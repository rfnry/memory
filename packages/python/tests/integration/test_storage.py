import asyncio
import copy
import json
import math
import os
import random
import re
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

import asyncpg
import pytest
from database import URL, Schemas, provision
from unit.fakes import HashEmbedding

from rfnry_memory.errors import InputError, NotFoundError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    Chunk,
    Counts,
    Entity,
    EntityIn,
    Hits,
    Kind,
    Leg,
    Relation,
    RelationIn,
    Scope,
    Section,
    Source,
    SourceIn,
    Story,
    Window,
)
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.postgres.graph import _NAMED_SQL
from rfnry_memory.postgres.locks import _language_lock, _sync_lock
from rfnry_memory.postgres.migrations import _CHUNK_SEARCHED
from rfnry_memory.postgres.stories import _text_lock
from rfnry_memory.postgres.vectors import _vectors_lock
from rfnry_memory.ranking import BM25
from rfnry_memory.rows import GraphWrite, SourceWrite, alias_keys, chunk_digest, name_key, text_hash
from rfnry_memory.tools import MemoryTools


async def _call(tools: MemoryTools, name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await tools.dispatch(name, args)
    assert isinstance(result, dict)
    return result


NOW = datetime(2026, 3, 1, tzinfo=UTC)


def ids(legs: list[Leg]) -> list[str]:
    return [i for _, i, _ in legs]


async def _text(storage: PostgresStorage, corpus_id: str, query: str, **options: Any) -> list[Leg]:
    found = await storage.search_text(
        [corpus_id], query, **{"ranking": BM25(), "match": "any", **options}
    )
    assert all(corpus == corpus_id for corpus, _ in found)
    return [leg for _, leg in found]


def _vec(text: str) -> list[float]:
    return [b / 255 for b in sha256(text.lower().encode()).digest()[:8]]


def _source(
    corpus: str, source_id: str, texts: list[str], *, metadata: dict[str, object] | None = None
) -> tuple[Source, list[Chunk], list[list[float]]]:
    source = Source(
        corpus_id=corpus,
        source_id=source_id,
        name=source_id,
        metadata=metadata or {},
        text="\n".join(texts),
        created_at=NOW,
        confirmed_at=NOW,
        chunks=len(texts),
    )
    chunks = [
        Chunk(
            corpus_id=corpus,
            chunk_id=f"{source_id}-{i}",
            source_id=source_id,
            source_name=source_id,
            position=i,
            title=None,
            locator={"page": i},
            text=t,
            metadata=dict(metadata or {}),
            created_at=NOW,
            confirmed_at=NOW,
        )
        for i, t in enumerate(texts)
    ]
    return source, chunks, [_vec(t) for t in texts]


_BARE: dict[str, Any] = {"subject": None, "metadata": {}}


def _pair(done: tuple[Counts, list[str], list[Source]] | None) -> tuple[Counts, list[str]] | None:
    return None if done is None else done[:2]


def _story(
    corpus: str,
    story_id: str,
    text: str,
    *,
    subject: str | None = None,
    valid_from: datetime = NOW,
    valid_to: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> Story:
    return Story(
        corpus_id=corpus,
        story_id=story_id,
        text=text,
        subject=subject,
        valid_from=valid_from,
        valid_to=valid_to,
        created_at=NOW,
        source_ids=(),
        metadata=metadata or {},
    )


async def _lock_key(storage: PostgresStorage, name: str) -> int:
    async with storage._pool.acquire() as conn:
        return int(
            await conn.fetchval("SELECT hashtextextended(current_schema() || ':' || $1, 0)", name)
        )


async def test_sources_roundtrip_list_and_delete(storage: PostgresStorage) -> None:
    source, chunks, vectors = _source(
        "a", "s1", ["pump seal replacement", "wiring of the panel"], metadata={"line": "A"}
    )
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    assert await storage.get_source("a", "s1") == source
    assert await storage.get_source("b", "s1") is None
    assert [c.chunk_id for c in await storage.source_chunks("a", "s1")] == ["s1-0", "s1-1"]
    listed, cursor = await storage.list_sources(
        "a", filters={"metadata.line": "A"}, limit=10, cursor=None
    )
    assert [s.source_id for s in listed] == ["s1"] and cursor is None
    listed, _ = await storage.list_sources(
        "a", filters={"metadata.line": "B"}, limit=10, cursor=None
    )
    assert listed == []
    deleted = await storage.delete_source("a", "s1")
    assert deleted.sources == 1 and deleted.chunks == 2
    assert (
        await storage.get_source("a", "s1") is None and await storage.source_chunks("a", "s1") == []
    )


async def test_list_sources_pages_with_a_cursor(storage: PostgresStorage) -> None:
    for i in range(5):
        source, chunks, vectors = _source("a", f"s{i}", [f"text {i}"])
        await storage.put_source(
            "a",
            source.model_copy(update={"created_at": NOW + timedelta(minutes=i)}),
            chunks,
            vectors,
            "hash-8",
        )
    first, cursor = await storage.list_sources("a", filters=None, limit=2, cursor=None)
    assert [s.source_id for s in first] == ["s4", "s3"] and cursor is not None
    second, cursor = await storage.list_sources("a", filters=None, limit=2, cursor=cursor)
    assert [s.source_id for s in second] == ["s2", "s1"]
    third, cursor = await storage.list_sources("a", filters=None, limit=2, cursor=cursor)
    assert [s.source_id for s in third] == ["s0"] and cursor is None


async def test_text_search_ranks_by_term_match_and_honours_kind_and_filters(
    storage: PostgresStorage,
) -> None:
    source, chunks, vectors = _source(
        "a",
        "s1",
        ["the pump seal must be replaced yearly", "wiring diagram of the control panel"],
        metadata={"line": "A"},
    )
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    await storage.put_story(
        "a",
        _story("a", "st1", "the seal on pump P-3 was replaced", metadata={"line": "B"}),
        _vec("seal"),
        "hash-8",
    )
    legs = await _text(
        storage,
        "a",
        "pump seal",
        kinds=("chunk", "story"),
        top_k=10,
        filters=None,
        window=Window(),
        subject=None,
    )
    assert [(k, i) for k, i, _ in legs][:2] == [("chunk", "s1-0"), ("story", "st1")] or [
        (k, i) for k, i, _ in legs
    ][:2] == [("story", "st1"), ("chunk", "s1-0")]
    assert all(score > 0 for _, _, score in legs) and ("chunk", "s1-1") not in [
        (k, i) for k, i, _ in legs
    ]
    only_chunks = await _text(
        storage,
        "a",
        "pump seal",
        kinds=("chunk",),
        top_k=10,
        filters=None,
        window=Window(),
        subject=None,
    )
    assert [i for _, i, _ in only_chunks] == ["s1-0"]
    faceted = await _text(
        storage,
        "a",
        "pump seal",
        kinds=("chunk", "story"),
        top_k=10,
        filters={"metadata.line": "B"},
        window=Window(),
        subject=None,
    )
    assert [i for _, i, _ in faceted] == ["st1"]
    asked = await _text(
        storage,
        "a",
        "how often is the seal replaced?",
        kinds=("chunk", "story"),
        top_k=10,
        filters=None,
        window=Window(),
        subject=None,
    )
    assert sorted(i for _, i, _ in asked) == ["s1-0", "st1"]
    covered = await _text(
        storage,
        "a",
        "which wiring diagram covers the seal?",
        kinds=("chunk", "story"),
        top_k=10,
        filters=None,
        window=Window(),
        subject=None,
    )
    assert [i for _, i, _ in covered][0] == "s1-1" and len(covered) == 3
    for nothing in ("what is the", "?!", ""):
        assert (
            await _text(
                storage,
                "a",
                nothing,
                kinds=("chunk", "story"),
                top_k=10,
                filters=None,
                window=Window(),
                subject=None,
            )
            == []
        )
    assert (
        await _text(
            storage,
            "b",
            "pump seal",
            kinds=("chunk", "story"),
            top_k=10,
            filters=None,
            window=Window(),
            subject=None,
        )
        == []
    )


async def test_the_text_pillar_stems_words_and_drops_stop_words(storage: PostgresStorage) -> None:
    source, chunks, vectors = _source(
        "a", "s1", ["The seal is replaced every year.", "Ground the panel first."]
    )
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    await storage.put_story(
        "a", _story("a", "st1", "Ana replaced the P-3 seal in March."), None, None
    )

    async def found(query: str) -> list[str]:
        legs = await _text(
            storage,
            "a",
            query,
            kinds=("chunk", "story"),
            top_k=10,
            filters=None,
            window=Window(),
            subject=None,
        )
        return sorted(ids(legs))

    assert await found("seals") == ["s1-0", "st1"]
    assert await found("Replacing SEALS") == ["s1-0", "st1"]
    assert await found("grounded panels") == ["s1-1"]
    for nothing in ("what is the", "The Of AND", "was it them?"):
        assert await found(nothing) == []


async def test_a_chunk_is_found_by_the_words_of_its_title(storage: PostgresStorage) -> None:
    source, chunks, vectors = _source("a", "s1", ["Do it every twelve months.", "Ground first."])
    titled = [
        chunks[0].model_copy(update={"title": "Pump manual > Seal replacement"}),
        chunks[1],
    ]
    await storage.put_source("a", source, titled, vectors, "hash-8")
    await storage.put_story("a", _story("a", "st1", "Nothing about it here."), None, None)
    for query in ("seals", "pump manual", "how is the seal replaced?"):
        legs = await _text(
            storage,
            "a",
            query,
            kinds=("chunk", "story"),
            top_k=10,
            filters=None,
            window=Window(),
            subject=None,
        )
        assert ids(legs) == ["s1-0"], query


async def test_vector_search_orders_by_cosine(storage: PostgresStorage) -> None:
    source, chunks, vectors = _source("a", "s1", ["alpha", "beta", "gamma"])
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    legs = await storage.search_vector(
        "a", _vec("beta"), kinds=("chunk",), top_k=2, filters=None, window=Window(), subject=None
    )
    assert legs[0][1] == "s1-1" and legs[0][2] == pytest.approx(1.0) and len(legs) == 2


async def test_story_windows_apply_inside_search(storage: PostgresStorage) -> None:
    jan, feb, mar = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3))
    await storage.put_story(
        "a",
        _story(
            "a", "old", "go-live is february twenty", subject="apollo", valid_from=jan, valid_to=feb
        ),
        _vec("go-live"),
        "hash-8",
    )
    await storage.put_story(
        "a",
        _story("a", "new", "go-live is march three", subject="apollo", valid_from=feb),
        _vec("go-live"),
        "hash-8",
    )
    assert set(
        ids(
            await _text(
                storage,
                "a",
                "go-live",
                kinds=("story",),
                top_k=10,
                filters=None,
                window=Window(),
                subject=None,
            )
        )
    ) == {"old", "new"}
    assert ids(
        await _text(
            storage,
            "a",
            "go-live",
            kinds=("story",),
            top_k=10,
            filters=None,
            window=Window(as_of=jan + timedelta(days=10)),
            subject=None,
        )
    ) == ["old"]
    assert ids(
        await _text(
            storage,
            "a",
            "go-live",
            kinds=("story",),
            top_k=10,
            filters=None,
            window=Window(as_of=mar),
            subject=None,
        )
    ) == ["new"]
    assert ids(
        await storage.search_vector(
            "a",
            _vec("go-live"),
            kinds=("story",),
            top_k=10,
            filters=None,
            window=Window(since=feb),
            subject=None,
        )
    ) == ["new"]
    assert ids(
        await storage.search_vector(
            "a",
            _vec("go-live"),
            kinds=("story",),
            top_k=10,
            filters=None,
            window=Window(until=jan),
            subject=None,
        )
    ) == ["old"]
    listed, _ = await storage.list_stories(
        "a", subject="apollo", window=Window(as_of=mar), filters=None, limit=10, cursor=None
    )
    assert [s.story_id for s in listed] == ["new"]


async def test_supersede_is_one_step_and_hash_finds_only_open_stories(
    storage: PostgresStorage,
) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    old = _story("a", "old", "The date is Feb 20.", valid_from=jan)
    await storage.put_story("a", old, None, None)
    assert (await storage.find_open_story("a", text_hash(old.text), **_BARE, at=NOW)) == old
    new = _story("a", "new", "The date is Mar 3.", valid_from=feb).model_copy(
        update={"supersedes": ("old",)}
    )
    [closed], created = await storage.supersede_story("a", ["old"], new, None, None)
    assert closed.valid_to == feb and created.supersedes == ("old",)
    assert await storage.find_open_story("a", text_hash(old.text), **_BARE, at=NOW) is None
    assert await storage.find_open_story("a", text_hash(new.text), **_BARE, at=NOW) == created
    with pytest.raises(NotFoundError):
        await storage.supersede_story(
            "a", ["ghost"], new.model_copy(update={"story_id": "x"}), None, None
        )
    with pytest.raises(
        InputError,
        match=r"^postgres: story 'old' is already superseded; supersede the story that replaced it$",
    ):
        await storage.supersede_story(
            "a",
            ["old"],
            new.model_copy(update={"story_id": "y", "text": "The date is Apr 4."}),
            None,
            None,
        )


async def test_graph_entities_relations_mentions_and_traversal(storage: PostgresStorage) -> None:
    source, chunks, vectors = _source("a", "s1", ["valve V-101 feeds pump P-3"])
    v = Entity(corpus_id="a", entity_id="e1", name="V-101", key="v-101", type="valve")
    p = Entity(corpus_id="a", entity_id="e2", name="P-3", key="p-3", type="pump")
    t = Entity(corpus_id="a", entity_id="e3", name="Tank T-9", key="tank t-9", type="tank")
    relations = [
        Relation(
            corpus_id="a",
            from_id="e1",
            to_id="e2",
            type="CONNECTS_TO",
            evidence_kind="chunk",
            evidence_id="s1-0",
        ),
        Relation(
            corpus_id="a",
            from_id="e2",
            to_id="e3",
            type="FEEDS",
            evidence_kind="chunk",
            evidence_id="s1-0",
        ),
    ]
    graph = GraphWrite(
        entities=[v, p, t],
        relations=relations,
        mentions=[("e1", "chunk", "s1-0", "V-101"), ("e2", "chunk", "s1-0", "P-3")],
    )
    await storage.put_source("a", source, chunks, vectors, "hash-8", graph=graph)
    assert (await storage.entities_named("a", ["p-3"]))[name_key("p-3")] == [p] and (
        (await storage.entities_named("a", ["P-3"]))[name_key("P-3")]
    ) == [p]
    assert [e.entity_id for e in await storage.entities_by_id("a", ["e3", "ghost", "e1"])] == [
        "e3",
        "e1",
    ]
    for spot in ("words", "exact", "substring"):
        named = await storage.entities_named_in("a", "is V-101 open?", spot=spot)
        assert [e.entity_id for e in named] == ["e1"], spot
    assert [e.entity_id for e in await storage.similar_entities("a", "tank T9", limit=5)][:1] == [
        "e3"
    ]
    hop = await storage.neighbors("a", ["e1"], relation_types=None, window=Window())
    assert [(r.from_id, r.to_id) for r in hop] == [("e1", "e2")]
    assert [
        (r.type)
        for r in await storage.neighbors("a", ["e2"], relation_types=["FEEDS"], window=Window())
    ] == ["FEEDS"]
    assert await storage.mentions(
        "a", ["e1", "e3"], kinds=("chunk", "story"), filters=None, window=Window(), subject=None
    ) == [("e1", ("chunk", "s1-0"))]
    await storage.put_story(
        "a",
        _story("a", "st", "P-3 is also called pump three"),
        None,
        None,
        graph=GraphWrite(aliases=[("e2", "Pump three")]),
    )
    assert (await storage.entities_named("a", ["pump three"]))[name_key("pump three")] == [
        p.model_copy(update={"aliases": ("Pump three",)})
    ]
    deleted = await storage.delete_source("a", "s1")
    assert deleted.mentions == 2 and deleted.relations == 2
    assert await storage.neighbors("a", ["e1"], relation_types=None, window=Window()) == []


async def test_passages_hydrate_in_order_and_reembed_helpers(storage: PostgresStorage) -> None:
    source, chunks, vectors = _source("a", "s1", ["one", "two"])
    await storage.put_source("a", source, chunks, vectors, "old-model")
    await storage.put_story("a", _story("a", "st", "three"), _vec("three"), "old-model")
    passages = await storage.get_passages(
        "a", [("story", "st"), ("chunk", "s1-1"), ("chunk", "ghost")]
    )
    assert [p.story_id if isinstance(p, Story) else p.chunk_id for p in passages] == ["st", "s1-1"]
    stale = await storage.passages_embedded_with("a", not_model="new-model", limit=10)
    assert len(stale) == 3
    await storage.set_vectors(
        "a", [("chunk", "s1-0"), ("chunk", "s1-1"), ("story", "st")], [_vec("x")] * 3, "new-model"
    )
    assert await storage.passages_embedded_with("a", not_model="new-model", limit=10) == []


async def test_delete_corpus_is_total_and_counted(storage: PostgresStorage) -> None:
    source, chunks, vectors = _source("a", "s1", ["one"])
    await storage.put_source("a", source, chunks, vectors, "m")
    await storage.put_story(
        "a",
        _story("a", "st", "two"),
        None,
        None,
        graph=GraphWrite(
            entities=[Entity(corpus_id="a", entity_id="e1", name="x", key="x", type="t")],
            mentions=[("e1", "story", "st", "x")],
        ),
    )
    other, other_chunks, other_vectors = _source("b", "s2", ["keep"])
    await storage.put_source("b", other, other_chunks, other_vectors, "m")
    residue = await storage.count_corpus("a")
    assert (
        residue.sources,
        residue.chunks,
        residue.stories,
        residue.entities,
        residue.mentions,
    ) == (1, 1, 1, 1, 1) and not residue.clean
    deleted = await storage.delete_corpus("a")
    assert (
        deleted.sources,
        deleted.chunks,
        deleted.stories,
        deleted.entities,
        deleted.mentions,
    ) == (1, 1, 1, 1, 1)
    assert (await storage.count_corpus("a")).clean and not (await storage.count_corpus("b")).clean


async def test_aliases_match_by_normalized_name(storage: PostgresStorage) -> None:
    entity = Entity(
        corpus_id="a",
        entity_id="e1",
        name="P-3",
        key="p-3",
        type="pump",
        aliases=("\t Pump \n  Three ",),
    )
    await storage.put_story(
        "a", _story("a", "st", "P-3"), None, None, graph=GraphWrite(entities=[entity])
    )
    assert (await storage.entities_named("a", ["pump three"]))[name_key("pump three")] != []
    for spot in ("words", "exact", "substring"):
        for query in ("where is PUMP three now", " where is\tPUMP \n three now"):
            named = await storage.entities_named_in("a", query, spot=spot)
            assert [e.entity_id for e in named] == ["e1"], (spot, query)
    assert [
        e.entity_id for e in (await storage.entities_named("a", [" \tP-3 "]))[name_key(" \tP-3 ")]
    ] == ["e1"]


async def test_metadata_and_locators_round_trip_exactly(storage: PostgresStorage) -> None:
    metadata: dict[str, object] = {
        "zeta": 1,
        "a": {"refund": "r", "no": "n", "billing": "b"},
        "big": 1e16,
        "tiny": 1.5e-7,
    }
    source, chunks, vectors = _source("a", "s1", ["one"], metadata=metadata)
    chunks = [c.model_copy(update={"locator": metadata}) for c in chunks]
    await storage.put_source("a", source, chunks, vectors, "m")
    await storage.put_story("a", _story("a", "st", "two", metadata=metadata), None, None)
    got = await storage.get_source("a", "s1")
    listed, _ = await storage.list_sources("a", filters={"metadata.zeta": 1}, limit=10, cursor=None)
    stories, _ = await storage.list_stories(
        "a", subject=None, window=Window(), filters={"metadata.a.no": "n"}, limit=10, cursor=None
    )
    passages = await storage.get_passages("a", [("chunk", "s1-0"), ("story", "st")])
    assert got is not None and len(listed) == 1 and len(stories) == 1 and len(passages) == 2
    stored = [
        got.metadata,
        listed[0].metadata,
        stories[0].metadata,
        (await storage.source_chunks("a", "s1"))[0].locator,
        (await storage.get_stories("a", ["st"]))[0].metadata,
    ]
    stored += [p.locator if isinstance(p, Chunk) else p.metadata for p in passages]
    for value in stored:
        assert value == metadata
        assert list(value) == list(metadata) and list(value["a"]) == ["refund", "no", "billing"]
        assert type(value["big"]) is float


async def test_ties_order_by_code_point_whatever_the_collation(storage: PostgresStorage) -> None:
    tied = ["ab", "B", "a-b", "_z", "a"]
    chunk_ids = sorted(f"{tie}-0" for tie in tied)
    for tie in tied:
        entity = Entity(corpus_id="a", entity_id=tie, name="valve", key="valve", type=f"type {tie}")
        feeds = Relation(
            corpus_id="a",
            from_id=tie,
            to_id="ab",
            type="FEEDS",
            evidence_kind="story",
            evidence_id=tie,
        )
        await storage.put_story(
            "a",
            _story("a", tie, f"story {tie}", valid_from=NOW),
            [1.0] * 8,
            "m",
            graph=GraphWrite(
                entities=[entity], relations=[feeds], mentions=[(tie, "story", tie, "valve")]
            ),
        )
        source, chunks, vectors = _source("a", tie, ["pump seal"])
        source = source.model_copy(update={"name": "manual"})
        chunks = [c.model_copy(update={"source_name": "manual"}) for c in chunks]
        await storage.put_source(
            "a",
            source,
            chunks,
            vectors,
            "m",
            graph=GraphWrite(mentions=[("ab", "chunk", f"{tie}-0", "valve")]),
        )
    sources, _ = await storage.list_sources("a", filters=None, limit=10, cursor=None)
    assert [s.source_id for s in sources] == sorted(tied, reverse=True)
    stories, _ = await storage.list_stories(
        "a", subject=None, window=Window(), filters=None, limit=10, cursor=None
    )
    assert [s.story_id for s in stories] == sorted(tied, reverse=True)
    for spot in ("words", "exact", "substring"):
        named = await storage.entities_named_in("a", "the valve", spot=spot)
        assert [e.entity_id for e in named] == sorted(tied), spot
    assert [e.entity_id for e in await storage.similar_entities("a", "valve", limit=10)] == sorted(
        tied
    )
    refs = await storage.mentions(
        "a", tied, kinds=("chunk", "story"), filters=None, window=Window(), subject=None
    )
    assert refs == [("ab", ("chunk", c)) for c in chunk_ids] + [
        (tie, ("story", tie)) for tie in sorted(tied)
    ]
    relations = await storage.neighbors("a", ["ab"], relation_types=None, window=Window())
    assert [r.from_id for r in relations] == sorted(tied)
    both: tuple[Kind, ...] = ("chunk", "story")
    assert (
        ids(
            await _text(
                storage,
                "a",
                "pump seal",
                kinds=both,
                top_k=10,
                filters=None,
                window=Window(),
                subject=None,
            )
        )
        == chunk_ids
    )
    assert ids(
        await storage.search_vector(
            "a", [1.0] * 8, kinds=("story",), top_k=10, filters=None, window=Window(), subject=None
        )
    ) == sorted(tied)
    assert (
        ids(
            await storage.search_vector(
                "a",
                _vec("pump seal"),
                kinds=("chunk",),
                top_k=10,
                filters=None,
                window=Window(),
                subject=None,
            )
        )
        == chunk_ids
    )


async def test_text_with_a_nul_character_is_refused(storage: PostgresStorage) -> None:
    memory = Memory(storage)
    with pytest.raises(InputError):
        await memory.ingest("a", name="manual", text="pump\x00seal")
    with pytest.raises(InputError):
        await memory.ingest("a", name="manual", text="pump seal", metadata={"line": "A\x00"})
    with pytest.raises(InputError):
        await memory.remember("a", "the seal\x00 was replaced")
    with pytest.raises(InputError):
        await memory.remember("a", "the seal was replaced", subject="P\x003")
    with pytest.raises(InputError):
        await memory.search("a", "pump\x00seal")
    with pytest.raises(InputError):
        await memory.get_source("a\x00", "s1")
    for source_in in (
        SourceIn(name="manual", text="pump\x00seal", metadata={"file": "a"}),
        SourceIn(name="sheet", sections=[Section(text="pump seal\x00")], metadata={"file": "a"}),
    ):
        with pytest.raises(InputError):
            await memory.sync("a", filters={"metadata.file": "a"}, sources=[source_in])
    assert (await storage.count_corpus("a")).clean


async def test_the_expand_tool_refuses_an_id_with_a_nul_character(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage)
    refused = r"^memory: text must not contain a NUL character$"
    for tools in (
        MemoryTools(memory, corpus_id="a"),
        MemoryTools(memory, corpus_id="a", filters={"metadata.line": "A"}),
    ):
        for kind in ("chunk", "story"):
            with pytest.raises(InputError, match=refused):
                await tools.dispatch("expand", {"kind": kind, "id": "s1\x00-0", "by": "document"})


async def test_a_number_that_is_not_finite_is_refused(storage: PostgresStorage) -> None:
    memory = Memory(storage)
    refused = r"^memory: numbers must be finite; NaN and infinity cannot be stored$"
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(InputError, match=refused):
            await memory.remember("a", "the seal was replaced", metadata={"v": value})
        with pytest.raises(InputError, match=refused):
            await memory.ingest(
                "a", name="manual", text="pump seal", metadata={"a": [{"v": value}]}
            )
        with pytest.raises(InputError, match=refused):
            await memory.ingest(
                "a", name="sheet", sections=[Section(text="pump seal", locator={"x": value})]
            )
        for source_in in (
            SourceIn(name="manual", text="pump seal", metadata={"file": "a", "v": value}),
            SourceIn(
                name="sheet",
                sections=[Section(text="pump seal", locator={"x": value})],
                metadata={"file": "a"},
            ),
        ):
            with pytest.raises(InputError, match=refused):
                await memory.sync("a", filters={"metadata.file": "a"}, sources=[source_in])
    assert (await storage.count_corpus("a")).clean


def _hit_ids(hits: Hits) -> set[str]:
    return {h.id for h in hits.hits}


async def test_story_reads_answer_with_what_is_true_now_unless_history_is_asked(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    apollo = [EntityIn(name="Apollo", type="project")]
    jan, feb = datetime(2026, 1, 10, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    closed = (
        await memory.remember(
            "a",
            "The go-live is February 20.",
            subject="project:apollo",
            valid_from=jan,
            metadata={"team": "ops"},
            entities=apollo,
        )
    ).story
    current = (
        await memory.supersede(
            "a", closed.story_id, "The go-live moved to March 3.", valid_from=feb
        )
    ).story
    later = (
        await memory.remember(
            "a",
            "The go-live rehearsal is planned.",
            subject="project:apollo",
            valid_from=datetime.now(UTC) + timedelta(days=30),
            entities=apollo,
        )
    ).story
    lead = (
        await memory.remember("a", "Ana leads Apollo.", subject="user:ana", entities=apollo)
    ).story
    everything = {closed.story_id, current.story_id, later.story_id}

    assert _hit_ids(await memory.recall("a", "go-live")) == {current.story_id}
    assert _hit_ids(await memory.recall("a", "go-live", history=True)) == everything
    assert _hit_ids(await memory.recall("a", "go-live", as_of=jan + timedelta(days=10))) == {
        closed.story_id
    }
    with pytest.raises(InputError):
        await memory.recall("a", "go-live", history=True, as_of=jan)
    assert _hit_ids(await memory.recall("a", "go-live", since=feb)) == {current.story_id}
    assert _hit_ids(await memory.recall("a", "go-live", since=feb, history=True)) == {
        current.story_id,
        later.story_id,
    }
    assert _hit_ids(await memory.recall("a", "go-live", until=feb, history=True)) == {
        closed.story_id,
        current.story_id,
    }
    assert _hit_ids(await memory.recall("a", "go-live", subject="project:apollo")) == {
        current.story_id
    }
    assert _hit_ids(
        await memory.recall("a", "go-live", filters={"metadata.team": "ops"}, history=True)
    ) == {closed.story_id, current.story_id}
    assert _hit_ids(await memory.recall("a", "go-live", filters={"metadata.team": "ops"})) == {
        current.story_id
    }

    assert _hit_ids(await memory.search("a", "go-live", kinds=("story",))) == {current.story_id}
    assert (
        _hit_ids(await memory.search("a", "go-live", kinds=("story",), history=True)) == everything
    )

    listed, _ = await memory.list_stories("a", subject="project:apollo")
    assert [s.story_id for s in listed] == [current.story_id]
    listed, _ = await memory.list_stories("a", subject="project:apollo", history=True)
    assert [s.story_id for s in listed] == [later.story_id, current.story_id, closed.story_id]
    listed, _ = await memory.list_stories("a", as_of=jan + timedelta(days=10))
    assert [s.story_id for s in listed] == [closed.story_id]
    with pytest.raises(InputError):
        await memory.list_stories("a", history=True, as_of=jan)

    assert {h.id for h in (await memory.neighbors("a", "Apollo")).passages} == {lead.story_id}
    assert {h.id for h in (await memory.neighbors("a", "Apollo", history=True)).passages} == {
        closed.story_id,
        later.story_id,
        lead.story_id,
    }


async def test_a_superseding_story_is_written_to_the_graph_like_a_remembered_one(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    feb = datetime(2026, 2, 1, tzinfo=UTC)
    closed = (
        await memory.remember(
            "a",
            "Ana owns the pump.",
            valid_from=datetime(2026, 1, 10, tzinfo=UTC),
            entities=[EntityIn(name="Ana", type="person"), EntityIn(name="P-3", type="pump")],
            relations=[RelationIn(from_name="Ana", to_name="P-3", type="OWNS")],
        )
    ).story
    current = (
        await memory.supersede(
            "a",
            closed.story_id,
            "Bo owns the pump.",
            valid_from=feb,
            entities=[EntityIn(name="Bo", type="person"), EntityIn(name="P-3", type="pump")],
            relations=[RelationIn(from_name="Bo", to_name="P-3", type="OWNS")],
        )
    ).story

    hood = await memory.neighbors("a", "P-3")
    assert {e.name for e in hood.entities} == {"P-3", "Bo"}
    assert {h.id for h in hood.passages} == {current.story_id}
    past = await memory.neighbors("a", "P-3", history=True)
    assert {e.name for e in past.entities} == {"P-3", "Ana", "Bo"}
    assert {h.id for h in (await memory.neighbors("a", "P-3", history=True)).passages} == {
        closed.story_id,
        current.story_id,
    }
    assert {h.id for h in (await memory.neighbors("a", "Bo")).passages} == {current.story_id}
    assert _hit_ids(await memory.recall("a", "Bo", pillars={"graph"})) == {current.story_id}
    with pytest.raises(InputError):
        await memory.supersede(
            "a",
            current.story_id,
            "Cy owns the pump.",
            relations=[RelationIn(from_name="Cy", to_name="P-3", type="OWNS")],
        )
    assert [s.story_id for s in (await memory.list_stories("a"))[0]] == [current.story_id]


def _bad_graph(corpus: str, kind: Kind, passage_id: str) -> GraphWrite:
    return GraphWrite(
        entities=[Entity(corpus_id=corpus, entity_id="e1", name="P-3", key="p-3", type="pump")],
        relations=[
            Relation(
                corpus_id=corpus,
                from_id="e1",
                to_id="ghost",
                type="FEEDS",
                evidence_kind=kind,
                evidence_id=passage_id,
            )
        ],
        mentions=[("e1", kind, passage_id, "P-3")],
    )


async def test_a_write_whose_graph_is_refused_leaves_nothing_behind(
    storage: PostgresStorage,
) -> None:
    source, chunks, vectors = _source("a", "s1", ["pump P-3 feeds tank T-9"])
    with pytest.raises(StorageError):
        await storage.put_source(
            "a", source, chunks, vectors, "m", graph=_bad_graph("a", "chunk", "s1-0")
        )
    with pytest.raises(StorageError):
        await storage.put_story(
            "a",
            _story("a", "st", "pump P-3 feeds tank T-9"),
            _vec("p-3"),
            "m",
            graph=_bad_graph("a", "story", "st"),
        )
    elsewhere = GraphWrite(
        entities=[Entity(corpus_id="a", entity_id="e1", name="P-3", key="p-3", type="pump")],
        mentions=[("e1", "story", "other", "P-3")],
    )
    with pytest.raises(StorageError):
        await storage.put_story("a", _story("a", "st", "pump P-3"), None, None, graph=elsewhere)
    assert (await storage.count_corpus("a")).clean
    assert (
        await storage.search_vector(
            "a",
            _vec("p-3"),
            kinds=("chunk", "story"),
            top_k=10,
            filters=None,
            window=Window(),
            subject=None,
        )
        == []
    )

    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    old = _story("a", "old", "P-3 feeds tank T-8", valid_from=jan)
    await storage.put_story("a", old, None, None)
    new = _story("a", "new", "P-3 feeds tank T-9", valid_from=feb).model_copy(
        update={"supersedes": ("old",)}
    )
    with pytest.raises(StorageError):
        await storage.supersede_story(
            "a", ["old"], new, _vec("t-9"), "m", graph=_bad_graph("a", "story", "new")
        )
    assert await storage.get_stories("a", ["old", "new"]) == [old]
    residue = await storage.count_corpus("a")
    assert (residue.stories, residue.entities, residue.relations, residue.mentions) == (1, 0, 0, 0)


async def test_a_write_carries_its_graph_in_the_same_step(storage: PostgresStorage) -> None:
    source, chunks, vectors = _source(
        "a", "s1", ["pump P-3 feeds tank T-9", "tank T-9 holds water"]
    )
    pump = Entity(corpus_id="a", entity_id="e1", name="P-3", key="p-3", type="pump")
    tank = Entity(corpus_id="a", entity_id="e2", name="T-9", key="t-9", type="tank")
    feeds = Relation(
        corpus_id="a",
        from_id="e1",
        to_id="e2",
        type="FEEDS",
        evidence_kind="chunk",
        evidence_id="s1-0",
    )
    await storage.put_source(
        "a",
        source,
        chunks,
        vectors,
        "m",
        graph=GraphWrite(
            entities=[pump, tank],
            relations=[feeds],
            mentions=[
                ("e1", "chunk", "s1-0", "P-3"),
                ("e2", "chunk", "s1-0", "T-9"),
                ("e2", "chunk", "s1-1", "T-9"),
            ],
        ),
    )
    residue = await storage.count_corpus("a")
    assert (
        residue.sources,
        residue.chunks,
        residue.entities,
        residue.relations,
        residue.mentions,
    ) == (1, 2, 2, 1, 3)
    assert ids(
        await storage.search_vector(
            "a",
            _vec("tank T-9 holds water"),
            kinds=("chunk",),
            top_k=1,
            filters=None,
            window=Window(),
            subject=None,
        )
    ) == ["s1-1"]

    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    old = _story("a", "old", "Ana runs P-3", valid_from=jan)
    await storage.put_story(
        "a",
        old,
        _vec("ana"),
        "m",
        graph=GraphWrite(
            entities=[Entity(corpus_id="a", entity_id="e3", name="Ana", key="ana", type="person")],
            relations=[
                Relation(
                    corpus_id="a",
                    from_id="e3",
                    to_id="e1",
                    type="RUNS",
                    evidence_kind="story",
                    evidence_id="old",
                )
            ],
            mentions=[("e3", "story", "old", "Ana"), ("e1", "story", "old", "P-3")],
        ),
    )
    new = _story("a", "new", "Bo runs P-3", valid_from=feb).model_copy(
        update={"supersedes": ("old",)}
    )
    [closed], created = await storage.supersede_story(
        "a",
        ["old"],
        new,
        _vec("bo"),
        "m",
        graph=GraphWrite(
            entities=[Entity(corpus_id="a", entity_id="e4", name="Bo", key="bo", type="person")],
            aliases=[("e1", "Pump three")],
            relations=[
                Relation(
                    corpus_id="a",
                    from_id="e4",
                    to_id="e1",
                    type="RUNS",
                    evidence_kind="story",
                    evidence_id="new",
                )
            ],
            mentions=[("e4", "story", "new", "Bo"), ("e1", "story", "new", "P-3")],
        ),
    )
    assert closed.valid_to == feb and created == new
    assert await storage.mentions(
        "a", ["e1"], kinds=("chunk", "story"), filters=None, window=Window(as_of=feb), subject=None
    ) == [("e1", ("chunk", "s1-0")), ("e1", ("story", "new"))]
    assert [
        (r.from_id, r.type)
        for r in await storage.neighbors("a", ["e4"], relation_types=None, window=Window())
    ] == [("e4", "RUNS")]
    assert (await storage.entities_named("a", ["pump three"]))[name_key("pump three")] == [
        pump.model_copy(update={"aliases": ("Pump three",)})
    ]
    residue = await storage.count_corpus("a")
    assert (residue.stories, residue.entities, residue.relations, residue.mentions) == (2, 4, 3, 7)


async def test_an_entity_created_by_a_concurrent_writer_is_joined_not_duplicated(
    storage: PostgresStorage,
) -> None:
    first = Entity(corpus_id="a", entity_id="first", name="P-3", key="p-3", type="pump")
    await storage.put_story(
        "a",
        _story("a", "st1", "P-3 hums"),
        None,
        None,
        graph=GraphWrite(entities=[first], mentions=[("first", "story", "st1", "P-3")]),
    )
    late = Entity(
        corpus_id="a", entity_id="late", name="P-3", key="p-3", type="pump", aliases=("Pump three",)
    )
    tank = Entity(corpus_id="a", entity_id="tank", name="T-9", key="t-9", type="tank")
    await storage.put_story(
        "a",
        _story("a", "st2", "pump three feeds T-9"),
        None,
        None,
        graph=GraphWrite(
            entities=[late, tank],
            relations=[
                Relation(
                    corpus_id="a",
                    from_id="late",
                    to_id="tank",
                    type="FEEDS",
                    evidence_kind="story",
                    evidence_id="st2",
                )
            ],
            mentions=[("late", "story", "st2", "P-3"), ("tank", "story", "st2", "T-9")],
        ),
    )
    assert (await storage.count_corpus("a")).entities == 2
    assert await storage.entities_by_id("a", ["late"]) == []
    assert (await storage.entities_named("a", ["pump three"]))[name_key("pump three")] == [
        first.model_copy(update={"aliases": ("Pump three",)})
    ]
    assert await storage.mentions(
        "a", ["first"], kinds=("story",), filters=None, window=Window(), subject=None
    ) == [("first", ("story", "st1")), ("first", ("story", "st2"))]
    assert [
        (r.from_id, r.to_id)
        for r in await storage.neighbors("a", ["first"], relation_types=None, window=Window())
    ] == [("first", "tank")]

    racers = [
        Entity(
            corpus_id="a",
            entity_id=f"racer{i}",
            name="V-7",
            key="v-7",
            type="valve",
            aliases=(f"valve {i}",),
        )
        for i in range(4)
    ]
    await asyncio.gather(
        *(
            storage.put_story(
                "a",
                _story("a", f"race{i}", f"V-7 report {i}"),
                None,
                None,
                graph=GraphWrite(
                    entities=[racer, tank], mentions=[(racer.entity_id, "story", f"race{i}", "V-7")]
                ),
            )
            for i, racer in enumerate(racers)
        )
    )
    [joined] = (await storage.entities_named("a", ["v-7"]))[name_key("v-7")]
    assert sorted(joined.aliases) == [f"valve {i}" for i in range(4)]
    assert (await storage.count_corpus("a")).entities == 3
    assert (
        len(
            await storage.mentions(
                "a",
                [joined.entity_id],
                kinds=("story",),
                filters=None,
                window=Window(),
                subject=None,
            )
        )
        == 4
    )


async def test_memory_writes_graph_and_passage_as_one_step(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    facts = [EntityIn(name="Ana", type="person"), EntityIn(name="P-3", type="pump")]
    ingested = await memory.ingest(
        "a",
        name="sheet",
        text="Ana runs P-3.",
        entities=facts,
        relations=[RelationIn(from_name="Ana", to_name="P-3", type="RUNS")],
    )
    story = (
        await memory.remember(
            "a", "P-3 was serviced.", entities=[EntityIn(name="p-3", type="pump")]
        )
    ).story
    current = (
        await memory.supersede(
            "a",
            story.story_id,
            "P-3 was replaced.",
            entities=[EntityIn(name="P-3", type="pump")],
            relations=[RelationIn(from_name="Ana", to_name="P-3", type="RUNS")],
        )
    ).story
    residue = await storage.count_corpus("a")
    assert (
        residue.sources,
        residue.stories,
        residue.entities,
        residue.relations,
        residue.mentions,
    ) == (1, 2, 2, 2, 5)
    hood = await memory.neighbors("a", "P-3")
    assert {h.id for h in hood.passages} == {f"{ingested.source.source_id}-0", current.story_id}


async def test_subject_narrows_inside_every_leg_not_after_the_fetch(
    storage: PostgresStorage,
) -> None:
    go_live = Entity(corpus_id="a", entity_id="go", name="go-live", key="go-live", type="event")
    for i in range(60):
        story = _story(
            "a", f"other-{i:02d}", f"go-live go-live go-live {i}", subject=f"project:{i}"
        )
        await storage.put_story(
            "a",
            story,
            _vec("go-live"),
            "hash-8",
            graph=GraphWrite(
                entities=[go_live] if i == 0 else [],
                mentions=[("go", "story", story.story_id, "go-live")],
            ),
        )
    await storage.put_story(
        "a",
        _story(
            "a",
            "zz-apollo",
            "the apollo go-live moved after a long review of the plan",
            subject="project:apollo",
        ),
        _vec("unrelated"),
        "hash-8",
        graph=GraphWrite(mentions=[("go", "story", "zz-apollo", "go-live")]),
    )
    source, chunks, vectors = _source("a", "s1", ["go-live"])
    await storage.put_source(
        "a",
        source,
        chunks,
        vectors,
        "hash-8",
        graph=GraphWrite(mentions=[("go", "chunk", "s1-0", "go-live")]),
    )
    both: tuple[Kind, ...] = ("chunk", "story")
    assert ids(
        await _text(
            storage,
            "a",
            "go-live",
            kinds=both,
            top_k=5,
            filters=None,
            window=Window(),
            subject="project:apollo",
        )
    ) == ["zz-apollo"]
    assert ids(
        await storage.search_vector(
            "a",
            _vec("go-live"),
            kinds=both,
            top_k=5,
            filters=None,
            window=Window(),
            subject="project:apollo",
        )
    ) == ["zz-apollo"]
    assert await storage.mentions(
        "a", ["go"], kinds=both, filters=None, window=Window(), subject="project:apollo"
    ) == [("go", ("story", "zz-apollo"))]
    assert "zz-apollo" not in ids(
        await _text(
            storage,
            "a",
            "go-live",
            kinds=both,
            top_k=5,
            filters=None,
            window=Window(),
            subject=None,
        )
    )
    assert (
        len(
            await storage.mentions(
                "a", ["go"], kinds=both, filters=None, window=Window(), subject=None
            )
        )
        == 62
    )


async def test_recall_by_subject_finds_the_subject_story_however_many_others_outrank_it(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    go_live = [EntityIn(name="go-live", type="event")]
    for i in range(60):
        await memory.remember(
            "a", f"go-live go-live go-live {i}", subject=f"project:{i}", entities=go_live
        )
    apollo = (
        await memory.remember(
            "a",
            "the apollo go-live moved after a long review of the plan",
            subject="project:apollo",
            entities=go_live,
        )
    ).story
    for pillars in ({"text"}, {"vector"}, {"graph"}, {"text", "vector", "graph"}):
        hits = await memory.recall(
            "a", "go-live", top_k=5, subject="project:apollo", pillars=pillars
        )
        assert [h.id for h in hits.hits] == [apollo.story_id], pillars


async def test_bound_tools_recall_what_they_remember(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    outside = (
        await memory.remember("a", "Line B owns the wiring review.", metadata={"line": "B"})
    ).story
    tools = MemoryTools(
        memory, corpus_id="a", filters={"metadata.line": "A", "metadata.site.zone": 3}
    )
    remembered = await _call(tools, "remember", {"text": "Ana owns the wiring review."})
    assert remembered["story"]["metadata"] == {"line": "A", "site": {"zone": 3}}
    recalled = await _call(tools, "recall", {"query": "wiring review"})
    assert [h["id"] for h in recalled["hits"]] == [remembered["story"]["story_id"]]
    repeated = (await _call(tools, "remember", {"text": "line b owns the  wiring review."}))[
        "story"
    ]
    assert repeated["story_id"] != outside.story_id
    assert repeated["metadata"] == {"line": "A", "site": {"zone": 3}}
    with pytest.raises(InputError, match=repeated["story_id"]):
        await _call(
            tools,
            "supersede",
            {"story_id": remembered["story"]["story_id"], "text": "Line B owns the wiring review."},
        )


async def test_neighbors_asks_for_a_type_when_a_name_names_several(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.ingest(
        "a", name="bom", text="Part P-3 is a gasket.", entities=[EntityIn(name="P-3", type="part")]
    )
    facts = [EntityIn(name="V-1", type="valve"), EntityIn(name="P-3", type="pump")]
    feeds = RelationIn(from_name="V-1", to_name="P-3", type="FEEDS")
    with pytest.raises(
        InputError,
        match=r"^memory: relation 'V-1' -> 'P-3' names 'P-3', which has several types: \['part', 'pump'\]",
    ):
        await memory.ingest(
            "a", name="sheet", text="Valve V-1 feeds pump P-3.", entities=facts, relations=[feeds]
        )
    assert (await storage.count_corpus("a")).sources == 1 and [
        e.type for e in (await storage.entities_named("a", ["p-3"]))[name_key("p-3")]
    ] == ["part"]
    await memory.ingest(
        "a",
        name="sheet",
        text="Valve V-1 feeds pump P-3.",
        entities=facts,
        relations=[feeds.model_copy(update={"to_type": "pump"})],
    )
    assert sorted(
        e.type for e in (await storage.entities_named("a", ["p-3"]))[name_key("p-3")]
    ) == ["part", "pump"]
    assert (await storage.entities_named("b", ["p-3"]))[name_key("p-3")] == []
    with pytest.raises(
        InputError,
        match=r"^memory: entity 'P-3' names entities of several types: \['part', 'pump'\]",
    ):
        await memory.neighbors("a", "P-3")
    assert {
        (e.name, e.type) for e in (await memory.neighbors("a", "P-3", entity_type="Pump")).entities
    } == {("P-3", "pump"), ("V-1", "valve")}
    assert [
        (e.name, e.type) for e in (await memory.neighbors("a", "P-3", entity_type="part")).entities
    ] == [("P-3", "part")]
    with pytest.raises(NotFoundError, match=r"^memory: entity 'P-3' of type 'valve' was not found"):
        await memory.neighbors("a", "P-3", entity_type="valve")
    assert {e.name for e in (await memory.neighbors("a", "V-1")).entities} == {"V-1", "P-3"}


class _Yielding(HashEmbedding):
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        await asyncio.sleep(0)
        return await super().embed(texts, purpose=purpose)


async def test_find_open_story_keys_on_text_subject_and_metadata(
    storage: PostgresStorage,
) -> None:
    story = _story("a", "s1", "Ana owns P-3.", subject="user:ana", metadata={"line": "A", "n": 1})
    assert await storage.put_story("a", story, None, None) == story
    hashed = text_hash("ana owns  p-3.")
    same: dict[str, Any] = {"subject": "user:ana", "metadata": {"n": 1, "line": "A"}}
    assert await storage.find_open_story("a", hashed, **same, at=NOW) == story
    others: list[dict[str, Any]] = [
        {"subject": None, "metadata": {"line": "A", "n": 1}},
        {"subject": "user:bo", "metadata": {"line": "A", "n": 1}},
        {"subject": "user:ana", "metadata": {}},
        {"subject": "user:ana", "metadata": {"line": "A", "n": True}},
    ]
    for other in others:
        assert await storage.find_open_story("a", hashed, **other, at=NOW) is None, other
    bo = _story("a", "s2", "Ana owns P-3.", subject="user:bo", metadata={"line": "A", "n": 1})
    assert await storage.put_story("a", bo, None, None) == bo
    again = _story("a", "s3", "ana owns  p-3.", subject="user:ana", metadata={"n": 1, "line": "A"})
    assert await storage.put_story("a", again, None, None) == story


async def test_concurrent_writers_of_one_text_keep_one_story_per_key(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=_Yielding())
    racers = await asyncio.gather(
        *(
            memory.remember(
                "a", "The bay is free.", subject=f"user:{i % 2}", metadata={"line": i % 3 == 0}
            )
            for i in range(12)
        )
    )
    keys: dict[tuple[str | None, bool], set[str]] = {
        (r.story.subject, r.story.metadata["line"]): set() for r in racers
    }
    for r in racers:
        keys[(r.story.subject, r.story.metadata["line"])].add(r.story.story_id)
    assert sorted(keys) == [
        ("user:0", False),
        ("user:0", True),
        ("user:1", False),
        ("user:1", True),
    ]
    assert all(len(ids) == 1 for ids in keys.values())
    stories, _ = await memory.list_stories("a", history=True)
    assert len(stories) == 4


async def test_one_open_story_per_text_under_concurrent_remembers(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=_Yielding())
    racers = await asyncio.gather(
        *(
            memory.remember("a", f"The go-live is  {'MARCH' if i % 2 else 'march'} 3.", subject="s")
            for i in range(6)
        )
    )
    assert len({r.story.story_id for r in racers}) == 1
    stories, _ = await memory.list_stories("a", history=True)
    assert [s.story_id for s in stories] == [racers[0].story.story_id]
    first = _story("a", "first", "Ana owns P-3.")
    assert await storage.put_story("a", first, None, None) == first
    assert (
        await storage.put_story(
            "a",
            _story("a", "second", "ana owns  p-3."),
            None,
            None,
            graph=GraphWrite(
                entities=[
                    Entity(corpus_id="a", entity_id="e1", name="Ana", key="ana", type="person")
                ],
                mentions=[("e1", "story", "second", "Ana")],
            ),
        )
        == first
    )
    assert (
        await storage.get_stories("a", ["second"]) == []
        and (await storage.count_corpus("a")).entities == 0
    )


async def test_a_supersede_racing_another_writer_is_refused_like_the_pre_check(
    storage: PostgresStorage,
) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    await storage.put_story(
        "a", _story("a", "old", "The date is Feb 20.", valid_from=jan), None, None
    )
    holder = _story("a", "holder", "The date is Mar 3.", valid_from=jan)
    await storage.put_story("a", holder, None, None)
    new = _story("a", "new", "the date is  MAR 3.", valid_from=feb).model_copy(
        update={"supersedes": ("old",)}
    )
    held = r"^postgres: story 'holder', true since 2026-01-01T00:00:00\+00:00, already says that; supersede or delete it instead$"
    with pytest.raises(InputError, match=held):
        await storage.supersede_story("a", ["old"], new, None, None)
    assert [s.valid_to for s in await storage.get_stories("a", ["old", "new"])] == [None]

    memory = Memory(storage, embedding=_Yielding())
    story = (await memory.remember("a", "The line runs nights.")).story
    racers = await asyncio.gather(
        *(
            memory.supersede("a", story.story_id, f"The line runs {when}.")
            for when in ("days", "weekends")
        ),
        return_exceptions=True,
    )
    refused = [r for r in racers if isinstance(r, BaseException)]
    assert (
        len(refused) == 1
        and isinstance(refused[0], InputError)
        and "is already superseded" in str(refused[0])
    )


async def test_sync_sources_swaps_the_matching_set_in_one_step(storage: PostgresStorage) -> None:
    line = {"metadata.file": "a"}
    pump = Entity(corpus_id="a", entity_id="e1", name="P-3", key="p-3", type="pump")
    for source_id in ("old1", "old2"):
        source, chunks, vectors = _source(
            "a", source_id, [f"pump P-3 {source_id}", f"wiring {source_id}"], metadata={"file": "a"}
        )
        await storage.put_source(
            "a",
            source,
            chunks,
            vectors,
            "m",
            graph=GraphWrite(entities=[pump], mentions=[("e1", "chunk", f"{source_id}-0", "P-3")]),
        )
    await storage.put_source(
        "a", *_source("a", "keep", ["pump elsewhere"], metadata={"file": "b"}), "m"
    )
    await storage.put_source("b", *_source("b", "old1", ["pump in b"], metadata={"file": "a"}), "m")
    tank = Entity(corpus_id="a", entity_id="e2", name="T-9", key="t-9", type="tank")
    new = SourceWrite(
        *_source("a", "new1", ["tank T-9 fills", "pump P-3 drains T-9"], metadata={"file": "a"})
    )
    done = await storage.sync_sources(
        "a",
        line,
        [new],
        "m",
        at=NOW,
        graph=GraphWrite(
            entities=[tank],
            relations=[
                Relation(
                    corpus_id="a",
                    from_id="e1",
                    to_id="e2",
                    type="DRAINS",
                    evidence_kind="chunk",
                    evidence_id="new1-1",
                )
            ],
            mentions=[
                ("e2", "chunk", "new1-0", "T-9"),
                ("e1", "chunk", "new1-1", "P-3"),
                ("e2", "chunk", "new1-1", "T-9"),
            ],
        ),
    )
    assert done is not None and done[:2] == (Counts(sources=2, chunks=4, mentions=2), [])
    listed, _ = await storage.list_sources("a", filters=line, limit=10, cursor=None)
    assert listed == [new.source]
    assert await storage.source_chunks("a", "new1") == new.chunks
    assert await storage.source_chunks("a", "old1") == []
    assert await storage.get_source("a", "keep") is not None
    assert await storage.get_source("b", "old1") is not None
    window = Window()
    assert (
        await _text(
            storage,
            "a",
            "wiring",
            kinds=("chunk",),
            top_k=10,
            filters=None,
            window=window,
            subject=None,
        )
        == []
    )
    assert ids(
        await storage.search_vector(
            "a",
            _vec("tank T-9 fills"),
            kinds=("chunk",),
            top_k=1,
            filters=None,
            window=window,
            subject=None,
        )
    ) == ["new1-0"]
    residue = await storage.count_corpus("a")
    assert (
        residue.sources,
        residue.chunks,
        residue.entities,
        residue.relations,
        residue.mentions,
    ) == (2, 3, 2, 1, 3)

    stray = GraphWrite(entities=[tank], mentions=[("e2", "chunk", "new1-0", "T-9")])
    for writes, graph, refused in (
        (
            [
                SourceWrite(
                    *_source("a", "new2", ["pump P-3 feeds tank T-9"], metadata={"file": "a"})
                )
            ],
            _bad_graph("a", "chunk", "new2-0"),
            r"graph facts name unknown entities",
        ),
        (
            [SourceWrite(*_source("a", "new2", ["tank T-9"], metadata={"file": "a"}))],
            stray,
            r"^rows: graph facts must be evidenced by a passage of the same write$",
        ),
        (
            [SourceWrite(*_source("b", "new2", ["tank T-9"], metadata={"file": "a"}))],
            None,
            r"^postgres: row corpus does not match the call corpus$",
        ),
    ):
        with pytest.raises(StorageError, match=refused):
            await storage.sync_sources("a", line, writes, "m", at=NOW, graph=graph)
    assert await storage.count_corpus("a") == residue
    assert (await storage.list_sources("a", filters=line, limit=10, cursor=None))[0] == [new.source]


async def test_a_sync_with_no_sources_removes_the_set(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    line = {"metadata.file": "a"}
    for i in range(2):
        await memory.ingest("a", name=f"v{i}", text=f"version {i}", metadata={"file": "a"})
    kept = (await memory.ingest("a", name="b", text="other file", metadata={"file": "b"})).source
    other = (await memory.ingest("b", name="v0", text="version 0", metadata={"file": "a"})).source
    synced = await memory.sync("a", filters=line, sources=[])
    assert (synced.sources, synced.deleted) == ([], Counts(sources=2, chunks=2))
    assert (await memory.list_sources("a", filters=line))[0] == []
    assert (await memory.list_sources("a"))[0] == [kept]
    assert (await memory.list_sources("b"))[0] == [other]
    residue = await memory.verify_absent("a")
    assert (residue.sources, residue.chunks) == (1, 1)
    again = await memory.sync("a", filters=line, sources=[])
    assert again.deleted.clean and (await memory.list_sources("a"))[0] == [kept]


async def test_sync_sources_keeps_what_it_is_told_and_writes_nothing_when_that_moved(
    storage: PostgresStorage,
) -> None:
    line = {"metadata.file": "a"}
    kept_source, kept_chunks, kept_vectors = _source(
        "a", "kept", ["pump P-3"], metadata={"file": "a"}
    )
    kept_source = kept_source.model_copy(update={"digest": "d-kept"})
    await storage.put_source("a", kept_source, kept_chunks, kept_vectors, "m")
    await storage.put_source("a", *_source("a", "old", ["old wiring"], metadata={"file": "a"}), "m")
    assert await storage.get_source("a", "kept") == kept_source
    new = SourceWrite(*_source("a", "new", ["new wiring"], metadata={"file": "a"}))
    before = await storage.count_corpus("a")
    for keep in ({"kept": "d-other"}, {"gone": "d-kept"}):
        assert await storage.sync_sources("a", line, [new], "m", at=NOW, keep=keep) is None
        assert await storage.count_corpus("a") == before
    assert _pair(
        await storage.sync_sources("a", line, [new], "m", at=NOW, keep={"kept": "d-kept"})
    ) == (Counts(sources=1, chunks=1), [])
    listed, _ = await storage.list_sources("a", filters=line, limit=10, cursor=None)
    assert sorted(s.source_id for s in listed) == ["kept", "new"]
    assert await storage.get_source("a", "kept") == kept_source
    assert await storage.source_chunks("a", "kept") == kept_chunks


async def test_chunk_vectors_answer_by_text_within_one_corpus_and_one_model(
    storage: PostgresStorage,
) -> None:
    await storage.put_source("a", *_source("a", "s1", ["pump seal", "wiring"]), "m")
    await storage.put_source("a", *_source("a", "s2", ["pump seal"]), "m2")
    await storage.put_source("b", *_source("b", "s3", ["gasket"]), "m")
    seal, wiring, gasket = (chunk_digest(t) for t in ("pump seal", "wiring", "gasket"))
    held = await storage.chunk_vectors("a", "m", [seal, wiring, gasket])
    assert held.keys() == {seal, wiring}
    assert held[seal] == pytest.approx(_vec("pump seal"), rel=1e-6)
    assert held[wiring] == pytest.approx(_vec("wiring"), rel=1e-6)
    other = await storage.chunk_vectors("a", "m2", [seal, wiring])
    assert other.keys() == {seal} and other[seal] == pytest.approx(_vec("pump seal"), rel=1e-6)
    assert await storage.chunk_vectors("b", "m", [seal]) == {}
    assert await storage.chunk_vectors("a", "m", []) == {}


async def test_a_corpus_language_reindexes_its_passages_and_no_other_corpus(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    for corpus in ("a", "b"):
        await memory.ingest(corpus, name="m", text="A informação das válvulas está no manual.")
        await memory.remember(corpus, "As bombas funcionam melhor.")

    async def found(corpus: str, query: str) -> list[str]:
        hits = await memory.search(corpus, query, pillars={"text"})
        return sorted(h.kind for h in hits.hits)

    assert await found("a", "informação funciona") == ["chunk"]
    assert await storage.set_language("a", "klingon") is None
    assert await storage.set_language("a", "portuguese") == 2
    assert await storage.get_language("a") == "portuguese"
    assert await storage.get_language("b") == "english"
    assert await found("a", "informação funciona") == ["chunk", "story"]
    assert await found("b", "informação funciona") == ["chunk"]
    await memory.ingest("a", name="n", text="Novas informações sobre as válvulas.")
    assert await found("a", "válvulas") == ["chunk", "chunk"]
    assert await storage.set_language("a", "portuguese") == 0
    assert (await storage.count_corpus("a")).corpora == 1
    assert (await storage.count_corpus("b")).corpora == 1
    deleted = await storage.delete_corpus("a")
    assert deleted.corpora == 1 and (await storage.count_corpus("a")).clean
    assert await storage.get_language("a") == "english"


async def test_a_write_waits_while_the_corpus_language_changes(storage: PostgresStorage) -> None:
    memory = Memory(storage)
    holder = await asyncpg.connect(os.environ[URL])
    try:
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock($1)", await _lock_key(storage, "memory.language:a")
            )
            writing = asyncio.create_task(memory.ingest("a", name="m", text="manual"))
            await asyncio.sleep(0.5)
            assert not writing.done()
        await asyncio.wait_for(writing, 10)
    finally:
        await holder.close()
    assert [s.name for s in (await memory.list_sources("a"))[0]] == ["m"]


async def test_a_hit_cites_its_source_by_name_and_metadata_on_every_read(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    ingested = await memory.ingest(
        "a",
        name="Pump manual v3",
        sections=[Section(title="Seals", text="Replace the seal yearly.", locator={"page": 27})],
        metadata={"document": "d-1", "version": 3, "site": {"zone": 2}},
    )
    hits = await memory.search("a", "seal yearly", kinds=("chunk",))
    [chunk] = [h.chunk for h in hits.hits if h.chunk is not None]
    assert (chunk.source_name, chunk.metadata, chunk.locator) == (
        "Pump manual v3",
        {"document": "d-1", "version": 3, "site": {"zone": 2}},
        {"page": 27},
    )
    assert await storage.source_chunks("a", ingested.source.source_id) == [chunk]
    found = await MemoryTools(memory, corpus_id="a").dispatch("search", {"query": "seal yearly"})
    assert isinstance(found, dict)
    assert (found["hits"][0]["source_name"], found["hits"][0]["metadata"]) == (
        "Pump manual v3",
        {"document": "d-1", "version": 3, "site": {"zone": 2}},
    )


async def test_a_deleted_passage_takes_the_names_only_it_carried(storage: PostgresStorage) -> None:
    person = Entity(
        corpus_id="a",
        entity_id="e1",
        name="John Smith",
        key="john smith",
        type="person",
        aliases=("J. Smith", "Johnny"),
    )
    plant = Entity(corpus_id="a", entity_id="e2", name="Plant 4", key="plant 4", type="site")
    await storage.put_source(
        "a",
        *_source("a", "memo", ["John Smith runs Plant 4."]),
        "m",
        graph=GraphWrite(
            entities=[person, plant],
            mentions=[
                ("e1", "chunk", "memo-0", "John Smith"),
                ("e2", "chunk", "memo-0", "Plant 4"),
            ],
        ),
    )
    await storage.put_source(
        "a",
        *_source("a", "notes", ["J. Smith signed it."]),
        "m",
        graph=GraphWrite(mentions=[("e1", "chunk", "notes-0", "J. Smith")]),
    )
    await storage.put_source(
        "a",
        *_source("a", "chat", ["Johnny said hi."]),
        "m",
        graph=GraphWrite(mentions=[("e1", "chunk", "chat-0", "Johnny")]),
    )
    assert (await storage.delete_source("a", "chat")).entities == 0
    [kept] = (await storage.entities_named("a", ["John Smith"]))[name_key("John Smith")]
    assert (kept.name, kept.aliases) == ("John Smith", ("J. Smith",))
    assert (await storage.entities_named("a", ["Johnny"]))[name_key("Johnny")] == []
    deleted = await storage.delete_source("a", "memo")
    assert (deleted.sources, deleted.mentions, deleted.entities) == (1, 2, 1)
    assert (await storage.entities_named("a", ["Plant 4"]))[name_key("Plant 4")] == []
    assert (await storage.entities_named("a", ["John Smith"]))[name_key("John Smith")] == []
    [renamed] = (await storage.entities_named("a", ["J. Smith"]))[name_key("J. Smith")]
    assert (renamed.entity_id, renamed.name, renamed.key, renamed.aliases) == (
        "e1",
        "J. Smith",
        "j. smith",
        (),
    )
    assert (await storage.delete_source("a", "notes")).entities == 1
    assert await storage.count_corpus("a") == Counts(corpora=1)


async def test_an_entity_still_related_or_named_by_a_kept_source_stays(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    line = {"metadata.file": "a"}
    pump = EntityIn(name="P-3", type="pump")
    sources = [
        SourceIn(name="manual", text="P-3 feeds T-9.", metadata={"file": "a"}, entities=[pump]),
        SourceIn(name="memo", text="P-3 was serviced.", metadata={"file": "a"}, entities=[pump]),
    ]
    first = await memory.sync("a", filters=line, sources=sources)
    [before] = (await storage.entities_named("a", ["P-3"]))[name_key("P-3")]
    second = await memory.sync("a", filters=line, sources=sources[:1])
    assert second.kept == [first.sources[0].source_id] and second.deleted.entities == 0
    assert (await storage.entities_named("a", ["P-3"]))[name_key("P-3")] == [before]
    await memory.ingest(
        "a",
        name="links",
        text="Bo runs P-3.",
        entities=[EntityIn(name="Bo", type="person")],
        relations=[RelationIn(from_name="Bo", to_name="P-3", type="RUNS")],
    )
    assert (await memory.sync("a", filters=line, sources=[])).deleted.entities == 0
    assert (await storage.entities_named("a", ["P-3"]))[name_key("P-3")] == [before]


async def test_deleting_every_document_leaves_no_graph_name_behind(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    for i in range(3):
        await memory.ingest(
            "a",
            name=f"doc {i}",
            text=f"V-{i} feeds P-1.",
            metadata={"doc": i},
            entities=[EntityIn(name=f"V-{i}", type="valve"), EntityIn(name="P-1", type="pump")],
            relations=[RelationIn(from_name=f"V-{i}", to_name="P-1", type="FEEDS")],
        )
    for source in (await memory.list_sources("a"))[0]:
        await memory.delete_source("a", source.source_id)
    assert await memory.verify_absent("a") == Counts(corpora=1)


async def test_a_write_racing_the_prune_of_an_entity_it_names_brings_it_back(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    pump = EntityIn(name="P-3", type="pump")
    outcomes: set[str] = set()
    for i in range(25):
        corpus = f"race-{i}"
        old = await memory.ingest(corpus, name="old", text="P-3 runs.", entities=[pump])
        deleted, written = await asyncio.gather(
            memory.delete_source(corpus, old.source.source_id),
            memory.ingest(corpus, name="new", text="P-3 stops.", entities=[pump]),
            return_exceptions=True,
        )
        assert isinstance(deleted, Counts) and not isinstance(written, BaseException)
        outcomes.add("pruned" if deleted.entities else "kept")
        async with storage._pool.acquire() as conn:
            dangling = await conn.fetchval(
                "SELECT count(*) FROM memory_mentions m WHERE m.corpus_id = $1 AND NOT EXISTS "
                "(SELECT 1 FROM memory_entities e WHERE e.corpus_id = m.corpus_id AND e.entity_id = m.entity_id)",
                corpus,
            )
        assert dangling == 0
        [named] = (await storage.entities_named(corpus, ["P-3"]))[name_key("P-3")]
        assert named.name == "P-3"
        assert [s.name for s in (await memory.list_sources(corpus))[0]] == ["new"]
    assert "pruned" in outcomes


def _acme(aliases: tuple[str, ...] = ("Acme",)) -> Entity:
    return Entity(
        corpus_id="a", entity_id="e1", name="Acme Inc", key="acme inc", type="org", aliases=aliases
    )


async def _named_in(
    storage: PostgresStorage, source_id: str, text: str, name: str, *new: Entity
) -> None:
    await storage.put_source(
        "a",
        *_source("a", source_id, [text]),
        "m",
        graph=GraphWrite(entities=list(new), mentions=[("e1", "chunk", f"{source_id}-0", name)]),
    )


async def _write_after(storage: PostgresStorage, source_id: str) -> None:
    original = storage.put_source

    async def racing(*args: Any, **kwargs: Any) -> Any:
        await storage.delete_source("a", source_id)
        return await original(*args, **kwargs)

    storage.put_source = racing  # type: ignore[method-assign]


@pytest.mark.parametrize("dropped", ["alias", "name"])
async def test_a_write_keeps_its_name_on_its_entity_when_a_delete_trims_it_meanwhile(
    storage: PostgresStorage, dropped: str
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    gone, kept, written = (
        ("Acme", "Acme Inc", "Acme")
        if dropped == "alias"
        else (
            "Acme Inc",
            "Acme",
            "Acme Inc",
        )
    )
    await _named_in(storage, "d1", f"{gone} ships.", gone, _acme())
    await _named_in(storage, "d2", f"{kept} ships.", kept)
    await _write_after(storage, "d1")
    await memory.ingest(
        "a", name="d3", text=f"{written} again.", entities=[EntityIn(name=written, type="org")]
    )
    for name in ("Acme", "Acme Inc"):
        assert [
            e.entity_id for e in (await storage.entities_named("a", [name]))[name_key(name)]
        ] == ["e1"], name


async def test_a_name_used_only_through_a_relation_is_evidence(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    pump = Entity(
        corpus_id="a", entity_id="e1", name="P-3", key="p-3", type="pump", aliases=("Pump three",)
    )
    await storage.put_source(
        "a",
        *_source("a", "a1", ["Pump three hums."]),
        "m",
        graph=GraphWrite(entities=[pump], mentions=[("e1", "chunk", "a1-0", "Pump three")]),
    )
    await storage.put_source(
        "a",
        *_source("a", "m1", ["P-3 runs."]),
        "m",
        graph=GraphWrite(mentions=[("e1", "chunk", "m1-0", "P-3")]),
    )
    await memory.ingest(
        "a",
        name="links",
        text="Bo runs Pump three.",
        entities=[EntityIn(name="Bo", type="person")],
        relations=[RelationIn(from_name="Bo", to_name="Pump three", type="RUNS")],
    )
    await storage.delete_source("a", "a1")
    [kept] = (await storage.entities_named("a", ["Pump three"]))[name_key("Pump three")]
    assert (kept.entity_id, kept.name, kept.aliases) == ("e1", "P-3", ("Pump three",))


async def test_an_entity_related_elsewhere_still_loses_a_name_only_deleted_text_used(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    person = Entity(
        corpus_id="a",
        entity_id="e1",
        name="John Smith",
        key="john smith",
        type="person",
        aliases=("Johnny",),
    )
    await storage.put_source(
        "a",
        *_source("a", "chat", ["Johnny said hi."]),
        "m",
        graph=GraphWrite(entities=[person], mentions=[("e1", "chunk", "chat-0", "Johnny")]),
    )
    await memory.ingest(
        "a",
        name="org",
        text="John Smith runs Plant 4.",
        entities=[EntityIn(name="Plant 4", type="site")],
        relations=[RelationIn(from_name="John Smith", to_name="Plant 4", type="RUNS")],
    )
    await storage.delete_source("a", "chat")
    assert (await storage.entities_named("a", ["Johnny"]))[name_key("Johnny")] == []
    [kept] = (await storage.entities_named("a", ["John Smith"]))[name_key("John Smith")]
    assert (kept.entity_id, kept.aliases) == ("e1", ())


async def test_two_entities_renamed_to_one_freed_name_at_once_leave_one_holder(
    storage: PostgresStorage,
) -> None:
    for entity_id, name, doc in (("e1", "Acme Corp", "c"), ("e2", "Acme Ltd", "l")):
        entity = Entity(
            corpus_id="a",
            entity_id=entity_id,
            name=name,
            key=name.lower(),
            type="org",
            aliases=("Acme",),
        )
        await storage.put_source(
            "a",
            *_source("a", f"{doc}-full", [f"{name} ships."]),
            "m",
            graph=GraphWrite(
                entities=[entity], mentions=[(entity_id, "chunk", f"{doc}-full-0", name)]
            ),
        )
        await storage.put_source(
            "a",
            *_source("a", f"{doc}-short", ["Acme ships."]),
            "m",
            graph=GraphWrite(mentions=[(entity_id, "chunk", f"{doc}-short-0", "Acme")]),
        )
    results = await asyncio.gather(
        storage.delete_source("a", "c-full"), storage.delete_source("a", "l-full")
    )
    assert all(r.sources == 1 for r in results)
    holders = (await storage.entities_named("a", ["Acme"]))[name_key("Acme")]
    assert sorted(e.entity_id for e in holders) == ["e1", "e2"]
    assert sorted(e.name for e in holders) in (["Acme", "Acme Ltd"], ["Acme", "Acme Corp"])


async def test_a_storage_recreates_only_known_entities_the_write_names(
    storage: PostgresStorage,
) -> None:
    ghost = Entity(corpus_id="a", entity_id="ghost", name="Ghost", key="ghost", type="t")
    done = await storage.sync_sources(
        "a",
        {"metadata.file": "a"},
        [SourceWrite(*_source("a", "s1", ["plain"], metadata={"file": "a"}))],
        "m",
        at=NOW,
        graph=GraphWrite(known=[ghost]),
    )
    assert done is not None and done[0].clean and done[1] == []
    assert (await storage.count_corpus("a")).entities == 0


async def test_a_chunk_reads_back_its_source_name_and_metadata_whatever_it_was_written_with(
    storage: PostgresStorage,
) -> None:
    source, chunks, vectors = _source("a", "s1", ["text"], metadata={"line": "A"})
    stray = [c.model_copy(update={"source_name": "stray", "metadata": {"x": 1}}) for c in chunks]
    await storage.put_source("a", source, stray, vectors, "m")
    [chunk] = await storage.source_chunks("a", "s1")
    assert (chunk.source_name, chunk.metadata) == ("s1", {"line": "A"})


async def test_concurrent_syncs_of_one_set_leave_exactly_one(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=_Yielding())
    line = {"metadata.file": "a"}
    await memory.ingest("a", name="v0", text="version zero", metadata={"file": "a"})
    for attempt in range(3):
        racers = await asyncio.gather(
            *(
                memory.sync(
                    "a",
                    filters=line,
                    sources=[
                        SourceIn(
                            name=f"{who} {i}",
                            text=f"{who} {attempt} part {i}",
                            metadata={"file": "a"},
                        )
                        for i in range(3)
                    ],
                )
                for who in ("left", "right")
            )
        )
        listed, _ = await memory.list_sources("a", filters=line)
        assert {s.source_id for s in listed} in [{s.source_id for s in r.sources} for r in racers]
        assert sum(r.deleted.sources for r in racers) == (1 if attempt == 0 else 3) + 3
        residue = await memory.verify_absent("a")
        assert (residue.sources, residue.chunks) == (3, 3)


async def test_syncs_with_overlapping_filters_serialize(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=_Yielding())
    for attempt in range(3):
        await asyncio.gather(
            memory.sync(
                "a",
                filters={"metadata.file": "a"},
                sources=[SourceIn(name="left a", text=f"left {attempt}", metadata={"file": "a"})],
            ),
            memory.sync(
                "a",
                filters={"metadata.file": ["a", "b"]},
                sources=[
                    SourceIn(name="right a", text=f"right a {attempt}", metadata={"file": "a"}),
                    SourceIn(name="right b", text=f"right b {attempt}", metadata={"file": "b"}),
                ],
            ),
        )
        listed, _ = await memory.list_sources("a", filters={"metadata.file": ["a", "b"]})
        assert sorted(s.name for s in listed) in (["right a", "right b"], ["left a", "right b"])


async def test_a_sync_takes_its_turn_under_the_corpus_lock(storage: PostgresStorage) -> None:
    memory = Memory(storage)
    holder = await asyncpg.connect(os.environ[URL])
    try:
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock($1)", await _lock_key(storage, "memory.sync:a")
            )
            syncing = asyncio.create_task(
                memory.sync(
                    "a",
                    filters={"metadata.file": "a"},
                    sources=[SourceIn(name="v1", text="version one", metadata={"file": "a"})],
                )
            )
            await asyncio.sleep(0.5)
            assert not syncing.done()
        await asyncio.wait_for(syncing, 10)
    finally:
        await holder.close()
    assert [s.name for s in (await memory.list_sources("a"))[0]] == ["v1"]


async def test_a_sync_that_fails_inside_its_transaction_leaves_the_old_set(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    line = {"metadata.file": "a"}
    old = (await memory.ingest("a", name="v0", text="version zero", metadata={"file": "a"})).source
    before = await memory.verify_absent("a")
    monkeypatch.setattr(
        storage, "_insert_chunk_sql", lambda: "INSERT INTO memory_nowhere VALUES ($1)"
    )
    with pytest.raises(StorageError, match=r"^postgres: storage rejected sync_sources: "):
        await memory.sync(
            "a",
            filters=line,
            sources=[SourceIn(name="v1", text="version one", metadata={"file": "a"})],
        )
    monkeypatch.undo()
    assert await memory.verify_absent("a") == before
    assert await memory.list_sources("a", filters=line) == ([old], None)
    assert await memory.read_source("a", old.source_id) == "version zero"


async def test_a_memory_without_embedding_leaves_the_vector_pillar_of_another_whole(
    storage: PostgresStorage,
) -> None:
    vectors = Memory(storage, embedding=HashEmbedding())
    await vectors.check()
    await Memory(storage).check()
    await vectors.ingest("a", name="manual", text="pump seal yearly")
    hits = await vectors.search("a", "pump seal yearly", pillars={"vector"})
    assert hits.pillars["vector"].error is None and len(hits.hits) == 1


async def test_a_cursor_no_listing_returned_is_refused(storage: PostgresStorage) -> None:
    memory = Memory(storage)
    for i in range(3):
        await memory.ingest("a", name=f"n{i}", text=f"alpha {i}")
        await memory.remember("a", f"story {i}")
    first, cursor = await memory.list_sources("a", limit=2)
    assert len(first) == 2 and cursor is not None
    assert len((await memory.list_sources("a", limit=2, cursor=cursor))[0]) == 1
    last = str(2**63 - 1)
    assert await memory.list_sources("a", cursor=last) == ([], None)
    assert await memory.list_stories("a", cursor=last) == ([], None)
    tools = MemoryTools(memory, corpus_id="a")
    for bad in ("-1", "1.5", "abc", " 2", "²", str(2**63), "9" * 20):
        refused = rf"^memory: cursor {re.escape(repr(bad))} is not one a listing returned$"
        with pytest.raises(InputError, match=refused):
            await memory.list_sources("a", cursor=bad)
        with pytest.raises(InputError, match=refused):
            await memory.list_stories("a", cursor=bad)
        for name in ("list_sources", "list_stories"):
            with pytest.raises(InputError, match=refused):
                await tools.dispatch(name, {"cursor": bad})


async def test_deleting_a_story_erases_its_mentions_relations_and_the_entities_only_it_named(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    pump = EntityIn(name="P-3", type="pump")
    kept = (
        await memory.remember(
            "a",
            "Bo runs P-3.",
            entities=[EntityIn(name="Bo", type="person"), pump],
            relations=[RelationIn(from_name="Bo", to_name="P-3", type="RUNS")],
        )
    ).story
    story = (
        await memory.remember(
            "a",
            "Ana owns P-3.",
            entities=[EntityIn(name="Ana", type="person"), pump],
            relations=[RelationIn(from_name="Ana", to_name="P-3", type="OWNS")],
        )
    ).story
    assert (await memory.delete_story("a", story.story_id)) == Counts(
        stories=1, mentions=2, relations=1, entities=1
    )
    residue = await storage.count_corpus("a")
    assert (residue.stories, residue.mentions, residue.relations, residue.entities) == (1, 2, 1, 2)
    assert (await storage.entities_named("a", ["Ana"]))[name_key("Ana")] == []
    assert [e.name for e in (await storage.entities_named("a", ["P-3"]))[name_key("P-3")]] == [
        "P-3"
    ]
    assert await storage.get_stories("a", [story.story_id]) == []
    hood = await memory.neighbors("a", "P-3", history=True)
    assert [r.type for r in hood.relations] == ["RUNS"]
    assert [h.id for h in hood.passages] == [kept.story_id]
    assert (await storage.delete_story("a", story.story_id)).clean


async def test_synced_sources_are_searched_like_ingested_ones(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    line = {"metadata.file": "a"}
    await memory.ingest(
        "a", name="v0", text="The gasket is swapped monthly.", metadata={"file": "a"}
    )
    synced = await memory.sync(
        "a",
        filters=line,
        sources=[
            SourceIn(
                name="manual",
                sections=[Section(title="Seal replacement", text="Do it every twelve months.")],
                metadata={"file": "a"},
            ),
            SourceIn(name="notes", text="Ground the panels first.", metadata={"file": "a"}),
            *(
                SourceIn(name=f"copy {i}", text="pump seal", metadata={"file": "a"})
                for i in range(3)
            ),
        ],
    )
    manual, notes, *copies = synced.sources

    async def found(query: str) -> list[str]:
        hits = await memory.search("a", query, pillars={"text"})
        return sorted({h.chunk.source_id for h in hits.hits if h.chunk is not None})

    assert await found("replacing") == [manual.source_id]
    assert await found("grounded panel") == [notes.source_id]
    assert await found("gaskets") == []
    assert await found("what is the") == []
    tied = sorted(f"{s.source_id}-0" for s in copies)
    window = Window()
    text = await _text(
        storage,
        "a",
        "pump seal",
        kinds=("chunk",),
        top_k=10,
        filters=None,
        window=window,
        subject=None,
    )
    assert ids(text)[:3] == tied
    vector = await storage.search_vector(
        "a", _vec("pump seal"), kinds=("chunk",), top_k=3, filters=None, window=window, subject=None
    )
    assert ids(vector) == tied


async def test_source_digests_name_each_source_by_its_name_metadata_and_text(
    storage: PostgresStorage,
) -> None:
    metadata: dict[str, object] = {
        "line": "north",
        "torque": 410,
        "weight": 0.5,
        "tags": ["press", "die"],
        "owner": {"name": "Mei", "ext": 318, "backup": None},
    }
    source, chunks, vectors = _source(
        "a",
        "s1",
        ["Tighten to 410 N·m — cross pattern.", "Re-torque after 50 strokes."],
        metadata=metadata,
    )
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    await storage.put_source("b", *_source("b", "s2", ["b's own text"]), "hash-8")
    assert await storage.source_digests("a") == [
        ("s1", metadata, sha256(source.text.encode("utf-8")).hexdigest(), 2, None, None)
    ]
    assert [name for name, *_ in await storage.source_digests("b")] == ["s2"]
    assert await storage.source_digests("c") == []


async def test_a_fingerprint_follows_content_not_ids(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    manual = SourceIn(name="manual", text="Replace the seal yearly.", metadata={"file": "m"})
    first = await memory.ingest("a", name=manual.name, text=manual.text, metadata=manual.metadata)
    await memory.ingest("b", name="handbook", text="Boilers are inspected every spring.")
    before, other = await memory.fingerprint("a"), await memory.fingerprint("b")
    assert await memory.fingerprint("a") == before
    await memory.delete_source("a", first.source.source_id)
    await memory.ingest("a", name=manual.name, text=manual.text, metadata=manual.metadata)
    assert await memory.fingerprint("a") == before
    await memory.sync("a", filters={"metadata.file": "m"}, sources=[manual])
    assert await memory.fingerprint("a") == before
    reworded = manual.model_copy(update={"text": "Replace the seal twice a year."})
    await memory.sync("a", filters={"metadata.file": "m"}, sources=[reworded])
    changed = await memory.fingerprint("a")
    retagged = manual.model_copy(update={"metadata": {"file": "m", "rev": 2}})
    await memory.sync("a", filters={"metadata.file": "m"}, sources=[retagged])
    tagged = await memory.fingerprint("a")
    assert len({before.content, changed.content, tagged.content}) == 3
    await memory.remember("a", "Ana owns the seal plan.", subject="user:ana")
    told = await memory.fingerprint("a")
    assert (told.content, told.sources, told.chunks) == (tagged.content, 1, 1)
    assert told.stories == 1 and told.story_content != tagged.story_content
    assert await memory.fingerprint("b") == other


async def test_count_vectors_counts_passages_by_the_model_of_their_vector(
    storage: PostgresStorage,
) -> None:
    source, chunks, vectors = _source("a", "s1", ["one", "two", "three"])
    await storage.put_source("a", source, chunks, vectors, "old-model")
    await storage.put_story("a", _story("a", "st1", "four"), _vec("four"), "old-model")
    await storage.put_story("a", _story("a", "st2", "five"), None, None)
    await storage.put_story("a", _story("a", "st3", "six"), None, "old-model")
    await storage.put_story("a", _story("a", "st4", "seven"), _vec("seven"), None)
    await storage.set_vectors("a", [("chunk", "s1-2")], [_vec("x")], "new-model")
    await storage.put_source("b", *_source("b", "s2", ["b's own text"]), "b-model")
    assert await storage.count_vectors("a") == {"old-model": 3, "new-model": 1, None: 3}
    assert await storage.count_vectors("b") == {"b-model": 1}
    assert await storage.count_vectors("c") == {}


async def test_a_fingerprint_counts_stale_vectors(storage: PostgresStorage) -> None:
    renamed = HashEmbedding()
    renamed.model = "hash-8-v2"
    old = Memory(storage, embedding=HashEmbedding(), chunk_chars=60, overlap_chars=10)
    new = Memory(storage, embedding=renamed, chunk_chars=60, overlap_chars=10)
    await old.ingest(
        "a",
        name="die change rev B",
        text="Tighten the clamp bolts to 410 N·m in a cross pattern.\n\nRe-torque after 50 strokes.",
    )
    await old.remember("a", "Mei signs the die log.")
    await old.ingest("b", name="handbook", text="Boilers are inspected every spring.")
    clean = await old.fingerprint("a")
    rows = clean.chunks + clean.stories
    assert rows > 2 and (clean.embedded, clean.stale) == ({"hash-8": rows}, 0)
    moved = await new.fingerprint("a")
    assert (moved.embedding, moved.embedded, moved.stale) == ("hash-8-v2", {"hash-8": rows}, rows)
    await new.remember("a", "Rui signs the die log.")
    half = await new.fingerprint("a")
    assert (half.embedded, half.stale) == ({"hash-8": rows, "hash-8-v2": 1}, rows)
    assert await new.reembed("a") == rows
    done = await new.fingerprint("a")
    assert (done.embedded, done.stale) == ({"hash-8-v2": rows + 1}, 0)
    assert (done.content, done.story_content) == (half.content, half.story_content)
    text = await Memory(storage).fingerprint("a")
    assert (text.embedding, text.embedded, text.stale) == (None, {"hash-8-v2": rows + 1}, 0)
    other = await old.fingerprint("b")
    assert (other.embedded, other.stale) == ({"hash-8": 1}, 0)


async def test_a_supersede_records_who_closed_the_story_and_when(storage: PostgresStorage) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    old = _story("a", "old", "The date is Feb 20.", valid_from=jan)
    await storage.put_story("a", old, None, None)
    new = _story("a", "new", "The date is Mar 3.", valid_from=feb).model_copy(
        update={"supersedes": ("old",), "created_at": NOW + timedelta(hours=1)}
    )
    [closed], created = await storage.supersede_story("a", ["old"], new, None, None)
    assert (closed.valid_to, closed.superseded_by, closed.closed_at) == (feb, "new", new.created_at)
    assert created.supersedes == ("old",) and created.superseded_by is None
    assert await storage.get_stories("a", ["old", "new"]) == [closed, created]


async def test_open_means_not_superseded_and_not_ended(storage: PostgresStorage) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    april = datetime(2026, 4, 1, tzinfo=UTC)
    ended = _story("a", "ended", "The promo runs.", valid_from=jan, valid_to=feb)
    assert await storage.put_story("a", ended, None, None) == ended
    hashed = text_hash("The promo runs.")
    assert await storage.find_open_story("a", hashed, **_BARE, at=NOW) is None
    assert await storage.find_open_story("a", hashed, **_BARE, at=jan + timedelta(days=1)) == ended
    repeat = _story("a", "repeat", "the promo  runs.", valid_from=jan, valid_to=feb)
    assert await storage.put_story("a", repeat, None, None) == ended
    fresh = _story("a", "fresh", "The promo runs.", valid_from=NOW)
    assert await storage.put_story("a", fresh, None, None) == fresh
    planned = _story("a", "planned", "The fair is on.", valid_from=jan, valid_to=april)
    await storage.put_story("a", planned, None, None)
    assert (
        await storage.put_story("a", _story("a", "dup", "The fair is on."), None, None) == planned
    )
    later = _story("a", "later", "The fair moved.", valid_from=feb).model_copy(
        update={"supersedes": ("ended",)}
    )
    with pytest.raises(
        InputError,
        match=r"^postgres: story 'ended' ended at 2026-02-01T00:00:00\+00:00; remember a new story instead$",
    ):
        await storage.supersede_story("a", ["ended"], later, None, None)


async def test_end_story_closes_one_story_with_no_successor_once(
    storage: PostgresStorage,
) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    told = _story("a", "s", "Ana prefers Tuesday.", valid_from=jan)
    await storage.put_story("a", told, None, None)
    with pytest.raises(NotFoundError, match=r"^postgres: story 's' was not found$"):
        await storage.end_story("b", "s", valid_to=feb, closed_at=NOW)
    with pytest.raises(
        InputError,
        match=r"^postgres: valid_to must not be before story 's' became true, at 2026-01-01T00:00:00\+00:00$",
    ):
        await storage.end_story("a", "s", valid_to=jan - timedelta(days=1), closed_at=NOW)
    ended = await storage.end_story("a", "s", valid_to=feb, closed_at=NOW)
    assert ended == told.model_copy(update={"valid_to": feb, "closed_at": NOW, "closed_by": {}})
    assert await storage.get_stories("a", ["s"]) == [ended]
    with pytest.raises(
        InputError, match=r"^postgres: story 's' already ended, at 2026-02-01T00:00:00\+00:00$"
    ):
        await storage.end_story("a", "s", valid_to=jan + timedelta(days=10), closed_at=NOW)
    later = _story("a", "later", "Ana prefers Monday.", valid_from=jan + timedelta(days=10))
    with pytest.raises(
        InputError,
        match=r"^postgres: story 's' ended at 2026-02-01T00:00:00\+00:00; remember a new story instead$",
    ):
        await storage.supersede_story(
            "a", ["s"], later.model_copy(update={"supersedes": ("s",)}), None, None
        )
    assert await storage.get_stories("a", ["s", "later"]) == [ended]
    await storage.put_story(
        "a", _story("a", "old", "Bo prefers Tuesday.", valid_from=jan), None, None
    )
    new = _story("a", "new", "Bo prefers Monday.", valid_from=feb)
    await storage.supersede_story(
        "a", ["old"], new.model_copy(update={"supersedes": ("old",)}), None, None
    )
    with pytest.raises(
        InputError,
        match=r"^postgres: story 'old' is already superseded; end the story that replaced it$",
    ):
        await storage.end_story("a", "old", valid_to=jan + timedelta(days=10), closed_at=NOW)


async def test_a_closing_waits_for_one_in_flight_and_is_refused_by_it(
    storage: PostgresStorage,
) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    lock = "SELECT 1 FROM memory_stories WHERE corpus_id = 'a' AND story_id = $1 FOR UPDATE"
    for story_id in ("s1", "s2"):
        story = _story("a", story_id, f"Ana prefers {story_id}.", valid_from=jan)
        await storage.put_story("a", story, None, None)
    async with storage._pool.acquire() as conn, conn.transaction():
        await conn.execute(lock, "s1")
        ending = asyncio.create_task(storage.end_story("a", "s1", valid_to=feb, closed_at=NOW))
        await asyncio.sleep(0.2)
        assert not ending.done()
        await conn.execute(
            "UPDATE memory_stories SET superseded_by = 'x', valid_to = $2, closed_at = $3 "
            "WHERE corpus_id = 'a' AND story_id = $1",
            "s1",
            jan + timedelta(days=1),
            NOW,
        )
    with pytest.raises(
        InputError,
        match=r"^postgres: story 's1' is already superseded; end the story that replaced it$",
    ):
        await ending
    later = _story("a", "later", "Ana prefers Monday.", valid_from=jan + timedelta(days=1))
    async with storage._pool.acquire() as conn, conn.transaction():
        await conn.execute(lock, "s2")
        superseding = asyncio.create_task(
            storage.supersede_story(
                "a", ["s2"], later.model_copy(update={"supersedes": ("s2",)}), None, None
            )
        )
        await asyncio.sleep(0.2)
        assert not superseding.done()
        await conn.execute(
            "UPDATE memory_stories SET valid_to = $2, closed_at = $3 "
            "WHERE corpus_id = 'a' AND story_id = $1",
            "s2",
            feb,
            NOW,
        )
    with pytest.raises(
        InputError,
        match=r"^postgres: story 's2' ended at 2026-02-01T00:00:00\+00:00; remember a new story instead$",
    ):
        await superseding
    assert await storage.get_stories("a", ["later"]) == []


async def test_a_closing_at_a_storys_own_start_records_it_was_never_true(
    storage: PostgresStorage,
) -> None:
    jan, april = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC)
    told = _story("a", "s", "Ana prefers Tuesday.", valid_from=jan)
    await storage.put_story("a", told, None, None)
    retracted = await storage.end_story("a", "s", valid_to=jan, closed_at=NOW)
    assert retracted == told.model_copy(update={"valid_to": jan, "closed_at": NOW, "closed_by": {}})
    assert await storage.get_stories("a", ["s"]) == [retracted]
    before = jan - timedelta(days=1)
    assert await storage.find_open_story("a", text_hash(told.text), **_BARE, at=before) is None
    with pytest.raises(
        InputError, match=r"^postgres: story 's' was retracted at 2026-03-01T00:00:00\+00:00$"
    ):
        await storage.end_story("a", "s", valid_to=jan, closed_at=NOW)
    instead = _story("a", "instead", "Ana prefers Monday.", valid_from=jan)
    with pytest.raises(
        InputError,
        match=r"^postgres: story 's' was retracted at 2026-03-01T00:00:00\+00:00; remember a new story instead$",
    ):
        await storage.supersede_story(
            "a", ["s"], instead.model_copy(update={"supersedes": ("s",)}), None, None
        )
    planned = _story("a", "p", "The new line opens.", valid_from=april)
    await storage.put_story("a", planned, None, None)
    await storage.end_story("a", "p", valid_to=april, closed_at=NOW)
    hashed = text_hash("The new line opens.")
    assert await storage.find_open_story("a", hashed, **_BARE, at=NOW) is None
    await storage.put_story(
        "a", _story("a", "b", "Bo prefers Tuesday.", valid_from=jan), None, None
    )
    early = _story("a", "early", "Bo prefers Monday.", valid_from=jan - timedelta(days=1))
    with pytest.raises(
        InputError,
        match=r"^postgres: valid_from must not be before story 'b' became true, at 2026-01-01T00:00:00\+00:00$",
    ):
        await storage.supersede_story(
            "a", ["b"], early.model_copy(update={"supersedes": ("b",)}), None, None
        )
    fixed = _story("a", "fixed", "Bo prefers Monday.", valid_from=jan)
    closed, _ = await storage.supersede_story(
        "a", ["b"], fixed.model_copy(update={"supersedes": ("b",)}), None, None
    )
    assert [(s.valid_from, s.valid_to, s.superseded_by) for s in closed] == [(jan, jan, "fixed")]


async def test_supersede_closes_several_stories_all_or_nothing(storage: PostgresStorage) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    for story_id, text in (("a", "Ana prefers Tuesday."), ("b", "Ana prefers mornings.")):
        await storage.put_story("a", _story("a", story_id, text, valid_from=jan), None, None)
    done = _story("a", "done", "Ana left.", valid_from=jan).model_copy(
        update={"superseded_by": "x", "valid_to": jan + timedelta(days=1)}
    )
    await storage.put_story("a", done, None, None)
    created_at = NOW + timedelta(hours=1)
    new = _story("a", "new", "Ana prefers Tuesday mornings.", valid_from=feb).model_copy(
        update={"supersedes": ("a", "b"), "created_at": created_at}
    )
    bad = new.model_copy(update={"story_id": "bad", "supersedes": ("a", "done")})
    with pytest.raises(
        InputError,
        match=r"^postgres: story 'done' is already superseded; supersede the story that replaced it$",
    ):
        await storage.supersede_story("a", ["a", "done"], bad, None, None)
    untouched = await storage.get_stories("a", ["a", "bad"])
    assert [(s.story_id, s.superseded_by, s.valid_to) for s in untouched] == [("a", None, None)]
    for ids in ([], ["a", "a"]):
        with pytest.raises(
            InputError, match=r"^postgres: supersede_story needs distinct story ids$"
        ):
            await storage.supersede_story("a", ids, new, None, None)
    closed, created = await storage.supersede_story("a", ["b", "a"], new, None, None)
    assert [s.story_id for s in closed] == ["b", "a"]
    for s in closed:
        assert (s.valid_to, s.superseded_by, s.closed_at) == (feb, "new", created_at)
    stored = await storage.get_stories("a", ["new"])
    assert stored == [created] and created.supersedes == ("a", "b")


def _on(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


async def test_known_at_answers_as_memory_knew_then(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 1, 10))
    old = (
        await memory.remember(
            "a", "The go-live is February 20.", subject="apollo", valid_from=_on(2026, 1, 1)
        )
    ).story
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 2, 10))
    new = (
        await memory.supersede(
            "a", old.story_id, "The go-live moved to March 3.", valid_from=_on(2026, 2, 1)
        )
    ).story
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 3, 1))

    async def recalled(**when: Any) -> list[str]:
        hits = await memory.recall("a", "go-live", subject="apollo", **when)
        return [h.id for h in hits.hits]

    assert await recalled() == [new.story_id]
    assert await recalled(as_of=_on(2026, 1, 20)) == [old.story_id]
    assert await recalled(known_at=_on(2026, 2, 5)) == [old.story_id]
    assert await recalled(known_at=datetime(2026, 2, 5)) == [old.story_id]
    assert await recalled(known_at=_on(2026, 1, 5)) == []
    assert await recalled(known_at=_on(2026, 3, 1), as_of=_on(2026, 2, 5)) == [new.story_id]
    assert await recalled(known_at=_on(2027, 1, 1)) == [new.story_id]
    [seen] = (await memory.recall("a", "go-live", subject="apollo", known_at=_on(2026, 2, 5))).hits
    assert seen.story is not None
    assert (seen.story.valid_to, seen.story.superseded_by, seen.story.closed_at) == (
        None,
        None,
        None,
    )
    then, _ = await memory.list_stories("a", known_at=_on(2026, 2, 5), history=True)
    assert [s.story_id for s in then] == [old.story_id] and then[0].valid_to is None
    searched = await memory.search("a", "go-live", kinds=("story",), known_at=_on(2026, 2, 5))
    assert [h.id for h in searched.hits] == [old.story_id]
    [closed] = (
        await memory.recall(
            "a", "go-live", subject="apollo", known_at=_on(2026, 3, 1), as_of=_on(2026, 1, 20)
        )
    ).hits
    assert closed.story is not None
    assert (closed.story.valid_to, closed.story.superseded_by, closed.story.closed_at) == (
        _on(2026, 2, 1),
        new.story_id,
        _on(2026, 2, 10),
    )


async def test_known_at_shows_the_end_a_story_was_written_with(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 2, 20))
    promo = (
        await memory.remember(
            "a",
            "The spring promotion runs.",
            valid_from=_on(2026, 3, 1),
            valid_to=_on(2026, 3, 31),
        )
    ).story
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 3, 16))
    early = (
        await memory.supersede(
            "a", promo.story_id, "The spring promotion ended early.", valid_from=_on(2026, 3, 15)
        )
    ).story
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 4, 1))
    believed, _ = await memory.list_stories("a", known_at=_on(2026, 3, 10), as_of=_on(2026, 3, 20))
    assert [(s.story_id, s.valid_to) for s in believed] == [(promo.story_id, _on(2026, 3, 31))]
    known, _ = await memory.list_stories("a", as_of=_on(2026, 3, 20))
    assert [s.story_id for s in known] == [early.story_id]


async def test_the_graph_answers_for_the_same_moment_as_the_passages(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    ana, bob, apollo = (
        EntityIn(name="Ana", type="person"),
        EntityIn(name="Bob", type="person"),
        EntityIn(name="Apollo", type="project"),
    )
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 1, 10))
    await memory.ingest(
        "a",
        name="programs",
        text="Apollo is part of Moonshot.",
        entities=[apollo, EntityIn(name="Moonshot", type="program")],
        relations=[RelationIn(from_name="Apollo", to_name="Moonshot", type="PART_OF")],
    )
    owned = (
        await memory.remember(
            "a",
            "Ana owns Apollo.",
            valid_from=_on(2026, 1, 1),
            entities=[ana, apollo],
            relations=[RelationIn(from_name="Ana", to_name="Apollo", type="OWNS")],
        )
    ).story
    await memory.remember(
        "a", "Ana runs the Tuesday review.", valid_from=_on(2026, 1, 1), entities=[ana]
    )
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 2, 10))
    handed = (
        await memory.supersede(
            "a",
            owned.story_id,
            "Bob owns Apollo.",
            valid_from=_on(2026, 2, 1),
            entities=[bob, apollo],
            relations=[RelationIn(from_name="Bob", to_name="Apollo", type="OWNS")],
        )
    ).story
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 3, 1))

    async def around(**when: Any) -> tuple[list[str], list[str], set[str]]:
        hood = await memory.neighbors("a", "Apollo", **when)
        names = {e.entity_id: e.name for e in hood.entities}
        return (
            sorted(names[r.from_id] for r in hood.relations if r.type == "OWNS"),
            sorted(names[r.to_id] for r in hood.relations if r.type == "PART_OF"),
            {h.text for h in hood.passages if h.kind == "story"},
        )

    assert await around() == (["Bob"], ["Moonshot"], {"Bob owns Apollo."})
    then = (["Ana"], ["Moonshot"], {"Ana owns Apollo.", "Ana runs the Tuesday review."})
    assert await around(as_of=_on(2026, 1, 20)) == then
    assert await around(known_at=_on(2026, 2, 5)) == then
    owners, _, _ = await around(history=True)
    assert owners == ["Ana", "Bob"]
    walked = await memory.search("a", "Apollo", pillars={"graph"}, kinds=("story",))
    assert [h.id for h in walked.hits] == [handed.story_id]


async def test_an_open_window_reads_stories_not_superseded_whether_or_not_they_began(
    storage: PostgresStorage,
) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    april = datetime(2026, 4, 1, tzinfo=UTC)
    vector = _vec("valve")

    def graph(name: str) -> GraphWrite:
        entity = Entity(corpus_id="a", entity_id=name, name=name, key=name, type="valve")
        return GraphWrite(entities=[entity], mentions=[(name, "story", name, name)])

    async def put(story_id: str, **kw: Any) -> None:
        story = _story("a", story_id, f"the valve {story_id} leaks", **kw)
        await storage.put_story("a", story, vector, "hash-8", graph=graph(story_id))

    await put("old", valid_from=jan)
    await put("cur", valid_from=jan)
    await put("ended", valid_from=jan, valid_to=feb)
    await put("future", valid_from=april)
    await put("retracted", valid_from=april)
    await storage.end_story("a", "retracted", valid_to=april, closed_at=NOW)
    rep = _story("a", "rep", "the valve rep leaks", valid_from=feb).model_copy(
        update={"supersedes": ("old",)}
    )
    await storage.supersede_story("a", ["old"], rep, vector, "hash-8", graph=graph("rep"))
    names = ["old", "cur", "ended", "future", "retracted", "rep"]

    async def seen(window: Window) -> list[set[str]]:
        text = await _text(
            storage,
            "a",
            "valve",
            kinds=("story",),
            top_k=10,
            filters=None,
            window=window,
            subject=None,
        )
        vec = await storage.search_vector(
            "a", vector, kinds=("story",), top_k=10, filters=None, window=window, subject=None
        )
        refs = await storage.mentions(
            "a", names, kinds=("story",), filters=None, window=window, subject=None
        )
        return [set(ids(text)), set(ids(vec)), {i for _, (_, i) in refs}]

    for found in await seen(Window(as_of=NOW, open=True)):
        assert found == {"cur", "future", "rep"}
    for found in await seen(Window(as_of=NOW)):
        assert found == {"cur", "rep"}


async def test_a_write_finds_the_story_it_may_repeat_with_every_pillar_clean(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    first = await memory.remember("a", "Ana prefers the Tuesday slot.", subject="user:ana")
    written = await memory.remember("a", "Ana prefers the Thursday slot.", subject="user:ana")
    assert first.story.story_id in [h.id for h in written.related.hits]
    assert written.related.pillars
    assert all(report.error is None for report in written.related.pillars.values())


async def _race(*writers: Awaitable[Any]) -> list[Any]:
    return await asyncio.wait_for(asyncio.gather(*writers, return_exceptions=True), timeout=30)


async def test_stress_one_open_story_per_text_with_and_without_ends(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=_Yielding())
    end = datetime.now(UTC) + timedelta(days=30)
    for round in range(10):
        text = f"The line runs shift {round}."
        racers = await _race(
            *(memory.remember("a", text, valid_to=end if i % 2 else None) for i in range(12))
        )
        assert [r for r in racers if isinstance(r, BaseException)] == []
        assert len({r.story.story_id for r in racers}) == 1
        stories, _ = await memory.list_stories("a", history=True, limit=500)
        assert len([s for s in stories if s.text == text]) == 1


async def test_stress_a_remember_racing_a_supersede_of_its_text(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=_Yielding())
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    for round in range(10):
        text = f"Bay {round} is free."
        story = (await memory.remember("a", text, valid_from=jan)).story
        racers = await _race(
            memory.supersede("a", story.story_id, f"Bay {round} is taken."),
            memory.remember("a", text),
            memory.remember("a", text),
        )
        assert [r for r in racers if isinstance(r, BaseException)] == []
        stories, _ = await memory.list_stories("a", history=True, limit=500)
        same = [s for s in stories if s.text == text]
        assert len([s for s in same if s.superseded_by is None]) <= 1
        assert [s.superseded_by is not None for s in same].count(True) == 1


async def test_stress_a_supersede_racing_remembers_of_its_new_text(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=_Yielding())
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    for round in range(10):
        old = (await memory.remember("a", f"Gate {round} is shut.", valid_from=jan)).story
        text = f"Gate {round} is open."
        racers = await _race(
            memory.supersede("a", old.story_id, text),
            memory.remember("a", text),
            memory.remember("a", text),
        )
        errors = [r for r in racers if isinstance(r, BaseException)]
        assert all(isinstance(e, InputError) and "already says that" in str(e) for e in errors)
        assert len(errors) <= 1
        now = datetime.now(UTC)
        stories, _ = await memory.list_stories("a", history=True, limit=500)
        says = [s for s in stories if s.text == text]
        open_ones = [
            s for s in says if s.superseded_by is None and (s.valid_to is None or s.valid_to > now)
        ]
        assert len(open_ones) == 1
        closed = (await memory.get_stories("a", [old.story_id]))[0]
        if errors:
            assert closed.superseded_by is None
        else:
            assert closed.superseded_by == open_ones[0].story_id


async def test_stress_two_supersedes_closing_overlapping_sets(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=_Yielding())
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    for round in range(10):
        a, b, c = [
            (await memory.remember("a", f"Fact {round}{x}.", valid_from=jan)).story.story_id
            for x in "abc"
        ]
        racers = await _race(
            memory.supersede("a", [a, b], f"Facts {round} a and b."),
            memory.supersede("a", [b, c], f"Facts {round} b and c."),
        )
        refused = [r for r in racers if isinstance(r, BaseException)]
        assert len(refused) == 1 and isinstance(refused[0], InputError)
        assert "is already superseded" in str(refused[0])
        winner = next(r for r in racers if not isinstance(r, BaseException)).story
        closed = await memory.get_stories("a", [a, b, c])
        assert [s.superseded_by for s in closed].count(winner.story_id) == 2


async def test_stress_syncs_under_history_leave_one_current_version(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=_Yielding())
    await memory.set_history("a", True)
    returns = {"metadata.doc": "returns"}
    for round in range(5):
        racers = await _race(
            *(
                memory.sync(
                    "a",
                    filters=returns,
                    sources=[
                        SourceIn(
                            name="Returns policy",
                            text=f"Round {round} writer {w}.",
                            metadata={"doc": "returns"},
                        )
                    ],
                )
                for w in range(4)
            )
        )
        assert [r for r in racers if isinstance(r, BaseException)] == []
        current, _ = await memory.list_sources("a", filters=returns)
        versions, _ = await memory.list_sources("a", filters=returns, history=True, limit=500)
        assert len(current) == 1 and len(versions) == 4 * (round + 1)
        ids = {s.source_id for s in versions}
        [now] = current
        live = await storage.source_chunks("a", now.source_id)
        assert live and all(c.retired_at is None for c in live)
        for s in versions:
            if s.retired_at is None:
                continue
            assert s.replaced_by is None or s.replaced_by in ids
            chunks = await storage.source_chunks("a", s.source_id)
            assert chunks and len(chunks) == s.chunks
            assert all(c.retired_at == s.retired_at for c in chunks)


async def test_stress_switching_history_while_syncs_run(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=_Yielding())
    returns = {"metadata.doc": "returns"}
    for round in range(5):
        racers = await _race(
            memory.set_history("a", True),
            memory.sync(
                "a",
                filters=returns,
                sources=[
                    SourceIn(
                        name="Returns policy", text=f"Round {round} A.", metadata={"doc": "returns"}
                    )
                ],
            ),
            memory.set_history("a", False),
            memory.sync(
                "a",
                filters=returns,
                sources=[
                    SourceIn(
                        name="Returns policy", text=f"Round {round} B.", metadata={"doc": "returns"}
                    )
                ],
            ),
        )
        assert [r for r in racers if isinstance(r, BaseException)] == []
        versions, _ = await memory.list_sources("a", history=True, limit=500)
        current = [s for s in versions if s.retired_at is None]
        assert len(current) == 1
        assert (await storage.count_corpus("a")).chunks == sum(s.chunks for s in versions)


async def test_a_write_at_the_planned_end_sees_the_story_ended(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    end = datetime(2026, 3, 31, tzinfo=UTC)
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 3, 1, tzinfo=UTC))
    promo = (await memory.remember("a", "The promotion runs.", valid_to=end)).story
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: end - timedelta(microseconds=1))
    assert (await memory.remember("a", "The promotion runs.")).story == promo
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: end)
    renewed = (await memory.remember("a", "The promotion runs.")).story
    assert renewed.story_id != promo.story_id and renewed.valid_from == end


def _version(source_id: str, name: str, text: str, at: datetime) -> SourceWrite:
    source, chunks, vectors = _source("a", source_id, [text], metadata={"doc": "returns"})
    return SourceWrite(
        source.model_copy(update={"name": name, "created_at": at, "digest": f"d-{source_id}"}),
        chunks,
        vectors,
    )


def _annex(source_id: str, at: datetime) -> SourceWrite:
    source, chunks, vectors = _source("a", source_id, ["Annex A."], metadata={"doc": "annex"})
    return SourceWrite(source.model_copy(update={"created_at": at}), chunks, vectors)


async def test_a_corpus_that_keeps_history_retires_what_sync_replaces(
    storage: PostgresStorage,
) -> None:
    returns = {"metadata.doc": "returns"}
    jan, feb, mar, apr = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3, 4))
    assert await storage.get_history("a") is False
    first = [
        _version("v1", "Returns policy", "30 days", jan),
        _version("m1", "Manual", "Ground first.", jan),
    ]
    assert _pair(await storage.sync_sources("a", returns, first, "m", at=jan)) == (Counts(), [])
    await storage.set_history("a", True)
    assert await storage.get_history("a") is True and await storage.get_history("b") is False
    second = [
        _version("v2", "Returns policy", "60 days", feb),
        _version("x1", "Annex", "Annex A.", feb),
    ]
    assert _pair(await storage.sync_sources("a", returns, second, "m", at=feb)) == (
        Counts(),
        ["m1", "v1"],
    )
    older = _version("v0", "Returns policy", "Older.", jan)
    assert await storage.sync_sources("a", returns, [older], "m", at=jan) is None
    assert await storage.get_source("a", "v0") is None
    v1, m1 = await storage.get_source("a", "v1"), await storage.get_source("a", "m1")
    assert v1 is not None and (v1.retired_at, v1.replaced_by) == (feb, "v2")
    assert m1 is not None and (m1.retired_at, m1.replaced_by) == (feb, None)
    assert [c.retired_at for c in await storage.source_chunks("a", "v1")] == [feb]

    async def listed(**kw: Any) -> list[str]:
        rows, _ = await storage.list_sources("a", filters=returns, limit=10, cursor=None, **kw)
        return sorted(s.source_id for s in rows)

    assert await listed() == ["v2", "x1"]
    assert await listed(history=True) == ["m1", "v1", "v2", "x1"]
    assert await listed(known_at=jan) == ["m1", "v1"]
    assert await listed(known_at=feb) == ["v2", "x1"]
    then, _ = await storage.list_sources("a", filters=returns, limit=10, cursor=None, known_at=jan)
    assert all((s.retired_at, s.replaced_by) == (None, None) for s in then)
    twins = [
        _version("v3", "Returns policy", "90 days", mar),
        _version("v4", "Returns policy", "Lifetime.", mar),
    ]
    assert _pair(await storage.sync_sources("a", returns, twins, "m", at=mar)) == (
        Counts(),
        ["v2", "x1"],
    )
    v2 = await storage.get_source("a", "v2")
    assert v2 is not None and v2.replaced_by is None
    same = _version("v5", "Returns policy", "30 days", mar)
    assert (
        await storage.sync_sources("a", returns, [same], "m", at=mar, keep={"v1": "d-v1"}) is None
    )
    assert _pair(await storage.sync_sources("a", returns, [], "m", at=apr)) == (
        Counts(),
        ["v3", "v4"],
    )
    for source_id in ("v3", "v4"):
        gone = await storage.get_source("a", source_id)
        assert gone is not None and (gone.retired_at, gone.replaced_by) == (apr, None)
    assert await listed() == []
    assert await listed(known_at=mar) == ["v3", "v4"]
    await storage.set_history("a", False)
    await storage.sync_sources("a", returns, [same], "m", at=apr)
    done = await storage.sync_sources("a", returns, [], "m", at=apr)
    assert done is not None and done[0].sources == 1 and done[1] == []
    assert await listed(history=True) == ["m1", "v1", "v2", "v3", "v4", "x1"]


async def test_a_corpus_without_history_syncs_at_an_earlier_moment_than_its_sources(
    storage: PostgresStorage,
) -> None:
    returns = {"metadata.doc": "returns"}
    jan, feb = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2))
    await storage.sync_sources(
        "a", returns, [_version("v1", "Returns policy", "30 days", feb)], "m", at=feb
    )
    done = await storage.sync_sources(
        "a", returns, [_version("v2", "Returns policy", "60 days", jan)], "m", at=jan
    )
    assert done is not None and done[0].sources == 1 and done[1] == []
    assert await storage.get_source("a", "v1") is None
    assert await storage.get_source("a", "v2") is not None


async def test_reads_follow_the_documents_memory_held(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    returns = {"metadata.doc": "returns"}
    policy = EntityIn(name="Returns", type="policy")

    def version(store: str) -> SourceIn:
        return SourceIn(
            name="Returns policy",
            text=f"The Returns policy covers the {store} store.",
            metadata={"doc": "returns"},
            entities=[policy, EntityIn(name=store, type="store")],
            relations=[RelationIn(from_name="Returns", to_name=store, type="COVERS")],
        )

    await memory.set_history("a", True)
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 1, 10))
    first = await memory.sync("a", filters=returns, sources=[version("Outlet")])
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 2, 10))
    second = await memory.sync("a", filters=returns, sources=[version("Main")])
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(2026, 3, 1))
    old, new = first.sources[0].source_id, second.sources[0].source_id

    async def found(**when: Any) -> dict[str, Any]:
        hits = await memory.search("a", "Returns policy store", kinds=("chunk",), **when)
        return {h.chunk.source_id: h.chunk.retired_at for h in hits.hits if h.chunk is not None}

    async def covered(**when: Any) -> list[str]:
        hood = await memory.neighbors("a", "Returns", **when)
        names = {e.entity_id: e.name for e in hood.entities}
        return sorted(names[r.to_id] for r in hood.relations if r.type == "COVERS")

    assert await found() == {new: None}
    assert await found(known_at=_on(2026, 1, 20)) == {old: None}
    assert await found(known_at=_on(2026, 1, 5)) == {}
    assert await found(known_at=_on(2026, 2, 20), history=True) == {
        old: _on(2026, 2, 10),
        new: None,
    }
    assert await found(history=True) == {old: _on(2026, 2, 10), new: None}
    assert await found(history=True, filters=returns) == {old: _on(2026, 2, 10), new: None}
    assert await covered() == ["Main"]
    assert await covered(known_at=_on(2026, 1, 20)) == ["Outlet"]
    assert await covered(history=True) == ["Main", "Outlet"]
    assert await memory.read_source("a", old) == "The Returns policy covers the Outlet store."
    fingerprint = await memory.fingerprint("a")
    assert (fingerprint.sources, fingerprint.chunks, fingerprint.retired, fingerprint.history) == (
        1,
        1,
        1,
        True,
    )


async def test_erasing_takes_a_version_with_the_ones_before_it_and_forgets_old_retirements(
    storage: PostgresStorage,
) -> None:
    returns = {"metadata.doc": "returns"}
    jan, feb, mar, apr = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3, 4))
    await storage.set_history("a", True)
    for source_id, at in (("v1", jan), ("v2", feb), ("v3", mar)):
        await storage.sync_sources(
            "a", returns, [_version(source_id, "Returns policy", source_id, at)], "m", at=at
        )
    await storage.sync_sources("a", {"metadata.doc": "annex"}, [_annex("a1", jan)], "m", at=jan)
    erased = await storage.delete_source("a", "v2")
    assert (erased.sources, erased.chunks) == (2, 2)
    assert (
        await storage.get_source("a", "v1") is None
        and await storage.get_source("a", "v3") is not None
    )
    await storage.sync_sources(
        "a", returns, [_version("v4", "Returns policy", "v4", apr)], "m", at=apr
    )
    assert await storage.forget_retired("a", before=apr) == Counts()
    forgot = await storage.forget_retired("a", before=apr + timedelta(days=1))
    assert forgot.sources == 1 and await storage.get_source("a", "v3") is None
    assert await storage.forget_retired("a", before=apr + timedelta(days=1)) == Counts()
    erased = await storage.delete_source("a", "v4")
    assert erased.sources == 1 and await storage.get_source("a", "a1") is not None
    assert await storage.delete_source("a", "ghost") == Counts()


async def test_keyword_search_matches_a_chunks_source_name_and_context(
    storage: PostgresStorage,
) -> None:
    source, chunks, vectors = _source(
        "a", "s1", ["Part 4471 ships in boxes of ten.", "Payment within thirty days."]
    )
    named = source.model_copy(update={"name": "Gasket catalogue"})
    framed = [chunks[0].model_copy(update={"context": "ACME invoice"}), chunks[1]]
    await storage.put_source("a", named, framed, vectors, "m")

    async def found(query: str) -> list[str]:
        legs = await _text(
            storage,
            "a",
            query,
            kinds=("chunk",),
            top_k=10,
            filters=None,
            window=Window(as_of=NOW),
            subject=None,
        )
        return sorted(ids(legs))

    assert await found("invoice") == ["s1-0"]
    assert await found("catalogue") == ["s1-0", "s1-1"]
    assert [c.context for c in await storage.source_chunks("a", "s1")] == ["ACME invoice", None]


async def _texts(
    storage: PostgresStorage, corpus: str, source_id: str, texts: list[str], **metadata: object
) -> None:
    source, chunks, vectors = _source(corpus, source_id, texts, metadata=metadata or None)
    await storage.put_source(corpus, source, chunks, vectors, "m")


async def _ranked(
    storage: PostgresStorage,
    corpus: str,
    query: str,
    *,
    kinds: tuple[Kind, ...] = ("chunk",),
    filters: dict[str, Any] | None = None,
    window: Window | None = None,
    scope: Scope | None = None,
) -> list[tuple[str, float]]:
    legs = await _text(
        storage,
        corpus,
        query,
        kinds=kinds,
        top_k=10,
        filters=filters,
        window=window or Window(as_of=NOW),
        subject=None,
        scope=scope,
    )
    return [(i, score) for _, i, score in legs]


async def test_bm25_ranks_a_rare_term_above_common_ones(storage: PostgresStorage) -> None:
    await _texts(
        storage,
        "a",
        "doc",
        ["pump pump pump pump motor", "gasket for the pump", *["pump motor"] * 4],
    )
    ranked = await _ranked(storage, "a", "pump motor gasket")
    assert ranked[0][0] == "doc-1" and len(ranked) == 6
    rarity = math.log(1 + 5.5 / 1.5) + math.log(1 + 0.5 / 6.5)
    length = 3 / 3.5
    assert ranked[0][1] == pytest.approx(rarity * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * length)))


async def test_bm25_takes_its_statistics_from_the_filtered_scope(storage: PostgresStorage) -> None:
    await _texts(storage, "a", "keep", ["gasket pump", "pump motor"], line="A")
    await _texts(storage, "a", "noise", ["gasket gasket", "gasket seal", "gasket ring"], line="B")
    await _texts(storage, "b", "keep", ["gasket pump", "pump motor"], line="A")
    filtered = await _ranked(storage, "a", "gasket pump", filters={"metadata.line": "A"})
    alone = await _ranked(storage, "b", "gasket pump")
    assert [i for i, _ in filtered] == [i for i, _ in alone]
    assert [s for _, s in filtered] == pytest.approx([s for _, s in alone])


async def test_bm25_leaves_retired_versions_out_of_a_current_read(storage: PostgresStorage) -> None:
    returns = {"metadata.doc": "returns"}
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)

    def version(corpus: str, source_id: str, text: str, at: datetime) -> SourceWrite:
        source, chunks, vectors = _source(corpus, source_id, [text], metadata={"doc": "returns"})
        named = source.model_copy(update={"name": "Returns policy", "created_at": at})
        return SourceWrite(named, chunks, vectors)

    await storage.set_history("a", True)
    await storage.sync_sources(
        "a", returns, [version("a", "v1", "gasket gasket gasket", jan)], "m", at=jan
    )
    await storage.sync_sources("a", returns, [version("a", "v2", "gasket pump", feb)], "m", at=feb)
    await storage.sync_sources("b", returns, [version("b", "v2", "gasket pump", feb)], "m", at=feb)
    current = await _ranked(storage, "a", "gasket pump", window=Window(as_of=feb))
    alone = await _ranked(storage, "b", "gasket pump", window=Window(as_of=feb))
    assert [i for i, _ in current] == [i for i, _ in alone] == ["v2-0"]
    assert [s for _, s in current] == pytest.approx([s for _, s in alone])
    every = await _ranked(storage, "a", "gasket pump", window=Window())
    assert sorted(i for i, _ in every) == ["v1-0", "v2-0"]


async def test_bm25_prefers_the_shorter_passage_for_the_same_term_count(
    storage: PostgresStorage,
) -> None:
    await _texts(storage, "a", "doc", ["gasket seal ring flange bolt washer", "gasket seal"])
    assert [i for i, _ in await _ranked(storage, "a", "gasket")] == ["doc-1", "doc-0"]


async def test_bm25_counts_every_occurrence_in_a_passage_length(storage: PostgresStorage) -> None:
    await _texts(storage, "a", "doc", ["gasket seal seal seal seal", "gasket ring flange"])
    assert [i for i, _ in await _ranked(storage, "a", "gasket")] == ["doc-1", "doc-0"]


async def test_bm25_lists_a_repeated_kind_once(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.remember("a", "The gasket is backordered.")
    await memory.remember("a", "The gasket is running.")
    window = Window(as_of=datetime.now(UTC) + timedelta(seconds=1))
    once = await _ranked(storage, "a", "gasket", kinds=("story",), window=window)
    twice = await _ranked(storage, "a", "gasket", kinds=("story", "story"), window=window)
    assert len(once) == 2 and twice == once


async def test_bm25_counts_a_repeated_query_word_once(storage: PostgresStorage) -> None:
    await _texts(storage, "a", "doc", ["gasket seal", "seal ring"])
    once = await _ranked(storage, "a", "gasket seal")
    twice = await _ranked(storage, "a", "gasket gasket seal")
    assert [i for i, _ in twice] == [i for i, _ in once]
    assert [s for _, s in twice] == pytest.approx([s for _, s in once])


async def test_bm25_ranks_stories_on_their_own_statistics(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.remember("a", "The gasket is backordered.")
    await memory.remember("a", "The pump is running.")
    await _texts(storage, "a", "doc", ["gasket gasket gasket", "gasket seal"])
    window = Window(as_of=datetime.now(UTC) + timedelta(seconds=1))
    both = dict(await _ranked(storage, "a", "gasket pump", kinds=("chunk", "story"), window=window))
    stories = await _ranked(storage, "a", "gasket pump", kinds=("story",), window=window)
    assert len(stories) == 2
    assert [both[i] for i, _ in stories] == pytest.approx([s for _, s in stories])


async def test_bm25_answers_a_query_of_stop_words_with_nothing(storage: PostgresStorage) -> None:
    await _texts(storage, "a", "doc", ["the gasket is in the box"])
    assert await _ranked(storage, "a", "the of and") == []


async def _pooled(
    storage: PostgresStorage,
    corpora: list[str],
    query: str,
    *,
    kinds: tuple[Kind, ...] = ("chunk",),
    filters: dict[str, Any] | None = None,
    window: Window | None = None,
    subject: str | None = None,
) -> list[tuple[str, str, float]]:
    found = await storage.search_text(
        corpora,
        query,
        kinds=kinds,
        top_k=50,
        filters=filters,
        window=window or Window(as_of=NOW),
        subject=subject,
        ranking=BM25(),
        match="any",
    )
    return [(corpus, i, score) for corpus, (_, i, score) in found]


MANUAL = [
    "gasket and pump assembly",
    "gasket material",
    "gasket storage",
    "gasket supplier",
    "gasket inspection",
    "gasket sizes",
    "pump curve",
    "pump wiring",
]
NOTES = ["pump noise", "pump restart", "pump alarm", "pump service", "gasket ordered"]


async def test_bm25_counts_the_corpora_it_is_given_as_one(storage: PostgresStorage) -> None:
    await _texts(storage, "whole", "manual", MANUAL)
    await _texts(storage, "whole", "notes", NOTES)
    await _texts(storage, "big", "manual", MANUAL)
    await _texts(storage, "small", "notes", NOTES)
    whole = await _pooled(storage, ["whole"], "gasket pump")
    split = await _pooled(storage, ["big", "small"], "gasket pump")
    assert len(split) == 13
    assert [i for _, i, _ in split] == [i for _, i, _ in whole]
    assert [s for _, _, s in split] == pytest.approx([s for _, _, s in whole])
    assert all(c == ("big" if i.startswith("manual") else "small") for c, i, _ in split)


async def test_bm25_counts_the_stories_of_the_corpora_it_is_given_as_one(
    storage: PostgresStorage,
) -> None:
    texts = {"st0": "the pump seal leaks", "st1": "the pump runs", "st2": "seal kit ordered"}
    for story_id, text in texts.items():
        await storage.put_story("whole", _story("whole", story_id, text), _vec(text), "m")
        corpus = "small" if story_id == "st2" else "big"
        await storage.put_story(corpus, _story(corpus, story_id, text), _vec(text), "m")
    whole = await _pooled(storage, ["whole"], "pump seal", kinds=("story",))
    split = await _pooled(storage, ["big", "small"], "pump seal", kinds=("story",))
    assert [i for _, i, _ in split] == [i for _, i, _ in whole] == ["st0", "st1", "st2"]
    assert [s for _, _, s in split] == pytest.approx([s for _, _, s in whole])


async def test_bm25_ranks_a_big_corpus_answer_above_a_small_corpus_mention(
    storage: PostgresStorage,
) -> None:
    await _texts(storage, "manual", "manual", MANUAL)
    await _texts(storage, "notes", "notes", NOTES)
    manual = await _pooled(storage, ["manual"], "gasket pump")
    notes = await _pooled(storage, ["notes"], "gasket pump")
    assert notes[0][1] == "notes-4" and notes[0][2] > manual[0][2]
    pooled = await _pooled(storage, ["notes", "manual"], "gasket pump")
    assert pooled[0][:2] == ("manual", "manual-0")


async def test_bm25_reads_no_corpus_it_is_not_given(storage: PostgresStorage) -> None:
    await _texts(storage, "a", "s", ["gasket pump", "pump motor"])
    await _texts(storage, "b", "t", ["gasket seal"])
    before = await _pooled(storage, ["a", "b"], "gasket pump")
    await _texts(storage, "c", "u", ["gasket gasket", "gasket pump", "pump"])
    after = await _pooled(storage, ["a", "b"], "gasket pump")
    assert after == before and {c for c, _, _ in after} == {"a", "b"}


async def test_bm25_holds_filters_in_every_corpus_it_is_given(storage: PostgresStorage) -> None:
    await _texts(storage, "a", "ka", ["gasket pump"], line="A")
    await _texts(storage, "a", "na", ["gasket gasket", "gasket seal"], line="B")
    await _texts(storage, "b", "kb", ["pump motor"], line="A")
    await _texts(storage, "b", "nb", ["pump pump"], line="B")
    await _texts(storage, "c", "ka", ["gasket pump"])
    await _texts(storage, "d", "kb", ["pump motor"])
    filtered = await _pooled(storage, ["a", "b"], "gasket pump", filters={"metadata.line": "A"})
    alone = await _pooled(storage, ["c", "d"], "gasket pump")
    assert [i for _, i, _ in filtered] == [i for _, i, _ in alone] == ["ka-0", "kb-0"]
    assert [s for _, _, s in filtered] == pytest.approx([s for _, _, s in alone])


async def test_bm25_holds_subject_and_window_in_every_corpus_it_is_given(
    storage: PostgresStorage,
) -> None:
    began, ended = NOW - timedelta(days=2), NOW - timedelta(days=1)
    rows = (
        ("a", "st0", "pump seal leaks", "pump", None),
        ("a", "st1", "pump seal pump seal", "valve", None),
        ("b", "st2", "seal kit for the pump", "pump", None),
        ("b", "st3", "pump seal replaced", "pump", ended),
        ("c", "st0", "pump seal leaks", "pump", None),
        ("d", "st2", "seal kit for the pump", "pump", None),
    )
    for corpus, story_id, text, subject, valid_to in rows:
        story = _story(corpus, story_id, text, subject=subject, valid_from=began, valid_to=valid_to)
        await storage.put_story(corpus, story, _vec(text), "m")
    scoped = await _pooled(storage, ["a", "b"], "pump seal", kinds=("story",), subject="pump")
    alone = await _pooled(storage, ["c", "d"], "pump seal", kinds=("story",), subject="pump")
    assert [i for _, i, _ in scoped] == [i for _, i, _ in alone] == ["st0", "st2"]
    assert [s for _, _, s in scoped] == pytest.approx([s for _, _, s in alone])


async def test_bm25_leaves_retired_versions_out_of_every_corpus_it_is_given(
    storage: PostgresStorage,
) -> None:
    returns = {"metadata.doc": "returns"}
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)

    def version(corpus: str, source_id: str, text: str, at: datetime) -> SourceWrite:
        source, chunks, vectors = _source(corpus, source_id, [text], metadata={"doc": "returns"})
        named = source.model_copy(update={"name": "Returns policy", "created_at": at})
        return SourceWrite(named, chunks, vectors)

    await storage.set_history("a", True)
    await storage.sync_sources(
        "a", returns, [version("a", "v1", "gasket gasket gasket", jan)], "m", at=jan
    )
    await storage.sync_sources("a", returns, [version("a", "v2", "gasket pump", feb)], "m", at=feb)
    await storage.sync_sources("b", returns, [version("b", "w1", "pump seal", feb)], "m", at=feb)
    await storage.sync_sources("c", returns, [version("c", "v2", "gasket pump", feb)], "m", at=feb)
    await storage.sync_sources("d", returns, [version("d", "w1", "pump seal", feb)], "m", at=feb)
    current = await _pooled(storage, ["a", "b"], "gasket pump", window=Window(as_of=feb))
    alone = await _pooled(storage, ["c", "d"], "gasket pump", window=Window(as_of=feb))
    assert [i for _, i, _ in current] == [i for _, i, _ in alone] == ["v2-0", "w1-0"]
    assert [s for _, _, s in current] == pytest.approx([s for _, _, s in alone])
    then = await _pooled(storage, ["a", "b"], "gasket pump", window=Window(known_at=jan))
    assert [(c, i) for c, i, _ in then] == [("a", "v1-0")]


async def test_bm25_orders_ties_by_passage_id_then_corpus(storage: PostgresStorage) -> None:
    for corpus, source_id in (("B", "s"), ("a", "t"), ("a", "s")):
        source, chunks, vectors = _source(corpus, source_id, ["gasket"])
        named = source.model_copy(update={"name": "manual"})
        renamed = [c.model_copy(update={"source_name": "manual"}) for c in chunks]
        await storage.put_source(corpus, named, renamed, vectors, "m")
    found = await _pooled(storage, ["B", "a"], "gasket")
    assert [(c, i) for c, i, _ in found] == [("B", "s-0"), ("a", "s-0"), ("a", "t-0")]
    assert len({s for _, _, s in found}) == 1


async def test_bm25_answers_stop_words_across_corpora_with_nothing(
    storage: PostgresStorage,
) -> None:
    await _texts(storage, "a", "doc", ["the gasket is in the box"])
    await _texts(storage, "b", "doc", ["the pump is on the shelf"])
    assert await _pooled(storage, ["a", "b"], "the of and") == []


async def test_bm25_reads_the_query_in_the_language_of_the_corpora_it_is_given(
    storage: PostgresStorage,
) -> None:
    for corpus in ("es1", "es2"):
        await storage.set_language(corpus, "spanish")
    await _texts(storage, "es1", "uno", ["las bombas funcionan"])
    await _texts(storage, "es2", "dos", ["la bomba funcionaba ayer"])
    await _texts(storage, "en", "tres", ["las bombas funcionan"])
    found = await _pooled(storage, ["es1", "es2"], "funcionando")
    assert sorted(c for c, _, _ in found) == ["es1", "es2"]
    assert await _pooled(storage, ["en"], "funcionando") == []


async def _scoped_corpus(storage: PostgresStorage) -> None:
    gasket = Entity(corpus_id="a", entity_id="g", name="gasket", key="gasket", type="part")
    for source_id, texts in (
        ("s1", ["gasket pump one", "gasket pump two"]),
        ("s2", ["gasket pump three"]),
    ):
        source, chunks, vectors = _source("a", source_id, texts)
        await storage.put_source(
            "a",
            source,
            chunks,
            vectors,
            "m",
            graph=GraphWrite(
                entities=[gasket], mentions=[("g", "chunk", c.chunk_id, "gasket") for c in chunks]
            ),
        )
    for story_id, cited in (("cites", ("s1",)), ("both", ("s2", "s9")), ("none", ())):
        story = _story("a", story_id, f"gasket pump {story_id}").model_copy(
            update={"source_ids": cited}
        )
        await storage.put_story(
            "a",
            story,
            _vec(story.text),
            "m",
            graph=GraphWrite(entities=[gasket], mentions=[("g", "story", story_id, "gasket")]),
        )


async def _inside(storage: PostgresStorage, scope: Scope | None) -> list[set[str]]:
    both: tuple[Kind, ...] = ("chunk", "story")
    window = Window(as_of=NOW)
    text = await _text(
        storage,
        "a",
        "gasket pump",
        kinds=both,
        top_k=20,
        filters=None,
        window=window,
        subject=None,
        scope=scope,
    )
    vector = await storage.search_vector(
        "a",
        _vec("gasket"),
        kinds=both,
        top_k=20,
        filters=None,
        window=window,
        subject=None,
        scope=scope,
    )
    found = await storage.mentions(
        "a", ["g"], kinds=both, filters=None, window=window, subject=None, scope=scope
    )
    return [set(ids(text)), set(ids(vector)), {i for _, (_, i) in found}]


async def test_searches_and_mentions_stay_inside_a_scope(storage: PostgresStorage) -> None:
    await _scoped_corpus(storage)
    every = {"s1-0", "s1-1", "s2-0", "cites", "both", "none"}
    for scope, expected in (
        (None, every),
        (Scope(), every),
        (Scope(sources=("s1",)), {"s1-0", "s1-1", "cites"}),
        (Scope(sources=("s2",)), {"s2-0", "both"}),
        (Scope(passages=(("chunk", "s1-1"), ("story", "none"))), {"s1-1", "none"}),
        (
            Scope(sources=("s1",), passages=(("chunk", "s1-1"), ("chunk", "s2-0"))),
            {"s1-1"},
        ),
    ):
        assert await _inside(storage, scope) == [expected] * 3, scope


async def test_an_empty_scope_reads_nothing(storage: PostgresStorage) -> None:
    await _scoped_corpus(storage)
    for scope in (Scope(sources=()), Scope(passages=()), Scope(sources=("ghost",))):
        assert await _inside(storage, scope) == [set(), set(), set()], scope


async def test_a_sources_scope_holds_the_stories_that_cite_one_of_them(
    storage: PostgresStorage,
) -> None:
    for story_id, cited in (
        ("one", ("s1",)),
        ("two", ("s2", "s9")),
        ("three", ("s9",)),
        ("bare", ()),
        ("other", ("s3",)),
    ):
        story = _story("a", story_id, f"the seal {story_id}").model_copy(
            update={"source_ids": cited}
        )
        await storage.put_story("a", story, _vec(story.text), "m")
    stories: tuple[Kind, ...] = ("story",)
    scope = Scope(sources=("s1", "s9"))
    text = await _text(
        storage,
        "a",
        "seal",
        kinds=stories,
        top_k=10,
        filters=None,
        window=Window(as_of=NOW),
        subject=None,
        scope=scope,
    )
    vector = await storage.search_vector(
        "a",
        _vec("seal"),
        kinds=stories,
        top_k=10,
        filters=None,
        window=Window(as_of=NOW),
        subject=None,
        scope=scope,
    )
    assert set(ids(text)) == set(ids(vector)) == {"one", "two", "three"}


async def test_top_k_counts_inside_the_scope(storage: PostgresStorage) -> None:
    await _texts(storage, "a", "big", ["gasket gasket"] * 6)
    await _texts(storage, "a", "small", ["gasket ring flange", "gasket seal bolt", "gasket"])
    window = Window(as_of=NOW)
    scope = Scope(sources=("small",))
    text = await _text(
        storage,
        "a",
        "gasket",
        kinds=("chunk",),
        top_k=2,
        filters=None,
        window=window,
        subject=None,
        scope=scope,
    )
    vector = await storage.search_vector(
        "a",
        _vec("gasket gasket"),
        kinds=("chunk",),
        top_k=2,
        filters=None,
        window=window,
        subject=None,
        scope=scope,
    )
    for legs in (text, vector):
        assert len(legs) == 2 and all(i.startswith("small-") for i in ids(legs))


async def test_bm25_inside_a_scope_equals_bm25_over_the_scope_alone(
    storage: PostgresStorage,
) -> None:
    keep = ["gasket pump", "pump motor", "gasket seal ring"]
    await _texts(storage, "a", "keep", keep)
    await _texts(storage, "a", "noise", ["gasket gasket", "gasket seal", "pump pump pump"])
    await _texts(storage, "b", "keep", keep)
    alone = await _ranked(storage, "b", "gasket pump")
    held = tuple(("chunk", f"keep-{i}") for i in range(len(keep)))
    for scope in (Scope(sources=("keep",)), Scope(passages=held)):
        scoped = await _ranked(storage, "a", "gasket pump", scope=scope)
        assert [i for i, _ in scoped] == [i for i, _ in alone]
        assert [s for _, s in scoped] == pytest.approx([s for _, s in alone])
    unscoped = dict(await _ranked(storage, "a", "gasket pump"))
    assert [unscoped[i] for i, _ in alone] != pytest.approx([s for _, s in alone])


async def test_mentions_name_the_entity_each_passage_mentions(storage: PostgresStorage) -> None:
    source, chunks, vectors = _source("a", "s1", ["valve V-1 feeds pump P-1", "pump P-1 alone"])
    valve = Entity(corpus_id="a", entity_id="v", name="V-1", key="v-1", type="valve")
    pump = Entity(corpus_id="a", entity_id="p", name="P-1", key="p-1", type="pump")
    await storage.put_source(
        "a",
        source,
        chunks,
        vectors,
        "m",
        graph=GraphWrite(
            entities=[valve, pump],
            mentions=[
                ("v", "chunk", "s1-0", "V-1"),
                ("p", "chunk", "s1-0", "P-1"),
                ("p", "chunk", "s1-0", "pump one"),
                ("p", "chunk", "s1-1", "P-1"),
            ],
        ),
    )
    await storage.put_story(
        "a",
        _story("a", "st", "P-1 hums"),
        None,
        None,
        graph=GraphWrite(mentions=[("p", "story", "st", "P-1")]),
    )
    both: tuple[Kind, ...] = ("chunk", "story")
    assert await storage.mentions(
        "a", ["v", "p"], kinds=both, filters=None, window=Window(), subject=None
    ) == [
        ("p", ("chunk", "s1-0")),
        ("v", ("chunk", "s1-0")),
        ("p", ("chunk", "s1-1")),
        ("p", ("story", "st")),
    ]
    assert await storage.mentions(
        "a", ["v"], kinds=both, filters=None, window=Window(), subject=None
    ) == [("v", ("chunk", "s1-0"))]


async def test_passage_vector_is_the_vector_a_passage_was_stored_with(
    storage: PostgresStorage,
) -> None:
    source, chunks, vectors = _source("a", "s1", ["one", "two"])
    await storage.put_source("a", source, chunks, vectors, "m")
    await storage.put_story("a", _story("a", "st", "three"), _vec("three"), "m")
    await storage.put_story("a", _story("a", "bare", "four"), None, None)
    assert await storage.passage_vector("a", ("chunk", "s1-1")) == pytest.approx(_vec("two"))
    assert await storage.passage_vector("a", ("story", "st")) == pytest.approx(_vec("three"))
    for corpus, ref in (
        ("a", ("story", "bare")),
        ("a", ("chunk", "ghost")),
        ("b", ("chunk", "s1-1")),
    ):
        assert await storage.passage_vector(corpus, ref) is None


async def test_chunks_around_reads_neighbouring_positions_of_one_source(
    storage: PostgresStorage,
) -> None:
    source, chunks, vectors = _source("a", "s1", [f"part {i}" for i in range(6)])
    await storage.put_source("a", source, chunks, vectors, "m")
    other, other_chunks, other_vectors = _source("a", "s2", ["elsewhere 0", "elsewhere 1"])
    await storage.put_source("a", other, other_chunks, other_vectors, "m")
    now = Window(as_of=NOW)
    around = await storage.chunks_around("a", "s1", 3, width=1, window=now, filters=None)
    assert [c.chunk_id for c in around] == ["s1-2", "s1-3", "s1-4"]
    assert around[0] == chunks[2]
    first = await storage.chunks_around("a", "s1", 0, width=2, window=now, filters=None)
    assert [c.position for c in first] == [0, 1, 2]
    whole = await storage.chunks_around("a", "s1", 5, width=9, window=now, filters=None)
    assert [c.position for c in whole] == [0, 1, 2, 3, 4, 5]
    assert await storage.chunks_around("b", "s1", 3, width=1, window=now, filters=None) == []


async def test_chunks_around_reads_only_what_a_search_in_the_window_sees(
    storage: PostgresStorage,
) -> None:
    source, chunks, vectors = _source("a", "s1", ["one", "two", "three"], metadata={"line": "A"})
    await storage.put_source("a", source, chunks, vectors, "m")
    await storage.set_history("a", True)
    retired = NOW + timedelta(days=30)
    line = {"metadata.line": "A"}
    assert _pair(await storage.sync_sources("a", line, [], "m", at=retired)) == (Counts(), ["s1"])

    async def around(window: Window, filters: dict[str, Any] | None = None) -> list[Chunk]:
        return await storage.chunks_around("a", "s1", 1, width=1, window=window, filters=filters)

    every = ["s1-0", "s1-1", "s1-2"]
    assert await around(Window(as_of=retired + timedelta(days=1))) == []
    assert [c.chunk_id for c in await around(Window())] == every
    assert [c.retired_at for c in await around(Window())] == [retired] * 3
    between = NOW + timedelta(days=10)
    known = await around(Window(as_of=between, known_at=between))
    assert [(c.chunk_id, c.retired_at) for c in known] == [(i, None) for i in every]
    before = NOW - timedelta(days=1)
    assert await around(Window(as_of=before, known_at=before)) == []
    assert await around(Window(known_at=before)) == []
    assert await around(Window(), {"metadata.line": "B"}) == []
    assert [c.chunk_id for c in await around(Window(), line)] == every


async def test_passage_entities_name_the_entities_of_each_passage(storage: PostgresStorage) -> None:
    source, chunks, vectors = _source("a", "s1", ["V-1 feeds P-1", "P-1 alone", "nothing"])
    valve = Entity(corpus_id="a", entity_id="v", name="V-1", key="v-1", type="valve")
    pump = Entity(corpus_id="a", entity_id="p", name="P-1", key="p-1", type="pump")
    await storage.put_source(
        "a",
        source,
        chunks,
        vectors,
        "m",
        graph=GraphWrite(
            entities=[valve, pump],
            mentions=[
                ("v", "chunk", "s1-0", "V-1"),
                ("p", "chunk", "s1-0", "P-1"),
                ("p", "chunk", "s1-0", "pump one"),
                ("p", "chunk", "s1-1", "P-1"),
            ],
        ),
    )
    await storage.put_story(
        "a",
        _story("a", "st", "P-1 hums"),
        None,
        None,
        graph=GraphWrite(mentions=[("p", "story", "st", "P-1")]),
    )
    found = await storage.passage_entities(
        "a",
        [
            ("story", "st"),
            ("chunk", "s1-1"),
            ("chunk", "s1-0"),
            ("chunk", "s1-2"),
            ("chunk", "ghost"),
        ],
    )
    assert [(ref, e.entity_id) for ref, e in found] == [
        (("chunk", "s1-0"), "p"),
        (("chunk", "s1-0"), "v"),
        (("chunk", "s1-1"), "p"),
        (("story", "st"), "p"),
    ]
    assert found[1][1] == valve
    assert found[0][1] == pump.model_copy(update={"aliases": ("pump one",)})
    assert await storage.passage_entities("b", [("chunk", "s1-0")]) == []


async def test_graph_types_count_entities_and_relations_by_type(storage: PostgresStorage) -> None:
    named = (("V-1", "valve"), ("V-2", "valve"), ("P-1", "pump"))
    entities = [
        Entity(corpus_id="a", entity_id=name, name=name, key=name.lower(), type=kind)
        for name, kind in named
    ]
    relations = [
        Relation(
            corpus_id="a",
            from_id=start,
            to_id=end,
            type=kind,
            evidence_kind="chunk",
            evidence_id="s1-0",
        )
        for start, end, kind in (
            ("V-1", "P-1", "FEEDS"),
            ("V-2", "P-1", "FEEDS"),
            ("V-1", "V-2", "NEAR"),
        )
    ]
    source, chunks, vectors = _source("a", "s1", ["V-1 and V-2 feed P-1"])
    await storage.put_source(
        "a",
        source,
        chunks,
        vectors,
        "m",
        graph=GraphWrite(
            entities=entities,
            relations=relations,
            mentions=[(name, "chunk", "s1-0", name) for name, _ in named],
        ),
    )
    assert await storage.graph_types("a") == ({"pump": 1, "valve": 2}, {"FEEDS": 2, "NEAR": 1})
    assert await storage.graph_types("b") == ({}, {})


async def test_graph_search_ranks_by_distance_and_rarity(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"graph"})
    apollo, borealis, cygnus = (
        Entity(corpus_id="a", entity_id=name.lower(), name=name, key=name.lower(), type="thing")
        for name in ("Apollo", "Borealis", "Cygnus")
    )

    async def told(story_id: str, named: list[Entity], *related: tuple[Entity, Entity]) -> None:
        await storage.put_story(
            "a",
            _story("a", story_id, f"story {story_id}", valid_from=NOW - timedelta(days=1)),
            None,
            None,
            graph=GraphWrite(
                entities=[apollo, borealis, cygnus],
                relations=[
                    Relation(
                        corpus_id="a",
                        from_id=start.entity_id,
                        to_id=end.entity_id,
                        type="FEEDS",
                        evidence_kind="story",
                        evidence_id=story_id,
                    )
                    for start, end in related
                ],
                mentions=[(e.entity_id, "story", story_id, e.name) for e in named],
            ),
        )

    await told("p1", [cygnus])
    await told("p2", [borealis])
    await told("p3", [apollo])
    await told("p4", [apollo, borealis])
    await told("r1", [], (apollo, borealis), (borealis, cygnus))
    hits = await memory.search("a", "status", entities=["Apollo"], kinds=("story",), as_of=NOW)
    assert [h.id for h in hits.hits] == ["p4", "p3", "p2", "p1"]
    near, next_door, far = math.log(2), math.log(2) / 2, math.log(1 + 3.5 / 1.5) / 4

    def scored(hop: int, rarity: float) -> float:
        return 2.0**-hop * (1 + rarity / (1 + rarity)) / 2

    best = scored(0, near + next_door)
    assert [h.provenance["graph"] for h in hits.hits] == pytest.approx(
        [1.0, scored(0, near) / best, scored(1, next_door) / best, scored(2, far) / best]
    )


async def test_expand_reads_by_document_by_vector_and_by_graph(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    texts = ("V-1 feeds P-3.", "P-3 runs hot.", "Ground the panel.", "Replace the seal.")
    source = (
        await memory.ingest(
            "a",
            name="manual",
            sections=[Section(text=t) for t in texts],
            entities=[EntityIn(name="P-3", type="pump"), EntityIn(name="V-1", type="valve")],
            relations=[RelationIn(from_name="V-1", to_name="P-3", type="FEEDS")],
        )
    ).source
    chunks = [c.chunk_id for c in await storage.source_chunks("a", source.source_id)]
    beside = await memory.expand("a", ("chunk", chunks[1]), by="document", width=1)
    assert [h.text for h in beside.hits] == ["V-1 feeds P-3.", "Ground the panel."]
    near = await memory.expand("a", ("chunk", chunks[0]), by="vector")
    assert len(near.hits) == 3 and chunks[0] not in {h.id for h in near.hits}
    connected = await memory.expand("a", ("chunk", chunks[0]), by="graph")
    assert [h.text for h in connected.hits] == ["P-3 runs hot."]


async def test_hits_carry_their_entities(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.ingest(
        "a",
        name="sheet",
        sections=[Section(text="V-1 feeds P-3."), Section(text="P-3 runs hot.")],
        entities=[EntityIn(name="V-1", type="valve"), EntityIn(name="P-3", type="pump")],
        relations=[RelationIn(from_name="V-1", to_name="P-3", type="FEEDS")],
    )
    hits = await memory.search("a", "V-1 feeds P-3")
    named = {h.text: [e.name for e in h.entities] for h in hits.hits}
    assert named == {"V-1 feeds P-3.": ["P-3", "V-1"], "P-3 runs hot.": ["P-3"]}
    assert all(e.corpus_id == "a" for h in hits.hits for e in h.entities)


async def test_a_language_that_is_not_text_names_no_configuration_on_the_storage_either(
    storage: PostgresStorage,
) -> None:
    assert await storage.set_language("acme", 5) is None  # type: ignore[arg-type]
    assert await storage.get_language("acme") == await storage.get_language("never-written")


async def test_writes_naming_the_same_entities_in_opposite_orders_all_finish(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    pair = [EntityIn(name="Pump P-3", type="pump"), EntityIn(name="Tank T-1", type="tank")]

    async def write(i: int) -> None:
        await memory.ingest(
            "acme",
            name=f"log {i}",
            text=f"Pump P-3 fills Tank T-1, entry {i}.",
            entities=pair if i % 2 else pair[::-1],
        )

    await asyncio.wait_for(asyncio.gather(*(write(i) for i in range(16))), 60)
    assert len((await memory.list_sources("acme", limit=50))[0]) == 16
    assert len((await storage.entities_named("acme", ["Pump P-3"]))[name_key("Pump P-3")]) == 1


async def test_neighbors_lists_what_lies_at_one_distance_in_one_order(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    hub = EntityIn(name="H-1", type="hub")
    for name in ("Z-9", "A-1", "M-5"):
        await memory.remember(
            "acme",
            f"H-1 feeds {name}.",
            entities=[hub, EntityIn(name=name, type="pump")],
            relations=[RelationIn(from_name="H-1", to_name=name, type="FEEDS")],
            related=0,
        )
    first = await memory.neighbors("acme", "H-1")
    assert await memory.neighbors("acme", "H-1") == first
    around = [e.entity_id for e in first.entities[1:]]
    assert len(around) == 3 and around == sorted(around)
    assert [(r.from_id, r.to_id) for r in first.relations] == sorted(
        (r.from_id, r.to_id) for r in first.relations
    )


async def test_bm25_reads_the_query_in_the_language_of_the_first_corpus_it_is_given(
    storage: PostgresStorage,
) -> None:
    await storage.set_language("es1", "spanish")
    await _texts(storage, "es1", "uno", ["las bombas funcionan"])
    await _texts(storage, "en", "tres", ["the pumps are running"])
    assert [c for c, _, _ in await _pooled(storage, ["es1", "en"], "funcionando")] == ["es1"]
    assert await _pooled(storage, ["en", "es1"], "funcionando") == []


async def test_bm25_over_several_corpora_keeps_the_best_top_k_of_them_all(
    storage: PostgresStorage,
) -> None:
    await _texts(storage, "whole", "manual", MANUAL)
    await _texts(storage, "whole", "notes", NOTES)
    await _texts(storage, "big", "manual", MANUAL)
    await _texts(storage, "small", "notes", NOTES)
    whole = await _pooled(storage, ["whole"], "gasket pump")
    found = await storage.search_text(
        ["big", "small"],
        "gasket pump",
        kinds=("chunk",),
        top_k=3,
        filters=None,
        window=Window(as_of=NOW),
        subject=None,
        ranking=BM25(),
        match="any",
    )
    assert [i for _, (_, i, _) in found] == [i for _, i, _ in whole[:3]]


async def test_bm25_keeps_chunk_and_story_statistics_apart_across_corpora(
    storage: PostgresStorage,
) -> None:
    await _texts(storage, "whole", "manual", MANUAL)
    await _texts(storage, "big", "manual", MANUAL)
    texts = {"st0": "the pump seal leaks", "st1": "the gasket is worn", "st2": "seal kit ordered"}
    for story_id, text in texts.items():
        await storage.put_story("whole", _story("whole", story_id, text), _vec(text), "m")
        corpus = "small" if story_id == "st2" else "big"
        await storage.put_story(corpus, _story(corpus, story_id, text), _vec(text), "m")
    both: tuple[Kind, ...] = ("chunk", "story")
    whole = await _pooled(storage, ["whole"], "gasket pump seal", kinds=both)
    split = await _pooled(storage, ["big", "small"], "gasket pump seal", kinds=both)
    assert [i for _, i, _ in split] == [i for _, i, _ in whole]
    assert [s for _, _, s in split] == pytest.approx([s for _, _, s in whole])
    stories = await _pooled(storage, ["big", "small"], "gasket pump seal", kinds=("story",))
    assert {i: s for _, i, s in stories} == pytest.approx(
        {i: s for _, i, s in split if i.startswith("st")}
    )


async def test_passage_entities_order_a_passages_entities_by_code_point(
    storage: PostgresStorage,
) -> None:
    source, chunks, vectors = _source("a", "s1", ["a and B side by side"])
    lower = Entity(corpus_id="a", entity_id="a", name="a", key="a", type="tag")
    upper = Entity(corpus_id="a", entity_id="B", name="B", key="b", type="tag")
    await storage.put_source(
        "a",
        source,
        chunks,
        vectors,
        "m",
        graph=GraphWrite(
            entities=[lower, upper],
            mentions=[("a", "chunk", "s1-0", "a"), ("B", "chunk", "s1-0", "B")],
        ),
    )
    found = await storage.passage_entities("a", [("chunk", "s1-0")])
    assert [e.entity_id for _, e in found] == ["B", "a"]


async def test_a_scoped_keyword_search_over_several_corpora_keeps_each_corpus_to_its_passages(
    storage: PostgresStorage,
) -> None:
    for corpus, source_id in (("a", "s1"), ("b", "t1")):
        source, chunks, vectors = _source(corpus, source_id, ["gasket pump one", "gasket pump two"])
        await storage.put_source(corpus, source, chunks, vectors, "m")
    scope = Scope(passages=(("chunk", "s1-1"), ("chunk", "t1-0")))
    found = await storage.search_text(
        ["a", "b"],
        "gasket pump",
        kinds=("chunk",),
        top_k=20,
        filters=None,
        window=Window(as_of=NOW),
        subject=None,
        ranking=BM25(),
        match="any",
        scope=scope,
    )
    assert sorted((c, i) for c, (_, i, _) in found) == [("a", "s1-1"), ("b", "t1-0")]


async def test_graph_types_count_the_relations_retired_versions_still_evidence(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    await memory.set_history("a", True)
    feeds = RelationIn(from_name="V-1", to_name="P-1", type="FEEDS")
    pair = [EntityIn(name="V-1", type="valve"), EntityIn(name="P-1", type="pump")]
    doc = {"metadata.doc": "line"}
    for text in ("V-1 feeds P-1.", "V-1 still feeds P-1."):
        await memory.sync(
            "a",
            filters=doc,
            sources=[
                SourceIn(
                    name="line",
                    text=text,
                    metadata={"doc": "line"},
                    entities=pair,
                    relations=[feeds],
                )
            ],
        )
    assert await storage.graph_types("a") == ({"pump": 1, "valve": 1}, {"FEEDS": 2})
    assert (await memory.describe("a")).relation_types == {"FEEDS": 2}


async def test_passage_entities_answer_for_a_retired_chunk_while_its_hits_stay_hidden(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    await memory.set_history("a", True)
    doc = {"metadata.doc": "line"}
    pump = [EntityIn(name="P-1", type="pump")]
    first = await memory.sync(
        "a",
        filters=doc,
        sources=[SourceIn(name="line", text="P-1 runs.", metadata={"doc": "line"}, entities=pump)],
    )
    await memory.sync(
        "a",
        filters=doc,
        sources=[SourceIn(name="line", text="Nothing now.", metadata={"doc": "line"})],
    )
    old = ("chunk", f"{first.sources[0].source_id}-0")
    assert [e.name for _, e in await storage.passage_entities("a", [old])] == ["P-1"]
    hits = await memory.search("a", "P-1 runs", pillars={"text", "vector", "graph"})
    assert old[1] not in [h.id for h in hits.hits]


async def test_chunks_around_with_no_width_reads_the_chunk_alone_and_below_reads_nothing(
    storage: PostgresStorage,
) -> None:
    source, chunks, vectors = _source("a", "s1", [f"part {i}" for i in range(3)])
    await storage.put_source("a", source, chunks, vectors, "m")
    now = Window(as_of=NOW)
    alone = await storage.chunks_around("a", "s1", 1, width=0, window=now, filters=None)
    assert [c.chunk_id for c in alone] == ["s1-1"]
    assert await storage.chunks_around("a", "s1", 1, width=-1, window=now, filters=None) == []


async def test_stress_an_end_and_a_supersede_racing_close_a_story_once(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=_Yielding())
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    for round in range(10):
        told = (await memory.remember("a", f"Fact {round}.", valid_from=jan)).story
        ending, superseding = await _race(
            memory.end_story("a", told.story_id),
            memory.supersede(
                "a", told.story_id, f"Fact {round} changed.", valid_from=jan + timedelta(days=1)
            ),
        )
        refused = [r for r in (ending, superseding) if isinstance(r, BaseException)]
        assert len(refused) == 1 and isinstance(refused[0], InputError)
        [closed] = await memory.get_stories("a", [told.story_id])
        if isinstance(ending, Story):
            assert closed == ending and closed.superseded_by is None
        else:
            assert closed.superseded_by == superseding.story.story_id
            assert closed.valid_to == jan + timedelta(days=1)


async def test_replace_source_closes_a_version_once_or_erases_it(storage: PostgresStorage) -> None:
    later = NOW + timedelta(hours=1)
    for corpus, history in (("kept", True), ("erased", False)):
        await storage.set_history(corpus, history)
        source, chunks, vectors = _source(corpus, "s1", ["thirty days"])
        await storage.put_source(corpus, source, chunks, vectors, "m")
        write = SourceWrite(*_source(corpus, "s2", ["sixty days"]))
        with pytest.raises(NotFoundError, match=r"^postgres: source 'missing' was not found$"):
            await storage.replace_source(corpus, "missing", write, "m", at=later)
        done = await storage.replace_source(corpus, "s1", write, "m", at=later)
        assert done is not None
        deleted, retired, written = done
        if history:
            old = await storage.get_source(corpus, "s1")
            assert old is not None and retired == ["s1"] and deleted == Counts()
            assert (old.retired_at, old.replaced_by, old.valid_to) == (later, "s2", later)
            assert written.valid_from == later
            with pytest.raises(
                InputError,
                match=r"^postgres: source 's1' was already replaced by 's2'; replace that one$",
            ):
                await storage.replace_source(
                    corpus, "s1", SourceWrite(*_source(corpus, "s3", ["x"])), "m", at=later
                )
        else:
            assert retired == [] and deleted.sources == 1 and written.valid_from is None
            assert await storage.get_source(corpus, "s1") is None


async def test_stress_two_replaces_of_one_source_close_it_once(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=_Yielding())
    await memory.set_history("a", True)
    for round in range(10):
        old = (await memory.ingest("a", name=f"doc {round}", text=f"Version {round}.")).source
        racers = await _race(
            *(
                memory.ingest(
                    "a", name=f"doc {round}", text=f"{who} {round}.", replaces=old.source_id
                )
                for who in ("Ana", "Bo")
            )
        )
        refused = [r for r in racers if isinstance(r, BaseException)]
        assert len(refused) == 1 and isinstance(refused[0], InputError)
        assert "was already replaced by" in str(refused[0])
        winner = next(r for r in racers if not isinstance(r, BaseException)).source
        closed = await memory.get_source("a", old.source_id)
        assert closed is not None and closed.replaced_by == winner.source_id


async def test_list_sources_reads_the_sources_in_force_at_a_moment(
    storage: PostgresStorage,
) -> None:
    jan, mar = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)
    for source_id, start, end in (
        ("open", None, None),
        ("winter", jan, mar),
        ("spring", mar, None),
    ):
        source, chunks, vectors = _source("a", source_id, [source_id])
        dated = {"valid_from": start, "valid_to": end}
        await storage.put_source(
            "a",
            source.model_copy(update=dated),
            [c.model_copy(update=dated) for c in chunks],
            vectors,
            "m",
        )

    async def listed(**window: Any) -> list[str]:
        found, _ = await storage.list_sources("a", filters=None, limit=10, cursor=None, **window)
        return sorted(s.source_id for s in found)

    assert await listed() == ["open", "spring", "winter"]
    assert await listed(as_of=jan + timedelta(days=1)) == ["open", "winter"]
    assert await listed(as_of=mar) == ["open", "spring"]
    assert await listed(as_of=datetime(2025, 1, 1, tzinfo=UTC)) == ["open"]


async def test_stress_a_replace_racing_a_sync_of_its_set_leaves_one_latest_version(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=_Yielding())
    await memory.set_history("a", True)
    for round in range(10):
        doc = {"metadata.doc": f"r{round}"}
        held = {"doc": f"r{round}"}
        [old] = (
            await memory.sync(
                "a", filters=doc, sources=[SourceIn(name="Rules", text="Old.", metadata=held)]
            )
        ).sources
        racers = await _race(
            memory.ingest(
                "a", name="Rules", text="Replaced.", metadata=held, replaces=old.source_id
            ),
            memory.sync(
                "a", filters=doc, sources=[SourceIn(name="Rules", text="Synced.", metadata=held)]
            ),
        )
        assert all(isinstance(r, InputError) for r in racers if isinstance(r, BaseException))
        latest, _ = await storage.list_sources("a", filters=doc, limit=10, cursor=None)
        assert len(latest) == 1


async def test_facets_select_by_range_in_every_storage_read(storage: PostgresStorage) -> None:
    for source_id, weight in (("light", 0.5), ("heavy", 2), ("named", "x")):
        source, chunks, vectors = _source("a", source_id, [source_id], metadata={"w": weight})
        await storage.put_source("a", source, chunks, vectors, "m")
        await storage.put_story(
            "a", _story("a", source_id, f"{source_id} story", metadata={"w": weight}), None, None
        )
    heavy = {"metadata.w": {"gt": 1}}
    listed, _ = await storage.list_sources("a", filters=heavy, limit=10, cursor=None)
    stories, _ = await storage.list_stories(
        "a", subject=None, window=Window(), filters=heavy, limit=10, cursor=None
    )
    found = await _text(
        storage,
        "a",
        "heavy light named",
        kinds=("chunk",),
        top_k=10,
        filters={"metadata.w": {"lte": 1}},
        window=Window(),
        subject=None,
    )
    assert [s.source_id for s in listed] == ["heavy"] and [s.story_id for s in stories] == ["heavy"]
    assert ids(found) == ["light-0"]


async def test_graph_mentions_select_by_range(storage: PostgresStorage) -> None:
    named = []
    for source_id, weight in (("light", 0.5), ("heavy", 2)):
        source, chunks, vectors = _source(
            "a", source_id, [f"{source_id} valve"], metadata={"w": weight}
        )
        entity = Entity(
            corpus_id="a",
            entity_id=f"e-{source_id}",
            name=f"V-{source_id}",
            key=f"v-{source_id}",
            type="valve",
        )
        graph = GraphWrite(
            entities=[entity],
            mentions=[(entity.entity_id, "chunk", chunks[0].chunk_id, entity.name)],
        )
        await storage.put_source("a", source, chunks, vectors, "m", graph=graph)
        named.append(entity.entity_id)
    found = await storage.mentions(
        "a",
        named,
        kinds=("chunk",),
        filters={"metadata.w": {"gt": 1}},
        window=Window(as_of=NOW),
        subject=None,
    )
    assert [ref for _, ref in found] == [("chunk", "heavy-0")]


def _by(write: SourceWrite, written_by: dict[str, Any]) -> SourceWrite:
    source = write.source.model_copy(update={"written_by": written_by})
    chunks = [c.model_copy(update={"written_by": written_by}) for c in write.chunks]
    return SourceWrite(source, chunks, write.vectors)


async def test_every_write_records_who_wrote_and_every_closing_who_closed(
    storage: PostgresStorage,
) -> None:
    seven, eight = {"run": 7}, {"run": 8}
    jan, feb, mar = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3))
    told = _story("a", "s1", "Ana owns the review.", valid_from=jan)
    assert (await storage.put_story("a", told, None, None)).written_by == {}
    told = await storage.put_story(
        "a",
        _story("a", "s2", "Bo owns the audit.", valid_from=jan).model_copy(
            update={"written_by": seven}
        ),
        None,
        None,
    )
    later = _story("a", "s3", "Cy owns both.", valid_from=feb).model_copy(
        update={"created_at": NOW + timedelta(days=1), "written_by": eight}
    )
    closed, new = await storage.supersede_story("a", ["s1", "s2"], later, None, None)
    assert [(s.written_by, s.closed_by) for s in closed] == [({}, eight), (seven, eight)]
    assert (await storage.get_stories("a", ["s3"]))[0].written_by == eight == new.written_by
    ended = await storage.end_story(
        "a", "s3", valid_to=mar, closed_at=NOW + timedelta(days=2), closed_by=seven
    )
    assert (ended.written_by, ended.closed_by) == (eight, seven)
    shut = _story("a", "s4", "Di owned QA.", valid_from=jan, valid_to=feb).model_copy(
        update={"closed_at": NOW, "written_by": seven}
    )
    assert (await storage.put_story("a", shut, None, None)).closed_by == {}
    assert (await storage.get_stories("a", ["s4"]))[0].closed_by == {}
    returns = {"metadata.doc": "returns"}
    await storage.set_history("a", True)
    first = _by(_version("v1", "Returns policy", "30 days", jan), seven)
    await storage.sync_sources("a", returns, [first], "m", at=jan, written_by=seven)
    second = _by(_version("v2", "Returns policy", "60 days", feb), eight)
    await storage.sync_sources("a", returns, [second], "m", at=feb, written_by=eight)
    v1, v2 = await storage.get_source("a", "v1"), await storage.get_source("a", "v2")
    assert v1 is not None and (v1.written_by, v1.closed_by) == (seven, eight)
    assert v2 is not None and (v2.written_by, v2.closed_by) == (eight, None)
    assert [c.written_by for c in await storage.source_chunks("a", "v2")] == [eight]
    third = _by(_version("v3", "Returns policy", "90 days", mar), seven)
    await storage.replace_source("a", "v2", third, "m", at=mar)
    v2 = await storage.get_source("a", "v2")
    assert v2 is not None and v2.closed_by == seven
    await storage.sync_sources("a", returns, [], "m", at=mar + timedelta(days=1))
    v3 = await storage.get_source("a", "v3")
    assert v3 is not None and (v3.written_by, v3.closed_by) == (seven, {})


def _prose(rng: random.Random, paragraphs: int) -> list[str]:
    words = [
        "seal",
        "pump",
        "valve",
        "tank",
        "line",
        "feed",
        "panel",
        "motor",
        "wiring",
        "order",
        "clause",
        "notice",
        "repair",
        "part",
    ]
    return [" ".join(rng.choice(words) for _ in range(40)) for _ in range(paragraphs)]


async def test_resembling_finds_every_edition_through_the_band_index_and_no_stranger(
    storage: PostgresStorage,
) -> None:
    rng = random.Random(11)
    base = _prose(rng, 24)
    writes = {"base": base}
    for n in range(30):
        rewritten = 1 + n % 6
        writes[f"edition-{n}"] = _prose(rng, rewritten) + base[rewritten:]
    for n in range(10):
        writes[f"stranger-{n}"] = _prose(rng, 24)
    for source_id, paragraphs in writes.items():
        source, chunks, vectors = _source("a", source_id, ["\n\n".join(paragraphs)])
        await storage.put_source("a", source, chunks, vectors, "m")
    copy, chunks, vectors = _source("b", "base", ["\n\n".join(base)])
    await storage.put_source("b", copy, chunks, vectors, "m")
    options: dict[str, Any] = {
        "filters": None,
        "known_at": None,
        "history": False,
        "as_of": None,
        "limit": 100,
    }
    found = await storage.resembling(["a"], ("a", "base"), **options)
    assert sorted(m.source.source_id for m in found) == sorted(f"edition-{n}" for n in range(30))
    assert all(0.6 < m.share < 1.0 for m in found)
    assert [m.share for m in found] == sorted((m.share for m in found), reverse=True)
    probed = await storage.resembling(["a"], "\n\n".join(base), **options)
    assert [(m.source.source_id, m.share) for m in probed][0] == ("base", 1.0)
    assert {m.source.corpus_id for m in probed} == {"a"}
    both = await storage.resembling(["a", "b"], "\n\n".join(base), **options)
    assert [(m.source.corpus_id, m.source.source_id) for m in both[:2]] == [
        ("a", "base"),
        ("b", "base"),
    ]
    for wordless in ("-- !! --", "② ③"):
        with pytest.raises(InputError, match=r"^postgres: the text holds no words to compare$"):
            await storage.resembling(["a"], wordless, **options)


async def test_a_probe_reads_the_band_index_in_a_custom_and_a_generic_plan(
    storage: PostgresStorage, pool: Any
) -> None:
    rng = random.Random(5)
    for n in range(300):
        source, chunks, vectors = _source("a", f"doc-{n}", ["\n\n".join(_prose(rng, 2))])
        await storage.put_source("a", source, chunks, vectors, "m")
    query = (
        "SELECT * FROM memory_sources WHERE corpus_id = ANY($1::text[]) "
        "AND memory_bands(signature) && memory_bands($2::bigint[]) AND retired_at IS NULL"
    )
    async with pool.acquire() as conn:
        await conn.execute("ANALYZE memory_sources")
        signature = await conn.fetchval("SELECT memory_signature($1)", _prose(rng, 1)[0])
        custom = [r[0] for r in await conn.fetch(f"EXPLAIN {query}", ["a"], signature)]
        await conn.execute("SET plan_cache_mode = force_generic_plan")
        await conn.execute(f"PREPARE probe(text[], bigint[]) AS {query}")
        try:
            literal = "{" + ",".join(str(v) for v in signature) + "}"
            generic = [
                r[0] for r in await conn.fetch(f"EXPLAIN EXECUTE probe('{{a}}', '{literal}')")
            ]
        finally:
            await conn.execute("DEALLOCATE probe; RESET plan_cache_mode")
    assert any("ix_memory_sources_bands" in line for line in custom), custom
    assert any("ix_memory_sources_bands" in line for line in generic), generic


async def test_racing_closings_of_one_version_leave_exactly_one(storage: PostgresStorage) -> None:
    await storage.set_history("a", True)
    later = NOW + timedelta(days=1)
    options: dict[str, Any] = {"replaced_by": None, "valid_to": None, "closed_by": {}}
    for round in range(5):
        old, chunks, vectors = _source("a", f"old-{round}", [f"round {round} text"])
        await storage.put_source("a", old, chunks, vectors, "m")
        retires = await _race(
            *(storage.retire_source("a", f"old-{round}", at=later, **options) for _ in range(4))
        )
        assert sum(isinstance(r, Source) for r in retires) == 1
        assert all(isinstance(r, InputError) for r in retires if not isinstance(r, Source))
        held, chunks, vectors = _source("a", f"held-{round}", [f"held {round} text"])
        await storage.put_source("a", held, chunks, vectors, "m")
        new = _version(f"new-{round}", "held", f"held {round} again", later)
        raced = await _race(
            storage.retire_source("a", f"held-{round}", at=later, **options),
            storage.replace_source("a", f"held-{round}", new, "m", at=later),
        )
        assert sum(not isinstance(r, BaseException) for r in raced) == 1
        closed = await storage.get_source("a", f"held-{round}")
        assert closed is not None and closed.retired_at == later


async def test_retire_source_closes_a_held_version_once_inside_its_transaction(
    storage: PostgresStorage,
) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    for source_id in ("old", "new", "other"):
        source, chunks, vectors = _source("a", source_id, [f"{source_id} text"])
        await storage.put_source("a", source, chunks, vectors, "m")
    options: dict[str, Any] = {"replaced_by": None, "valid_to": None, "closed_by": {}}
    with pytest.raises(
        InputError,
        match=r"^postgres: corpus 'a' keeps no history, so a retired document would be erased; "
        r"delete_source erases it$",
    ):
        await storage.retire_source("a", "old", at=NOW + timedelta(days=1), **options)
    await storage.set_history("a", True)
    with pytest.raises(NotFoundError, match=r"^postgres: source 'missing' was not found$"):
        await storage.retire_source("a", "missing", at=NOW + timedelta(days=1), **options)
    with pytest.raises(
        InputError,
        match=r"^postgres: source 'old' was written at 2026-03-01T00:00:00\+00:00, after this "
        r"retirement's moment$",
    ):
        await storage.retire_source("a", "old", at=feb, **options)
    later = NOW + timedelta(days=1)
    retired = await storage.retire_source(
        "a", "old", at=later, replaced_by="new", valid_to=None, closed_by={"run": 7}
    )
    assert (retired.retired_at, retired.replaced_by, retired.valid_to, retired.closed_by) == (
        later,
        "new",
        later,
        {"run": 7},
    )
    assert [c.retired_at for c in await storage.source_chunks("a", "old")] == [later]
    with pytest.raises(
        InputError,
        match=r"^postgres: source 'old' was already retired at 2026-03-02T00:00:00\+00:00, "
        r"replaced by 'new'$",
    ):
        await storage.retire_source("a", "old", at=later, **options)
    with pytest.raises(
        InputError,
        match=r"^postgres: source 'old' was retired at 2026-03-02T00:00:00\+00:00; name the "
        r"version in force$",
    ):
        await storage.retire_source("a", "other", at=later, **{**options, "replaced_by": "old"})
    ended = await storage.retire_source(
        "a", "other", at=later, replaced_by=None, valid_to=jan, closed_by={}
    )
    assert (ended.valid_to, ended.closed_by, ended.replaced_by) == (jan, {}, None)


_COUNTED = "(SELECT sum(coalesce(array_length(u.positions, 1), 1)) FROM unnest(fts) AS u)::int"


async def test_every_passage_stores_the_length_keyword_ranking_reads(
    storage: PostgresStorage, pool: Any
) -> None:
    await storage.set_language("b", "simple")
    text = "The seal of the pump is replaced by the night shift every year."
    for corpus in ("a", "b"):
        source, chunks, vectors = _source(corpus, "s", [text, "Wiring of the panel."])
        titled = [
            chunks[0].model_copy(
                update={"title": "The pump", "context": "Of the maintenance plan"}
            ),
            chunks[1],
        ]
        await storage.put_source(corpus, source, titled, vectors, "m")
        await storage.put_story(corpus, _story(corpus, "t", text), _vec(text), "m")
    async with pool.acquire() as conn:
        rows = [
            (r["kind"], r["corpus_id"], r["fts_length"], r["counted"])
            for r in await conn.fetch(
                f"SELECT 'chunk' AS kind, corpus_id, fts_length, {_COUNTED} AS counted "
                "FROM memory_chunks UNION ALL "
                f"SELECT 'story', corpus_id, fts_length, {_COUNTED} FROM memory_stories "
                "ORDER BY 1, 2, 3"
            )
        ]
    assert len(rows) == 6 and all(stored == counted for _, _, stored, counted in rows)
    lengths = {(kind, corpus): stored for kind, corpus, stored, _ in rows if kind == "story"}
    assert lengths[("story", "a")] < lengths[("story", "b")]


class _Counted:
    def __init__(self, conn: Any, calls: list[str]) -> None:
        self._conn, self._calls = conn, calls

    def __getattr__(self, name: str) -> Any:
        found = getattr(self._conn, name)
        if name not in ("execute", "executemany", "fetch", "fetchrow", "fetchval"):
            return found

        async def counted(*args: Any, **kwargs: Any) -> Any:
            self._calls.append(str(args[0]) if args else name)
            return await found(*args, **kwargs)

        return counted


class _CountingPool:
    def __init__(self, pool: Any, calls: list[str]) -> None:
        self._pool, self._calls = pool, calls

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        async with self._pool.acquire() as conn:
            yield _Counted(conn, self._calls)


def _counting(storage: PostgresStorage) -> tuple[PostgresStorage, list[str]]:
    calls: list[str] = []
    counted = copy.copy(storage)
    counted._pool = _CountingPool(storage._pool, calls)
    return counted, calls


async def test_a_sync_writes_and_deletes_any_number_of_documents_in_the_same_statements(
    storage: PostgresStorage,
) -> None:
    counted, calls = _counting(storage)

    async def synced(name: str, count: int, round: int) -> int:
        writes = [
            SourceWrite(
                *_source(
                    "a", f"{name}-{round}-{n}", [f"{name} {round} {n}"], metadata={"set": name}
                )
            )
            for n in range(count)
        ]
        calls.clear()
        done = await counted.sync_sources("a", {"metadata.set": name}, writes, "m", at=NOW)
        assert done is not None and len(done[2]) == count
        return len(calls)

    assert await synced("one", 1, 0) == await synced("many", 20, 0)
    assert await synced("one", 1, 1) == await synced("many", 20, 1)
    listed, _ = await storage.list_sources(
        "a", filters={"metadata.set": "many"}, limit=100, cursor=None
    )
    assert sorted(s.source_id for s in listed) == sorted(f"many-1-{n}" for n in range(20))


def _graphed(source_id: str, count: int) -> tuple[SourceWrite, GraphWrite]:
    write = SourceWrite(*_source("a", source_id, [f"{source_id} names its parts"]))
    chunk_id = write.chunks[0].chunk_id
    entities = [
        Entity(
            corpus_id="a",
            entity_id=f"{source_id}-e{n}",
            name=f"Part {source_id} {n}",
            key=f"part {source_id} {n}",
            type="part",
        )
        for n in range(count)
    ]
    return write, GraphWrite(
        entities=entities,
        aliases=[(e.entity_id, f"alias {e.entity_id}") for e in entities],
        mentions=[
            (e.entity_id, "chunk", chunk_id, name)
            for e in entities
            for name in (e.name, f"alias {e.entity_id}", f"code {e.entity_id}")
        ],
    )


async def test_a_graph_write_upserts_any_number_of_entities_in_the_same_statements(
    storage: PostgresStorage,
) -> None:
    counted, calls = _counting(storage)
    sizes = []
    for source_id, count in (("one", 1), ("many", 20)):
        write, graph = _graphed(source_id, count)
        calls.clear()
        await counted.put_source("a", *write, "m", graph=graph)
        sizes.append(len(calls))
    assert sizes[0] == sizes[1]
    [entity] = await storage.entities_by_id("a", ["many-e7"])
    assert (entity.name, entity.aliases) == ("Part many 7", ("alias many-e7", "code many-e7"))


async def test_merging_and_erasing_any_number_of_rows_takes_the_same_statements(
    storage: PostgresStorage,
) -> None:
    counted, calls = _counting(storage)
    merged = []
    for name, count in (("one", 1), ("many", 10)):
        olds = [_story("a", f"{name}-{n}", f"{name} story {n}") for n in range(count)]
        for old in olds:
            await storage.put_story("a", old, None, None)
        new = _story("a", f"{name}-new", f"{name} merged").model_copy(
            update={"created_at": NOW + timedelta(days=1)}
        )
        calls.clear()
        await counted.supersede_story("a", [o.story_id for o in olds], new, None, None)
        merged.append(len(calls))
    assert merged[0] == merged[1]
    await storage.set_history("a", True)

    def named(source_id: str, name: str, facet: str, value: str) -> SourceWrite:
        source, chunks, vectors = _source("a", source_id, [source_id], metadata={facet: value})
        return SourceWrite(source.model_copy(update={"name": name}), chunks, vectors)

    erased, forgotten = [], []
    for name, count in (("one", 1), ("many", 10)):
        for n in range(count + 1):
            await storage.sync_sources(
                "a",
                {"metadata.chain": name},
                [named(f"{name}-v{n}", name, "chain", name)],
                "m",
                at=NOW + timedelta(days=n + 1),
            )
        calls.clear()
        done = await counted.delete_source("a", f"{name}-v{count}")
        assert done.sources == count + 1
        erased.append(len(calls))
    for name, count in (("one", 1), ("many", 10)):
        for n in range(count + 1):
            await storage.sync_sources(
                "a",
                {"metadata.kept": name},
                [named(f"{name}-k{n}", name, "kept", name)],
                "m",
                at=NOW + timedelta(days=20 + n),
            )
        calls.clear()
        done = await counted.forget_retired("a", before=NOW + timedelta(days=40))
        assert done.sources == count
        forgotten.append(len(calls))
    assert erased[0] == erased[1] and forgotten[0] == forgotten[1]


async def test_stress_writers_mentioning_each_others_entities_never_deadlock(
    storage: PostgresStorage,
) -> None:
    def part(entity_id: str, key: str) -> Entity:
        return Entity(corpus_id="a", entity_id=entity_id, name=key.upper(), key=key, type="part")

    await storage.put_source(
        "a",
        *_source("a", "seed", ["P and Q"]),
        "m",
        graph=GraphWrite(
            entities=[part("P", "p"), part("Q", "q")],
            mentions=[("P", "chunk", "seed-0", "P"), ("Q", "chunk", "seed-0", "Q")],
        ),
    )
    for round in range(20):

        def write(own: str, other: str, round: int = round) -> Awaitable[Any]:
            source_id = f"w{round}-{own}"
            staged = part(f"{source_id}-e", own.lower())
            return storage.put_source(
                "a",
                *_source("a", source_id, [f"{own} and {other}"]),
                "m",
                graph=GraphWrite(
                    entities=[staged],
                    mentions=[
                        (staged.entity_id, "chunk", f"{source_id}-0", own),
                        (other, "chunk", f"{source_id}-0", other),
                    ],
                ),
            )

        raced = await _race(write("Q", "P"), write("P", "Q"))
        assert not [r for r in raced if isinstance(r, BaseException)], raced


async def _jittered(rng: random.Random, writer: Awaitable[Any]) -> Any:
    await asyncio.sleep(rng.uniform(0, 0.02))
    return await writer


async def test_stress_ingests_racing_a_delete_of_their_shared_entities_never_deadlock(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    rng = random.Random(23)
    for round in range(20):
        names = [f"Pump P-{round}{i:03d}" for i in range(30)]
        entities = [EntityIn(name=name, type="pump") for name in names]
        relations = [
            RelationIn(from_name=a, to_name=b, type="FEEDS")
            for a, b in zip(names, names[1:] + names[:1], strict=True)
        ]
        text = ". ".join(names)
        first = await memory.ingest(
            "a", name=f"m{round}", text=text, entities=entities, relations=relations
        )
        rng.shuffle(entities)
        raced = await _race(
            *(
                _jittered(
                    rng,
                    memory.ingest(
                        "a",
                        name=f"m{round}-{w}",
                        text=text,
                        entities=entities,
                        relations=relations,
                    ),
                )
                for w in range(2)
            ),
            _jittered(rng, memory.delete_source("a", first.source.source_id)),
        )
        assert not [r for r in raced if isinstance(r, BaseException)], raced
        assert (await storage.count_corpus("a")).entities == 30 * (round + 1)


def _filed(source_id: str, text: str, doc: str) -> SourceWrite:
    source, chunks, vectors = _source("a", source_id, [text], metadata={"doc": doc})
    return SourceWrite(source, chunks, vectors)


def _appending(held: list[Entity], passage_id: str, alias: str) -> GraphWrite:
    return GraphWrite(
        aliases=[(e.entity_id, f"{e.key} {alias}") for e in held],
        mentions=[
            (e.entity_id, "chunk", passage_id, name)
            for e in held
            for name in (e.key, f"{e.key} {alias}")
        ],
        known=held,
    )


async def test_stress_an_ingest_appending_aliases_racing_a_sync_never_deadlocks(
    storage: PostgresStorage,
) -> None:
    rng = random.Random(29)
    for round in range(30):
        keys = [f"q{round}-{i:03d}" for i in range(30)]
        held = [
            Entity(corpus_id="a", entity_id=f"{key}-e", name=key, key=key, type="part")
            for key in keys
        ]
        doc, text = f"d{round}", " ".join(keys)
        await storage.sync_sources(
            "a",
            {"metadata.doc": doc},
            [_filed(f"v{round}-0", text, doc)],
            "m",
            at=NOW + timedelta(minutes=round),
            graph=GraphWrite(
                entities=held,
                mentions=[(e.entity_id, "chunk", f"v{round}-0-0", e.key) for e in held],
            ),
        )
        raced = await _race(
            _jittered(
                rng,
                storage.put_source(
                    "a",
                    *_source("a", f"i{round}", [text]),
                    "m",
                    graph=_appending(held, f"i{round}-0", "by ingest"),
                ),
            ),
            _jittered(
                rng,
                storage.sync_sources(
                    "a",
                    {"metadata.doc": doc},
                    [_filed(f"v{round}-1", text + " again", doc)],
                    "m",
                    at=NOW + timedelta(minutes=round, seconds=30),
                    graph=_appending(held, f"v{round}-1-0", "by sync"),
                ),
            ),
        )
        assert not [r for r in raced if isinstance(r, BaseException)], raced
        for entity in await storage.entities_by_id("a", [e.entity_id for e in held]):
            assert set(entity.aliases) == {f"{entity.key} by ingest", f"{entity.key} by sync"}


async def _blocked(admin: Any, by: list[int], count: int) -> list[int]:
    for _ in range(1000):
        waiting = [
            r["pid"]
            for r in await admin.fetch(
                "SELECT pid FROM pg_stat_activity WHERE pg_blocking_pids(pid) && $1::int[] "
                "ORDER BY pid",
                by,
            )
        ]
        if len(waiting) >= count:
            return waiting
        await asyncio.sleep(0.01)
    raise AssertionError(f"tests: {count} writers never waited behind {by}")


def _part(entity_id: str, key: str, *aliases: str) -> Entity:
    return Entity(
        corpus_id="a", entity_id=entity_id, name=key, key=key, type="part", aliases=aliases
    )


async def test_a_stale_ingest_and_a_delete_of_its_entities_take_turns(
    storage: PostgresStorage,
) -> None:
    await storage.put_source(
        "a",
        *_source("a", "s0", ["X and Y"]),
        "m",
        graph=GraphWrite(
            entities=[_part("zz-x", "a-x"), _part("yy-y", "b-y")],
            mentions=[("zz-x", "chunk", "s0-0", "a-x"), ("yy-y", "chunk", "s0-0", "b-y")],
        ),
    )
    stale = GraphWrite(
        entities=[_part("fresh-x", "a-x"), _part("fresh-y", "b-y")],
        mentions=[("fresh-x", "chunk", "s1-0", "a-x"), ("fresh-y", "chunk", "s1-0", "b-y")],
    )
    async with storage._pool.acquire() as admin, storage._pool.acquire() as gate:
        async with gate.transaction():
            await gate.execute(
                "SELECT 1 FROM memory_entities WHERE corpus_id = 'a' FOR NO KEY UPDATE"
            )
            ingest = asyncio.ensure_future(
                storage.put_source("a", *_source("a", "s1", ["X and Y again"]), "m", graph=stale)
            )
            [first] = await _blocked(admin, [gate.get_server_pid()], 1)
            delete = asyncio.ensure_future(storage.delete_source("a", "s0"))
            await _blocked(admin, [gate.get_server_pid(), first], 2)
        raced = await _race(ingest, delete)
    assert not [r for r in raced if isinstance(r, BaseException)], raced
    assert [
        (e.entity_id, e.key) for e in (await storage.entities_named("a", ["a-x"]))[name_key("a-x")]
    ] == [("zz-x", "a-x")]
    assert (await storage.count_corpus("a")).entities == 2


async def test_a_rename_on_delete_and_an_ingest_of_that_name_take_turns(
    storage: PostgresStorage,
) -> None:
    await storage.put_source(
        "a",
        *_source("a", "s0", ["old"]),
        "m",
        graph=GraphWrite(
            entities=[_part("e", "old", "new")], mentions=[("e", "chunk", "s0-0", "old")]
        ),
    )
    await storage.put_source(
        "a",
        *_source("a", "s1", ["new", "r"]),
        "m",
        graph=GraphWrite(
            entities=[_part("a-r", "r")],
            mentions=[("e", "chunk", "s1-0", "new"), ("a-r", "chunk", "s1-1", "r")],
        ),
    )
    racing = GraphWrite(
        entities=[_part("n2", "new")],
        mentions=[
            ("n2", "chunk", "s2-0", "new"),
            ("a-r", "chunk", "s2-0", "r"),
            ("e", "chunk", "s2-0", "old"),
        ],
    )
    async with storage._pool.acquire() as admin, storage._pool.acquire() as gate:
        async with gate.transaction():
            await gate.execute(
                "SELECT 1 FROM memory_entities WHERE corpus_id = 'a' AND entity_id = 'a-r' FOR UPDATE"
            )
            ingest = asyncio.ensure_future(
                storage.put_source("a", *_source("a", "s2", ["new r old"]), "m", graph=racing)
            )
            [first] = await _blocked(admin, [gate.get_server_pid()], 1)
            delete = asyncio.ensure_future(storage.delete_source("a", "s0"))
            await _blocked(admin, [gate.get_server_pid(), first], 2)
        raced = await _race(ingest, delete)
    assert not [r for r in raced if isinstance(r, BaseException)], raced
    [kept] = await storage.entities_by_id("a", ["e"])
    assert (kept.key, kept.aliases) == ("old", ("new",))


async def test_graph_writes_in_a_corpus_wait_for_its_graph_lock(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    pump = [EntityIn(name="P-3", type="pump")]
    told = await memory.remember("a", "P-3 feeds the line.", entities=pump)
    holder = await asyncpg.connect(os.environ[URL])
    try:
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock($1)", await _lock_key(storage, "memory.graph:a")
            )
            writing = asyncio.create_task(
                memory.ingest("a", name="m", text="P-3 runs.", entities=pump)
            )
            deleting = asyncio.create_task(memory.delete_story("a", told.story.story_id))
            await asyncio.wait_for(memory.ingest("a", name="plain", text="no names here"), 5)
            await asyncio.wait_for(memory.ingest("b", name="m", text="P-3 runs.", entities=pump), 5)
            await asyncio.sleep(0.5)
            assert not writing.done() and not deleting.done()
        await asyncio.wait_for(asyncio.gather(writing, deleting), 10)
    finally:
        await holder.close()
    assert [e.name for e in (await memory.neighbors("a", "P-3")).entities] == ["P-3"]


def _counted_bm25(
    settings: BM25,
    *,
    table: str,
    id_column: str,
    matches: str,
    ranks: str,
    terms: str,
    where: str,
    limit: str,
    kind: str,
) -> str:
    k1, b = settings.k1, settings.b
    searched = _CHUNK_SEARCHED if table == "memory_chunks" else "text"
    return (
        f"WITH matched AS (SELECT corpus_id AS corpus, {id_column} AS id, fts, "
        f"to_tsvector(config::regconfig, {searched}) AS words FROM {table} "
        f"WHERE corpus_id = ANY($1::text[]) AND fts @@ {matches}{where}), "
        f"terms AS (SELECT DISTINCT l AS lexeme FROM unnest(tsvector_to_array({terms})) AS l), "
        "sized AS (SELECT m.corpus, m.id, (SELECT sum(coalesce(array_length(u.positions, 1), 1)) "
        "FROM unnest(m.words) AS u)::float8 AS len FROM matched AS m), "
        "freq AS (SELECT m.corpus, m.id, u.lexeme, "
        "coalesce(array_length(u.positions, 1), 1)::float8 AS tf "
        "FROM matched AS m CROSS JOIN LATERAL unnest(m.fts) AS u "
        "WHERE u.lexeme IN (SELECT lexeme FROM terms)), "
        "stats AS (SELECT count(*)::float8 AS n, avg(len) AS avgdl FROM sized), "
        "df AS (SELECT lexeme, count(*)::float8 AS df FROM freq GROUP BY lexeme) "
        f"SELECT f.corpus, f.id, sum(round(ln(1 + (stats.n - df.df + 0.5) / (df.df + 0.5)) * f.tf * {k1 + 1} "
        f"/ (f.tf + {k1} * (1 - {b} + {b} * sized.len / stats.avgdl)) * 1e9::float8)) / 1e9::float8 AS score, "
        "0::float8 AS tie "
        "FROM freq AS f JOIN df ON df.lexeme = f.lexeme "
        "JOIN sized ON sized.corpus = f.corpus AND sized.id = f.id CROSS JOIN stats "
        "GROUP BY f.corpus, f.id "
        f'ORDER BY score DESC, f.id COLLATE "C", f.corpus COLLATE "C" LIMIT {limit}'
    )


async def test_bm25_scores_equal_the_query_that_counted_each_passage(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts = [
        "The seal of the pump is replaced every year by the night shift.",
        "Seal seal seal: the pump seal again, and the seal once more.",
        "Wiring of the panel and the pump.",
        " ".join(["pump"] * 3 + ["valve"] * 300),
        "the of and",
        "Replace part A-1001 and seal A-1001 on Line3_PLC:Motor1_Run.",
    ]
    for corpus in ("a", "b"):
        source, chunks, vectors = _source(corpus, "s", texts)
        titled = [chunks[0].model_copy(update={"title": "Pump seal", "context": "Maintenance"})]
        await storage.put_source(corpus, source, titled + chunks[1:], vectors, "m")
        for n, text in enumerate(texts):
            await storage.put_story(corpus, _story(corpus, f"t{n}", text), None, None)
    options: dict[str, Any] = {
        "kinds": ("chunk", "story"),
        "top_k": 50,
        "filters": None,
        "window": Window(),
        "subject": None,
        "ranking": BM25(),
    }
    for corpora in (["a"], ["a", "b"]):
        for match in ("any", "all", "phrase", "web"):
            for query in (
                "pump seal",
                "the seal of the pump",
                "valve",
                "seal -wiring",
                "seal A-1001",
            ):
                stored = await storage.search_text(corpora, query, match=match, **options)
                with monkeypatch.context() as patched:
                    patched.setattr("rfnry_memory.ranking.bm25.sql", _counted_bm25)
                    counted = await storage.search_text(corpora, query, match=match, **options)
                assert [(c, leg[0], leg[1]) for c, leg in stored] == [
                    (c, leg[0], leg[1]) for c, leg in counted
                ]
                assert [leg[2] for _, leg in stored] == pytest.approx(
                    [leg[2] for _, leg in counted], rel=1e-12
                )


async def test_a_graph_write_merges_entities_staged_twice_under_one_key(
    storage: PostgresStorage,
) -> None:
    def part(entity_id: str, name: str, aliases: tuple[str, ...]) -> Entity:
        return Entity(
            corpus_id="a",
            entity_id=entity_id,
            name=name,
            key="pump 3",
            type="part",
            aliases=aliases,
        )

    held = part("held", "Pump 3", ("P-3",))
    await storage.put_source(
        "a",
        *_source("a", "first", ["Pump 3"]),
        "m",
        graph=GraphWrite(entities=[held], mentions=[("held", "chunk", "first-0", "Pump 3")]),
    )
    twice = [part("x1", "Pump 3", ("P3", "P-3")), part("x2", "pump 3", ("P3", "Pump three"))]
    await storage.put_source(
        "a",
        *_source("a", "second", ["Pump three"]),
        "m",
        graph=GraphWrite(
            entities=twice,
            mentions=[("x1", "chunk", "second-0", "Pump 3"), ("x2", "chunk", "second-0", "P3")],
        ),
    )
    [entity] = await storage.entities_by_id("a", ["held"])
    assert entity.aliases == ("P-3", "P3", "Pump three")
    assert await storage.entities_by_id("a", ["x1", "x2"]) == []
    mentioned = await storage.passage_entities("a", [("chunk", "second-0")])
    assert {e.entity_id for _, e in mentioned} == {"held"}


async def test_a_batch_of_names_finds_what_each_name_finds_alone(storage: PostgresStorage) -> None:
    def entity(corpus: str, entity_id: str, key: str, kind: str, *aliases: str) -> Entity:
        return Entity(
            corpus_id=corpus, entity_id=entity_id, name=key, key=key, type=kind, aliases=aliases
        )

    held = {
        "a": [
            entity("a", "e1", "p-3", "pump", "Pump three", "P3"),
            entity("a", "e2", "p-3", "part"),
            entity("a", "e4", "valve", "part", "P-3"),
            entity("a", "tank", "tank", "part", "TANK"),
        ],
        "b": [entity("b", "e5", "p-3", "pump", "Pump three"), entity("b", "e6", "nothing", "part")],
    }
    for corpus, entities in held.items():
        await storage.put_story(
            corpus,
            _story(corpus, "names", "names"),
            None,
            None,
            graph=GraphWrite(
                entities=entities,
                mentions=[(e.entity_id, "story", "names", e.name) for e in entities],
            ),
        )
    asked = ["P-3", " p-3 ", "Pump three", "pump  THREE", "p3", "tank", "nothing", "TANK"]
    batch = await storage.entities_named("a", asked)
    alone: dict[str, list[Entity]] = {}
    for name in asked:
        alone.update(await storage.entities_named("a", [name]))
    assert batch == alone
    assert {key: [e.entity_id for e in found] for key, found in batch.items()} == {
        "p-3": ["e1", "e2", "e4"],
        "pump three": ["e1"],
        "p3": ["e1"],
        "tank": ["tank"],
        "nothing": [],
    }
    assert all(e.corpus_id == "a" for found in batch.values() for e in found)
    assert await storage.entities_named("a", []) == {}
    assert [e.entity_id for e in (await storage.entities_named("b", asked))["p-3"]] == ["e5"]


async def test_every_alias_write_keeps_the_keys_names_are_read_by(
    storage: PostgresStorage, pool: Any
) -> None:
    async def keyed() -> list[str]:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT entity_id, aliases, alias_keys FROM memory_entities ORDER BY entity_id"
            )
        for r in rows:
            assert r["alias_keys"] == alias_keys(json.loads(r["aliases"])), r["entity_id"]
        return [k for r in rows for k in r["alias_keys"]]

    async def write(source_id: str, graph: GraphWrite) -> None:
        await storage.put_source("a", *_source("a", source_id, [source_id]), "m", graph=graph)

    pump = Entity(
        corpus_id="a",
        entity_id="e1",
        name="Pump 3",
        key="pump 3",
        type="part",
        aliases=("Pump\xa0P-3", "ΟΔΟΣ", "pump p-3"),
    )
    await write(
        "s1",
        GraphWrite(
            entities=[pump],
            mentions=[("e1", "chunk", "s1-0", "Pump 3"), ("e1", "chunk", "s1-0", "ΟΔΟΣ")],
        ),
    )
    assert await keyed() == ["pump p-3", "οδος"]
    again = pump.model_copy(update={"entity_id": "e1b", "aliases": ("İzmir Plant", "PUMP P-3")})
    await write(
        "s2", GraphWrite(entities=[again], mentions=[("e1b", "chunk", "s2-0", "İzmir Plant")])
    )
    assert await keyed() == ["pump p-3", "οδος", "i\u0307zmir plant"]
    await write(
        "s3",
        GraphWrite(aliases=[("e1", "Line\x1cA")], mentions=[("e1", "chunk", "s3-0", "Line\x1cA")]),
    )
    await write("s4", GraphWrite(mentions=[("e1", "chunk", "s4-0", "Pump\tthree")]))
    assert await keyed() == ["pump p-3", "οδος", "i\u0307zmir plant", "line a", "pump three"]
    await storage.delete_source("a", "s2")
    assert await keyed() == ["οδος", "line a", "pump three"]
    await storage.delete_source("a", "s1")
    [renamed] = await storage.entities_by_id("a", ["e1"])
    assert (renamed.key, renamed.aliases) == ("line a", ("Pump\tthree",))
    assert await keyed() == ["pump three"]
    tank = Entity(
        corpus_id="a",
        entity_id="e9",
        name="Tank",
        key="tank",
        type="part",
        aliases=("Tank\xa0T-9",),
    )
    await write("s5", GraphWrite(entities=[tank], mentions=[("e9", "chunk", "s5-0", "Tank")]))
    await storage.delete_source("a", "s5")
    assert await storage.entities_by_id("a", ["e9"]) == []
    await write(
        "s6",
        GraphWrite(
            mentions=[("e9", "chunk", "s6-0", "Tank"), ("e9", "chunk", "s6-0", "Tank\xa0T-9")],
            known=[tank],
        ),
    )
    assert await keyed() == ["pump three", "tank t-9"]


async def test_two_schemas_never_wait_on_each_others_locks_for_one_corpus_id(
    schemas: Schemas,
) -> None:
    held_pool = await schemas.pool(await schemas.create())
    other_pool = await schemas.pool(await schemas.create())
    await provision(held_pool)
    other = await provision(other_pool)
    async with held_pool.acquire() as conn, conn.transaction():
        await _sync_lock(conn, "a")
        await _language_lock(conn, "a", shared=False)
        await _text_lock(conn, "a", [text_hash("Ana owns the review.")])
        async with _vectors_lock(conn, "a"):
            write = SourceWrite(*_source("a", "s", ["seal"], metadata={"set": "x"}))
            done = await asyncio.wait_for(
                other.sync_sources("a", {"metadata.set": "x"}, [write], "m", at=NOW), timeout=5
            )
            assert done is not None
            told = await asyncio.wait_for(
                other.put_story("a", _story("a", "t", "Ana owns the review."), None, None),
                timeout=5,
            )
            assert told.story_id == "t"
            async with other_pool.acquire() as elsewhere:
                await asyncio.wait_for(_vectors_lock(elsewhere, "a").__aenter__(), timeout=5)
                await elsewhere.execute("SELECT pg_advisory_unlock_all()")


async def test_the_alias_lookup_reads_the_alias_index_analyzed_or_not(schemas: Schemas) -> None:
    pool = await schemas.pool(await schemas.create())
    await provision(pool, dimension=None)
    async with pool.acquire() as conn:
        await conn.execute("ALTER TABLE memory_entities SET (autovacuum_enabled = false)")
        await conn.execute(
            "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases, alias_keys, "
            "config) SELECT c, c || '-' || i, 'part ' || i, 'Part ' || i, 'part', "
            "CASE WHEN i % 2 = 0 THEN jsonb_build_array('P' || i) ELSE '[]' END, "
            "CASE WHEN i % 2 = 0 THEN ARRAY['p' || i] ELSE '{}' END, 'pg_catalog.english' "
            "FROM generate_series(1, 3000) AS i, unnest(ARRAY['a', 'b']) AS c"
        )
        await conn.execute(f"PREPARE named_probe(text, text[]) AS {_NAMED_SQL}")
        for analyzed in (False, True):
            if analyzed:
                await conn.execute("ANALYZE memory_entities")
            assert (
                await conn.fetchval(
                    "SELECT reltuples >= 0 FROM pg_class WHERE oid = 'memory_entities'::regclass"
                )
                == analyzed
            )
            for mode in ("force_custom_plan", "force_generic_plan"):
                await conn.execute(f"SET plan_cache_mode = {mode}")
                for keys in ("ARRAY['p8']", "ARRAY['part 7', 'p8', 'nothing']"):
                    [plan] = json.loads(
                        await conn.fetchval(
                            f"EXPLAIN (FORMAT JSON) EXECUTE named_probe('a', {keys})"
                        )
                    )
                    nodes = list(_nodes(plan["Plan"]))
                    used = {n.get("Index Name") for n in nodes} - {None}
                    assert "ix_memory_entities_alias_keys" in used, (analyzed, mode, used)
                    assert "memory_entities_pkey" not in used, (analyzed, mode, used)
                    assert not any(n["Node Type"] == "Seq Scan" for n in nodes), (analyzed, mode)
                [ran] = json.loads(
                    await conn.fetchval(
                        "EXPLAIN (ANALYZE, FORMAT JSON) EXECUTE named_probe('a', ARRAY['p8'])"
                    )
                )
                read = [
                    n["Actual Rows"]
                    for n in _nodes(ran["Plan"])
                    if n["Node Type"] == "Bitmap Heap Scan"
                    and any(
                        c.get("Index Name") == "ix_memory_entities_alias_keys" for c in n["Plans"]
                    )
                ]
                assert read == [1], (analyzed, mode, read)
        await conn.execute("RESET plan_cache_mode")
        await conn.execute("DEALLOCATE named_probe")


def _nodes(plan: dict[str, Any]) -> Any:
    yield plan
    for child in plan.get("Plans", []):
        yield from _nodes(child)
