import asyncio
import os
import re
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256
from typing import Any

import pytest
from unit.fakes import HashEmbedding

from rfnry_memory.errors import InputError, NotFoundError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    Chunk,
    CorpusFingerprint,
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
from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.storage.storage import GraphWrite, SourceWrite, Storage, chunk_digest, text_hash
from rfnry_memory.tools import MemoryTools


async def _call(tools: MemoryTools, name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await tools.dispatch(name, args)
    assert isinstance(result, dict)
    return result


NOW = datetime(2026, 3, 1, tzinfo=UTC)


def ids(legs: list[Leg]) -> list[str]:
    return [i for _, i, _ in legs]


async def _text(storage: Storage, corpus_id: str, query: str, **options: Any) -> list[Leg]:
    found = await storage.search_text([corpus_id], query, **options)
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
        )
        for i, t in enumerate(texts)
    ]
    return source, chunks, [_vec(t) for t in texts]


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


async def test_sources_roundtrip_list_and_delete(storage: Storage) -> None:
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


async def test_list_sources_pages_with_a_cursor(storage: Storage) -> None:
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
    storage: Storage,
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


async def test_the_text_pillar_stems_words_and_drops_stop_words(storage: Storage) -> None:
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


async def test_a_chunk_is_found_by_the_words_of_its_title(storage: Storage) -> None:
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


async def test_vector_search_orders_by_cosine(storage: Storage) -> None:
    source, chunks, vectors = _source("a", "s1", ["alpha", "beta", "gamma"])
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    legs = await storage.search_vector(
        "a", _vec("beta"), kinds=("chunk",), top_k=2, filters=None, window=Window(), subject=None
    )
    assert legs[0][1] == "s1-1" and legs[0][2] == pytest.approx(1.0) and len(legs) == 2


async def test_story_windows_apply_inside_search(storage: Storage) -> None:
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


async def test_supersede_is_one_step_and_hash_finds_only_open_stories(storage: Storage) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    old = _story("a", "old", "The date is Feb 20.", valid_from=jan)
    await storage.put_story("a", old, None, None)
    assert (await storage.find_open_story("a", text_hash(old.text), at=NOW)) == old
    new = _story("a", "new", "The date is Mar 3.", valid_from=feb).model_copy(
        update={"supersedes": ("old",)}
    )
    [closed], created = await storage.supersede_story("a", ["old"], new, None, None)
    assert closed.valid_to == feb and created.supersedes == ("old",)
    assert await storage.find_open_story("a", text_hash(old.text), at=NOW) is None
    assert await storage.find_open_story("a", text_hash(new.text), at=NOW) == created
    with pytest.raises(NotFoundError):
        await storage.supersede_story(
            "a", ["ghost"], new.model_copy(update={"story_id": "x"}), None, None
        )
    with pytest.raises(
        InputError,
        match=r"^(inmemory|postgres): story 'old' is already superseded; supersede the story that replaced it$",
    ):
        await storage.supersede_story(
            "a",
            ["old"],
            new.model_copy(update={"story_id": "y", "text": "The date is Apr 4."}),
            None,
            None,
        )


async def test_graph_entities_relations_mentions_and_traversal(storage: Storage) -> None:
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
    assert (await storage.entities_named("a", "p-3")) == [p] and (
        await storage.entities_named("a", "P-3")
    ) == [p]
    assert [e.entity_id for e in await storage.entities_by_id("a", ["e3", "ghost", "e1"])] == [
        "e3",
        "e1",
    ]
    assert [e.entity_id for e in await storage.entities_named_in("a", "is V-101 open?")] == ["e1"]
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
    assert (await storage.entities_named("a", "pump three")) == [
        p.model_copy(update={"aliases": ("Pump three",)})
    ]
    deleted = await storage.delete_source("a", "s1")
    assert deleted.mentions == 2 and deleted.relations == 2
    assert await storage.neighbors("a", ["e1"], relation_types=None, window=Window()) == []


async def test_passages_hydrate_in_order_and_reembed_helpers(storage: Storage) -> None:
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


async def test_delete_corpus_is_total_and_counted(storage: Storage) -> None:
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


async def test_aliases_match_by_normalized_name(storage: Storage) -> None:
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
    assert (await storage.entities_named("a", "pump three")) != []
    assert [
        e.entity_id for e in await storage.entities_named_in("a", "where is PUMP three now")
    ] == ["e1"]
    assert [
        e.entity_id for e in await storage.entities_named_in("a", " where is\tPUMP \n three now")
    ] == ["e1"]
    assert [e.entity_id for e in await storage.entities_named("a", " \tP-3 ")] == ["e1"]


async def test_metadata_and_locators_round_trip_exactly(storage: Storage) -> None:
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


async def test_ties_order_by_code_point_whatever_the_collation(storage: Storage) -> None:
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
    assert [e.entity_id for e in await storage.entities_named_in("a", "the valve")] == sorted(tied)
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


async def test_text_with_a_nul_character_is_refused_on_every_storage(storage: Storage) -> None:
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


async def test_a_number_that_is_not_finite_is_refused_on_every_storage(storage: Storage) -> None:
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
    storage: Storage,
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
    storage: Storage,
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


async def test_a_write_whose_graph_is_refused_leaves_nothing_behind(storage: Storage) -> None:
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


async def test_a_write_carries_its_graph_in_the_same_step(storage: Storage) -> None:
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
    assert (await storage.entities_named("a", "pump three")) == [
        pump.model_copy(update={"aliases": ("Pump three",)})
    ]
    residue = await storage.count_corpus("a")
    assert (residue.stories, residue.entities, residue.relations, residue.mentions) == (2, 4, 3, 7)


async def test_an_entity_created_by_a_concurrent_writer_is_joined_not_duplicated(
    storage: Storage,
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
    assert (await storage.entities_named("a", "pump three")) == [
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
    [joined] = await storage.entities_named("a", "v-7")
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


async def test_memory_writes_graph_and_passage_as_one_step(storage: Storage) -> None:
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


async def test_subject_narrows_inside_every_leg_not_after_the_fetch(storage: Storage) -> None:
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
    storage: Storage,
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


async def test_bound_tools_recall_what_they_remember(storage: Storage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.remember("a", "Line B owns the wiring review.", metadata={"line": "B"})
    tools = MemoryTools(
        memory, corpus_id="a", filters={"metadata.line": "A", "metadata.site.zone": 3}
    )
    remembered = await _call(tools, "remember", {"text": "Ana owns the wiring review."})
    assert remembered["story"]["metadata"] == {"line": "A", "site": {"zone": 3}}
    recalled = await _call(tools, "recall", {"query": "wiring review"})
    assert [h["id"] for h in recalled["hits"]] == [remembered["story"]["story_id"]]
    hidden = r"^tools: this text already exists outside your view$"
    with pytest.raises(InputError, match=hidden):
        await _call(tools, "remember", {"text": "line b owns the  wiring review."})
    with pytest.raises(InputError, match=hidden):
        await _call(
            tools,
            "supersede",
            {"story_id": remembered["story"]["story_id"], "text": "Line B owns the wiring review."},
        )


async def test_neighbors_asks_for_a_type_when_a_name_names_several(storage: Storage) -> None:
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
        e.type for e in await storage.entities_named("a", "p-3")
    ] == ["part"]
    await memory.ingest(
        "a",
        name="sheet",
        text="Valve V-1 feeds pump P-3.",
        entities=facts,
        relations=[feeds.model_copy(update={"to_type": "pump"})],
    )
    assert sorted(e.type for e in await storage.entities_named("a", "p-3")) == ["part", "pump"]
    assert await storage.entities_named("b", "p-3") == []
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


async def test_one_open_story_per_text_under_concurrent_remembers(storage: Storage) -> None:
    memory = Memory(storage, embedding=_Yielding())
    racers = await asyncio.gather(
        *(
            memory.remember(
                "a", f"The go-live is  {'MARCH' if i % 2 else 'march'} 3.", subject=f"s{i}"
            )
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
    storage: Storage,
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
    held = r"^(inmemory|postgres): story 'holder', true since 2026-01-01T00:00:00\+00:00, already says that; supersede or delete it instead$"
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


async def test_sync_sources_swaps_the_matching_set_in_one_step(storage: Storage) -> None:
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
    assert done == (Counts(sources=2, chunks=4, mentions=2), [])
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
            r"^storage: graph facts must be evidenced by a passage of the same write$",
        ),
        (
            [SourceWrite(*_source("b", "new2", ["tank T-9"], metadata={"file": "a"}))],
            None,
            r"^(inmemory|postgres): row corpus does not match the call corpus$",
        ),
    ):
        with pytest.raises(StorageError, match=refused):
            await storage.sync_sources("a", line, writes, "m", at=NOW, graph=graph)
    assert await storage.count_corpus("a") == residue
    assert (await storage.list_sources("a", filters=line, limit=10, cursor=None))[0] == [new.source]


async def test_a_sync_with_no_sources_removes_the_set(storage: Storage) -> None:
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
    storage: Storage,
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
    assert await storage.sync_sources("a", line, [new], "m", at=NOW, keep={"kept": "d-kept"}) == (
        Counts(sources=1, chunks=1),
        [],
    )
    listed, _ = await storage.list_sources("a", filters=line, limit=10, cursor=None)
    assert sorted(s.source_id for s in listed) == ["kept", "new"]
    assert await storage.get_source("a", "kept") == kept_source
    assert await storage.source_chunks("a", "kept") == kept_chunks


async def test_chunk_vectors_answer_by_text_within_one_corpus_and_one_model(
    storage: Storage,
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
    storage: Storage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    for corpus in ("a", "b"):
        await memory.ingest(corpus, name="m", text="A informação das válvulas está no manual.")
        await memory.remember(corpus, "As bombas funcionam melhor.")

    async def found(corpus: str, query: str) -> list[str]:
        hits = await memory.search(corpus, query, pillars={"text"})
        return sorted(h.kind for h in hits.hits)

    assert await found("a", "informacao funciona") == []
    assert await storage.set_language("a", "portuguese") == 2
    assert await storage.get_language("a") == "portuguese"
    assert await storage.get_language("b") == "english"
    assert await found("a", "informacao funciona") == ["chunk", "story"]
    assert await found("b", "informacao funciona") == []
    await memory.ingest("a", name="n", text="Novas informações sobre as válvulas.")
    assert await found("a", "valvulas") == ["chunk", "chunk"]
    assert await storage.set_language("a", "portuguese") == 0
    assert (await storage.count_corpus("a")).corpora == 1
    assert (await storage.count_corpus("b")).corpora == 0
    deleted = await storage.delete_corpus("a")
    assert deleted.corpora == 1 and (await storage.count_corpus("a")).clean
    assert await storage.get_language("a") == "english"


@pytest.mark.postgres
@pytest.mark.parametrize("storage", ["postgres"], indirect=True)
async def test_a_write_waits_while_the_corpus_language_changes(storage: Storage) -> None:
    asyncpg = pytest.importorskip("asyncpg")
    memory = Memory(storage)
    holder = await asyncpg.connect(os.environ["RFNRY_TEST_POSTGRES_URL"])
    try:
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "memory.language:a"
            )
            writing = asyncio.create_task(memory.ingest("a", name="m", text="manual"))
            await asyncio.sleep(0.5)
            assert not writing.done()
        await asyncio.wait_for(writing, 10)
    finally:
        await holder.close()
    assert [s.name for s in (await memory.list_sources("a"))[0]] == ["m"]


async def test_a_hit_cites_its_source_by_name_and_metadata_on_every_read(storage: Storage) -> None:
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


async def test_a_deleted_passage_takes_the_names_only_it_carried(storage: Storage) -> None:
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
    [kept] = await storage.entities_named("a", "John Smith")
    assert (kept.name, kept.aliases) == ("John Smith", ("J. Smith",))
    assert await storage.entities_named("a", "Johnny") == []
    deleted = await storage.delete_source("a", "memo")
    assert (deleted.sources, deleted.mentions, deleted.entities) == (1, 2, 1)
    assert await storage.entities_named("a", "Plant 4") == []
    assert await storage.entities_named("a", "John Smith") == []
    [renamed] = await storage.entities_named("a", "J. Smith")
    assert (renamed.entity_id, renamed.name, renamed.key, renamed.aliases) == (
        "e1",
        "J. Smith",
        "j. smith",
        (),
    )
    assert (await storage.delete_source("a", "notes")).entities == 1
    assert (await storage.count_corpus("a")).clean


async def test_an_entity_still_related_or_named_by_a_kept_source_stays(storage: Storage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    line = {"metadata.file": "a"}
    pump = EntityIn(name="P-3", type="pump")
    sources = [
        SourceIn(name="manual", text="P-3 feeds T-9.", metadata={"file": "a"}, entities=[pump]),
        SourceIn(name="memo", text="P-3 was serviced.", metadata={"file": "a"}, entities=[pump]),
    ]
    first = await memory.sync("a", filters=line, sources=sources)
    [before] = await storage.entities_named("a", "P-3")
    second = await memory.sync("a", filters=line, sources=sources[:1])
    assert second.kept == [first.sources[0].source_id] and second.deleted.entities == 0
    assert await storage.entities_named("a", "P-3") == [before]
    await memory.ingest(
        "a",
        name="links",
        text="Bo runs P-3.",
        entities=[EntityIn(name="Bo", type="person")],
        relations=[RelationIn(from_name="Bo", to_name="P-3", type="RUNS")],
    )
    assert (await memory.sync("a", filters=line, sources=[])).deleted.entities == 0
    assert await storage.entities_named("a", "P-3") == [before]


async def test_deleting_every_document_leaves_no_graph_name_behind(storage: Storage) -> None:
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
    assert (await memory.verify_absent("a")).clean


@pytest.mark.postgres
@pytest.mark.parametrize("storage", ["postgres"], indirect=True)
async def test_a_write_racing_the_prune_of_an_entity_it_names_brings_it_back(
    storage: Storage,
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
        async with storage._pool.acquire() as conn:  # type: ignore[attr-defined]
            dangling = await conn.fetchval(
                "SELECT count(*) FROM memory_mentions m WHERE m.corpus_id = $1 AND NOT EXISTS "
                "(SELECT 1 FROM memory_entities e WHERE e.corpus_id = m.corpus_id AND e.entity_id = m.entity_id)",
                corpus,
            )
        assert dangling == 0
        [named] = await storage.entities_named(corpus, "P-3")
        assert named.name == "P-3"
        assert [s.name for s in (await memory.list_sources(corpus))[0]] == ["new"]
    assert "pruned" in outcomes


def _acme(aliases: tuple[str, ...] = ("Acme",)) -> Entity:
    return Entity(
        corpus_id="a", entity_id="e1", name="Acme Inc", key="acme inc", type="org", aliases=aliases
    )


async def _named_in(storage: Storage, source_id: str, text: str, name: str, *new: Entity) -> None:
    await storage.put_source(
        "a",
        *_source("a", source_id, [text]),
        "m",
        graph=GraphWrite(entities=list(new), mentions=[("e1", "chunk", f"{source_id}-0", name)]),
    )


async def _write_after(storage: Storage, source_id: str) -> None:
    original = storage.put_source

    async def racing(*args: Any, **kwargs: Any) -> Any:
        await storage.delete_source("a", source_id)
        return await original(*args, **kwargs)

    storage.put_source = racing  # type: ignore[method-assign]


@pytest.mark.parametrize("dropped", ["alias", "name"])
async def test_a_write_keeps_its_name_on_its_entity_when_a_delete_trims_it_meanwhile(
    storage: Storage, dropped: str
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
        assert [e.entity_id for e in await storage.entities_named("a", name)] == ["e1"], name


async def test_a_name_used_only_through_a_relation_is_evidence(storage: Storage) -> None:
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
    [kept] = await storage.entities_named("a", "Pump three")
    assert (kept.entity_id, kept.name, kept.aliases) == ("e1", "P-3", ("Pump three",))


async def test_an_entity_related_elsewhere_still_loses_a_name_only_deleted_text_used(
    storage: Storage,
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
    assert await storage.entities_named("a", "Johnny") == []
    [kept] = await storage.entities_named("a", "John Smith")
    assert (kept.entity_id, kept.aliases) == ("e1", ())


async def test_two_entities_renamed_to_one_freed_name_at_once_leave_one_holder(
    storage: Storage,
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
    holders = await storage.entities_named("a", "Acme")
    assert sorted(e.entity_id for e in holders) == ["e1", "e2"]
    assert sorted(e.name for e in holders) in (["Acme", "Acme Ltd"], ["Acme", "Acme Corp"])


async def test_a_storage_recreates_only_known_entities_the_write_names(storage: Storage) -> None:
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
    storage: Storage,
) -> None:
    source, chunks, vectors = _source("a", "s1", ["text"], metadata={"line": "A"})
    stray = [c.model_copy(update={"source_name": "stray", "metadata": {"x": 1}}) for c in chunks]
    await storage.put_source("a", source, stray, vectors, "m")
    [chunk] = await storage.source_chunks("a", "s1")
    assert (chunk.source_name, chunk.metadata) == ("s1", {"line": "A"})


async def test_concurrent_syncs_of_one_set_leave_exactly_one(storage: Storage) -> None:
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


async def test_syncs_with_overlapping_filters_serialize(storage: Storage) -> None:
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


@pytest.mark.postgres
@pytest.mark.parametrize("storage", ["postgres"], indirect=True)
async def test_a_sync_takes_its_turn_under_the_corpus_lock(storage: Storage) -> None:
    asyncpg = pytest.importorskip("asyncpg")
    memory = Memory(storage)
    holder = await asyncpg.connect(os.environ["RFNRY_TEST_POSTGRES_URL"])
    try:
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "memory.sync:a"
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


@pytest.mark.postgres
@pytest.mark.parametrize("storage", ["postgres"], indirect=True)
async def test_a_sync_that_fails_inside_its_transaction_leaves_the_old_set(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
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
    storage: Storage,
) -> None:
    vectors = Memory(storage, embedding=HashEmbedding())
    await vectors.initialize()
    await Memory(storage).initialize()
    await vectors.ingest("a", name="manual", text="pump seal yearly")
    hits = await vectors.search("a", "pump seal yearly", pillars={"vector"})
    assert hits.pillars["vector"].error is None and len(hits.hits) == 1


async def test_a_cursor_no_listing_returned_is_refused(storage: Storage) -> None:
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
    storage: Storage,
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
    assert await storage.entities_named("a", "Ana") == []
    assert [e.name for e in await storage.entities_named("a", "P-3")] == ["P-3"]
    assert await storage.get_stories("a", [story.story_id]) == []
    hood = await memory.neighbors("a", "P-3", history=True)
    assert [r.type for r in hood.relations] == ["RUNS"]
    assert [h.id for h in hood.passages] == [kept.story_id]
    assert (await storage.delete_story("a", story.story_id)).clean


async def test_synced_sources_are_searched_like_ingested_ones(storage: Storage) -> None:
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
    storage: Storage,
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
        ("s1", metadata, sha256(source.text.encode("utf-8")).hexdigest(), 2)
    ]
    assert [name for name, *_ in await storage.source_digests("b")] == ["s2"]
    assert await storage.source_digests("c") == []


async def test_a_fingerprint_reads_the_same_on_every_storage(storage: Storage) -> None:
    async def fingerprint(target: Storage) -> CorpusFingerprint:
        memory = Memory(target, embedding=HashEmbedding(), chunk_chars=60, overlap_chars=10)
        await memory.ingest(
            "a",
            name="die change rev B",
            text="Tighten the clamp bolts to 410 N·m in a cross pattern.\n\nRe-torque after 50 strokes.",
            metadata={"line": "north", "torque": 410, "weight": 0.5, "owner": {"b": 1, "a": 2}},
        )
        await memory.ingest(
            "a",
            name="sheet 7",
            sections=[Section(title="Feed", text="V-101 feeds P-3.", locator={"sheet": 7})],
        )
        told = await memory.remember(
            "a",
            "Mei signs the die log.",
            subject="user:mei",
            valid_from=datetime(2026, 5, 12, 9, 30, tzinfo=timezone(timedelta(hours=2))),
            metadata={"team": "north"},
        )
        await memory.supersede(
            "a", told.story.story_id, "Rui signs the die log.", valid_from=NOW + timedelta(days=90)
        )
        return await memory.fingerprint("a")

    reference = InMemoryStorage()
    await reference.initialize(dimension=8)
    assert await fingerprint(storage) == await fingerprint(reference)


async def test_a_fingerprint_follows_content_not_ids_on_every_storage(storage: Storage) -> None:
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
    storage: Storage,
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


async def test_a_fingerprint_counts_stale_vectors_on_every_storage(storage: Storage) -> None:
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


async def test_a_supersede_records_who_closed_the_story_and_when(storage: Storage) -> None:
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


async def test_open_means_not_superseded_and_not_ended(storage: Storage) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    april = datetime(2026, 4, 1, tzinfo=UTC)
    ended = _story("a", "ended", "The promo runs.", valid_from=jan, valid_to=feb)
    assert await storage.put_story("a", ended, None, None) == ended
    hashed = text_hash("The promo runs.")
    assert await storage.find_open_story("a", hashed, at=NOW) is None
    assert await storage.find_open_story("a", hashed, at=jan + timedelta(days=1)) == ended
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
        match=r"^(inmemory|postgres): story 'ended' ended at 2026-02-01T00:00:00\+00:00; remember a new story instead$",
    ):
        await storage.supersede_story("a", ["ended"], later, None, None)


async def test_supersede_closes_several_stories_all_or_nothing(storage: Storage) -> None:
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
        match=r"^(inmemory|postgres): story 'done' is already superseded; supersede the story that replaced it$",
    ):
        await storage.supersede_story("a", ["a", "done"], bad, None, None)
    untouched = await storage.get_stories("a", ["a", "bad"])
    assert [(s.story_id, s.superseded_by, s.valid_to) for s in untouched] == [("a", None, None)]
    for ids in ([], ["a", "a"]):
        with pytest.raises(
            InputError, match=r"^(inmemory|postgres): supersede_story needs distinct story ids$"
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
    storage: Storage, monkeypatch: pytest.MonkeyPatch
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
    storage: Storage, monkeypatch: pytest.MonkeyPatch
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
    storage: Storage, monkeypatch: pytest.MonkeyPatch
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
    storage: Storage,
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
    rep = _story("a", "rep", "the valve rep leaks", valid_from=feb).model_copy(
        update={"supersedes": ("old",)}
    )
    await storage.supersede_story("a", ["old"], rep, vector, "hash-8", graph=graph("rep"))
    names = ["old", "cur", "ended", "future", "rep"]

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
    storage: Storage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    first = await memory.remember("a", "Ana prefers the Tuesday slot.", subject="user:ana")
    written = await memory.remember("a", "Ana prefers the Thursday slot.", subject="user:ana")
    assert first.story.story_id in [h.id for h in written.related.hits]
    assert written.related.pillars
    assert all(report.error is None for report in written.related.pillars.values())


async def _race(*writers: Awaitable[Any]) -> list[Any]:
    return await asyncio.wait_for(asyncio.gather(*writers, return_exceptions=True), timeout=30)


async def test_stress_one_open_story_per_text_with_and_without_ends(storage: Storage) -> None:
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


async def test_stress_a_remember_racing_a_supersede_of_its_text(storage: Storage) -> None:
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


async def test_stress_a_supersede_racing_remembers_of_its_new_text(storage: Storage) -> None:
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


async def test_stress_two_supersedes_closing_overlapping_sets(storage: Storage) -> None:
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


async def test_stress_syncs_under_history_leave_one_current_version(storage: Storage) -> None:
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
        for s in versions:
            if s.retired_at is None:
                continue
            assert s.replaced_by is None or s.replaced_by in ids
            chunks = await storage.source_chunks("a", s.source_id)
            assert len(chunks) == s.chunks
            assert all(c.retired_at == s.retired_at for c in chunks)


async def test_stress_switching_history_while_syncs_run(storage: Storage) -> None:
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
    storage: Storage, monkeypatch: pytest.MonkeyPatch
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


async def test_a_corpus_that_keeps_history_retires_what_sync_replaces(storage: Storage) -> None:
    returns = {"metadata.doc": "returns"}
    jan, feb, mar, apr = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3, 4))
    assert await storage.get_history("a") is False
    first = [
        _version("v1", "Returns policy", "30 days", jan),
        _version("m1", "Manual", "Ground first.", jan),
    ]
    assert await storage.sync_sources("a", returns, first, "m", at=jan) == (Counts(), [])
    await storage.set_history("a", True)
    assert await storage.get_history("a") is True and await storage.get_history("b") is False
    second = [
        _version("v2", "Returns policy", "60 days", feb),
        _version("x1", "Annex", "Annex A.", feb),
    ]
    assert await storage.sync_sources("a", returns, second, "m", at=feb) == (Counts(), ["m1", "v1"])
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
    assert await storage.sync_sources("a", returns, twins, "m", at=mar) == (Counts(), ["v2", "x1"])
    v2 = await storage.get_source("a", "v2")
    assert v2 is not None and v2.replaced_by is None
    same = _version("v5", "Returns policy", "30 days", mar)
    assert (
        await storage.sync_sources("a", returns, [same], "m", at=mar, keep={"v1": "d-v1"}) is None
    )
    assert await storage.sync_sources("a", returns, [], "m", at=apr) == (Counts(), ["v3", "v4"])
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
    storage: Storage,
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
    storage: Storage, monkeypatch: pytest.MonkeyPatch
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
    storage: Storage,
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


async def test_keyword_search_matches_a_chunks_source_name_and_context(storage: Storage) -> None:
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
    storage: Storage, corpus: str, source_id: str, texts: list[str], **metadata: object
) -> None:
    source, chunks, vectors = _source(corpus, source_id, texts, metadata=metadata or None)
    await storage.put_source(corpus, source, chunks, vectors, "m")


async def _ranked(
    storage: Storage,
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


async def test_bm25_ranks_a_rare_term_above_common_ones(storage: Storage) -> None:
    await _texts(
        storage,
        "a",
        "doc",
        ["pump pump pump pump motor", "gasket for the pump", *["pump motor"] * 4],
    )
    ranked = await _ranked(storage, "a", "pump motor gasket")
    assert ranked[0][0] == "doc-1" and len(ranked) == 6
    assert ranked[0][1] == pytest.approx(1.7254765025505487)


async def test_bm25_takes_its_statistics_from_the_filtered_scope(storage: Storage) -> None:
    await _texts(storage, "a", "keep", ["gasket pump", "pump motor"], line="A")
    await _texts(storage, "a", "noise", ["gasket gasket", "gasket seal", "gasket ring"], line="B")
    await _texts(storage, "b", "keep", ["gasket pump", "pump motor"], line="A")
    filtered = await _ranked(storage, "a", "gasket pump", filters={"metadata.line": "A"})
    alone = await _ranked(storage, "b", "gasket pump")
    assert [i for i, _ in filtered] == [i for i, _ in alone]
    assert [s for _, s in filtered] == pytest.approx([s for _, s in alone])


async def test_bm25_leaves_retired_versions_out_of_a_current_read(storage: Storage) -> None:
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


async def test_bm25_prefers_the_shorter_passage_for_the_same_term_count(storage: Storage) -> None:
    await _texts(storage, "a", "doc", ["gasket seal ring flange bolt washer", "gasket seal"])
    assert [i for i, _ in await _ranked(storage, "a", "gasket")] == ["doc-1", "doc-0"]


async def test_bm25_counts_every_occurrence_in_a_passage_length(storage: Storage) -> None:
    await _texts(storage, "a", "doc", ["gasket seal seal seal seal", "gasket ring flange"])
    assert [i for i, _ in await _ranked(storage, "a", "gasket")] == ["doc-1", "doc-0"]


async def test_bm25_lists_a_repeated_kind_once(storage: Storage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.remember("a", "The gasket is backordered.")
    await memory.remember("a", "The gasket is running.")
    window = Window(as_of=datetime.now(UTC) + timedelta(seconds=1))
    once = await _ranked(storage, "a", "gasket", kinds=("story",), window=window)
    twice = await _ranked(storage, "a", "gasket", kinds=("story", "story"), window=window)
    assert len(once) == 2 and twice == once


async def test_bm25_counts_a_repeated_query_word_once(storage: Storage) -> None:
    await _texts(storage, "a", "doc", ["gasket seal", "seal ring"])
    once = await _ranked(storage, "a", "gasket seal")
    twice = await _ranked(storage, "a", "gasket gasket seal")
    assert [i for i, _ in twice] == [i for i, _ in once]
    assert [s for _, s in twice] == pytest.approx([s for _, s in once])


async def test_bm25_ranks_stories_on_their_own_statistics(storage: Storage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.remember("a", "The gasket is backordered.")
    await memory.remember("a", "The pump is running.")
    await _texts(storage, "a", "doc", ["gasket gasket gasket", "gasket seal"])
    window = Window(as_of=datetime.now(UTC) + timedelta(seconds=1))
    both = dict(await _ranked(storage, "a", "gasket pump", kinds=("chunk", "story"), window=window))
    stories = await _ranked(storage, "a", "gasket pump", kinds=("story",), window=window)
    assert len(stories) == 2
    assert [both[i] for i, _ in stories] == pytest.approx([s for _, s in stories])


async def test_bm25_answers_a_query_of_stop_words_with_nothing(storage: Storage) -> None:
    await _texts(storage, "a", "doc", ["the gasket is in the box"])
    assert await _ranked(storage, "a", "the of and") == []


async def _pooled(
    storage: Storage,
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


async def test_bm25_counts_the_corpora_it_is_given_as_one(storage: Storage) -> None:
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


async def test_bm25_counts_the_stories_of_the_corpora_it_is_given_as_one(storage: Storage) -> None:
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
    storage: Storage,
) -> None:
    await _texts(storage, "manual", "manual", MANUAL)
    await _texts(storage, "notes", "notes", NOTES)
    manual = await _pooled(storage, ["manual"], "gasket pump")
    notes = await _pooled(storage, ["notes"], "gasket pump")
    assert notes[0][1] == "notes-4" and notes[0][2] > manual[0][2]
    pooled = await _pooled(storage, ["notes", "manual"], "gasket pump")
    assert pooled[0][:2] == ("manual", "manual-0")


async def test_bm25_reads_no_corpus_it_is_not_given(storage: Storage) -> None:
    await _texts(storage, "a", "s", ["gasket pump", "pump motor"])
    await _texts(storage, "b", "t", ["gasket seal"])
    before = await _pooled(storage, ["a", "b"], "gasket pump")
    await _texts(storage, "c", "u", ["gasket gasket", "gasket pump", "pump"])
    after = await _pooled(storage, ["a", "b"], "gasket pump")
    assert after == before and {c for c, _, _ in after} == {"a", "b"}


async def test_bm25_holds_filters_in_every_corpus_it_is_given(storage: Storage) -> None:
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


async def test_bm25_holds_subject_and_window_in_every_corpus_it_is_given(storage: Storage) -> None:
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
    storage: Storage,
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


async def test_bm25_orders_ties_by_passage_id_then_corpus(storage: Storage) -> None:
    for corpus, source_id in (("B", "s"), ("a", "t"), ("a", "s")):
        source, chunks, vectors = _source(corpus, source_id, ["gasket"])
        named = source.model_copy(update={"name": "manual"})
        renamed = [c.model_copy(update={"source_name": "manual"}) for c in chunks]
        await storage.put_source(corpus, named, renamed, vectors, "m")
    found = await _pooled(storage, ["B", "a"], "gasket")
    assert [(c, i) for c, i, _ in found] == [("B", "s-0"), ("a", "s-0"), ("a", "t-0")]
    assert len({s for _, _, s in found}) == 1


async def test_bm25_answers_stop_words_across_corpora_with_nothing(storage: Storage) -> None:
    await _texts(storage, "a", "doc", ["the gasket is in the box"])
    await _texts(storage, "b", "doc", ["the pump is on the shelf"])
    assert await _pooled(storage, ["a", "b"], "the of and") == []


async def test_bm25_reads_the_query_in_the_language_of_the_corpora_it_is_given(
    storage: Storage,
) -> None:
    for corpus in ("es1", "es2"):
        await storage.set_language(corpus, "spanish")
    await _texts(storage, "es1", "uno", ["la información del equipo"])
    await _texts(storage, "es2", "dos", ["más información aquí"])
    await _texts(storage, "en", "tres", ["la información del equipo"])
    found = await _pooled(storage, ["es1", "es2"], "informacion")
    assert sorted(c for c, _, _ in found) == ["es1", "es2"]
    assert await _pooled(storage, ["en"], "informacion") == []


async def _scoped_corpus(storage: Storage) -> None:
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


async def _inside(storage: Storage, scope: Scope | None) -> list[set[str]]:
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


async def test_searches_and_mentions_stay_inside_a_scope(storage: Storage) -> None:
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


async def test_an_empty_scope_reads_nothing(storage: Storage) -> None:
    await _scoped_corpus(storage)
    for scope in (Scope(sources=()), Scope(passages=()), Scope(sources=("ghost",))):
        assert await _inside(storage, scope) == [set(), set(), set()], scope


async def test_a_sources_scope_holds_the_stories_that_cite_one_of_them(storage: Storage) -> None:
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


async def test_top_k_counts_inside_the_scope(storage: Storage) -> None:
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


async def test_bm25_inside_a_scope_equals_bm25_over_the_scope_alone(storage: Storage) -> None:
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


async def test_mentions_name_the_entity_each_passage_mentions(storage: Storage) -> None:
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
