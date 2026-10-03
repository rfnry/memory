import asyncio
import os
import re
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
    Section,
    Source,
    SourceIn,
    Story,
    Window,
)
from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.storage.storage import GraphWrite, SourceWrite, Storage, text_hash
from rfnry_memory.tools import MemoryTools


async def _call(tools: MemoryTools, name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await tools.dispatch(name, args)
    assert isinstance(result, dict)
    return result


NOW = datetime(2026, 3, 1, tzinfo=UTC)


def ids(legs: list[Leg]) -> list[str]:
    return [i for _, i, _ in legs]


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
            position=i,
            title=None,
            locator={"page": i},
            text=t,
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
        supersedes=None,
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
    legs = await storage.search_text(
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
    only_chunks = await storage.search_text(
        "a", "pump seal", kinds=("chunk",), top_k=10, filters=None, window=Window(), subject=None
    )
    assert [i for _, i, _ in only_chunks] == ["s1-0"]
    faceted = await storage.search_text(
        "a",
        "pump seal",
        kinds=("chunk", "story"),
        top_k=10,
        filters={"metadata.line": "B"},
        window=Window(),
        subject=None,
    )
    assert [i for _, i, _ in faceted] == ["st1"]
    asked = await storage.search_text(
        "a",
        "how often is the seal replaced?",
        kinds=("chunk", "story"),
        top_k=10,
        filters=None,
        window=Window(),
        subject=None,
    )
    assert sorted(i for _, i, _ in asked) == ["s1-0", "st1"]
    covered = await storage.search_text(
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
            await storage.search_text(
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
        await storage.search_text(
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
        legs = await storage.search_text(
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
        legs = await storage.search_text(
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
            await storage.search_text(
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
        await storage.search_text(
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
        await storage.search_text(
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
    assert (await storage.find_open_story("a", text_hash(old.text))) == old
    new = _story("a", "new", "The date is Mar 3.", valid_from=feb).model_copy(
        update={"supersedes": "old"}
    )
    closed, created = await storage.supersede_story("a", "old", feb, new, None, None)
    assert closed.valid_to == feb and created.supersedes == "old"
    assert await storage.find_open_story("a", text_hash(old.text)) is None
    assert await storage.find_open_story("a", text_hash(new.text)) == created
    with pytest.raises(NotFoundError):
        await storage.supersede_story(
            "a", "ghost", feb, new.model_copy(update={"story_id": "x"}), None, None
        )
    with pytest.raises(
        InputError,
        match=r"^(inmemory|postgres): story 'old' is already superseded; supersede the story that replaced it$",
    ):
        await storage.supersede_story(
            "a",
            "old",
            feb,
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
        mentions=[("e1", "chunk", "s1-0"), ("e2", "chunk", "s1-0")],
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
    hop = await storage.neighbors("a", ["e1"], relation_types=None)
    assert [(r.from_id, r.to_id) for r in hop] == [("e1", "e2")]
    assert [(r.type) for r in await storage.neighbors("a", ["e2"], relation_types=["FEEDS"])] == [
        "FEEDS"
    ]
    assert await storage.mentions(
        "a", ["e1", "e3"], kinds=("chunk", "story"), filters=None, window=Window(), subject=None
    ) == [("chunk", "s1-0")]
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
    assert await storage.neighbors("a", ["e1"], relation_types=None) == []


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
            mentions=[("e1", "story", "st")],
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
            graph=GraphWrite(entities=[entity], relations=[feeds], mentions=[(tie, "story", tie)]),
        )
        source, chunks, vectors = _source("a", tie, ["pump seal"])
        await storage.put_source(
            "a",
            source,
            chunks,
            vectors,
            "m",
            graph=GraphWrite(mentions=[("ab", "chunk", f"{tie}-0")]),
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
    assert refs == [("chunk", c) for c in chunk_ids] + [("story", tie) for tie in sorted(tied)]
    relations = await storage.neighbors("a", ["ab"], relation_types=None)
    assert [r.from_id for r in relations] == sorted(tied)
    both: tuple[Kind, ...] = ("chunk", "story")
    assert (
        ids(
            await storage.search_text(
                "a", "pump seal", kinds=both, top_k=10, filters=None, window=Window(), subject=None
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
    assert {e.name for e in hood.entities} == {"P-3", "Ana", "Bo"}
    assert {h.id for h in hood.passages} == {current.story_id}
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
        mentions=[("e1", kind, passage_id)],
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
        mentions=[("e1", "story", "other")],
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
        update={"supersedes": "old"}
    )
    with pytest.raises(StorageError):
        await storage.supersede_story(
            "a", "old", feb, new, _vec("t-9"), "m", graph=_bad_graph("a", "story", "new")
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
            mentions=[("e1", "chunk", "s1-0"), ("e2", "chunk", "s1-0"), ("e2", "chunk", "s1-1")],
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
            mentions=[("e3", "story", "old"), ("e1", "story", "old")],
        ),
    )
    new = _story("a", "new", "Bo runs P-3", valid_from=feb).model_copy(update={"supersedes": "old"})
    closed, created = await storage.supersede_story(
        "a",
        "old",
        feb,
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
            mentions=[("e4", "story", "new"), ("e1", "story", "new")],
        ),
    )
    assert closed.valid_to == feb and created == new
    assert await storage.mentions(
        "a", ["e1"], kinds=("chunk", "story"), filters=None, window=Window(as_of=feb), subject=None
    ) == [("chunk", "s1-0"), ("story", "new")]
    assert [
        (r.from_id, r.type) for r in await storage.neighbors("a", ["e4"], relation_types=None)
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
        graph=GraphWrite(entities=[first], mentions=[("first", "story", "st1")]),
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
            mentions=[("late", "story", "st2"), ("tank", "story", "st2")],
        ),
    )
    assert (await storage.count_corpus("a")).entities == 2
    assert await storage.entities_by_id("a", ["late"]) == []
    assert (await storage.entities_named("a", "pump three")) == [
        first.model_copy(update={"aliases": ("Pump three",)})
    ]
    assert await storage.mentions(
        "a", ["first"], kinds=("story",), filters=None, window=Window(), subject=None
    ) == [("story", "st1"), ("story", "st2")]
    assert [
        (r.from_id, r.to_id) for r in await storage.neighbors("a", ["first"], relation_types=None)
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
                    entities=[racer, tank], mentions=[(racer.entity_id, "story", f"race{i}")]
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
    ) == (1, 2, 2, 2, 4)
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
                entities=[go_live] if i == 0 else [], mentions=[("go", "story", story.story_id)]
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
        graph=GraphWrite(mentions=[("go", "story", "zz-apollo")]),
    )
    source, chunks, vectors = _source("a", "s1", ["go-live"])
    await storage.put_source(
        "a", source, chunks, vectors, "hash-8", graph=GraphWrite(mentions=[("go", "chunk", "s1-0")])
    )
    both: tuple[Kind, ...] = ("chunk", "story")
    assert ids(
        await storage.search_text(
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
    ) == [("story", "zz-apollo")]
    assert "zz-apollo" not in ids(
        await storage.search_text(
            "a", "go-live", kinds=both, top_k=5, filters=None, window=Window(), subject=None
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
    async def embed(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(0)
        return await super().embed(texts)


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
                mentions=[("e1", "story", "second")],
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
        update={"supersedes": "old"}
    )
    held = r"^(inmemory|postgres): story 'holder', true since 2026-01-01T00:00:00\+00:00, already says that; supersede or delete it instead$"
    with pytest.raises(InputError, match=held):
        await storage.supersede_story("a", "old", feb, new, None, None)
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
            graph=GraphWrite(entities=[pump], mentions=[("e1", "chunk", f"{source_id}-0")]),
        )
    await storage.put_source(
        "a", *_source("a", "keep", ["pump elsewhere"], metadata={"file": "b"}), "m"
    )
    await storage.put_source("b", *_source("b", "old1", ["pump in b"], metadata={"file": "a"}), "m")
    tank = Entity(corpus_id="a", entity_id="e2", name="T-9", key="t-9", type="tank")
    new = SourceWrite(
        *_source("a", "new1", ["tank T-9 fills", "pump P-3 drains T-9"], metadata={"file": "a"})
    )
    deleted = await storage.sync_sources(
        "a",
        line,
        [new],
        "m",
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
                ("e2", "chunk", "new1-0"),
                ("e1", "chunk", "new1-1"),
                ("e2", "chunk", "new1-1"),
            ],
        ),
    )
    assert deleted == Counts(sources=2, chunks=4, mentions=2)
    listed, _ = await storage.list_sources("a", filters=line, limit=10, cursor=None)
    assert listed == [new.source]
    assert await storage.source_chunks("a", "new1") == new.chunks
    assert await storage.source_chunks("a", "old1") == []
    assert await storage.get_source("a", "keep") is not None
    assert await storage.get_source("b", "old1") is not None
    window = Window()
    assert (
        await storage.search_text(
            "a", "wiring", kinds=("chunk",), top_k=10, filters=None, window=window, subject=None
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

    stray = GraphWrite(entities=[tank], mentions=[("e2", "chunk", "new1-0")])
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
            await storage.sync_sources("a", line, writes, "m", graph=graph)
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


async def test_deleting_a_story_erases_its_mentions_and_relations(storage: Storage) -> None:
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
        stories=1, mentions=2, relations=1
    )
    residue = await storage.count_corpus("a")
    assert (residue.stories, residue.mentions, residue.relations, residue.entities) == (1, 2, 1, 3)
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
    text = await storage.search_text(
        "a", "pump seal", kinds=("chunk",), top_k=10, filters=None, window=window, subject=None
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
        ("s1", metadata, sha256(source.text.encode("utf-8")).hexdigest())
    ]
    assert [name for name, _, _ in await storage.source_digests("b")] == ["s2"]
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
