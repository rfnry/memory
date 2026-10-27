import re
from datetime import UTC, datetime
from typing import Any

import pytest
from unit.fakes import HashEmbedding

from rfnry_memory.errors import InputError
from rfnry_memory.index import HNSW
from rfnry_memory.memory import Memory
from rfnry_memory.models import Chunk, Entity, Relation, Source, SourceIn, Story, Window
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.postgres.vectors import _vectors_lock
from rfnry_memory.ranking import BM25, CoverDensity
from rfnry_memory.rows import GraphWrite
from rfnry_memory.tools import MemoryTools

JAN, FEB, MAR, APR, MAY = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3, 4, 5))
DEC = datetime(2026, 12, 1, tzinfo=UTC)


async def _memory(storage: PostgresStorage, clock: list[datetime], **kw: Any) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding(), clock=lambda: clock[0], **kw)
    await memory.check()
    return memory


def _doc(name: str, text: str, doc: str, valid_to: datetime | None = None) -> SourceIn:
    return SourceIn(name=name, text=text, metadata={"doc": doc}, valid_to=valid_to)


async def test_keyword_search_filters_by_dates_under_both_rankings_and_two_corpora(
    storage: PostgresStorage,
) -> None:
    clock = [JAN]
    memory = await _memory(storage, clock)
    old_a = await memory.ingest("a", name="old a", text="pump seal replacement yearly")
    old_b = await memory.ingest("b", name="old b", text="pump seal replacement monthly")
    clock[0] = MAR
    new_a = await memory.ingest(
        "a", name="new a", text="pump seal replacement weekly", valid_to=DEC
    )
    new_b = await memory.ingest("b", name="new b", text="pump seal replacement daily", valid_to=DEC)
    for text in (BM25(), CoverDensity()):
        for match in ("any", "all", "phrase", "web"):
            hits = await memory.search(
                ["a", "b"],
                "pump seal replacement",
                kinds=("chunk",),
                pillars={"text"},
                text=text,
                match=match,
                filters={"created_at": {"lt": FEB}},
            )
            assert hits.pillars["text"].error is None, (text, match)
            assert {h.chunk.source_id for h in hits.hits if h.chunk} == {
                old_a.source.source_id,
                old_b.source.source_id,
            }, (text, match)
            hits = await memory.search(
                ["a", "b"],
                "pump seal replacement",
                kinds=("chunk",),
                pillars={"text"},
                text=text,
                match=match,
                filters={"valid_to": {"lte": DEC}, "confirmed_at": {"gte": "2026-02-01"}},
            )
            assert {h.chunk.source_id for h in hits.hits if h.chunk} == {
                new_a.source.source_id,
                new_b.source.source_id,
            }, (text, match)


async def test_vector_search_and_expand_filter_by_dates_through_an_index(
    storage: PostgresStorage,
) -> None:
    clock = [JAN]
    memory = await _memory(storage, clock)
    old = await memory.ingest("a", name="old", text="pump seal replacement yearly")
    await memory.remember("a", "The seal was replaced.")
    clock[0] = MAR
    new = await memory.ingest("a", name="new", text="pump seal replacement weekly", valid_to=DEC)
    await memory.remember("a", "The seal will be replaced again.", valid_to=DEC)
    assert await memory.set_vector_index("a", HNSW()) == 4

    async def ids(filters: dict[str, Any]) -> set[str]:
        hits = await memory.search(
            "a", "seal replacement", pillars={"vector"}, top_k=10, filters=filters
        )
        assert hits.pillars["vector"].error is None
        return {h.chunk.source_id if h.chunk else h.text for h in hits.hits}

    assert await ids({"created_at": {"lt": FEB}}) == {
        old.source.source_id,
        "The seal was replaced.",
    }
    assert await ids({"confirmed_at": {"gte": MAR}}) == {
        new.source.source_id,
        "The seal will be replaced again.",
    }
    assert await ids({"valid_to": {"lt": "2027-01-01"}}) == {
        new.source.source_id,
        "The seal will be replaced again.",
    }
    assert await ids({"valid_from": {"gte": FEB}}) == {"The seal will be replaced again."}
    around = await memory.expand(
        "a",
        ("chunk", f"{old.source.source_id}-0"),
        by="vector",
        filters={"created_at": {"gte": MAR}},
    )
    assert {h.chunk.source_id if h.chunk else h.text for h in around.hits} == {
        new.source.source_id,
        "The seal will be replaced again.",
    }


async def test_graph_mentions_filter_chunks_and_stories_by_dates(storage: PostgresStorage) -> None:
    pump = Entity(corpus_id="a", entity_id="e1", name="P-3", key="p-3", type="pump")
    tank = Entity(corpus_id="a", entity_id="e2", name="T-9", key="t-9", type="tank")
    for sid, at in (("s1", JAN), ("s2", MAR)):
        source = Source(
            corpus_id="a",
            source_id=sid,
            name=sid,
            metadata={},
            text="pump P-3 feeds tank T-9",
            created_at=at,
            confirmed_at=at,
            chunks=1,
        )
        chunk = Chunk(
            corpus_id="a",
            chunk_id=f"{sid}-0",
            source_id=sid,
            source_name=sid,
            position=0,
            title=None,
            locator={},
            text="pump P-3 feeds tank T-9",
            metadata={},
            created_at=at,
            confirmed_at=at,
        )
        graph = GraphWrite(
            entities=[pump, tank],
            relations=[
                Relation(
                    corpus_id="a",
                    from_id="e1",
                    to_id="e2",
                    type="FEEDS",
                    evidence_kind="chunk",
                    evidence_id=f"{sid}-0",
                )
            ],
            mentions=[("e1", "chunk", f"{sid}-0", "P-3")],
        )
        await storage.put_source("a", source, [chunk], None, None, graph=graph)
    for tid, at in (("t1", JAN), ("t2", MAR)):
        story = Story(
            corpus_id="a",
            story_id=tid,
            text=f"story {tid} about P-3",
            subject=None,
            valid_from=at,
            valid_to=DEC if tid == "t2" else None,
            created_at=at,
            source_ids=(),
            metadata={},
        )
        await storage.put_story(
            "a",
            story,
            None,
            None,
            graph=GraphWrite(entities=[pump], mentions=[("e1", "story", tid, "P-3")]),
        )
    window = Window(as_of=MAY)

    async def found(filters: dict[str, Any]) -> set[tuple[str, str]]:
        rows = await storage.mentions(
            "a", ["e1"], kinds=("chunk", "story"), filters=filters, window=window, subject=None
        )
        return {ref for _, ref in rows}

    assert await found({"created_at": {"lt": FEB}}) == {("chunk", "s1-0"), ("story", "t1")}
    assert await found({"confirmed_at": {"gte": MAR}}) == {("chunk", "s2-0"), ("story", "t2")}
    assert await found({"valid_to": {"lt": "2027-01-01"}}) == {("story", "t2")}
    assert await found({"valid_from": {"gte": FEB}}) == {("story", "t2")}
    memory = Memory(storage, pillars={"text", "graph"})
    hits = await memory.search(
        "a", "P-3", around=["P-3"], hops=0, filters={"created_at": {"lt": FEB}}
    )
    assert {(h.kind, h.id) for h in hits.hits} == {("chunk", "s1-0"), ("story", "t1")}
    around = await memory.expand(
        "a", ("chunk", "s1-0"), by="graph", depth=1, filters={"created_at": {"gte": MAR}}
    )
    assert {(h.kind, h.id) for h in around.hits} == {("chunk", "s2-0"), ("story", "t2")}


async def test_stories_under_known_at_compare_the_valid_to_they_show(
    storage: PostgresStorage,
) -> None:
    clock = [JAN]
    memory = await _memory(storage, clock)
    first = (await memory.remember("acme", "Ana owns the review.")).story
    ending = (await memory.remember("acme", "Bo owns the audit.", valid_to=DEC)).story
    clock[0] = MAR
    await memory.supersede("acme", first.story_id, "Cy owns the review.")
    await memory.end_story("acme", ending.story_id, valid_to=MAR)
    for known_at in (None, FEB, APR):
        every, _ = await memory.list_stories("acme", history=True, known_at=known_at)
        shown = {s.story_id: s.valid_to for s in every}
        for bounds in ({"lt": DEC}, {"gte": DEC}, {"lt": APR}, {"gte": APR, "lte": DEC}):
            expected = {
                sid
                for sid, end in shown.items()
                if end is not None
                and all(
                    {"lt": end < b, "lte": end <= b, "gt": end > b, "gte": end >= b}[op]
                    for op, b in bounds.items()
                )
            }
            listed, _ = await memory.list_stories(
                "acme", history=True, known_at=known_at, filters={"valid_to": bounds}
            )
            assert {s.story_id for s in listed} == expected, (known_at, bounds)
            recalled = await memory.recall(
                "acme", "owns", history=True, known_at=known_at, filters={"valid_to": bounds}
            )
            assert {h.id for h in recalled.hits} == expected, (known_at, bounds)
    then, _ = await memory.list_stories("acme", known_at=FEB, filters={"valid_to": {"lt": APR}})
    assert then == []
    then, _ = await memory.list_stories("acme", known_at=FEB, filters={"valid_to": {"gte": DEC}})
    assert [s.story_id for s in then] == [ending.story_id] and then[0].valid_to == DEC


async def test_chunks_and_sources_under_known_at_compare_the_valid_to_they_show(
    storage: PostgresStorage,
) -> None:
    clock = [JAN]
    memory = await _memory(storage, clock)
    await memory.set_history("acme", True)
    doc = {"metadata.doc": "a"}
    [old] = (
        await memory.sync("acme", filters=doc, sources=[_doc("Returns", "30 days.", "a")])
    ).sources
    clock[0] = MAR
    [new] = (
        await memory.sync("acme", filters=doc, sources=[_doc("Returns", "60 days.", "a", DEC)])
    ).sources
    retired = await memory.list_sources("acme", history=True)
    shown = {s.source_id: s.valid_to for (s) in retired[0]}
    assert shown == {old.source_id: MAR, new.source_id: DEC}
    for known_at, expected in (
        (None, {old.source_id, new.source_id}),
        (FEB, set()),
        (APR, {old.source_id, new.source_id}),
    ):
        listed, _ = await memory.list_sources(
            "acme", history=True, known_at=known_at, filters={"valid_to": {"lte": DEC}}
        )
        assert {s.source_id for s in listed} == expected, known_at
        for s in listed:
            assert s.valid_to is not None and s.valid_to <= DEC
        hits = await memory.search(
            "acme",
            "days",
            kinds=("chunk",),
            history=True,
            known_at=known_at,
            filters={"valid_to": {"lte": DEC}},
        )
        assert {h.chunk.source_id for h in hits.hits if h.chunk} == expected, known_at
        for h in hits.hits:
            assert h.chunk is not None and h.chunk.valid_to is not None
        alike = await memory.resembling(
            "acme",
            text="30 days.",
            history=True,
            known_at=known_at,
            filters={"valid_to": {"lte": DEC}},
        )
        assert {m.source.source_id for m in alike.matches} <= expected, known_at
    then, _ = await memory.list_sources("acme", known_at=FEB)
    assert [s.source_id for s in then] == [old.source_id] and then[0].valid_to is None
    with pytest.raises(InputError, match=r"^memory: confirmed_at keeps no history"):
        await memory.search("acme", "days", known_at=FEB, filters={"confirmed_at": {"lt": DEC}})
    with pytest.raises(InputError, match=r"^memory: confirmed_at keeps no history"):
        await memory.recall("acme", "days", known_at=FEB, filters={"confirmed_at": {"lt": DEC}})
    with pytest.raises(InputError, match=r"^memory: confirmed_at keeps no history"):
        await memory.resembling(
            "acme", text="days", known_at=FEB, filters={"confirmed_at": {"lt": DEC}}
        )
    with pytest.raises(InputError, match=r"^memory: confirmed_at keeps no history"):
        await memory.list_stories("acme", known_at=FEB, filters={"confirmed_at": {"lt": DEC}})
    with pytest.raises(InputError, match=r"^memory: confirmed_at keeps no history"):
        await memory.expand(
            "acme",
            ("chunk", f"{old.source_id}-0"),
            by="document",
            known_at=FEB,
            filters={"confirmed_at": {"lt": DEC}},
        )
    around = await memory.expand(
        "acme",
        ("chunk", f"{old.source_id}-0"),
        by="document",
        history=True,
        filters={"created_at": {"lt": FEB}},
    )
    assert around.hits == []
    told = await memory.remember(
        "acme", "Returns take 30 days.", filters={"confirmed_at": {"lt": DEC}}, related=5
    )
    assert told.related.hits == []


async def test_what_a_date_filter_accepts_and_refuses_on_every_path(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage, [JAN])
    accepted: list[Any] = [
        {"lt": "2026-02-01T00:00:00Z"},
        {"lt": "2026-02-01T00:00:00+02:00"},
        {"lt": "2026-02-01"},
        {"lt": datetime(2026, 2, 1)},
        {"lt": FEB},
        {"gt": "2026-01-01", "lte": FEB},
        {"lt": "2026-W05"},
    ]
    for condition in accepted:
        await memory.list_sources("acme", filters={"valid_to": condition})
    ranged = "filters: filter 'valid_to' is a range of gt, gte, lt and lte over datetimes or ISO 8601 strings"
    refused: list[Any] = [
        {"lt": True},
        {"lt": 1},
        {"lt": 1.5},
        {"lt": None},
        {"eq": FEB},
        {"lt": "2025"},
        {"lt": FEB.date()},
        [FEB],
        FEB,
        "2026-02-01",
        {},
        [],
        {"lt": FEB, "gt": "x"},
    ]
    for condition in refused:
        with pytest.raises(InputError, match=f"^{re.escape(ranged)}$"):
            await memory.list_sources("acme", filters={"valid_to": condition})
    with pytest.raises(InputError, match=r"^filters: filter key 'created' must address a facet"):
        await memory.list_sources("acme", filters={"created": {"lt": FEB}})
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    with pytest.raises(InputError, match=r"^memory: confirmed_at keeps no history"):
        await tools.dispatch(
            "list_sources",
            {"filters": {"confirmed_at": {"lt": "2026-02-01"}}, "known_at": "2026-02-01T00:00:00Z"},
        )
    with pytest.raises(InputError, match=r"^filters: filter 'created_at' is a range"):
        await tools.dispatch("search", {"query": "x", "filters": {"created_at": {"lt": True}}})
    for name in ("search", "recall", "expand", "list_sources", "list_stories", "resembling"):
        [tool] = [d for d in tools.definitions() if d["name"] == name]
        assert "confirmed_at" in tool["input_schema"]["properties"]["filters"]["description"], name


async def test_the_vector_lock_releases_the_key_it_took(storage: PostgresStorage) -> None:
    async with storage._pool.acquire() as conn:
        async with _vectors_lock(conn, "a"):
            held = await conn.fetchval(
                "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
            )
            assert held == 1
        held = await conn.fetchval(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
        )
        assert held == 0
