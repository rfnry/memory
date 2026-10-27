from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn
from rfnry_memory.postgres import PostgresStorage, keys
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding

RUN_7 = {"run": 7, "agent": "planner"}
RUN_8 = {"run": 8, "agent": "reviewer"}
RUN_9 = {"run": 9, "agent": {"name": "auditor"}}


async def _memory(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    return memory


async def test_a_story_records_who_wrote_it_and_who_closed_it(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    told = (await memory.remember("acme", "Ana owns the review.", written_by=RUN_7)).story
    assert (told.written_by, told.closed_by) == (RUN_7, None)
    again = (await memory.remember("acme", "Ana owns the review.", written_by=RUN_8)).story
    assert again.story_id == told.story_id and again.confirmed_by == RUN_8
    later = (
        await memory.supersede("acme", told.story_id, "Bo owns the review.", written_by=RUN_8)
    ).story
    [closed] = await memory.get_stories("acme", [told.story_id])
    assert (closed.written_by, closed.closed_by, later.written_by) == (RUN_7, RUN_8, RUN_8)
    before = closed.closed_at - timedelta(microseconds=1) if closed.closed_at else None
    [believed] = (await memory.list_stories("acme", known_at=before, history=True))[0]
    assert (believed.closed_by, believed.closed_at) == (None, None)
    ended = await memory.end_story("acme", later.story_id, written_by=RUN_7)
    assert ended.closed_by == RUN_7
    plain = (await memory.remember("acme", "Cy owns the audit.")).story
    retracted = await memory.end_story("acme", plain.story_id, valid_to=plain.valid_from)
    assert (plain.written_by, retracted.closed_by) == ({}, {})


async def test_a_document_records_who_wrote_it_and_who_retired_it(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    await memory.set_history("acme", True)
    doc = {"metadata.doc": "returns"}
    policy = SourceIn(name="Returns policy", text="30 days.", metadata={"doc": "returns"})
    [first] = (await memory.sync("acme", filters=doc, sources=[policy], written_by=RUN_7)).sources
    assert first.written_by == RUN_7
    [kept] = (await memory.sync("acme", filters=doc, sources=[policy], written_by=RUN_8)).sources
    assert kept.source_id == first.source_id and kept.written_by == RUN_7
    [hit] = [h for h in (await memory.search("acme", "30 days")).hits if h.chunk is not None]
    assert hit.chunk is not None and hit.chunk.written_by == RUN_7
    changed = policy.model_copy(update={"text": "60 days."})
    [second] = (await memory.sync("acme", filters=doc, sources=[changed], written_by=RUN_8)).sources
    retired = await memory.get_source("acme", first.source_id)
    assert retired is not None and (retired.closed_by, second.written_by) == (RUN_8, RUN_8)
    await memory.sync("acme", filters=doc, sources=[], written_by=RUN_7)
    removed = await memory.get_source("acme", second.source_id)
    assert removed is not None and removed.closed_by == RUN_7
    note = (await memory.ingest("acme", name="note", text="Ground first.", written_by=RUN_7)).source
    newer = await memory.ingest(
        "acme", name="note", text="Ground first, then test.", replaces=note.source_id
    )
    replaced = await memory.get_source("acme", note.source_id)
    assert replaced is not None and (replaced.closed_by, newer.source.written_by) == ({}, {})


async def test_reads_select_by_who_wrote_and_a_sync_names_its_set_by_metadata(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.remember("acme", "Ana owns the review.", written_by=RUN_7)
    await memory.remember("acme", "Bo owns the audit.", written_by=RUN_8)
    await memory.ingest(
        "acme", name="seal notes", text="Replace the seal yearly.", written_by=RUN_8
    )
    by_eight = {"written_by.run": 8}
    stories, _ = await memory.list_stories("acme", filters=by_eight)
    assert [s.text for s in stories] == ["Bo owns the audit."]
    sources, _ = await memory.list_sources("acme", filters={"written_by.agent": "reviewer"})
    assert [s.name for s in sources] == ["seal notes"]
    hits = await memory.search("acme", "owns", filters={"written_by.run": {"lt": 8}})
    assert [h.text for h in hits.hits] == ["Ana owns the review."]
    with pytest.raises(
        InputError, match=r"^memory: a sync names its set by metadata; 'written_by.run' is not"
    ):
        await memory.sync("acme", filters={"written_by.run": 8}, sources=[])


async def test_the_tools_write_as_the_host_declares_and_show_who_wrote(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    tools = MemoryTools(memory, corpus_id="acme", written_by=RUN_7)
    remember = next(d for d in tools.definitions() if d["name"] == "remember")
    assert "written_by" not in remember["input_schema"]["properties"]
    told = await tools.dispatch("remember", {"text": "Ana owns the review."})
    assert isinstance(told, dict) and told["story"]["written_by"] == RUN_7
    ended = await tools.dispatch("end_story", {"story_id": told["story"]["story_id"]})
    assert isinstance(ended, dict) and ended["story"]["closed_by"] == RUN_7
    filed = await tools.dispatch("ingest", {"name": "seal notes", "text": "Replace the seal."})
    assert isinstance(filed, dict) and filed["source"]["written_by"] == RUN_7
    found = await tools.dispatch("search", {"query": "seal", "kinds": ["chunk"]})
    assert isinstance(found, dict) and found["hits"][0]["written_by"] == RUN_7
    near = await tools.dispatch("remember", {"text": "Ana owns the review again."})
    assert isinstance(near, dict) and all("written_by" in r for r in near["related"])
    with pytest.raises(
        ConfigurationError, match=r"^tools: a view is bound by metadata; 'written_by.run' is not"
    ):
        MemoryTools(memory, corpus_id="acme", filters={"written_by.run": 7})


async def test_a_merge_a_correction_and_a_withdrawal_record_who_closed(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    a = (await memory.remember("acme", "Ana owns the review.", written_by=RUN_7)).story
    b = (await memory.remember("acme", "Bo owns the audit.", written_by=RUN_8)).story
    both = await memory.supersede(
        "acme", [a.story_id, b.story_id], "Cy owns both.", written_by=RUN_9
    )
    closed = await memory.get_stories("acme", [a.story_id, b.story_id])
    assert [(s.written_by, s.closed_by) for s in closed] == [(RUN_7, RUN_9), (RUN_8, RUN_9)]
    assert (both.story.written_by, both.story.closed_by) == (RUN_9, None)
    old = (await memory.remember("acme", "Di owns QA.", valid_from=jan, written_by=RUN_7)).story
    fixed = await memory.supersede(
        "acme", old.story_id, "Di owns QA and docs.", valid_from=jan, written_by=RUN_8
    )
    [corrected] = await memory.get_stories("acme", [old.story_id])
    assert corrected.valid_to == corrected.valid_from == jan
    assert (corrected.closed_by, fixed.story.written_by) == (RUN_8, RUN_8)
    future = datetime.now(UTC) + timedelta(days=30)
    planned = await memory.remember(
        "acme", "Ed starts in a month.", valid_from=future, written_by=RUN_7
    )
    withdrawn = await memory.end_story(
        "acme", planned.story.story_id, valid_to=future, written_by=RUN_8
    )
    assert withdrawn.valid_to == withdrawn.valid_from == future
    assert withdrawn.closed_by == RUN_8 and withdrawn.closed_at is not None
    before = withdrawn.closed_at - timedelta(microseconds=1)
    [believed] = (await memory.list_stories("acme", known_at=before, history=True, since=future))[0]
    assert (believed.closed_by, believed.closed_at, believed.valid_to) == (None, None, None)


async def test_a_document_known_before_its_retirement_shows_no_closer_and_its_passages_their_writer(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.set_history("acme", True)
    doc = {"metadata.doc": "returns"}
    policy = SourceIn(name="Returns policy", text="30 days.", metadata={"doc": "returns"})
    [first] = (await memory.sync("acme", filters=doc, sources=[policy], written_by=RUN_7)).sources
    changed = policy.model_copy(update={"text": "60 days."})
    await memory.sync("acme", filters=doc, sources=[changed], written_by=RUN_9)
    retired = await memory.get_source("acme", first.source_id)
    assert retired is not None and retired.retired_at is not None and retired.closed_by == RUN_9
    before = retired.retired_at - timedelta(microseconds=1)
    rows, _ = await memory.list_sources("acme", known_at=before, history=True)
    [believed] = [s for s in rows if s.source_id == first.source_id]
    assert (believed.written_by, believed.closed_by, believed.retired_at) == (RUN_7, None, None)
    then = await memory.search("acme", "days", known_at=before, kinds=("chunk",))
    assert [(h.chunk.written_by, h.chunk.retired_at) for h in then.hits if h.chunk] == [
        (RUN_7, None)
    ]
    for filters, text in (
        ({"written_by.run": 7}, "30 days."),
        ({"written_by.agent.name": "auditor"}, "60 days."),
    ):
        found = await memory.search("acme", "days", history=True, filters=filters, kinds=("chunk",))
        assert [h.chunk.text for h in found.hits if h.chunk] == [text]


async def test_a_sync_prepared_again_writes_with_the_written_by_of_the_call(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _memory(storage)
    await memory.set_history("acme", True)
    doc = {"metadata.doc": "returns"}
    policy = SourceIn(name="Returns policy", text="30 days.", metadata={"doc": "returns"})
    [first] = (await memory.sync("acme", filters=doc, sources=[policy], written_by=RUN_7)).sources
    sync_sources, calls = storage.sync_sources, []

    async def moved(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["written_by"])
        return None if len(calls) == 1 else await sync_sources(*args, **kwargs)

    monkeypatch.setattr(storage, "sync_sources", moved)
    changed = policy.model_copy(update={"text": "60 days."})
    [second] = (await memory.sync("acme", filters=doc, sources=[changed], written_by=RUN_8)).sources
    assert calls == [RUN_8, RUN_8] and second.written_by == RUN_8
    retired = await memory.get_source("acme", first.source_id)
    assert retired is not None and retired.closed_by == RUN_8
    assert [c.written_by for c in await storage.source_chunks("acme", second.source_id)] == [RUN_8]


async def test_an_erasure_records_no_closer(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    note = (await memory.ingest("acme", name="note", text="Ground first.", written_by=RUN_7)).source
    newer = await memory.ingest(
        "acme", name="note", text="Ground, then test.", replaces=note.source_id, written_by=RUN_8
    )
    assert await memory.get_source("acme", note.source_id) is None
    assert (newer.source.written_by, newer.retired, newer.deleted.sources) == (RUN_8, [], 1)
    await memory.set_history("acme", True)
    third = await memory.ingest(
        "acme",
        name="note",
        text="Ground, test, log.",
        replaces=newer.source.source_id,
        written_by=RUN_9,
    )
    replaced = await memory.get_source("acme", newer.source.source_id)
    assert replaced is not None and replaced.closed_by == RUN_9
    assert replaced.replaced_by == third.source.source_id


async def test_a_repeat_ended_on_write_keeps_its_first_teller(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    first = (
        await memory.remember(
            "acme", "Ana ran the drill.", valid_from=jan, valid_to=feb, written_by=RUN_7
        )
    ).story
    again = (
        await memory.remember(
            "acme", "Ana ran the drill.", valid_from=jan, valid_to=feb, written_by=RUN_8
        )
    ).story
    assert again.story_id == first.story_id and again.written_by == RUN_7
    tagged = await memory.remember(
        "acme",
        "Ana ran the drill.",
        valid_from=jan,
        valid_to=feb,
        metadata={"k": 1},
        written_by=RUN_8,
    )
    assert tagged.story.story_id != first.story_id and tagged.story.written_by == RUN_8


async def test_the_tools_close_as_the_host_declares_and_the_model_cannot_say_otherwise(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    tools = MemoryTools(memory, corpus_id="acme", written_by=RUN_7)
    for definition in tools.definitions():
        assert "written_by" not in definition["input_schema"]["properties"], definition["name"]
    told = await tools.dispatch("remember", {"text": "Ana owns it.", "written_by": RUN_8})
    assert isinstance(told, dict) and told["story"]["written_by"] == RUN_7
    moved = await tools.dispatch(
        "supersede", {"story_id": told["story"]["story_id"], "text": "Bo owns it."}
    )
    assert isinstance(moved, dict) and moved["story"]["written_by"] == RUN_7
    found = await tools.dispatch("search", {"query": "owns", "kinds": ["story"], "history": True})
    assert isinstance(found, dict)
    rows = {r["id"]: r["story"] for r in found["hits"]}
    assert rows[told["story"]["story_id"]]["closed_by"] == RUN_7
    assert rows[moved["story"]["story_id"]]["closed_by"] is None
    listed = await tools.dispatch("list_stories", {"filters": {"written_by.run": 7}})
    assert isinstance(listed, dict) and len(listed["stories"]) == 1


async def test_reads_select_by_who_closed_a_record_as_memory_held_it(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = await _memory(storage)
    await memory.set_history("acme", True)
    jan5, feb1 = datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    told = {
        text: (await memory.remember("acme", text, valid_from=jan5)).story.story_id
        for text in (
            "Ana owns the review.",
            "Bo owns the audit.",
            "Cy owns the plan.",
            "Di owns it.",
        )
    }
    ana, bo, cy, di = told.values()
    doc = (await memory.ingest("acme", name="seal notes", text="Replace the seal yearly.")).source
    clock[0] = datetime(2026, 1, 20, tzinfo=UTC)
    await memory.end_story("acme", ana, valid_to=feb1, written_by=RUN_7)
    await memory.end_story("acme", bo, valid_to=feb1, written_by=RUN_8)
    await memory.end_story("acme", di, valid_to=feb1)
    await memory.retire_source("acme", doc.source_id, valid_to=feb1, written_by=RUN_7)
    clock[0] = datetime(2026, 1, 30, tzinfo=UTC)
    await memory.end_story("acme", ana, valid_to=feb1 + timedelta(days=1), written_by=RUN_8)
    await memory.retire_source(
        "acme", doc.source_id, valid_to=feb1 + timedelta(days=1), written_by=RUN_8
    )
    clock[0] = datetime(2026, 2, 5, tzinfo=UTC)
    by7, by8 = {"closed_by.run": 7}, {"closed_by.run": 8}
    closer = {"closed_by.run": {"gte": 0}}
    before, between = datetime(2026, 1, 15, tzinfo=UTC), datetime(2026, 1, 25, tzinfo=UTC)

    async def stories(filters: dict[str, Any], **when: Any) -> list[str]:
        listed, _ = await memory.list_stories("acme", filters=filters, **when)
        recalled = await memory.recall("acme", "owns", filters=filters, top_k=50, **when)
        searched = await memory.search(
            "acme", "owns", kinds=("story",), filters=filters, top_k=50, **when
        )
        assert sorted(h.id for h in recalled.hits) == sorted(h.id for h in searched.hits)
        assert sorted(h.id for h in recalled.hits) == sorted(s.story_id for s in listed)
        return sorted(s.story_id for s in listed)

    async def sources(filters: dict[str, Any], **when: Any) -> list[str]:
        listed, _ = await memory.list_sources("acme", filters=filters, **when)
        alike = await memory.resembling(
            "acme", text="Replace the seal yearly.", filters=filters, **when
        )
        hits = await memory.search(
            "acme", "seal", kinds=("chunk",), filters=filters, top_k=50, **when
        )
        found = sorted(s.source_id for s in listed)
        assert sorted(m.source.source_id for m in alike.matches) == found
        assert sorted({h.chunk.source_id for h in hits.hits if h.chunk}) == found
        return found

    every = {"history": True}
    assert await stories(by7, **every) == [] and await stories(by8, **every) == sorted([ana, bo])
    assert await stories(closer, **every) == sorted([ana, bo])
    assert await stories(by7, known_at=between, history=True) == [ana]
    assert await stories(by8, known_at=between, history=True) == [bo]
    assert await stories(closer, known_at=before, history=True) == []
    assert await stories(by8, as_of=between) == sorted([ana, bo])
    assert await sources(by7, **every) == [] and await sources(by8, **every) == [doc.source_id]
    assert await sources(by7, known_at=between, history=True) == [doc.source_id]
    assert await sources(closer, known_at=before, history=True) == []
    [believed] = (
        await memory.resembling(
            "acme", text="Replace the seal yearly.", known_at=between, history=True
        )
    ).matches
    assert (believed.source.closed_by, believed.source.valid_to) == (RUN_7, feb1)
    assert believed.source.revised_at is None
    around = await memory.expand(
        "acme", ("story", cy), by="vector", filters=by8, history=True, top_k=10
    )
    assert sorted(h.id for h in around.hits) == sorted([ana, bo, f"{doc.source_id}-0"])
    [hit] = (await memory.search("acme", "seal", kinds=("chunk",), history=True)).hits
    assert hit.chunk is not None and hit.chunk.closed_by == RUN_8
    [then] = (
        await memory.search("acme", "seal", kinds=("chunk",), known_at=between, history=True)
    ).hits
    assert then.chunk is not None and then.chunk.closed_by == RUN_7
    [early] = (
        await memory.search("acme", "seal", kinds=("chunk",), known_at=before, history=True)
    ).hits
    assert early.chunk is not None and early.chunk.closed_by is None
    tools = MemoryTools(memory, corpus_id="acme")
    found = await tools.dispatch(
        "search", {"query": "seal", "filters": by8, "history": True, "kinds": ["chunk"]}
    )
    assert isinstance(found, dict) and [r["closed_by"] for r in found["hits"]] == [RUN_8]
    alike = await tools.dispatch(
        "resembling", {"text": "Replace the seal yearly.", "history": True}
    )
    assert isinstance(alike, dict) and [m["closed_by"] for m in alike["matches"]] == [RUN_8]
    with pytest.raises(
        InputError, match=r"^memory: a sync names its set by metadata; 'closed_by.run' is not"
    ):
        await memory.sync("acme", filters=by7, sources=[])
    with pytest.raises(
        ConfigurationError,
        match=r"^tools: a view is bound by metadata; 'closed_by.run' is not one$",
    ):
        MemoryTools(memory, corpus_id="acme", filters=by7)
    with pytest.raises(
        InputError, match=r"^filters: filter key 'closed_by\.' must address a facet"
    ):
        await memory.list_stories("acme", filters={"closed_by.": 7})
    with pytest.raises(InputError, match=r"^postgres: an indexed key is a filter key"):
        keys._checked("closed_by.run")


async def test_a_repeat_records_the_latest_confirmation_and_never_moves_it_back(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = await _memory(storage)
    jan10, jan20 = clock[0], datetime(2026, 1, 20, tzinfo=UTC)
    told = await memory.remember("acme", "Ana owns the review.", subject="ana", written_by=RUN_7)
    first = told.story
    assert (first.confirmed_at, first.confirmed_by, told.repeated) == (jan10, RUN_7, False)
    clock[0] = jan20
    again = await memory.remember("acme", "ana owns  the review.", subject="ana", written_by=RUN_8)
    assert again.repeated and again.story.story_id == first.story_id
    assert (again.story.confirmed_at, again.story.confirmed_by) == (jan20, RUN_8)
    assert (again.story.created_at, again.story.written_by) == (jan10, RUN_7)
    clock[0] = datetime(2026, 1, 15, tzinfo=UTC)
    late = await memory.remember("acme", "Ana owns the review.", subject="ana", written_by=RUN_9)
    assert late.repeated and (late.story.confirmed_at, late.story.confirmed_by) == (jan20, RUN_8)
    assert await memory.get_stories("acme", [first.story_id]) == [again.story]
    assert len((await memory.list_stories("acme", history=True))[0]) == 1
    since = datetime(2026, 1, 18, tzinfo=UTC)
    told_since, _ = await memory.list_stories("acme", filters={"confirmed_at": {"gte": since}})
    assert [s.story_id for s in told_since] == [first.story_id]
    assert (await memory.list_stories("acme", filters={"confirmed_at": {"lt": since}}))[0] == []
    clock[0] = datetime(2026, 1, 25, tzinfo=UTC)
    fair: dict[str, Any] = {
        "valid_from": datetime(2020, 1, 1, tzinfo=UTC),
        "valid_to": datetime(2020, 1, 5, tzinfo=UTC),
    }
    ran = await memory.remember("acme", "The fair ran.", written_by=RUN_7, **fair)
    clock[0] = datetime(2026, 1, 26, tzinfo=UTC)
    rerun = await memory.remember("acme", "The fair ran.", written_by=RUN_8, **fair)
    assert rerun.repeated and rerun.story.story_id == ran.story.story_id
    assert (rerun.story.confirmed_at, rerun.story.confirmed_by) == (clock[0], RUN_8)
    later = await memory.supersede("acme", first.story_id, "Bo owns the review.", written_by=RUN_9)
    assert later.repeated is False
    assert (later.story.confirmed_at, later.story.confirmed_by) == (later.story.created_at, RUN_9)
    clock[0] = datetime(2026, 1, 27, tzinfo=UTC)
    tools = MemoryTools(memory, corpus_id="acme", written_by=RUN_7)
    row = await tools.dispatch("remember", {"text": "Bo owns the review.", "subject": "ana"})
    assert isinstance(row, dict) and row["repeated"] is True
    assert row["story"]["confirmed_by"] == RUN_7 and row["story"]["written_by"] == RUN_9


async def test_a_kept_document_records_who_confirmed_it_last(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = await _memory(storage)
    doc = {"metadata.doc": "returns"}
    policy = SourceIn(name="Returns policy", text="30 days.", metadata={"doc": "returns"})
    [first] = (await memory.sync("acme", filters=doc, sources=[policy], written_by=RUN_7)).sources
    assert (first.confirmed_at, first.confirmed_by) == (clock[0], RUN_7)
    clock[0] = datetime(2026, 1, 20, tzinfo=UTC)
    [kept] = (await memory.sync("acme", filters=doc, sources=[policy], written_by=RUN_8)).sources
    assert (kept.confirmed_at, kept.confirmed_by, kept.written_by) == (clock[0], RUN_8, RUN_7)
    clock[0] = datetime(2026, 1, 15, tzinfo=UTC)
    [held] = (await memory.sync("acme", filters=doc, sources=[policy], written_by=RUN_9)).sources
    assert (held.confirmed_at, held.confirmed_by) == (kept.confirmed_at, RUN_8)
    assert await memory.get_source("acme", first.source_id) == held
    filed = (
        await memory.ingest("acme", name="note", text="Ground first.", written_by=RUN_9)
    ).source
    assert (filed.confirmed_at, filed.confirmed_by) == (filed.created_at, RUN_9)
    tools = MemoryTools(memory, corpus_id="acme")
    alike = await tools.dispatch("resembling", {"text": "30 days."})
    assert isinstance(alike, dict)
    assert [m["confirmed_by"] for m in alike["matches"] if m["source_id"] == first.source_id] == [
        RUN_8
    ]
