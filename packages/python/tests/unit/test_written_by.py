from datetime import timedelta

import pytest

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding

RUN_7 = {"run": 7, "agent": "planner"}
RUN_8 = {"run": 8, "agent": "reviewer"}


async def _memory(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    return memory


async def test_a_story_records_who_wrote_it_and_who_closed_it(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    told = (await memory.remember("acme", "Ana owns the review.", written_by=RUN_7)).story
    assert (told.written_by, told.closed_by) == (RUN_7, None)
    again = (await memory.remember("acme", "Ana owns the review.", written_by=RUN_8)).story
    assert again == told
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
