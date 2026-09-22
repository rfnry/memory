from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError
from rfnry_memory.memory import Memory
from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.tools import MemoryTools, Tool, Tools

from .fakes import Fixed, HashEmbedding

ALL = {"search", "recall", "read_source", "list_sources", "remember", "supersede", "forget", "list_stories", "neighbors"}


async def _memory(**kwargs: Any) -> Memory:
    memory = Memory(InMemoryStorage(), embedding=HashEmbedding(), chunk_chars=200, overlap_chars=20, **kwargs)
    await memory.initialize()
    await memory.ingest("acme", name="manual", text="Replace the pump seal yearly.", metadata={"line": "A"})
    await memory.ingest("acme", name="other", text="Replace the pump seal monthly on line B.", metadata={"line": "B"})
    return memory


def _names(tools: MemoryTools) -> set[str]:
    return {d["name"] for d in tools.definitions()}


async def test_definitions_are_gated_and_shaped() -> None:
    plain = await _memory()
    assert _names(MemoryTools(plain, corpus_id="acme")) == ALL - {"neighbors"}
    graph = await _memory(pillars={"text", "vector", "graph"})
    assert _names(MemoryTools(graph, corpus_id="acme")) == ALL
    judged = await _memory(decision=Fixed({}))
    search = next(d for d in MemoryTools(judged, corpus_id="acme").definitions() if d["name"] == "search")
    assert "judge" in search["input_schema"]["properties"]
    search = next(d for d in MemoryTools(plain, corpus_id="acme").definitions() if d["name"] == "search")
    assert "judge" not in search["input_schema"]["properties"]
    assert _names(MemoryTools(plain, corpus_id="acme", include={"search", "recall"})) == {"search", "recall"}
    assert all({"name", "description", "input_schema"} <= set(d) for d in MemoryTools(plain, corpus_id="acme").definitions())


async def test_corpus_is_required_and_bound_filters_win() -> None:
    memory = await _memory()
    with pytest.raises(ConfigurationError):
        MemoryTools(memory, corpus_id="")
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    result = await tools.dispatch("search", {"query": "pump seal"})
    assert result["hits"] and all(h["kind"] == "chunk" for h in result["hits"]) and all("yearly" in h["text"] for h in result["hits"])
    widened = await tools.dispatch("search", {"query": "pump seal", "filters": {"metadata.line": "B"}})
    assert widened == result or all("yearly" in h["text"] for h in widened["hits"])


async def test_story_verbs_round_trip_through_dispatch() -> None:
    tools = MemoryTools(await _memory(), corpus_id="acme")
    remembered = await tools.dispatch("remember", {"text": "Ana owns the wiring review.", "subject": "user:ana", "occurred_at": "2026-03-05T00:00:00Z"})
    story_id = remembered["story"]["story_id"]
    recalled = await tools.dispatch("recall", {"query": "wiring review", "subject": "user:ana"})
    assert [h["id"] for h in recalled["hits"]] == [story_id]
    superseded = await tools.dispatch("supersede", {"story_id": story_id, "text": "Ana handed the wiring review to Bo.", "effective_at": "2026-04-01T00:00:00Z"})
    assert superseded["closed"]["valid_to"].startswith("2026-04-01") and superseded["current"]["supersedes"] == story_id
    listed = await tools.dispatch("list_stories", {"subject": "user:ana", "as_of": "2026-03-10T00:00:00Z"})
    assert [s["story_id"] for s in listed["stories"]] == [story_id]
    assert await tools.dispatch("forget", {"story_id": superseded["current"]["story_id"]}) == {"forgotten": superseded["current"]["story_id"]}


async def test_errors_are_results_never_exceptions() -> None:
    tools = MemoryTools(await _memory(), corpus_id="acme")
    assert (await tools.dispatch("ghost", {}))["error"]["type"] == "UnknownTool"
    assert (await tools.dispatch("search", {}))["error"]["type"] == "InputError"
    assert (await tools.dispatch("forget", {"story_id": "missing"}))["error"] == {
        "type": "NotFoundError", "message": "memory: story 'missing' was not found"}
    assert (await tools.dispatch("ghost", {}))["error"]["message"].startswith("tools: unknown tool 'ghost'")
    assert (await tools.dispatch("search", {}))["error"]["message"].startswith("tools: invalid arguments for search: ")
    assert (await tools.dispatch("search", {"query": "x", "filters": {"corpus_id": "other"}}))["error"]["type"] == "InputError"


async def test_read_and_list_sources_and_clipping() -> None:
    memory = await _memory()
    tools = MemoryTools(memory, corpus_id="acme", max_result_chars=1000)
    listed = await tools.dispatch("list_sources", {})
    assert {s["name"] for s in listed["sources"]} == {"manual", "other"}
    source_id = listed["sources"][0]["source_id"]
    assert "pump seal" in (await tools.dispatch("read_source", {"source_id": source_id}))["text"]
    await memory.ingest("acme", name="long", text="seal " * 3000)
    clipped = await tools.dispatch("search", {"query": "seal", "top_k": 10})
    assert clipped["truncated"] is True and len(str(clipped)) <= 1400


async def test_memory_tools_compose_with_other_tools_and_keep_their_posture() -> None:
    async def now(args: dict[str, Any]) -> str:
        return "2026-09-22T00:00:00Z"

    combined = MemoryTools(await _memory(), corpus_id="acme").only("search", "recall") + Tools([Tool("now", "UTC time", body=now)])
    assert combined.names == ["search", "recall", "now"]
    assert await combined.dispatch("now", {}) == "2026-09-22T00:00:00Z"
    result = await combined.dispatch("search", {"query": "pump seal"})
    assert isinstance(result, dict) and result["hits"]
    failed = await combined.dispatch("search", {"filters": {"corpus_id": "x"}, "query": "q"})
    assert isinstance(failed, dict) and failed["error"]["type"] == "InputError"


async def test_recall_answers_now_and_history_marks_what_was_superseded() -> None:
    tools = MemoryTools(await _memory(), corpus_id="acme")
    recall = next(d for d in tools.definitions() if d["name"] == "recall")
    assert "history" in recall["input_schema"]["properties"] and "current facts" in recall["description"]
    assert all("history" in d["input_schema"]["properties"] for d in tools.definitions() if d["name"] in {"search", "list_stories"})
    remembered = await tools.dispatch("remember", {"text": "The go-live is February 20.", "subject": "project:apollo", "occurred_at": "2026-01-10T00:00:00Z"})
    old_id = remembered["story"]["story_id"]
    superseded = await tools.dispatch("supersede", {"story_id": old_id, "text": "The go-live moved to March 3.", "effective_at": "2026-02-01T00:00:00Z"})
    new_id = superseded["current"]["story_id"]
    now = await tools.dispatch("recall", {"query": "go-live"})
    assert [h["id"] for h in now["hits"]] == [new_id]
    history = await tools.dispatch("recall", {"query": "go-live", "history": True})
    marks = {h["id"]: h["story"]["valid_to"] for h in history["hits"]}
    assert set(marks) == {old_id, new_id} and marks[new_id] is None and marks[old_id].startswith("2026-02-01")
    both = await tools.dispatch("recall", {"query": "go-live", "history": True, "as_of": "2026-01-20T00:00:00Z"})
    assert both["error"]["type"] == "InputError"
    listed = await tools.dispatch("list_stories", {"subject": "project:apollo", "history": True})
    assert [s["story_id"] for s in listed["stories"]] == [new_id, old_id]


async def test_bound_filters_are_stamped_on_every_write_so_the_tools_see_their_own_writes() -> None:
    memory = await _memory()
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A", "metadata.site.zone": 3})
    remembered = await tools.dispatch("remember", {"text": "Ana owns the wiring review.", "metadata": {"line": "A", "rev": 2}})
    assert remembered["story"]["metadata"] == {"line": "A", "rev": 2, "site": {"zone": 3}}
    story_id = remembered["story"]["story_id"]
    assert [h["id"] for h in (await tools.dispatch("recall", {"query": "wiring review"}))["hits"]] == [story_id]
    assert [s["story_id"] for s in (await tools.dispatch("list_stories", {}))["stories"]] == [story_id]
    superseded = await tools.dispatch("supersede", {"story_id": story_id, "text": "Bo owns the wiring review."})
    assert superseded["current"]["metadata"] == {"line": "A", "rev": 2, "site": {"zone": 3}}
    replaced = await tools.dispatch("supersede", {"story_id": superseded["current"]["story_id"], "text": "Cy owns the wiring review.",
                                                  "metadata": {"note": "handover"}})
    assert replaced["current"]["metadata"] == {"note": "handover", "line": "A", "site": {"zone": 3}}
    assert [h["id"] for h in (await tools.dispatch("recall", {"query": "wiring review"}))["hits"]] == [replaced["current"]["story_id"]]


async def test_a_write_outside_the_bound_view_is_refused() -> None:
    memory = await _memory()
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A", "metadata.site.zone": 3})
    conflict = await tools.dispatch("remember", {"text": "Line B runs nights.", "metadata": {"line": "B"}})
    assert conflict["error"]["type"] == "InputError" and "'metadata.line'" in conflict["error"]["message"]
    assert conflict["error"]["message"].startswith("filters: ")
    nested = await tools.dispatch("remember", {"text": "Zone north.", "metadata": {"site": "north"}})
    assert nested["error"]["type"] == "InputError" and "'metadata.site.zone'" in nested["error"]["message"]
    assert (await memory.list_stories("acme"))[0] == []
    outside = await memory.remember("acme", "Line B runs nights.", metadata={"line": "B"})
    for name, args in (("supersede", {"story_id": outside.story_id, "text": "Line B runs days."}), ("forget", {"story_id": outside.story_id})):
        assert (await tools.dispatch(name, args))["error"] == {"type": "NotFoundError", "message": f"tools: story {outside.story_id!r} was not found"}
    assert [s.story_id for s in (await memory.list_stories("acme"))[0]] == [outside.story_id]


async def test_a_bound_filter_that_cannot_be_stamped_withholds_the_writing_tools() -> None:
    memory = await _memory()
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": ["A", "B"]})
    assert _names(tools) == ALL - {"neighbors", "remember", "supersede"}
    for name in ("remember", "supersede"):
        refused = await tools.dispatch(name, {"text": "Ana owns the wiring review.", "story_id": "x"})
        assert refused["error"]["type"] == "InputError" and "'metadata.line'" in refused["error"]["message"]
    assert (await memory.list_stories("acme"))[0] == []
    assert _names(MemoryTools(memory, corpus_id="acme", filters={"metadata.line": ["A"]}, include={"remember", "recall"})) == {"recall"}
