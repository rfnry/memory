import json
from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Answer, CheckAnswer, EntityIn, RelationIn
from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.tools import MemoryTools, Tool, Tools

from .fakes import Fixed, HashEmbedding


async def _call(tools: MemoryTools, name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await tools.dispatch(name, args)
    assert isinstance(result, dict)
    return result


ALL = {
    "search",
    "recall",
    "read_source",
    "list_sources",
    "ingest",
    "remember",
    "supersede",
    "delete_story",
    "list_stories",
    "neighbors",
}


async def _memory(**kwargs: Any) -> Memory:
    memory = Memory(
        InMemoryStorage(), embedding=HashEmbedding(), chunk_chars=200, overlap_chars=20, **kwargs
    )
    await memory.initialize()
    await memory.ingest(
        "acme", name="manual", text="Replace the pump seal yearly.", metadata={"line": "A"}
    )
    await memory.ingest(
        "acme",
        name="other",
        text="Replace the pump seal monthly on line B.",
        metadata={"line": "B"},
    )
    return memory


def _names(tools: MemoryTools) -> set[str]:
    return {d["name"] for d in tools.definitions()}


async def test_definitions_are_gated_and_shaped() -> None:
    plain = await _memory()
    assert _names(MemoryTools(plain, corpus_id="acme")) == ALL - {"neighbors"}
    graph = await _memory(pillars={"text", "vector", "graph"})
    assert _names(MemoryTools(graph, corpus_id="acme")) == ALL
    judged = await _memory(decision=Fixed({}))
    search = next(
        d for d in MemoryTools(judged, corpus_id="acme").definitions() if d["name"] == "search"
    )
    assert "judge" in search["input_schema"]["properties"]
    search = next(
        d for d in MemoryTools(plain, corpus_id="acme").definitions() if d["name"] == "search"
    )
    assert "judge" not in search["input_schema"]["properties"]
    assert _names(MemoryTools(plain, corpus_id="acme", include={"search", "recall"})) == {
        "search",
        "recall",
    }
    assert all(
        {"name", "description", "input_schema"} <= set(d)
        for d in MemoryTools(plain, corpus_id="acme").definitions()
    )


async def test_corpus_is_required_and_bound_filters_win() -> None:
    memory = await _memory()
    with pytest.raises(ConfigurationError):
        MemoryTools(memory, corpus_id="")
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    result = await _call(tools, "search", {"query": "pump seal"})
    assert (
        result["hits"]
        and all(h["kind"] == "chunk" for h in result["hits"])
        and all("yearly" in h["text"] for h in result["hits"])
    )
    widened = await _call(
        tools, "search", {"query": "pump seal", "filters": {"metadata.line": "B"}}
    )
    assert widened == result or all("yearly" in h["text"] for h in widened["hits"])


async def test_story_verbs_round_trip_through_dispatch() -> None:
    tools = MemoryTools(await _memory(), corpus_id="acme")
    remembered = await _call(
        tools,
        "remember",
        {
            "text": "Ana owns the wiring review.",
            "subject": "user:ana",
            "valid_from": "2026-03-05T00:00:00Z",
        },
    )
    assert set(remembered) == {"story"}
    story_id = remembered["story"]["story_id"]
    recalled = await _call(tools, "recall", {"query": "wiring review", "subject": "user:ana"})
    assert [h["id"] for h in recalled["hits"]] == [story_id]
    superseded = await _call(
        tools,
        "supersede",
        {
            "story_id": story_id,
            "text": "Ana handed the wiring review to Bo.",
            "valid_from": "2026-04-01T00:00:00Z",
        },
    )
    assert (
        superseded == {"story": superseded["story"]}
        and superseded["story"]["valid_from"].startswith("2026-04-01")
        and superseded["story"]["supersedes"] == story_id
    )
    listed = await _call(
        tools, "list_stories", {"subject": "user:ana", "as_of": "2026-03-10T00:00:00Z"}
    )
    assert [s["story_id"] for s in listed["stories"]] == [story_id]
    assert await _call(tools, "delete_story", {"story_id": superseded["story"]["story_id"]}) == {
        "deleted": superseded["story"]["story_id"]
    }


async def test_errors_raise_the_librarys_exceptions_as_any_tools_does() -> None:
    async def now(args: dict[str, Any]) -> str:
        return "2026-09-22T00:00:00Z"

    tools = MemoryTools(await _memory(), corpus_id="acme")
    for surface in (
        tools,
        tools.only("search", "delete_story"),
        tools + Tools([Tool("now", "UTC time", body=now)]),
    ):
        with pytest.raises(InputError, match=r"^tools: unknown tool 'ghost'$"):
            await surface.dispatch("ghost", {})
        with pytest.raises(InputError, match=r"^tools: invalid arguments for search: "):
            await surface.dispatch("search", {})
        with pytest.raises(NotFoundError, match=r"^memory: story 'missing' was not found$"):
            await surface.dispatch("delete_story", {"story_id": "missing"})
        with pytest.raises(InputError, match=r"^filters: filter key 'corpus_id' is reserved"):
            await surface.dispatch("search", {"query": "x", "filters": {"corpus_id": "other"}})


async def test_read_and_list_sources_and_clipping() -> None:
    memory = await _memory()
    tools = MemoryTools(memory, corpus_id="acme", max_result_chars=1000)
    listed = await _call(tools, "list_sources", {})
    assert {s["name"] for s in listed["sources"]} == {"manual", "other"}
    source_id = listed["sources"][0]["source_id"]
    assert "pump seal" in (await _call(tools, "read_source", {"source_id": source_id}))["text"]
    await memory.ingest("acme", name="long", text="seal " * 3000)
    clipped = await _call(tools, "search", {"query": "seal", "top_k": 10})
    assert clipped["truncated"] is True and len(str(clipped)) <= 1400


async def test_a_neighbors_result_is_clipped_like_any_other() -> None:
    memory = Memory(InMemoryStorage(), pillars={"text", "graph"})
    names = [f"Valve-{i:04d}-" + "x" * 40 for i in range(60)]
    await memory.ingest(
        "acme",
        name="sheet",
        text="Hub H-1 connects many valves.",
        entities=[
            EntityIn(name="H-1", type="hub"),
            *(EntityIn(name=n, type="valve") for n in names),
        ],
        relations=[RelationIn(from_name="H-1", to_name=n, type="CONNECTS_TO") for n in names],
    )
    tools = MemoryTools(memory, corpus_id="acme", max_result_chars=1000)
    walked = await _call(tools, "neighbors", {"entity": "H-1"})
    assert walked["truncated"] is True and len(json.dumps(walked)) <= 1000
    assert walked["passages"] == [] and walked["relations"] == [] and walked["entities"]


async def test_memory_tools_compose_with_other_tools_and_keep_their_posture() -> None:
    async def now(args: dict[str, Any]) -> str:
        return "2026-09-22T00:00:00Z"

    combined = MemoryTools(await _memory(), corpus_id="acme").only("search", "recall") + Tools(
        [Tool("now", "UTC time", body=now)]
    )
    assert combined.names == ["search", "recall", "now"]
    assert await combined.dispatch("now", {}) == "2026-09-22T00:00:00Z"
    result = await combined.dispatch("search", {"query": "pump seal"})
    assert isinstance(result, dict) and result["hits"]
    with pytest.raises(InputError):
        await combined.dispatch("search", {"filters": {"corpus_id": "x"}, "query": "q"})


async def test_tools_refuse_a_duplicate_name_a_missing_body_and_an_unknown_name() -> None:
    bodiless = Tool("now", "UTC time")
    with pytest.raises(ConfigurationError, match=r"^tools: duplicate tool name 'now'$"):
        Tools([bodiless, bodiless])
    with pytest.raises(ConfigurationError, match=r"^tools: tool 'now' has no body$"):
        await Tools([bodiless]).dispatch("now", {})
    with pytest.raises(InputError, match=r"^tools: unknown tools \['ghost'\]$"):
        Tools([bodiless]).only("now", "ghost")


async def test_without_drops_the_named_tools_and_keeps_the_rest() -> None:
    tools = MemoryTools(await _memory(), corpus_id="acme")
    kept = tools.without("remember", "delete_story", "ghost")
    assert kept.names == [n for n in tools.names if n not in {"remember", "delete_story"}]
    assert {d["name"] for d in kept.definitions()} == set(kept.names)
    with pytest.raises(InputError, match=r"^tools: unknown tool 'delete_story'$"):
        await kept.dispatch("delete_story", {"story_id": "missing"})
    found = await kept.dispatch("search", {"query": "pump seal"})
    assert isinstance(found, dict) and found["hits"]
    assert "delete_story" in tools and len(tools) == len(kept) + 2


async def test_recall_answers_now_and_history_marks_what_was_superseded() -> None:
    tools = MemoryTools(await _memory(), corpus_id="acme")
    recall = next(d for d in tools.definitions() if d["name"] == "recall")
    assert (
        "history" in recall["input_schema"]["properties"]
        and "current facts" in recall["description"]
    )
    assert all(
        "history" in d["input_schema"]["properties"]
        for d in tools.definitions()
        if d["name"] in {"search", "list_stories"}
    )
    remembered = await _call(
        tools,
        "remember",
        {
            "text": "The go-live is February 20.",
            "subject": "project:apollo",
            "valid_from": "2026-01-10T00:00:00Z",
        },
    )
    old_id = remembered["story"]["story_id"]
    superseded = await _call(
        tools,
        "supersede",
        {
            "story_id": old_id,
            "text": "The go-live moved to March 3.",
            "valid_from": "2026-02-01T00:00:00Z",
        },
    )
    new_id = superseded["story"]["story_id"]
    now = await _call(tools, "recall", {"query": "go-live"})
    assert [h["id"] for h in now["hits"]] == [new_id]
    history = await _call(tools, "recall", {"query": "go-live", "history": True})
    marks = {h["id"]: h["story"]["valid_to"] for h in history["hits"]}
    assert (
        set(marks) == {old_id, new_id}
        and marks[new_id] is None
        and marks[old_id].startswith("2026-02-01")
    )
    with pytest.raises(
        InputError, match=r"^memory: history=True reads every moment; it takes no as_of$"
    ):
        await _call(
            tools, "recall", {"query": "go-live", "history": True, "as_of": "2026-01-20T00:00:00Z"}
        )
    listed = await _call(tools, "list_stories", {"subject": "project:apollo", "history": True})
    assert [s["story_id"] for s in listed["stories"]] == [new_id, old_id]


async def test_bound_filters_are_stamped_on_every_write_so_the_tools_see_their_own_writes() -> None:
    memory = await _memory()
    tools = MemoryTools(
        memory, corpus_id="acme", filters={"metadata.line": "A", "metadata.site.zone": 3}
    )
    remembered = await _call(
        tools,
        "remember",
        {"text": "Ana owns the wiring review.", "metadata": {"line": "A", "rev": 2}},
    )
    assert remembered["story"]["metadata"] == {"line": "A", "rev": 2, "site": {"zone": 3}}
    story_id = remembered["story"]["story_id"]
    assert [
        h["id"] for h in (await _call(tools, "recall", {"query": "wiring review"}))["hits"]
    ] == [story_id]
    assert [s["story_id"] for s in (await _call(tools, "list_stories", {}))["stories"]] == [
        story_id
    ]
    superseded = await _call(
        tools, "supersede", {"story_id": story_id, "text": "Bo owns the wiring review."}
    )
    assert superseded["story"]["metadata"] == {"line": "A", "rev": 2, "site": {"zone": 3}}
    replaced = await _call(
        tools,
        "supersede",
        {
            "story_id": superseded["story"]["story_id"],
            "text": "Cy owns the wiring review.",
            "metadata": {"note": "handover"},
        },
    )
    assert replaced["story"]["metadata"] == {"note": "handover", "line": "A", "site": {"zone": 3}}
    assert [
        h["id"] for h in (await _call(tools, "recall", {"query": "wiring review"}))["hits"]
    ] == [replaced["story"]["story_id"]]


async def test_a_write_outside_the_bound_view_is_refused() -> None:
    memory = await _memory()
    tools = MemoryTools(
        memory, corpus_id="acme", filters={"metadata.line": "A", "metadata.site.zone": 3}
    )
    with pytest.raises(
        InputError, match=r"^filters: metadata must agree with the bound filter 'metadata.line'"
    ):
        await _call(tools, "remember", {"text": "Line B runs nights.", "metadata": {"line": "B"}})
    with pytest.raises(
        InputError,
        match=r"^filters: metadata must agree with the bound filter 'metadata.site.zone'",
    ):
        await _call(tools, "remember", {"text": "Zone north.", "metadata": {"site": "north"}})
    assert (await memory.list_stories("acme"))[0] == []
    outside = (await memory.remember("acme", "Line B runs nights.", metadata={"line": "B"})).story
    for name, args in (
        ("supersede", {"story_id": outside.story_id, "text": "Line B runs days."}),
        ("delete_story", {"story_id": outside.story_id}),
    ):
        with pytest.raises(
            NotFoundError, match=rf"^tools: story '{outside.story_id}' was not found$"
        ):
            await _call(tools, name, args)
    assert [s.story_id for s in (await memory.list_stories("acme"))[0]] == [outside.story_id]


async def test_a_bound_filter_takes_one_value() -> None:
    memory = await _memory()
    for bound in (["A", "B"], ["A"]):
        with pytest.raises(
            ConfigurationError,
            match=r"^tools: a bound filter takes one value; 'metadata.line' is bound to a list — "
            r"build one MemoryTools per value$",
        ):
            MemoryTools(
                memory, corpus_id="acme", filters={"metadata.site": "north", "metadata.line": bound}
            )
    assert _names(MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})) == ALL - {
        "neighbors"
    }


async def test_neighbors_takes_an_entity_type_when_a_name_is_ambiguous() -> None:
    memory = await _memory(pillars={"text", "vector", "graph"})
    await memory.ingest(
        "acme",
        name="bom",
        text="Part P-3 is a gasket.",
        entities=[EntityIn(name="P-3", type="part")],
    )
    await memory.ingest(
        "acme", name="sheet", text="Pump P-3 runs.", entities=[EntityIn(name="P-3", type="pump")]
    )
    tools = MemoryTools(memory, corpus_id="acme")
    neighbors = next(d for d in tools.definitions() if d["name"] == "neighbors")
    assert "entity_type" in neighbors["input_schema"]["properties"]
    with pytest.raises(InputError, match=r"\['part', 'pump'\]"):
        await _call(tools, "neighbors", {"entity": "P-3"})
    assert [
        e["type"]
        for e in (await _call(tools, "neighbors", {"entity": "P-3", "entity_type": "pump"}))[
            "entities"
        ]
    ] == ["pump"]


async def test_a_duplicate_outside_the_bound_view_reveals_nothing() -> None:
    memory = await _memory()
    outside = (await memory.remember("acme", "Line B runs nights.", metadata={"line": "B"})).story
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    hidden = r"^tools: this text already exists outside your view$"
    with pytest.raises(InputError, match=hidden):
        await _call(tools, "remember", {"text": "line b   runs NIGHTS."})
    inside = (await _call(tools, "remember", {"text": "Line A runs days."}))["story"]
    assert (await _call(tools, "remember", {"text": "line a runs days."}))["story"] == inside
    with pytest.raises(InputError, match=hidden):
        await _call(
            tools, "supersede", {"story_id": inside["story_id"], "text": "Line B runs nights."}
        )
    other = (await _call(tools, "remember", {"text": "Line A runs weekends."}))["story"]
    with pytest.raises(InputError, match=inside["story_id"]):
        await _call(
            tools, "supersede", {"story_id": other["story_id"], "text": "Line A runs days."}
        )
    stories, _ = await memory.list_stories("acme")
    assert sorted(s.story_id for s in stories) == sorted(
        [outside.story_id, inside["story_id"], other["story_id"]]
    )
    assert all(s.valid_to is None for s in stories)


def test_descriptions_say_which_kinds_each_search_covers() -> None:
    definitions = {
        d["name"]: d["description"]
        for d in MemoryTools(Memory(InMemoryStorage()), corpus_id="acme").definitions()
    }
    assert "both sources (ingested documents) and stories" in definitions["search"]
    assert "stories only, not sources; use search for documents" in definitions["recall"]


async def test_a_bound_view_reads_nothing_outside_it() -> None:
    memory = await _memory(pillars={"text", "vector", "graph"})
    await memory.ingest(
        "acme",
        name="sheet B",
        text="Valve V-9 feeds pump P-9.",
        metadata={"line": "B"},
        entities=[EntityIn(name="V-9", type="valve"), EntityIn(name="P-9", type="pump")],
    )
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    assert "neighbors" not in _names(tools) and "neighbors" in _names(
        MemoryTools(memory, corpus_id="acme")
    )
    with pytest.raises(InputError, match=r"^tools: unknown tool 'neighbors'$"):
        await tools.dispatch("neighbors", {"entity": "V-9"})
    sources = {s.name: s.source_id for s in (await memory.list_sources("acme"))[0]}
    assert "yearly" in (await _call(tools, "read_source", {"source_id": sources["manual"]}))["text"]
    for source_id in (sources["other"], sources["sheet B"], "missing"):
        with pytest.raises(NotFoundError, match=rf"^tools: source '{source_id}' was not found$"):
            await tools.dispatch("read_source", {"source_id": source_id})
    outside = (await memory.remember("acme", "Line B runs nights.", metadata={"line": "B"})).story
    for name in ("delete_story", "supersede"):
        for story_id in (outside.story_id, "missing"):
            with pytest.raises(NotFoundError, match=rf"^tools: story '{story_id}' was not found$"):
                await tools.dispatch(name, {"story_id": story_id, "text": "Line A runs days."})


async def test_ingest_files_a_source_the_same_tools_then_find() -> None:
    memory = await _memory()
    tools = MemoryTools(memory, corpus_id="acme")
    ingest = next(d for d in tools.definitions() if d["name"] == "ingest")
    assert set(ingest["input_schema"]["properties"]) == {"name", "text", "metadata"}
    ingested = await _call(
        tools,
        "ingest",
        {
            "name": "die-change.md",
            "text": "Torque the die clamp bolts to 180 Nm.",
            "metadata": {"rev": "B"},
        },
    )
    assert set(ingested) == {"source"} and "text" not in ingested["source"]
    source = ingested["source"]
    assert (
        source["name"] == "die-change.md"
        and source["chunks"] == 1
        and source["metadata"] == {"rev": "B"}
    )
    hits = (await _call(tools, "search", {"query": "die clamp bolts torque", "kinds": ["chunk"]}))[
        "hits"
    ]
    assert hits[0]["source_id"] == source["source_id"] and "180 Nm" in hits[0]["text"]
    assert (
        "180 Nm" in (await _call(tools, "read_source", {"source_id": source["source_id"]}))["text"]
    )
    with pytest.raises(InputError, match=r"^tools: invalid arguments for ingest: "):
        await tools.dispatch("ingest", {"name": "empty"})


async def test_ingest_stamps_the_bound_view_and_refuses_to_leave_it() -> None:
    memory = await _memory()
    tools = MemoryTools(
        memory, corpus_id="acme", filters={"metadata.line": "A", "metadata.site.zone": 3}
    )
    source = (
        await _call(
            tools,
            "ingest",
            {
                "name": "wiring",
                "text": "The wiring loom is checked monthly.",
                "metadata": {"rev": 2},
            },
        )
    )["source"]
    assert source["metadata"] == {"rev": 2, "line": "A", "site": {"zone": 3}}
    assert source["source_id"] in {
        s["source_id"] for s in (await _call(tools, "list_sources", {}))["sources"]
    }
    assert [
        h["source_id"] for h in (await _call(tools, "search", {"query": "wiring loom"}))["hits"]
    ][0] == source["source_id"]
    with pytest.raises(
        InputError, match=r"^filters: metadata must agree with the bound filter 'metadata.line'"
    ):
        await _call(
            tools,
            "ingest",
            {"name": "line b", "text": "Line B runs nights.", "metadata": {"line": "B"}},
        )
    with pytest.raises(
        InputError,
        match=r"^filters: metadata must agree with the bound filter 'metadata.site.zone'",
    ):
        await _call(
            tools, "ingest", {"name": "zone", "text": "Zone north.", "metadata": {"site": "north"}}
        )
    assert sorted(s.name for s in (await memory.list_sources("acme"))[0]) == [
        "manual",
        "other",
        "wiring",
    ]


async def test_ingest_lands_only_in_the_pinned_corpus() -> None:
    memory = await _memory()
    tools = MemoryTools(memory, corpus_id="tenant-b")
    source = (await _call(tools, "ingest", {"name": "secret", "text": "The MARKER valve is red."}))[
        "source"
    ]
    assert source["corpus_id"] == "tenant-b"
    assert [s.name for s in (await memory.list_sources("tenant-b"))[0]] == ["secret"]
    assert "secret" not in {s.name for s in (await memory.list_sources("acme"))[0]}
    hits = (
        await _call(MemoryTools(memory, corpus_id="acme"), "search", {"query": "MARKER valve"})
    )["hits"]
    assert all(
        h.get("source_id") != source["source_id"] and "MARKER" not in h["text"] for h in hits
    )


async def test_search_and_recall_order_by_a_judgement_and_show_it() -> None:
    def answers(state: Any, questions: Any) -> dict[str, Answer]:
        text = state["passage"]["text"]
        evidence = 0.9 if "monthly" in text or "Tuesday" in text else 0.1
        return {
            name: CheckAnswer(probability=evidence if name == "evidence" else 0.3)
            for name in ("relevant", "evidence", "contradicts", "injection")
        }

    memory = await _memory(decision=Fixed(answers))
    await memory.remember("acme", "Ana prefers the Monday slot.")
    await memory.remember("acme", "Ana prefers the Tuesday slot.")
    tools = MemoryTools(memory, corpus_id="acme")
    search = next(d for d in tools.definitions() if d["name"] == "search")
    assert search["input_schema"]["properties"]["order_by"]
    found = await _call(
        tools, "search", {"query": "pump seal", "kinds": ["chunk"], "order_by": "evidence"}
    )
    assert "monthly" in found["hits"][0]["text"]
    assert [h["judge"]["evidence"] for h in found["hits"]] == [0.9, 0.1]
    assert set(found["hits"][0]["judge"]) == {"relevant", "evidence", "contradicts", "injection"}
    recalled = await _call(tools, "recall", {"query": "Ana slot", "order_by": "evidence"})
    assert (
        "Tuesday" in recalled["hits"][0]["text"] and recalled["hits"][0]["judge"]["evidence"] == 0.9
    )
    fused = await _call(tools, "search", {"query": "pump seal", "kinds": ["chunk"]})
    assert all("judge" not in h for h in fused["hits"])
    with pytest.raises(InputError):
        await tools.dispatch("search", {"query": "pump seal", "order_by": "kindness"})
    plain = await _memory()
    search = next(
        d for d in MemoryTools(plain, corpus_id="acme").definitions() if d["name"] == "search"
    )
    assert "order_by" not in search["input_schema"]["properties"]
