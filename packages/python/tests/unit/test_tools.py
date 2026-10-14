import json
from datetime import UTC, datetime, timedelta
from itertools import count
from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Answer, CheckAnswer, EntityIn, RelationIn, SourceIn
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.tools import MemoryTools, Tool, Tools

from .fakes import Fixed, HashEmbedding


async def _call(tools: MemoryTools, name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await tools.dispatch(name, args)
    assert isinstance(result, dict)
    return result


ALL = {
    "search",
    "recall",
    "expand",
    "read_source",
    "list_sources",
    "ingest",
    "remember",
    "supersede",
    "delete_story",
    "list_stories",
    "neighbors",
    "describe",
}


async def _memory(storage: PostgresStorage, **kwargs: Any) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding(), chunk_chars=200, overlap_chars=20, **kwargs)
    await memory.check()
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


async def test_definitions_are_gated_and_shaped(storage: PostgresStorage) -> None:
    plain = await _memory(storage)
    assert _names(MemoryTools(plain, corpus_id="acme")) == ALL - {"neighbors"}
    graph = await _memory(storage, pillars={"text", "vector", "graph"})
    assert _names(MemoryTools(graph, corpus_id="acme")) == ALL
    judged = await _memory(storage, decision=Fixed({}))
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


async def test_corpus_is_required_and_a_bound_filter_is_never_widened(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    with pytest.raises(ConfigurationError):
        MemoryTools(memory, corpus_id="")
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    result = await _call(tools, "search", {"query": "pump seal"})
    assert (
        result["hits"]
        and all(h["kind"] == "chunk" for h in result["hits"])
        and all("yearly" in h["text"] for h in result["hits"])
    )
    same = await _call(tools, "search", {"query": "pump seal", "filters": {"metadata.line": "A"}})
    assert same["hits"] == result["hits"]
    for widened in ("B", ["A", "B"]):
        with pytest.raises(
            InputError,
            match=r"^tools: filter 'metadata.line' can narrow its bound value 'A', never widen it$",
        ):
            await _call(
                tools, "search", {"query": "pump seal", "filters": {"metadata.line": widened}}
            )


async def test_story_verbs_round_trip_through_dispatch(storage: PostgresStorage) -> None:
    tools = MemoryTools(await _memory(storage), corpus_id="acme")
    remembered = await _call(
        tools,
        "remember",
        {
            "text": "Ana owns the wiring review.",
            "subject": "user:ana",
            "valid_from": "2026-03-05T00:00:00Z",
        },
    )
    assert set(remembered) == {"story", "related"}
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
        set(superseded) == {"story", "related"}
        and superseded["story"]["valid_from"].startswith("2026-04-01")
        and superseded["story"]["supersedes"] == [story_id]
    )
    listed = await _call(
        tools, "list_stories", {"subject": "user:ana", "as_of": "2026-03-10T00:00:00Z"}
    )
    assert [s["story_id"] for s in listed["stories"]] == [story_id]
    assert await _call(tools, "delete_story", {"story_id": superseded["story"]["story_id"]}) == {
        "deleted": superseded["story"]["story_id"]
    }


async def test_errors_raise_the_librarys_exceptions_as_any_tools_does(
    storage: PostgresStorage,
) -> None:
    async def now(args: dict[str, Any]) -> str:
        return "2026-09-22T00:00:00Z"

    tools = MemoryTools(await _memory(storage), corpus_id="acme")
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


async def test_read_and_list_sources_and_clipping(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    tools = MemoryTools(memory, corpus_id="acme", max_result_chars=1000)
    listed = await _call(tools, "list_sources", {})
    assert {s["name"] for s in listed["sources"]} == {"manual", "other"}
    source_id = listed["sources"][0]["source_id"]
    assert "pump seal" in (await _call(tools, "read_source", {"source_id": source_id}))["text"]
    await memory.ingest("acme", name="long", text="seal " * 3000)
    clipped = await _call(tools, "search", {"query": "seal", "top_k": 10})
    assert clipped["truncated"] is True and len(str(clipped)) <= 1400


async def test_a_neighbors_result_is_clipped_like_any_other(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
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


async def test_memory_tools_compose_with_other_tools_and_keep_their_posture(
    storage: PostgresStorage,
) -> None:
    async def now(args: dict[str, Any]) -> str:
        return "2026-09-22T00:00:00Z"

    combined = MemoryTools(await _memory(storage), corpus_id="acme").only(
        "search", "recall"
    ) + Tools([Tool("now", "UTC time", body=now)])
    assert combined.names == ["search", "recall", "now"]
    assert await combined.dispatch("now", {}) == "2026-09-22T00:00:00Z"
    result = await combined.dispatch("search", {"query": "pump seal"})
    assert isinstance(result, dict) and result["hits"]
    with pytest.raises(InputError):
        await combined.dispatch("search", {"filters": {"corpus_id": "x"}, "query": "q"})


async def test_tools_keep_limits_of_their_own(storage: PostgresStorage) -> None:
    tools = MemoryTools(
        await _memory(storage, pillars={"text", "vector", "graph"}), corpus_id="acme"
    )
    schemas = {d["name"]: d["input_schema"]["properties"] for d in tools.definitions()}
    for name in ("list_sources", "list_stories"):
        assert (schemas[name]["limit"]["default"], schemas[name]["limit"]["maximum"]) == (50, 500)
    for name in ("search", "recall"):
        assert (schemas[name]["top_k"]["default"], schemas[name]["top_k"]["maximum"]) == (10, 100)
    assert (schemas["neighbors"]["hops"]["minimum"], schemas["neighbors"]["hops"]["maximum"]) == (
        1,
        3,
    )
    with pytest.raises(InputError, match=r"^tools: invalid arguments for list_sources: "):
        await tools.dispatch("list_sources", {"limit": 501})
    with pytest.raises(InputError, match=r"^tools: invalid arguments for neighbors: "):
        await tools.dispatch("neighbors", {"entity": "pump", "hops": 4})


async def test_tools_refuse_a_duplicate_name_a_missing_body_and_an_unknown_name() -> None:
    bodiless = Tool("now", "UTC time")
    with pytest.raises(ConfigurationError, match=r"^tools: duplicate tool name 'now'$"):
        Tools([bodiless, bodiless])
    with pytest.raises(ConfigurationError, match=r"^tools: tool 'now' has no body$"):
        await Tools([bodiless]).dispatch("now", {})
    with pytest.raises(InputError, match=r"^tools: unknown tools \['ghost'\]$"):
        Tools([bodiless]).only("now", "ghost")


async def test_without_drops_the_named_tools_and_keeps_the_rest(storage: PostgresStorage) -> None:
    tools = MemoryTools(await _memory(storage), corpus_id="acme")
    kept = tools.without("remember", "delete_story", "ghost")
    assert kept.names == [n for n in tools.names if n not in {"remember", "delete_story"}]
    assert {d["name"] for d in kept.definitions()} == set(kept.names)
    with pytest.raises(InputError, match=r"^tools: unknown tool 'delete_story'$"):
        await kept.dispatch("delete_story", {"story_id": "missing"})
    found = await kept.dispatch("search", {"query": "pump seal"})
    assert isinstance(found, dict) and found["hits"]
    assert "delete_story" in tools and len(tools) == len(kept) + 2


async def test_recall_answers_now_and_history_marks_what_was_superseded(
    storage: PostgresStorage,
) -> None:
    tools = MemoryTools(await _memory(storage), corpus_id="acme")
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


async def test_the_tools_read_a_past_moment_and_what_memory_knew_then(
    storage: PostgresStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    apollo = EntityIn(name="Apollo", type="project")
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 1, 10, tzinfo=UTC))
    old = (
        await memory.remember(
            "acme",
            "The go-live is February 20.",
            valid_from=datetime(2026, 1, 1, tzinfo=UTC),
            entities=[apollo],
        )
    ).story
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 2, 10, tzinfo=UTC))
    new = (
        await memory.supersede(
            "acme",
            old.story_id,
            "The go-live moved to March 3.",
            valid_from=datetime(2026, 2, 1, tzinfo=UTC),
            entities=[apollo],
        )
    ).story
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 3, 1, tzinfo=UTC))
    tools = MemoryTools(memory, corpus_id="acme")
    then = {"known_at": "2026-02-05T00:00:00Z"}
    past = {"as_of": "2026-01-20T00:00:00Z"}
    for name in ("search", "recall"):
        for when in (then, past):
            found = await _call(tools, name, {"query": "go-live", **when})
            assert [h["id"] for h in found["hits"]] == [old.story_id]
        current = await _call(tools, name, {"query": "go-live"})
        assert [h["id"] for h in current["hits"]] == [new.story_id]
    for when in (then, past):
        listed = await _call(tools, "list_stories", when)
        assert [s["story_id"] for s in listed["stories"]] == [old.story_id]
        hood = await _call(tools, "neighbors", {"entity": "Apollo", **when})
        assert [p["id"] for p in hood["passages"]] == [old.story_id]


async def test_bound_filters_are_stamped_on_every_write_so_the_tools_see_their_own_writes(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
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


async def test_a_write_outside_the_bound_view_is_refused(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
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


async def test_a_filter_bound_to_several_values_reads_them_all_and_narrows_to_one(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.ingest(
        "acme", name="c", text="Line C runs the pump seal check.", metadata={"line": "C"}
    )
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": ["A", "B"]})
    assert _names(tools) == ALL - {"neighbors", "describe"}

    async def lines(filters: dict[str, Any] | None = None) -> list[str]:
        listed = await _call(tools, "list_sources", {"filters": filters} if filters else {})
        return sorted(s["metadata"]["line"] for s in listed["sources"])

    assert await lines() == ["A", "B"]
    assert await lines({"metadata.line": "B"}) == ["B"]
    assert await lines({"metadata.line": ["B", "A"]}) == ["A", "B"]
    hits = await _call(tools, "search", {"query": "pump seal", "filters": {"metadata.line": "A"}})
    assert hits["hits"] and all("yearly" in h["text"] for h in hits["hits"])
    for widened in ("C", ["A", "C"]):
        with pytest.raises(
            InputError,
            match=r"^tools: filter 'metadata.line' can narrow its bound value \['A', 'B'\], never widen it$",
        ):
            await lines({"metadata.line": widened})
    outside = (await memory.list_sources("acme", filters={"metadata.line": "C"}))[0][0]
    with pytest.raises(
        NotFoundError, match=rf"^tools: source '{outside.source_id}' was not found$"
    ):
        await _call(tools, "read_source", {"source_id": outside.source_id})


async def test_a_write_under_several_bound_values_names_one_of_them(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = count()
    monkeypatch.setattr(
        "rfnry_memory.memory._now",
        lambda: datetime(2026, 3, 1, tzinfo=UTC) + timedelta(minutes=next(clock)),
    )
    memory = await _memory(storage)
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": ["A", "B"]})
    with pytest.raises(
        InputError,
        match=r"^filters: metadata must name one of the values bound to 'metadata.line': \['A', 'B'\]$",
    ):
        await _call(tools, "remember", {"text": "Line A runs nights."})
    remembered = await _call(
        tools, "remember", {"text": "Line B runs nights.", "metadata": {"line": "B"}}
    )
    assert remembered["story"]["metadata"] == {"line": "B"}
    superseded = await _call(
        tools,
        "supersede",
        {"story_id": remembered["story"]["story_id"], "text": "Line B runs days."},
    )
    assert superseded["story"]["metadata"] == {"line": "B"}
    for stray in ("C", ["B", "C"]):
        with pytest.raises(
            InputError, match=r"^filters: metadata must agree with the bound filter 'metadata.line'"
        ):
            await _call(tools, "remember", {"text": "Line C runs.", "metadata": {"line": stray}})
    one = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": ["A"]})
    stamped = await _call(one, "remember", {"text": "Line A runs days."})
    assert stamped["story"]["metadata"] == {"line": "A"}


async def test_a_write_cannot_reach_past_a_single_bound_value_with_a_list(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    with pytest.raises(
        InputError, match=r"^filters: metadata must agree with the bound filter 'metadata.line'"
    ):
        await _call(
            tools, "remember", {"text": "Both lines stop.", "metadata": {"line": ["A", "B"]}}
        )
    assert (await memory.list_stories("acme"))[0] == []
    kept = await _call(tools, "remember", {"text": "Line A stops.", "metadata": {"line": ["A"]}})
    assert kept["story"]["metadata"] == {"line": ["A"]}


async def test_neighbors_takes_an_entity_type_when_a_name_is_ambiguous(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage, pillars={"text", "vector", "graph"})
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


async def test_a_duplicate_outside_the_bound_view_reveals_nothing(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
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


async def test_a_duplicate_outside_the_view_is_refused_before_the_judge_is_paid(
    storage: PostgresStorage,
) -> None:
    def judged(state: Any, questions: Any) -> dict[str, Answer]:
        return {name: CheckAnswer(probability=0.5) for name in questions}

    decision = Fixed(judged)
    memory = await _memory(storage, decision=decision)
    await memory.remember("acme", "Line B runs nights.", metadata={"line": "B"})
    await memory.remember("acme", "Line A runs nights too.", metadata={"line": "A"})
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    with pytest.raises(InputError, match=r"^tools: this text already exists outside your view$"):
        await _call(tools, "remember", {"text": "line b runs nights.", "judge": True})
    assert decision.calls == []
    past = await _call(
        tools,
        "remember",
        {
            "text": "Line B runs nights.",
            "valid_from": "2020-01-01T00:00:00Z",
            "valid_to": "2020-02-01T00:00:00Z",
        },
    )
    assert past["story"]["metadata"] == {"line": "A"}
    await memory.remember(
        "acme",
        "Line B ran days.",
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        valid_to=datetime(2020, 2, 1, tzinfo=UTC),
        metadata={"line": "B"},
    )
    with pytest.raises(InputError, match=r"^tools: this text already exists outside your view$"):
        await _call(
            tools,
            "remember",
            {
                "text": "line b ran days.",
                "valid_from": "2020-01-01T00:00:00",
                "valid_to": "2020-02-01T00:00:00Z",
                "judge": True,
            },
        )
    assert decision.calls == []
    later = await _call(
        tools,
        "remember",
        {
            "text": "Line B ran days.",
            "valid_from": "2020-03-01T00:00:00Z",
            "valid_to": "2020-04-01T00:00:00Z",
        },
    )
    assert later["story"]["metadata"] == {"line": "A"}


async def test_bound_and_unbound_tools_both_refuse_a_repeated_story_id(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    for tools in (
        MemoryTools(memory, corpus_id="acme"),
        MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"}),
    ):
        with pytest.raises(InputError, match=r"names a story more than once$"):
            await _call(tools, "supersede", {"story_id": ["ghost", "ghost"], "text": "Anything."})


async def test_related_rows_write_dates_as_the_story_does(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    await memory.remember(
        "acme",
        "Ana prefers the Tuesday slot.",
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_to=datetime(2030, 1, 1, tzinfo=UTC),
    )
    tools = MemoryTools(memory, corpus_id="acme")
    answer = await _call(tools, "remember", {"text": "Ana prefers the Wednesday slot."})
    [row] = answer["related"]
    assert (row["valid_from"], row["valid_to"]) == ("2026-01-01T00:00:00Z", "2030-01-01T00:00:00Z")
    assert answer["story"]["valid_from"].endswith("Z")


def test_descriptions_say_which_kinds_each_search_covers(storage: PostgresStorage) -> None:
    definitions = {
        d["name"]: d["description"]
        for d in MemoryTools(Memory(storage), corpus_id="acme").definitions()
    }
    assert "both sources (ingested documents) and stories" in definitions["search"]
    assert "stories only, not sources; use search for documents" in definitions["recall"]


async def test_a_bound_view_reads_nothing_outside_it(storage: PostgresStorage) -> None:
    memory = await _memory(storage, pillars={"text", "vector", "graph"})
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


async def test_ingest_files_a_source_the_same_tools_then_find(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
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


async def test_ingest_stamps_the_bound_view_and_refuses_to_leave_it(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
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


async def test_ingest_lands_only_in_the_pinned_corpus(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
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


async def test_search_and_recall_order_by_a_judgement_and_show_it(storage: PostgresStorage) -> None:
    def answers(state: Any, questions: Any) -> dict[str, Answer]:
        text = state["passage"]["text"]
        evidence = 0.9 if "monthly" in text or "Tuesday" in text else 0.1
        return {
            name: CheckAnswer(probability=evidence if name == "evidence" else 0.3)
            for name in ("relevant", "evidence", "contradicts", "injection")
        }

    memory = await _memory(storage, decision=Fixed(answers))
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
    assert set(found["hits"][0]["judge"]) == {"evidence"}
    recalled = await _call(tools, "recall", {"query": "Ana slot", "order_by": "evidence"})
    assert (
        "Tuesday" in recalled["hits"][0]["text"] and recalled["hits"][0]["judge"]["evidence"] == 0.9
    )
    fused = await _call(tools, "search", {"query": "pump seal", "kinds": ["chunk"]})
    assert all("judge" not in h for h in fused["hits"])
    with pytest.raises(InputError):
        await tools.dispatch("search", {"query": "pump seal", "order_by": "kindness"})
    plain = await _memory(storage)
    search = next(
        d for d in MemoryTools(plain, corpus_id="acme").definitions() if d["name"] == "search"
    )
    assert "order_by" not in search["input_schema"]["properties"]


async def test_supersede_carries_forward_metadata_the_host_wrote_wider_than_the_view(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    story = (
        await memory.remember("acme", "Both lines stop at noon.", metadata={"line": ["A", "C"]})
    ).story
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    superseded = await _call(
        tools, "supersede", {"story_id": story.story_id, "text": "Both lines stop at one."}
    )
    assert superseded["story"]["metadata"] == {"line": ["A", "C"]}
    with pytest.raises(
        InputError, match=r"^filters: metadata must agree with the bound filter 'metadata.line'"
    ):
        await _call(
            tools,
            "supersede",
            {
                "story_id": superseded["story"]["story_id"],
                "text": "Both lines stop at two.",
                "metadata": {"line": ["A", "C"]},
            },
        )


async def test_a_bound_supersede_of_several_keeps_metadata_only_when_they_agree(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    jan = datetime(2026, 1, 10, tzinfo=UTC)
    agree = [
        (
            await memory.remember(
                "acme", text, valid_from=jan, metadata={"line": "A", "team": "core"}
            )
        ).story.story_id
        for text in ("Ana prefers Tuesday.", "Ana prefers mornings.")
    ]
    differ = [
        (
            await memory.remember(
                "acme", text, valid_from=jan, metadata={"line": "A", "team": team}
            )
        ).story.story_id
        for text, team in (("Bo prefers Fridays.", "ops"), ("Cy prefers Mondays.", "core"))
    ]
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    feb = "2026-02-01T00:00:00Z"
    kept = await _call(
        tools,
        "supersede",
        {"story_id": agree, "text": "Ana prefers Tuesday mornings.", "valid_from": feb},
    )
    assert kept["story"]["metadata"] == {"line": "A", "team": "core"}
    stamped = await _call(
        tools,
        "supersede",
        {
            "story_id": differ,
            "text": "Bo and Cy share Mondays.",
            "valid_from": feb,
            "valid_to": "2030-01-01T00:00:00Z",
        },
    )
    assert stamped["story"]["metadata"] == {"line": "A"}
    assert stamped["story"]["valid_to"] == "2030-01-01T00:00:00Z"


async def test_a_source_reaches_the_model_without_its_text_or_digest(
    storage: PostgresStorage,
) -> None:
    tools = MemoryTools(await _memory(storage), corpus_id="acme")
    listed = await _call(tools, "list_sources", {})
    filed = await _call(tools, "ingest", {"name": "notes", "text": "Ground the panel first."})
    for source in [*listed["sources"], filed["source"]]:
        assert "text" not in source and "digest" not in source


async def test_supersede_takes_several_stories_through_the_tool(storage: PostgresStorage) -> None:
    tools = MemoryTools(await _memory(storage), corpus_id="acme")
    jan = "2026-01-01T00:00:00Z"
    a = await _call(tools, "remember", {"text": "Ana prefers Tuesday.", "valid_from": jan})
    b = await _call(tools, "remember", {"text": "Ana prefers mornings.", "valid_from": jan})
    ids = [a["story"]["story_id"], b["story"]["story_id"]]
    merged = await _call(
        tools,
        "supersede",
        {
            "story_id": ids,
            "text": "Ana prefers Tuesday mornings.",
            "valid_from": "2026-02-01T00:00:00Z",
        },
    )
    assert merged["story"]["supersedes"] == ids


async def test_remember_shows_related_stories_inside_the_view_only(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.remember("acme", "Ana prefers the Tuesday slot.", metadata={"line": "A"})
    await memory.remember("acme", "Ana prefers the Monday slot.", metadata={"line": "B"})
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    answer = await _call(tools, "remember", {"text": "Ana prefers the Wednesday slot."})
    assert set(answer) == {"story", "related"}
    assert [r["text"] for r in answer["related"]] == ["Ana prefers the Tuesday slot."]
    assert set(answer["related"][0]) == {"story_id", "text", "subject", "valid_from", "valid_to"}


async def test_a_judged_related_story_reaches_the_model_and_a_long_answer_is_clipped(
    storage: PostgresStorage,
) -> None:
    def judged(state: Any, questions: Any) -> dict[str, Answer]:
        return {name: CheckAnswer(probability=0.5) for name in questions}

    memory = await _memory(storage, decision=Fixed(judged))
    await memory.remember("acme", "Ana prefers the Tuesday slot.")
    tools = MemoryTools(memory, corpus_id="acme")
    answer = await _call(
        tools, "remember", {"text": "Ana prefers the Wednesday slot.", "judge": True}
    )
    [row] = answer["related"]
    assert set(row["judge"]) == {"relevant", "evidence", "contradicts", "injection"}
    for n in range(6):
        await memory.remember("acme", f"Ana noted item {n}: " + "the slot moved again. " * 12)
    short = MemoryTools(memory, corpus_id="acme", max_result_chars=1000)
    clipped = await _call(short, "remember", {"text": "Ana prefers the Friday slot.", "related": 8})
    assert clipped["truncated"] is True and 0 < len(clipped["related"]) < 8
    assert len(json.dumps(clipped)) <= 1000


async def test_related_stories_fill_from_the_view_when_closer_ones_sit_outside_it(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    for day in ("Monday", "Tuesday", "Thursday"):
        await memory.remember("acme", f"Ana prefers the {day} slot.", metadata={"line": "B"})
    await memory.remember("acme", "Ana sometimes takes Fridays.", metadata={"line": "A"})
    await memory.remember("acme", "Bo likes tea.", metadata={"line": "A"})
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    written = await _call(
        tools, "remember", {"text": "Ana prefers the Wednesday slot.", "related": 2}
    )
    assert {r["text"] for r in written["related"]} == {
        "Ana sometimes takes Fridays.",
        "Bo likes tea.",
    }
    moved = await _call(
        tools,
        "supersede",
        {"story_id": written["story"]["story_id"], "text": "Ana moved.", "related": 2},
    )
    assert {r["text"] for r in moved["related"]} == {
        "Ana sometimes takes Fridays.",
        "Bo likes tea.",
    }


async def test_the_tools_read_document_versions_inside_the_view(
    storage: PostgresStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", True)
    returns = {"metadata.doc": "returns"}

    def policy(text: str, line: str) -> SourceIn:
        return SourceIn(name="Returns policy", text=text, metadata={"doc": "returns", "line": line})

    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 1, 10, tzinfo=UTC))
    await memory.sync("acme", filters=returns, sources=[policy("Within 30 days.", "A")])
    await memory.sync(
        "acme",
        filters={"metadata.doc": "other"},
        sources=[
            SourceIn(name="Other", text="Within 7 days.", metadata={"doc": "other", "line": "B"})
        ],
    )
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 2, 10, tzinfo=UTC))
    await memory.sync("acme", filters=returns, sources=[policy("Within 60 days.", "A")])
    other = {"metadata.doc": "other"}
    await memory.sync(
        "acme",
        filters=other,
        sources=[
            SourceIn(name="Other", text="Within 14 days.", metadata={"doc": "other", "line": "B"})
        ],
    )
    retired_b = next(
        s
        for s in (await memory.list_sources("acme", filters=other, history=True))[0]
        if s.retired_at
    )
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    listed = await _call(tools, "list_sources", {"history": True})
    assert sorted(s["retired_at"] or "" for s in listed["sources"]) == ["", "2026-02-10T00:00:00Z"]
    then = await _call(tools, "list_sources", {"known_at": "2026-01-20T00:00:00Z"})
    assert [s["retired_at"] for s in then["sources"]] == [None]
    found = await _call(tools, "search", {"query": "days", "kinds": ["chunk"], "history": True})
    assert {h["text"]: h.get("retired_at") for h in found["hits"]} == {
        "Within 30 days.": "2026-02-10T00:00:00Z",
        "Within 60 days.": None,
    }
    assert {x["source_id"] for x in listed["sources"]}.isdisjoint({retired_b.source_id})
    assert all(h["source_id"] != retired_b.source_id for h in found["hits"])
    every = await _call(tools, "search", {"query": "days", "history": True, "top_k": 50})
    assert all(h.get("source_id") != retired_b.source_id for h in every["hits"])
    assert "Within 7 days." not in {h["text"] for h in every["hits"]}
    with pytest.raises(
        NotFoundError, match=rf"^tools: source '{retired_b.source_id}' was not found$"
    ):
        await tools.dispatch("read_source", {"source_id": retired_b.source_id})
    search = next(d for d in tools.definitions() if d["name"] == "search")
    assert "documents are read as they are now" not in search["description"]


def _order_by_choices(definition: dict[str, Any]) -> list[str]:
    choices = definition["input_schema"]["properties"]["order_by"]["anyOf"]
    return next(option["enum"] for option in choices if "enum" in option)


async def _reranked_memory(storage: PostgresStorage, decision: Fixed) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding(), decision=decision)
    await memory.check()
    await memory.ingest("acme", name="a", text="The pump runs all day.")
    await memory.ingest("acme", name="b", text="The gasket on the pump leaks.")
    return memory


async def test_tools_bound_to_a_reranker_ask_only_its_judgement(storage: PostgresStorage) -> None:
    def answers(state: Any, questions: Any) -> dict[str, Answer]:
        assert isinstance(state, dict)
        gasket = "gasket" in state["passage"]["text"]
        return {"relevant": CheckAnswer(probability=0.9 if gasket else 0.2)}

    decision = Fixed(answers)
    memory = await _reranked_memory(storage, decision)
    tools = MemoryTools(memory, corpus_id="acme", judge=("relevant",))
    search = next(d for d in tools.definitions() if d["name"] == "search")
    assert _order_by_choices(search) == ["relevant"]
    judged = await _call(tools, "search", {"query": "pump", "kinds": ["chunk"], "judge": True})
    assert all(set(row["judge"]) == {"relevant"} for row in judged["hits"])
    ordered = await _call(
        tools, "search", {"query": "pump", "kinds": ["chunk"], "order_by": "relevant"}
    )
    assert [row["source_name"] for row in ordered["hits"]] == ["b", "a"]
    assert all(set(q) == {"relevant"} for _, q in decision.calls)
    asked = len(decision.calls)
    with pytest.raises(InputError, match=r"^tools: order_by is one of relevant$"):
        await _call(tools, "search", {"query": "pump", "order_by": "evidence"})
    assert len(decision.calls) == asked


async def test_order_by_alone_asks_only_that_judgement(storage: PostgresStorage) -> None:
    decision = Fixed(
        lambda state, questions: {name: CheckAnswer(probability=0.5) for name in questions}
    )
    tools = MemoryTools(await _reranked_memory(storage, decision), corpus_id="acme")
    await _call(tools, "search", {"query": "pump", "order_by": "evidence"})
    assert decision.calls and all(set(q) == {"evidence"} for _, q in decision.calls)


@pytest.mark.parametrize("judge", [(), ("score",), ("relevant", "relevant"), (["relevant"],)])
async def test_tools_refuse_a_judge_naming_no_judgement_or_one_twice(
    storage: PostgresStorage, judge: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), decision=Fixed({}))
    with pytest.raises(
        ConfigurationError,
        match=r"^tools: judge names relevant, evidence, contradicts or injection, each once$",
    ):
        MemoryTools(memory, corpus_id="acme", judge=judge)
