from datetime import UTC, datetime
from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import CorpusDescription, EntityIn, RelationIn, Section, SourceIn
from rfnry_memory.storage.postgres import PostgresStorage
from rfnry_memory.tools import MemoryTools

from .fakes import Fixed, HashEmbedding

FEED = "V-101 feeds P-3."
RETURN = "P-3 returns to the tank."


async def _sheet(memory: Memory, corpus: str = "acme", **metadata: str) -> None:
    await memory.ingest(
        corpus,
        name="sheet",
        sections=[Section(text=FEED), Section(text=RETURN)],
        metadata=dict(metadata),
        entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump")],
        relations=[RelationIn(from_name="V-101", to_name="P-3", type="FEEDS")],
    )


def _graphed(storage: PostgresStorage) -> Memory:
    return Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})


async def test_hits_carry_the_entities_their_passages_mention(storage: PostgresStorage) -> None:
    memory = _graphed(storage)
    await _sheet(memory)
    hits = await memory.search("acme", "V-101 feeds P-3")
    named = {h.text: [(e.name, e.type) for e in h.entities] for h in hits.hits}
    assert named == {FEED: [("P-3", "pump"), ("V-101", "valve")], RETURN: [("P-3", "pump")]}
    hood = await memory.neighbors("acme", "V-101")
    assert {h.text: len(h.entities) for h in hood.passages} == {FEED: 2, RETURN: 1}
    start = next(h for h in hits.hits if h.text == FEED)
    beside = await memory.expand("acme", (start.kind, start.id), by="document")
    assert [[e.name for e in h.entities] for h in beside.hits] == [["P-3"]]
    plain = Memory(storage, embedding=HashEmbedding())
    await plain.ingest("acme", name="sheet", text=FEED)
    assert all(h.entities == [] for h in (await plain.search("acme", "feeds")).hits)


async def test_tool_rows_carry_entities_only_where_the_graph_is_offered(
    storage: PostgresStorage,
) -> None:
    memory = _graphed(storage)
    await _sheet(memory, line="A")

    async def rows(tools: MemoryTools) -> dict[str, Any]:
        found = await tools.dispatch("search", {"query": "V-101 feeds P-3"})
        assert isinstance(found, dict)
        return {h["text"]: h.get("entities") for h in found["hits"]}

    assert await rows(MemoryTools(memory, corpus_id="acme")) == {
        FEED: [{"name": "P-3", "type": "pump"}, {"name": "V-101", "type": "valve"}],
        RETURN: [{"name": "P-3", "type": "pump"}],
    }
    bound = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    assert await rows(bound) == {FEED: None, RETURN: None}


async def test_the_search_tools_offer_only_the_configured_pillars(storage: PostgresStorage) -> None:
    def offered(tools: MemoryTools, name: str) -> list[str]:
        definition = next(d for d in tools.definitions() if d["name"] == name)
        [items] = [
            option["items"]
            for option in definition["input_schema"]["properties"]["pillars"]["anyOf"]
            if "items" in option
        ]
        return list(items["enum"])

    plain = Memory(storage)
    embedded = Memory(storage, embedding=HashEmbedding())
    graphed = _graphed(storage)
    for name in ("search", "recall"):
        assert offered(MemoryTools(plain, corpus_id="acme"), name) == ["text"]
        assert offered(MemoryTools(embedded, corpus_id="acme"), name) == ["text", "vector"]
        assert offered(MemoryTools(graphed, corpus_id="acme"), name) == ["text", "vector", "graph"]
        bound = MemoryTools(graphed, corpus_id="acme", filters={"metadata.line": "A"})
        assert offered(bound, name) == ["text", "vector"]
    await embedded.ingest("acme", name="parts", text="Part 4471-B ships in boxes of ten.")
    await embedded.ingest("acme", name="notes", text="The boxes were stacked by the door.")
    tools = MemoryTools(embedded, corpus_id="acme")
    exact = await tools.dispatch("search", {"query": "4471", "pillars": ["text"]})
    assert isinstance(exact, dict) and set(exact["pillars"]) == {"text"}
    assert [h["text"] for h in exact["hits"]] == ["Part 4471-B ships in boxes of ten."]


async def test_describe_reports_what_a_corpus_holds_and_what_works(
    storage: PostgresStorage,
) -> None:
    memory = Memory(
        storage,
        embedding=HashEmbedding(),
        decision=Fixed({}),
        pillars={"text", "vector", "graph"},
    )
    await memory.set_history("acme", True)
    entities = [
        EntityIn(name="V-101", type="valve"),
        EntityIn(name="P-3", type="pump"),
        EntityIn(name="T-9", type="tank"),
    ]
    relations = [
        RelationIn(from_name="V-101", to_name="P-3", type="FEEDS"),
        RelationIn(from_name="P-3", to_name="T-9", type="FEEDS"),
    ]

    def sheet(rev: str) -> SourceIn:
        return SourceIn(
            name="sheet",
            sections=[
                Section(text=f"V-101 feeds P-3, {rev}."),
                Section(text=f"P-3 feeds T-9, {rev}."),
            ],
            metadata={"doc": "sheet"},
            entities=entities,
            relations=relations,
        )

    for rev in ("rev 1", "rev 2"):
        await memory.sync("acme", filters={"metadata.doc": "sheet"}, sources=[sheet(rev)])
    await memory.remember("acme", "P-3 hums.", related=0)
    serviced = await memory.remember(
        "acme", "P-3 was serviced.", valid_from=datetime(2026, 1, 1, tzinfo=UTC), related=0
    )
    await memory.supersede(
        "acme",
        serviced.story.story_id,
        "P-3 was serviced twice.",
        valid_from=datetime(2026, 2, 1, tzinfo=UTC),
        related=0,
    )
    assert await memory.describe("acme") == CorpusDescription(
        sources=1,
        chunks=2,
        stories=2,
        retired=1,
        entities=3,
        relations=4,
        entity_types={"pump": 1, "tank": 1, "valve": 1},
        relation_types={"FEEDS": 4},
        language="english",
        history=True,
        pillars=["graph", "text", "vector"],
        judge=True,
        expand=["document", "vector", "graph"],
    )
    plain = await Memory(storage).describe("other")
    assert (plain.sources, plain.stories, plain.entity_types, plain.history) == (0, 0, {}, False)
    assert (plain.pillars, plain.judge, plain.expand) == (["text"], False, ["document"])


async def test_the_describe_tool_describes_each_bound_corpus(storage: PostgresStorage) -> None:
    memory = _graphed(storage)
    await _sheet(memory, "hr")
    await memory.ingest("ops", name="note", text="The ops line runs nights.")
    tools = MemoryTools(memory, corpus_id=["hr", "ops"])
    schema = next(d for d in tools.definitions() if d["name"] == "describe")["input_schema"]
    assert schema["properties"]["corpus_id"]["enum"] == ["hr", "ops"]
    hr = await tools.dispatch("describe", {"corpus_id": "hr"})
    ops = await tools.dispatch("describe", {"corpus_id": "ops"})
    assert isinstance(hr, dict) and isinstance(ops, dict)
    assert (hr["chunks"], hr["entity_types"]) == (2, {"pump": 1, "valve": 1})
    assert (ops["chunks"], ops["entity_types"]) == (1, {})
    with pytest.raises(
        InputError, match=r"^tools: corpus_id can name only the corpora you may read: hr, ops$"
    ):
        await tools.dispatch("describe", {"corpus_id": "finance"})
    one = MemoryTools(memory, corpus_id="hr")
    alone = next(d for d in one.definitions() if d["name"] == "describe")["input_schema"]
    assert alone["properties"] == {}
    assert await one.dispatch("describe", {}) == hr
    bound = MemoryTools(memory, corpus_id="hr", filters={"metadata.line": "A"})
    assert "describe" not in bound


async def test_the_search_tools_refuse_a_pillar_they_do_not_offer(storage: PostgresStorage) -> None:
    plain = MemoryTools(Memory(storage), corpus_id="acme")
    bound = MemoryTools(_graphed(storage), corpus_id="acme", filters={"metadata.line": "A"})
    for name in ("search", "recall"):
        with pytest.raises(InputError, match=r"^tools: pillars are among text$"):
            await plain.dispatch(name, {"query": "x", "pillars": ["vector"]})
        with pytest.raises(InputError, match=r"^tools: pillars are among text, vector$"):
            await bound.dispatch(name, {"query": "x", "pillars": ["graph"]})
        with pytest.raises(InputError, match=r"^tools: pillars are among text$"):
            await plain.dispatch(name, {"query": "x", "pillars": []})


def test_the_tool_descriptions_say_how_to_get_past_each_dead_end(storage: PostgresStorage) -> None:
    tools = MemoryTools(_graphed(storage), corpus_id="acme")
    definitions = {d["name"]: d for d in tools.definitions()}
    expand = definitions["expand"]
    arguments = expand["input_schema"]["properties"]
    assert "retired_at" in arguments["history"]["description"]
    assert "history=true to expand" in arguments["history"]["description"]
    assert "past moment" in arguments["as_of"]["description"]
    assert "as memory knew them then" in arguments["known_at"]["description"]
    assert "top_k other stories about its subject nearest it in time" in expand["description"]
    assert "not by document from a chunk" in arguments["top_k"]["description"]
    assert "not by document" in arguments["kinds"]["description"]
    for name in ("search", "recall"):
        properties = definitions[name]["input_schema"]["properties"]
        assert "filters switch graph search off" in properties["pillars"]["description"]
        assert "filters switch graph search off" in properties["entities"]["description"]
        assert "similar names to retry with" in properties["around"]["description"]
    assert "relation rows" in definitions["describe"]["description"]
    assert "relations it holds" not in definitions["describe"]["description"]
