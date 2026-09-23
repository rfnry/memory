from datetime import UTC, datetime
from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.graph.extraction import extract, schema_with_vocabulary
from rfnry_memory.graph.identity import SAME, Staging, resolve
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    Decided,
    Entity,
    EntityIn,
    RelationIn,
    Remembered,
    ScoreAnswer,
    Section,
    Story,
    Usage,
    Window,
)
from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.storage.storage import GraphWrite

from .fakes import Fixed, HashEmbedding, Scripted, structured_reply

EXTRACTED = {
    "entities": [{"name": "V-101", "type": "valve"}, {"name": "P-3", "type": "pump"}],
    "relations": [
        {"from_name": "V-101", "from_type": "valve", "to_name": "P-3", "to_type": "pump", "type": "CONNECTS_TO"},
        {"from_name": "P-3", "from_type": "pump", "to_name": "V-101", "to_type": "valve", "type": "INVENTED"},
    ],
}


async def _graph(memory: Memory) -> tuple[int, int]:
    residue = await memory.storage.count_corpus("acme")
    return residue.entities, residue.relations


def test_schema_constrains_relation_types() -> None:
    schema = schema_with_vocabulary(["CONNECTS_TO", "PART_OF"])
    assert schema["$defs"]["ExtractedRelation"]["properties"]["type"]["enum"] == ["CONNECTS_TO", "PART_OF"]


async def test_extract_drops_relations_outside_the_vocabulary() -> None:
    llm = Scripted(structured_reply(EXTRACTED))
    extraction, _ = await extract(llm, "Valve V-101 feeds pump P-3.", ["CONNECTS_TO"])
    assert [e.name for e in extraction.entities] == ["V-101", "P-3"]
    assert [r.type for r in extraction.relations] == ["CONNECTS_TO"]
    assert "Valve V-101 feeds pump P-3." in str(llm.calls[0][1][0].content)


async def test_ingest_extracts_and_links() -> None:
    llm = Scripted(structured_reply(EXTRACTED))
    memory = Memory(InMemoryStorage(), embedding=HashEmbedding(), generation=llm, relation_types=["CONNECTS_TO"])
    await memory.initialize()
    await memory.ingest("acme", name="sheet", text="Valve V-101 feeds pump P-3.")
    assert await _graph(memory) == (2, 1)
    hood = await memory.neighbors("acme", "V-101")
    assert {e.name for e in hood.entities} == {"V-101", "P-3"} and hood.relations[0].type == "CONNECTS_TO"


async def test_an_extracted_relation_to_an_unknown_name_is_dropped_even_beside_supplied_facts() -> None:
    llm = Scripted(structured_reply({
        "entities": [{"name": "V-101", "type": "valve"}],
        "relations": [{"from_name": "V-101", "from_type": "valve", "to_name": "Ghost", "to_type": "ghost", "type": "CONNECTS_TO"}],
    }), structured_reply({"entities": [], "relations": []}))
    memory = Memory(InMemoryStorage(), generation=llm, relation_types=["CONNECTS_TO"])
    await memory.initialize()
    await memory.ingest("acme", name="sheet", text="Valve V-101.", entities=[EntityIn(name="P-3", type="pump")])
    assert await _graph(memory) == (2, 0)
    with pytest.raises(InputError):
        await memory.remember("acme", "z", relations=[RelationIn(from_name="V-101", to_name="Nobody", type="CONNECTS_TO")])


async def test_identity_merges_when_the_decision_says_same() -> None:
    storage = InMemoryStorage()
    await storage.initialize(dimension=None)
    llm = Scripted(structured_reply({"entities": [{"name": "Acme Corp", "type": "company"}], "relations": []}),
                   structured_reply({"entities": [{"name": "ACME Corporation", "type": "company"}], "relations": []}))

    def decide(state: Any, questions: Any) -> Decided:
        assert isinstance(state, dict)
        assert set(questions) == {"same"} and state["a"]["name"] == "ACME Corporation" and state["b"]["name"] == "Acme Corp"
        return Decided(answers={"same": ScoreAnswer(score=1.9, probabilities=[0.05, 0.05, 0.9])}, usage=Usage(input=30))

    memory = Memory(storage, generation=llm, decision=Fixed(decide), relation_types=["OWNS"])
    await memory.remember("acme", "Acme Corp signed the contract.")
    ingested = await memory.ingest("acme", name="press", text="ACME Corporation announced results.")
    assert ingested.usage == Usage(input=30)
    [entity] = await storage.entities_named("acme", "acme corp")
    assert entity.aliases == ("ACME Corporation",)
    assert (await storage.count_corpus("acme")).entities == 1


async def test_identity_with_a_decision_model_only_resolves_supplied_entities() -> None:
    storage = InMemoryStorage()
    await storage.initialize(dimension=None)
    judge = Fixed({"same": ScoreAnswer(score=1.8, probabilities=[0.1, 0.1, 0.8])})
    memory = Memory(storage, decision=judge, pillars={"text", "graph"})
    await memory.remember("acme", "x", entities=[EntityIn(name="Acme Corp", type="company")])
    ingested = await memory.ingest("acme", name="press", text="y", entities=[EntityIn(name="ACME Corporation", type="company")])
    assert len(judge.calls) == 1 and ingested.usage == Usage()
    assert (await storage.count_corpus("acme")).entities == 1


async def test_the_most_probable_level_names_the_outcome() -> None:
    storage = InMemoryStorage()
    await storage.initialize(dimension=None)
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.remember("acme", "x", entities=[EntityIn(name="Acme Corp", type="company")])
    related = Fixed({"same": ScoreAnswer(score=1.2, probabilities=[0.2, 0.45, 0.35])})
    staging = Staging("acme")
    entity, _ = await resolve(storage, related, staging, EntityIn(name="Acme Corp Labs", type="company"), "p")
    assert entity.name == "Acme Corp Labs" and staging.write().entities == [entity]
    assert (await storage.count_corpus("acme")).entities == 1


async def test_identity_is_exact_without_a_decision_model_and_new_when_different() -> None:
    storage = InMemoryStorage()
    await storage.initialize(dimension=None)
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.remember("acme", "x", entities=[EntityIn(name="Acme Corp", type="company")])
    await memory.remember("acme", "y", entities=[EntityIn(name="ACME Corporation", type="company")])
    assert (await storage.count_corpus("acme")).entities == 2
    different = Fixed({"same": ScoreAnswer(score=0.1, probabilities=[0.9, 0.05, 0.05])})
    staging = Staging("acme")
    entity, usage = await resolve(storage, different, staging, EntityIn(name="Acme Labs", type="company"), "Acme Labs is a startup.")
    assert entity.name == "Acme Labs" and staging.write().entities == [entity] and usage == Usage()
    assert isinstance(SAME.levels, list) and len(SAME.levels) == 3


async def test_supersede_extracts_the_new_story_like_remember() -> None:
    llm = Scripted(structured_reply({"entities": [{"name": "Ana", "type": "person"}], "relations": []}),
                   structured_reply(EXTRACTED))
    memory = Memory(InMemoryStorage(), embedding=HashEmbedding(), generation=llm, relation_types=["CONNECTS_TO"])
    await memory.initialize()
    story = (await memory.remember("acme", "Ana checks the feed line.")).story
    current = (await memory.supersede("acme", story.story_id, "Valve V-101 feeds pump P-3.")).story
    assert "Valve V-101 feeds pump P-3." in str(llm.calls[1][1][0].content)
    hood = await memory.neighbors("acme", "V-101")
    assert {e.name for e in hood.entities} == {"V-101", "P-3"} and [h.id for h in hood.passages] == [current.story_id]


async def test_identity_within_one_write_sees_what_that_write_staged() -> None:
    storage = InMemoryStorage()
    await storage.initialize(dimension=None)
    llm = Scripted(structured_reply({"entities": [{"name": "Acme Corp", "type": "company"}], "relations": []}),
                   structured_reply({"entities": [{"name": "ACME Corporation", "type": "company"}], "relations": []}))

    def decide(state: Any, questions: Any) -> Decided:
        assert state["b"] == {"name": "Acme Corp", "type": "company", "aliases": [], "passage": "Acme Corp signed."}
        return Decided(answers={"same": ScoreAnswer(score=1.9, probabilities=[0.05, 0.05, 0.9])})

    memory = Memory(storage, generation=llm, decision=Fixed(decide), relation_types=["OWNS"])
    await memory.ingest("acme", name="press", sections=[Section(text="Acme Corp signed."), Section(text="ACME Corporation grew.")])
    assert (await storage.count_corpus("acme")).entities == 1
    [entity] = await storage.entities_named("acme", "acme corporation")
    assert entity.name == "Acme Corp" and entity.aliases == ("ACME Corporation",)
    assert len(await storage.mentions("acme", [entity.entity_id], kinds=("chunk",), filters=None, window=Window(), subject=None)) == 2


async def test_a_supplied_relation_to_a_name_of_several_types_asks_for_the_type() -> None:
    memory = Memory(InMemoryStorage(), pillars={"text", "graph"})
    await memory.initialize()
    await memory.ingest("acme", name="bom", text="Part P-3 is a gasket.", entities=[EntityIn(name="P-3", type="part")])
    story = (await memory.remember("acme", "Ana runs the line.", entities=[EntityIn(name="Ana", type="person")])).story
    before = await memory.storage.count_corpus("acme")
    feeds = RelationIn(from_name="V-1", to_name="P-3", type="FEEDS")
    facts = [EntityIn(name="V-1", type="valve"), EntityIn(name="P-3", type="pump")]
    ambiguous = r"^memory: relation 'V-1' -> 'P-3' names 'P-3', which has several types: \['part', 'pump'\]; pass to_type"
    with pytest.raises(InputError, match=ambiguous):
        await memory.ingest("acme", name="sheet", text="Valve V-1 feeds pump P-3.", entities=facts, relations=[feeds])
    with pytest.raises(InputError, match=ambiguous):
        await memory.remember("acme", "V-1 feeds P-3.", entities=facts, relations=[feeds])
    with pytest.raises(InputError, match=ambiguous):
        await memory.supersede("acme", story.story_id, "V-1 feeds P-3.", entities=facts, relations=[feeds])
    with pytest.raises(InputError, match=r"^memory: relation 'V-1' -> 'P-3' names an unknown entity"):
        await memory.remember("acme", "V-1 feeds P-3.", entities=facts, relations=[feeds.model_copy(update={"to_type": "motor"})])
    assert await memory.storage.count_corpus("acme") == before
    assert (await memory.get_stories("acme", [story.story_id]))[0].valid_to is None
    await memory.ingest("acme", name="sheet", text="Valve V-1 feeds pump P-3.", entities=facts,
                        relations=[feeds.model_copy(update={"to_type": "Pump", "from_type": "valve"})])
    assert await _graph(memory) == (before.entities + 2, before.relations + 1)
    hood = await memory.neighbors("acme", "V-1")
    assert {(e.name, e.type) for e in hood.entities} == {("V-1", "valve"), ("P-3", "pump")}


async def test_one_write_may_supply_one_name_under_several_types() -> None:
    memory = Memory(InMemoryStorage(), pillars={"text", "graph"})
    await memory.initialize()
    facts = [EntityIn(name="P-3", type="part"), EntityIn(name="P-3", type="pump"), EntityIn(name="Ana", type="person")]
    with pytest.raises(InputError, match=r"\['part', 'pump'\]; pass to_type"):
        await memory.remember("acme", "Ana owns pump P-3.", entities=facts, relations=[RelationIn(from_name="Ana", to_name="P-3", type="OWNS")])
    await memory.remember("acme", "Ana owns pump P-3.", entities=facts,
                          relations=[RelationIn(from_name="Ana", to_name="P-3", to_type="pump", type="OWNS")])
    assert sorted(e.type for e in await memory.storage.entities_named("acme", "p-3")) == ["part", "pump"]
    assert [(e.name, e.type) for e in (await memory.neighbors("acme", "P-3", entity_type="part")).entities] == [("P-3", "part")]
    assert {e.type for e in (await memory.neighbors("acme", "P-3", entity_type="pump")).entities} == {"pump", "person"}


async def test_an_extracted_relation_names_its_endpoint_types_and_lands_on_them() -> None:
    def sheet(to_type: str) -> Any:
        return structured_reply({
            "entities": [{"name": "V-1", "type": "valve"}, {"name": "P-3", "type": "Pump"}],
            "relations": [{"from_name": "V-1", "from_type": "valve", "to_name": "P-3", "to_type": to_type, "type": "FEEDS"}],
        })

    llm = Scripted(structured_reply({"entities": [], "relations": []}), sheet("pump"), sheet("part"), sheet("motor"))
    memory = Memory(InMemoryStorage(), generation=llm, relation_types=["FEEDS"])
    await memory.initialize()
    await memory.ingest("acme", name="bom", text="Part P-3 is a gasket.", entities=[EntityIn(name="P-3", type="part")])
    await memory.ingest("acme", name="sheet", text="Valve V-1 feeds pump P-3.")
    assert await _graph(memory) == (3, 1)
    hood = await memory.neighbors("acme", "V-1")
    assert {(e.name, e.type) for e in hood.entities} == {("V-1", "valve"), ("P-3", "pump")}
    for other in ("sheet 2", "sheet 3"):
        await memory.ingest("acme", name=other, text="Valve V-1 feeds pump P-3.")
        assert (await _graph(memory))[1] == 1
    assert [(e.name, e.type) for e in (await memory.neighbors("acme", "P-3", entity_type="part")).entities] == [("P-3", "part")]


def test_the_schema_asks_for_each_endpoint_type() -> None:
    relation = schema_with_vocabulary(["FEEDS"])["$defs"]["ExtractedRelation"]
    assert {"from_type", "to_type"} <= set(relation["required"])


async def test_remember_and_supersede_count_what_extraction_and_identity_cost() -> None:
    spent = Usage(input=100, output=20)
    llm = Scripted(structured_reply({"entities": [{"name": "Acme Corp", "type": "company"}], "relations": []}).model_copy(update={"usage": spent}),
                   structured_reply({"entities": [{"name": "ACME Corporation", "type": "company"}], "relations": []}).model_copy(update={"usage": spent}))
    judge = Fixed(lambda s, q: Decided(answers={"same": ScoreAnswer(score=0.1, probabilities=[0.9, 0.05, 0.05])}, usage=Usage(input=30)))
    memory = Memory(InMemoryStorage(), generation=llm, decision=judge, relation_types=["OWNS"])
    await memory.initialize()
    remembered = await memory.remember("acme", "Acme Corp signed.")
    assert isinstance(remembered, Remembered) and remembered.usage == spent
    superseded = await memory.supersede("acme", remembered.story.story_id, "ACME Corporation grew.")
    assert superseded.usage == spent + Usage(input=30) and superseded.story.supersedes == remembered.story.story_id
    again = await memory.remember("acme", "acme corporation  grew.")
    assert again.story == superseded.story and again.usage == Usage() and len(llm.calls) == 2


async def test_identity_without_a_decision_model_finds_a_known_alias_of_the_same_type() -> None:
    storage = InMemoryStorage()
    known = Entity(corpus_id="acme", entity_id="e1", name="P-3", key="p-3", type="pump", aliases=("Pump three",))
    await storage.put_story("acme", Story(corpus_id="acme", story_id="s0", text="P-3 runs.", subject=None, valid_from=datetime.now(UTC),
                                          created_at=datetime.now(UTC), valid_to=None, supersedes=None, source_ids=(), metadata={}),
                            None, None, graph=GraphWrite(entities=[known], mentions=[("e1", "story", "s0")]))
    memory = Memory(storage, pillars={"text", "graph"})
    story = (await memory.remember("acme", "pump  THREE was serviced.", entities=[EntityIn(name="pump  THREE", type="pump")])).story
    [pump] = await storage.entities_named("acme", "p-3")
    assert (await storage.count_corpus("acme")).entities == 1
    assert ("story", story.story_id) in await storage.mentions("acme", [pump.entity_id], kinds=("story",), filters=None, window=Window(), subject=None)
    await memory.remember("acme", "Part three ships.", entities=[EntityIn(name="Pump three", type="part")])
    assert sorted(e.type for e in await storage.entities_named("acme", "pump three")) == ["part", "pump"]


async def test_a_supplied_relation_to_an_entity_merged_into_another_type_is_refused_before_writing() -> None:
    same = Fixed({"same": ScoreAnswer(score=1.9, probabilities=[0.05, 0.05, 0.9])})
    memory = Memory(InMemoryStorage(), decision=same, pillars={"text", "graph"})
    await memory.remember("acme", "Acme Corp signed.", entities=[EntityIn(name="Acme Corp", type="company")])
    before = await memory.storage.count_corpus("acme")
    with pytest.raises(InputError, match=r"^memory: relation 'Bo' -> 'ACME Corporation' names an unknown entity$"):
        await memory.remember("acme", "Bo joined ACME Corporation.",
                              entities=[EntityIn(name="ACME Corporation", type="organization"), EntityIn(name="Bo", type="person")],
                              relations=[RelationIn(from_name="Bo", to_name="ACME Corporation", to_type="organization", type="WORKS_AT")])
    assert await memory.storage.count_corpus("acme") == before


async def test_supplied_facts_land_on_the_chunks_that_name_them() -> None:
    memory = Memory(InMemoryStorage(), pillars={"text", "graph"})
    await memory.initialize()
    ingested = await memory.ingest("acme", name="sheet", sections=[
        Section(title="Feed", text="Valve V-101 feeds pump P-3."),
        Section(title="Drive", text="Motor M-7 drives pump p-3."),
    ], entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump"), EntityIn(name="M-7", type="motor"),
                 EntityIn(name="T-1", type="tank")],
       relations=[RelationIn(from_name="V-101", to_name="P-3", type="FEEDS"), RelationIn(from_name="M-7", to_name="P-3", type="DRIVES"),
                  RelationIn(from_name="V-101", to_name="M-7", type="NEAR")])
    feed, drive = (c.chunk_id for c in await memory.storage.source_chunks("acme", ingested.source.source_id))
    storage = memory.storage

    async def chunks_of(name: str) -> list[str]:
        [entity] = await storage.entities_named("acme", name)
        return [i for _, i in await storage.mentions("acme", [entity.entity_id], kinds=("chunk",), filters=None, window=Window(), subject=None)]

    assert await chunks_of("V-101") == [feed] and await chunks_of("M-7") == [drive]
    assert sorted(await chunks_of("P-3")) == sorted([feed, drive]) and await chunks_of("T-1") == [feed]
    [valve] = await storage.entities_named("acme", "V-101")
    [motor] = await storage.entities_named("acme", "M-7")
    evidence = {r.type: r.evidence_id for r in await storage.neighbors("acme", [valve.entity_id, motor.entity_id], relation_types=None)}
    assert evidence == {"FEEDS": feed, "DRIVES": drive, "NEAR": feed}
