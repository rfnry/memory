from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.graph.extraction import extract, schema_with_vocabulary
from rfnry_memory.graph.identity import SAME, Staging, resolve
from rfnry_memory.memory import Memory
from rfnry_memory.models import Decided, EntityIn, RelationIn, ScoreAnswer, Section, Usage, Window
from rfnry_memory.storage.inmemory import InMemoryStorage

from .fakes import Fixed, HashEmbedding, Scripted, structured_reply

EXTRACTED = {
    "entities": [{"name": "V-101", "type": "valve"}, {"name": "P-3", "type": "pump"}],
    "relations": [
        {"from_name": "V-101", "to_name": "P-3", "type": "CONNECTS_TO"},
        {"from_name": "P-3", "to_name": "V-101", "type": "INVENTED"},
    ],
}


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
    ingested = await memory.ingest("acme", name="sheet", text="Valve V-101 feeds pump P-3.")
    assert (ingested.entities, ingested.relations) == (2, 1)
    hood = await memory.neighbors("acme", "V-101")
    assert {e.name for e in hood.entities} == {"V-101", "P-3"} and hood.relations[0].type == "CONNECTS_TO"


async def test_an_extracted_relation_to_an_unknown_name_is_dropped_even_beside_supplied_facts() -> None:
    llm = Scripted(structured_reply({
        "entities": [{"name": "V-101", "type": "valve"}],
        "relations": [{"from_name": "V-101", "to_name": "Ghost", "type": "CONNECTS_TO"}],
    }), structured_reply({"entities": [], "relations": []}))
    memory = Memory(InMemoryStorage(), generation=llm, relation_types=["CONNECTS_TO"])
    await memory.initialize()
    ingested = await memory.ingest("acme", name="sheet", text="Valve V-101.", entities=[EntityIn(name="P-3", type="pump")])
    assert (ingested.entities, ingested.relations) == (2, 0)
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
    assert ingested.entities == 0 and ingested.usage == Usage(input=30)
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
    assert ingested.entities == 0 and len(judge.calls) == 1
    assert (await storage.count_corpus("acme")).entities == 1


async def test_the_most_probable_level_names_the_outcome() -> None:
    storage = InMemoryStorage()
    await storage.initialize(dimension=None)
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.remember("acme", "x", entities=[EntityIn(name="Acme Corp", type="company")])
    related = Fixed({"same": ScoreAnswer(score=1.2, probabilities=[0.2, 0.45, 0.35])})
    staging = Staging("acme")
    entity, created, _ = await resolve(storage, related, staging, EntityIn(name="Acme Corp Labs", type="company"), "p")
    assert created and entity.name == "Acme Corp Labs" and staging.write().entities == [entity]
    assert (await storage.count_corpus("acme")).entities == 1


async def test_identity_is_exact_without_a_decision_model_and_new_when_different() -> None:
    storage = InMemoryStorage()
    await storage.initialize(dimension=None)
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.remember("acme", "x", entities=[EntityIn(name="Acme Corp", type="company")])
    await memory.remember("acme", "y", entities=[EntityIn(name="ACME Corporation", type="company")])
    assert (await storage.count_corpus("acme")).entities == 2
    different = Fixed({"same": ScoreAnswer(score=0.1, probabilities=[0.9, 0.05, 0.05])})
    entity, created, usage = await resolve(storage, different, Staging("acme"), EntityIn(name="Acme Labs", type="company"), "Acme Labs is a startup.")
    assert created and entity.name == "Acme Labs" and usage == Usage()
    assert isinstance(SAME.levels, list) and len(SAME.levels) == 3


async def test_supersede_extracts_the_new_story_like_remember() -> None:
    llm = Scripted(structured_reply({"entities": [{"name": "Ana", "type": "person"}], "relations": []}),
                   structured_reply(EXTRACTED))
    memory = Memory(InMemoryStorage(), embedding=HashEmbedding(), generation=llm, relation_types=["CONNECTS_TO"])
    await memory.initialize()
    story = await memory.remember("acme", "Ana checks the feed line.")
    _, current = await memory.supersede("acme", story.story_id, "Valve V-101 feeds pump P-3.")
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
    ingested = await memory.ingest("acme", name="press", sections=[Section(text="Acme Corp signed."), Section(text="ACME Corporation grew.")])
    assert ingested.entities == 1
    [entity] = await storage.entities_named("acme", "acme corporation")
    assert entity.name == "Acme Corp" and entity.aliases == ("ACME Corporation",)
    assert len(await storage.mentions("acme", [entity.entity_id], kinds=("chunk",), filters=None, window=Window(), subject=None)) == 2


async def test_a_supplied_relation_to_a_name_of_several_types_asks_for_the_type() -> None:
    memory = Memory(InMemoryStorage(), pillars={"text", "graph"})
    await memory.initialize()
    await memory.ingest("acme", name="bom", text="Part P-3 is a gasket.", entities=[EntityIn(name="P-3", type="part")])
    story = await memory.remember("acme", "Ana runs the line.", entities=[EntityIn(name="Ana", type="person")])
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
    ingested = await memory.ingest("acme", name="sheet", text="Valve V-1 feeds pump P-3.", entities=facts,
                                   relations=[feeds.model_copy(update={"to_type": "Pump", "from_type": "valve"})])
    assert (ingested.entities, ingested.relations) == (2, 1)
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


async def test_an_extracted_relation_to_a_name_of_several_types_is_dropped() -> None:
    llm = Scripted(structured_reply({"entities": [], "relations": []}), structured_reply({
        "entities": [{"name": "V-1", "type": "valve"}, {"name": "P-3", "type": "pump"}],
        "relations": [{"from_name": "V-1", "to_name": "P-3", "type": "FEEDS"}],
    }))
    memory = Memory(InMemoryStorage(), generation=llm, relation_types=["FEEDS"])
    await memory.initialize()
    await memory.ingest("acme", name="bom", text="Part P-3 is a gasket.", entities=[EntityIn(name="P-3", type="part")])
    ingested = await memory.ingest("acme", name="sheet", text="Valve V-1 feeds pump P-3.")
    assert (ingested.entities, ingested.relations) == (2, 0)
    assert sorted(e.type for e in await memory.storage.entities_named("acme", "p-3")) == ["part", "pump"]


async def test_a_write_counts_what_extraction_and_identity_cost() -> None:
    spent = Usage(input=100, output=20)
    llm = Scripted(structured_reply({"entities": [{"name": "Acme Corp", "type": "company"}], "relations": []}).model_copy(update={"usage": spent}),
                   structured_reply({"entities": [{"name": "ACME Corporation", "type": "company"}], "relations": []}).model_copy(update={"usage": spent}),
                   structured_reply({"entities": [], "relations": []}))
    judge = Fixed(lambda s, q: Decided(answers={"same": ScoreAnswer(score=0.1, probabilities=[0.9, 0.05, 0.05])}, usage=Usage(input=30)))
    memory = Memory(InMemoryStorage(), generation=llm, decision=judge, relation_types=["OWNS"])
    await memory.initialize()
    first = await memory.ingest("acme", name="a", text="Acme Corp signed.")
    assert first.usage == spent
    second = await memory.ingest("acme", name="b", sections=[Section(text="ACME Corporation grew."), Section(text="Nothing here.")])
    assert second.usage == spent + Usage(input=30)
