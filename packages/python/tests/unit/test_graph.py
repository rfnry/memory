from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.graph.extraction import extract, schema_with_vocabulary
from rfnry_memory.graph.identity import SAME, resolve
from rfnry_memory.memory import Memory
from rfnry_memory.models import Decided, EntityIn, RelationIn, ScoreAnswer, Usage
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
    extraction = await extract(llm, "Valve V-101 feeds pump P-3.", ["CONNECTS_TO"])
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
    entity = await storage.get_entity("acme", "acme corp")
    assert entity is not None and entity.aliases == ("ACME Corporation",)
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
    entity, created, _ = await resolve(storage, related, "acme", EntityIn(name="Acme Corp Labs", type="company"), "p")
    assert created and entity.name == "Acme Corp Labs"


async def test_identity_is_exact_without_a_decision_model_and_new_when_different() -> None:
    storage = InMemoryStorage()
    await storage.initialize(dimension=None)
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.remember("acme", "x", entities=[EntityIn(name="Acme Corp", type="company")])
    await memory.remember("acme", "y", entities=[EntityIn(name="ACME Corporation", type="company")])
    assert (await storage.count_corpus("acme")).entities == 2
    different = Fixed({"same": ScoreAnswer(score=0.1, probabilities=[0.9, 0.05, 0.05])})
    entity, created, usage = await resolve(storage, different, "acme", EntityIn(name="Acme Labs", type="company"), "Acme Labs is a startup.")
    assert created and entity.name == "Acme Labs" and usage == Usage()
    assert isinstance(SAME.levels, list) and len(SAME.levels) == 3
