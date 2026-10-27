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
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.rows import GraphWrite, alias_keys, name_key, reconciled
from rfnry_memory.tools import MemoryTools

from .fakes import Fixed, HashEmbedding, Scripted, structured_reply

EXTRACTED = {
    "entities": [{"name": "V-101", "type": "valve"}, {"name": "P-3", "type": "pump"}],
    "relations": [
        {
            "from_name": "V-101",
            "from_type": "valve",
            "to_name": "P-3",
            "to_type": "pump",
            "type": "CONNECTS_TO",
        },
        {
            "from_name": "P-3",
            "from_type": "pump",
            "to_name": "V-101",
            "to_type": "valve",
            "type": "INVENTED",
        },
    ],
}


async def _graph(memory: Memory) -> tuple[int, int]:
    residue = await memory.storage.count_corpus("acme")
    return residue.entities, residue.relations


def test_schema_constrains_relation_types() -> None:
    schema = schema_with_vocabulary(["CONNECTS_TO", "PART_OF"])
    assert schema["$defs"]["ExtractedRelation"]["properties"]["type"]["enum"] == [
        "CONNECTS_TO",
        "PART_OF",
    ]


async def test_extract_drops_relations_outside_the_vocabulary() -> None:
    llm = Scripted(structured_reply(EXTRACTED))
    extraction, _ = await extract(llm, "Valve V-101 feeds pump P-3.", ["CONNECTS_TO"])
    assert [e.name for e in extraction.entities] == ["V-101", "P-3"]
    assert [r.type for r in extraction.relations] == ["CONNECTS_TO"]
    assert "Valve V-101 feeds pump P-3." in str(llm.calls[0][1][0].content)


async def test_ingest_extracts_and_links(storage: PostgresStorage) -> None:
    llm = Scripted(structured_reply(EXTRACTED))
    memory = Memory(
        storage, embedding=HashEmbedding(), generation=llm, relation_types=["CONNECTS_TO"]
    )
    await memory.check()
    await memory.ingest("acme", name="sheet", text="Valve V-101 feeds pump P-3.")
    assert await _graph(memory) == (2, 1)
    hood = await memory.neighbors("acme", "V-101")
    assert {e.name for e in hood.entities} == {"V-101", "P-3"} and hood.relations[
        0
    ].type == "CONNECTS_TO"


async def test_an_extracted_relation_to_an_unknown_name_is_dropped_even_beside_supplied_facts(
    storage: PostgresStorage,
) -> None:
    llm = Scripted(
        structured_reply(
            {
                "entities": [{"name": "V-101", "type": "valve"}],
                "relations": [
                    {
                        "from_name": "V-101",
                        "from_type": "valve",
                        "to_name": "Ghost",
                        "to_type": "ghost",
                        "type": "CONNECTS_TO",
                    }
                ],
            }
        ),
        structured_reply({"entities": [], "relations": []}),
    )
    memory = Memory(storage, generation=llm, relation_types=["CONNECTS_TO"])
    await memory.check()
    await memory.ingest(
        "acme", name="sheet", text="Valve V-101.", entities=[EntityIn(name="P-3", type="pump")]
    )
    assert await _graph(memory) == (2, 0)
    with pytest.raises(InputError):
        await memory.remember(
            "acme",
            "z",
            relations=[RelationIn(from_name="V-101", to_name="Nobody", type="CONNECTS_TO")],
        )


async def test_identity_merges_when_the_decision_says_same(storage: PostgresStorage) -> None:
    await storage.check(dimension=None, graph=True)
    llm = Scripted(
        structured_reply({"entities": [{"name": "Acme Corp", "type": "company"}], "relations": []}),
        structured_reply(
            {"entities": [{"name": "ACME Corporation", "type": "company"}], "relations": []}
        ),
    )

    def decide(state: Any, questions: Any) -> Decided:
        assert isinstance(state, dict)
        assert (
            set(questions) == {"same"}
            and state["a"]["name"] == "ACME Corporation"
            and state["b"]["name"] == "Acme Corp"
        )
        return Decided(
            answers={"same": ScoreAnswer(score=1.9, probabilities=[0.05, 0.05, 0.9])},
            usage=Usage(input=30),
        )

    memory = Memory(storage, generation=llm, decision=Fixed(decide), relation_types=["OWNS"])
    await memory.remember("acme", "Acme Corp signed the contract.")
    ingested = await memory.ingest("acme", name="press", text="ACME Corporation announced results.")
    assert ingested.usage == Usage(input=30)
    [entity] = (await storage.entities_named("acme", ["acme corp"]))[name_key("acme corp")]
    assert entity.aliases == ("ACME Corporation",)
    assert (await storage.count_corpus("acme")).entities == 1


async def test_identity_with_a_decision_model_only_resolves_supplied_entities(
    storage: PostgresStorage,
) -> None:
    await storage.check(dimension=None, graph=True)
    judge = Fixed({"same": ScoreAnswer(score=1.8, probabilities=[0.1, 0.1, 0.8])})
    memory = Memory(storage, decision=judge, pillars={"text", "graph"})
    await memory.remember("acme", "x", entities=[EntityIn(name="Acme Corp", type="company")])
    ingested = await memory.ingest(
        "acme", name="press", text="y", entities=[EntityIn(name="ACME Corporation", type="company")]
    )
    assert len(judge.calls) == 1 and ingested.usage == Usage()
    assert (await storage.count_corpus("acme")).entities == 1


async def test_the_most_probable_level_names_the_outcome(storage: PostgresStorage) -> None:
    await storage.check(dimension=None, graph=True)
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.remember("acme", "x", entities=[EntityIn(name="Acme Corp", type="company")])
    related = Fixed({"same": ScoreAnswer(score=1.2, probabilities=[0.2, 0.45, 0.35])})
    staging = Staging("acme")
    entity, _ = await resolve(
        storage, related, staging, EntityIn(name="Acme Corp Labs", type="company"), "p"
    )
    assert entity.name == "Acme Corp Labs" and staging.write().entities == [entity]
    assert (await storage.count_corpus("acme")).entities == 1


async def test_identity_is_exact_without_a_decision_model_and_new_when_different(
    storage: PostgresStorage,
) -> None:
    await storage.check(dimension=None, graph=True)
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.remember("acme", "x", entities=[EntityIn(name="Acme Corp", type="company")])
    await memory.remember("acme", "y", entities=[EntityIn(name="ACME Corporation", type="company")])
    assert (await storage.count_corpus("acme")).entities == 2
    different = Fixed({"same": ScoreAnswer(score=0.1, probabilities=[0.9, 0.05, 0.05])})
    staging = Staging("acme")
    entity, usage = await resolve(
        storage,
        different,
        staging,
        EntityIn(name="Acme Labs", type="company"),
        "Acme Labs is a startup.",
    )
    assert entity.name == "Acme Labs" and staging.write().entities == [entity] and usage == Usage()
    assert isinstance(SAME.levels, list) and len(SAME.levels) == 3


async def test_supersede_extracts_the_new_story_like_remember(storage: PostgresStorage) -> None:
    llm = Scripted(
        structured_reply({"entities": [{"name": "Ana", "type": "person"}], "relations": []}),
        structured_reply(EXTRACTED),
    )
    memory = Memory(
        storage, embedding=HashEmbedding(), generation=llm, relation_types=["CONNECTS_TO"]
    )
    await memory.check()
    story = (await memory.remember("acme", "Ana checks the feed line.")).story
    current = (await memory.supersede("acme", story.story_id, "Valve V-101 feeds pump P-3.")).story
    assert "Valve V-101 feeds pump P-3." in str(llm.calls[1][1][0].content)
    hood = await memory.neighbors("acme", "V-101")
    assert {e.name for e in hood.entities} == {"V-101", "P-3"} and [
        h.id for h in hood.passages
    ] == [current.story_id]


async def test_identity_within_one_write_sees_what_that_write_staged(
    storage: PostgresStorage,
) -> None:
    await storage.check(dimension=None, graph=True)
    llm = Scripted(
        structured_reply({"entities": [{"name": "Acme Corp", "type": "company"}], "relations": []}),
        structured_reply(
            {"entities": [{"name": "ACME Corporation", "type": "company"}], "relations": []}
        ),
    )

    def decide(state: Any, questions: Any) -> Decided:
        assert state["b"] == {
            "name": "Acme Corp",
            "type": "company",
            "aliases": [],
            "passage": "Acme Corp signed.",
        }
        return Decided(answers={"same": ScoreAnswer(score=1.9, probabilities=[0.05, 0.05, 0.9])})

    memory = Memory(storage, generation=llm, decision=Fixed(decide), relation_types=["OWNS"])
    await memory.ingest(
        "acme",
        name="press",
        sections=[Section(text="Acme Corp signed."), Section(text="ACME Corporation grew.")],
    )
    assert (await storage.count_corpus("acme")).entities == 1
    [entity] = (await storage.entities_named("acme", ["acme corporation"]))[
        name_key("acme corporation")
    ]
    assert entity.name == "Acme Corp" and entity.aliases == ("ACME Corporation",)
    assert (
        len(
            await storage.mentions(
                "acme",
                [entity.entity_id],
                kinds=("chunk",),
                filters=None,
                window=Window(),
                subject=None,
            )
        )
        == 2
    )


async def test_a_supplied_relation_to_a_name_of_several_types_asks_for_the_type(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.check()
    await memory.ingest(
        "acme",
        name="bom",
        text="Part P-3 is a gasket.",
        entities=[EntityIn(name="P-3", type="part")],
    )
    story = (
        await memory.remember(
            "acme", "Ana runs the line.", entities=[EntityIn(name="Ana", type="person")]
        )
    ).story
    before = await memory.storage.count_corpus("acme")
    feeds = RelationIn(from_name="V-1", to_name="P-3", type="FEEDS")
    facts = [EntityIn(name="V-1", type="valve"), EntityIn(name="P-3", type="pump")]
    ambiguous = r"^memory: relation 'V-1' -> 'P-3' names 'P-3', which has several types: \['part', 'pump'\]; pass to_type"
    with pytest.raises(InputError, match=ambiguous):
        await memory.ingest(
            "acme",
            name="sheet",
            text="Valve V-1 feeds pump P-3.",
            entities=facts,
            relations=[feeds],
        )
    with pytest.raises(InputError, match=ambiguous):
        await memory.remember("acme", "V-1 feeds P-3.", entities=facts, relations=[feeds])
    with pytest.raises(InputError, match=ambiguous):
        await memory.supersede(
            "acme", story.story_id, "V-1 feeds P-3.", entities=facts, relations=[feeds]
        )
    with pytest.raises(
        InputError, match=r"^memory: relation 'V-1' -> 'P-3' names an unknown entity"
    ):
        await memory.remember(
            "acme",
            "V-1 feeds P-3.",
            entities=facts,
            relations=[feeds.model_copy(update={"to_type": "motor"})],
        )
    assert await memory.storage.count_corpus("acme") == before
    assert (await memory.get_stories("acme", [story.story_id]))[0].valid_to is None
    await memory.ingest(
        "acme",
        name="sheet",
        text="Valve V-1 feeds pump P-3.",
        entities=facts,
        relations=[feeds.model_copy(update={"to_type": "Pump", "from_type": "valve"})],
    )
    assert await _graph(memory) == (before.entities + 2, before.relations + 1)
    hood = await memory.neighbors("acme", "V-1")
    assert {(e.name, e.type) for e in hood.entities} == {("V-1", "valve"), ("P-3", "pump")}


async def test_one_write_may_supply_one_name_under_several_types(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.check()
    facts = [
        EntityIn(name="P-3", type="part"),
        EntityIn(name="P-3", type="pump"),
        EntityIn(name="Ana", type="person"),
    ]
    with pytest.raises(InputError, match=r"\['part', 'pump'\]; pass to_type"):
        await memory.remember(
            "acme",
            "Ana owns pump P-3.",
            entities=facts,
            relations=[RelationIn(from_name="Ana", to_name="P-3", type="OWNS")],
        )
    await memory.remember(
        "acme",
        "Ana owns pump P-3.",
        entities=facts,
        relations=[RelationIn(from_name="Ana", to_name="P-3", to_type="pump", type="OWNS")],
    )
    assert sorted(
        e.type for e in (await memory.storage.entities_named("acme", ["p-3"]))[name_key("p-3")]
    ) == [
        "part",
        "pump",
    ]
    assert [
        (e.name, e.type)
        for e in (await memory.neighbors("acme", "P-3", entity_type="part")).entities
    ] == [("P-3", "part")]
    assert {
        e.type for e in (await memory.neighbors("acme", "P-3", entity_type="pump")).entities
    } == {"pump", "person"}


async def test_an_extracted_relation_names_its_endpoint_types_and_lands_on_them(
    storage: PostgresStorage,
) -> None:
    def sheet(to_type: str) -> Any:
        return structured_reply(
            {
                "entities": [{"name": "V-1", "type": "valve"}, {"name": "P-3", "type": "Pump"}],
                "relations": [
                    {
                        "from_name": "V-1",
                        "from_type": "valve",
                        "to_name": "P-3",
                        "to_type": to_type,
                        "type": "FEEDS",
                    }
                ],
            }
        )

    llm = Scripted(
        structured_reply({"entities": [], "relations": []}),
        sheet("pump"),
        sheet("part"),
        sheet("motor"),
    )
    memory = Memory(storage, generation=llm, relation_types=["FEEDS"])
    await memory.check()
    await memory.ingest(
        "acme",
        name="bom",
        text="Part P-3 is a gasket.",
        entities=[EntityIn(name="P-3", type="part")],
    )
    await memory.ingest("acme", name="sheet", text="Valve V-1 feeds pump P-3.")
    assert await _graph(memory) == (3, 1)
    hood = await memory.neighbors("acme", "V-1")
    assert {(e.name, e.type) for e in hood.entities} == {("V-1", "valve"), ("P-3", "pump")}
    for other in ("sheet 2", "sheet 3"):
        await memory.ingest("acme", name=other, text="Valve V-1 feeds pump P-3.")
        assert (await _graph(memory))[1] == 1
    assert [
        (e.name, e.type)
        for e in (await memory.neighbors("acme", "P-3", entity_type="part")).entities
    ] == [("P-3", "part")]


def test_the_schema_asks_for_each_endpoint_type() -> None:
    relation = schema_with_vocabulary(["FEEDS"])["$defs"]["ExtractedRelation"]
    assert {"from_type", "to_type"} <= set(relation["required"])


async def test_remember_and_supersede_count_what_extraction_and_identity_cost(
    storage: PostgresStorage,
) -> None:
    spent = Usage(input=100, output=20)
    llm = Scripted(
        structured_reply(
            {"entities": [{"name": "Acme Corp", "type": "company"}], "relations": []}
        ).model_copy(update={"usage": spent}),
        structured_reply(
            {"entities": [{"name": "ACME Corporation", "type": "company"}], "relations": []}
        ).model_copy(update={"usage": spent}),
    )
    judge = Fixed(
        lambda s, q: Decided(
            answers={"same": ScoreAnswer(score=0.1, probabilities=[0.9, 0.05, 0.05])},
            usage=Usage(input=30),
        )
    )
    memory = Memory(storage, generation=llm, decision=judge, relation_types=["OWNS"])
    await memory.check()
    remembered = await memory.remember("acme", "Acme Corp signed.")
    assert isinstance(remembered, Remembered) and remembered.usage == spent
    superseded = await memory.supersede("acme", remembered.story.story_id, "ACME Corporation grew.")
    assert superseded.usage == spent + Usage(input=30)
    assert superseded.story.supersedes == (remembered.story.story_id,)
    again = await memory.remember("acme", "acme corporation  grew.")
    assert again.story == superseded.story and again.usage == Usage() and len(llm.calls) == 2


async def test_identity_without_a_decision_model_finds_a_known_alias_of_the_same_type(
    storage: PostgresStorage,
) -> None:
    known = Entity(
        corpus_id="acme",
        entity_id="e1",
        name="P-3",
        key="p-3",
        type="pump",
        aliases=("Pump three",),
    )
    await storage.put_story(
        "acme",
        Story(
            corpus_id="acme",
            story_id="s0",
            text="P-3 runs.",
            subject=None,
            valid_from=datetime.now(UTC),
            created_at=datetime.now(UTC),
            valid_to=None,
            source_ids=(),
            metadata={},
        ),
        None,
        None,
        graph=GraphWrite(entities=[known], mentions=[("e1", "story", "s0", "P-3")]),
    )
    memory = Memory(storage, pillars={"text", "graph"})
    story = (
        await memory.remember(
            "acme",
            "pump  THREE was serviced.",
            entities=[EntityIn(name="pump  THREE", type="pump")],
        )
    ).story
    [pump] = (await storage.entities_named("acme", ["p-3"]))[name_key("p-3")]
    assert (await storage.count_corpus("acme")).entities == 1
    assert (pump.entity_id, ("story", story.story_id)) in await storage.mentions(
        "acme", [pump.entity_id], kinds=("story",), filters=None, window=Window(), subject=None
    )
    await memory.remember(
        "acme", "Part three ships.", entities=[EntityIn(name="Pump three", type="part")]
    )
    assert sorted(
        e.type
        for e in (await storage.entities_named("acme", ["pump three"]))[name_key("pump three")]
    ) == [
        "part",
        "pump",
    ]


async def test_a_supplied_relation_to_an_entity_merged_into_another_type_is_refused_before_writing(
    storage: PostgresStorage,
) -> None:
    same = Fixed({"same": ScoreAnswer(score=1.9, probabilities=[0.05, 0.05, 0.9])})
    memory = Memory(storage, decision=same, pillars={"text", "graph"})
    await memory.remember(
        "acme", "Acme Corp signed.", entities=[EntityIn(name="Acme Corp", type="company")]
    )
    before = await memory.storage.count_corpus("acme")
    with pytest.raises(
        InputError, match=r"^memory: relation 'Bo' -> 'ACME Corporation' names an unknown entity$"
    ):
        await memory.remember(
            "acme",
            "Bo joined ACME Corporation.",
            entities=[
                EntityIn(name="ACME Corporation", type="organization"),
                EntityIn(name="Bo", type="person"),
            ],
            relations=[
                RelationIn(
                    from_name="Bo",
                    to_name="ACME Corporation",
                    to_type="organization",
                    type="WORKS_AT",
                )
            ],
        )
    assert await memory.storage.count_corpus("acme") == before


async def test_supplied_facts_land_on_the_chunks_that_name_them(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.check()
    ingested = await memory.ingest(
        "acme",
        name="sheet",
        sections=[
            Section(title="Feed", text="Valve V-101 feeds pump P-3."),
            Section(title="Drive", text="Motor M-7 drives pump p-3."),
        ],
        entities=[
            EntityIn(name="V-101", type="valve"),
            EntityIn(name="P-3", type="pump"),
            EntityIn(name="M-7", type="motor"),
            EntityIn(name="T-1", type="tank"),
        ],
        relations=[
            RelationIn(from_name="V-101", to_name="P-3", type="FEEDS"),
            RelationIn(from_name="M-7", to_name="P-3", type="DRIVES"),
            RelationIn(from_name="V-101", to_name="M-7", type="NEAR"),
        ],
    )
    feed, drive = (
        c.chunk_id for c in await memory.storage.source_chunks("acme", ingested.source.source_id)
    )

    async def chunks_of(name: str) -> list[str]:
        [entity] = (await storage.entities_named("acme", [name]))[name_key(name)]
        return [
            i
            for _, (_, i) in await storage.mentions(
                "acme",
                [entity.entity_id],
                kinds=("chunk",),
                filters=None,
                window=Window(),
                subject=None,
            )
        ]

    assert await chunks_of("V-101") == [feed]
    assert sorted(await chunks_of("M-7")) == sorted([feed, drive])
    assert sorted(await chunks_of("P-3")) == sorted([feed, drive]) and await chunks_of("T-1") == [
        feed
    ]
    [valve] = (await storage.entities_named("acme", ["V-101"]))[name_key("V-101")]
    [motor] = (await storage.entities_named("acme", ["M-7"]))[name_key("M-7")]
    evidence = {
        r.type: r.evidence_id
        for r in await storage.neighbors(
            "acme", [valve.entity_id, motor.entity_id], relation_types=None, window=Window()
        )
    }
    assert evidence == {"FEEDS": feed, "DRIVES": drive, "NEAR": feed}


async def test_neighbors_reads_relation_types_as_they_were_written(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.remember(
        "acme",
        "Ana owns P-3.",
        entities=[EntityIn(name="Ana", type="person"), EntityIn(name="P-3", type="pump")],
        relations=[RelationIn(from_name="Ana", to_name="P-3", type=" owns")],
    )
    everything = (await memory.neighbors("acme", "Ana")).relations
    assert [r.type for r in everything] == ["OWNS"]
    for asked in (["owns"], [" Owns "], ["OWNS"]):
        assert (await memory.neighbors("acme", "Ana", relation_types=asked)).relations == everything
    tools = MemoryTools(memory, corpus_id="acme")
    walked = await tools.dispatch("neighbors", {"entity": "Ana", "relation_types": ["owns"]})
    assert isinstance(walked, dict) and [r["type"] for r in walked["relations"]] == ["OWNS"]


def test_an_entity_keeps_only_the_names_its_remaining_passages_use() -> None:
    entity = Entity(
        corpus_id="a",
        entity_id="e1",
        name="John Smith",
        key="john smith",
        type="person",
        aliases=("J. Smith", "Johnny"),
    )
    assert reconciled(entity, [], related=False, taken=set()) is None
    assert reconciled(entity, [], related=True, taken=set()) == entity
    same = reconciled(entity, ["john  smith", "J. Smith"], related=False, taken=set())
    assert same is not None and (same.name, same.aliases) == ("John Smith", ("J. Smith",))
    renamed = reconciled(entity, ["Johnny", "J. Smith"], related=False, taken=set())
    assert renamed is not None and (renamed.name, renamed.key, renamed.aliases) == (
        "J. Smith",
        "j. smith",
        ("Johnny",),
    )
    blocked = reconciled(entity, ["J. Smith"], related=False, taken={"j. smith"})
    assert blocked is not None and (blocked.name, blocked.aliases) == ("John Smith", ("J. Smith",))


async def test_a_write_or_a_read_looks_its_names_up_in_one_call_however_many_it_names(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    looked = storage.entities_named

    async def counted(corpus_id: str, names: list[str]) -> dict[str, list[Entity]]:
        calls.append(corpus_id)
        return await looked(corpus_id, names)

    monkeypatch.setattr(storage, "entities_named", counted)
    memory = Memory(storage, pillars={"text", "graph"})
    made: dict[int, list[int]] = {}
    for count in (1, 50):
        names = [f"P-{count}-{i}" for i in range(count)]
        entities = [EntityIn(name=n, type="pump") for n in names]
        relations = [
            RelationIn(from_name=a, to_name=b, type="FEEDS")
            for a, b in zip(names, names[1:] + names[:1], strict=True)
        ]
        made[count] = []
        for write in (
            memory.ingest(
                "acme",
                name=f"m{count}",
                sections=[Section(text=" ".join(names[: count // 2 + 1])), Section(text="more")],
                entities=entities,
                relations=relations,
            ),
            memory.remember("acme", " ".join(names), entities=entities, relations=relations),
            memory.search(["acme", "other"], "pumps", entities=[*names, "Nobody"]),
        ):
            calls.clear()
            await write
            made[count].append(len(calls))
    assert made == {1: [1, 1, 2], 50: [1, 1, 2]}
    extracted = Scripted(
        *(
            structured_reply({"entities": [{"name": f"V-{i}", "type": "valve"} for i in range(n)]})
            for n in (1, 30)
        )
    )
    reading = Memory(storage, generation=extracted, relation_types=["FEEDS"])
    for text in ("V-0 opens.", "Thirty valves open."):
        calls.clear()
        await reading.ingest("acme", name=text, text=text)
        assert len(calls) == 1, text


def test_a_name_is_keyed_one_way_and_aliases_by_their_distinct_keys() -> None:
    assert [name_key(n) for n in ("  Pump\tP-3 ", "Line\x1cA", "Pump\xa0P-3", "ΟΔΟΣ")] == [
        "pump p-3",
        "line a",
        "pump p-3",
        "οδος",
    ]
    assert alias_keys(["Pump\xa0P-3", " ", "ΟΔΟΣ", "pump  p-3", "\x1c", "οδος"]) == [
        "pump p-3",
        "οδος",
    ]
    assert alias_keys([]) == []


async def test_a_call_with_keys_outside_the_schema_is_refused_and_retried() -> None:
    entity = {"name": "P-3", "type": "pump"}
    model = Scripted(
        structured_reply({"parameters": {"entities": [entity], "relations": []}}),
        structured_reply({"entities": [entity], "relations": []}),
    )
    found, _ = await extract(model, "Pump P-3 runs.", ["FEEDS"])
    assert [(e.name, e.type) for e in found.entities] == [("P-3", "pump")]
    assert len(model.calls) == 2
    feedback = model.calls[1][1][-1]
    assert feedback.role == "tool" and feedback.is_error
    assert str(feedback.content).startswith("invalid; fix and call again")
    schema = schema_with_vocabulary(["FEEDS"])
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["ExtractedEntity"]["additionalProperties"] is False
    assert schema["$defs"]["ExtractedRelation"]["additionalProperties"] is False


async def test_an_entity_or_relation_with_keys_outside_the_schema_is_refused_and_retried() -> None:
    pump, tank = {"name": "P-3", "type": "pump"}, {"name": "T-1", "type": "tank"}
    feeds = {"from_name": "P-3", "from_type": "pump", "to_name": "T-1", "to_type": "tank"}
    model = Scripted(
        structured_reply({"entities": [{**pump, "confidence": 0.9}, tank], "relations": []}),
        structured_reply(
            {"entities": [pump, tank], "relations": [{**feeds, "type": "FEEDS", "why": "stated"}]}
        ),
        structured_reply({"entities": [pump, tank], "relations": [{**feeds, "type": "FEEDS"}]}),
    )
    found, _ = await extract(model, "Pump P-3 feeds tank T-1.", ["FEEDS"])
    assert [(r.from_name, r.type, r.to_name) for r in found.relations] == [("P-3", "FEEDS", "T-1")]
    assert len(model.calls) == 3


async def test_a_write_whose_entity_a_delete_removed_meanwhile_brings_it_back(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.check()
    pump = EntityIn(name="P-3", type="pump")
    old = (await memory.ingest("acme", name="old", text="P-3 runs.", entities=[pump])).source
    [before] = (await memory.storage.entities_named("acme", ["P-3"]))[name_key("P-3")]
    original = storage.put_source

    async def after_a_delete(*args: Any, **kwargs: Any) -> Any:
        assert (await storage.delete_source("acme", old.source_id)).entities == 1
        assert (await storage.entities_named("acme", ["P-3"]))[name_key("P-3")] == []
        return await original(*args, **kwargs)

    storage.put_source = after_a_delete  # type: ignore[method-assign]
    await memory.ingest("acme", name="new", text="P-3 stops.", entities=[pump])
    assert (await storage.entities_named("acme", ["P-3"]))[name_key("P-3")] == [before]
