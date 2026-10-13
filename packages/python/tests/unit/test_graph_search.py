import math
from datetime import UTC, datetime

import pytest

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Entity, EntityIn, Hits, Relation, Story
from rfnry_memory.rows import GraphWrite
from rfnry_memory.storage.postgres import PostgresStorage
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding

JAN = datetime(2026, 1, 1, tzinfo=UTC)


def _entity(
    name: str, kind: str = "thing", *, corpus: str = "acme", aliases: tuple[str, ...] = ()
) -> Entity:
    return Entity(
        corpus_id=corpus,
        entity_id=f"{kind}:{name.lower()}",
        name=name,
        key=name.lower(),
        type=kind,
        aliases=aliases,
    )


async def _story(
    memory: Memory,
    story_id: str,
    mentions: list[Entity],
    relations: tuple[tuple[Entity, str, Entity], ...] = (),
    *,
    corpus: str = "acme",
) -> None:
    story = Story(
        corpus_id=corpus,
        story_id=story_id,
        text=f"story {story_id}",
        subject=None,
        valid_from=JAN,
        valid_to=None,
        created_at=JAN,
        source_ids=(),
        metadata={},
    )
    ends = [entity for start, _, end in relations for entity in (start, end)]
    await memory.storage.put_story(
        corpus,
        story,
        None,
        None,
        graph=GraphWrite(
            entities=list({e.entity_id: e for e in [*mentions, *ends]}.values()),
            relations=[
                Relation(
                    corpus_id=corpus,
                    from_id=start.entity_id,
                    to_id=end.entity_id,
                    type=kind,
                    evidence_kind="story",
                    evidence_id=story_id,
                )
                for start, kind, end in relations
            ],
            mentions=[(e.entity_id, "story", story_id, e.name) for e in mentions],
        ),
    )


async def _graphed(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, pillars={"graph"})
    await memory.initialize()
    return memory


def _ids(hits: Hits) -> list[str]:
    return [h.id for h in hits.hits]


def _score(near: int, rarity: float) -> float:
    return 2.0**-near * (1 + rarity / (1 + rarity)) / 2


async def test_a_passage_naming_the_entity_outranks_one_a_hop_away_and_two_hops_away(
    storage: PostgresStorage,
) -> None:
    memory = await _graphed(storage)
    apollo, borealis, cygnus = _entity("Apollo"), _entity("Borealis"), _entity("Cygnus")
    await _story(memory, "p1", [cygnus])
    await _story(memory, "p2", [borealis])
    await _story(memory, "p3", [apollo])
    await _story(memory, "r1", [], ((apollo, "FEEDS", borealis),))
    await _story(memory, "r2", [], ((borealis, "FEEDS", cygnus),))
    hits = await memory.search("acme", "what about Apollo?", kinds=("story",))
    assert _ids(hits) == ["p3", "p2", "p1"]
    rarity = math.log(1 + 2.5 / 1.5)
    top = _score(0, rarity)
    assert [h.provenance["graph"] for h in hits.hits] == pytest.approx(
        [1.0, _score(1, rarity / 2) / top, _score(2, rarity / 4) / top]
    )


async def test_every_passage_naming_the_entity_outranks_every_passage_a_hop_away(
    storage: PostgresStorage,
) -> None:
    memory = await _graphed(storage)
    pump = _entity("P-3", "pump")
    for i in range(1, 7):
        await _story(memory, f"inspection-{i}", [pump])
    for i in range(1, 5):
        valve = _entity(f"V-10{i}", "valve")
        await _story(memory, f"repaint-{i}", [valve], ((pump, "FEEDS", valve),))
    hits = await memory.search(
        "acme", "status", entities=["P-3"], pillars={"graph"}, kinds=("story",), top_k=20
    )
    assert _ids(hits) == [f"inspection-{i}" for i in range(1, 7)] + [
        f"repaint-{i}" for i in range(1, 5)
    ]
    named, neighbour = math.log(1 + 4.5 / 6.5), math.log(1 + 9.5 / 1.5) / 2
    assert [h.provenance["graph"] for h in hits.hits] == pytest.approx(
        [1.0] * 6 + [_score(1, neighbour) / _score(0, named)] * 4
    )


async def test_a_direct_mention_in_a_small_corpus_outranks_a_two_hop_one_in_a_big_corpus(
    storage: PostgresStorage,
) -> None:
    memory = await _graphed(storage)
    await _story(memory, "small-apollo", [_entity("Apollo", corpus="small")], corpus="small")
    apollo, borealis, cygnus = (
        _entity(name, corpus="big") for name in ("Apollo", "Borealis", "Cygnus")
    )
    for i in range(7):
        await _story(memory, f"big-apollo-{i}", [apollo], corpus="big")
    await _story(memory, "big-cygnus", [cygnus], corpus="big")
    await _story(
        memory,
        "big-chain",
        [],
        ((apollo, "FEEDS", borealis), (borealis, "FEEDS", cygnus)),
        corpus="big",
    )
    hits = await memory.search(
        ["big", "small"], "status", entities=["Apollo"], kinds=("story",), top_k=20
    )
    assert [(h.corpus_id, h.id) for h in hits.hits] == [
        ("small", "small-apollo"),
        *(("big", f"big-apollo-{i}") for i in range(7)),
        ("big", "big-cygnus"),
    ]
    small, hub, far = math.log(1 + 0.5 / 1.5), math.log(1 + 1.5 / 7.5), math.log(1 + 7.5 / 1.5) / 4
    top = _score(0, small)
    assert [h.provenance["graph"] for h in hits.hits] == pytest.approx(
        [1.0] + [_score(0, hub) / top] * 7 + [_score(2, far) / top]
    )


async def test_a_passage_naming_two_reached_entities_outranks_one_naming_one(
    storage: PostgresStorage,
) -> None:
    memory = await _graphed(storage)
    apollo, borealis = _entity("Apollo"), _entity("Borealis")
    await _story(memory, "q1", [apollo])
    await _story(memory, "q2", [apollo, borealis])
    await _story(memory, "r1", [], ((apollo, "FEEDS", borealis),))
    hits = await memory.search("acme", "what about Apollo?", kinds=("story",))
    assert _ids(hits) == ["q2", "q1"]
    near, far = math.log(1 + 0.5 / 2.5), math.log(1 + 1.5 / 1.5) / 2
    assert hits.hits[1].provenance["graph"] == pytest.approx(
        _score(0, near) / _score(0, near + far)
    )


async def test_a_hub_entity_counts_less_than_a_rare_one_and_ties_order_by_id(
    storage: PostgresStorage,
) -> None:
    memory = await _graphed(storage)
    hub, rare = _entity("Hub"), _entity("Rare")
    for story_id in ("h3", "h1", "h4", "h2"):
        await _story(memory, story_id, [hub])
    await _story(memory, "z", [rare])
    hits = await memory.search("acme", "Hub or Rare", kinds=("story",))
    assert _ids(hits) == ["z", "h1", "h2", "h3", "h4"]
    common, uncommon = math.log(1 + 1.5 / 4.5), math.log(1 + 4.5 / 1.5)
    assert [h.provenance["graph"] for h in hits.hits] == pytest.approx(
        [1.0] + [_score(0, common) / _score(0, uncommon)] * 4
    )


async def test_relation_types_limit_the_walk(storage: PostgresStorage) -> None:
    memory = await _graphed(storage)
    apollo, borealis, cygnus = _entity("Apollo"), _entity("Borealis"), _entity("Cygnus")
    await _story(memory, "m-a", [apollo])
    await _story(memory, "m-b", [borealis])
    await _story(memory, "m-c", [cygnus])
    await _story(memory, "r1", [], ((apollo, "OWNS", borealis), (apollo, "FEEDS", cygnus)))
    every = await memory.search("acme", "status", entities=["Apollo"], kinds=("story",))
    assert _ids(every) == ["m-a", "m-b", "m-c"]
    owned = await memory.search(
        "acme", "status", entities=["Apollo"], relation_types=["owns"], kinds=("story",)
    )
    assert _ids(owned) == ["m-a", "m-b"]


async def test_named_entities_replace_the_names_spotted_in_the_query(
    storage: PostgresStorage,
) -> None:
    memory = await _graphed(storage)
    await _story(memory, "n-a", [_entity("Apollo")])
    await _story(memory, "n-c", [_entity("Cygnus")])
    assert _ids(await memory.search("acme", "Cygnus status", kinds=("story",))) == ["n-c"]
    named = await memory.search("acme", "Cygnus status", entities=["Apollo"], kinds=("story",))
    assert _ids(named) == ["n-a"] and named.unresolved == {}
    none = await memory.search("acme", "Cygnus status", entities=[], kinds=("story",))
    assert none.hits == []


async def test_names_resolve_by_name_alias_and_every_type_that_holds_them(
    storage: PostgresStorage,
) -> None:
    memory = await _graphed(storage)
    await _story(memory, "by-alias", [_entity("Apollo", aliases=("Moonshot",))])
    await _story(memory, "pump", [_entity("P-3", "pump")])
    await _story(memory, "part", [_entity("P-3", "part")])
    aliased = await memory.search("acme", "status", entities=["moonshot"], kinds=("story",))
    assert _ids(aliased) == ["by-alias"]
    both = await memory.search("acme", "status", entities=["p-3"], kinds=("story",))
    assert _ids(both) == ["part", "pump"] and both.unresolved == {}


async def test_unresolved_names_come_back_with_up_to_five_similar_names(
    storage: PostgresStorage,
) -> None:
    memory = await _graphed(storage)
    await _story(memory, "prog", [_entity("Apollo Program")])
    for i in range(1, 8):
        await _story(memory, f"x{i}", [_entity(f"Gemini {i}")])
    hits = await memory.search(
        "acme",
        "status",
        entities=["Apollo Programme", "Apollo Program", "Gemini X"],
        kinds=("story",),
    )
    assert _ids(hits) == ["prog"]
    assert hits.unresolved == {
        "Apollo Programme": ["Apollo Program"],
        "Gemini X": ["Gemini 1", "Gemini 2", "Gemini 3", "Gemini 4", "Gemini 5"],
    }


async def test_a_name_held_by_one_corpus_of_several_is_resolved(storage: PostgresStorage) -> None:
    memory = await _graphed(storage)
    await _story(memory, "in-b", [_entity("Apollo", corpus="b")], corpus="b")
    await _story(memory, "in-a", [_entity("Cygnus", corpus="a")], corpus="a")
    hits = await memory.search(["a", "b"], "status", entities=["Apollo"], kinds=("story",))
    assert [(h.corpus_id, h.id) for h in hits.hits] == [("b", "in-b")]
    assert hits.unresolved == {}
    missing = await memory.search(["a", "b"], "status", entities=["Apolo"], kinds=("story",))
    assert missing.hits == [] and missing.unresolved == {"Apolo": ["Apollo"]}


async def test_graph_search_needs_the_graph_pillar_and_storable_names(
    storage: PostgresStorage,
) -> None:
    plain = Memory(storage, embedding=HashEmbedding())
    needs = r"^memory: entities and relation_types need the graph pillar$"
    with pytest.raises(ConfigurationError, match=needs):
        await plain.search("acme", "status", entities=["Apollo"])
    with pytest.raises(ConfigurationError, match=needs):
        await plain.search("acme", "status", relation_types=["OWNS"])
    with pytest.raises(ConfigurationError, match=needs):
        await plain.recall("acme", "status", entities=["Apollo"])
    with pytest.raises(ConfigurationError, match=needs):
        await plain.recall("acme", "status", relation_types=["OWNS"])
    with pytest.raises(InputError, match=r"^memory: text must not contain a NUL character$"):
        await (await _graphed(storage)).search("acme", "status", entities=["Apo\x00llo"])


async def test_a_new_story_finds_related_stories_by_graph_rank(storage: PostgresStorage) -> None:
    memory = await _graphed(storage)
    apollo, borealis = _entity("Apollo"), _entity("Borealis")
    await _story(memory, "a-borealis", [borealis])
    await _story(memory, "b-both", [apollo, borealis], ((apollo, "FEEDS", borealis),))
    await _story(memory, "c-apollo", [apollo])
    written = await memory.remember(
        "acme", "Apollo slips to June.", entities=[EntityIn(name="Apollo", type="thing")]
    )
    assert _ids(written.related) == ["b-both", "c-apollo", "a-borealis"]


async def test_a_new_story_finds_the_stories_naming_its_entity_before_its_neighbours(
    storage: PostgresStorage,
) -> None:
    memory = await _graphed(storage)
    pump = _entity("P-3", "pump")
    for i in range(1, 5):
        await _story(memory, f"reading-{i}", [pump])
    for i in range(1, 5):
        valve = _entity(f"V-10{i}", "valve")
        await _story(memory, f"repaint-{i}", [valve], ((pump, "FEEDS", valve),))
    written = await memory.remember(
        "acme",
        "P-3 pressure is abnormal.",
        entities=[EntityIn(name="P-3", type="pump")],
        related=4,
    )
    assert _ids(written.related) == [f"reading-{i}" for i in range(1, 5)]


def _properties(tools: MemoryTools, name: str) -> set[str]:
    definition = next(d for d in tools.definitions() if d["name"] == name)
    return set(definition["input_schema"]["properties"])


async def test_the_search_tools_name_entities_only_where_the_graph_is_offered(
    storage: PostgresStorage,
) -> None:
    plain = Memory(storage, embedding=HashEmbedding())
    graphed = await _graphed(storage)
    bound = MemoryTools(graphed, corpus_id="acme", filters={"metadata.line": "A"})
    named = {"entities", "relation_types"}
    for name in ("search", "recall"):
        assert named <= _properties(MemoryTools(graphed, corpus_id="acme"), name)
        assert not named & _properties(MemoryTools(plain, corpus_id="acme"), name)
        assert not named & _properties(bound, name)
        with pytest.raises(InputError, match=r"^tools: entities is not offered here$"):
            await bound.dispatch(name, {"query": "status", "entities": ["Apollo"]})


async def test_the_search_tools_report_names_they_could_not_find(storage: PostgresStorage) -> None:
    memory = await _graphed(storage)
    await _story(memory, "prog", [_entity("Apollo Program")])
    tools = MemoryTools(memory, corpus_id="acme")
    found = await tools.dispatch(
        "search", {"query": "status", "entities": ["Apollo Programme", "Apollo Program"]}
    )
    assert isinstance(found, dict) and [h["id"] for h in found["hits"]] == ["prog"]
    assert found["unresolved"] == {"Apollo Programme": ["Apollo Program"]}
    recalled = await tools.dispatch(
        "recall", {"query": "status", "entities": ["Apollo Program"], "relation_types": ["owns"]}
    )
    assert isinstance(recalled, dict) and [h["id"] for h in recalled["hits"]] == ["prog"]
    assert "unresolved" not in recalled
