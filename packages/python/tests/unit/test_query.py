from datetime import UTC, datetime

import pytest

from rfnry_memory import BM25, CoverDensity, Memory, MemoryTools
from rfnry_memory.errors import InputError
from rfnry_memory.models import Entity, Relation, Story
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.rows import GraphWrite

from .fakes import HashEmbedding

JAN = datetime(2026, 1, 1, tzinfo=UTC)

KIT = "the pump seal kit"
YEARLY = "replace the pump seal yearly"
REVERSED = "a seal for the pump"
MOTOR = "the pump motor"


async def _filled(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    for name, text in (("kit", KIT), ("yearly", YEARLY), ("reversed", REVERSED), ("motor", MOTOR)):
        await memory.ingest("a", name=name, text=text)
    return memory


async def _matched(memory: Memory, query: str, match: str) -> set[str]:
    hits = await memory.search(
        "a",
        query,
        kinds=("chunk",),
        pillars={"text"},
        match=match,  # type: ignore[arg-type]
    )
    return {hit.text for hit in hits.hits}


async def test_match_decides_which_passages_the_words_find(storage: PostgresStorage) -> None:
    memory = await _filled(storage)
    assert await _matched(memory, "pump seal", "any") == {KIT, YEARLY, REVERSED, MOTOR}
    assert await _matched(memory, "pump seal", "all") == {KIT, YEARLY, REVERSED}
    assert await _matched(memory, "pump seal", "phrase") == {KIT, YEARLY}
    assert await _matched(memory, '"pump seal" -kit', "web") == {YEARLY}
    assert await _matched(memory, "motor or yearly", "web") == {YEARLY, MOTOR}


async def test_any_is_the_default_and_keeps_todays_reading(storage: PostgresStorage) -> None:
    memory = await _filled(storage)
    hits = await memory.search("a", "seal motor", kinds=("chunk",), pillars={"text"})
    assert {hit.text for hit in hits.hits} == {KIT, YEARLY, REVERSED, MOTOR}


async def test_a_strict_match_that_finds_nothing_still_answers_from_the_other_pillars(
    storage: PostgresStorage,
) -> None:
    memory = await _filled(storage)
    hits = await memory.search("a", "seal pump kit", kinds=("chunk",), match="phrase")
    assert hits.pillars["text"].returned == 0 and hits.pillars["text"].error is None
    assert hits.hits and all(set(hit.provenance) == {"vector"} for hit in hits.hits)


async def test_web_reads_any_input_without_an_error(storage: PostgresStorage) -> None:
    memory = await _filled(storage)
    for query in ('"pump seal', "-", "or", "pump -", '""', "--seal"):
        hits = await memory.search("a", query, kinds=("chunk",), pillars={"text"}, match="web")
        assert hits.pillars["text"].error is None, query


async def test_a_query_of_stop_words_matches_nothing_in_any_mode(storage: PostgresStorage) -> None:
    memory = await _filled(storage)
    for match in ("any", "all", "phrase", "web"):
        assert await _matched(memory, "the of a", match) == set()


async def test_cover_density_ranks_within_a_strict_match(storage: PostgresStorage) -> None:
    memory = await _filled(storage)
    hits = await memory.search(
        "a", "pump seal", kinds=("chunk",), pillars={"text"}, match="all", text=CoverDensity()
    )
    assert [hit.text for hit in hits.hits][-1] == REVERSED


async def test_match_and_depth_refuse_what_they_cannot_mean(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    with pytest.raises(InputError, match=r"^memory: match is any, all, phrase or web$"):
        await memory.search("a", "pump", match="fuzzy")  # type: ignore[arg-type]
    for depth in (-1, 4, True, 1.0):
        with pytest.raises(InputError, match=r"^memory: depth is 0, 1, 2 or 3$"):
            await memory.search("a", "pump", depth=depth)  # type: ignore[arg-type]
        with pytest.raises(InputError, match=r"^memory: depth is 0, 1, 2 or 3$"):
            await memory.recall("a", "pump", depth=depth)  # type: ignore[arg-type]
        with pytest.raises(InputError, match=r"^memory: depth is 0, 1, 2 or 3$"):
            await memory.expand("a", ("story", "s"), by="graph", depth=depth)  # type: ignore[arg-type]


def _entity(name: str) -> Entity:
    return Entity(
        corpus_id="acme",
        entity_id=f"thing:{name.lower()}",
        name=name,
        key=name.lower(),
        type="thing",
        aliases=(),
    )


async def _story(
    memory: Memory,
    story_id: str,
    mentions: list[Entity],
    relations: tuple[tuple[Entity, Entity], ...] = (),
) -> None:
    story = Story(
        corpus_id="acme",
        story_id=story_id,
        text=f"story {story_id}",
        subject=None,
        valid_from=JAN,
        valid_to=None,
        created_at=JAN,
        source_ids=(),
        metadata={},
    )
    ends = [entity for pair in relations for entity in pair]
    await memory.storage.put_story(
        "acme",
        story,
        None,
        None,
        graph=GraphWrite(
            entities=list({e.entity_id: e for e in [*mentions, *ends]}.values()),
            relations=[
                Relation(
                    corpus_id="acme",
                    from_id=start.entity_id,
                    to_id=end.entity_id,
                    type="FEEDS",
                    evidence_kind="story",
                    evidence_id=story_id,
                )
                for start, end in relations
            ],
            mentions=[(e.entity_id, "story", story_id, e.name) for e in mentions],
        ),
    )


async def _chain(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, pillars={"graph"})
    await memory.check()
    apollo, borealis, cygnus, draco = (
        _entity(name) for name in ("Apollo", "Borealis", "Cygnus", "Draco")
    )
    await _story(memory, "pa", [apollo])
    await _story(memory, "pb", [borealis])
    await _story(memory, "pc", [cygnus])
    await _story(memory, "pd", [draco])
    await _story(memory, "r1", [], ((apollo, borealis),))
    await _story(memory, "r2", [], ((borealis, cygnus),))
    await _story(memory, "r3", [], ((cygnus, draco),))
    return memory


async def test_depth_sets_how_far_the_graph_walks_from_what_the_query_names(
    storage: PostgresStorage,
) -> None:
    memory = await _chain(storage)
    reached = []
    for depth in (0, 1, 2, 3):
        hits = await memory.search("acme", "what about Apollo?", kinds=("story",), depth=depth)
        reached.append([hit.id for hit in hits.hits])
    assert reached == [["pa"], ["pa", "pb"], ["pa", "pb", "pc"], ["pa", "pb", "pc", "pd"]]
    default = await memory.search("acme", "what about Apollo?", kinds=("story",))
    assert [hit.id for hit in default.hits] == reached[2]
    recalled = await memory.recall("acme", "what about Apollo?", depth=1)
    assert [hit.id for hit in recalled.hits] == ["pa", "pb"]


async def test_depth_sets_how_far_expanding_by_graph_walks(storage: PostgresStorage) -> None:
    memory = await _chain(storage)
    near = await memory.expand("acme", ("story", "pa"), by="graph", depth=1)
    assert [hit.id for hit in near.hits] == ["pb"]
    default = await memory.expand("acme", ("story", "pa"), by="graph")
    assert [hit.id for hit in default.hits] == ["pb", "pc"]
    far = await memory.expand("acme", ("story", "pa"), by="graph", depth=3)
    assert [hit.id for hit in far.hits] == ["pb", "pc", "pd"]


async def test_the_tools_offer_match_and_depth_and_never_ranking_or_fusion(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    schemas = {
        d["name"]: d["input_schema"]["properties"]
        for d in MemoryTools(memory, corpus_id="acme").definitions()
    }
    for name in ("search", "recall"):
        assert schemas[name]["match"]["enum"] == ["any", "all", "phrase", "web"]
        assert schemas[name]["match"]["default"] == "any"
        depth = schemas[name]["depth"]
        assert (depth["default"], depth["minimum"], depth["maximum"]) == (2, 0, 3)
        assert "text" not in schemas[name] and "fusion" not in schemas[name]
    assert (schemas["expand"]["depth"]["default"], schemas["expand"]["depth"]["maximum"]) == (2, 3)
    flat = Memory(storage, embedding=HashEmbedding())
    plain = {
        d["name"]: d["input_schema"]["properties"]
        for d in MemoryTools(flat, corpus_id="acme").definitions()
    }
    assert "depth" not in plain["search"] and plain["search"]["match"]["default"] == "any"


async def test_the_tools_pass_match_and_depth_through(storage: PostgresStorage) -> None:
    memory = await _filled(storage)
    tools = MemoryTools(memory, corpus_id="a")
    found = await tools.dispatch(
        "search", {"query": "pump seal", "match": "phrase", "pillars": ["text"]}
    )
    assert isinstance(found, dict)
    assert {hit["text"] for hit in found["hits"]} == {KIT, YEARLY}
    chain = await _chain(storage)
    graph = MemoryTools(chain, corpus_id="acme")
    walked = await graph.dispatch("search", {"query": "what about Apollo?", "depth": 0})
    assert isinstance(walked, dict) and [hit["id"] for hit in walked["hits"]] == ["pa"]
    with pytest.raises(InputError, match=r"^tools: invalid arguments for search: "):
        await graph.dispatch("search", {"query": "Apollo", "depth": 4})


async def test_a_web_query_of_only_excluded_words_finds_nothing(storage: PostgresStorage) -> None:
    memory = await _filled(storage)
    for query in ("-kit", '-"pump seal"', "-pump -seal"):
        for ranking in (BM25(), CoverDensity()):
            hits = await memory.search(
                "a", query, kinds=("chunk",), pillars={"text"}, match="web", text=ranking
            )
            assert hits.hits == [], (query, ranking)
    kept = await memory.search("a", "pump -seal", kinds=("chunk",), pillars={"text"}, match="web")
    assert {hit.text for hit in kept.hits} == {MOTOR}


async def test_a_deep_walk_hands_in_no_more_than_the_window(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"graph"})
    await memory.check()
    hub = _entity("Hub")
    await _story(memory, "named", [hub])
    for i in range(12):
        spoke = _entity(f"Spoke{i}")
        await _story(memory, f"s{i}", [spoke], ((hub, spoke),))
    hits = await memory.search("acme", "what about Hub?", kinds=("story",), top_k=2, depth=3)
    assert hits.pillars["graph"].returned == 8 and len(hits.hits) == 2


async def test_the_tools_offer_match_only_with_the_text_pillar(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"vector"})
    tools = MemoryTools(memory, corpus_id="a")
    schemas = {d["name"]: d["input_schema"]["properties"] for d in tools.definitions()}
    assert "match" not in schemas["search"] and "match" not in schemas["recall"]
    with pytest.raises(InputError, match=r"^tools: match is not offered here$"):
        await tools.dispatch("search", {"query": "pump", "match": "phrase"})
