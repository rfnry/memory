from datetime import UTC, datetime
from typing import Any, cast

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Entity, Hits, Story
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.rows import GraphWrite
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding

JAN = datetime(2026, 1, 1, tzinfo=UTC)


def _entity(name: str, corpus: str = "acme") -> Entity:
    return Entity(
        corpus_id=corpus,
        entity_id=f"thing:{name.lower()}",
        name=name,
        key=name.lower(),
        type="thing",
        aliases=(),
    )


async def _story(memory: Memory, story_id: str, named: Entity, corpus: str = "acme") -> None:
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
    await memory.storage.put_story(
        corpus,
        story,
        None,
        None,
        graph=GraphWrite(
            entities=[named], mentions=[(named.entity_id, "story", story_id, named.name)]
        ),
    )


async def _plant(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, pillars={"graph"})
    await memory.check()
    await _story(memory, "pump", _entity("P-3"))
    await _story(memory, "valve", _entity("valve"))
    return memory


def _ids(hits: Hits) -> list[str]:
    return [h.id for h in hits.hits]


async def test_left_out_spot_reads_the_query_by_its_words(storage: PostgresStorage) -> None:
    memory = await _plant(storage)
    assert _ids(await memory.search("acme", "the valves leak")) == ["valve"]
    assert _ids(await memory.search("acme", "P-30 tripped")) == []
    assert _ids(await memory.search("acme", "P-30 tripped", spot="substring")) == ["pump"]
    assert _ids(await memory.search("acme", "the valves leak", spot="exact")) == []
    assert _ids(await memory.search("acme", "the valve leaks", spot="exact")) == ["valve"]
    off = await memory.search("acme", "the valves leak", spot="off")
    assert off.hits == [] and off.pillars["graph"].returned == 0
    assert _ids(await memory.recall("acme", "the valves leak")) == ["valve"]
    assert _ids(await memory.recall("acme", "the valves leak", spot="off")) == []


async def test_naming_entities_leaves_spotting_off_unless_asked(storage: PostgresStorage) -> None:
    memory = await _plant(storage)
    named = await memory.search("acme", "the valves leak", entities=["P-3"])
    assert _ids(named) == ["pump"]
    around = await memory.search("acme", "the valves leak", around=["P-3"], hops=0)
    assert _ids(around) == ["pump"]
    both = await memory.search("acme", "the valves leak", entities=["P-3"], spot="words")
    assert sorted(_ids(both)) == ["pump", "valve"]
    recalled = await memory.recall("acme", "the valves leak", entities=["P-3"], spot="words")
    assert sorted(_ids(recalled)) == ["pump", "valve"]


async def test_a_spot_outside_the_four_is_refused(storage: PostgresStorage) -> None:
    memory = await _plant(storage)
    refused = r"^memory: spot is words, exact, substring or off$"
    for spot in ("fuzzy", "", "WORDS", True):
        with pytest.raises(InputError, match=refused):
            await memory.search("acme", "valve", spot=cast(Any, spot))
        with pytest.raises(InputError, match=refused):
            await memory.recall("acme", "valve", spot=cast(Any, spot))
    with pytest.raises(InputError, match=refused):
        await memory.search("acme", "   ", spot=cast(Any, "fuzzy"))


async def test_a_new_story_finds_related_stories_by_the_words_it_names(
    storage: PostgresStorage,
) -> None:
    memory = await _plant(storage)
    near = await memory.remember("acme", "P-30 tripped twice.")
    assert _ids(near.related) == []
    named = await memory.remember("acme", "Pump P-3 tripped twice.")
    assert _ids(named.related) == ["pump"]
    plural = await memory.remember("acme", "The valves were replaced.")
    assert _ids(plural.related) == ["valve"]


async def test_each_corpus_spots_in_its_own_language(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"graph"})
    await memory.check()
    await memory.set_language("plain", "simple")
    await _story(memory, "en-valve", _entity("valve", "en"), "en")
    await _story(memory, "plain-valve", _entity("valve", "plain"), "plain")
    both = await memory.search(["en", "plain"], "the valves leak")
    assert _ids(both) == ["en-valve"]
    singular = await memory.search(["en", "plain"], "the valve leaks")
    assert sorted(_ids(singular)) == ["en-valve", "plain-valve"]


def _properties(tools: MemoryTools, name: str) -> dict[str, Any]:
    definition = next(d for d in tools.definitions() if d["name"] == name)
    return cast(dict[str, Any], definition["input_schema"]["properties"])


async def test_the_tools_offer_spot_only_where_the_graph_is_offered(
    storage: PostgresStorage,
) -> None:
    memory = await _plant(storage)
    tools = MemoryTools(memory, corpus_id="acme")
    plain = MemoryTools(Memory(storage, embedding=HashEmbedding()), corpus_id="acme")
    bound = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    for name in ("search", "recall"):
        spot = _properties(tools, name)["spot"]
        assert spot["default"] is None
        assert {"enum": ["words", "exact", "substring", "off"], "type": "string"} in spot["anyOf"]
        assert "exact" in spot["description"] and "off" in spot["description"]
        assert "spot" not in _properties(plain, name)
        assert "spot" not in _properties(bound, name)
        with pytest.raises(InputError, match=r"^tools: spot is not offered here$"):
            await bound.dispatch(name, {"query": "valve", "spot": "exact"})
        with pytest.raises(InputError, match=rf"^tools: invalid arguments for {name}: "):
            await tools.dispatch(name, {"query": "valve", "spot": "fuzzy"})
    found = await tools.dispatch("search", {"query": "P-30 tripped", "spot": "substring"})
    assert isinstance(found, dict) and [h["id"] for h in found["hits"]] == ["pump"]
    left = await tools.dispatch("recall", {"query": "P-30 tripped"})
    assert isinstance(left, dict) and left["hits"] == []
