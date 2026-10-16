from datetime import UTC, datetime

import pytest

from rfnry_memory.models import Entity, Spot, Story
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.rows import GraphWrite, name_key

JAN = datetime(2026, 1, 1, tzinfo=UTC)

READINGS: tuple[Spot, ...] = ("words", "exact", "substring")


def _entity(corpus: str, name: str, *aliases: str) -> Entity:
    return Entity(
        corpus_id=corpus,
        entity_id=f"e:{name_key(name)}",
        name=name,
        key=name_key(name),
        type="thing",
        aliases=aliases,
    )


async def _hold(storage: PostgresStorage, corpus: str, entities: list[Entity]) -> None:
    story = Story(
        corpus_id=corpus,
        story_id=f"{corpus}-names",
        text="names",
        subject=None,
        valid_from=JAN,
        valid_to=None,
        created_at=JAN,
        source_ids=(),
        metadata={},
    )
    await storage.put_story(
        corpus,
        story,
        None,
        None,
        graph=GraphWrite(
            entities=entities,
            mentions=[(e.entity_id, "story", story.story_id, e.name) for e in entities],
        ),
    )


async def _spotted(storage: PostgresStorage, corpus: str, query: str, spot: Spot) -> set[str]:
    return {e.name for e in await storage.entities_named_in(corpus, query, spot=spot)}


PLANT = [
    _entity("plant", "valve"),
    _entity("plant", "P-3", "Feed valve"),
    _entity("plant", "Run"),
    _entity("plant", "Motor1_Run"),
    _entity("plant", "Dana"),
    _entity("plant", "EN2T"),
    _entity("plant", "IT"),
    _entity("plant", "AND gate"),
    _entity("plant", "Acme"),
]

CASES: list[tuple[str, set[str], set[str], set[str]]] = [
    ("the valves leak", {"valve"}, set(), {"valve"}),
    ("P-30 tripped", set(), set(), {"P-3"}),
    ("Line3_PLC:Motor1_Run stopped", {"Run", "Motor1_Run"}, {"Motor1_Run"}, {"Run", "Motor1_Run"}),
    ("Dana's shift starts", {"Dana"}, {"Dana"}, {"Dana"}),
    ("Danalyn called", set(), set(), {"Dana"}),
    ("1756-EN2T module", {"EN2T"}, set(), {"EN2T"}),
    ("who in IT approves laptops?", set(), {"IT"}, {"IT"}),
    ("is it working?", set(), {"IT"}, {"IT"}),
    ("the OR gate output", {"AND gate"}, set(), set()),
    ("Acme's order", {"Acme"}, {"Acme"}, {"Acme"}),
    ("the feed valves leak", {"valve", "P-3"}, set(), {"valve", "P-3"}),
    ("a feed valve", {"valve", "P-3"}, {"valve", "P-3"}, {"valve", "P-3"}),
    ("  A FEED\tvalve ", {"valve", "P-3"}, {"valve", "P-3"}, {"valve", "P-3"}),
]


@pytest.mark.parametrize(("query", "words", "exact", "substring"), CASES)
async def test_each_reading_finds_the_names_it_promises(
    storage: PostgresStorage, query: str, words: set[str], exact: set[str], substring: set[str]
) -> None:
    await _hold(storage, "plant", PLANT)
    found = {spot: await _spotted(storage, "plant", query, spot) for spot in READINGS}
    assert found == {"words": words, "exact": exact, "substring": substring}


async def test_off_reads_nothing(storage: PostgresStorage) -> None:
    await _hold(storage, "plant", PLANT)
    assert await storage.entities_named_in("plant", "the valves leak", spot="off") == []


async def test_words_reads_names_in_the_corpus_language(storage: PostgresStorage) -> None:
    await storage.set_language("pt", "portuguese")
    await storage.set_language("plain", "simple")
    await _hold(storage, "pt", [_entity("pt", "bomba")])
    await _hold(storage, "plain", [_entity("plain", "valve"), _entity("plain", "IT")])
    assert await _spotted(storage, "pt", "as bombas falharam", "words") == {"bomba"}
    assert await _spotted(storage, "pt", "as bombas falharam", "exact") == set()
    assert await _spotted(storage, "plain", "the valves leak", "words") == set()
    assert await _spotted(storage, "plain", "who in IT approves?", "words") == {"IT"}


async def test_names_with_pattern_syntax_are_read_literally(storage: PostgresStorage) -> None:
    names = ["C++", "(X)", "A|B", "x:*", "50%", "a_b", "o'neil", "[v]"]
    await _hold(storage, "odd", [_entity("odd", name) for name in names])
    query = "we use c++ and (x) here, a|b too, at 50% with o'neil"
    assert await _spotted(storage, "odd", query, "exact") == {"C++", "(X)", "A|B", "50%", "o'neil"}
    assert await _spotted(storage, "odd", query, "substring") == {
        "C++",
        "(X)",
        "A|B",
        "50%",
        "o'neil",
    }
    for spot in READINGS:
        await _spotted(storage, "odd", "[v] x:* a_b ' \\ % _ ( ) | & ! <-> :*", spot)


async def test_a_query_without_words_spots_nothing_by_words(storage: PostgresStorage) -> None:
    await _hold(storage, "plant", PLANT)
    for query in ("the and of", "?!", "   "):
        assert await _spotted(storage, "plant", query, "words") == set()
    assert await _spotted(storage, "plant", "it, and", "exact") == {"IT"}
