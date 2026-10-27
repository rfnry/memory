import os
import random
from datetime import UTC, datetime
from functools import partial
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from database import URL, Schemas

from rfnry_memory.memory import Memory
from rfnry_memory.models import Entity, EntityIn, Section, Spot, Story
from rfnry_memory.postgres import PostgresStorage, migrations
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
    ("feed the valve", {"valve"}, {"valve"}, {"valve"}),
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


async def test_a_code_that_begins_with_a_stop_word_loses_it_under_words(
    storage: PostgresStorage,
) -> None:
    await _hold(storage, "codes", [_entity("codes", n) for n in ("A-203", "T-1", "HB-60")])
    assert await _spotted(storage, "codes", "is B-203 open?", "words") == {"A-203"}
    assert await _spotted(storage, "codes", "P-1 tripped", "words") == {"T-1"}
    assert await _spotted(storage, "codes", "HB-600 alarm", "words") == set()
    for query in ("is B-203 open?", "P-1 tripped", "HB-600 alarm"):
        assert await _spotted(storage, "codes", query, "exact") == set(), query
    assert await _spotted(storage, "codes", "is A-203 open?", "exact") == {"A-203"}


KEEPING = (
    "CREATE TEXT SEARCH DICTIONARY spot_stem_all (TEMPLATE = snowball, Language = english); "
    "CREATE TEXT SEARCH CONFIGURATION spot_all (COPY = pg_catalog.english); "
    "ALTER TEXT SEARCH CONFIGURATION spot_all ALTER MAPPING FOR asciiword, asciihword, "
    "hword_asciipart, word, hword, hword_part WITH spot_stem_all"
)


async def test_a_configuration_without_stop_words_keeps_them_in_names(
    storage: PostgresStorage, pool: Any
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(KEEPING)
    try:
        await storage.set_language("kept", "spot_all")
        names = ["AND gate", "OR gate", "IT", "The Who", "valve"]
        await _hold(storage, "kept", [_entity("kept", n) for n in names])
        assert await _spotted(storage, "kept", "the OR gate output", "words") == {"OR gate"}
        assert await _spotted(storage, "kept", "the AND gate output", "words") == {"AND gate"}
        assert await _spotted(storage, "kept", "who in IT approves?", "words") == {"IT"}
        assert await _spotted(storage, "kept", "when did The Who play?", "words") == {"The Who"}
        assert await _spotted(storage, "kept", "the valves leak", "words") == {"valve"}
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DROP TEXT SEARCH CONFIGURATION spot_all; DROP TEXT SEARCH DICTIONARY spot_stem_all"
            )


async def test_the_stored_query_does_not_depend_on_the_order_of_aliases(
    storage: PostgresStorage, pool: Any
) -> None:
    held = [
        Entity(
            corpus_id="plant",
            entity_id=entity_id,
            name="pump",
            key="pump",
            type=kind,
            aliases=aliases,
        )
        for entity_id, kind, aliases in (
            ("a", "part", ("zeta line", "alpha line")),
            ("b", "tag", ("alpha line", "zeta line")),
        )
    ]
    await _hold(storage, "plant", held)
    async with pool.acquire() as conn:
        stored = await conn.fetch(
            "SELECT spotted::text AS q FROM memory_entities WHERE corpus_id = 'plant' "
            "ORDER BY entity_id"
        )
    assert [r["q"] for r in stored] == ["'alpha' <-> 'line' | 'pump' | 'zeta' <-> 'line'"] * 2


async def test_words_reads_a_corpus_without_a_language_of_its_own_in_the_default(
    storage: PostgresStorage, pool: Any
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases, alias_keys, "
            "config) VALUES ('orphan', 'e1', 'valve', 'valve', 'part', '[]', '{}', "
            "(SELECT config FROM memory_settings))"
        )
    assert await _spotted(storage, "orphan", "the valves leak", "words") == {"valve"}


async def test_an_empty_name_is_never_spotted(storage: PostgresStorage, pool: Any) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) "
            "SELECT 'old', language, config FROM memory_settings"
        )
        await conn.execute(
            "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases, alias_keys, "
            "config) VALUES ('old', 'blank', '', '  ', 'part', '[]', '{}', 'pg_catalog.english'), "
            """('old', 'aliased', 'pump', 'pump', 'part', '["  "]', '{}', 'pg_catalog.english')"""
        )
    for query in ("a, b", "the line runs", "--"):
        for spot in READINGS:
            assert await _spotted(storage, "old", query, spot) == set(), (query, spot)


SPELLINGS = ["Pump\tP-3", "Line\x1cA", "Pump\xa0P-3", "ΟΔΟΣ", "İzmir Plant"]


@pytest.mark.parametrize("alias", SPELLINGS)
async def test_an_alias_is_found_by_its_own_spelling(storage: PostgresStorage, alias: str) -> None:
    held = Entity(
        corpus_id="plant", entity_id="held", name="Held", key="held", type="site", aliases=(alias,)
    )
    await _hold(storage, "plant", [held])
    assert [
        e.entity_id for e in (await storage.entities_named("plant", [alias]))[name_key(alias)]
    ] == ["held"]
    for spot in READINGS:
        named = await storage.entities_named_in("plant", f"the {alias} runs", spot=spot)
        assert [e.entity_id for e in named] == ["held"], spot
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.ingest(
        "plant", name="m", text=f"{alias} runs.", entities=[EntityIn(name=alias, type="site")]
    )
    assert (await storage.count_corpus("plant")).entities == 1
    assert [e.entity_id for e in (await memory.neighbors("plant", alias)).entities] == ["held"]


async def test_a_name_is_keyed_the_same_under_any_collation(schemas: Schemas, pool: Any) -> None:
    database = f"{schemas.run}_tr"
    async with pool.acquire() as conn:
        await conn.execute(
            f'CREATE DATABASE "{database}" TEMPLATE template0 LOCALE_PROVIDER icu '
            "ICU_LOCALE 'tr-TR'"
        )
    url = urlunsplit(urlsplit(os.environ[URL])._replace(path=f"/{database}"))
    try:
        turkish = await asyncpg.create_pool(
            url, min_size=1, max_size=4, server_settings={"search_path": "memory, public"}
        )
        try:
            async with turkish.acquire() as conn:
                await conn.execute("CREATE SCHEMA memory")
                assert await conn.fetchval("SELECT lower('IZMIR')") == "ızmır"
                await migrations.apply(
                    conn, partial(migrations.plan_create, dimension=None, language="english")
                )
            storage = PostgresStorage(turkish)
            await storage.check(dimension=None, graph=True)
            held = Entity(
                corpus_id="plant",
                entity_id="held",
                name="Plant",
                key="plant",
                type="site",
                aliases=("IZMIR PLANT",),
            )
            await _hold(storage, "plant", [held])
            assert [
                e.entity_id
                for e in (await storage.entities_named("plant", ["IZMIR PLANT"]))[
                    name_key("IZMIR PLANT")
                ]
            ] == ["held"]
            for spot in READINGS:
                named = await storage.entities_named_in("plant", "the IZMIR PLANT runs", spot=spot)
                assert [e.entity_id for e in named] == ["held"], spot
        finally:
            await turkish.close()
    finally:
        async with pool.acquire() as conn:
            await conn.execute(f'DROP DATABASE "{database}" WITH (FORCE)')


VOCABULARY = [
    "P-3",
    "P-30",
    "tag:C/Motor1",
    "tag:C/Motor1_Run",
    "Run",
    "Motor1",
    "valve",
    "valves",
    "Feed valve",
    "Dana",
    "IT",
    "A-203",
    "B-203",
    "1756-EN2T",
    "EN2T",
    "ΟΔΟΣ",
    "Ação",
    "3.5",
    "13.5",
    "the",
]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("spot", ["words", "exact", "substring"])
async def test_a_write_places_each_name_where_a_read_spots_it(
    storage: PostgresStorage, spot: Spot, seed: int
) -> None:
    rng = random.Random(seed)
    corpus = f"place-{spot}-{seed}"
    filler = ["stopped", "tripped", "on", "line", "and", ",", ".", "is", "open?"]
    texts = [
        " ".join(rng.choice(VOCABULARY + filler + filler) for _ in range(rng.randint(1, 9)))
        for _ in range(rng.randint(2, 7))
    ]
    names = rng.sample(VOCABULARY, rng.randint(3, 12))
    memory = Memory(storage, pillars={"text", "graph"}, spot=spot)
    ingested = await memory.ingest(
        corpus,
        name="log",
        sections=[Section(text=t) for t in texts],
        entities=[EntityIn(name=name, type="thing") for name in names],
    )
    chunks = await storage.source_chunks(corpus, ingested.source.source_id)
    mentioned = await storage.passage_entities(corpus, [("chunk", c.chunk_id) for c in chunks])
    placed: dict[str, set[str]] = {c.chunk_id: set() for c in chunks}
    for (_, chunk_id), entity in mentioned:
        placed[chunk_id].add(entity.key)
    spotted = {
        c.chunk_id: {e.key for e in await storage.entities_named_in(corpus, c.text, spot=spot)}
        for c in chunks
    }
    nowhere = {name_key(n) for n in names} - set().union(*spotted.values())
    spotted[chunks[0].chunk_id] |= nowhere
    assert placed == spotted
