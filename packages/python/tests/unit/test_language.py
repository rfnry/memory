import re
from typing import Any

import pytest
from database import Schemas

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Counts
from rfnry_memory.postgres import PostgresStorage


async def _memory(storage: PostgresStorage) -> Memory:
    memory = Memory(storage)
    await memory.initialize()
    await memory.ingest(
        "acme", name="manual", text="La información de las válvulas está en el manual."
    )
    await memory.remember("acme", "Las bombas funcionan mejor después del cambio.")
    return memory


async def _found(memory: Memory, query: str, corpus: str = "acme") -> list[str]:
    hits = await memory.search(corpus, query, pillars={"text"})
    return [h.kind for h in hits.hits]


async def _defaulting_to(schemas: Schemas, language: str) -> Memory:
    pool = await schemas.pool(schemas.run, default_text_search_config=language)
    return Memory(PostgresStorage(pool))


async def test_a_corpus_reads_its_own_language(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    assert await _found(memory, "funcionando") == []
    assert await memory.set_language("acme", "spanish") == 2
    assert await _found(memory, "funcionando") == ["story"]
    assert await _found(memory, "informacion valvula") == ["chunk"]
    assert await memory.set_language("acme", "spanish") == 0
    assert await memory.set_language("acme", "simple") == 2
    assert await _found(memory, "valvula bomba") == []
    assert await _found(memory, "VALVULAS") == []
    assert await _found(memory, "VÁLVULAS") == ["chunk"]


async def test_a_built_in_language_reads_text_as_postgres_ships_it(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage)
    await memory.ingest("acme", name="menu", text="The café opens at noon.")
    assert await _found(memory, "cafe") == []
    assert await _found(memory, "café") == ["chunk"]
    assert (await memory.fingerprint("acme")).language == "english"


async def test_a_corpus_takes_the_default_language_at_its_first_write_and_keeps_it(
    storage: PostgresStorage, schemas: Schemas
) -> None:
    portuguese = await _defaulting_to(schemas, "pg_catalog.portuguese")
    english = Memory(storage)
    await portuguese.ingest("pt", name="nota", text="As bombas funcionavam ontem.")
    await english.ingest("en", name="note", text="The pumps were running yesterday.")
    assert (await english.fingerprint("pt")).language == "portuguese"
    assert (await portuguese.fingerprint("en")).language == "english"
    assert await _found(english, "funcionando", "pt") == ["chunk"]
    await english.remember("pt", "As bombas funcionam de novo.")
    assert sorted(await _found(english, "funcionando", "pt")) == ["chunk", "story"]


async def test_a_corpus_never_written_reads_the_default_and_writes_nothing(
    storage: PostgresStorage, schemas: Schemas
) -> None:
    portuguese = await _defaulting_to(schemas, "pg_catalog.portuguese")
    english = Memory(storage)
    assert (await portuguese.fingerprint("unseen")).language == "portuguese"
    assert (await english.describe("unseen")).language == "english"
    assert await _found(portuguese, "funcionando", "unseen") == []
    assert await english.verify_absent("unseen") == Counts()


async def test_keeping_history_pins_the_default_language_too(
    storage: PostgresStorage, schemas: Schemas
) -> None:
    portuguese = await _defaulting_to(schemas, "pg_catalog.portuguese")
    await portuguese.set_history("pt", True)
    english = Memory(storage)
    await english.ingest("pt", name="nota", text="As bombas funcionavam ontem.")
    assert (await english.fingerprint("pt")).language == "portuguese"
    assert await _found(english, "funcionando", "pt") == ["chunk"]


async def test_a_folding_configuration_you_build_finds_words_typed_without_accents(
    storage: PostgresStorage, pool: Any, schemas: Schemas
) -> None:
    schema = await schemas.create()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
        await conn.execute(
            f"CREATE TEXT SEARCH CONFIGURATION {schema}.pt_folded (COPY = pg_catalog.portuguese)"
        )
        await conn.execute(
            f"ALTER TEXT SEARCH CONFIGURATION {schema}.pt_folded "
            "ALTER MAPPING FOR hword, hword_part, word WITH unaccent, portuguese_stem"
        )
    memory = Memory(storage)
    await memory.ingest("pt", name="nota", text="A ação começou cedo.")
    assert await memory.set_language("pt", "portuguese") == 1
    assert await _found(memory, "acao", "pt") == []
    assert await memory.set_language("pt", f"{schema}.pt_folded") == 1
    assert await _found(memory, "acao", "pt") == ["chunk"]
    assert (await memory.fingerprint("pt")).language == f"{schema}.pt_folded"


async def test_any_built_in_language_reads_its_own_words(storage: PostgresStorage) -> None:
    memory = Memory(storage)
    await memory.ingest("acme", name="note", text="Les chevaux mangeaient près de l'école.")
    assert await _found(memory, "cheval") == []
    assert await memory.set_language("acme", "French") == 1
    assert (await memory.fingerprint("acme")).language == "french"
    assert await _found(memory, "cheval") == ["chunk"]
    assert await _found(memory, "ecole") == []
    assert await _found(memory, "école") == ["chunk"]
    assert await memory.set_language("acme", "pg_catalog.french") == 0
    assert await memory.set_language("acme", "german") == 1
    assert (await memory.describe("acme")).language == "german"


async def test_a_configuration_you_created_is_read_exactly_as_you_made_it(
    storage: PostgresStorage, pool: Any, schemas: Schemas
) -> None:
    schema = await schemas.create()
    async with pool.acquire() as conn:
        await conn.execute(
            f"CREATE TEXT SEARCH CONFIGURATION {schema}.legal (COPY = pg_catalog.english)"
        )
    memory = Memory(storage)
    await memory.ingest("acme", name="brief", text="The café was replacing its seals.")
    assert await _found(memory, "café") == ["chunk"]
    assert await memory.set_language("acme", f"{schema}.legal") == 1
    assert (await memory.fingerprint("acme")).language == f"{schema}.legal"
    assert await _found(memory, "cafe") == []
    assert await _found(memory, "café replaced") == ["chunk"]


async def test_a_configuration_created_after_the_text_is_read_exactly_as_made(
    storage: PostgresStorage, pool: Any, schemas: Schemas
) -> None:
    memory = Memory(storage)
    await memory.ingest("acme", name="brief", text="The café was replacing its seals.")
    schema = await schemas.create()
    async with pool.acquire() as conn:
        await conn.execute(
            f"CREATE TEXT SEARCH CONFIGURATION {schema}.later (COPY = pg_catalog.simple)"
        )
    assert await _found(memory, "replaced") == ["chunk"]
    assert await memory.set_language("acme", f"{schema}.later") == 1
    assert await _found(memory, "replaced") == []
    assert await _found(memory, "replacing") == ["chunk"]


async def test_naming_your_configuration_again_reads_the_text_as_it_is_now(
    storage: PostgresStorage, pool: Any, schemas: Schemas
) -> None:
    schema = await schemas.create()
    async with pool.acquire() as conn:
        await conn.execute(
            f"CREATE TEXT SEARCH CONFIGURATION {schema}.legal (COPY = pg_catalog.simple)"
        )
    memory = Memory(storage)
    assert await memory.set_language("acme", f"{schema}.legal") == 0
    await memory.ingest("acme", name="brief", text="The pumps were replaced.")
    await memory.remember("acme", "The pumps run again.")
    async with pool.acquire() as conn:
        await conn.execute(
            f"ALTER TEXT SEARCH CONFIGURATION {schema}.legal "
            "ALTER MAPPING FOR asciiword WITH english_stem"
        )
    assert await _found(memory, "pumps") == []
    assert await memory.set_language("acme", f"{schema}.legal") == 2
    assert sorted(await _found(memory, "pumps")) == ["chunk", "story"]


async def test_a_language_belongs_to_one_corpus_and_goes_with_it(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.ingest("other", name="manual", text="La información de las válvulas.")
    await memory.set_language("acme", "portuguese")
    assert (await memory.fingerprint("acme")).language == "portuguese"
    assert (await memory.fingerprint("other")).language == "english"
    assert (await memory.verify_absent("acme")).corpora == 1
    assert (await memory.verify_absent("other")).corpora == 1
    deleted = await memory.delete_corpus("acme")
    assert deleted.corpora == 1 and (await memory.verify_absent("acme")) == Counts()
    assert (await memory.fingerprint("acme")).language == "english"


@pytest.mark.parametrize(
    "name", ["klingon", "", "x y", "a.b.c", "nowhere.french", 3, "-", "0", "3748", "99999999999"]
)
async def test_a_name_the_database_does_not_know_is_refused(
    storage: PostgresStorage, name: Any
) -> None:
    memory = await _memory(storage)
    with pytest.raises(
        InputError,
        match=rf"^memory: {re.escape(repr(name))} is not a text search configuration in this database$",
    ):
        await memory.set_language("acme", name)
    assert (await memory.fingerprint("acme")).language == "english"
    assert await _found(memory, "funcionando") == []


async def test_a_language_needs_a_corpus(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    with pytest.raises(InputError, match=r"^memory: corpus_id is required$"):
        await memory.set_language(" ", "spanish")


async def test_memory_makes_no_text_search_configuration_of_its_own(schemas: Schemas) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    memory = Memory(PostgresStorage(pool))
    await memory.initialize()
    await memory.ingest("acme", name="manual", text="The café opens at noon.")
    await memory.set_language("acme", "french")
    async with pool.acquire() as conn:
        made = await conn.fetchval(
            "SELECT count(*) FROM pg_ts_config WHERE cfgnamespace = $1::regnamespace", schema
        )
    assert made == 0
