import re
from typing import Any

import pytest
from database import Schemas

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Counts
from rfnry_memory.storage.postgres import PostgresStorage


async def _memory(storage: PostgresStorage) -> Memory:
    memory = Memory(storage)
    await memory.initialize()
    await memory.ingest(
        "acme", name="manual", text="La información de las válvulas está en el manual."
    )
    await memory.remember("acme", "Las bombas funcionan mejor después del cambio.")
    return memory


async def _found(memory: Memory, query: str) -> list[str]:
    hits = await memory.search("acme", query, pillars={"text"})
    return [h.kind for h in hits.hits]


async def test_a_corpus_reads_its_own_language_and_folds_accents(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    assert await _found(memory, "funcionando") == []
    assert await memory.set_language("acme", "spanish") == 2
    assert await _found(memory, "funcionando") == ["story"]
    assert await _found(memory, "informacion valvula") == ["chunk"]
    assert await memory.set_language("acme", "spanish") == 0
    assert await memory.set_language("acme", "simple") == 2
    assert await _found(memory, "valvula bomba") == []
    assert await _found(memory, "VALVULAS") == ["chunk"]


async def test_english_folds_accents_as_every_built_in_language_does(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage)
    await memory.ingest("acme", name="menu", text="The café opens at noon.")
    assert await _found(memory, "cafe") == ["chunk"]
    assert (await memory.fingerprint("acme")).language == "english"


async def test_any_built_in_language_reads_its_own_words(storage: PostgresStorage) -> None:
    memory = Memory(storage)
    await memory.ingest("acme", name="note", text="Les chevaux mangeaient près de l'école.")
    assert await _found(memory, "cheval") == []
    assert await memory.set_language("acme", "French") == 1
    assert (await memory.fingerprint("acme")).language == "french"
    assert await _found(memory, "cheval") == ["chunk"]
    assert await _found(memory, "ecole") == ["chunk"]
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
    assert await _found(memory, "cafe") == ["chunk"]
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
            f"CREATE TEXT SEARCH CONFIGURATION {schema}.later (COPY = pg_catalog.english)"
        )
    assert await _found(memory, "cafe") == ["chunk"]
    assert await memory.set_language("acme", f"{schema}.later") == 1
    assert await _found(memory, "cafe") == []
    assert await _found(memory, "café") == ["chunk"]


async def test_a_language_belongs_to_one_corpus_and_goes_with_it(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.ingest("other", name="manual", text="La información de las válvulas.")
    await memory.set_language("acme", "portuguese")
    assert (await memory.fingerprint("acme")).language == "portuguese"
    assert (await memory.fingerprint("other")).language == "english"
    assert (await memory.verify_absent("acme")).corpora == 1
    assert (await memory.verify_absent("other")).corpora == 0
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
