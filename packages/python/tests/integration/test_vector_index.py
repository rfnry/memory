import asyncio
import re
from hashlib import sha256
from typing import Any

import pytest
from database import Schemas
from unit.fakes import HashEmbedding

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Section, VectorIndex
from rfnry_memory.storage.postgres import PostgresStorage

DEFAULT = VectorIndex(m=16, ef_construction=64, precision="full")


class Wide(HashEmbedding):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [[(i % 5 + 1) / 5 for i in range(self.dimension)] for _ in texts]


def _table(corpus_id: str) -> str:
    return f"memory_vectors_{sha256(corpus_id.encode('utf-8')).hexdigest()[:24]}"


async def _indexes(pool: Any, corpus_id: str) -> dict[str, tuple[int, bool]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT c.relname, c.oid, i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE i.indrelid = to_regclass(format('%I.%I', current_schema(), $1::text)) AND NOT i.indisprimary",
            _table(corpus_id),
        )
    return {r["relname"]: (r["oid"], r["indisvalid"]) for r in rows}


async def _rows(pool: Any, corpus_id: str) -> list[tuple[str, str]] | None:
    async with pool.acquire() as conn:
        table = await conn.fetchval(
            "SELECT format('%I.%I', current_schema(), $1::text) "
            "WHERE to_regclass(format('%I.%I', current_schema(), $1::text)) IS NOT NULL",
            _table(corpus_id),
        )
        if table is None:
            return None
        rows = await conn.fetch(f"SELECT passage_kind, passage_id FROM {table}")
    return sorted((r["passage_kind"], r["passage_id"]) for r in rows)


async def _embedded(pool: Any, corpus_id: str) -> list[tuple[str, str]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT 'chunk' AS kind, chunk_id AS id FROM memory_chunks WHERE corpus_id = $1 AND embedding IS NOT NULL "
            "UNION ALL SELECT 'story', story_id FROM memory_stories WHERE corpus_id = $1 AND embedding IS NOT NULL",
            corpus_id,
        )
    return sorted((r["kind"], r["id"]) for r in rows)


async def _manuals(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.ingest(
        "manuals",
        name="manual",
        sections=[Section(text="Replace the pump seal yearly."), Section(text="Ground the panel.")],
    )
    await memory.remember("manuals", "Ana owns the seal plan.")
    plain = Memory(storage, embedding=HashEmbedding(), pillars={"text"})
    await plain.ingest("manuals", name="notes", text="Notes kept without a vector.")
    await memory.ingest("other", name="manual", text="Another tenant's manual.")
    return memory


async def test_a_corpus_switches_to_an_index_of_its_own_and_back(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", True) == 3
    assert (await memory.fingerprint("manuals")).vector_index == DEFAULT
    held = await _rows(pool, "manuals")
    assert held == await _embedded(pool, "manuals") and held is not None and len(held) == 3
    hashed = _table("manuals")
    indexes = await _indexes(pool, "manuals")
    assert sorted(re.sub(r"_[0-9a-f]{8}$", "", name) for name in indexes) == [
        f"ix_{hashed}_chunk",
        f"ix_{hashed}_story",
    ]
    assert all(valid for _, valid in indexes.values())
    assert (await memory.fingerprint("other")).vector_index is None
    assert await _rows(pool, "other") is None
    assert await memory.set_vector_index("manuals", False) == 0
    assert (await memory.fingerprint("manuals")).vector_index is None
    assert await _rows(pool, "manuals") is None
    assert await memory.set_vector_index("manuals", False) == 0


async def test_a_corpus_that_holds_nothing_yet_can_be_switched_on(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    assert await memory.set_vector_index("empty", True) == 0
    assert (await memory.fingerprint("empty")).vector_index == DEFAULT
    assert await _rows(pool, "empty") == []


async def test_a_corpus_id_that_is_no_sql_name_is_indexed_like_any_other(
    storage: PostgresStorage, pool: Any
) -> None:
    corpus = 'acme\'s "manuals" ü ' + "x" * 180
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.ingest(corpus, name="manual", text="Replace the pump seal yearly.")
    assert await memory.set_vector_index(corpus, True) == 1
    assert (await memory.fingerprint(corpus)).vector_index == DEFAULT
    assert await _rows(pool, corpus) == await _embedded(pool, corpus)
    assert len(await _indexes(pool, corpus)) == 2
    assert await memory.set_vector_index(corpus, False) == 0
    assert await _rows(pool, corpus) is None


async def test_the_same_values_change_nothing_and_new_ones_replace_the_old_indexes(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", True) == 3
    first = await _indexes(pool, "manuals")
    assert await memory.set_vector_index("manuals", True, m=16, ef_construction=64) == 3
    assert await memory.set_vector_index("manuals", True, precision="full") == 3
    assert await _indexes(pool, "manuals") == first
    assert await memory.set_vector_index("manuals", True, m=32) == 3
    retuned = await _indexes(pool, "manuals")
    assert len(retuned) == 2 and not set(retuned) & set(first)
    assert all(valid for _, valid in retuned.values())
    assert (await memory.fingerprint("manuals")).vector_index == VectorIndex(
        m=32, ef_construction=64, precision="full"
    )
    assert await memory.set_vector_index("manuals", True, m=32, precision="half") == 3
    halved = await _indexes(pool, "manuals")
    assert len(halved) == 2 and not set(halved) & set(retuned)
    assert (await memory.fingerprint("manuals")).vector_index == VectorIndex(
        m=32, ef_construction=64, precision="half"
    )


async def test_a_build_that_never_finished_reports_none_until_the_next_call_replaces_it(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    await memory.set_vector_index("manuals", True)
    built = await _indexes(pool, "manuals")
    chunk = next(name for name in built if "_chunk_" in name)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE pg_index SET indisvalid = false WHERE indexrelid = $1", built[chunk][0]
        )
    assert (await memory.fingerprint("manuals")).vector_index is None
    assert await memory.set_vector_index("manuals", True) == 3
    rebuilt = await _indexes(pool, "manuals")
    assert set(rebuilt) == set(built) and rebuilt[chunk][0] != built[chunk][0]
    assert all(valid for _, valid in rebuilt.values())
    assert (await memory.fingerprint("manuals")).vector_index == DEFAULT


async def test_the_dimension_decides_the_precision_up_to_what_pgvector_indexes(
    schemas: Schemas,
) -> None:
    async def wide(dimension: int) -> Memory:
        pool = await schemas.pool(await schemas.create())
        memory = Memory(PostgresStorage(pool), embedding=Wide(dimension))
        await memory.initialize()
        await memory.ingest("a", name="manual", text="Replace the pump seal yearly.")
        return memory

    full = await wide(2000)
    assert await full.set_vector_index("a", True) == 1
    assert (await full.fingerprint("a")).vector_index == DEFAULT
    half = await wide(2001)
    with pytest.raises(
        InputError,
        match="^"
        + re.escape(
            "postgres: full precision indexes up to 2000 dimensions; this memory embeds 2001 — pass precision='half'"
        )
        + "$",
    ):
        await half.set_vector_index("a", True, precision="full")
    assert (await half.fingerprint("a")).vector_index is None
    assert await half.set_vector_index("a", True) == 1
    assert (await half.fingerprint("a")).vector_index == VectorIndex(
        m=16, ef_construction=64, precision="half"
    )
    beyond = await wide(4001)
    refused = "^postgres: a vector index reaches 4000 dimensions at half precision; this memory embeds 4001$"
    with pytest.raises(ConfigurationError, match=refused):
        await beyond.set_vector_index("a", True)
    with pytest.raises(ConfigurationError, match=refused):
        await beyond.set_vector_index("a", True, precision="half")
    assert (await beyond.fingerprint("a")).vector_index is None


async def test_a_value_pgvector_refuses_is_an_input_error_and_builds_nothing(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    out_of_bounds = (
        "^"
        + re.escape(
            'postgres: value 1 out of bounds for option "m" — Valid values are between "2" and "100".'
        )
        + "$"
    )
    with pytest.raises(InputError, match=out_of_bounds):
        await memory.set_vector_index("manuals", True, m=1)
    with pytest.raises(
        InputError,
        match=r"^postgres: ef_construction must be greater than or equal to 2 \* m$",
    ):
        await memory.set_vector_index("manuals", True, ef_construction=16)
    assert await _rows(pool, "manuals") is None
    assert await memory.set_vector_index("manuals", True) == 3
    built = await _indexes(pool, "manuals")
    with pytest.raises(InputError, match=out_of_bounds):
        await memory.set_vector_index("manuals", True, m=1)
    assert await _indexes(pool, "manuals") == built
    assert (await memory.fingerprint("manuals")).vector_index == DEFAULT


async def test_pgvector_older_than_0_8_0_is_a_configuration_error(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def old(conn: Any) -> str:
        return "0.7.4"

    monkeypatch.setattr("rfnry_memory.storage.postgres._pgvector", old)
    memory = await _manuals(storage)
    with pytest.raises(
        ConfigurationError,
        match=r"^postgres: a vector index needs pgvector 0\.8\.0 or later; this database has 0\.7\.4$",
    ):
        await memory.set_vector_index("manuals", True)
    assert await _rows(pool, "manuals") is None


async def test_set_vector_index_checks_its_arguments_and_needs_an_embedding_model(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    with pytest.raises(InputError, match=r"^memory: on is True or False$"):
        await memory.set_vector_index("manuals", 1)  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: m is an int$"):
        await memory.set_vector_index("manuals", True, m=True)
    with pytest.raises(InputError, match=r"^memory: m is an int$"):
        await memory.set_vector_index("manuals", True, m="16")  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: ef_construction is an int$"):
        await memory.set_vector_index("manuals", True, ef_construction=64.0)  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: precision is 'full', 'half' or None$"):
        await memory.set_vector_index("manuals", True, precision="quarter")  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: corpus_id is required$"):
        await memory.set_vector_index(" ", True)
    unembedded = Memory(storage)
    for on in (True, False):
        with pytest.raises(
            ConfigurationError, match=r"^memory: a vector index needs an embedding model$"
        ):
            await unembedded.set_vector_index("manuals", on)
    with pytest.raises(
        ConfigurationError,
        match=r"^postgres: a vector index needs the embedding column — initialize this storage with a dimension$",
    ):
        await PostgresStorage(pool).set_vector_index(
            "manuals", True, m=16, ef_construction=64, precision=None
        )
    assert await _rows(pool, "manuals") is None


async def test_two_services_switching_one_corpus_run_one_after_the_other(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    built = await asyncio.gather(
        memory.set_vector_index("manuals", True),
        memory.set_vector_index("manuals", True, m=24, ef_construction=96),
    )
    assert list(built) == [3, 3]
    indexes = await _indexes(pool, "manuals")
    assert len(indexes) == 2 and all(valid for _, valid in indexes.values())
    assert (await memory.fingerprint("manuals")).vector_index in (
        DEFAULT,
        VectorIndex(m=24, ef_construction=96, precision="full"),
    )
    switched = await asyncio.gather(
        memory.set_vector_index("manuals", True, m=8, ef_construction=32),
        memory.set_vector_index("manuals", False),
    )
    assert list(switched) == [3, 0]
    indexed = (await memory.fingerprint("manuals")).vector_index
    assert indexed in (None, VectorIndex(m=8, ef_construction=32, precision="full"))
    assert (await _rows(pool, "manuals") is None) == (indexed is None)
