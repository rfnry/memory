import asyncio
import os
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest
from database import URL, Database, Schemas, provision
from unit.fakes import HashEmbedding

from rfnry_memory.errors import StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn, Window
from rfnry_memory.postgres import PostgresStorage


async def _lock_key(storage: PostgresStorage, name: str) -> int:
    async with storage._pool.acquire() as conn:
        return int(
            await conn.fetchval("SELECT hashtextextended(current_schema() || ':' || $1, 0)", name)
        )


async def test_memory_without_an_embedding_model_writes_and_reads_on_postgres(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 3, 1, tzinfo=UTC))
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    memory = Memory(await provision(pool, dimension=None))
    await memory.check()
    async with pool.acquire() as conn:
        columns = await conn.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = $1 AND table_name = 'memory_chunks'",
            schema,
        )
    assert "embedding" not in {r["column_name"] for r in columns}
    ingested = await memory.ingest(
        "a", name="manual", text="The pump seal on P-3 must be replaced every year."
    )
    assert ingested.source.chunks == 1
    story = (await memory.remember("a", "The seal on pump P-3 was replaced in March.")).story
    assert (
        await memory.remember("a", "the seal on pump  P-3 was replaced in march.")
    ).story == story
    hits = await memory.search("a", "pump seal")
    assert set(hits.pillars) == {"text"} and hits.pillars["text"].error is None
    with pytest.raises(
        StorageError,
        match=r"^postgres: vectors were given but this storage has no vector dimension — call check\(\) first$",
    ):
        await memory.storage.search_vector(
            "a",
            [0.0] * 8,
            kinds=("chunk",),
            top_k=1,
            filters=None,
            window=Window(),
            subject=None,
        )
    assert {h.kind for h in hits.hits} == {"chunk", "story"}
    assert (await memory.search("b", "pump seal")).hits == []
    fingerprint = await memory.fingerprint("a")
    assert (fingerprint.embedding, fingerprint.embedded, fingerprint.stale) == (None, {}, 0)


async def test_a_storage_that_was_not_checked_refuses_vectors_rather_than_drop_them(
    pool: Any,
) -> None:
    first = Memory(PostgresStorage(pool), embedding=HashEmbedding())
    await first.check()
    story = (await first.remember("a", "Ana owns the pump.")).story
    kept = (await first.ingest("a", name="v0", text="version zero", metadata={"file": "a"})).source
    restarted = Memory(PostgresStorage(pool), embedding=HashEmbedding())
    first_write = r"^postgres: vectors were given but this storage has no vector dimension — call check\(\) first$"
    with pytest.raises(StorageError, match=first_write):
        await restarted.ingest("a", name="manual", text="pump seal yearly")
    with pytest.raises(StorageError, match=first_write):
        await restarted.remember("a", "Bo owns the pump.")
    with pytest.raises(StorageError, match=first_write):
        await restarted.supersede("a", story.story_id, "Cy owns the pump.")
    with pytest.raises(StorageError, match=first_write):
        await restarted.sync(
            "a",
            filters={"metadata.file": "a"},
            sources=[SourceIn(name="v1", text="version one", metadata={"file": "a"})],
        )
    assert await restarted.list_sources("a") == ([kept], None)
    held = await first.fingerprint("a")
    assert (held.embedded, held.stale) == ({"hash-8": 2}, 0)
    assert await restarted.fingerprint("a") == held
    hits = await restarted.search("a", "pump", pillars={"vector"})
    assert hits.pillars["vector"].error == "StorageError: " + first_write[1:-1].replace("\\", "")
    renamed = HashEmbedding()
    renamed.model = "hash-8-v2"
    with pytest.raises(StorageError, match=first_write):
        await Memory(PostgresStorage(pool), embedding=renamed).reembed("a")
    residue = await restarted.verify_absent("a")
    assert (residue.sources, residue.stories) == (1, 1)
    plain = PostgresStorage(pool)
    await plain.check(dimension=None, graph=True)
    await Memory(plain, embedding=HashEmbedding()).ingest("a", name="m", text="pump seal yearly")
    hits = await first.search("a", "pump seal yearly", pillars={"vector"})
    assert hits.pillars["vector"].error is None and hits.hits[0].kind == "chunk"
    assert await first.reembed("a") == 0


async def test_no_column_has_a_reg_type_so_the_database_can_be_upgraded(pool: Any) -> None:
    async with pool.acquire() as conn:
        typed = await conn.fetch(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name LIKE 'memory%' AND data_type LIKE 'reg%'"
        )
    assert typed == []


async def test_a_write_that_waited_for_a_language_change_lands_under_it_at_repeatable_read(
    pool: Any, schemas: Schemas
) -> None:
    repeatable = await schemas.pool(schemas.run, default_transaction_isolation="repeatable read")
    storage = PostgresStorage(repeatable)
    await storage.check(dimension=8, graph=True)
    await storage.set_language("es", "spanish")
    memory = Memory(storage)
    async with pool.acquire() as holder, holder.transaction():
        await holder.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || ':' || $1, 0))",
            "memory.language:rr",
        )
        await holder.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) "
            "SELECT 'rr', language, config FROM memory_corpora WHERE corpus_id = 'es'"
        )
        writing = asyncio.create_task(
            memory.ingest("rr", name="m", text="La información de las válvulas.")
        )
        remembering = asyncio.create_task(memory.remember("rr", "Las bombas funcionan."))
        await asyncio.sleep(0.5)
        assert not writing.done() and not remembering.done()
    await asyncio.wait_for(asyncio.gather(writing, remembering), 10)
    async with pool.acquire() as conn:
        configs = await conn.fetch(
            "SELECT config FROM memory_chunks WHERE corpus_id = 'rr' "
            "UNION ALL SELECT config FROM memory_stories WHERE corpus_id = 'rr'"
        )
    assert [r["config"] for r in configs] == ["pg_catalog.spanish"] * 2
    assert [h.kind for h in (await memory.search("rr", "funcionando", pillars={"text"})).hits] == [
        "story"
    ]


async def test_set_history_waits_for_a_running_sync(storage: PostgresStorage) -> None:
    holder = await asyncpg.connect(os.environ[URL])
    try:
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock($1)", await _lock_key(storage, "memory.sync:hist")
            )
            switching = asyncio.create_task(storage.set_history("hist", True))
            await asyncio.sleep(0.5)
            assert not switching.done() and await storage.get_history("hist") is False
        await asyncio.wait_for(switching, 10)
    finally:
        await holder.close()
    assert await storage.get_history("hist") is True


async def test_erasing_waits_for_a_running_sync(storage: PostgresStorage) -> None:
    holder = await asyncpg.connect(os.environ[URL])
    try:
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock($1)", await _lock_key(storage, "memory.sync:erase")
            )
            deleting = asyncio.create_task(storage.delete_source("erase", "ghost"))
            forgetting = asyncio.create_task(
                storage.forget_retired("erase", before=datetime(2030, 1, 1, tzinfo=UTC))
            )
            dropping = asyncio.create_task(storage.delete_corpus("erase"))
            await asyncio.sleep(0.5)
            assert not (deleting.done() or forgetting.done() or dropping.done())
        await asyncio.wait_for(asyncio.gather(deleting, forgetting, dropping), 10)
    finally:
        await holder.close()


async def test_neighbors_skips_a_relation_whose_chunk_is_gone(
    pool: Any, storage: PostgresStorage
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_relations (corpus_id, from_id, to_id, type, evidence_kind, evidence_id) "
            "VALUES ('orphan', 'e1', 'e2', 'FEEDS', 'chunk', 'ghost-0')"
        )
    assert await storage.neighbors("orphan", ["e1"], relation_types=None, window=Window()) == []


async def test_closing_the_test_schemas_drops_every_one_even_when_one_drop_fails(
    database: Database,
) -> None:
    made = Schemas(database)
    first, second = await made.create(), await made.create()
    admin = database.admin

    class Refusing:
        async def execute(self, sql: str, *args: Any) -> Any:
            if first in sql:
                raise RuntimeError("drop refused")
            return await admin.execute(sql, *args)

    made._admin = Refusing()
    with pytest.raises(RuntimeError, match=r"^drop refused$"):
        await made.close()
    assert await admin.fetchval("SELECT to_regnamespace($1) IS NULL", second)
    await admin.execute(f"DROP SCHEMA {first} CASCADE")
