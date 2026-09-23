import os
from uuid import uuid4

import pytest

from rfnry_memory.errors import ConfigurationError
from rfnry_memory.memory import Memory
from rfnry_memory.storage.postgres import PostgresStorage

pytestmark = pytest.mark.postgres


async def test_memory_without_an_embedding_model_writes_and_reads_on_postgres() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    schema = f"nodim_{uuid4().hex[:12]}"
    admin = await asyncpg.connect(url)
    await admin.execute(f"CREATE SCHEMA {schema}")
    pool = await asyncpg.create_pool(url, server_settings={"search_path": f"{schema}, public"})
    try:
        memory = Memory(PostgresStorage(pool))
        await memory.initialize()
        async with pool.acquire() as conn:
            columns = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = $1 AND table_name = 'memory_chunks'", schema)
        assert "embedding" not in {r["column_name"] for r in columns}
        ingested = await memory.ingest("a", name="manual", text="The pump seal on P-3 must be replaced every year.")
        assert ingested.source.chunks == 1
        story = (await memory.remember("a", "The seal on pump P-3 was replaced in March.")).story
        assert (await memory.remember("a", "the seal on pump  P-3 was replaced in march.")).story == story
        hits = await memory.search("a", "pump seal")
        assert set(hits.pillars) == {"text"} and hits.pillars["text"].error is None
        assert {h.kind for h in hits.hits} == {"chunk", "story"}
        assert (await memory.search("b", "pump seal")).hits == []
    finally:
        await pool.close()
        await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        await admin.close()


async def test_initialize_refuses_tables_made_for_another_dimension() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    admin = await asyncpg.connect(url)
    schemas = [f"dims_{uuid4().hex[:12]}" for _ in range(2)]
    pools = []
    try:
        for schema in schemas:
            await admin.execute(f"CREATE SCHEMA {schema}")
            pools.append(await asyncpg.create_pool(url, server_settings={"search_path": f"{schema}, public"}))
        eight, bare = pools
        await PostgresStorage(eight).initialize(dimension=8)
        await PostgresStorage(eight).initialize(dimension=8)
        await PostgresStorage(eight).initialize(dimension=None)
        with pytest.raises(ConfigurationError, match=r"^postgres: tables were created for dimension 8; this memory embeds 1024 — re-provision or use another schema$"):
            await PostgresStorage(eight).initialize(dimension=1024)
        await PostgresStorage(bare).initialize(dimension=None)
        with pytest.raises(ConfigurationError, match=r"^postgres: tables were created without a vector column; this memory embeds 1024 — re-provision or use another schema$"):
            await PostgresStorage(bare).initialize(dimension=1024)
    finally:
        for pool in pools:
            await pool.close()
        for schema in schemas:
            await admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await admin.close()


async def test_initialize_checks_the_tables_it_would_create_not_others_on_the_search_path() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    admin = await asyncpg.connect(url)
    below, above = f"below_{uuid4().hex[:12]}", f"above_{uuid4().hex[:12]}"
    pools = []
    try:
        for schema in (below, above):
            await admin.execute(f"CREATE SCHEMA {schema}")
        pools.append(await asyncpg.create_pool(url, server_settings={"search_path": f"{below}, public"}))
        await PostgresStorage(pools[0]).initialize(dimension=8)
        pools.append(await asyncpg.create_pool(url, server_settings={"search_path": f"{above}, {below}, public"}))
        await PostgresStorage(pools[1]).initialize(dimension=16)
        created = await admin.fetch("SELECT table_schema FROM information_schema.tables WHERE table_name = 'memory_stories' AND table_schema = ANY($1::text[])",
                                    [below, above])
        assert sorted(r["table_schema"] for r in created) == sorted([below, above])
    finally:
        for pool in pools:
            await pool.close()
        for schema in (below, above):
            await admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await admin.close()
