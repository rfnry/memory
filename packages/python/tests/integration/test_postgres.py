import os
from uuid import uuid4

import pytest

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
        assert ingested.chunks == 1
        story = await memory.remember("a", "The seal on pump P-3 was replaced in March.")
        assert await memory.remember("a", "the seal on pump  P-3 was replaced in march.") == story
        hits = await memory.search("a", "pump seal")
        assert set(hits.pillars) == {"text"} and hits.pillars["text"].error is None
        assert {h.kind for h in hits.hits} == {"chunk", "story"}
        assert (await memory.search("b", "pump seal")).hits == []
    finally:
        await pool.close()
        await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        await admin.close()
