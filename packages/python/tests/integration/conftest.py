import os
from collections.abc import AsyncIterator
from typing import Any

import pytest

from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.storage.postgres import PostgresStorage
from rfnry_memory.storage.storage import Storage

TABLES = "memory_mentions, memory_relations, memory_entities, memory_stories, memory_chunks, memory_sources, memory_corpora"


@pytest.fixture(params=["inmemory", pytest.param("postgres", marks=pytest.mark.postgres)])
async def storage(request: Any) -> AsyncIterator[Storage]:
    if request.param == "inmemory":
        memory = InMemoryStorage()
        await memory.initialize(dimension=8)
        yield memory
        return
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(url)
    postgres = PostgresStorage(pool)
    await postgres.initialize(dimension=8)
    async with pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {TABLES}")
    yield postgres
    await pool.close()
