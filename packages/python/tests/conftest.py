import asyncio
import copy
import os
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import asyncpg
import pytest
from database import URL, Database, Schemas

from rfnry_memory.storage.postgres import PostgresStorage

PREFIX = "memory_test_"
EMPTY = "; ".join(
    f"DELETE FROM {table}"
    for table in (
        "memory_mentions",
        "memory_relations",
        "memory_entities",
        "memory_stories",
        "memory_chunks",
        "memory_sources",
        "memory_corpora",
    )
)


async def _probe() -> None:
    connection = await asyncpg.connect(os.environ[URL], timeout=5)
    await connection.close()


def pytest_sessionstart(session: pytest.Session) -> None:
    if not os.environ.get(URL):
        raise pytest.UsageError(
            f"{URL} is not set: start the test database with `docker start rfnry-memory-pg` "
            f"and set {URL}=postgresql://memory:memory@127.0.0.1:55434/memory"
        )
    try:
        asyncio.run(_probe())
    except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError, TimeoutError) as error:
        raise pytest.UsageError(
            f"cannot reach {URL} ({type(error).__name__}: {error}) — "
            "start the test database with `docker start rfnry-memory-pg`"
        ) from None


@pytest.fixture(scope="session")
async def database() -> AsyncIterator[Database]:
    admin = await asyncpg.connect(os.environ[URL])
    run = f"{PREFIX}{uuid4().hex[:12]}"
    await admin.execute(f"CREATE SCHEMA {run}")
    try:
        pool = await asyncpg.create_pool(
            os.environ[URL], server_settings={"search_path": f"{run}, public"}
        )
        try:
            storage = PostgresStorage(pool)
            await storage.initialize(dimension=8)
            yield Database(run, admin, pool, storage)
        finally:
            await pool.close()
    finally:
        await admin.execute(f"DROP SCHEMA {run} CASCADE")
        await admin.close()


@pytest.fixture
async def pool(database: Database) -> Any:
    async with database.pool.acquire() as conn:
        await conn.execute(EMPTY)
    return database.pool


@pytest.fixture
async def storage(database: Database, pool: Any) -> PostgresStorage:
    return copy.copy(database.storage)


@pytest.fixture
async def schemas(database: Database) -> AsyncIterator[Schemas]:
    made = Schemas(database)
    try:
        yield made
    finally:
        await made.close()
