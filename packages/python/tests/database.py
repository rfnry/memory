import asyncio
import io
import os
import sys
from collections.abc import Awaitable
from contextlib import redirect_stderr
from dataclasses import dataclass
from functools import partial
from typing import Any, TextIO
from uuid import uuid4

import asyncpg

from rfnry_memory import cli
from rfnry_memory.postgres import PostgresStorage, migrations

URL = "RFNRY_TEST_POSTGRES_URL"
LOCK_TIMEOUT = {"lock_timeout": "60s"}
WAIT = 30


EMPTY = "; ".join(
    [
        *(
            f"DELETE FROM {table}"
            for table in (
                "memory_term_counts",
                "memory_term_totals",
                "memory_retrieval_labels",
                "memory_retrievals",
                "memory_closings",
                "memory_mentions",
                "memory_relations",
                "memory_entities",
                "memory_stories",
                "memory_chunks",
                "memory_sources",
                "memory_corpora",
            )
        ),
        "DO $$ DECLARE t text; BEGIN FOR t IN SELECT tablename FROM pg_tables "
        "WHERE schemaname = current_schema() AND starts_with(tablename, 'memory_vectors_') "
        "LOOP EXECUTE format('DROP TABLE %I.%I', current_schema(), t); END LOOP; END $$",
    ]
)


class Spawned:
    def __init__(self) -> None:
        self.tasks: list[asyncio.Future[Any]] = []

    def __call__[T](self, awaitable: Awaitable[T]) -> asyncio.Future[T]:
        task = asyncio.ensure_future(awaitable)
        self.tasks.append(task)
        return task

    async def settle(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), WAIT)


class Ran(int):
    printed = ""

    def __repr__(self) -> str:
        return f"{int(self)} (stderr: {self.printed!r})"


class _Both(io.StringIO):
    def __init__(self, shown: TextIO) -> None:
        super().__init__()
        self.shown = shown

    def write(self, text: str) -> int:
        self.shown.write(text)
        return super().write(text)


def _ran(argv: list[str]) -> Ran:
    printed = _Both(sys.stderr)
    with redirect_stderr(printed):
        ran = Ran(cli.main(argv))
    ran.printed = printed.getvalue()
    return ran


async def run_cli(*argv: str) -> Ran:
    return await asyncio.to_thread(_ran, [argv[0], "--url", os.environ[URL], *argv[1:]])


async def provision(
    pool: Any, *, dimension: int | None = 8, graph: bool = True, language: str = "english"
) -> PostgresStorage:
    async with pool.acquire() as conn:
        await migrations.apply(
            conn,
            partial(migrations.plan_create, dimension=dimension, graph=graph, language=language),
        )
    storage = PostgresStorage(pool)
    await storage.check(dimension=dimension, graph=graph)
    return storage


@dataclass
class Database:
    run: str
    admin: Any
    pool: Any
    storage: PostgresStorage


class Schemas:
    def __init__(self, database: Database) -> None:
        self.run = database.run
        self._admin = database.admin
        self._made: list[str] = []
        self._pools: list[Any] = []

    async def create(self) -> str:
        name = f"{self.run}_{uuid4().hex[:8]}"
        await self._admin.execute(f"CREATE SCHEMA {name}")
        self._made.append(name)
        return name

    async def pool(self, *schemas: str, public: bool = True, **settings: str) -> Any:
        path = ", ".join([*schemas, *(["public"] if public else [])])
        pool = await asyncpg.create_pool(
            os.environ[URL],
            min_size=1,
            max_size=5,
            server_settings={"search_path": path, **LOCK_TIMEOUT, **settings},
        )
        self._pools.append(pool)
        return pool

    async def close(self) -> None:
        for pool in self._pools:
            await pool.close()
        failed: Exception | None = None
        for name in self._made:
            try:
                await self._admin.execute(f"DROP SCHEMA {name} CASCADE")
            except Exception as exc:
                failed = failed or exc
        if failed is not None:
            raise failed
