import os
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import asyncpg

from rfnry_memory.postgres import PostgresStorage

URL = "RFNRY_TEST_POSTGRES_URL"


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

    async def pool(self, *schemas: str, **settings: str) -> Any:
        path = ", ".join([*schemas, "public"])
        pool = await asyncpg.create_pool(
            os.environ[URL], min_size=1, server_settings={"search_path": path, **settings}
        )
        self._pools.append(pool)
        return pool

    async def close(self) -> None:
        for pool in self._pools:
            await pool.close()
        for name in self._made:
            await self._admin.execute(f"DROP SCHEMA {name} CASCADE")
