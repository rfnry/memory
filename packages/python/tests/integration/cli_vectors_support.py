import asyncio
import json
import os
import signal
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from database import URL, WAIT, Database, Ran, Schemas, provision, run_cli
from unit.fakes import HashEmbedding

from rfnry_memory import cli
from rfnry_memory.memory import Memory
from rfnry_memory.models import Section
from rfnry_memory.postgres import PostgresStorage

NEW = "new-model"


OLD = "hash-8"


WIDE = "wide-model"


MODEL = """
import asyncio
import os
import sys
from hashlib import sha256


class New:
    model = "new-model"
    dimension = 8

    async def embed(self, texts, *, purpose="document"):
        return [[b / 255 for b in sha256(t.encode()).digest()[:8]] for t in texts]
"""


def _adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str = "") -> str:
    name = f"adapter_{uuid4().hex[:8]}"
    (tmp_path / f"{name}.py").write_text(MODEL + body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [*sys.path])
    return name


async def _vectors(action: str, schema: str, *argv: str) -> Ran:
    return await run_cli("vectors", action, "--schema", schema, *argv)


async def _filled(storage: PostgresStorage, corpus_id: str = "acme") -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.ingest(
        corpus_id,
        name="manual",
        sections=[Section(text=f"Line {i} of {corpus_id} runs.") for i in range(3)],
        metadata={"line": "A", "year": 2021},
    )
    await memory.ingest(
        corpus_id,
        name="sheet",
        text=f"Valve notes of {corpus_id}.",
        metadata={"line": "B", "year": 2019},
    )
    await memory.remember(corpus_id, f"The pump of {corpus_id} runs.", metadata={"line": "A"})
    await memory.remember(corpus_id, f"The valve of {corpus_id} leaks.", metadata={"line": "B"})
    return memory


def _failing_after(calls: int, error: str) -> str:
    return f"""
class Failing(New):
    calls = 0

    async def embed(self, texts, *, purpose="document"):
        Failing.calls += 1
        if Failing.calls > {calls}:
            {error}
        return await super().embed(texts, purpose=purpose)
"""


class Turned(HashEmbedding):
    model = NEW
    settings = {"task": "retrieval"}

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [[1 - x for x in vector] for vector in await super().embed(texts, purpose=purpose)]


class Wide(HashEmbedding):
    model = WIDE
    dimension = 16

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [[*vector, *vector] for vector in await super().embed(texts, purpose=purpose)]


Row = tuple[str | None, str | None, str | None]


async def _rows(pool: Any, corpus_id: str = "acme") -> dict[tuple[str, str], Row]:
    held: dict[tuple[str, str], Row] = {}
    async with pool.acquire() as conn:
        for kind, table, key in (
            ("chunk", "memory_chunks", "chunk_id"),
            ("story", "memory_stories", "story_id"),
        ):
            for r in await conn.fetch(
                f"SELECT {key} AS id, embedding_model AS model, embedding::text AS vector, "
                f"embedding_settings::text AS settings FROM {table} WHERE corpus_id = $1",
                corpus_id,
            ):
                held[(kind, r["id"])] = (r["model"], r["vector"], r["settings"])
    return held


async def _staged(
    pool: Any, corpus_id: str = "acme"
) -> dict[tuple[str, str, str], tuple[Any, Any]]:
    async with pool.acquire() as conn:
        return {
            (r["model"], r["passage_kind"], r["passage_id"]): (r["vector"], r["settings"])
            for r in await conn.fetch(
                "SELECT model, passage_kind, passage_id, vector::text AS vector, "
                "settings::text AS settings FROM memory_staged_vectors WHERE corpus_id = $1",
                corpus_id,
            )
        }


async def _filling(
    storage: PostgresStorage, corpus_id: str = "acme", *, limit: int | None = None
) -> Memory:
    new = Memory(storage, embedding=Turned())
    await new.stage_vectors(corpus_id, limit=limit)
    return new


async def _swap(schema: str, model: str = NEW, *argv: str) -> Ran:
    return await _vectors("swap", schema, "--model", model, "--yes", *argv)


async def _forget(schema: str, model: str, *argv: str) -> Ran:
    return await _vectors("forget", schema, "--model", model, "--yes", *argv)


async def _record(pool: Any, corpus_id: str = "acme") -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        held = await conn.fetchval(
            "SELECT vector_index::text FROM memory_corpora WHERE corpus_id = $1", corpus_id
        )
    return None if held is None else json.loads(held)


async def _own(schemas: Schemas) -> tuple[str, Any, PostgresStorage]:
    name = await schemas.create()
    pool = await schemas.pool(name)
    return name, pool, await provision(pool)


async def _waiting(admin: Any, count: int) -> None:
    async with asyncio.timeout(WAIT):
        while count > await admin.fetchval(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
            "AND datname = current_database() AND pid <> pg_backend_pid()"
        ):
            await asyncio.sleep(0.01)


PUMP = "The pump of acme runs."


class LostAtCommit:
    def __init__(self, lost: BaseException) -> None:
        self.lost = lost

    async def execute(self, statement: str) -> None:
        pass

    @asynccontextmanager
    async def transaction(self, isolation: str) -> AsyncIterator[None]:
        yield
        raise self.lost


ODD_CORPUS = "o'brien; $$ 100% \\ drop"


ODD_MODEL = "it's $$ 100% \\ model"


class Odd(Turned):
    model = ODD_MODEL


@asynccontextmanager
async def _running(
    *argv: str, cwd: Path | None = None
) -> AsyncIterator[asyncio.subprocess.Process]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "rfnry_memory.cli",
        *argv[:2],
        "--url",
        os.environ[URL],
        *argv[2:],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(Path(cli.__file__).parents[1])},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        yield process
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def _interrupted(database: Database, *argv: str) -> tuple[int | None, str]:
    async with _running(*argv) as process:
        await _waiting(database.admin, 1)
        process.send_signal(signal.SIGINT)
        async with asyncio.timeout(WAIT):
            _, left = await process.communicate()
    return process.returncode, left.decode()
