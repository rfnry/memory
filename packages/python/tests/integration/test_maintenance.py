import asyncio
import os
from typing import Any

from database import URL, Schemas, provision
from unit.fakes import HashEmbedding

from rfnry_memory import cli
from rfnry_memory.index import HNSW
from rfnry_memory.memory import Memory
from rfnry_memory.models import EntityIn, RelationIn


async def _run(*argv: str) -> int:
    return await asyncio.to_thread(cli.main, [argv[0], "--url", os.environ[URL], *argv[1:]])


async def _filled(schemas: Schemas) -> tuple[str, Memory]:
    name = await schemas.create()
    storage = await provision(await schemas.pool(name))
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    for n in range(12):
        await memory.ingest(
            "acme",
            name=f"manual {n}",
            text=f"Pump P-{n} seal ERR-{n:03d} replaced on line A. The seal kit SK_{n} fits.",
            metadata={"line": "A", "rev": n},
            entities=[
                EntityIn(name=f"Pump P-{n}", type="machine"),
                EntityIn(name="Line A", type="line"),
            ],
            relations=[RelationIn(from_name=f"Pump P-{n}", to_name="Line A", type="on")],
        )
        await memory.remember("acme", f"Ana prefers slot {n} for ERR-{n:03d}.", subject=f"user:{n}")
    await memory.set_vector_index("acme", HNSW(precision="binary"))
    assert (
        await _run("index", "--schema", name, "--key", "metadata.line", "--type", "text", "--yes")
        == 0
    )
    assert (
        await _run("index", "--schema", name, "--key", "metadata.rev", "--type", "number", "--yes")
        == 0
    )
    assert await _run("index", "--schema", name, "--current", "--yes") == 0
    return name, memory


async def _answers(memory: Memory) -> list[list[str]]:
    return [
        [h.id for h in (await memory.search("acme", query)).hits]
        for query in ("seal ERR-007", "Ana slot", "Pump P-3 line A")
    ]


async def _invalid(pool: Any, name: str) -> int:
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                "SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND NOT i.indisvalid",
                name,
            )
        )


async def test_a_schema_is_rebuilt_and_maintained_without_itself_on_the_search_path(
    schemas: Schemas,
) -> None:
    name, memory = await _filled(schemas)
    before = await _answers(memory)
    bare = await schemas.pool(public=False)
    async with bare.acquire() as conn:
        assert await conn.fetchval("SHOW search_path") == ""
        for table in ("memory_chunks", "memory_stories", "memory_entities"):
            await conn.execute(f"UPDATE {name}.{table} SET config = config")
        await conn.execute(f"UPDATE {name}.memory_sources SET text = text")
        await conn.execute(f"REINDEX SCHEMA {name}")
        await conn.execute(
            f"VACUUM (ANALYZE) {name}.memory_sources, {name}.memory_chunks, "
            f"{name}.memory_stories, {name}.memory_entities"
        )
    assert await _invalid(bare, name) == 0
    assert await _answers(memory) == before
