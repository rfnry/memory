import asyncio
import os
from typing import Any

import pytest
from database import URL, Schemas, provision
from unit.fakes import HashEmbedding

from rfnry_memory import cli
from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn
from rfnry_memory.postgres import keys

TABLES = ("chunks", "sources", "stories")


async def _run(*argv: str) -> int:
    return await asyncio.to_thread(cli.main, [argv[0], "--url", os.environ[URL], *argv[1:]])


async def _schema(schemas: Schemas) -> tuple[str, Any, Memory]:
    name = await schemas.create()
    pool = await schemas.pool(name)
    storage = await provision(pool)
    return name, pool, Memory(storage, embedding=HashEmbedding())


async def _objects(pool: Any, key: str) -> dict[str, tuple[bool | None, bool | None]]:
    async with pool.acquire() as conn:
        return await keys._state(conn, keys.tag_of(key))


async def _held(pool: Any) -> list[tuple[str, str, bool]]:
    async with pool.acquire() as conn:
        return await keys.indexed_keys(conn)


async def test_index_builds_a_key_s_indexes_and_check_lists_it(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, memory = await _schema(schemas)
    await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": 120})
    await memory.remember("acme", "Ana owns the seal plan.", metadata={"price": 3.5})
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "  index metadata.price as a number on chunks, sources and stories\n" in out
    assert f"schema {name}: metadata.price is indexed as number" in out
    assert await _objects(pool, "metadata.price") == {table: (True, True) for table in TABLES}
    assert await _held(pool) == [("metadata.price", "number", True)]
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    assert f"schema {name}: nothing to do" in capsys.readouterr().out
    assert await _run("check", "--schema", name) == 0
    assert "\n  metadata.price: number\n" in capsys.readouterr().out


async def test_index_with_sql_prints_its_statements_and_builds_nothing(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, _ = await _schema(schemas)
    assert (
        await _run("index", "--schema", name, "--key", "metadata.rev", "--type", "text", "--sql")
        == 0
    )
    out = capsys.readouterr().out
    assert "CREATE INDEX CONCURRENTLY ix_memory_key_chunks_" in out
    assert "memory_key_text(metadata, ARRAY['rev']::text[]) COLLATE \"C\"" in out
    assert "memory_key_text(metadata::jsonb, ARRAY['rev']::text[])" in out
    assert await _objects(pool, "metadata.rev") == {table: (None, None) for table in TABLES}
    assert await _held(pool) == []


async def test_index_refuses_current_rows_that_break_the_rule_and_ignores_the_past(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, memory = await _schema(schemas)
    await memory.set_history("acme", True)
    old = await memory.ingest(
        "acme", name="pump", text="Pump seal v1.", metadata={"doc": "pump", "price": "120"}
    )
    await memory.sync(
        "acme",
        filters={"metadata.doc": "pump"},
        sources=[
            SourceIn(name="pump", text="Pump seal v2.", metadata={"doc": "pump", "price": 99})
        ],
    )
    bad = await memory.ingest("ops", name="valve", text="Valve.", metadata={"price": [1, 2]})
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 1
    )
    err = capsys.readouterr().err
    assert old.source.source_id not in err
    assert (
        f"cli: 1 current sources hold metadata.price as something other than a number — "
        f"corpus 'ops', source {bad.source.source_id!r}; fix them before indexing it"
    ) in err
    assert (
        "cli: 1 current chunks hold metadata.price as something other than a number — corpus 'ops'"
        in err
    )
    assert await _objects(pool, "metadata.price") == {table: (None, None) for table in TABLES}
    await memory.delete_source("ops", bad.source.source_id)
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    long = "x" * 2001
    await memory.ingest("acme", name="rev", text="Rev.", metadata={"rev": long})
    assert (
        await _run("index", "--schema", name, "--key", "metadata.rev", "--type", "text", "--yes")
        == 1
    )
    assert "as something other than a string of at most 2000 bytes" in capsys.readouterr().err


async def test_a_write_holding_another_type_at_an_indexed_key_is_refused(
    schemas: Schemas,
) -> None:
    name, pool, memory = await _schema(schemas)
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    assert (
        await _run("index", "--schema", name, "--key", "written_by.run", "--type", "text", "--yes")
        == 0
    )
    number = (
        "^postgres: metadata.price is indexed as a number, so a write holding another type or an array "
        "there is refused$"
    )
    for value in ("120", [1, 2], True, {"amount": 1}):
        with pytest.raises(InputError, match=number):
            await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": value})
        with pytest.raises(InputError, match=number):
            await memory.remember("acme", "Ana owns the seal plan.", metadata={"price": value})
    assert (await memory.list_sources("acme"))[0] == []
    text = (
        "^postgres: written_by.run is indexed as text, so a write holding another type, an array or a "
        "string over 2000 bytes there is refused$"
    )
    for run in (5, "r" * 2001):
        with pytest.raises(InputError, match=text):
            await memory.remember("acme", "Ana owns the seal plan.", written_by={"run": run})
    await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": 120})
    await memory.ingest("acme", name="valve", text="Valve.", metadata={"price": None})
    story = await memory.remember("acme", "Ana owns the seal plan.", written_by={"run": "r1"})
    await memory.end_story("acme", story.story.story_id)
    assert len((await memory.list_sources("acme"))[0]) == 2


async def test_drop_removes_the_indexes_and_lifts_the_rule(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, memory = await _schema(schemas)
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    assert await _run("index", "--schema", name, "--key", "metadata.price", "--drop", "--yes") == 0
    assert f"schema {name}: metadata.price is no longer indexed" in capsys.readouterr().out
    assert await _objects(pool, "metadata.price") == {table: (None, None) for table in TABLES}
    assert await _held(pool) == []
    await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": "120"})
    assert await _run("index", "--schema", name, "--key", "metadata.price", "--drop", "--yes") == 0
    assert f"schema {name}: nothing to do" in capsys.readouterr().out


async def test_an_interrupted_build_is_listed_not_ready_and_finished_by_running_again(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, memory = await _schema(schemas)
    async with pool.acquire() as conn:
        plan = await keys.plan_key(conn, "metadata.price", "number")
        for statement in plan.statements[:3]:
            await conn.execute(statement)
    assert await _held(pool) == [("metadata.price", "number", False)]
    assert await _run("check", "--schema", name) == 0
    assert (
        f"  metadata.price: number, not ready — run rfnry-memory index --key metadata.price "
        f"--type number --schema {name}"
    ) in capsys.readouterr().out
    with pytest.raises(InputError, match="^postgres: metadata.price is indexed as a number"):
        await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": "120"})
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    assert await _objects(pool, "metadata.price") == {table: (True, True) for table in TABLES}
    assert await _held(pool) == [("metadata.price", "number", True)]


async def test_index_refuses_a_bad_key_a_type_change_and_an_old_schema(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, _ = await _schema(schemas)
    assert await _run("index", "--schema", name, "--key", "price", "--type", "number") == 1
    assert (
        "postgres: an indexed key is a filter key, metadata.<key> or written_by.<key>; got 'price'"
        in capsys.readouterr().err
    )
    assert await _run("index", "--schema", name, "--key", "metadata.a..b", "--type", "number") == 1
    assert "postgres: 'metadata.a..b' names an empty part" in capsys.readouterr().err
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    assert (
        await _run("index", "--schema", name, "--key", "metadata.price", "--type", "text", "--yes")
        == 1
    )
    assert (
        "postgres: metadata.price is indexed as number — drop it first with rfnry-memory index "
        "--key metadata.price --drop"
    ) in capsys.readouterr().err
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memory_migrations WHERE version = 9")
    assert await _run("index", "--schema", name, "--key", "metadata.size", "--type", "number") == 1
    assert "run rfnry-memory upgrade first" in capsys.readouterr().err


async def test_a_write_refused_by_a_key_dropped_meanwhile_still_says_why(
    schemas: Schemas,
) -> None:
    _, pool, memory = await _schema(schemas)
    async with pool.acquire() as conn:
        plan = await keys.plan_key(conn, "metadata.price", "number")
        for statement in plan.statements[:3]:
            await conn.execute(statement)
        await conn.execute("DELETE FROM memory_indexed_keys")
    with pytest.raises(
        InputError,
        match=r"^postgres: a write holds a value an indexed key refuses \(ck_memory_key_",
    ):
        await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": "120"})
