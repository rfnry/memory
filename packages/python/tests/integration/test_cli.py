import asyncio
import os
from typing import Any
from uuid import uuid4

import pytest
from database import URL, Schemas, provision
from unit.fakes import HashEmbedding

from rfnry_memory import cli
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import migrations
from rfnry_memory.postgres.migrations import Migration


async def _run(*argv: str) -> int:
    return await asyncio.to_thread(cli.main, [argv[0], "--url", os.environ[URL], *argv[1:]])


def _named(schemas: Schemas, suffix: str = "") -> str:
    return f"{schemas.run}_{uuid4().hex[:8]}{suffix}"


async def _made(schemas: Schemas, name: str) -> Any:
    quoted = '"' + name.replace('"', '""') + '"'
    schemas._made.append(quoted)
    return await schemas.pool(quoted)


async def _exists(pool: Any, name: str) -> bool:
    async with pool.acquire() as conn:
        return bool(
            await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)", name
            )
        )


async def test_create_makes_a_schema_the_app_can_check(
    schemas: Schemas, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    name = _named(schemas)
    assert await _run("create", "--schema", name, "--dimension", "8", "--language", "french") == 0
    made = await _made(schemas, name)
    out = capsys.readouterr().out
    assert "0001 create" in out and f"schema {name} is at version 1" in out
    memory = Memory(await _storage(made), embedding=HashEmbedding())
    assert (await memory.fingerprint("a")).language == "french"
    assert await _run("check", "--schema", name) == 0
    assert (
        f"schema {name} fits rfnry-memory: version 1, vectors of 8 dimensions"
        in capsys.readouterr().out
    )


async def _storage(pool: Any) -> Any:
    from rfnry_memory.postgres import PostgresStorage

    storage = PostgresStorage(pool)
    await storage.check(dimension=8, graph=True)
    return storage


async def test_create_refuses_a_shared_or_system_schema(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("public", "pg_memory", "information_schema"):
        assert await _run("create", "--schema", name, "--dimension", "8") == 1
        assert (
            capsys.readouterr().err
            == f"cli: memory is not created in {name!r} — name a schema of its own\n"
        )


async def test_create_refuses_a_schema_that_holds_memory_tables(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    schema = await schemas.create()
    await provision(await schemas.pool(schema))
    assert await _run("create", "--schema", schema, "--dimension", "8") == 1
    assert "already holds memory tables" in capsys.readouterr().err


async def test_create_with_sql_prints_every_statement_and_makes_nothing(
    schemas: Schemas, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    name = _named(schemas)
    assert await _run("create", "--schema", name, "--no-vector", "--sql") == 0
    out = capsys.readouterr().out
    assert out.startswith(f'CREATE SCHEMA IF NOT EXISTS "{name}";\n')
    assert "CREATE TABLE memory_chunks" in out and "embedding vector(" not in out
    assert not await _exists(pool, name)


async def test_an_unknown_language_creates_nothing(
    schemas: Schemas, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    name = _named(schemas)
    assert await _run("create", "--schema", name, "--dimension", "8", "--language", "klingon") == 1
    assert (
        capsys.readouterr().err
        == "postgres: 'klingon' is not a text search configuration in this database\n"
    )
    assert not await _exists(pool, name)


async def test_a_schema_name_that_needs_quoting_works_through_every_command(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name = _named(schemas, " Odd-Name")
    assert await _run("create", "--schema", name, "--dimension", "8") == 0
    await _made(schemas, name)
    assert await _run("check", "--schema", name) == 0
    assert await _run("upgrade", "--schema", name) == 0
    assert await _run("set", "--schema", name, "--language", "german", "--yes") == 0
    assert f"schema {name}: nothing to do" in capsys.readouterr().out


async def test_upgrade_asks_before_it_runs_and_sql_runs_nothing(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    await provision(pool)
    assert await _run("upgrade", "--schema", schema) == 0
    assert capsys.readouterr().out == f"schema {schema}: nothing to do\n"
    notes = Migration(2, "notes", lambda settings: ["CREATE TABLE memory_notes (id INTEGER)"])
    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, notes])
    monkeypatch.setattr(migrations, "LATEST", 2)
    assert await _run("check", "--schema", schema) == 1
    assert "is at version 1, this rfnry-memory needs 2" in capsys.readouterr().err
    assert await _run("upgrade", "--schema", schema, "--sql") == 0
    assert capsys.readouterr().out == (
        "CREATE TABLE memory_notes (id INTEGER);\n"
        "INSERT INTO memory_migrations (version, name) VALUES (2, 'notes');\n"
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert await _run("upgrade", "--schema", schema) == 0
    out = capsys.readouterr().out
    assert (
        "  0002 notes\n" in out and "take a snapshot" in out and out.endswith("nothing was run\n")
    )
    assert await _run("check", "--schema", schema) == 1
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert await _run("upgrade", "--schema", schema) == 0
    assert capsys.readouterr().out.endswith(f"schema {schema} is at version 2\n")
    assert await _run("check", "--schema", schema) == 0


async def test_set_dimension_with_yes_clears_the_vectors(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    memory = Memory(await provision(pool), embedding=HashEmbedding())
    await memory.ingest("a", name="manual", text="Replace the pump seal yearly.")
    assert await _run("set", "--schema", schema, "--dimension", "16", "--yes") == 0
    assert "every stored vector and vector index is deleted" in capsys.readouterr().out
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(embedding) FROM memory_chunks") == 0
        assert await conn.fetchval("SELECT dimension FROM memory_settings") == 16


async def test_check_and_upgrade_name_a_schema_that_does_not_exist(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name = _named(schemas)
    for command in ("check", "upgrade"):
        assert await _run(command, "--schema", name) == 1
        assert (
            capsys.readouterr().err
            == f"cli: there is no schema {name} — run rfnry-memory create --schema {name}\n"
        )


async def test_a_connection_string_never_appears_in_what_the_command_prints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    url = "postgresql://memory:hunter2@127.0.0.1:1/memory"
    assert await asyncio.to_thread(cli.main, ["check", "--url", url]) == 1
    printed = capsys.readouterr()
    assert printed.err.startswith("cli: ")
    assert "hunter2" not in printed.out + printed.err and url not in printed.err


def test_a_message_that_quotes_the_connection_string_is_scrubbed() -> None:
    url = "postgresql://memory:hunter2@db/memory"
    assert cli._scrubbed(f"cannot reach {url}", url) == "cannot reach <url>"
    assert cli._scrubbed("dsn memory:hunter2@db is bad", url) == "dsn memory:***@db is bad"
    assert cli._scrubbed("memory refused", url) == "memory refused"
