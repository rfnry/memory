import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest
from database import URL, Schemas, provision
from unit.fakes import HashEmbedding

from rfnry_memory import cli
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import migrations
from rfnry_memory.postgres.migrations import Migration

NOW = migrations.LATEST
NEXT = NOW + 1


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
    assert "0001 create" in out and "0002 spotting" in out
    assert f"schema {name} is at version {NOW}" in out
    assert "  settings (dimension 8, graph on, language french)\n" in out
    memory = Memory(await _storage(made), embedding=HashEmbedding())
    assert (await memory.fingerprint("a")).language == "french"
    assert await _run("check", "--schema", name) == 0
    assert (
        f"schema {name} fits rfnry-memory: version {NOW}, vectors of 8 dimensions"
        in capsys.readouterr().out
    )


async def _storage(pool: Any) -> Any:
    from rfnry_memory.postgres import PostgresStorage

    storage = PostgresStorage(pool)
    await storage.check(dimension=8, graph=True)
    return storage


async def test_create_installs_the_extensions_into_public_on_a_database_without_them(
    schemas: Schemas, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    database = f"{schemas.run}_fresh"
    async with pool.acquire() as conn:
        await conn.execute(f'CREATE DATABASE "{database}"')
    try:
        url = urlunsplit(urlsplit(os.environ[URL])._replace(path=f"/{database}"))
        argv = ["create", "--url", url, "--schema", "memory", "--dimension", "8"]
        assert await asyncio.to_thread(cli.main, argv) == 0
        assert "  extension vector in public\n" in capsys.readouterr().out
        fresh = await asyncpg.connect(url)
        try:
            homes = await fresh.fetch(
                "SELECT e.extname, n.nspname FROM pg_extension e "
                "JOIN pg_namespace n ON n.oid = e.extnamespace "
                "WHERE e.extname IN ('vector', 'pg_trgm') ORDER BY 1"
            )
        finally:
            await fresh.close()
        assert [tuple(r) for r in homes] == [("pg_trgm", "public"), ("vector", "public")]
    finally:
        async with pool.acquire() as conn:
            await conn.execute(f'DROP DATABASE "{database}" WITH (FORCE)')


async def test_create_refuses_a_shared_or_system_schema(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("public", "pg_memory", "information_schema"):
        assert await _run("create", "--schema", name, "--dimension", "8") == 1
        assert (
            capsys.readouterr().err
            == f"cli: memory is not created in {name!r} — name a schema of its own\n"
        )


async def test_every_schema_change_refuses_a_server_older_than_postgres_17(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    held = await schemas.create()
    await provision(await schemas.pool(held))
    empty = await schemas.create()
    served = "current_setting('server_version_num')::int"
    assert served in migrations._STATE
    monkeypatch.setattr(migrations, "_STATE", migrations._STATE.replace(served, "160015"))
    refused = "postgres: memory needs Postgres 17 or later; this server is 16.15\n"
    for argv in (
        ("create", "--schema", empty, "--dimension", "8"),
        ("upgrade", "--schema", held, "--yes"),
        ("set", "--schema", held, "--language", "simple", "--yes"),
        ("index", "--schema", held, "--key", "metadata.line", "--type", "text", "--yes"),
        ("index", "--schema", held, "--current", "--yes"),
    ):
        assert await _run(*argv) == 1, argv
        assert capsys.readouterr().err == refused, argv
    async with (await schemas.pool(empty)).acquire() as conn:
        assert (
            await conn.fetchval("SELECT count(*) FROM pg_tables WHERE schemaname = $1", empty) == 0
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
    assert out.startswith(
        f'SET search_path TO "{name}", public;\nCREATE SCHEMA IF NOT EXISTS "{name}";\n'
    )
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
    notes = Migration(NEXT, "notes", lambda settings: ["CREATE TABLE memory_notes (id INTEGER)"])
    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, notes])
    monkeypatch.setattr(migrations, "LATEST", NEXT)
    assert await _run("check", "--schema", schema) == 1
    assert f"is at version {NOW}, this rfnry-memory needs {NEXT}" in capsys.readouterr().err
    assert await _run("upgrade", "--schema", schema, "--sql") == 0
    assert capsys.readouterr().out == (
        f'SET search_path TO "{schema}", public;\n'
        "CREATE TABLE memory_notes (id INTEGER);\n"
        f"INSERT INTO memory_migrations (version, name) VALUES ({NEXT}, 'notes');\n"
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert await _run("upgrade", "--schema", schema) == 0
    out = capsys.readouterr().out
    assert (
        f"  {NEXT:04d} notes\n" in out
        and "take a snapshot" in out
        and out.endswith("nothing was run\n")
    )
    assert await _run("check", "--schema", schema) == 1
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert await _run("upgrade", "--schema", schema) == 0
    assert capsys.readouterr().out.endswith(f"schema {schema} is at version {NEXT}\n")
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


def test_without_asyncpg_the_command_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None)
    assert cli.main(["check"]) == 1
    assert capsys.readouterr().err == "cli: install rfnry-memory[postgres] to use this command\n"


async def test_a_closed_input_at_the_prompt_runs_nothing(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    schema = await schemas.create()
    await provision(await schemas.pool(schema))
    notes = Migration(NEXT, "notes", lambda settings: ["CREATE TABLE memory_notes (id INTEGER)"])
    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, notes])
    monkeypatch.setattr(migrations, "LATEST", NEXT)

    def closed(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)
    assert await _run("upgrade", "--schema", schema) == 0
    assert capsys.readouterr().out.endswith("nothing was run\n")
    assert await _run("check", "--schema", schema) == 1


def test_an_interrupt_at_the_prompt_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupted)
    with ThreadPoolExecutor(1) as worker:
        asked = worker.submit(cli._confirmed, argparse.Namespace(yes=False))
        assert asked.exception() is None
        assert asked.result() is False
