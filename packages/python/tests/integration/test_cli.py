import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest
from database import URL, Schemas, provision, run_cli
from unit.fakes import HashEmbedding

from rfnry_memory import cli
from rfnry_memory.errors import ConfigurationError
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import PostgresStorage, migrations, schema as schema_state
from rfnry_memory.postgres.migrations import Migration

NOW = migrations.LATEST
NEXT = NOW + 1
LOCK = "SELECT pg_advisory_xact_lock(hashtextextended('memory.migrate:' || current_schema(), 0));"
SERVED = "current_setting('server_version_num')::int"
NAMED = "split_part(current_setting('server_version'), ' ', 1)"
TOO_OLD = "postgres: memory needs Postgres 18 or later; this server is 17.11"


async def _run(*argv: str) -> int:
    return await run_cli(*argv)


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


def _on_postgres_17(state: str) -> str:
    return state.replace(SERVED, "170011").replace(NAMED, "'17.11'")


async def test_every_schema_change_refuses_a_server_older_than_postgres_18(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    held = await schemas.create()
    await provision(await schemas.pool(held))
    empty = await schemas.create()
    assert SERVED in migrations._STATE and NAMED in migrations._STATE
    monkeypatch.setattr(migrations, "_STATE", _on_postgres_17(migrations._STATE))
    for argv in (
        ("create", "--schema", empty, "--dimension", "8"),
        ("upgrade", "--schema", held, "--yes"),
        ("set", "--schema", held, "--language", "simple", "--yes"),
        ("index", "--schema", held, "--key", "metadata.line", "--type", "text", "--yes"),
        ("index", "--schema", held, "--current", "--yes"),
    ):
        assert await _run(*argv) == 1, argv
        assert capsys.readouterr().err == f"{TOO_OLD}\n", argv
    async with (await schemas.pool(empty)).acquire() as conn:
        assert (
            await conn.fetchval("SELECT count(*) FROM pg_tables WHERE schemaname = $1", empty) == 0
        )


async def test_check_and_vectors_refuse_a_server_older_than_postgres_18(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    held = await schemas.create()
    storage = await provision(await schemas.pool(held))
    monkeypatch.setattr(schema_state, "_STATE", _on_postgres_17(schema_state._STATE))
    for argv in (("check", "--schema", held), ("vectors", "status", "--schema", held)):
        assert await _run(*argv) == 1, argv
        assert capsys.readouterr().err == f"{TOO_OLD}\n", argv
    with pytest.raises(ConfigurationError, match=f"^{TOO_OLD}$"):
        await Memory(storage, embedding=HashEmbedding()).check()


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
        f'SET search_path TO "{name}", public;\nBEGIN;\nCREATE SCHEMA IF NOT EXISTS "{name}";\n'
        f"{LOCK}\n"
    )
    assert out.endswith("\nCOMMIT;\n")
    assert "CREATE TABLE memory_chunks" in out and "embedding vector(" not in out
    assert not await _exists(pool, name)
    async with pool.acquire() as conn:
        await conn.execute(out)
        await conn.execute("RESET search_path")
    made = await _made(schemas, name)
    await Memory(PostgresStorage(made), pillars={"text"}).check()


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
        "BEGIN;\n"
        f"{LOCK}\n"
        "CREATE TABLE memory_notes (id INTEGER);\n"
        f"INSERT INTO memory_migrations (version, name) VALUES ({NEXT}, 'notes');\n"
        "COMMIT;\n"
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


async def test_upgrade_sql_names_a_step_rfnry_memory_computes_and_prints_no_data(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:13])
    monkeypatch.setattr(migrations, "LATEST", 13)
    async with pool.acquire() as conn:
        await migrations.apply(
            conn, partial(migrations.plan_create, dimension=8, language="english")
        )
        await conn.execute(
            "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases, config) "
            """VALUES ('a', 'e', 'pump', 'Pump', 'part', '["Secret Name"]', 'pg_catalog.english')"""
        )
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    assert await _run("upgrade", "--schema", schema, "--sql") == 0
    out = capsys.readouterr().out
    assert "\n-- fill memory_entities.alias_keys from aliases, computed by rfnry-memory\n" in out
    assert "Secret Name" not in out and "ALTER TABLE memory_entities ADD COLUMN alias_keys" in out
    assert await _run("upgrade", "--schema", schema, "--yes") == 0
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT alias_keys FROM memory_entities") == ["secret name"]


async def test_set_dimension_with_yes_clears_the_vectors(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    memory = Memory(await provision(pool), embedding=HashEmbedding())
    await memory.ingest("a", name="manual", text="Replace the pump seal yearly.")
    assert await _run("set", "--schema", schema, "--dimension", "16", "--yes") == 0
    assert (
        "every vector on the rows and every vector index is deleted, and staged vectors stay"
        in capsys.readouterr().out
    )
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
    assert cli._scrubbed("dsn memory:hunter2@db is bad", url) == "dsn <url>"
    assert cli._scrubbed("memory refused", url) == "memory refused"


PASTED = [
    ("postgresql://memory:hunter2@db/memory", "hunter2"),
    ("memory:hunter2@db/memory", "hunter2"),
    ("memory:hun ter2@db/memory", "ter2"),
    ("host=db user=memory password=hunter2 dbname=memory", "hunter2"),
    ("host=db password='hun ter2' dbname=memory", "ter2"),
    ("host=db password = hunter2", "hunter2"),
]
PASTED_IDS = [
    "a url",
    "no scheme",
    "a space in the password and no scheme",
    "a keyword dsn",
    "a quoted password with a space",
    "spaces around the equals sign",
]


@pytest.mark.parametrize(("pasted", "secret"), PASTED, ids=PASTED_IDS)
def test_a_line_masks_what_reads_as_a_connection_string_with_no_url_given(
    pasted: str, secret: str
) -> None:
    masked = cli._scrubbed(f"cli: schema memory holds no corpus {pasted!r}", None)
    assert masked.startswith("cli: schema memory holds no corpus ")
    assert "<url>" in masked and secret not in masked


@pytest.mark.parametrize(
    "line",
    [
        "cli: schema memory holds no corpus 'acme'",
        "cli: --embedding is module:attribute; it was given 'adapter'",
        "vendor: no quota left for ops@acme.example",
        "cli: stopped by ValueError: a passage of 12:30 has no vector",
    ],
)
def test_a_line_with_nothing_url_shaped_reads_as_it_was_written(line: str) -> None:
    assert cli._scrubbed(line, None) == line


@pytest.mark.parametrize(("pasted", "secret"), PASTED, ids=PASTED_IDS)
async def test_a_url_pasted_into_schema_or_corpus_never_prints(
    pasted: str, secret: str, schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name = await schemas.create()
    await provision(await schemas.pool(name))
    for argv in (
        ("check", "--schema", pasted),
        ("upgrade", "--schema", pasted),
        ("vectors", "status", "--schema", name, "--corpus", pasted),
    ):
        assert await _run(*argv) == 1
        printed = capsys.readouterr()
        assert printed.err.startswith("cli: "), printed.err
        assert secret not in printed.out + printed.err


def test_a_schema_refused_by_name_does_not_print_a_url_either(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["check", "--schema", "pg_memory://memory:hunter2@db"]) == 1
    printed = capsys.readouterr()
    assert printed.err.startswith("cli: memory is not created in ")
    assert "hunter2" not in printed.out + printed.err


SECRET = "postgresql://memory:hunter2@127.0.0.1:1/memory"


@pytest.mark.parametrize(
    ("argv", "secret"),
    [
        (["vectors", "--url", SECRET, "status"], "hunter2"),
        (["vectors", f"--url={SECRET}", "status"], "hunter2"),
        (["vectors", "status", SECRET], "hunter2"),
        (["vectors", "status", f"--ulr={SECRET}"], "hunter2"),
        (["--url", SECRET, "check"], "hunter2"),
        ([f"--url={SECRET}", "check"], "hunter2"),
        (["check", SECRET], "hunter2"),
        (["create", "--dimension", SECRET], "hunter2"),
        (["vectors", "memory:hunter2@127.0.0.1:1/memory", "status"], "hunter2"),
        (["vectors", "postgresql://memory:hun ter2@127.0.0.1:1/memory", "status"], "ter2"),
        (["vectors", "memory:hun ter2@127.0.0.1:1/memory", "status"], "ter2"),
        (["vectors", "host=db password=hunter2", "status"], "hunter2"),
    ],
    ids=[
        "--url before the action",
        "--url= before the action",
        "a stray url after the action",
        "a mistyped flag after the action",
        "--url before the command",
        "--url= before the command",
        "a stray url after the command",
        "a url where a number goes",
        "no scheme",
        "a space in the password",
        "a space in the password and no scheme",
        "a keyword dsn",
    ],
)
def test_a_bad_command_line_never_prints_the_connection_string(
    argv: list[str], secret: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(argv)
    assert exited.value.code == 2
    printed = capsys.readouterr()
    assert "error: " in printed.err and "<url>" in printed.err
    assert secret not in printed.out + printed.err


def test_a_bad_command_line_with_no_url_reads_as_argparse_wrote_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(["vectors", "statsu"])
    assert "argument action: invalid choice: 'statsu' (choose from " in capsys.readouterr().err


def test_a_malformed_url_still_scrubs_and_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    url = "postgres://u:sekret@[bad/db"
    assert cli._scrubbed(f"reach {url} as u:sekret@db", url) == "reach <url> as <url>"
    assert cli.main(["check", "--url", url]) == 1
    printed = capsys.readouterr()
    assert printed.err.startswith("cli: ")
    assert "sekret" not in printed.out + printed.err


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


async def test_a_printed_upgrade_that_fails_part_way_leaves_the_schema_as_it_was(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    await provision(pool)
    assert await _run("set", "--schema", schema, "--language", "german", "--sql") == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[1:3] == ["BEGIN;", LOCK] and printed[-1] == "COMMIT;"
    assert printed[3].startswith("UPDATE memory_settings SET language = 'german'")
    assert await _run("index", "--schema", schema, "--current", "--sql") == 0
    assert "BEGIN;" not in capsys.readouterr().out
    broken = Migration(
        NEXT,
        "broken",
        lambda settings: ["CREATE TABLE memory_notes (id INTEGER)", "SELECT 1 / 0"],
    )
    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, broken])
    monkeypatch.setattr(migrations, "LATEST", NEXT)
    assert await _run("upgrade", "--schema", schema, "--sql") == 0
    out = capsys.readouterr().out
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.DivisionByZeroError):
            await conn.execute(out)
        await conn.execute("ROLLBACK")
        assert await conn.fetchval("SELECT to_regclass('memory_notes')") is None
        assert await conn.fetchval("SELECT max(version) FROM memory_migrations") == NOW


async def test_upgrade_to_stories_listed_prints_the_index_it_swaps_and_check_then_passes(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:15])
    monkeypatch.setattr(migrations, "LATEST", 15)
    await provision(pool)
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:16])
    monkeypatch.setattr(migrations, "LATEST", 16)
    assert await _run("check", "--schema", schema) == 1
    assert "is at version 15, this rfnry-memory needs 16" in capsys.readouterr().err
    assert await _run("upgrade", "--schema", schema, "--sql") == 0
    assert capsys.readouterr().out == (
        f'SET search_path TO "{schema}", public;\n'
        "BEGIN;\n"
        f"{LOCK}\n"
        "DROP INDEX ix_memory_stories_valid_from;\n"
        'CREATE INDEX ix_memory_stories_listed ON memory_stories (corpus_id, valid_from DESC, story_id COLLATE "C" DESC);\n'
        "INSERT INTO memory_migrations (version, name) VALUES (16, 'stories listed');\n"
        "COMMIT;\n"
    )
    assert await _run("check", "--schema", schema) == 1
    capsys.readouterr()
    assert await _run("upgrade", "--schema", schema, "--yes") == 0
    assert capsys.readouterr().out.endswith(f"schema {schema} is at version 16\n")
    assert await _run("check", "--schema", schema) == 0
    assert "version 16" in capsys.readouterr().out
