from functools import partial
from typing import Any

import pytest
from database import Schemas, provision
from unit.fakes import HashEmbedding

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import PostgresStorage, migrations
from rfnry_memory.postgres.migrations import Migration, Plan


async def _fresh(schemas: Schemas) -> tuple[str, Any]:
    schema = await schemas.create()
    return schema, await schemas.pool(schema)


async def _apply(pool: Any, planner: Any, **options: Any) -> Plan:
    async with pool.acquire() as conn:
        return await migrations.apply(conn, planner, **options)


async def _tables(pool: Any) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() ORDER BY 1"
        )
    return [r["tablename"] for r in rows]


def _notes(monkeypatch: pytest.MonkeyPatch) -> Migration:
    notes = Migration(2, "notes", lambda settings: ["CREATE TABLE memory_notes (id INTEGER)"])
    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, notes])
    monkeypatch.setattr(migrations, "LATEST", 2)
    return notes


async def test_create_records_the_version_and_the_settings(schemas: Schemas) -> None:
    _, pool = await _fresh(schemas)
    plan = await _apply(
        pool, partial(migrations.plan_create, dimension=8, graph=True, language="Spanish")
    )
    assert plan.steps == ["extension vector", "extension pg_trgm", "0001 create", "settings"]
    async with pool.acquire() as conn:
        recorded = [
            tuple(r) for r in await conn.fetch("SELECT version, name FROM memory_migrations")
        ]
        settings = tuple(
            await conn.fetchrow("SELECT dimension, language, config FROM memory_settings")
        )
    assert recorded == [(1, "create")]
    assert settings == (8, "spanish", "pg_catalog.spanish")
    await PostgresStorage(pool).check(dimension=8, graph=True)


async def test_create_without_vectors_makes_no_embedding_column(schemas: Schemas) -> None:
    schema, pool = await _fresh(schemas)
    await provision(pool, dimension=None)
    async with pool.acquire() as conn:
        columns = await conn.fetch(
            "SELECT table_name FROM information_schema.columns WHERE table_schema = $1 "
            "AND column_name = 'embedding'",
            schema,
        )
        held = await conn.fetchval("SELECT dimension FROM memory_settings")
    assert columns == [] and held is None


async def test_create_refuses_a_schema_that_already_holds_memory_tables(schemas: Schemas) -> None:
    schema, pool = await _fresh(schemas)
    await provision(pool)
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: {schema} already holds memory tables — run rfnry-memory upgrade, or create into another schema$",
    ):
        await _apply(pool, partial(migrations.plan_create, dimension=8))


async def test_create_refuses_a_shared_schema_and_runs_nothing_there(schemas: Schemas) -> None:
    pool = await schemas.pool()
    with pytest.raises(
        InputError,
        match=r"^postgres: memory is not created in 'public' — name a schema of its own$",
    ):
        await _apply(pool, partial(migrations.plan_create, dimension=8))


async def test_an_unknown_language_fails_before_anything_runs(schemas: Schemas) -> None:
    _, pool = await _fresh(schemas)
    with pytest.raises(
        InputError,
        match=r"^postgres: 'klingon' is not a text search configuration in this database$",
    ):
        await _apply(pool, partial(migrations.plan_create, dimension=8, language="klingon"))
    assert await _tables(pool) == []


async def test_a_dry_run_shows_the_plan_and_runs_nothing(schemas: Schemas) -> None:
    _, pool = await _fresh(schemas)
    plan = await _apply(pool, partial(migrations.plan_create, dimension=8), run=False)
    assert any(s.startswith("CREATE TABLE memory_chunks") for s in plan.statements)
    assert plan.statements[-1].startswith("INSERT INTO memory_settings")
    assert await _tables(pool) == []


async def test_upgrade_with_nothing_pending_does_nothing(schemas: Schemas) -> None:
    _, pool = await _fresh(schemas)
    await provision(pool)
    assert await _apply(pool, migrations.plan_upgrade) == Plan([], [])


async def test_upgrade_runs_a_new_migration_once_and_records_it(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema, pool = await _fresh(schemas)
    storage = await provision(pool)
    _notes(monkeypatch)
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: {schema} is at version 1, this rfnry-memory needs 2 — run rfnry-memory upgrade --schema {schema}$",
    ):
        await storage.check(dimension=8, graph=True)
    shown = await _apply(pool, migrations.plan_upgrade, run=False)
    assert shown == Plan(
        ["0002 notes"],
        [
            "CREATE TABLE memory_notes (id INTEGER)",
            "INSERT INTO memory_migrations (version, name) VALUES (2, 'notes')",
        ],
    )
    assert "memory_notes" not in await _tables(pool)
    assert await _apply(pool, migrations.plan_upgrade, expected=shown) == shown
    assert "memory_notes" in await _tables(pool)
    await storage.check(dimension=8, graph=True)
    assert await _apply(pool, migrations.plan_upgrade) == Plan([], [])


async def test_a_plan_the_schema_has_moved_past_is_refused(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pool = await _fresh(schemas)
    await provision(pool)
    _notes(monkeypatch)
    shown = await _apply(pool, migrations.plan_upgrade, run=False)
    await _apply(pool, migrations.plan_upgrade)
    with pytest.raises(
        ConfigurationError,
        match=r"^postgres: the schema changed since the plan was shown — run the command again$",
    ):
        await _apply(pool, migrations.plan_upgrade, expected=shown)


async def test_a_schema_ahead_of_this_version_is_refused(schemas: Schemas) -> None:
    schema, pool = await _fresh(schemas)
    storage = await provision(pool)
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO memory_migrations (version, name) VALUES (2, 'later')")
    ahead = (
        rf"^postgres: {schema} is at version 2, this rfnry-memory knows 1 — upgrade rfnry-memory$"
    )
    with pytest.raises(ConfigurationError, match=ahead):
        await storage.check(dimension=8, graph=True)
    with pytest.raises(ConfigurationError, match=ahead):
        await _apply(pool, migrations.plan_upgrade)


async def test_a_new_dimension_deletes_every_vector_and_vector_index(schemas: Schemas) -> None:
    schema, pool = await _fresh(schemas)
    memory = Memory(await provision(pool), embedding=HashEmbedding())
    await memory.ingest("a", name="manual", text="Replace the pump seal yearly.")
    await memory.remember("a", "The seal was replaced.")
    assert await memory.set_vector_index("a", True) == 2
    plan = await _apply(pool, partial(migrations.plan_set, dimension=16))
    assert plan.steps == [
        "vectors of 16 dimensions: every stored vector and vector index is deleted"
    ]
    async with pool.acquire() as conn:
        typed = await conn.fetch(
            "SELECT atttypmod FROM pg_attribute WHERE attname = 'embedding' AND attrelid IN "
            "(to_regclass(format('%I.memory_chunks', current_schema())), "
            "to_regclass(format('%I.memory_stories', current_schema())))"
        )
        stored = await conn.fetchval(
            "SELECT count(*) FROM (SELECT embedding, embedding_model FROM memory_chunks UNION ALL "
            "SELECT embedding, embedding_model FROM memory_stories) AS p "
            "WHERE embedding IS NOT NULL OR embedding_model IS NOT NULL"
        )
    assert [r["atttypmod"] for r in typed] == [16, 16] and stored == 0
    assert not [t for t in await _tables(pool) if t.startswith("memory_vectors_")]
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: {schema} holds vectors of 16 dimensions; this memory embeds 8 — run rfnry-memory set --dimension 8 --schema {schema}, or use another schema$",
    ):
        await memory.check()
    await PostgresStorage(pool).check(dimension=16, graph=True)
    assert await _apply(pool, partial(migrations.plan_set, dimension=16)) == Plan([], [])


async def test_a_dimension_on_a_schema_without_vectors_adds_the_columns(schemas: Schemas) -> None:
    _, pool = await _fresh(schemas)
    await provision(pool, dimension=None)
    await _apply(pool, partial(migrations.plan_set, dimension=8))
    memory = Memory(PostgresStorage(pool), embedding=HashEmbedding())
    await memory.check()
    await memory.ingest("a", name="manual", text="Replace the pump seal yearly.")
    hits = await memory.search("a", "pump seal", pillars={"vector"})
    assert [h.kind for h in hits.hits] == ["chunk"]


async def test_graph_on_a_database_that_has_pg_trgm_changes_nothing(schemas: Schemas) -> None:
    _, pool = await _fresh(schemas)
    await provision(pool)
    assert await _apply(pool, partial(migrations.plan_set, graph=True)) == Plan([], [])


async def test_set_changes_exactly_one_setting(schemas: Schemas) -> None:
    _, pool = await _fresh(schemas)
    await provision(pool)
    with pytest.raises(
        InputError,
        match=r"^postgres: set changes one setting: a dimension, the graph or a language$",
    ):
        await _apply(pool, partial(migrations.plan_set, dimension=16, language="french"))


async def test_check_names_a_schema_without_memory(schemas: Schemas) -> None:
    schema, pool = await _fresh(schemas)
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: no memory schema in {schema} — run rfnry-memory create --schema {schema}$",
    ):
        await PostgresStorage(pool).check(dimension=None, graph=False)


async def test_check_names_tables_from_before_migrations(schemas: Schemas) -> None:
    schema, pool = await _fresh(schemas)
    async with pool.acquire() as conn:
        await conn.execute("CREATE TABLE memory_sources (corpus_id TEXT)")
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: the memory tables in {schema} predate migrations — create a new schema with rfnry-memory create$",
    ):
        await PostgresStorage(pool).check(dimension=None, graph=False)


async def test_check_names_a_dimension_the_schema_does_not_hold(schemas: Schemas) -> None:
    eight_schema, eight = await _fresh(schemas)
    bare_schema, bare = await _fresh(schemas)
    await provision(eight)
    await provision(bare, dimension=None)
    await PostgresStorage(eight).check(dimension=None, graph=True)
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: {eight_schema} holds vectors of 8 dimensions; this memory embeds 1024 — ",
    ):
        await PostgresStorage(eight).check(dimension=1024, graph=True)
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: {bare_schema} holds no vectors; this memory embeds 1024 — ",
    ):
        await PostgresStorage(bare).check(dimension=1024, graph=True)


async def test_check_reads_the_first_schema_on_the_search_path(schemas: Schemas) -> None:
    below, above = await schemas.create(), await schemas.create()
    lower, both = await schemas.pool(below), await schemas.pool(above, below)
    await provision(lower)
    await provision(both, dimension=16)
    await PostgresStorage(both).check(dimension=16, graph=True)
    await PostgresStorage(lower).check(dimension=8, graph=True)
    with pytest.raises(ConfigurationError, match=rf"^postgres: {above} holds vectors of 16"):
        await PostgresStorage(both).check(dimension=8, graph=True)


async def test_check_names_every_extension_the_search_path_cannot_reach(schemas: Schemas) -> None:
    schema, pool = await _fresh(schemas)
    await provision(pool)
    hidden = await schemas.pool(schema, public=False)
    with pytest.raises(ConfigurationError) as raised:
        await PostgresStorage(hidden).check(dimension=8, graph=True)
    assert str(raised.value) == (
        f"postgres: the vector extension is not reachable from search_path '{schema}' — add the schema that holds it; "
        f"the pg_trgm extension is not reachable from search_path '{schema}' — add the schema that holds it"
    )
