from datetime import UTC, datetime
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from database import Schemas, provision
from unit.fakes import HashEmbedding

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.index import HNSW
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import PostgresStorage, migrations
from rfnry_memory.postgres.migrations import Migration, Plan
from rfnry_memory.postgres.schema import _first


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


NOW = migrations.LATEST
NEXT = NOW + 1


def _notes(monkeypatch: pytest.MonkeyPatch) -> Migration:
    notes = Migration(NEXT, "notes", lambda settings: ["CREATE TABLE memory_notes (id INTEGER)"])
    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, notes])
    monkeypatch.setattr(migrations, "LATEST", NEXT)
    return notes


async def test_create_records_the_version_and_the_settings(schemas: Schemas) -> None:
    _, pool = await _fresh(schemas)
    plan = await _apply(
        pool, partial(migrations.plan_create, dimension=8, graph=True, language="Spanish")
    )
    assert plan.steps == [
        "extension vector in public",
        "extension pg_trgm in public",
        "0001 create",
        "0002 spotting",
        "0003 validity",
        "settings (dimension 8, graph on, language spanish)",
    ]
    assert plan.statements[:2] == [
        "CREATE EXTENSION IF NOT EXISTS vector SCHEMA public",
        "CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA public",
    ]
    async with pool.acquire() as conn:
        recorded = [
            tuple(r) for r in await conn.fetch("SELECT version, name FROM memory_migrations")
        ]
        settings = tuple(
            await conn.fetchrow("SELECT dimension, language, config FROM memory_settings")
        )
    assert recorded == [(1, "create"), (2, "spotting"), (3, "validity")]
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
        match=rf"^postgres: {schema} is at version {NOW}, this rfnry-memory needs {NEXT} — run rfnry-memory upgrade --schema {schema}$",
    ):
        await storage.check(dimension=8, graph=True)
    shown = await _apply(pool, migrations.plan_upgrade, run=False)
    assert shown == Plan(
        [f"{NEXT:04d} notes"],
        [
            "CREATE TABLE memory_notes (id INTEGER)",
            f"INSERT INTO memory_migrations (version, name) VALUES ({NEXT}, 'notes')",
        ],
    )
    assert "memory_notes" not in await _tables(pool)
    assert await _apply(pool, migrations.plan_upgrade, expected=shown) == shown
    assert "memory_notes" in await _tables(pool)
    await storage.check(dimension=8, graph=True)
    assert await _apply(pool, migrations.plan_upgrade) == Plan([], [])


async def test_upgrading_from_version_1_reads_every_entity_name_in_its_corpus_language(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema, pool = await _fresh(schemas)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:1])
    monkeypatch.setattr(migrations, "LATEST", 1)
    await _apply(pool, partial(migrations.plan_create, dimension=8, language="english"))
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) VALUES "
            "('en', 'english', 'pg_catalog.english'), ('plain', 'simple', 'pg_catalog.simple')"
        )
        await conn.execute(
            "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases) VALUES "
            """('en', 'e1', 'feed valves', 'Feed valves', 'valve', '["V-101"]'), """
            """('plain', 'e2', 'feed valves', 'Feed valves', 'valve', '["Main  Valve"]'), """
            """('orphan', 'e3', 'feed valves', 'Feed valves', 'valve', '[]'), """
            """('en', 'e4', 'it', 'IT', 'team', '["The Who"]')"""
        )
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    storage = PostgresStorage(pool)
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: {schema} is at version 1, this rfnry-memory needs {NOW} — run rfnry-memory upgrade --schema {schema}$",
    ):
        await storage.check(dimension=8, graph=True)
    assert (await _apply(pool, migrations.plan_upgrade)).steps[0] == "0002 spotting"
    await storage.check(dimension=8, graph=True)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT entity_id, config, spotted::text AS spotted FROM memory_entities "
            "ORDER BY entity_id"
        )
    assert [tuple(r) for r in rows] == [
        ("e1", "pg_catalog.english", "'feed' <-> 'valv' | 'v' <-> '-101'"),
        ("e2", "pg_catalog.simple", "'feed' <-> 'valves' | 'main' <-> 'valve'"),
        ("e3", "pg_catalog.english", "'feed' <-> 'valv'"),
        ("e4", "pg_catalog.english", None),
    ]
    english = await storage.entities_named_in("en", "the feed valve leaks", spot="words")
    assert [e.entity_id for e in english] == ["e1"]
    plain = await storage.entities_named_in("plain", "the feed valves leak", spot="words")
    assert [e.entity_id for e in plain] == ["e2"]
    assert await storage.entities_named_in("plain", "the feed valve leaks", spot="words") == []


async def test_upgrading_to_validity_closes_each_retired_version_where_its_successor_took_over(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pool = await _fresh(schemas)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:2])
    monkeypatch.setattr(migrations, "LATEST", 2)
    await _apply(pool, partial(migrations.plan_create, dimension=8, language="english"))
    jan, feb, mar = (datetime(2026, month, 1, tzinfo=UTC) for month in (1, 2, 3))
    versions = [
        ("v1", jan, feb, "v2"),
        ("v2", feb, mar, "v3"),
        ("v3", mar, None, None),
        ("gone", jan, feb, None),
        ("plain", jan, None, None),
        ("twin-a", jan, feb, "both"),
        ("twin-b", jan, mar, "both"),
        ("both", feb, None, None),
    ]
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config, history) "
            "VALUES ('a', 'english', 'pg_catalog.english', true)"
        )
        for source_id, created, retired, replaced_by in versions:
            await conn.execute(
                "INSERT INTO memory_sources (corpus_id, source_id, name, text, created_at, chunks, "
                "retired_at, replaced_by) VALUES ('a', $1, 'doc', 'x', $2, 1, $3, $4)",
                source_id,
                created,
                retired,
                replaced_by,
            )
            await conn.execute(
                "INSERT INTO memory_chunks (corpus_id, chunk_id, source_id, position, source_name, "
                "text, digest, config, retired_at) "
                "VALUES ('a', $1 || '-0', $1, 0, 'doc', 'x', 'd', 'pg_catalog.english', $2)",
                source_id,
                retired,
            )
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    assert (await _apply(pool, migrations.plan_upgrade)).steps[-1] == f"{NOW:04d} validity"
    async with pool.acquire() as conn:
        sources = await conn.fetch(
            "SELECT source_id, valid_from, valid_to, written_to, confirmed_at = created_at AS same "
            "FROM memory_sources ORDER BY source_id"
        )
        chunks = await conn.fetch(
            "SELECT source_id, valid_from, valid_to, written_to FROM memory_chunks ORDER BY source_id"
        )
    expected = {
        "both": (feb, None),
        "gone": (None, feb),
        "plain": (None, None),
        "twin-a": (None, feb),
        "twin-b": (None, mar),
        "v1": (None, feb),
        "v2": (feb, mar),
        "v3": (mar, None),
    }
    assert {r["source_id"]: (r["valid_from"], r["valid_to"]) for r in sources} == expected
    assert {r["source_id"]: (r["valid_from"], r["valid_to"]) for r in chunks} == expected
    assert all(r["written_to"] is None for r in [*sources, *chunks])
    assert all(r["same"] for r in sources)


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
        await conn.execute(
            "INSERT INTO memory_migrations (version, name) VALUES ($1, 'later')", NEXT
        )
    ahead = rf"^postgres: {schema} is at version {NEXT}, this rfnry-memory knows {NOW} — upgrade rfnry-memory$"
    with pytest.raises(ConfigurationError, match=ahead):
        await storage.check(dimension=8, graph=True)
    with pytest.raises(ConfigurationError, match=ahead):
        await _apply(pool, migrations.plan_upgrade)


async def test_a_new_dimension_deletes_every_vector_and_vector_index(schemas: Schemas) -> None:
    schema, pool = await _fresh(schemas)
    memory = Memory(await provision(pool), embedding=HashEmbedding())
    await memory.ingest("a", name="manual", text="Replace the pump seal yearly.")
    await memory.remember("a", "The seal was replaced.")
    assert await memory.set_vector_index("a", HNSW()) == 2
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
    planned = await _apply(pool, partial(migrations.plan_set, dimension=8))
    assert planned.steps[0] == "extension vector in public"
    assert planned.statements[0] == "CREATE EXTENSION IF NOT EXISTS vector SCHEMA public"
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


async def test_check_names_the_schema_the_search_path_asks_for_when_it_does_not_exist(
    schemas: Schemas,
) -> None:
    missing = f"{schemas.run}_{uuid4().hex[:8]}"
    pool = await schemas.pool(missing)
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: there is no schema {missing} on search_path '{missing}, public' — run rfnry-memory create --schema {missing}$",
    ):
        await PostgresStorage(pool).check(dimension=None, graph=False)


def test_the_first_schema_of_a_search_path_is_read_as_postgres_reads_it() -> None:
    assert _first('"$user", public', "memory", "memory") == "memory"
    assert _first('"$user", public', "memory", "public") == "public"
    assert _first('"Odd ""x"" Name", public', "memory", None) == 'Odd "x" Name'
    assert _first("Memory, public", "memory", None) == "memory"
    assert _first('"$user"', "memory", None) is None


async def test_check_never_sends_memory_into_public(schemas: Schemas) -> None:
    pool = await schemas.pool()
    with pytest.raises(
        ConfigurationError,
        match=r"^postgres: no memory schema on search_path 'public' — run rfnry-memory create and put its schema first on the pool's search_path$",
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


async def test_check_asks_for_the_vector_extension_whenever_the_schema_holds_vectors(
    schemas: Schemas,
) -> None:
    schema, pool = await _fresh(schemas)
    await provision(pool)
    hidden = await schemas.pool(schema, public=False)
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: the vector extension is not reachable from search_path '{schema}' — add the schema that holds it$",
    ):
        await Memory(PostgresStorage(hidden), pillars={"text"}).check()


async def test_upgrade_names_the_command_for_a_schema_without_memory(schemas: Schemas) -> None:
    schema, pool = await _fresh(schemas)
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: no memory schema in {schema} — run rfnry-memory create --schema {schema}$",
    ):
        await _apply(pool, migrations.plan_upgrade)
