import json
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
from rfnry_memory.models import Window
from rfnry_memory.postgres import PostgresStorage, migrations
from rfnry_memory.postgres.migrations import Call, Migration, Plan
from rfnry_memory.postgres.schema import _first
from rfnry_memory.rows import alias_keys, name_key


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
        "0004 attribution",
        "0005 resemblance",
        "0006 lengths",
        "0007 indexes",
        "0008 refinery",
        "0009 indexed keys",
        "0010 codes",
        "0011 statistics",
        "0012 key moments",
        "0013 function bodies",
        "0014 graph writes",
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
    assert recorded == [
        (1, "create"),
        (2, "spotting"),
        (3, "validity"),
        (4, "attribution"),
        (5, "resemblance"),
        (6, "lengths"),
        (7, "indexes"),
        (8, "refinery"),
        (9, "indexed keys"),
        (10, "codes"),
        (11, "statistics"),
        (12, "key moments"),
        (13, "function bodies"),
        (14, "graph writes"),
    ]
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
    assert any(str(s).startswith("CREATE TABLE memory_chunks") for s in plan.statements)
    assert str(plan.statements[-1]).startswith("INSERT INTO memory_settings")
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
    assert (await _apply(pool, migrations.plan_upgrade)).steps[0] == "0003 validity"
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


async def test_upgrading_to_attribution_marks_every_closed_row_as_closed_by_no_one(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pool = await _fresh(schemas)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:3])
    monkeypatch.setattr(migrations, "LATEST", 3)
    await _apply(pool, partial(migrations.plan_create, dimension=8, language="english"))
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) "
            "VALUES ('a', 'english', 'pg_catalog.english')"
        )
        for story_id, closed in (("open", None), ("closed", feb)):
            await conn.execute(
                "INSERT INTO memory_stories (corpus_id, story_id, text, text_hash, valid_from, valid_to, "
                "created_at, closed_at, written_to, metadata, config, embedding_model) "
                "VALUES ('a', $1, $1, $1, $2, $3, $2, $3, NULL, '{}', 'pg_catalog.english', NULL)",
                story_id,
                jan,
                closed,
            )
        for source_id, retired in (("held", None), ("retired", feb)):
            await conn.execute(
                "INSERT INTO memory_sources (corpus_id, source_id, name, text, created_at, chunks, "
                "retired_at, confirmed_at) VALUES ('a', $1, 'doc', 'x', $2, 0, $3, $2)",
                source_id,
                jan,
                retired,
            )
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    assert (await _apply(pool, migrations.plan_upgrade)).steps[0] == "0004 attribution"
    async with pool.acquire() as conn:
        stories = await conn.fetch("SELECT story_id, written_by, closed_by FROM memory_stories")
        sources = await conn.fetch("SELECT source_id, written_by, closed_by FROM memory_sources")
    assert {r["story_id"]: (r["written_by"], r["closed_by"]) for r in stories} == {
        "open": ("{}", None),
        "closed": ("{}", "{}"),
    }
    assert {r["source_id"]: (r["written_by"], r["closed_by"]) for r in sources} == {
        "held": ("{}", None),
        "retired": ("{}", "{}"),
    }


async def test_upgrading_to_resemblance_signs_every_stored_document(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pool = await _fresh(schemas)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:4])
    monkeypatch.setattr(migrations, "LATEST", 4)
    await _apply(pool, partial(migrations.plan_create, dimension=8, language="english"))
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    texts = {
        "policy": "Returns are accepted within thirty days of delivery with a receipt.",
        "renamed": "Returns are accepted within thirty days of delivery with a receipt.",
        "other": "The pump seal is replaced every year by the night shift.",
        "empty": "-- !! --",
    }
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) "
            "VALUES ('a', 'english', 'pg_catalog.english')"
        )
        for source_id, text in texts.items():
            await conn.execute(
                "INSERT INTO memory_sources (corpus_id, source_id, name, text, created_at, chunks, "
                "confirmed_at) VALUES ('a', $1, $1, $2, $3, 0, $3)",
                source_id,
                text,
                jan,
            )
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    assert (await _apply(pool, migrations.plan_upgrade)).steps[0] == "0005 resemblance"
    async with pool.acquire() as conn:
        signed = {
            r["source_id"]: r["signature"]
            for r in await conn.fetch("SELECT source_id, signature FROM memory_sources")
        }
        shares = await conn.fetchrow(
            "SELECT memory_share($1::bigint[], $2::bigint[]) AS same, "
            "memory_share($1::bigint[], $3::bigint[]) AS apart",
            signed["policy"],
            signed["renamed"],
            signed["other"],
        )
    assert signed["policy"] == signed["renamed"] and len(signed["policy"]) == 128
    assert signed["empty"] is None
    assert (shares["same"], shares["apart"]) == (1.0, 0.0)


async def test_upgrading_to_lengths_counts_every_stored_passage(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pool = await _fresh(schemas)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:5])
    monkeypatch.setattr(migrations, "LATEST", 5)
    await _apply(pool, partial(migrations.plan_create, dimension=8, language="english"))
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    text = "The seal of the pump is replaced every year."
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) "
            "VALUES ('a', 'english', 'pg_catalog.english')"
        )
        await conn.execute(
            "INSERT INTO memory_sources (corpus_id, source_id, name, text, created_at, chunks, "
            "confirmed_at) VALUES ('a', 's', 'manual', $1, $2, 1, $2)",
            text,
            jan,
        )
        await conn.execute(
            "INSERT INTO memory_chunks (corpus_id, chunk_id, source_id, position, source_name, text, "
            "digest, config) VALUES ('a', 's-0', 's', 0, 'manual', $1, 'd', 'pg_catalog.english')",
            text,
        )
        await conn.execute(
            "INSERT INTO memory_stories (corpus_id, story_id, text, text_hash, valid_from, created_at, "
            "config) VALUES ('a', 't', $1, 'h', $2, $2, 'pg_catalog.english')",
            text,
            jan,
        )
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    assert (await _apply(pool, migrations.plan_upgrade)).steps[0] == "0006 lengths"
    counted = "(SELECT sum(coalesce(array_length(u.positions, 1), 1)) FROM unnest(fts) AS u)::int"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT fts_length, {counted} AS counted FROM memory_chunks UNION ALL "
            f"SELECT fts_length, {counted} FROM memory_stories"
        )
    assert [(r["fts_length"], r["counted"]) for r in rows] == [(6, 6), (5, 5)]


async def test_upgrading_to_indexes_serves_versions_retirements_and_listings(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pool = await _fresh(schemas)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:6])
    monkeypatch.setattr(migrations, "LATEST", 6)
    await _apply(pool, partial(migrations.plan_create, dimension=8, language="english"))
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    assert (await _apply(pool, migrations.plan_upgrade)).steps[0] == "0007 indexes"
    async with pool.acquire() as conn:
        made = {
            r["indexname"]: r["indexdef"].split(" USING ")[1]
            for r in await conn.fetch(
                "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = current_schema() "
                "AND tablename = 'memory_sources'"
            )
        }
    assert made["ix_memory_sources_replaced_by"] == (
        "btree (corpus_id, replaced_by) WHERE (replaced_by IS NOT NULL)"
    )
    assert made["ix_memory_sources_retired"] == (
        "btree (corpus_id, retired_at) WHERE (retired_at IS NOT NULL)"
    )
    assert made["ix_memory_sources_listed"] == (
        'btree (corpus_id, created_at DESC, source_id COLLATE "C" DESC)'
    )


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


async def test_upgrading_to_codes_keeps_every_stored_code_whole(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pool = await _fresh(schemas)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:9])
    monkeypatch.setattr(migrations, "LATEST", 9)
    await _apply(pool, partial(migrations.plan_create, dimension=8, language="english"))
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    text = "Replace part A-1001 on Line3_PLC:Motor1_Run, token " + "-".join(["ab12"] * 600) + "."
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) "
            "VALUES ('a', 'english', 'pg_catalog.english')"
        )
        await conn.execute(
            "INSERT INTO memory_sources (corpus_id, source_id, name, text, created_at, chunks, "
            "confirmed_at) VALUES ('a', 's', 'manual', $1, $2, 1, $2)",
            text,
            jan,
        )
        await conn.execute(
            "INSERT INTO memory_chunks (corpus_id, chunk_id, source_id, position, source_name, text, "
            "digest, config) VALUES ('a', 's-0', 's', 0, 'manual', $1, 'd', 'pg_catalog.english')",
            text,
        )
        await conn.execute(
            "INSERT INTO memory_stories (corpus_id, story_id, text, text_hash, valid_from, created_at, "
            "config) VALUES ('a', 't', $1, 'h', $2, $2, 'pg_catalog.english')",
            text,
            jan,
        )
        before = await conn.fetch(
            "SELECT fts_length FROM memory_chunks UNION ALL SELECT fts_length FROM memory_stories"
        )
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    assert (await _apply(pool, migrations.plan_upgrade)).steps[0] == "0010 codes"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT fts_length, tsvector_to_array(fts) AS words FROM memory_chunks UNION ALL "
            "SELECT fts_length, tsvector_to_array(fts) FROM memory_stories"
        )
    for row in rows:
        assert {"a-1001", "line3_plc:motor1_run"} <= set(row["words"])
    assert [r["fts_length"] for r in rows] == [r["fts_length"] for r in before]


async def test_upgrading_to_function_bodies_lets_a_schema_be_rebuilt_without_its_search_path(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema, pool = await _fresh(schemas)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:12])
    monkeypatch.setattr(migrations, "LATEST", 12)
    await _apply(pool, partial(migrations.plan_create, dimension=8, language="english"))
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    text = "Replace part A-1001 on line 3."
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) "
            "VALUES ('a', 'english', 'pg_catalog.english')"
        )
        await conn.execute(
            "INSERT INTO memory_sources (corpus_id, source_id, name, text, created_at, chunks, "
            "confirmed_at) VALUES ('a', 's', 'manual', $1, $2, 1, $2)",
            text,
            jan,
        )
        await conn.execute(
            "INSERT INTO memory_chunks (corpus_id, chunk_id, source_id, position, source_name, text, "
            "digest, config) VALUES ('a', 's-0', 's', 0, 'manual', $1, 'd', 'pg_catalog.english')",
            text,
        )
        await conn.execute(
            "INSERT INTO memory_stories (corpus_id, story_id, text, text_hash, valid_from, created_at, "
            "config) VALUES ('a', 't', $1, 'h', $2, $2, 'pg_catalog.english')",
            text,
            jan,
        )
        before = await conn.fetch(
            "SELECT fts FROM memory_chunks UNION ALL SELECT fts FROM memory_stories"
        )
    bare = await schemas.pool(public=False)
    async with bare.acquire() as conn:
        with pytest.raises(Exception, match="function memory_codes"):
            await conn.execute(f"UPDATE {schema}.memory_chunks SET config = config")
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    assert (await _apply(pool, migrations.plan_upgrade)).steps[0] == "0013 function bodies"
    async with bare.acquire() as conn:
        await conn.execute(f"UPDATE {schema}.memory_chunks SET config = config")
        await conn.execute(f"UPDATE {schema}.memory_stories SET config = config")
        required = await conn.fetchval(
            f"SELECT {schema}.memory_required(plainto_tsquery('english', 'part'), $1)::text", text
        )
        after = await conn.fetch(
            f"SELECT fts FROM {schema}.memory_chunks UNION ALL SELECT fts FROM {schema}.memory_stories"
        )
    assert after == before
    assert required == "'part' & 'a-1001'"


ALIASED = {
    "e1": ("feed valves", ["V-101", "Main  Valve"]),
    "e2": ("road", ["ΟΔΟΣ", "Pump\xa0P-3", "Line\x1cA", "İzmir Plant", "  ", "pump p-3"]),
    "e3": ("it", []),
    "e4": ("pump", ["P-3", "p-3"]),
}


async def _at_function_bodies(schemas: Schemas, monkeypatch: pytest.MonkeyPatch) -> Any:
    _, pool = await _fresh(schemas)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:13])
    monkeypatch.setattr(migrations, "LATEST", 13)
    await _apply(pool, partial(migrations.plan_create, dimension=8, language="english"))
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) "
            "VALUES ('a', 'english', 'pg_catalog.english')"
        )
        await conn.executemany(
            "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases, config) "
            "VALUES ('a', $1, $2, $2, 'thing', $3::jsonb, 'pg_catalog.english')",
            [
                (entity_id, key, json.dumps(aliases))
                for entity_id, (key, aliases) in ALIASED.items()
            ],
        )
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    return pool


async def test_upgrading_to_graph_writes_keys_every_alias_as_memory_keys_a_name(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = await _at_function_bodies(schemas, monkeypatch)
    spotted = "SELECT entity_id, spotted::text AS spotted FROM memory_entities ORDER BY entity_id"
    async with pool.acquire() as conn:
        before = [tuple(r) for r in await conn.fetch(spotted)]
    shown = await _apply(pool, migrations.plan_upgrade, run=False)
    assert shown.steps == ["0014 graph writes"]
    [fill] = [s for s in shown.statements if isinstance(s, Call)]
    assert fill.name == "fill memory_entities.alias_keys from aliases, computed by rfnry-memory"
    assert not any("V-101" in s or "ΟΔΟΣ" in s for s in shown.statements if isinstance(s, str))
    assert await _apply(pool, migrations.plan_upgrade, expected=shown) == shown
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT entity_id, alias_keys FROM memory_entities ORDER BY entity_id"
        )
        after = [tuple(r) for r in await conn.fetch(spotted)]
        default = await conn.fetchval(
            "SELECT column_default FROM information_schema.columns WHERE table_schema = "
            "current_schema() AND table_name = 'memory_entities' AND column_name = 'alias_keys'"
        )
    assert {r["entity_id"]: r["alias_keys"] for r in rows} == {
        entity_id: alias_keys(aliases) for entity_id, (_, aliases) in ALIASED.items()
    }
    assert [r for r in after if r[0] in ("e1", "e3")] == [r for r in before if r[0] in ("e1", "e3")]
    assert [r for r in before if r[0] == "e4"] == [("e4", "'p' <-> '-3' | 'p' <-> '-3' | 'pump'")]
    assert [r for r in after if r[0] == "e4"] == [("e4", "'p' <-> '-3' | 'pump'")]
    assert default is None
    async with pool.acquire() as conn:
        named = await conn.fetch("SELECT entity_id, name FROM memory_entity_names")
    assert {(r["entity_id"], r["name"]) for r in named} == {
        (entity_id, n) for entity_id, (key, aliases) in ALIASED.items() for n in [key, *aliases]
    }
    assert await _indexed(pool, "ix_memory_entity_names")
    assert await _apply(pool, migrations.plan_upgrade) == Plan([], [])
    storage = PostgresStorage(pool)
    await storage.check(dimension=8, graph=True)
    for alias in ALIASED["e2"][1][:4]:
        assert [
            e.entity_id for e in (await storage.entities_named("a", [alias]))[name_key(alias)]
        ] == ["e2"], alias
    named = await storage.entities_named_in("a", "is ΟΔΟΣ open?", spot="words")
    assert [e.entity_id for e in named] == ["e2"]
    async with pool.acquire() as conn:
        with pytest.raises(Exception, match="alias_keys"):
            await conn.execute(
                "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases, config) "
                "VALUES ('a', 'old-app', 'x', 'x', 'thing', '[\"y\"]', 'pg_catalog.english')"
            )


async def _indexed(pool: Any, name: str) -> bool:
    async with pool.acquire() as conn:
        return bool(
            await conn.fetchval(
                "SELECT to_regclass(format('%I.%I', current_schema(), $1::text)) IS NOT NULL", name
            )
        )


async def test_upgrading_without_pg_trgm_builds_no_index_of_similar_names_until_set_graph(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = await _at_function_bodies(schemas, monkeypatch)
    settled = migrations._settings

    async def without_trigrams(conn: Any) -> migrations.Settings:
        return (await settled(conn)).model_copy(update={"graph": False})

    monkeypatch.setattr(migrations, "_settings", without_trigrams)
    shown = await _apply(pool, migrations.plan_upgrade, run=False)
    assert shown.steps == ["0014 graph writes"]
    assert migrations._NAMES_INDEX not in shown.statements
    assert await _apply(pool, migrations.plan_upgrade, expected=shown) == shown
    assert not await _indexed(pool, "ix_memory_entity_names")
    assert await _apply(pool, migrations.plan_upgrade) == Plan([], [])
    storage = PostgresStorage(pool)
    await storage.check(dimension=8, graph=False)
    monkeypatch.setattr(migrations, "_settings", settled)
    planned = await _apply(pool, partial(migrations.plan_set, graph=True))
    assert planned.steps == ["the index of similar names"]
    await storage.check(dimension=8, graph=True)
    [named] = (await storage.entities_named("a", ["ΟΔΟΣ"]))["οδος"]
    assert [e.entity_id for e in await storage.similar_entities("a", "ΟΔΟΣ", limit=5)] == [
        named.entity_id
    ]


async def test_set_graph_builds_the_index_of_similar_names_where_pg_trgm_already_is(
    schemas: Schemas,
) -> None:
    schema, pool = await _fresh(schemas)
    await provision(pool, graph=False)
    assert not await _indexed(pool, "ix_memory_entity_names")
    storage = PostgresStorage(pool)
    with pytest.raises(
        ConfigurationError,
        match=rf"^postgres: the graph pillar needs the index of similar names — run rfnry-memory set --graph --schema {schema}$",
    ):
        await storage.check(dimension=8, graph=True)
    await storage.check(dimension=8, graph=False)
    planned = await _apply(pool, partial(migrations.plan_set, graph=True))
    assert planned.steps == ["the index of similar names"]
    assert await _indexed(pool, "ix_memory_entity_names")
    await storage.check(dimension=8, graph=True)
    assert await _apply(pool, partial(migrations.plan_set, graph=True)) == Plan([], [])


async def test_what_was_written_before_graph_writes_reads_as_unknown(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = await _at_function_bodies(schemas, monkeypatch)
    every = migrations.MIGRATIONS
    monkeypatch.setattr(migrations, "MIGRATIONS", every[:13])
    monkeypatch.setattr(migrations, "LATEST", 13)
    jan = datetime(2026, 1, 1, tzinfo=UTC)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_sources (corpus_id, source_id, name, text, created_at, chunks, "
            "confirmed_at) VALUES ('a', 's', 'manual', 'feed valves', $1, 1, $1)",
            jan,
        )
        await conn.execute(
            "INSERT INTO memory_chunks (corpus_id, chunk_id, source_id, position, source_name, text, "
            "digest, config) VALUES ('a', 's-0', 's', 0, 'manual', 'feed valves', 'd', "
            "'pg_catalog.english')"
        )
        await conn.execute(
            "INSERT INTO memory_stories (corpus_id, story_id, text, text_hash, valid_from, created_at, "
            "config) VALUES ('a', 't', 'feed valves', 'h', $1, $1, 'pg_catalog.english')",
            jan,
        )
        await conn.execute(
            "INSERT INTO memory_relations (corpus_id, from_id, to_id, type, evidence_kind, evidence_id) "
            "VALUES ('a', 'e1', 'e3', 'FEEDS', 'chunk', 's-0'), ('a', 'e1', 'e3', 'FEEDS', 'story', 't')"
        )
        await conn.execute(
            "INSERT INTO memory_mentions (corpus_id, entity_id, passage_kind, passage_id, name) "
            "VALUES ('a', 'e1', 'chunk', 's-0', 'feed valves'), ('a', 'e3', 'story', 't', 'it')"
        )
    monkeypatch.setattr(migrations, "MIGRATIONS", every)
    monkeypatch.setattr(migrations, "LATEST", NOW)
    await _apply(pool, migrations.plan_upgrade)
    storage = PostgresStorage(pool)
    await storage.check(dimension=8, graph=True)
    [chunk] = await storage.source_chunks("a", "s")
    [story] = await storage.get_stories("a", ["t"])
    assert (chunk.extracted_by, story.extracted_by) == (None, None)
    related = await storage.neighbors("a", ["e1"], relation_types=None, window=Window())
    assert [(r.evidence_kind, r.extracted) for r in related] == [("chunk", None), ("story", None)]
    for entity in await storage.entities_by_id("a", list(ALIASED)):
        assert (entity.aliased_by, entity.resolved_by) == ({}, None), entity.entity_id
