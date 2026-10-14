from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NamedTuple

from pydantic import BaseModel

from rfnry_memory.errors import ConfigurationError, InputError, StorageError
from rfnry_memory.postgres.language import _configuration


class Settings(BaseModel):
    dimension: int | None
    language: str
    config: str
    graph: bool = True


class Migration(NamedTuple):
    version: int
    name: str
    plan: Callable[[Settings], list[str]]


class Plan(NamedTuple):
    steps: list[str]
    statements: list[str]


Planner = Callable[[Any], Awaitable[Plan]]


_ALIAS_KEY = r"btrim(regexp_replace(lower(a), '\s+', ' ', 'g'))"


_CHUNK_SEARCHED = "coalesce(source_name, '') || ' ' || coalesce(title, '') || ' ' || coalesce(context, '') || ' ' || text"


def _create(settings: Settings) -> list[str]:
    vector = (
        f"embedding vector({settings.dimension}) NULL, embedding_model TEXT NULL,"
        if settings.dimension
        else "embedding_model TEXT NULL,"
    )
    return [
        "CREATE FUNCTION memory_tsvector(config text, body text) RETURNS tsvector LANGUAGE sql "
        "IMMUTABLE PARALLEL SAFE AS $$ SELECT to_tsvector(config::regconfig, body) $$",
        "CREATE FUNCTION memory_alias_keys(aliases jsonb) RETURNS text[] LANGUAGE sql IMMUTABLE PARALLEL SAFE "
        f"AS $$ SELECT array_agg({_ALIAS_KEY}) FROM jsonb_array_elements_text(aliases) AS a $$",
        """CREATE TABLE memory_migrations (
            version INTEGER NOT NULL PRIMARY KEY, name TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
        """CREATE TABLE memory_settings (
            id BOOLEAN NOT NULL PRIMARY KEY DEFAULT true CHECK (id), dimension INTEGER NULL,
            language TEXT NOT NULL, config TEXT NOT NULL)""",
        """CREATE TABLE memory_corpora (
            corpus_id TEXT NOT NULL PRIMARY KEY, language TEXT NOT NULL, config TEXT NOT NULL,
            history BOOLEAN NOT NULL DEFAULT false)""",
        """CREATE TABLE memory_sources (
            corpus_id TEXT NOT NULL, source_id TEXT NOT NULL, name TEXT NOT NULL, metadata JSON NOT NULL DEFAULT '{}',
            text TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL, chunks INTEGER NOT NULL, digest TEXT NULL,
            retired_at TIMESTAMPTZ NULL, replaced_by TEXT NULL,
            PRIMARY KEY (corpus_id, source_id))""",
        "CREATE INDEX ix_memory_sources_metadata ON memory_sources USING GIN ((metadata::jsonb))",
        f"""CREATE TABLE memory_chunks (
            corpus_id TEXT NOT NULL, chunk_id TEXT NOT NULL, source_id TEXT NOT NULL, position INTEGER NOT NULL,
            source_name TEXT NOT NULL, title TEXT NULL, locator JSON NOT NULL DEFAULT '{{}}', text TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}',
            digest TEXT NOT NULL, {vector} config TEXT NOT NULL,
            retired_at TIMESTAMPTZ NULL, context TEXT NULL,
            fts tsvector GENERATED ALWAYS AS (memory_tsvector(config, {_CHUNK_SEARCHED})) STORED,
            PRIMARY KEY (corpus_id, chunk_id),
            FOREIGN KEY (corpus_id, source_id) REFERENCES memory_sources (corpus_id, source_id) ON DELETE CASCADE)""",
        "CREATE INDEX ix_memory_chunks_source ON memory_chunks (corpus_id, source_id, position)",
        "CREATE INDEX ix_memory_chunks_fts ON memory_chunks USING GIN (fts)",
        "CREATE INDEX ix_memory_chunks_metadata ON memory_chunks USING GIN (metadata)",
        "CREATE INDEX ix_memory_chunks_digest ON memory_chunks (corpus_id, digest)",
        f"""CREATE TABLE memory_stories (
            corpus_id TEXT NOT NULL, story_id TEXT NOT NULL, text TEXT NOT NULL, text_hash TEXT NOT NULL,
            subject TEXT NULL, valid_from TIMESTAMPTZ NOT NULL, valid_to TIMESTAMPTZ NULL, created_at TIMESTAMPTZ NOT NULL,
            supersedes JSONB NOT NULL DEFAULT '[]', superseded_by TEXT NULL, closed_at TIMESTAMPTZ NULL,
            written_to TIMESTAMPTZ NULL, source_ids JSONB NOT NULL DEFAULT '[]', metadata JSON NOT NULL DEFAULT '{{}}',
            {vector} config TEXT NOT NULL,
            fts tsvector GENERATED ALWAYS AS (memory_tsvector(config, text)) STORED,
            PRIMARY KEY (corpus_id, story_id))""",
        "CREATE INDEX ix_memory_stories_text ON memory_stories (corpus_id, text_hash)",
        "CREATE INDEX ix_memory_stories_subject ON memory_stories (corpus_id, subject)",
        "CREATE INDEX ix_memory_stories_valid_from ON memory_stories (corpus_id, valid_from)",
        "CREATE INDEX ix_memory_stories_fts ON memory_stories USING GIN (fts)",
        "CREATE INDEX ix_memory_stories_metadata ON memory_stories USING GIN ((metadata::jsonb))",
        """CREATE TABLE memory_entities (
            corpus_id TEXT NOT NULL, entity_id TEXT NOT NULL, key TEXT NOT NULL, name TEXT NOT NULL, type TEXT NOT NULL,
            aliases JSONB NOT NULL DEFAULT '[]',
            PRIMARY KEY (corpus_id, entity_id), UNIQUE (corpus_id, key, type))""",
        "CREATE INDEX ix_memory_entities_alias_keys ON memory_entities USING GIN (memory_alias_keys(aliases)) "
        "WITH (fastupdate = off) WHERE aliases <> '[]'::jsonb",
        """CREATE TABLE memory_relations (
            corpus_id TEXT NOT NULL, from_id TEXT NOT NULL, to_id TEXT NOT NULL, type TEXT NOT NULL,
            evidence_kind TEXT NOT NULL, evidence_id TEXT NOT NULL,
            PRIMARY KEY (corpus_id, from_id, to_id, type, evidence_kind, evidence_id))""",
        "CREATE INDEX ix_memory_relations_from ON memory_relations (corpus_id, from_id)",
        "CREATE INDEX ix_memory_relations_to ON memory_relations (corpus_id, to_id)",
        """CREATE TABLE memory_mentions (
            corpus_id TEXT NOT NULL, entity_id TEXT NOT NULL, passage_kind TEXT NOT NULL, passage_id TEXT NOT NULL,
            name TEXT NOT NULL, PRIMARY KEY (corpus_id, entity_id, passage_kind, passage_id, name))""",
        "CREATE INDEX ix_memory_mentions_passage ON memory_mentions (corpus_id, passage_kind, passage_id)",
    ]


MIGRATIONS = [Migration(1, "create", _create)]


LATEST = MIGRATIONS[-1].version


_STATE = (
    "SELECT current_schema() AS schema, "
    "to_regclass(format('%I.memory_migrations', current_schema())) IS NOT NULL AS migrated, "
    "EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = current_schema() "
    "AND starts_with(tablename, 'memory_')) AS tables, "
    "EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') AS trigram"
)


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _extensions(settings: Settings) -> list[str]:
    return [*(["vector"] if settings.dimension else []), *(["pg_trgm"] if settings.graph else [])]


def _install(name: str) -> str:
    return f"CREATE EXTENSION IF NOT EXISTS {name} SCHEMA public"


def _recorded(migration: Migration) -> str:
    return f"INSERT INTO memory_migrations (version, name) VALUES ({migration.version}, {_literal(migration.name)})"


def _steps(migrations: Sequence[Migration], settings: Settings) -> Plan:
    return Plan(
        [f"{m.version:04d} {m.name}" for m in migrations],
        [s for m in migrations for s in [*m.plan(settings), _recorded(m)]],
    )


def _shared(schema: str) -> bool:
    return schema in ("public", "information_schema") or schema.startswith("pg_")


def _own(schema: str | None) -> str:
    if schema is None or _shared(schema):
        raise InputError(
            f"postgres: memory is not created in {schema!r} — name a schema of its own"
        )
    return schema


async def _language(conn: Any, name: str | None) -> tuple[str, str]:
    if name is None:
        found = await conn.fetchrow(
            "SELECT c.oid::regconfig::text AS language, format('%I.%I', n.nspname, c.cfgname) AS config "
            "FROM pg_ts_config c JOIN pg_namespace n ON n.oid = c.cfgnamespace "
            "WHERE c.oid = current_setting('default_text_search_config')::regconfig"
        )
        return str(found["language"]), str(found["config"])
    resolved = await _configuration(conn, name)
    if resolved is None:
        raise InputError(f"postgres: {name!r} is not a text search configuration in this database")
    return resolved[0], resolved[1]


async def _settings(conn: Any) -> Settings:
    row = await conn.fetchrow("SELECT dimension, language, config FROM memory_settings")
    trigram = await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')"
    )
    return Settings(
        dimension=row["dimension"], language=row["language"], config=row["config"], graph=trigram
    )


async def _applied(conn: Any) -> list[int]:
    state = await conn.fetchrow(_STATE)
    if not state["migrated"]:
        if state["tables"]:
            raise ConfigurationError(
                f"postgres: the memory tables in {state['schema']} predate migrations — create a new schema with rfnry-memory create"
            )
        raise ConfigurationError(
            f"postgres: no memory schema in {state['schema']} — run rfnry-memory create"
        )
    rows = await conn.fetch("SELECT version FROM memory_migrations ORDER BY version")
    return [int(r["version"]) for r in rows]


async def plan_create(
    conn: Any, *, dimension: int | None, graph: bool = True, language: str | None = None
) -> Plan:
    if dimension is not None and (isinstance(dimension, bool) or dimension < 1):
        raise InputError("postgres: a dimension is a positive int")
    state = await conn.fetchrow(_STATE)
    schema = _own(state["schema"])
    if state["tables"]:
        raise ConfigurationError(
            f"postgres: {schema} already holds memory tables — run rfnry-memory upgrade, or create into another schema"
        )
    name, config = await _language(conn, language)
    settings = Settings(dimension=dimension, language=name, config=config, graph=graph)
    planned = _steps(MIGRATIONS, settings)
    record = (
        "INSERT INTO memory_settings (dimension, language, config) "
        f"VALUES ({dimension if dimension else 'NULL'}, {_literal(name)}, {_literal(config)})"
    )
    extensions = _extensions(settings)
    vectors = f"dimension {dimension}" if dimension else "no vectors"
    return Plan(
        [
            *(f"extension {name} in public" for name in extensions),
            *planned.steps,
            f"settings ({vectors}, graph {'on' if graph else 'off'}, language {name})",
        ],
        [*(_install(name) for name in extensions), *planned.statements, record],
    )


async def plan_upgrade(conn: Any) -> Plan:
    applied = await _applied(conn)
    newest = max(applied, default=0)
    schema = await conn.fetchval("SELECT current_schema()")
    if newest > LATEST:
        raise ConfigurationError(
            f"postgres: {schema} is at version {newest}, this rfnry-memory knows {LATEST} — upgrade rfnry-memory"
        )
    pending = [m for m in MIGRATIONS if m.version not in applied]
    return _steps(pending, await _settings(conn)) if pending else Plan([], [])


async def plan_set(
    conn: Any,
    *,
    dimension: int | None = None,
    graph: bool = False,
    language: str | None = None,
) -> Plan:
    if sum((dimension is not None, graph, language is not None)) != 1:
        raise InputError("postgres: set changes one setting: a dimension, the graph or a language")
    applied = await _applied(conn)
    if max(applied, default=0) != LATEST:
        raise ConfigurationError(
            "postgres: the schema is not at this rfnry-memory's version — run rfnry-memory upgrade first"
        )
    settings = await _settings(conn)
    if graph:
        return (
            Plan([], [])
            if settings.graph
            else Plan(["extension pg_trgm in public"], [_install("pg_trgm")])
        )
    if language is not None:
        name, config = await _language(conn, language)
        if (name, config) == (settings.language, settings.config):
            return Plan([], [])
        return Plan(
            [f"default language {name}"],
            [
                f"UPDATE memory_settings SET language = {_literal(name)}, config = {_literal(config)}"
            ],
        )
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise InputError("postgres: a dimension is a positive int")
    if dimension == settings.dimension:
        return Plan([], [])
    schema = await conn.fetchval("SELECT quote_ident(current_schema())")
    indexes = [
        r["name"]
        for r in await conn.fetch(
            "SELECT quote_ident(tablename) AS name FROM pg_tables WHERE schemaname = current_schema() "
            "AND starts_with(tablename, 'memory_vectors_') ORDER BY tablename"
        )
    ]
    if settings.dimension is None:
        columns = [
            f"ALTER TABLE {schema}.{table} ADD COLUMN embedding vector({dimension}) NULL"
            for table in ("memory_chunks", "memory_stories")
        ]
    else:
        columns = [
            f"ALTER TABLE {schema}.{table} ALTER COLUMN embedding TYPE vector({dimension}) USING NULL"
            for table in ("memory_chunks", "memory_stories")
        ]
    return Plan(
        [
            *(["extension vector in public"] if settings.dimension is None else []),
            f"vectors of {dimension} dimensions: every stored vector and vector index is deleted",
        ],
        [
            *([_install("vector")] if settings.dimension is None else []),
            *(f"DROP TABLE {schema}.{name}" for name in indexes),
            *columns,
            *(
                f"UPDATE {schema}.{table} SET embedding_model = NULL WHERE embedding_model IS NOT NULL"
                for table in ("memory_chunks", "memory_stories")
            ),
            f"UPDATE memory_settings SET dimension = {dimension}",
        ],
    )


async def apply(
    conn: Any,
    planner: Planner,
    *,
    run: bool = True,
    expected: Plan | None = None,
    before: Sequence[str] = (),
) -> Plan:
    transaction = conn.transaction(isolation="read_committed")
    await transaction.start()
    try:
        for statement in before:
            await conn.execute(statement)
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended('memory.migrate:' || current_schema(), 0))"
        )
        planned = await planner(conn)
        plan = Plan(planned.steps, [*before, *planned.statements])
        if expected is not None and plan != expected:
            raise ConfigurationError(
                "postgres: the schema changed since the plan was shown — run the command again"
            )
        if run:
            for statement in planned.statements:
                await _execute(conn, statement)
    except BaseException:
        await transaction.rollback()
        raise
    if run:
        await transaction.commit()
    else:
        await transaction.rollback()
    return plan


async def _execute(conn: Any, statement: str) -> None:
    try:
        await conn.execute(statement)
    except Exception as exc:
        if statement.startswith("CREATE EXTENSION"):
            raise ConfigurationError(
                f"postgres: {statement} failed: {exc} — run it as a privileged role, then run this command again"
            ) from exc
        raise StorageError(f"postgres: {statement.split(chr(10))[0]} failed: {exc}") from exc
