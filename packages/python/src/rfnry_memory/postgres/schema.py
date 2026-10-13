from __future__ import annotations

from typing import Any

from rfnry_memory.errors import ConfigurationError
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.language import _folded

_ALIAS_KEY = r"btrim(regexp_replace(lower(a), '\s+', ' ', 'g'))"


_EXTENSIONS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "CREATE EXTENSION IF NOT EXISTS unaccent",
]


def _indexed(text: str) -> str:
    return f"memory_tsvector(config, {text})"


_CHUNK_SEARCHED = "coalesce(source_name, '') || ' ' || coalesce(title, '') || ' ' || coalesce(context, '') || ' ' || text"


_TSVECTOR = """DO $$ BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended('memory.upgrade', 0));
    IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'memory_tsvector' AND pronamespace = current_schema()::regnamespace) THEN
        CREATE FUNCTION memory_tsvector(config text, body text) RETURNS tsvector LANGUAGE sql IMMUTABLE PARALLEL SAFE
            AS $body$ SELECT to_tsvector(config::regconfig, body) $body$;
    END IF;
END $$"""


_ALIAS_INDEX = f"""DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'memory_alias_keys' AND pronamespace = current_schema()::regnamespace) THEN
        CREATE FUNCTION memory_alias_keys(aliases jsonb) RETURNS text[] LANGUAGE sql IMMUTABLE PARALLEL SAFE
            AS $keys$ SELECT array_agg({_ALIAS_KEY}) FROM jsonb_array_elements_text(aliases) AS a $keys$;
    END IF;
    IF to_regclass(format('%I.ix_memory_entities_alias_keys', current_schema())) IS NULL THEN
        CREATE INDEX ix_memory_entities_alias_keys ON memory_entities USING GIN (memory_alias_keys(aliases))
            WITH (fastupdate = off) WHERE aliases <> '[]'::jsonb;
    END IF;
END $$"""


def _ddl(dimension: int | None) -> list[str]:
    vector = (
        f"embedding vector({dimension}) NULL, embedding_model TEXT NULL,"
        if dimension
        else "embedding_model TEXT NULL,"
    )
    return [
        _TSVECTOR,
        """CREATE TABLE IF NOT EXISTS memory_corpora (
            corpus_id TEXT NOT NULL PRIMARY KEY, language TEXT NOT NULL,
            history BOOLEAN NOT NULL DEFAULT false, config TEXT NULL)""",
        """CREATE TABLE IF NOT EXISTS memory_sources (
            corpus_id TEXT NOT NULL, source_id TEXT NOT NULL, name TEXT NOT NULL, metadata JSON NOT NULL DEFAULT '{}',
            text TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL, chunks INTEGER NOT NULL, digest TEXT NULL,
            retired_at TIMESTAMPTZ NULL, replaced_by TEXT NULL,
            PRIMARY KEY (corpus_id, source_id))""",
        "CREATE INDEX IF NOT EXISTS ix_memory_sources_metadata ON memory_sources USING GIN ((metadata::jsonb))",
        f"""CREATE TABLE IF NOT EXISTS memory_chunks (
            corpus_id TEXT NOT NULL, chunk_id TEXT NOT NULL, source_id TEXT NOT NULL, position INTEGER NOT NULL,
            source_name TEXT NOT NULL, title TEXT NULL, locator JSON NOT NULL DEFAULT '{{}}', text TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}',
            digest TEXT NOT NULL, {vector} config TEXT NOT NULL,
            retired_at TIMESTAMPTZ NULL, context TEXT NULL,
            fts tsvector GENERATED ALWAYS AS ({_indexed(_CHUNK_SEARCHED)}) STORED,
            PRIMARY KEY (corpus_id, chunk_id),
            FOREIGN KEY (corpus_id, source_id) REFERENCES memory_sources (corpus_id, source_id) ON DELETE CASCADE)""",
        "CREATE INDEX IF NOT EXISTS ix_memory_chunks_source ON memory_chunks (corpus_id, source_id, position)",
        "CREATE INDEX IF NOT EXISTS ix_memory_chunks_fts ON memory_chunks USING GIN (fts)",
        "CREATE INDEX IF NOT EXISTS ix_memory_chunks_metadata ON memory_chunks USING GIN (metadata)",
        "CREATE INDEX IF NOT EXISTS ix_memory_chunks_digest ON memory_chunks (corpus_id, digest)",
        f"""CREATE TABLE IF NOT EXISTS memory_stories (
            corpus_id TEXT NOT NULL, story_id TEXT NOT NULL, text TEXT NOT NULL, text_hash TEXT NOT NULL,
            subject TEXT NULL, valid_from TIMESTAMPTZ NOT NULL, valid_to TIMESTAMPTZ NULL, created_at TIMESTAMPTZ NOT NULL,
            supersedes JSONB NOT NULL DEFAULT '[]', superseded_by TEXT NULL, closed_at TIMESTAMPTZ NULL,
            written_to TIMESTAMPTZ NULL, source_ids JSONB NOT NULL DEFAULT '[]', metadata JSON NOT NULL DEFAULT '{{}}',
            {vector} config TEXT NOT NULL,
            fts tsvector GENERATED ALWAYS AS ({_indexed("text")}) STORED,
            PRIMARY KEY (corpus_id, story_id))""",
        "CREATE INDEX IF NOT EXISTS ix_memory_stories_text ON memory_stories (corpus_id, text_hash)",
        "CREATE INDEX IF NOT EXISTS ix_memory_stories_subject ON memory_stories (corpus_id, subject)",
        "CREATE INDEX IF NOT EXISTS ix_memory_stories_valid_from ON memory_stories (corpus_id, valid_from)",
        "CREATE INDEX IF NOT EXISTS ix_memory_stories_fts ON memory_stories USING GIN (fts)",
        "CREATE INDEX IF NOT EXISTS ix_memory_stories_metadata ON memory_stories USING GIN ((metadata::jsonb))",
        """CREATE TABLE IF NOT EXISTS memory_entities (
            corpus_id TEXT NOT NULL, entity_id TEXT NOT NULL, key TEXT NOT NULL, name TEXT NOT NULL, type TEXT NOT NULL,
            aliases JSONB NOT NULL DEFAULT '[]',
            PRIMARY KEY (corpus_id, entity_id), UNIQUE (corpus_id, key, type))""",
        """CREATE TABLE IF NOT EXISTS memory_relations (
            corpus_id TEXT NOT NULL, from_id TEXT NOT NULL, to_id TEXT NOT NULL, type TEXT NOT NULL,
            evidence_kind TEXT NOT NULL, evidence_id TEXT NOT NULL,
            PRIMARY KEY (corpus_id, from_id, to_id, type, evidence_kind, evidence_id))""",
        "CREATE INDEX IF NOT EXISTS ix_memory_relations_from ON memory_relations (corpus_id, from_id)",
        "CREATE INDEX IF NOT EXISTS ix_memory_relations_to ON memory_relations (corpus_id, to_id)",
        """CREATE TABLE IF NOT EXISTS memory_mentions (
            corpus_id TEXT NOT NULL, entity_id TEXT NOT NULL, passage_kind TEXT NOT NULL, passage_id TEXT NOT NULL,
            name TEXT NOT NULL, PRIMARY KEY (corpus_id, entity_id, passage_kind, passage_id, name))""",
        "CREATE INDEX IF NOT EXISTS ix_memory_mentions_passage ON memory_mentions (corpus_id, passage_kind, passage_id)",
    ]


_ADDED_COLUMNS = [
    ("memory_corpora", "history", "BOOLEAN NOT NULL DEFAULT false"),
    ("memory_sources", "retired_at", "TIMESTAMPTZ NULL"),
    ("memory_sources", "replaced_by", "TEXT NULL"),
    ("memory_chunks", "retired_at", "TIMESTAMPTZ NULL"),
    ("memory_chunks", "context", "TEXT NULL"),
    ("memory_corpora", "config", "TEXT NULL"),
]


_SUPERSEDES_TO_LIST = [
    "ALTER TABLE memory_stories ALTER COLUMN supersedes TYPE JSONB USING "
    "CASE WHEN supersedes IS NULL THEN '[]'::jsonb ELSE jsonb_build_array(supersedes) END",
    "ALTER TABLE memory_stories ALTER COLUMN supersedes SET DEFAULT '[]'::jsonb",
    "ALTER TABLE memory_stories ALTER COLUMN supersedes SET NOT NULL",
    "UPDATE memory_stories AS old SET superseded_by = new.story_id, closed_at = new.created_at "
    "FROM memory_stories AS new WHERE new.corpus_id = old.corpus_id AND new.supersedes->>0 = old.story_id "
    "AND old.superseded_by IS NULL",
    "UPDATE memory_stories SET written_to = valid_to WHERE superseded_by IS NULL AND written_to IS NULL",
]


class Schema(Base):
    async def initialize(self, *, dimension: int | None) -> None:
        async with self._pool.acquire() as conn:
            for statement in _EXTENSIONS:
                try:
                    await conn.execute(statement)
                except Exception as exc:
                    raise ConfigurationError(
                        f"postgres: {statement} failed: {exc}. create the extension with a privileged role, then initialize again."
                    ) from exc
            async with self._guard("initialize"):
                await self._check_dimension(conn, dimension)
                for statement in _ddl(dimension):
                    await conn.execute(statement)
                await self._upgrade(conn)
                self._dimension = await conn.fetchval(
                    "SELECT atttypmod FROM pg_attribute WHERE attrelid = to_regclass(format('%I.%I', current_schema(), 'memory_chunks')) "
                    "AND attname = 'embedding' AND NOT attisdropped"
                )

    async def _upgrade(self, conn: Any) -> None:
        async with conn.transaction(isolation="read_committed"):
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "memory.upgrade"
            )
            present = await conn.fetch(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = ANY($1::text[])",
                ["memory_corpora", "memory_sources", "memory_chunks", "memory_stories"],
            )
            have = {(r["table_name"], r["column_name"]) for r in present}
            columns = {
                r["column_name"]: r["data_type"]
                for r in present
                if r["table_name"] == "memory_stories"
            }
            for table, column, kind in _ADDED_COLUMNS:
                if (table, column) not in have:
                    await conn.execute(
                        f"DO $$ BEGIN EXECUTE format('ALTER TABLE %I.{table} ADD COLUMN IF NOT EXISTS {column} {kind}', "
                        "current_schema()); END $$"
                    )
            await _folded(conn, "english")
            if ("memory_corpora", "config") not in have:
                await conn.execute(
                    "UPDATE memory_corpora SET config = format('%I.%I', current_schema(), 'memory_' || language)"
                )
            schema = await conn.fetchval("SELECT quote_ident(current_schema())")
            for table, searched in (("memory_chunks", _CHUNK_SEARCHED), ("memory_stories", "text")):
                if (table, "config") in have:
                    continue
                await conn.execute(f"ALTER TABLE {schema}.{table} RENAME COLUMN language TO config")
                await conn.execute(
                    f"ALTER TABLE {schema}.{table} DROP COLUMN fts, ALTER COLUMN config DROP DEFAULT, "
                    "ALTER COLUMN config TYPE TEXT USING format('%I.%I', current_schema(), 'memory_' || config), "
                    f"ADD COLUMN fts tsvector GENERATED ALWAYS AS ({_indexed(searched)}) STORED"
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS ix_{table}_fts ON {schema}.{table} USING GIN (fts)"
                )
            for column, kind in (
                ("superseded_by", "TEXT"),
                ("closed_at", "TIMESTAMPTZ"),
                ("written_to", "TIMESTAMPTZ"),
            ):
                if column not in columns:
                    await conn.execute(
                        f"ALTER TABLE memory_stories ADD COLUMN {column} {kind} NULL"
                    )
            if columns.get("supersedes") == "text":
                for statement in _SUPERSEDES_TO_LIST:
                    await conn.execute(statement)
            await conn.execute(
                "DO $$ BEGIN EXECUTE format('DROP INDEX IF EXISTS %I.ux_memory_stories_open_text', "
                "current_schema()); END $$"
            )
            await conn.execute(_ALIAS_INDEX)

    async def _check_dimension(self, conn: Any, dimension: int | None) -> None:
        if not dimension:
            return
        rows = await conn.fetch(
            """SELECT t.name, a.atttypmod AS dimension FROM unnest($1::text[]) AS t(name)
               CROSS JOIN LATERAL (SELECT to_regclass(format('%I.%I', current_schema(), t.name)) AS relation) AS r
               LEFT JOIN pg_attribute a ON a.attrelid = r.relation AND a.attname = 'embedding' AND NOT a.attisdropped
               WHERE r.relation IS NOT NULL""",
            ["memory_chunks", "memory_stories"],
        )
        for row in rows:
            if row["dimension"] != dimension:
                held = (
                    f"for dimension {row['dimension']}"
                    if row["dimension"] is not None
                    else "without a vector column"
                )
                raise ConfigurationError(
                    f"postgres: tables were created {held}; this memory embeds {dimension} — re-provision or use another schema"
                )
