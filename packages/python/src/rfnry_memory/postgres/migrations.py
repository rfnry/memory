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


_NAMES = (
    f"SELECT key AS n UNION ALL SELECT {_ALIAS_KEY} FROM jsonb_array_elements_text(aliases) AS a"
)


def _spotting(settings: Settings) -> list[str]:
    return [
        "CREATE FUNCTION memory_spotted(config text, key text, aliases jsonb) RETURNS tsquery LANGUAGE sql "
        "IMMUTABLE PARALLEL SAFE AS $$ SELECT string_agg(format('(%s)', phraseto_tsquery(config::regconfig, n)), "
        f"""' | ' ORDER BY n COLLATE "C")::tsquery FROM ({_NAMES}) AS names """
        "WHERE length(to_tsvector(config::regconfig, n)) > 0 $$",
        "ALTER TABLE memory_entities ADD COLUMN config TEXT NULL",
        "UPDATE memory_entities e SET config = coalesce((SELECT c.config FROM memory_corpora c "
        "WHERE c.corpus_id = e.corpus_id), (SELECT s.config FROM memory_settings s))",
        "ALTER TABLE memory_entities ALTER COLUMN config SET NOT NULL",
        "ALTER TABLE memory_entities ADD COLUMN spotted tsquery "
        "GENERATED ALWAYS AS (memory_spotted(config, key, aliases)) STORED",
    ]


def _validity(settings: Settings) -> list[str]:
    return [
        "ALTER TABLE memory_sources ADD COLUMN valid_from TIMESTAMPTZ NULL, "
        "ADD COLUMN valid_to TIMESTAMPTZ NULL, ADD COLUMN written_to TIMESTAMPTZ NULL, "
        "ADD COLUMN confirmed_at TIMESTAMPTZ NULL",
        "ALTER TABLE memory_chunks ADD COLUMN valid_from TIMESTAMPTZ NULL, "
        "ADD COLUMN valid_to TIMESTAMPTZ NULL, ADD COLUMN written_to TIMESTAMPTZ NULL",
        "UPDATE memory_sources SET confirmed_at = created_at",
        "ALTER TABLE memory_sources ALTER COLUMN confirmed_at SET NOT NULL",
        "UPDATE memory_sources SET valid_to = retired_at WHERE retired_at IS NOT NULL",
        "UPDATE memory_sources s SET valid_from = p.retired_at FROM ("
        "SELECT corpus_id, replaced_by, min(retired_at) AS retired_at FROM memory_sources "
        "WHERE replaced_by IS NOT NULL GROUP BY corpus_id, replaced_by) p "
        "WHERE s.corpus_id = p.corpus_id AND s.source_id = p.replaced_by",
        "UPDATE memory_chunks c SET valid_from = s.valid_from, valid_to = s.valid_to "
        "FROM memory_sources s WHERE c.corpus_id = s.corpus_id AND c.source_id = s.source_id "
        "AND (s.valid_from IS NOT NULL OR s.valid_to IS NOT NULL)",
    ]


def _attribution(settings: Settings) -> list[str]:
    return [
        "ALTER TABLE memory_stories ADD COLUMN written_by JSONB NOT NULL DEFAULT '{}', "
        "ADD COLUMN closed_by JSONB NULL",
        "ALTER TABLE memory_sources ADD COLUMN written_by JSONB NOT NULL DEFAULT '{}', "
        "ADD COLUMN closed_by JSONB NULL",
        "ALTER TABLE memory_chunks ADD COLUMN written_by JSONB NOT NULL DEFAULT '{}'",
        "UPDATE memory_stories SET closed_by = '{}' WHERE closed_at IS NOT NULL",
        "UPDATE memory_sources SET closed_by = '{}' WHERE retired_at IS NOT NULL",
    ]


def _resemblance(settings: Settings) -> list[str]:
    return [
        """CREATE FUNCTION memory_signature(body text) RETURNS bigint[] LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
            WITH words AS (
                SELECT word, row_number() OVER (ORDER BY ord) AS i, count(*) OVER () AS total
                FROM regexp_split_to_table(lower(body), '[^[:alnum:]]+') WITH ORDINALITY AS w(word, ord)
                WHERE word <> ''
            ), slots AS (
                SELECT h & 127 AS slot, min(h) AS m FROM (
                    SELECT ('x' || left(md5(concat_ws(' ', word, lead(word, 1) OVER run, lead(word, 2) OVER run,
                        lead(word, 3) OVER run, lead(word, 4) OVER run)), 16))::bit(64)::bigint AS h, i, total
                    FROM words WINDOW run AS (ORDER BY i)
                ) AS runs WHERE i <= greatest(total - 4, 1) GROUP BY 1
            ), ring AS (
                SELECT p, s.m FROM generate_series(0, 255) AS p LEFT JOIN slots AS s ON s.slot = p % 128
            ), nearest AS (
                SELECT p, min(p) FILTER (WHERE m IS NOT NULL) OVER (ORDER BY p DESC) AS q FROM ring
            )
            SELECT array_agg(r.m # (n.q - n.p) ORDER BY n.p)
            FROM nearest AS n JOIN ring AS r ON r.p = n.q WHERE n.p < 128 $$""",
        """CREATE FUNCTION memory_bands(signature bigint[]) RETURNS bigint[] LANGUAGE sql IMMUTABLE PARALLEL SAFE
            COST 10000 AS $$
            SELECT coalesce(array_agg(('x' || left(md5(b || ':' || array_to_string(signature[b * 3 + 1:b * 3 + 3], ',')),
                16))::bit(64)::bigint ORDER BY b), '{}')
            FROM generate_series(0, 41) AS b WHERE signature[1] IS NOT NULL $$""",
        """CREATE FUNCTION memory_share(a bigint[], b bigint[]) RETURNS double precision LANGUAGE sql IMMUTABLE
            PARALLEL SAFE AS $$
            SELECT 2 * j / (1 + j) FROM (
                SELECT count(*) FILTER (WHERE x = y)::double precision / greatest(count(*), 1) AS j
                FROM unnest(a, b) AS t(x, y)
            ) AS q $$""",
        "ALTER TABLE memory_sources ADD COLUMN signature bigint[] "
        "GENERATED ALWAYS AS (memory_signature(text)) STORED",
        "CREATE INDEX ix_memory_sources_bands ON memory_sources USING GIN (memory_bands(signature))",
    ]


def _lengths(settings: Settings) -> list[str]:
    return [
        "CREATE FUNCTION memory_length(config text, body text) RETURNS integer LANGUAGE sql IMMUTABLE "
        "PARALLEL SAFE AS $$ SELECT coalesce(sum(coalesce(array_length(u.positions, 1), 1)), 0)::integer "
        "FROM unnest(memory_tsvector(config, body)) AS u $$",
        "ALTER TABLE memory_chunks ADD COLUMN fts_length integer "
        f"GENERATED ALWAYS AS (memory_length(config, {_CHUNK_SEARCHED})) STORED",
        "ALTER TABLE memory_stories ADD COLUMN fts_length integer "
        "GENERATED ALWAYS AS (memory_length(config, text)) STORED",
    ]


def _indexes(settings: Settings) -> list[str]:
    return [
        "CREATE INDEX ix_memory_sources_replaced_by ON memory_sources (corpus_id, replaced_by) "
        "WHERE replaced_by IS NOT NULL",
        "CREATE INDEX ix_memory_sources_retired ON memory_sources (corpus_id, retired_at) "
        "WHERE retired_at IS NOT NULL",
        "CREATE INDEX ix_memory_sources_listed ON memory_sources "
        '(corpus_id, created_at DESC, source_id COLLATE "C" DESC)',
    ]


def _refinery(settings: Settings) -> list[str]:
    return [
        """CREATE TABLE memory_retrievals (
            retrieval_id TEXT NOT NULL PRIMARY KEY, corpus_ids TEXT[] NOT NULL, verb TEXT NOT NULL,
            at TIMESTAMPTZ NOT NULL, recorded TIMESTAMPTZ NOT NULL, duration_ms INTEGER NOT NULL,
            body JSON NOT NULL)""",
        "CREATE INDEX ix_memory_retrievals_corpora ON memory_retrievals USING GIN (corpus_ids)",
        "CREATE INDEX ix_memory_retrievals_recorded ON memory_retrievals (recorded)",
        """CREATE TABLE memory_retrieval_labels (
            retrieval_id TEXT NOT NULL REFERENCES memory_retrievals (retrieval_id) ON DELETE CASCADE,
            label_id TEXT NOT NULL, source TEXT NOT NULL, at TIMESTAMPTZ NOT NULL, body JSON NOT NULL,
            PRIMARY KEY (retrieval_id, label_id))""",
    ]


KEY_TEXT_BYTES = 2000


def _keys(settings: Settings) -> list[str]:
    return [
        """CREATE TABLE memory_indexed_keys (
            key TEXT NOT NULL PRIMARY KEY, type TEXT NOT NULL CHECK (type IN ('number', 'text')),
            tag TEXT NOT NULL UNIQUE, ready BOOLEAN NOT NULL DEFAULT false)""",
        "CREATE FUNCTION memory_key_number(doc jsonb, path text[]) RETURNS numeric LANGUAGE sql IMMUTABLE "
        "PARALLEL SAFE AS $$ SELECT CASE WHEN jsonb_typeof(doc #> path) = 'number' "
        "THEN (doc #>> path)::numeric END $$",
        "CREATE FUNCTION memory_key_text(doc jsonb, path text[]) RETURNS text LANGUAGE sql IMMUTABLE "
        "PARALLEL SAFE AS $$ SELECT CASE WHEN jsonb_typeof(doc #> path) = 'string' "
        f"AND octet_length(doc #>> path) <= {KEY_TEXT_BYTES} THEN doc #>> path END $$",
        "CREATE FUNCTION memory_key_fits(doc jsonb, path text[], kind text) RETURNS boolean LANGUAGE sql "
        "IMMUTABLE PARALLEL SAFE AS $$ SELECT CASE coalesce(jsonb_typeof(doc #> path), 'null') "
        "WHEN 'null' THEN true WHEN 'number' THEN kind = 'number' "
        f"WHEN 'string' THEN kind = 'text' AND octet_length(doc #>> path) <= {KEY_TEXT_BYTES} "
        "ELSE false END $$",
    ]


_CODE = "([[:alnum:]]+(?:[-_:#][[:alnum:]]+)+)"


def _codes(settings: Settings) -> list[str]:
    return [
        "CREATE FUNCTION memory_codes(body text) RETURNS tsvector LANGUAGE sql IMMUTABLE PARALLEL SAFE "
        "AS $$ SELECT coalesce(string_agg(format('''%s'':%s', c.code, "
        "(SELECT string_agg(g::text, ',') FROM generate_series(1, least(c.n, 256)) AS g)), ' ')::tsvector, "
        "''::tsvector) FROM (SELECT lower(m[1]) AS code, count(*)::int AS n "
        f"FROM regexp_matches(body, '{_CODE}', 'g') AS m WHERE m[1] ~ '[0-9_:#]' "
        "AND octet_length(lower(m[1])) <= 2046 GROUP BY 1) AS c $$",
        "CREATE FUNCTION memory_required(query tsquery, body text) RETURNS tsquery LANGUAGE sql IMMUTABLE "
        "PARALLEL SAFE AS $$ SELECT CASE WHEN c.codes IS NULL THEN query WHEN numnode(query) = 0 "
        "THEN c.codes ELSE query && c.codes END FROM (SELECT string_agg(quote_literal(l), ' & ')::tsquery "
        "AS codes FROM unnest(tsvector_to_array(memory_codes(body))) AS l) AS c $$",
        "CREATE OR REPLACE FUNCTION memory_length(config text, body text) RETURNS integer LANGUAGE sql "
        "IMMUTABLE PARALLEL SAFE AS $$ SELECT coalesce(sum(coalesce(array_length(u.positions, 1), 1)), 0)"
        "::integer FROM unnest(to_tsvector(config::regconfig, body)) AS u $$",
        "CREATE OR REPLACE FUNCTION memory_tsvector(config text, body text) RETURNS tsvector LANGUAGE sql "
        "IMMUTABLE PARALLEL SAFE AS $$ SELECT w || ts_delete(memory_codes(body), tsvector_to_array(w)) "
        "FROM to_tsvector(config::regconfig, body) AS w $$",
        "UPDATE memory_chunks SET config = config",
        "UPDATE memory_stories SET config = config",
    ]


_COUNTED = (
    "SELECT fts, fts_length FROM memory_chunks WHERE the_kind = 'chunk' AND corpus_id = the_corpus "
    "AND source_id = ANY(the_ids) AND retired_at IS NULL UNION ALL "
    "SELECT fts, fts_length FROM memory_stories WHERE the_kind = 'story' AND corpus_id = the_corpus "
    "AND story_id = ANY(the_ids) AND closed_at IS NULL"
)


_HELD = (
    "SELECT 'chunk' AS kind, fts, fts_length FROM memory_chunks WHERE corpus_id = the_corpus "
    "AND retired_at IS NULL UNION ALL SELECT 'story', fts, fts_length FROM memory_stories "
    "WHERE corpus_id = the_corpus AND closed_at IS NULL"
)


def _statistics(settings: Settings) -> list[str]:
    return [
        "ALTER TABLE memory_corpora ADD COLUMN statistics BOOLEAN NOT NULL DEFAULT false",
        """CREATE TABLE memory_term_counts (
            corpus_id TEXT NOT NULL, kind TEXT NOT NULL, lexeme TEXT NOT NULL, passages BIGINT NOT NULL,
            PRIMARY KEY (corpus_id, kind, lexeme))""",
        """CREATE TABLE memory_term_totals (
            corpus_id TEXT NOT NULL, kind TEXT NOT NULL, passages BIGINT NOT NULL, length BIGINT NOT NULL,
            PRIMARY KEY (corpus_id, kind))""",
        "CREATE FUNCTION memory_count(the_corpus text, the_kind text, the_ids text[], the_sign integer) "
        "RETURNS void LANGUAGE plpgsql AS $$ BEGIN "
        "IF cardinality(the_ids) = 0 OR NOT EXISTS (SELECT 1 FROM memory_corpora "
        "WHERE corpus_id = the_corpus AND statistics) THEN RETURN; END IF; "
        "INSERT INTO memory_term_totals AS t (corpus_id, kind, passages, length) "
        "SELECT the_corpus, the_kind, the_sign * count(*), the_sign * coalesce(sum(p.fts_length), 0) "
        f"FROM ({_COUNTED}) AS p ON CONFLICT (corpus_id, kind) DO UPDATE "
        "SET passages = t.passages + EXCLUDED.passages, length = t.length + EXCLUDED.length; "
        "INSERT INTO memory_term_counts AS t (corpus_id, kind, lexeme, passages) "
        f"SELECT the_corpus, the_kind, l.lexeme, the_sign * count(*) FROM ({_COUNTED}) AS p "
        "CROSS JOIN LATERAL unnest(tsvector_to_array(p.fts)) AS l(lexeme) "
        'GROUP BY l.lexeme ORDER BY l.lexeme COLLATE "C" '
        "ON CONFLICT (corpus_id, kind, lexeme) DO UPDATE SET passages = t.passages + EXCLUDED.passages; "
        "IF the_sign < 0 THEN DELETE FROM memory_term_counts AS t WHERE t.corpus_id = the_corpus "
        "AND t.kind = the_kind AND t.passages <= 0 AND t.lexeme IN "
        f"(SELECT unnest(tsvector_to_array(p.fts)) FROM ({_COUNTED}) AS p); END IF; END $$",
        "CREATE FUNCTION memory_recount(the_corpus text) RETURNS bigint LANGUAGE plpgsql AS $$ "
        "DECLARE counted bigint; BEGIN "
        "DELETE FROM memory_term_counts WHERE corpus_id = the_corpus; "
        "DELETE FROM memory_term_totals WHERE corpus_id = the_corpus; "
        "IF NOT EXISTS (SELECT 1 FROM memory_corpora WHERE corpus_id = the_corpus AND statistics) "
        "THEN RETURN 0; END IF; "
        f"WITH held AS ({_HELD}) INSERT INTO memory_term_totals (corpus_id, kind, passages, length) "
        "SELECT the_corpus, k.kind, count(h.kind), coalesce(sum(h.fts_length), 0) "
        "FROM (VALUES ('chunk'), ('story')) AS k(kind) LEFT JOIN held AS h ON h.kind = k.kind "
        "GROUP BY k.kind; "
        f"WITH held AS ({_HELD}) INSERT INTO memory_term_counts (corpus_id, kind, lexeme, passages) "
        "SELECT the_corpus, h.kind, l.lexeme, count(*) FROM held AS h "
        "CROSS JOIN LATERAL unnest(tsvector_to_array(h.fts)) AS l(lexeme) GROUP BY h.kind, l.lexeme; "
        "SELECT coalesce(sum(passages), 0) INTO counted FROM memory_term_totals "
        "WHERE corpus_id = the_corpus; RETURN counted; END $$",
    ]


def _key_moments(settings: Settings) -> list[str]:
    return [
        "ALTER TABLE memory_indexed_keys ADD COLUMN built_at TIMESTAMPTZ NULL",
        "UPDATE memory_indexed_keys SET built_at = now() WHERE ready",
    ]


def _function_bodies(settings: Settings) -> list[str]:
    return [
        "CREATE OR REPLACE FUNCTION memory_tsvector(config text, body text) RETURNS tsvector LANGUAGE sql "
        "IMMUTABLE PARALLEL SAFE BEGIN ATOMIC SELECT w || ts_delete(memory_codes(body), tsvector_to_array(w)) "
        "FROM to_tsvector(config::regconfig, body) AS w; END",
        "CREATE OR REPLACE FUNCTION memory_required(query tsquery, body text) RETURNS tsquery LANGUAGE sql "
        "IMMUTABLE PARALLEL SAFE BEGIN ATOMIC SELECT CASE WHEN c.codes IS NULL THEN query "
        "WHEN numnode(query) = 0 THEN c.codes ELSE query && c.codes END FROM "
        "(SELECT string_agg(quote_literal(l), ' & ')::tsquery AS codes "
        "FROM unnest(tsvector_to_array(memory_codes(body))) AS l) AS c; END",
    ]


MIGRATIONS = [
    Migration(1, "create", _create),
    Migration(2, "spotting", _spotting),
    Migration(3, "validity", _validity),
    Migration(4, "attribution", _attribution),
    Migration(5, "resemblance", _resemblance),
    Migration(6, "lengths", _lengths),
    Migration(7, "indexes", _indexes),
    Migration(8, "refinery", _refinery),
    Migration(9, "indexed keys", _keys),
    Migration(10, "codes", _codes),
    Migration(11, "statistics", _statistics),
    Migration(12, "key moments", _key_moments),
    Migration(13, "function bodies", _function_bodies),
]


LATEST = MIGRATIONS[-1].version


_STATE = (
    "SELECT current_schema() AS schema, "
    "current_setting('server_version_num')::int AS server, "
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


def _served(server: int) -> None:
    if server < 170000:
        raise ConfigurationError(
            f"postgres: memory needs Postgres 17 or later; this server is {server // 10000}.{server % 10000}"
        )


async def _applied(conn: Any) -> list[int]:
    state = await conn.fetchrow(_STATE)
    _served(state["server"])
    if not state["migrated"]:
        if state["tables"]:
            raise ConfigurationError(
                f"postgres: the memory tables in {state['schema']} predate migrations — create a new schema with rfnry-memory create"
            )
        raise ConfigurationError(
            f"postgres: no memory schema in {state['schema']} — run rfnry-memory create --schema {state['schema']}"
        )
    rows = await conn.fetch("SELECT version FROM memory_migrations ORDER BY version")
    return [int(r["version"]) for r in rows]


async def plan_create(
    conn: Any, *, dimension: int | None, graph: bool = True, language: str | None = None
) -> Plan:
    if dimension is not None and (isinstance(dimension, bool) or dimension < 1):
        raise InputError("postgres: a dimension is a positive int")
    state = await conn.fetchrow(_STATE)
    _served(state["server"])
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
