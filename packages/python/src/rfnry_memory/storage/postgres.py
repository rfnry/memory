from __future__ import annotations

import json
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, cast

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError, StorageError
from rfnry_memory.filters import FACET_PREFIX
from rfnry_memory.models import (
    Chunk,
    Counts,
    Entity,
    Kind,
    Language,
    Leg,
    Ref,
    Relation,
    Source,
    Story,
    Window,
)
from rfnry_memory.storage.storage import (
    GraphWrite,
    SourceWrite,
    chunk_digest,
    closable,
    ended_on_write,
    name_key,
    reconciled,
    text_hash,
    total,
)

_ALIAS_KEY = r"btrim(regexp_replace(lower(a), '\s+', ' ', 'g'))"
_EXTENSIONS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "CREATE EXTENSION IF NOT EXISTS unaccent",
]
_CONFIGS: dict[Language, str] = {
    "spanish": "memory_spanish",
    "portuguese": "memory_portuguese",
    "simple": "memory_simple",
}
_LANGUAGE = "coalesce((SELECT language FROM memory_corpora WHERE corpus_id = $1), 'english')"


def _config(language: str) -> str:
    cases = " ".join(
        f"WHEN '{name}' THEN '{config}'::regconfig" for name, config in _CONFIGS.items()
    )
    return f"CASE {language} {cases} ELSE 'english'::regconfig END"


def _indexed(text: str) -> str:
    cases = " ".join(
        f"WHEN '{name}' THEN to_tsvector('{config}'::regconfig, {text})"
        for name, config in _CONFIGS.items()
    )
    return f"CASE language {cases} ELSE to_tsvector('english'::regconfig, {text}) END"


_ANY_TERM = (
    "(SELECT string_agg(quote_literal(l), ' | ')::tsquery "
    f"FROM unnest(tsvector_to_array(to_tsvector({_config(_LANGUAGE)}, $2))) AS l)"
)


def _folding(name: str, base: str, dictionary: str) -> str:
    return f"""DO $$ DECLARE folding text; BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = '{name}' AND cfgnamespace = current_schema()::regnamespace) THEN
            SELECT format('%I.unaccent', n.nspname) INTO folding FROM pg_extension e
                JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'unaccent';
            CREATE TEXT SEARCH CONFIGURATION {name} (COPY = pg_catalog.{base});
            EXECUTE format('ALTER TEXT SEARCH CONFIGURATION {name} ALTER MAPPING FOR hword, hword_part, word WITH %s, pg_catalog.{dictionary}', folding);
        END IF;
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$"""


def _ddl(dimension: int | None) -> list[str]:
    vector = (
        f"embedding vector({dimension}) NULL, embedding_model TEXT NULL,"
        if dimension
        else "embedding_model TEXT NULL,"
    )
    return [
        _folding("memory_spanish", "spanish", "spanish_stem"),
        _folding("memory_portuguese", "portuguese", "portuguese_stem"),
        _folding("memory_simple", "simple", "simple"),
        """CREATE TABLE IF NOT EXISTS memory_corpora (
            corpus_id TEXT NOT NULL PRIMARY KEY, language TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS memory_sources (
            corpus_id TEXT NOT NULL, source_id TEXT NOT NULL, name TEXT NOT NULL, metadata JSON NOT NULL DEFAULT '{}',
            text TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL, chunks INTEGER NOT NULL, digest TEXT NULL,
            PRIMARY KEY (corpus_id, source_id))""",
        "CREATE INDEX IF NOT EXISTS ix_memory_sources_metadata ON memory_sources USING GIN ((metadata::jsonb))",
        f"""CREATE TABLE IF NOT EXISTS memory_chunks (
            corpus_id TEXT NOT NULL, chunk_id TEXT NOT NULL, source_id TEXT NOT NULL, position INTEGER NOT NULL,
            source_name TEXT NOT NULL, title TEXT NULL, locator JSON NOT NULL DEFAULT '{{}}', text TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}',
            digest TEXT NOT NULL, {vector} language TEXT NOT NULL DEFAULT 'english',
            fts tsvector GENERATED ALWAYS AS ({_indexed("coalesce(title, '') || ' ' || text")}) STORED,
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
            {vector} language TEXT NOT NULL DEFAULT 'english',
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


class _Moved(StorageError):
    pass


async def _language_lock(conn: Any, corpus_id: str, *, shared: bool) -> None:
    lock = "pg_advisory_xact_lock_shared" if shared else "pg_advisory_xact_lock"
    await conn.execute(f"SELECT {lock}(hashtextextended($1, 0))", f"memory.language:{corpus_id}")


_OPEN_SQL = (
    "SELECT * FROM memory_stories WHERE corpus_id = $1 AND text_hash = $2 AND superseded_by IS NULL "
    'AND (valid_to IS NULL OR valid_to > $3) ORDER BY story_id COLLATE "C" LIMIT 1'
)


async def _text_lock(conn: Any, corpus_id: str, hashes: list[str]) -> None:
    for hash in sorted(set(hashes)):
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"memory.story:{corpus_id}:{hash}",
        )


_SUPERSEDES_TO_LIST = [
    "ALTER TABLE memory_stories ALTER COLUMN supersedes TYPE JSONB USING "
    "CASE WHEN supersedes IS NULL THEN '[]'::jsonb ELSE jsonb_build_array(supersedes) END",
    "ALTER TABLE memory_stories ALTER COLUMN supersedes SET DEFAULT '[]'::jsonb",
    "ALTER TABLE memory_stories ALTER COLUMN supersedes SET NOT NULL",
    "UPDATE memory_stories AS old SET superseded_by = new.story_id, closed_at = new.created_at "
    "FROM memory_stories AS new WHERE new.corpus_id = old.corpus_id AND new.supersedes ? old.story_id "
    "AND old.superseded_by IS NULL",
    "UPDATE memory_stories SET written_to = valid_to WHERE superseded_by IS NULL AND written_to IS NULL",
]


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _vector(vector: list[float] | None) -> str | None:
    return None if vector is None else "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _source(r: Any) -> Source:
    return Source(
        corpus_id=r["corpus_id"],
        source_id=r["source_id"],
        name=r["name"],
        metadata=_json(r["metadata"]),
        text=r["text"],
        created_at=r["created_at"],
        chunks=r["chunks"],
        digest=r["digest"],
    )


def _chunk(r: Any) -> Chunk:
    return Chunk(
        corpus_id=r["corpus_id"],
        chunk_id=r["chunk_id"],
        source_id=r["source_id"],
        source_name=r["source_name"],
        position=r["position"],
        title=r["title"],
        locator=_json(r["locator"]),
        text=r["text"],
        metadata=_json(r["metadata"]),
    )


def _story(r: Any) -> Story:
    return Story(
        corpus_id=r["corpus_id"],
        story_id=r["story_id"],
        text=r["text"],
        subject=r["subject"],
        valid_from=r["valid_from"],
        valid_to=r["valid_to"],
        created_at=r["created_at"],
        supersedes=tuple(_json(r["supersedes"])),
        superseded_by=r["superseded_by"],
        closed_at=r["closed_at"],
        source_ids=tuple(_json(r["source_ids"])),
        metadata=_json(r["metadata"]),
    )


def _entity(r: Any) -> Entity:
    return Entity(
        corpus_id=r["corpus_id"],
        entity_id=r["entity_id"],
        name=r["name"],
        key=r["key"],
        type=r["type"],
        aliases=tuple(_json(r["aliases"])),
    )


def _relation(r: Any) -> Relation:
    return Relation(
        corpus_id=r["corpus_id"],
        from_id=r["from_id"],
        to_id=r["to_id"],
        type=r["type"],
        evidence_kind=r["evidence_kind"],
        evidence_id=r["evidence_id"],
    )


def _count(status: str) -> int:
    return int(status.rsplit(" ", 1)[-1])


def _nested(path: str, value: Any) -> dict[str, Any]:
    parts = path.split(".")
    node: Any = value
    for part in reversed(parts):
        node = {part: node}
    return dict(node)


def _facet_sql(filters: dict[str, Any] | None, params: list[Any], column: str) -> str:
    if not filters:
        return ""
    clauses = []
    for key, condition in filters.items():
        path = key[len(FACET_PREFIX) :]
        values = condition if isinstance(condition, list) else [condition]
        alternatives = []
        for value in values:
            for shape in (value, [value]):
                params.append(json.dumps(_nested(path, shape)))
                alternatives.append(f"{column} @> ${len(params)}::jsonb")
        clauses.append("(" + " OR ".join(alternatives) + ")")
    return " AND " + " AND ".join(clauses)


def _window_sql(window: Window, params: list[Any], alias: str = "") -> str:
    start = f"{alias}valid_from"
    sql = ""
    if window.as_of is not None:
        params.append(window.as_of)
        sql += f" AND {start} <= ${len(params)} AND ({alias}valid_to IS NULL OR {alias}valid_to > ${len(params)})"
    if window.since is not None:
        params.append(window.since)
        sql += f" AND {start} >= ${len(params)}"
    if window.until is not None:
        params.append(window.until)
        sql += f" AND {start} <= ${len(params)}"
    return sql


def _subject_sql(subject: str | None, params: list[Any], alias: str = "") -> str:
    if subject is None:
        return ""
    params.append(subject)
    return f" AND {alias}subject = ${len(params)}"


def _subject_kinds(kinds: tuple[Kind, ...], subject: str | None) -> tuple[Kind, ...]:
    wanted: tuple[Kind, ...] = ("chunk", "story") if subject is None else ("story",)
    return tuple(kind for kind in wanted if kind in kinds)


class PostgresStorage:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._dimension: int | None = None

    @asynccontextmanager
    async def _guard(self, operation: str) -> AsyncIterator[None]:
        try:
            yield
        except (StorageError, NotFoundError, ConfigurationError, InputError):
            raise
        except Exception as exc:
            raise StorageError(f"postgres: storage rejected {operation}: {exc}") from exc

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
                await self._upgrade_stories(conn)
                self._dimension = await conn.fetchval(
                    "SELECT atttypmod FROM pg_attribute WHERE attrelid = to_regclass(format('%I.%I', current_schema(), 'memory_chunks')) "
                    "AND attname = 'embedding' AND NOT attisdropped"
                )

    async def _upgrade_stories(self, conn: Any) -> None:
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "memory.upgrade"
            )
            columns = {
                r["column_name"]: r["data_type"]
                for r in await conn.fetch(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name = 'memory_stories'"
                )
            }
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
            await conn.execute("DROP INDEX IF EXISTS ux_memory_stories_open_text")

    def _check_vectors(self, vectors: object) -> None:
        if vectors is not None and not self._dimension:
            raise StorageError(
                "postgres: vectors were given but this storage has no vector dimension — call initialize() first"
            )

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

    async def put_source(
        self,
        corpus_id: str,
        source: Source,
        chunks: list[Chunk],
        vectors: list[list[float]] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> None:
        if source.corpus_id != corpus_id or any(c.corpus_id != corpus_id for c in chunks):
            raise StorageError("postgres: row corpus does not match the call corpus")
        self._check_vectors(vectors)
        graph = graph or GraphWrite()
        graph.check(corpus_id, {("chunk", c.chunk_id) for c in chunks})
        async with (
            self._guard("put_source"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=True)
            touched: set[str] = set()
            await self._delete_source(conn, corpus_id, source.source_id, touched)
            await self._insert_source(conn, corpus_id, SourceWrite(source, chunks, vectors), model)
            await self._write_graph(conn, corpus_id, graph, touched)
            await self._reconcile(conn, corpus_id, touched)

    async def sync_sources(
        self,
        corpus_id: str,
        filters: dict[str, Any],
        writes: list[SourceWrite],
        model: str | None,
        *,
        graph: GraphWrite | None = None,
        keep: dict[str, str] | None = None,
    ) -> Counts | None:
        if any(
            source.corpus_id != corpus_id or any(c.corpus_id != corpus_id for c in chunks)
            for source, chunks, _ in writes
        ):
            raise StorageError("postgres: row corpus does not match the call corpus")
        keep = keep or {}
        for write in writes:
            self._check_vectors(write.vectors)
        graph = graph or GraphWrite()
        graph.check(corpus_id, {("chunk", c.chunk_id) for w in writes for c in w.chunks})
        params: list[Any] = [corpus_id]
        facets = _facet_sql(filters, params, "metadata::jsonb")
        try:
            async with (
                self._guard("sync_sources"),
                self._pool.acquire() as conn,
                conn.transaction(isolation="read_committed"),
            ):
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    f"memory.sync:{corpus_id}",
                )
                await _language_lock(conn, corpus_id, shared=True)
                matching = {
                    r["source_id"]: r["digest"]
                    for r in await conn.fetch(
                        f'SELECT source_id, digest FROM memory_sources WHERE corpus_id = $1{facets} ORDER BY source_id COLLATE "C" FOR UPDATE',
                        *params,
                    )
                }
                if any(matching.get(source_id) != digest for source_id, digest in keep.items()):
                    raise _Moved
                touched: set[str] = set()
                deleted = total(
                    [
                        await self._delete_source(conn, corpus_id, source_id, touched)
                        for source_id in matching
                        if source_id not in keep
                    ]
                )
                for write in writes:
                    await self._insert_source(conn, corpus_id, write, model)
                await self._write_graph(conn, corpus_id, graph, touched)
                pruned = await self._reconcile(conn, corpus_id, touched)
        except _Moved:
            return None
        return deleted.model_copy(update={"entities": pruned})

    async def chunk_vectors(
        self, corpus_id: str, model: str, digests: list[str]
    ) -> dict[str, list[float]]:
        if not self._dimension or not digests:
            return {}
        async with self._guard("chunk_vectors"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT DISTINCT ON (digest) digest, embedding::text AS embedding FROM memory_chunks "
                "WHERE corpus_id = $1 AND embedding_model = $2 AND digest = ANY($3::text[]) AND embedding IS NOT NULL",
                corpus_id,
                model,
                digests,
            )
        return {r["digest"]: [float(x) for x in json.loads(r["embedding"])] for r in records}

    async def _insert_source(
        self, conn: Any, corpus_id: str, write: SourceWrite, model: str | None
    ) -> None:
        source, chunks, vectors = write
        await conn.execute(
            "INSERT INTO memory_sources (corpus_id, source_id, name, metadata, text, created_at, chunks, digest) "
            "VALUES ($1, $2, $3, $4::json, $5, $6, $7, $8)",
            corpus_id,
            source.source_id,
            source.name,
            json.dumps(source.metadata),
            source.text,
            source.created_at,
            source.chunks,
            source.digest,
        )
        rows = [
            (
                corpus_id,
                c.chunk_id,
                c.source_id,
                source.name,
                c.position,
                c.title,
                json.dumps(c.locator),
                c.text,
                json.dumps(source.metadata),
                chunk_digest(c.text),
                *(
                    [_vector(vectors[i] if vectors is not None else None)]
                    if self._dimension
                    else []
                ),
                model,
            )
            for i, c in enumerate(chunks)
        ]
        if rows:
            await conn.executemany(self._insert_chunk_sql(), rows)

    async def _write_graph(
        self, conn: Any, corpus_id: str, graph: GraphWrite, touched: set[str]
    ) -> None:
        ids: dict[str, str] = {}
        for entity in sorted(graph.entities, key=lambda e: (e.key, e.type)):
            ids[entity.entity_id] = await conn.fetchval(
                "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases) VALUES ($1, $2, $3, $4, $5, $6::jsonb) "
                "ON CONFLICT (corpus_id, key, type) DO UPDATE SET aliases = memory_entities.aliases || coalesce((SELECT jsonb_agg(a) "
                "FROM jsonb_array_elements_text(EXCLUDED.aliases) AS a WHERE NOT memory_entities.aliases ? a), '[]'::jsonb) "
                "RETURNING entity_id",
                corpus_id,
                entity.entity_id,
                entity.key,
                entity.name,
                entity.type,
                json.dumps(list(dict.fromkeys(entity.aliases))),
            )
        named = sorted(graph.named(ids))
        found = {
            r["entity_id"]
            for r in await conn.fetch(
                "SELECT entity_id FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[]) "
                'ORDER BY entity_id COLLATE "C" FOR KEY SHARE',
                corpus_id,
                named,
            )
        }
        known = {e.entity_id: e for e in graph.known}
        for entity_id in sorted((set(named) - found) & set(known)):
            entity = known[entity_id]
            ids[entity_id] = await conn.fetchval(
                "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases) VALUES ($1, $2, $3, $4, $5, $6::jsonb) "
                "ON CONFLICT DO NOTHING RETURNING entity_id",
                corpus_id,
                entity.entity_id,
                entity.key,
                entity.name,
                entity.type,
                json.dumps(list(dict.fromkeys(entity.aliases))),
            ) or await conn.fetchval(
                "SELECT entity_id FROM memory_entities WHERE corpus_id = $1 AND (entity_id = $2 OR (key = $3 AND type = $4)) "
                "ORDER BY entity_id = $2 DESC LIMIT 1 FOR KEY SHARE",
                corpus_id,
                entity.entity_id,
                entity.key,
                entity.type,
            )
            touched.add(ids[entity_id])
        missing = set(named) - found - set(known)
        if missing:
            raise StorageError(f"postgres: graph facts name unknown entities: {sorted(missing)}")
        for entity_id, alias in sorted((ids.get(e, e), a) for e, a in graph.aliases):
            await conn.execute(
                "UPDATE memory_entities SET aliases = CASE WHEN aliases ? $3 THEN aliases ELSE aliases || to_jsonb($3::text) END "
                "WHERE corpus_id = $1 AND entity_id = $2",
                corpus_id,
                entity_id,
                alias,
            )
        if graph.relations:
            await conn.executemany(
                "INSERT INTO memory_relations (corpus_id, from_id, to_id, type, evidence_kind, evidence_id) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING",
                [
                    (
                        corpus_id,
                        ids.get(r.from_id, r.from_id),
                        ids.get(r.to_id, r.to_id),
                        r.type,
                        r.evidence_kind,
                        r.evidence_id,
                    )
                    for r in graph.relations
                ],
            )
        if graph.mentions:
            await conn.executemany(
                "INSERT INTO memory_mentions (corpus_id, entity_id, passage_kind, passage_id, name) VALUES ($1, $2, $3, $4, $5) ON CONFLICT DO NOTHING",
                [
                    (corpus_id, ids.get(e, e), kind, passage_id, name)
                    for e, kind, passage_id, name in graph.mentions
                ],
            )
            await self._carry(conn, corpus_id, graph.mentions, ids)

    async def _carry(
        self,
        conn: Any,
        corpus_id: str,
        mentions: list[tuple[str, Kind, str, str]],
        ids: dict[str, str],
    ) -> None:
        said: dict[str, list[str]] = {}
        for entity_id, _, _, name in mentions:
            said.setdefault(ids.get(entity_id, entity_id), []).append(name)
        for r in await conn.fetch(
            "SELECT entity_id, key, aliases FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[]) "
            'ORDER BY entity_id COLLATE "C"',
            corpus_id,
            sorted(said),
        ):
            carried = {r["key"], *map(name_key, _json(r["aliases"]))}
            new: list[str] = []
            for name in sorted(said[r["entity_id"]]):
                if name_key(name) not in carried:
                    carried.add(name_key(name))
                    new.append(name)
            if new:
                await conn.execute(
                    "UPDATE memory_entities SET aliases = aliases || $3::jsonb WHERE corpus_id = $1 AND entity_id = $2",
                    corpus_id,
                    r["entity_id"],
                    json.dumps(new),
                )

    def _insert_chunk_sql(self) -> str:
        columns = "corpus_id, chunk_id, source_id, source_name, position, title, locator, text, metadata, digest, language"
        if self._dimension:
            return (
                f"INSERT INTO memory_chunks ({columns}, embedding, embedding_model) "
                f"VALUES ($1, $2, $3, $4, $5, $6, $7::json, $8, $9::jsonb, $10, {_LANGUAGE}, $11::vector, $12)"
            )
        return (
            f"INSERT INTO memory_chunks ({columns}, embedding_model) "
            f"VALUES ($1, $2, $3, $4, $5, $6, $7::json, $8, $9::jsonb, $10, {_LANGUAGE}, $11)"
        )

    async def get_source(self, corpus_id: str, source_id: str) -> Source | None:
        async with self._guard("get_source"), self._pool.acquire() as conn:
            record = await conn.fetchrow(
                "SELECT * FROM memory_sources WHERE corpus_id = $1 AND source_id = $2",
                corpus_id,
                source_id,
            )
        return _source(record) if record else None

    async def list_sources(
        self, corpus_id: str, *, filters: dict[str, Any] | None, limit: int, cursor: str | None
    ) -> tuple[list[Source], str | None]:
        params: list[Any] = [corpus_id]
        facets = _facet_sql(filters, params, "metadata::jsonb")
        offset = int(cursor) if cursor else 0
        params.extend([limit + 1, offset])
        async with self._guard("list_sources"), self._pool.acquire() as conn:
            records = await conn.fetch(
                f'SELECT * FROM memory_sources WHERE corpus_id = $1{facets} ORDER BY created_at DESC, source_id COLLATE "C" DESC '
                f"LIMIT ${len(params) - 1} OFFSET ${len(params)}",
                *params,
            )
        rows = [_source(r) for r in records]
        return rows[:limit], (str(offset + limit) if len(rows) > limit else None)

    async def source_chunks(self, corpus_id: str, source_id: str) -> list[Chunk]:
        async with self._guard("source_chunks"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_chunks WHERE corpus_id = $1 AND source_id = $2 ORDER BY position",
                corpus_id,
                source_id,
            )
        return [_chunk(r) for r in records]

    async def source_digests(self, corpus_id: str) -> list[tuple[str, dict[str, Any], str]]:
        async with self._guard("source_digests"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT name, metadata, encode(sha256(convert_to(text, 'UTF8')), 'hex') AS text_sha256 "
                "FROM memory_sources WHERE corpus_id = $1",
                corpus_id,
            )
        return [(r["name"], _json(r["metadata"]), r["text_sha256"]) for r in records]

    async def _delete_source(
        self, conn: Any, corpus_id: str, source_id: str, touched: set[str]
    ) -> Counts:
        chunk_ids = [
            r["chunk_id"]
            for r in await conn.fetch(
                "SELECT chunk_id FROM memory_chunks WHERE corpus_id = $1 AND source_id = $2",
                corpus_id,
                source_id,
            )
        ]
        mentions, relations = await self._drop_evidence(
            conn, corpus_id, "chunk", chunk_ids, touched
        )
        sources = await conn.execute(
            "DELETE FROM memory_sources WHERE corpus_id = $1 AND source_id = $2",
            corpus_id,
            source_id,
        )
        return Counts(
            sources=_count(sources), chunks=len(chunk_ids), mentions=mentions, relations=relations
        )

    async def _drop_evidence(
        self, conn: Any, corpus_id: str, kind: Kind, passage_ids: list[str], touched: set[str]
    ) -> tuple[int, int]:
        if not passage_ids:
            return 0, 0
        mentioned = await conn.fetch(
            "DELETE FROM memory_mentions WHERE corpus_id = $1 AND passage_kind = $2 AND passage_id = ANY($3::text[]) "
            "RETURNING entity_id",
            corpus_id,
            kind,
            passage_ids,
        )
        related = await conn.fetch(
            "DELETE FROM memory_relations WHERE corpus_id = $1 AND evidence_kind = $2 AND evidence_id = ANY($3::text[]) "
            "RETURNING from_id, to_id",
            corpus_id,
            kind,
            passage_ids,
        )
        touched.update(r["entity_id"] for r in mentioned)
        touched.update(e for r in related for e in (r["from_id"], r["to_id"]))
        return len(mentioned), len(related)

    async def _reconcile(self, conn: Any, corpus_id: str, touched: set[str]) -> int:
        if not touched:
            return 0
        ids = sorted(touched)
        entities = [
            _entity(r)
            for r in await conn.fetch(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[]) "
                'ORDER BY entity_id COLLATE "C" FOR UPDATE',
                corpus_id,
                ids,
            )
        ]
        names: dict[str, list[str]] = {}
        for r in await conn.fetch(
            "SELECT entity_id, name FROM memory_mentions WHERE corpus_id = $1 AND entity_id = ANY($2::text[])",
            corpus_id,
            ids,
        ):
            names.setdefault(r["entity_id"], []).append(r["name"])
        related = {
            r["entity_id"]
            for r in await conn.fetch(
                "SELECT from_id AS entity_id FROM memory_relations WHERE corpus_id = $1 AND from_id = ANY($2::text[]) "
                "UNION SELECT to_id FROM memory_relations WHERE corpus_id = $1 AND to_id = ANY($2::text[])",
                corpus_id,
                ids,
            )
        }
        holders: dict[tuple[str, str], set[str]] = {}
        for r in await conn.fetch(
            "SELECT entity_id, key, type FROM memory_entities WHERE corpus_id = $1 AND key = ANY($2::text[])",
            corpus_id,
            sorted({name_key(n) for held in names.values() for n in held}),
        ):
            holders.setdefault((r["key"], r["type"]), set()).add(r["entity_id"])
        pruned: list[str] = []
        for entity in entities:
            held = names.get(entity.entity_id, [])
            taken = {
                key
                for key in map(name_key, held)
                if holders.get((key, entity.type), set()) - {entity.entity_id}
            }
            kept = reconciled(entity, held, entity.entity_id in related, taken)
            if kept is None:
                pruned.append(entity.entity_id)
            elif kept != entity:
                blocked = reconciled(
                    entity, held, entity.entity_id in related, {name_key(n) for n in held}
                )
                assert blocked is not None
                await self._settle(conn, corpus_id, entity, kept, blocked)
                holders.setdefault((kept.key, kept.type), set()).add(kept.entity_id)
                holders.get((entity.key, entity.type), set()).discard(entity.entity_id)
        if pruned:
            await conn.execute(
                "DELETE FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[])",
                corpus_id,
                pruned,
            )
        return len(pruned)

    async def _settle(
        self, conn: Any, corpus_id: str, entity: Entity, kept: Entity, blocked: Entity
    ) -> None:
        update = (
            "UPDATE memory_entities SET name = $3, key = $4, aliases = $5::jsonb "
            "WHERE corpus_id = $1 AND entity_id = $2"
        )
        if kept.key != entity.key:
            try:
                async with conn.transaction():
                    await conn.execute(
                        update,
                        corpus_id,
                        entity.entity_id,
                        kept.name,
                        kept.key,
                        json.dumps(list(kept.aliases)),
                    )
                return
            except Exception as exc:
                if getattr(exc, "sqlstate", None) != "23505":
                    raise
                kept = blocked
        await conn.execute(
            update,
            corpus_id,
            entity.entity_id,
            kept.name,
            kept.key,
            json.dumps(list(kept.aliases)),
        )

    async def delete_source(self, corpus_id: str, source_id: str) -> Counts:
        async with (
            self._guard("delete_source"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            touched: set[str] = set()
            deleted = await self._delete_source(conn, corpus_id, source_id, touched)
            pruned = await self._reconcile(conn, corpus_id, touched)
        return deleted.model_copy(update={"entities": pruned})

    def _insert_story_sql(self) -> str:
        columns = (
            "corpus_id, story_id, text, text_hash, subject, valid_from, valid_to, created_at, "
            "supersedes, superseded_by, closed_at, written_to, source_ids, metadata, language"
        )
        values = (
            "$1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11, $12, $13::jsonb, $14::json, "
            + _LANGUAGE
        )
        if self._dimension:
            return (
                f"INSERT INTO memory_stories ({columns}, embedding, embedding_model) "
                f"VALUES ({values}, $15::vector, $16)"
            )
        return f"INSERT INTO memory_stories ({columns}, embedding_model) VALUES ({values}, $15)"

    def _story_params(
        self, story: Story, vector: list[float] | None, model: str | None
    ) -> list[Any]:
        return [
            story.corpus_id,
            story.story_id,
            story.text,
            text_hash(story.text),
            story.subject,
            story.valid_from,
            story.valid_to,
            story.created_at,
            json.dumps(list(story.supersedes)),
            story.superseded_by,
            story.closed_at,
            story.valid_to,
            json.dumps(list(story.source_ids)),
            json.dumps(story.metadata),
            *([_vector(vector)] if self._dimension else []),
            model,
        ]

    async def _held(self, conn: Any, corpus_id: str, story: Story) -> Story | None:
        hash = text_hash(story.text)
        if ended_on_write(story):
            record = await conn.fetchrow(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND text_hash = $2 "
                'AND valid_from = $3 AND valid_to = $4 ORDER BY story_id COLLATE "C" LIMIT 1',
                corpus_id,
                hash,
                story.valid_from,
                story.valid_to,
            )
        else:
            record = await conn.fetchrow(_OPEN_SQL, corpus_id, hash, story.created_at)
        return _story(record) if record else None

    async def put_story(
        self,
        corpus_id: str,
        story: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> Story:
        if story.corpus_id != corpus_id:
            raise StorageError("postgres: row corpus does not match the call corpus")
        self._check_vectors(vector)
        graph = graph or GraphWrite()
        graph.check(corpus_id, {("story", story.story_id)})
        async with (
            self._guard("put_story"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=True)
            await _text_lock(conn, corpus_id, [text_hash(story.text)])
            holder = await self._held(conn, corpus_id, story)
            if holder is not None and holder.story_id != story.story_id:
                return holder
            await conn.execute(
                "DELETE FROM memory_stories WHERE corpus_id = $1 AND story_id = $2",
                corpus_id,
                story.story_id,
            )
            await conn.execute(self._insert_story_sql(), *self._story_params(story, vector, model))
            recreated: set[str] = set()
            await self._write_graph(conn, corpus_id, graph, recreated)
            await self._reconcile(conn, corpus_id, recreated)
        return story

    async def find_open_story(self, corpus_id: str, hash: str, *, at: datetime) -> Story | None:
        async with self._guard("find_open_story"), self._pool.acquire() as conn:
            record = await conn.fetchrow(_OPEN_SQL, corpus_id, hash, at)
        return _story(record) if record else None

    async def supersede_story(
        self,
        corpus_id: str,
        old_id: str,
        valid_to: datetime,
        new: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> tuple[Story, Story]:
        if new.corpus_id != corpus_id:
            raise StorageError("postgres: row corpus does not match the call corpus")
        self._check_vectors(vector)
        graph = graph or GraphWrite()
        graph.check(corpus_id, {("story", new.story_id)})
        async with (
            self._guard("supersede_story"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=True)
            old_hash = await conn.fetchval(
                "SELECT text_hash FROM memory_stories WHERE corpus_id = $1 AND story_id = $2",
                corpus_id,
                old_id,
            )
            await _text_lock(
                conn, corpus_id, [text_hash(new.text), *([old_hash] if old_hash else [])]
            )
            current = await conn.fetchrow(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = $2 FOR UPDATE",
                corpus_id,
                old_id,
            )
            if current is None:
                raise NotFoundError(f"postgres: story {old_id!r} was not found")
            closable(_story(current), new, "postgres")
            holder = await conn.fetchrow(_OPEN_SQL, corpus_id, text_hash(new.text), new.created_at)
            if holder is not None and holder["story_id"] != old_id:
                raise InputError(
                    f"postgres: story {holder['story_id']!r}, true since {holder['valid_from'].isoformat()}, already says that; supersede or delete it instead"
                )
            closed = await conn.fetchrow(
                "UPDATE memory_stories SET valid_to = $3, superseded_by = $4, closed_at = $5 "
                "WHERE corpus_id = $1 AND story_id = $2 RETURNING *",
                corpus_id,
                old_id,
                valid_to,
                new.story_id,
                new.created_at,
            )
            await conn.execute(self._insert_story_sql(), *self._story_params(new, vector, model))
            recreated: set[str] = set()
            await self._write_graph(conn, corpus_id, graph, recreated)
            await self._reconcile(conn, corpus_id, recreated)
        return _story(closed), new

    async def get_stories(self, corpus_id: str, story_ids: list[str]) -> list[Story]:
        async with self._guard("get_stories"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = ANY($2::text[])",
                corpus_id,
                list(story_ids),
            )
        by_id = {r["story_id"]: _story(r) for r in records}
        return [by_id[i] for i in story_ids if i in by_id]

    async def list_stories(
        self,
        corpus_id: str,
        *,
        subject: str | None,
        window: Window,
        filters: dict[str, Any] | None,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[Story], str | None]:
        params: list[Any] = [corpus_id]
        sql = "SELECT * FROM memory_stories WHERE corpus_id = $1"
        sql += (
            _subject_sql(subject, params)
            + _window_sql(window, params)
            + _facet_sql(filters, params, "metadata::jsonb")
        )
        offset = int(cursor) if cursor else 0
        params.extend([limit + 1, offset])
        sql += f' ORDER BY valid_from DESC, story_id COLLATE "C" DESC LIMIT ${len(params) - 1} OFFSET ${len(params)}'
        async with self._guard("list_stories"), self._pool.acquire() as conn:
            records = await conn.fetch(sql, *params)
        rows = [_story(r) for r in records]
        return rows[:limit], (str(offset + limit) if len(rows) > limit else None)

    async def delete_story(self, corpus_id: str, story_id: str) -> Counts:
        async with (
            self._guard("delete_story"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            touched: set[str] = set()
            mentions, relations = await self._drop_evidence(
                conn, corpus_id, "story", [story_id], touched
            )
            stories = await conn.execute(
                "DELETE FROM memory_stories WHERE corpus_id = $1 AND story_id = $2",
                corpus_id,
                story_id,
            )
            pruned = await self._reconcile(conn, corpus_id, touched)
        return Counts(
            stories=_count(stories), mentions=mentions, relations=relations, entities=pruned
        )

    async def _search_kind(
        self,
        conn: Any,
        kind: Kind,
        score: str,
        condition: str,
        base: list[Any],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        top_k: int,
    ) -> list[Leg]:
        table, id_column = (
            ("memory_chunks", "chunk_id") if kind == "chunk" else ("memory_stories", "story_id")
        )
        params = list(base)
        extra = _facet_sql(filters, params, "metadata" if kind == "chunk" else "metadata::jsonb")
        if kind == "story":
            extra += _window_sql(window, params) + _subject_sql(subject, params)
        params.append(top_k)
        records = await conn.fetch(
            f"SELECT {id_column} AS id, {score} AS score FROM {table} WHERE corpus_id = $1 AND {condition}{extra} "
            f'ORDER BY score DESC, {id_column} COLLATE "C" LIMIT ${len(params)}',
            *params,
        )
        return [(kind, r["id"], float(r["score"])) for r in records]

    async def _search(
        self,
        operation: str,
        kinds: tuple[Kind, ...],
        score: str,
        condition: str,
        base: list[Any],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        top_k: int,
    ) -> list[Leg]:
        legs: list[Leg] = []
        async with self._guard(operation), self._pool.acquire() as conn:
            for kind in _subject_kinds(kinds, subject):
                legs.extend(
                    await self._search_kind(
                        conn, kind, score, condition, base, filters, window, subject, top_k
                    )
                )
        legs.sort(key=lambda leg: (-leg[2], leg[1]))
        return legs[:top_k]

    async def search_text(
        self,
        corpus_id: str,
        query: str,
        *,
        kinds: tuple[Kind, ...],
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> list[Leg]:
        return await self._search(
            "search_text",
            kinds,
            f"ts_rank_cd(fts, {_ANY_TERM})",
            f"fts @@ {_ANY_TERM}",
            [corpus_id, query],
            filters,
            window,
            subject,
            top_k,
        )

    async def search_vector(
        self,
        corpus_id: str,
        vector: list[float],
        *,
        kinds: tuple[Kind, ...],
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> list[Leg]:
        self._check_vectors(vector)
        return await self._search(
            "search_vector",
            kinds,
            "1 - (embedding <=> $2::vector)",
            "embedding IS NOT NULL",
            [corpus_id, _vector(vector)],
            filters,
            window,
            subject,
            top_k,
        )

    async def get_passages(self, corpus_id: str, refs: list[Ref]) -> list[Chunk | Story]:
        chunk_ids = [i for k, i in refs if k == "chunk"]
        story_ids = [i for k, i in refs if k == "story"]
        async with self._guard("get_passages"), self._pool.acquire() as conn:
            chunks = await conn.fetch(
                "SELECT * FROM memory_chunks WHERE corpus_id = $1 AND chunk_id = ANY($2::text[])",
                corpus_id,
                chunk_ids,
            )
            stories = await conn.fetch(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = ANY($2::text[])",
                corpus_id,
                story_ids,
            )
        found: dict[Ref, Chunk | Story] = {("chunk", r["chunk_id"]): _chunk(r) for r in chunks}
        found.update({("story", r["story_id"]): _story(r) for r in stories})
        return [found[ref] for ref in refs if ref in found]

    async def similar_entities(self, corpus_id: str, name: str, *, limit: int) -> list[Entity]:
        async with self._guard("similar_entities"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT *, greatest(similarity(name, $2), coalesce((SELECT max(similarity(a, $2)) "
                "FROM jsonb_array_elements_text(aliases) AS a), 0)) AS sim FROM memory_entities "
                'WHERE corpus_id = $1 ORDER BY sim DESC, entity_id COLLATE "C" LIMIT $3',
                corpus_id,
                name,
                limit,
            )
        return [_entity(r) for r in records if float(r["sim"]) > 0.3]

    async def entities_named(self, corpus_id: str, name_or_key: str) -> list[Entity]:
        key = name_key(name_or_key)
        async with self._guard("entities_named"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND (key = $2 OR EXISTS "
                f'(SELECT 1 FROM jsonb_array_elements_text(aliases) AS a WHERE {_ALIAS_KEY} = $2)) ORDER BY entity_id COLLATE "C"',
                corpus_id,
                key,
            )
        return [_entity(r) for r in records]

    async def entities_by_id(self, corpus_id: str, entity_ids: list[str]) -> list[Entity]:
        async with self._guard("entities_by_id"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[])",
                corpus_id,
                list(entity_ids),
            )
        by_id = {r["entity_id"]: _entity(r) for r in records}
        return [by_id[i] for i in entity_ids if i in by_id]

    async def entities_named_in(self, corpus_id: str, text: str) -> list[Entity]:
        haystack = name_key(text)
        async with self._guard("entities_named_in"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND (position(key IN $2) > 0 OR EXISTS "
                f'(SELECT 1 FROM jsonb_array_elements_text(aliases) AS a WHERE position({_ALIAS_KEY} IN $2) > 0)) ORDER BY entity_id COLLATE "C"',
                corpus_id,
                haystack,
            )
        return [_entity(r) for r in records]

    async def neighbors(
        self, corpus_id: str, entity_ids: list[str], *, relation_types: list[str] | None
    ) -> list[Relation]:
        params: list[Any] = [corpus_id, list(entity_ids)]
        sql = "SELECT * FROM memory_relations WHERE corpus_id = $1 AND (from_id = ANY($2::text[]) OR to_id = ANY($2::text[]))"
        if relation_types is not None:
            params.append(list(relation_types))
            sql += " AND type = ANY($3::text[])"
        async with self._guard("neighbors"), self._pool.acquire() as conn:
            records = await conn.fetch(
                sql
                + ' ORDER BY from_id COLLATE "C", to_id COLLATE "C", type COLLATE "C", evidence_kind COLLATE "C", evidence_id COLLATE "C"',
                *params,
            )
        return [_relation(r) for r in records]

    async def mentions(
        self,
        corpus_id: str,
        entity_ids: list[str],
        *,
        kinds: tuple[Kind, ...],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> list[Ref]:
        refs: list[Ref] = []
        kinds = _subject_kinds(kinds, subject)
        async with self._guard("mentions"), self._pool.acquire() as conn:
            if "chunk" in kinds:
                params: list[Any] = [corpus_id, list(entity_ids)]
                facets = _facet_sql(filters, params, "c.metadata")
                records = await conn.fetch(
                    'SELECT DISTINCT c.chunk_id COLLATE "C" AS chunk_id FROM memory_mentions m JOIN memory_chunks c ON c.corpus_id = m.corpus_id AND c.chunk_id = m.passage_id '
                    f"WHERE m.corpus_id = $1 AND m.passage_kind = 'chunk' AND m.entity_id = ANY($2::text[]){facets} ORDER BY chunk_id",
                    *params,
                )
                refs.extend(("chunk", r["chunk_id"]) for r in records)
            if "story" in kinds:
                params = [corpus_id, list(entity_ids)]
                facets = (
                    _facet_sql(filters, params, "s.metadata::jsonb")
                    + _window_sql(window, params, alias="s.")
                    + _subject_sql(subject, params, alias="s.")
                )
                records = await conn.fetch(
                    'SELECT DISTINCT s.story_id COLLATE "C" AS story_id FROM memory_mentions m JOIN memory_stories s ON s.corpus_id = m.corpus_id AND s.story_id = m.passage_id '
                    f"WHERE m.corpus_id = $1 AND m.passage_kind = 'story' AND m.entity_id = ANY($2::text[]){facets} ORDER BY story_id",
                    *params,
                )
                refs.extend(("story", r["story_id"]) for r in records)
        return refs

    async def passages_embedded_with(
        self, corpus_id: str, *, not_model: str, limit: int
    ) -> list[Chunk | Story]:
        async with self._guard("passages_embedded_with"), self._pool.acquire() as conn:
            chunks = await conn.fetch(
                'SELECT * FROM memory_chunks WHERE corpus_id = $1 AND embedding_model IS DISTINCT FROM $2 ORDER BY chunk_id COLLATE "C" LIMIT $3',
                corpus_id,
                not_model,
                limit,
            )
            stories = await conn.fetch(
                'SELECT * FROM memory_stories WHERE corpus_id = $1 AND embedding_model IS DISTINCT FROM $2 ORDER BY story_id COLLATE "C" LIMIT $3',
                corpus_id,
                not_model,
                max(limit - len(chunks), 0),
            )
        return [*(_chunk(r) for r in chunks), *(_story(r) for r in stories)]

    async def set_vectors(
        self, corpus_id: str, refs: list[Ref], vectors: list[list[float]], model: str
    ) -> None:
        self._check_vectors(vectors)
        chunk_rows = [
            (corpus_id, i, _vector(v), model)
            for (k, i), v in zip(refs, vectors, strict=True)
            if k == "chunk"
        ]
        story_rows = [
            (corpus_id, i, _vector(v), model)
            for (k, i), v in zip(refs, vectors, strict=True)
            if k == "story"
        ]
        async with self._guard("set_vectors"), self._pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                "UPDATE memory_chunks SET embedding = $3::vector, embedding_model = $4 WHERE corpus_id = $1 AND chunk_id = $2",
                chunk_rows,
            )
            await conn.executemany(
                "UPDATE memory_stories SET embedding = $3::vector, embedding_model = $4 WHERE corpus_id = $1 AND story_id = $2",
                story_rows,
            )

    async def delete_corpus(self, corpus_id: str) -> Counts:
        async with self._guard("delete_corpus"), self._pool.acquire() as conn, conn.transaction():
            counts = {}
            for field, table in (
                ("mentions", "memory_mentions"),
                ("relations", "memory_relations"),
                ("entities", "memory_entities"),
                ("stories", "memory_stories"),
                ("chunks", "memory_chunks"),
                ("sources", "memory_sources"),
                ("corpora", "memory_corpora"),
            ):
                counts[field] = _count(
                    await conn.execute(f"DELETE FROM {table} WHERE corpus_id = $1", corpus_id)
                )
        return Counts(**counts)

    async def count_corpus(self, corpus_id: str) -> Counts:
        async with self._guard("count_corpus"), self._pool.acquire() as conn:
            counts = {}
            for field, table in (
                ("sources", "memory_sources"),
                ("chunks", "memory_chunks"),
                ("stories", "memory_stories"),
                ("entities", "memory_entities"),
                ("relations", "memory_relations"),
                ("mentions", "memory_mentions"),
                ("corpora", "memory_corpora"),
            ):
                counts[field] = int(
                    await conn.fetchval(
                        f"SELECT count(*) FROM {table} WHERE corpus_id = $1", corpus_id
                    )
                )
        return Counts(**counts)

    async def get_language(self, corpus_id: str) -> Language:
        async with self._guard("get_language"), self._pool.acquire() as conn:
            language = await conn.fetchval(
                "SELECT language FROM memory_corpora WHERE corpus_id = $1", corpus_id
            )
        return cast(Language, language or "english")

    async def set_language(self, corpus_id: str, language: Language) -> int:
        if language != "english" and language not in _CONFIGS:
            raise StorageError(f"postgres: unknown language {language!r}")
        async with (
            self._guard("set_language"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=False)
            await conn.execute(
                "INSERT INTO memory_corpora (corpus_id, language) VALUES ($1, $2) "
                "ON CONFLICT (corpus_id) DO UPDATE SET language = EXCLUDED.language",
                corpus_id,
                language,
            )
            changed = 0
            for table in ("memory_chunks", "memory_stories"):
                changed += _count(
                    await conn.execute(
                        f"UPDATE {table} SET language = $2 WHERE corpus_id = $1 AND language <> $2",
                        corpus_id,
                        language,
                    )
                )
        return changed

    async def count_vectors(self, corpus_id: str) -> dict[str | None, int]:
        model = (
            "CASE WHEN embedding IS NOT NULL THEN embedding_model END"
            if self._dimension
            else "embedding_model"
        )
        counted: Counter[str | None] = Counter()
        async with self._guard("count_vectors"), self._pool.acquire() as conn:
            for table in ("memory_chunks", "memory_stories"):
                for r in await conn.fetch(
                    f"SELECT {model} AS model, count(*) AS n FROM {table} WHERE corpus_id = $1 GROUP BY 1",
                    corpus_id,
                ):
                    counted[r["model"]] += r["n"]
        return dict(counted)
