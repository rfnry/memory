from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from rfnry_memory.errors import ConfigurationError, NotFoundError, StorageError
from rfnry_memory.filters import FACET_PREFIX
from rfnry_memory.models import (
    Chunk,
    Deleted,
    Entity,
    Kind,
    Leg,
    Ref,
    Relation,
    Residue,
    Source,
    Story,
    Window,
)
from rfnry_memory.storage.storage import text_hash

_ALIAS_KEY = r"btrim(regexp_replace(lower(a), '\s+', ' ', 'g'))"
_EXTENSIONS = ["CREATE EXTENSION IF NOT EXISTS vector", "CREATE EXTENSION IF NOT EXISTS pg_trgm"]


def _ddl(dimension: int | None) -> list[str]:
    vector = f"embedding vector({dimension}) NULL, embedding_model TEXT NULL," if dimension else "embedding_model TEXT NULL,"
    statements = [
        """CREATE TABLE IF NOT EXISTS memory_sources (
            corpus_id TEXT NOT NULL, source_id TEXT NOT NULL, name TEXT NOT NULL, metadata JSON NOT NULL DEFAULT '{}',
            text TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL, chunks INTEGER NOT NULL,
            PRIMARY KEY (corpus_id, source_id))""",
        "CREATE INDEX IF NOT EXISTS ix_memory_sources_metadata ON memory_sources USING GIN ((metadata::jsonb))",
        f"""CREATE TABLE IF NOT EXISTS memory_chunks (
            corpus_id TEXT NOT NULL, chunk_id TEXT NOT NULL, source_id TEXT NOT NULL, position INTEGER NOT NULL,
            title TEXT NULL, locator JSON NOT NULL DEFAULT '{{}}', text TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{{}}',
            {vector}
            fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || text)) STORED,
            PRIMARY KEY (corpus_id, chunk_id),
            FOREIGN KEY (corpus_id, source_id) REFERENCES memory_sources (corpus_id, source_id) ON DELETE CASCADE)""",
        "CREATE INDEX IF NOT EXISTS ix_memory_chunks_source ON memory_chunks (corpus_id, source_id, position)",
        "CREATE INDEX IF NOT EXISTS ix_memory_chunks_fts ON memory_chunks USING GIN (fts)",
        "CREATE INDEX IF NOT EXISTS ix_memory_chunks_metadata ON memory_chunks USING GIN (metadata)",
        f"""CREATE TABLE IF NOT EXISTS memory_stories (
            corpus_id TEXT NOT NULL, story_id TEXT NOT NULL, text TEXT NOT NULL, text_hash TEXT NOT NULL,
            subject TEXT NULL, occurred_at TIMESTAMPTZ NULL, created_at TIMESTAMPTZ NOT NULL, valid_to TIMESTAMPTZ NULL,
            supersedes TEXT NULL, source_ids JSONB NOT NULL DEFAULT '[]', metadata JSON NOT NULL DEFAULT '{{}}',
            {vector}
            fts tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
            PRIMARY KEY (corpus_id, story_id))""",
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_memory_stories_open_text ON memory_stories (corpus_id, text_hash) WHERE valid_to IS NULL",
        "CREATE INDEX IF NOT EXISTS ix_memory_stories_subject ON memory_stories (corpus_id, subject)",
        "CREATE INDEX IF NOT EXISTS ix_memory_stories_event ON memory_stories (corpus_id, (coalesce(occurred_at, created_at)))",
        "CREATE INDEX IF NOT EXISTS ix_memory_stories_fts ON memory_stories USING GIN (fts)",
        "CREATE INDEX IF NOT EXISTS ix_memory_stories_metadata ON memory_stories USING GIN ((metadata::jsonb))",
        """CREATE TABLE IF NOT EXISTS memory_entities (
            corpus_id TEXT NOT NULL, entity_id TEXT NOT NULL, key TEXT NOT NULL, name TEXT NOT NULL, type TEXT NOT NULL,
            aliases JSONB NOT NULL DEFAULT '[]',
            PRIMARY KEY (corpus_id, entity_id), UNIQUE (corpus_id, key, type))""",
        "CREATE INDEX IF NOT EXISTS ix_memory_entities_name_trgm ON memory_entities USING GIN (name gin_trgm_ops)",
        """CREATE TABLE IF NOT EXISTS memory_relations (
            corpus_id TEXT NOT NULL, from_id TEXT NOT NULL, to_id TEXT NOT NULL, type TEXT NOT NULL,
            evidence_kind TEXT NOT NULL, evidence_id TEXT NOT NULL,
            PRIMARY KEY (corpus_id, from_id, to_id, type, evidence_kind, evidence_id))""",
        "CREATE INDEX IF NOT EXISTS ix_memory_relations_from ON memory_relations (corpus_id, from_id)",
        "CREATE INDEX IF NOT EXISTS ix_memory_relations_to ON memory_relations (corpus_id, to_id)",
        """CREATE TABLE IF NOT EXISTS memory_mentions (
            corpus_id TEXT NOT NULL, entity_id TEXT NOT NULL, passage_kind TEXT NOT NULL, passage_id TEXT NOT NULL,
            PRIMARY KEY (corpus_id, entity_id, passage_kind, passage_id))""",
        "CREATE INDEX IF NOT EXISTS ix_memory_mentions_passage ON memory_mentions (corpus_id, passage_kind, passage_id)",
    ]
    if dimension:
        statements += [
            "CREATE INDEX IF NOT EXISTS ix_memory_chunks_embedding ON memory_chunks USING hnsw (embedding vector_cosine_ops)",
            "CREATE INDEX IF NOT EXISTS ix_memory_stories_embedding ON memory_stories USING hnsw (embedding vector_cosine_ops)",
        ]
    return statements


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _vector(vector: list[float] | None) -> str | None:
    return None if vector is None else "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _source(r: Any) -> Source:
    return Source(corpus_id=r["corpus_id"], source_id=r["source_id"], name=r["name"], metadata=_json(r["metadata"]),
                  text=r["text"], created_at=r["created_at"], chunks=r["chunks"])


def _chunk(r: Any) -> Chunk:
    return Chunk(corpus_id=r["corpus_id"], chunk_id=r["chunk_id"], source_id=r["source_id"], position=r["position"],
                 title=r["title"], locator=_json(r["locator"]), text=r["text"])


def _story(r: Any) -> Story:
    return Story(corpus_id=r["corpus_id"], story_id=r["story_id"], text=r["text"], subject=r["subject"],
                 occurred_at=r["occurred_at"], created_at=r["created_at"], valid_to=r["valid_to"], supersedes=r["supersedes"],
                 source_ids=tuple(_json(r["source_ids"])), metadata=_json(r["metadata"]))


def _entity(r: Any) -> Entity:
    return Entity(corpus_id=r["corpus_id"], entity_id=r["entity_id"], name=r["name"], key=r["key"], type=r["type"],
                  aliases=tuple(_json(r["aliases"])))


def _relation(r: Any) -> Relation:
    return Relation(corpus_id=r["corpus_id"], from_id=r["from_id"], to_id=r["to_id"], type=r["type"],
                    evidence_kind=r["evidence_kind"], evidence_id=r["evidence_id"])


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
        path = key[len(FACET_PREFIX):]
        values = condition if isinstance(condition, list) else [condition]
        alternatives = []
        for value in values:
            for shape in (value, [value]):
                params.append(json.dumps(_nested(path, shape)))
                alternatives.append(f"{column} @> ${len(params)}::jsonb")
        clauses.append("(" + " OR ".join(alternatives) + ")")
    return " AND " + " AND ".join(clauses)


def _window_sql(window: Window, params: list[Any], alias: str = "") -> str:
    start = f"coalesce({alias}occurred_at, {alias}created_at)"
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


class PostgresStorage:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._dimension: int | None = None

    @asynccontextmanager
    async def _guard(self, operation: str) -> AsyncIterator[None]:
        try:
            yield
        except (StorageError, NotFoundError, ConfigurationError):
            raise
        except Exception as exc:
            raise StorageError(f"storage rejected {operation}: {exc}") from exc

    async def initialize(self, *, dimension: int | None) -> None:
        self._dimension = dimension
        async with self._pool.acquire() as conn:
            for statement in _EXTENSIONS:
                try:
                    await conn.execute(statement)
                except Exception as exc:
                    raise ConfigurationError(
                        f"{statement} failed: {exc}. Create the extension with a privileged role, then initialize again."
                    ) from exc
            async with self._guard("initialize"):
                for statement in _ddl(dimension):
                    await conn.execute(statement)

    async def put_source(
        self, corpus_id: str, source: Source, chunks: list[Chunk], vectors: list[list[float]] | None, model: str | None
    ) -> None:
        if source.corpus_id != corpus_id or any(c.corpus_id != corpus_id for c in chunks):
            raise StorageError("row corpus does not match the call corpus")
        async with self._guard("put_source"), self._pool.acquire() as conn, conn.transaction():
            await self._delete_source(conn, corpus_id, source.source_id)
            await conn.execute(
                "INSERT INTO memory_sources (corpus_id, source_id, name, metadata, text, created_at, chunks) "
                "VALUES ($1, $2, $3, $4::json, $5, $6, $7)",
                corpus_id, source.source_id, source.name, json.dumps(source.metadata), source.text, source.created_at, source.chunks,
            )
            rows = [
                (corpus_id, c.chunk_id, c.source_id, c.position, c.title, json.dumps(c.locator), c.text,
                 json.dumps(source.metadata), *([_vector(vectors[i] if vectors is not None else None)] if self._dimension else []), model)
                for i, c in enumerate(chunks)
            ]
            if rows:
                await conn.executemany(self._insert_chunk_sql(), rows)

    def _insert_chunk_sql(self) -> str:
        columns = "corpus_id, chunk_id, source_id, position, title, locator, text, metadata"
        if self._dimension:
            return (f"INSERT INTO memory_chunks ({columns}, embedding, embedding_model) "
                    "VALUES ($1, $2, $3, $4, $5, $6::json, $7, $8::jsonb, $9::vector, $10)")
        return (f"INSERT INTO memory_chunks ({columns}, embedding_model) "
                "VALUES ($1, $2, $3, $4, $5, $6::json, $7, $8::jsonb, $9)")

    async def get_source(self, corpus_id: str, source_id: str) -> Source | None:
        async with self._guard("get_source"), self._pool.acquire() as conn:
            record = await conn.fetchrow("SELECT * FROM memory_sources WHERE corpus_id = $1 AND source_id = $2", corpus_id, source_id)
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
                f"SELECT * FROM memory_sources WHERE corpus_id = $1{facets} ORDER BY created_at DESC, source_id COLLATE \"C\" DESC "
                f"LIMIT ${len(params) - 1} OFFSET ${len(params)}",
                *params,
            )
        rows = [_source(r) for r in records]
        return rows[:limit], (str(offset + limit) if len(rows) > limit else None)

    async def source_chunks(self, corpus_id: str, source_id: str) -> list[Chunk]:
        async with self._guard("source_chunks"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_chunks WHERE corpus_id = $1 AND source_id = $2 ORDER BY position", corpus_id, source_id
            )
        return [_chunk(r) for r in records]

    async def _delete_source(self, conn: Any, corpus_id: str, source_id: str) -> Deleted:
        chunk_ids = [r["chunk_id"] for r in await conn.fetch(
            "SELECT chunk_id FROM memory_chunks WHERE corpus_id = $1 AND source_id = $2", corpus_id, source_id)]
        mentions = relations = "DELETE 0"
        if chunk_ids:
            mentions = await conn.execute(
                "DELETE FROM memory_mentions WHERE corpus_id = $1 AND passage_kind = 'chunk' AND passage_id = ANY($2::text[])",
                corpus_id, chunk_ids)
            relations = await conn.execute(
                "DELETE FROM memory_relations WHERE corpus_id = $1 AND evidence_kind = 'chunk' AND evidence_id = ANY($2::text[])",
                corpus_id, chunk_ids)
        sources = await conn.execute("DELETE FROM memory_sources WHERE corpus_id = $1 AND source_id = $2", corpus_id, source_id)
        return Deleted(sources=_count(sources), chunks=len(chunk_ids), mentions=_count(mentions), relations=_count(relations))

    async def delete_source(self, corpus_id: str, source_id: str) -> Deleted:
        async with self._guard("delete_source"), self._pool.acquire() as conn, conn.transaction():
            return await self._delete_source(conn, corpus_id, source_id)

    def _insert_story_sql(self) -> str:
        columns = "corpus_id, story_id, text, text_hash, subject, occurred_at, created_at, valid_to, supersedes, source_ids, metadata"
        if self._dimension:
            return (f"INSERT INTO memory_stories ({columns}, embedding, embedding_model) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::json, $12::vector, $13)")
        return (f"INSERT INTO memory_stories ({columns}, embedding_model) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::json, $12)")

    def _story_params(self, story: Story, vector: list[float] | None, model: str | None) -> list[Any]:
        return [story.corpus_id, story.story_id, story.text, text_hash(story.text), story.subject, story.occurred_at,
                story.created_at, story.valid_to, story.supersedes, json.dumps(list(story.source_ids)),
                json.dumps(story.metadata), *([_vector(vector)] if self._dimension else []), model]

    async def put_story(self, corpus_id: str, story: Story, vector: list[float] | None, model: str | None) -> None:
        if story.corpus_id != corpus_id:
            raise StorageError("row corpus does not match the call corpus")
        async with self._guard("put_story"), self._pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM memory_stories WHERE corpus_id = $1 AND story_id = $2", corpus_id, story.story_id)
            await conn.execute(self._insert_story_sql(), *self._story_params(story, vector, model))

    async def find_open_story(self, corpus_id: str, hash: str) -> Story | None:
        async with self._guard("find_open_story"), self._pool.acquire() as conn:
            record = await conn.fetchrow(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND text_hash = $2 AND valid_to IS NULL", corpus_id, hash)
        return _story(record) if record else None

    async def supersede_story(
        self, corpus_id: str, old_id: str, valid_to: datetime, new: Story, vector: list[float] | None, model: str | None
    ) -> tuple[Story, Story]:
        async with self._guard("supersede_story"), self._pool.acquire() as conn, conn.transaction():
            current = await conn.fetchrow("SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = $2 FOR UPDATE", corpus_id, old_id)
            if current is None:
                raise NotFoundError(f"story {old_id!r}")
            if current["valid_to"] is not None:
                raise StorageError(f"story {old_id!r} is already closed")
            closed = await conn.fetchrow(
                "UPDATE memory_stories SET valid_to = $3 WHERE corpus_id = $1 AND story_id = $2 RETURNING *", corpus_id, old_id, valid_to)
            await conn.execute(self._insert_story_sql(), *self._story_params(new, vector, model))
        return _story(closed), new

    async def get_stories(self, corpus_id: str, story_ids: list[str]) -> list[Story]:
        async with self._guard("get_stories"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = ANY($2::text[])", corpus_id, list(story_ids))
        by_id = {r["story_id"]: _story(r) for r in records}
        return [by_id[i] for i in story_ids if i in by_id]

    async def list_stories(
        self, corpus_id: str, *, subject: str | None, window: Window, filters: dict[str, Any] | None, limit: int, cursor: str | None
    ) -> tuple[list[Story], str | None]:
        params: list[Any] = [corpus_id]
        sql = "SELECT * FROM memory_stories WHERE corpus_id = $1"
        if subject is not None:
            params.append(subject)
            sql += f" AND subject = ${len(params)}"
        sql += _window_sql(window, params) + _facet_sql(filters, params, "metadata::jsonb")
        offset = int(cursor) if cursor else 0
        params.extend([limit + 1, offset])
        sql += f" ORDER BY coalesce(occurred_at, created_at) DESC, story_id COLLATE \"C\" DESC LIMIT ${len(params) - 1} OFFSET ${len(params)}"
        async with self._guard("list_stories"), self._pool.acquire() as conn:
            records = await conn.fetch(sql, *params)
        rows = [_story(r) for r in records]
        return rows[:limit], (str(offset + limit) if len(rows) > limit else None)

    async def delete_story(self, corpus_id: str, story_id: str) -> Deleted:
        async with self._guard("delete_story"), self._pool.acquire() as conn, conn.transaction():
            mentions = await conn.execute(
                "DELETE FROM memory_mentions WHERE corpus_id = $1 AND passage_kind = 'story' AND passage_id = $2", corpus_id, story_id)
            relations = await conn.execute(
                "DELETE FROM memory_relations WHERE corpus_id = $1 AND evidence_kind = 'story' AND evidence_id = $2", corpus_id, story_id)
            stories = await conn.execute("DELETE FROM memory_stories WHERE corpus_id = $1 AND story_id = $2", corpus_id, story_id)
        return Deleted(stories=_count(stories), mentions=_count(mentions), relations=_count(relations))

    async def _search_kind(
        self, conn: Any, kind: Kind, score: str, condition: str, base: list[Any], filters: dict[str, Any] | None, window: Window, top_k: int
    ) -> list[Leg]:
        table, id_column = ("memory_chunks", "chunk_id") if kind == "chunk" else ("memory_stories", "story_id")
        params = list(base)
        extra = _facet_sql(filters, params, "metadata" if kind == "chunk" else "metadata::jsonb")
        if kind == "story":
            extra += _window_sql(window, params)
        params.append(top_k)
        records = await conn.fetch(
            f"SELECT {id_column} AS id, {score} AS score FROM {table} WHERE corpus_id = $1 AND {condition}{extra} "
            f'ORDER BY score DESC, {id_column} COLLATE "C" LIMIT ${len(params)}',
            *params,
        )
        return [(kind, r["id"], float(r["score"])) for r in records]

    async def _search(
        self, operation: str, kinds: tuple[Kind, ...], score: str, condition: str, base: list[Any],
        filters: dict[str, Any] | None, window: Window, top_k: int,
    ) -> list[Leg]:
        legs: list[Leg] = []
        async with self._guard(operation), self._pool.acquire() as conn:
            for kind in ("chunk", "story"):
                if kind in kinds:
                    legs.extend(await self._search_kind(conn, kind, score, condition, base, filters, window, top_k))
        legs.sort(key=lambda leg: (-leg[2], leg[1]))
        return legs[:top_k]

    async def search_text(
        self, corpus_id: str, query: str, *, kinds: tuple[Kind, ...], top_k: int, filters: dict[str, Any] | None, window: Window
    ) -> list[Leg]:
        return await self._search(
            "search_text", kinds, "ts_rank_cd(fts, websearch_to_tsquery('english', $2))",
            "fts @@ websearch_to_tsquery('english', $2)", [corpus_id, query], filters, window, top_k,
        )

    async def search_vector(
        self, corpus_id: str, vector: list[float], *, kinds: tuple[Kind, ...], top_k: int, filters: dict[str, Any] | None, window: Window
    ) -> list[Leg]:
        if not self._dimension:
            raise StorageError("this storage was initialized without a vector dimension")
        return await self._search(
            "search_vector", kinds, "1 - (embedding <=> $2::vector)", "embedding IS NOT NULL",
            [corpus_id, _vector(vector)], filters, window, top_k,
        )

    async def get_passages(self, corpus_id: str, refs: list[Ref]) -> list[Chunk | Story]:
        chunk_ids = [i for k, i in refs if k == "chunk"]
        story_ids = [i for k, i in refs if k == "story"]
        async with self._guard("get_passages"), self._pool.acquire() as conn:
            chunks = await conn.fetch("SELECT * FROM memory_chunks WHERE corpus_id = $1 AND chunk_id = ANY($2::text[])", corpus_id, chunk_ids)
            stories = await conn.fetch("SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = ANY($2::text[])", corpus_id, story_ids)
        found: dict[Ref, Chunk | Story] = {("chunk", r["chunk_id"]): _chunk(r) for r in chunks}
        found.update({("story", r["story_id"]): _story(r) for r in stories})
        return [found[ref] for ref in refs if ref in found]

    async def find_entity(self, corpus_id: str, key: str, type: str) -> Entity | None:
        async with self._guard("find_entity"), self._pool.acquire() as conn:
            record = await conn.fetchrow(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND key = $2 AND type = $3", corpus_id, key, type)
        return _entity(record) if record else None

    async def similar_entities(self, corpus_id: str, name: str, *, limit: int) -> list[Entity]:
        async with self._guard("similar_entities"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT *, greatest(similarity(name, $2), coalesce((SELECT max(similarity(a, $2)) "
                "FROM jsonb_array_elements_text(aliases) AS a), 0)) AS sim FROM memory_entities "
                "WHERE corpus_id = $1 ORDER BY sim DESC, entity_id COLLATE \"C\" LIMIT $3",
                corpus_id, name, limit)
        return [_entity(r) for r in records if float(r["sim"]) > 0.3]

    async def get_entity(self, corpus_id: str, name_or_key: str) -> Entity | None:
        key = " ".join(name_or_key.strip().lower().split())
        async with self._guard("get_entity"), self._pool.acquire() as conn:
            record = await conn.fetchrow(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND (key = $2 OR EXISTS "
                f"(SELECT 1 FROM jsonb_array_elements_text(aliases) AS a WHERE {_ALIAS_KEY} = $2)) ORDER BY entity_id COLLATE \"C\" LIMIT 1",
                corpus_id, key)
        return _entity(record) if record else None

    async def entities_by_id(self, corpus_id: str, entity_ids: list[str]) -> list[Entity]:
        async with self._guard("entities_by_id"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[])", corpus_id, list(entity_ids))
        by_id = {r["entity_id"]: _entity(r) for r in records}
        return [by_id[i] for i in entity_ids if i in by_id]

    async def put_entity(self, corpus_id: str, entity: Entity) -> None:
        if entity.corpus_id != corpus_id:
            raise StorageError("row corpus does not match the call corpus")
        async with self._guard("put_entity"), self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases) VALUES ($1, $2, $3, $4, $5, $6::jsonb) "
                "ON CONFLICT (corpus_id, entity_id) DO UPDATE SET key = EXCLUDED.key, name = EXCLUDED.name, type = EXCLUDED.type, aliases = EXCLUDED.aliases",
                corpus_id, entity.entity_id, entity.key, entity.name, entity.type, json.dumps(list(entity.aliases)))

    async def add_alias(self, corpus_id: str, entity_id: str, alias: str) -> None:
        async with self._guard("add_alias"), self._pool.acquire() as conn:
            status = await conn.execute(
                "UPDATE memory_entities SET aliases = CASE WHEN aliases ? $3 THEN aliases ELSE aliases || to_jsonb($3::text) END "
                "WHERE corpus_id = $1 AND entity_id = $2",
                corpus_id, entity_id, alias)
        if _count(status) == 0:
            raise NotFoundError(f"entity {entity_id!r}")

    async def put_relations(self, corpus_id: str, relations: list[Relation]) -> None:
        if any(r.corpus_id != corpus_id for r in relations):
            raise StorageError("row corpus does not match the call corpus")
        async with self._guard("put_relations"), self._pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO memory_relations (corpus_id, from_id, to_id, type, evidence_kind, evidence_id) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING",
                [(corpus_id, r.from_id, r.to_id, r.type, r.evidence_kind, r.evidence_id) for r in relations])

    async def put_mentions(self, corpus_id: str, entity_ids: list[str], passage_kind: Kind, passage_id: str) -> None:
        async with self._guard("put_mentions"), self._pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO memory_mentions (corpus_id, entity_id, passage_kind, passage_id) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                [(corpus_id, e, passage_kind, passage_id) for e in entity_ids])

    async def entities_named_in(self, corpus_id: str, text: str) -> list[Entity]:
        haystack = " ".join(text.strip().lower().split())
        async with self._guard("entities_named_in"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND (position(key IN $2) > 0 OR EXISTS "
                f"(SELECT 1 FROM jsonb_array_elements_text(aliases) AS a WHERE position({_ALIAS_KEY} IN $2) > 0)) ORDER BY entity_id COLLATE \"C\"",
                corpus_id, haystack)
        return [_entity(r) for r in records]

    async def neighbors(self, corpus_id: str, entity_ids: list[str], *, relation_types: list[str] | None) -> list[Relation]:
        params: list[Any] = [corpus_id, list(entity_ids)]
        sql = "SELECT * FROM memory_relations WHERE corpus_id = $1 AND (from_id = ANY($2::text[]) OR to_id = ANY($2::text[]))"
        if relation_types is not None:
            params.append(list(relation_types))
            sql += " AND type = ANY($3::text[])"
        async with self._guard("neighbors"), self._pool.acquire() as conn:
            records = await conn.fetch(sql + ' ORDER BY from_id COLLATE "C", to_id COLLATE "C", type COLLATE "C", evidence_kind COLLATE "C", evidence_id COLLATE "C"', *params)
        return [_relation(r) for r in records]

    async def mentions(
        self, corpus_id: str, entity_ids: list[str], *, kinds: tuple[Kind, ...], filters: dict[str, Any] | None, window: Window
    ) -> list[Ref]:
        refs: list[Ref] = []
        async with self._guard("mentions"), self._pool.acquire() as conn:
            if "chunk" in kinds:
                params: list[Any] = [corpus_id, list(entity_ids)]
                facets = _facet_sql(filters, params, "c.metadata")
                records = await conn.fetch(
                    'SELECT DISTINCT c.chunk_id COLLATE "C" AS chunk_id FROM memory_mentions m JOIN memory_chunks c ON c.corpus_id = m.corpus_id AND c.chunk_id = m.passage_id '
                    f"WHERE m.corpus_id = $1 AND m.passage_kind = 'chunk' AND m.entity_id = ANY($2::text[]){facets} ORDER BY chunk_id",
                    *params)
                refs.extend(("chunk", r["chunk_id"]) for r in records)
            if "story" in kinds:
                params = [corpus_id, list(entity_ids)]
                facets = _facet_sql(filters, params, "s.metadata::jsonb") + _window_sql(window, params, alias="s.")
                records = await conn.fetch(
                    'SELECT DISTINCT s.story_id COLLATE "C" AS story_id FROM memory_mentions m JOIN memory_stories s ON s.corpus_id = m.corpus_id AND s.story_id = m.passage_id '
                    f"WHERE m.corpus_id = $1 AND m.passage_kind = 'story' AND m.entity_id = ANY($2::text[]){facets} ORDER BY story_id",
                    *params)
                refs.extend(("story", r["story_id"]) for r in records)
        return refs

    async def passages_embedded_with(self, corpus_id: str, *, not_model: str, limit: int) -> list[Chunk | Story]:
        async with self._guard("passages_embedded_with"), self._pool.acquire() as conn:
            chunks = await conn.fetch(
                "SELECT * FROM memory_chunks WHERE corpus_id = $1 AND embedding_model IS DISTINCT FROM $2 ORDER BY chunk_id COLLATE \"C\" LIMIT $3",
                corpus_id, not_model, limit)
            stories = await conn.fetch(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND embedding_model IS DISTINCT FROM $2 ORDER BY story_id COLLATE \"C\" LIMIT $3",
                corpus_id, not_model, max(limit - len(chunks), 0))
        return [*(_chunk(r) for r in chunks), *(_story(r) for r in stories)]

    async def set_vectors(self, corpus_id: str, refs: list[Ref], vectors: list[list[float]], model: str) -> None:
        if not self._dimension:
            raise StorageError("this storage was initialized without a vector dimension")
        chunk_rows = [(corpus_id, i, _vector(v), model) for (k, i), v in zip(refs, vectors, strict=True) if k == "chunk"]
        story_rows = [(corpus_id, i, _vector(v), model) for (k, i), v in zip(refs, vectors, strict=True) if k == "story"]
        async with self._guard("set_vectors"), self._pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                "UPDATE memory_chunks SET embedding = $3::vector, embedding_model = $4 WHERE corpus_id = $1 AND chunk_id = $2", chunk_rows)
            await conn.executemany(
                "UPDATE memory_stories SET embedding = $3::vector, embedding_model = $4 WHERE corpus_id = $1 AND story_id = $2", story_rows)

    async def delete_corpus(self, corpus_id: str) -> Deleted:
        async with self._guard("delete_corpus"), self._pool.acquire() as conn, conn.transaction():
            counts = {}
            for field, table in (("mentions", "memory_mentions"), ("relations", "memory_relations"), ("entities", "memory_entities"),
                                 ("stories", "memory_stories"), ("chunks", "memory_chunks"), ("sources", "memory_sources")):
                counts[field] = _count(await conn.execute(f"DELETE FROM {table} WHERE corpus_id = $1", corpus_id))
        return Deleted(**counts)

    async def count_corpus(self, corpus_id: str) -> Residue:
        async with self._guard("count_corpus"), self._pool.acquire() as conn:
            counts = {}
            for field, table in (("sources", "memory_sources"), ("chunks", "memory_chunks"), ("stories", "memory_stories"),
                                 ("entities", "memory_entities"), ("relations", "memory_relations"), ("mentions", "memory_mentions")):
                counts[field] = int(await conn.fetchval(f"SELECT count(*) FROM {table} WHERE corpus_id = $1", corpus_id))
        return Residue(**counts)
