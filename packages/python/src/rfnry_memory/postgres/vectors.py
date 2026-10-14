from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any, Literal, NamedTuple

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.index import hnsw
from rfnry_memory.models import (
    Chunk,
    Kind,
    Ref,
    Story,
    VectorIndex,
)
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.locks import _language_lock
from rfnry_memory.postgres.records import _chunk, _count, _story, _vector
from rfnry_memory.postgres.sql import _table

_FULL_DIMENSIONS = 2000


_HALF_DIMENSIONS = 4000


_VECTORS_TABLE = (
    "(SELECT format('%I.%I', schemaname, tablename) FROM pg_tables "
    "WHERE schemaname = current_schema() AND tablename = $1::text)"
)


_INDEXED = (
    "SELECT format('%I.%I', n.nspname, t.relname) AS vectors, split_part(c.relname, '_', 5) AS kind, "
    "i.indexprs IS NOT NULL AS half FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
    "JOIN pg_class t ON t.oid = i.indrelid JOIN pg_namespace n ON n.oid = t.relnamespace "
    "WHERE n.nspname = current_schema() AND t.relname = $1::text AND i.indisvalid "
    "AND starts_with(c.relname, 'ix_') ORDER BY c.oid"
)


_VECTOR_INDEXES = (
    "SELECT c.relname AS name, i.indisvalid AS valid, i.indexprs IS NOT NULL AS half, "
    "coalesce(c.reloptions, '{}') AS options FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
    "WHERE i.indrelid = to_regclass(format('%I.%I', current_schema(), $1::text)) ORDER BY c.oid DESC"
)


def _vectors_name(corpus_id: str) -> str:
    return f"memory_vectors_{sha256(corpus_id.encode('utf-8')).hexdigest()[:24]}"


async def _vectors_table(conn: Any, corpus_id: str) -> str | None:
    table = await conn.fetchval(f"SELECT {_VECTORS_TABLE}", _vectors_name(corpus_id))
    return None if table is None else str(table)


async def _copy_vectors(
    conn: Any, table: str | None, corpus_id: str, kind: Kind, ids: list[str]
) -> None:
    if table is None or not ids:
        return
    passages, id_column = _table(kind)
    await conn.execute(
        f"INSERT INTO {table} (passage_kind, passage_id, vector) SELECT $2::text, {id_column}, embedding "
        f"FROM {passages} WHERE corpus_id = $1 AND {id_column} = ANY($3::text[]) AND embedding IS NOT NULL "
        "ON CONFLICT (passage_kind, passage_id) DO UPDATE SET vector = EXCLUDED.vector",
        corpus_id,
        kind,
        ids,
    )


async def _delete_vectors(conn: Any, table: str | None, kind: Kind, ids: list[str]) -> int:
    if table is None or not ids:
        return 0
    return _count(
        await conn.execute(
            f"DELETE FROM {table} WHERE passage_kind = $1 AND passage_id = ANY($2::text[])",
            kind,
            ids,
        )
    )


@asynccontextmanager
async def _vectors_lock(conn: Any, corpus_id: str) -> AsyncIterator[None]:
    key = f"memory.vectors:{corpus_id}"
    while not await conn.fetchval("SELECT pg_try_advisory_lock(hashtextextended($1, 0))", key):
        await asyncio.sleep(0.1)
    try:
        yield
    finally:
        await conn.execute("SELECT pg_advisory_unlock(hashtextextended($1, 0))", key)


async def _pgvector(conn: Any) -> str:
    return str(await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))


def _precision(
    dimension: int, precision: Literal["full", "half"] | None
) -> Literal["full", "half"]:
    if dimension > _HALF_DIMENSIONS:
        raise ConfigurationError(
            f"postgres: a vector index reaches {_HALF_DIMENSIONS} dimensions at half precision; this memory embeds {dimension}"
        )
    if precision == "full" and dimension > _FULL_DIMENSIONS:
        raise InputError(
            f"postgres: full precision indexes up to {_FULL_DIMENSIONS} dimensions; this memory embeds {dimension} — pass precision='half'"
        )
    if precision is None:
        return "full" if dimension <= _FULL_DIMENSIONS else "half"
    return precision


class _Index(NamedTuple):
    name: str
    kind: str
    settings: VectorIndex
    valid: bool


async def _vector_indexes(conn: Any, name: str) -> list[_Index]:
    prefix = f"ix_{name}_"
    found: list[_Index] = []
    for r in await conn.fetch(_VECTOR_INDEXES, name):
        kind = r["name"].removeprefix(prefix).partition("_")[0]
        if not r["name"].startswith(prefix) or kind not in ("chunk", "story"):
            continue
        found.append(_Index(r["name"], kind, hnsw.read(r["options"], r["half"]), r["valid"]))
    return found


def _vectors_ddl(table: str, dimension: int) -> str:
    return (
        f"CREATE TABLE {table} (passage_kind TEXT NOT NULL, passage_id TEXT NOT NULL, "
        f"vector vector({dimension}) NOT NULL, PRIMARY KEY (passage_kind, passage_id))"
    )


async def _check_index(conn: Any, table: str, dimension: int, clause: str) -> None:
    check = conn.transaction()
    await check.start()
    try:
        await conn.execute(_vectors_ddl(table, dimension))
        await conn.execute(f"CREATE INDEX ON {table} {clause}")
    except Exception as exc:
        if getattr(exc, "sqlstate", None) != "22023":
            raise
        detail = getattr(exc, "detail", None)
        reason = getattr(exc, "message", str(exc)) + (f" — {detail}" if detail else "")
        raise InputError(f"postgres: {reason}") from exc
    finally:
        await check.rollback()


class Vectors(Base):
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
        async with (
            self._guard("set_vectors"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=True)
            table = await _vectors_table(conn, corpus_id)
            await conn.executemany(
                "UPDATE memory_chunks SET embedding = $3::vector, embedding_model = $4 WHERE corpus_id = $1 AND chunk_id = $2",
                chunk_rows,
            )
            await conn.executemany(
                "UPDATE memory_stories SET embedding = $3::vector, embedding_model = $4 WHERE corpus_id = $1 AND story_id = $2",
                story_rows,
            )
            await _copy_vectors(
                conn, table, corpus_id, "chunk", [i for k, i in refs if k == "chunk"]
            )
            await _copy_vectors(
                conn, table, corpus_id, "story", [i for k, i in refs if k == "story"]
            )

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

    async def get_vector_index(self, corpus_id: str) -> VectorIndex | None:
        async with self._guard("get_vector_index"), self._pool.acquire() as conn:
            indexes = await _vector_indexes(conn, _vectors_name(corpus_id))
        built = [i for i in indexes if i.valid]
        stories = [i.settings for i in built if i.kind == "story"]
        return next(
            (i.settings for i in built if i.kind == "chunk" and i.settings in stories), None
        )

    async def set_vector_index(
        self,
        corpus_id: str,
        on: bool,
        *,
        m: int,
        ef_construction: int,
        precision: Literal["full", "half"] | None,
    ) -> int:
        async with self._guard("set_vector_index"):
            if not on:
                await self._drop_vectors(corpus_id)
                return 0
            dimension = self._dimension
            if not dimension:
                raise ConfigurationError(
                    "postgres: a vector index needs the embedding column — run rfnry-memory set --dimension, then check() again"
                )
            settings = VectorIndex(
                m=m, ef_construction=ef_construction, precision=_precision(dimension, precision)
            )
            async with self._pool.acquire() as conn:
                version = await _pgvector(conn)
                if tuple(int(part) for part in version.split(".")[:2]) < (0, 8):
                    raise ConfigurationError(
                        f"postgres: a vector index needs pgvector 0.8.0 or later; this database has {version}"
                    )
                async with _vectors_lock(conn, corpus_id):
                    return await self._index_vectors(conn, corpus_id, settings, dimension)

    async def _drop_vectors(self, corpus_id: str) -> None:
        async with (
            self._pool.acquire() as conn,
            _vectors_lock(conn, corpus_id),
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=False)
            table = await _vectors_table(conn, corpus_id)
            if table is not None:
                await conn.execute(f"DROP TABLE {table}")

    async def _index_vectors(
        self, conn: Any, corpus_id: str, settings: VectorIndex, dimension: int
    ) -> int:
        name = _vectors_name(corpus_id)
        schema = await conn.fetchval("SELECT quote_ident(current_schema())")
        table = f"{schema}.{name}"
        await _check_index(
            conn, f"{table}_check", dimension, hnsw.clause("chunk", settings, dimension)
        )
        if await _vectors_table(conn, corpus_id) is None:
            async with conn.transaction(isolation="read_committed"):
                await _language_lock(conn, corpus_id, shared=False)
                await conn.execute(_vectors_ddl(table, dimension))
                await conn.execute(
                    f"INSERT INTO {table} (passage_kind, passage_id, vector) "
                    "SELECT 'chunk', chunk_id, embedding FROM memory_chunks WHERE corpus_id = $1 AND embedding IS NOT NULL "
                    "UNION ALL SELECT 'story', story_id, embedding FROM memory_stories WHERE corpus_id = $1 AND embedding IS NOT NULL",
                    corpus_id,
                )
            await conn.execute(f"ANALYZE {table}")
        indexes = await _vector_indexes(conn, name)
        for index in indexes:
            if not index.valid:
                await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}.{index.name}")
        tuning = f"{settings.m}:{settings.ef_construction}:{settings.precision}"
        tuned = sha256(tuning.encode("utf-8")).hexdigest()[:8]
        for kind in ("chunk", "story"):
            wanted = f"ix_{name}_{kind}_{tuned}"
            if not any(index.valid and index.name == wanted for index in indexes):
                await conn.execute(
                    f"CREATE INDEX CONCURRENTLY {wanted} ON {table} {hnsw.clause(kind, settings, dimension)}"
                )
            for index in indexes:
                if index.valid and index.kind == kind and index.name != wanted:
                    await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}.{index.name}")
        return int(await conn.fetchval(f"SELECT count(*) FROM {table}"))
