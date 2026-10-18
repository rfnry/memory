import json
from datetime import datetime
from typing import Any

from rfnry_memory.errors import StorageError
from rfnry_memory.models import (
    Chunk,
    Counts,
    Source,
)
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.graph import _drop_evidence, _reconcile, _write_graph
from rfnry_memory.postgres.language import _CONFIG, _pin
from rfnry_memory.postgres.locks import _language_lock, _sync_lock
from rfnry_memory.postgres.records import _chunk, _count, _json, _source, _vector
from rfnry_memory.postgres.sql import _facet_sql
from rfnry_memory.postgres.vectors import _copy_vectors, _delete_vectors, _vectors_table
from rfnry_memory.rows import (
    GraphWrite,
    SourceWrite,
    chunk_digest,
    successors,
    total,
)


class _Moved(StorageError):
    pass


_VERSIONS_SQL = (
    "WITH RECURSIVE chain(source_id) AS ("
    "SELECT source_id FROM memory_sources WHERE corpus_id = $1 AND source_id = $2 "
    "UNION SELECT s.source_id FROM memory_sources s JOIN chain c ON s.replaced_by = c.source_id "
    'WHERE s.corpus_id = $1) SELECT source_id FROM chain ORDER BY source_id COLLATE "C"'
)


def _sources_sql(
    known_at: datetime | None, history: bool, params: list[Any], alias: str = ""
) -> str:
    sql = ""
    if known_at is not None:
        params.append(known_at)
        learned = f"${len(params)}"
        sql += f" AND {alias}created_at <= {learned}"
        if not history:
            sql += f" AND ({alias}retired_at IS NULL OR {alias}retired_at > {learned})"
    elif not history:
        sql += f" AND {alias}retired_at IS NULL"
    return sql


class Sources(Base):
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
            await _pin(conn, corpus_id)
            table = await _vectors_table(conn, corpus_id)
            touched: set[str] = set()
            await self._delete_source(conn, corpus_id, source.source_id, touched, table)
            await self._insert_source(
                conn, corpus_id, SourceWrite(source, chunks, vectors), model, table
            )
            await _write_graph(conn, corpus_id, graph, touched)
            await _reconcile(conn, corpus_id, touched)

    async def sync_sources(
        self,
        corpus_id: str,
        filters: dict[str, Any],
        writes: list[SourceWrite],
        model: str | None,
        *,
        at: datetime,
        graph: GraphWrite | None = None,
        keep: dict[str, str] | None = None,
    ) -> tuple[Counts, list[str]] | None:
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
                await _sync_lock(conn, corpus_id)
                await _language_lock(conn, corpus_id, shared=True)
                await _pin(conn, corpus_id)
                table = await _vectors_table(conn, corpus_id)
                matching = {
                    r["source_id"]: (r["digest"], r["name"], r["created_at"])
                    for r in await conn.fetch(
                        f"SELECT source_id, digest, name, created_at FROM memory_sources WHERE corpus_id = $1 AND retired_at IS NULL{facets} "
                        'ORDER BY source_id COLLATE "C" FOR UPDATE',
                        *params,
                    )
                }
                history = await conn.fetchval(
                    "SELECT history FROM memory_corpora WHERE corpus_id = $1", corpus_id
                )
                if any(
                    source_id not in matching or matching[source_id][0] != digest
                    for source_id, digest in keep.items()
                ) or (history and any(created > at for _, _, created in matching.values())):
                    raise _Moved
                dropped = [source_id for source_id in matching if source_id not in keep]
                touched: set[str] = set()
                retired: list[str] = []
                if history:
                    later = successors(writes)
                    await self._retire(
                        conn,
                        corpus_id,
                        dropped,
                        at,
                        [later.get(matching[source_id][1]) for source_id in dropped],
                    )
                    deleted, retired = Counts(), dropped
                else:
                    deleted = total(
                        [
                            await self._delete_source(conn, corpus_id, source_id, touched, table)
                            for source_id in dropped
                        ]
                    )
                for write in writes:
                    await self._insert_source(conn, corpus_id, write, model, table)
                await _write_graph(conn, corpus_id, graph, touched)
                pruned = await _reconcile(conn, corpus_id, touched)
        except _Moved:
            return None
        return deleted.model_copy(update={"entities": pruned}), retired

    async def _retire(
        self,
        conn: Any,
        corpus_id: str,
        source_ids: list[str],
        at: datetime,
        replaced_by: list[str | None],
    ) -> None:
        if not source_ids:
            return
        await conn.execute(
            "UPDATE memory_sources s SET retired_at = $3, replaced_by = r.replaced_by "
            "FROM unnest($2::text[], $4::text[]) AS r(source_id, replaced_by) "
            "WHERE s.corpus_id = $1 AND s.source_id = r.source_id",
            corpus_id,
            source_ids,
            at,
            replaced_by,
        )
        await conn.execute(
            "UPDATE memory_chunks SET retired_at = $3 WHERE corpus_id = $1 AND source_id = ANY($2::text[])",
            corpus_id,
            source_ids,
            at,
        )

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
        self, conn: Any, corpus_id: str, write: SourceWrite, model: str | None, table: str | None
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
                c.context,
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
        await _copy_vectors(conn, table, corpus_id, "chunk", [c.chunk_id for c in chunks])

    def _insert_chunk_sql(self) -> str:
        columns = "corpus_id, chunk_id, source_id, source_name, position, title, context, locator, text, metadata, digest, config"
        if self._dimension:
            return (
                f"INSERT INTO memory_chunks ({columns}, embedding, embedding_model) "
                f"VALUES ($1, $2, $3, $4, $5, $6, $7, $8::json, $9, $10::jsonb, $11, {_CONFIG}, $12::vector, $13)"
            )
        return (
            f"INSERT INTO memory_chunks ({columns}, embedding_model) "
            f"VALUES ($1, $2, $3, $4, $5, $6, $7, $8::json, $9, $10::jsonb, $11, {_CONFIG}, $12)"
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
        self,
        corpus_id: str,
        *,
        filters: dict[str, Any] | None,
        limit: int,
        cursor: str | None,
        known_at: datetime | None = None,
        history: bool = False,
    ) -> tuple[list[Source], str | None]:
        params: list[Any] = [corpus_id]
        facets = _facet_sql(filters, params, "metadata::jsonb") + _sources_sql(
            known_at, history, params
        )
        offset = int(cursor) if cursor else 0
        params.extend([limit + 1, offset])
        async with self._guard("list_sources"), self._pool.acquire() as conn:
            records = await conn.fetch(
                f'SELECT * FROM memory_sources WHERE corpus_id = $1{facets} ORDER BY created_at DESC, source_id COLLATE "C" DESC '
                f"LIMIT ${len(params) - 1} OFFSET ${len(params)}",
                *params,
            )
        rows = [_source(r, known_at) for r in records]
        return rows[:limit], (str(offset + limit) if len(rows) > limit else None)

    async def source_chunks(self, corpus_id: str, source_id: str) -> list[Chunk]:
        async with self._guard("source_chunks"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_chunks WHERE corpus_id = $1 AND source_id = $2 ORDER BY position",
                corpus_id,
                source_id,
            )
        return [_chunk(r) for r in records]

    async def count_retired(self, corpus_id: str) -> int:
        async with self._guard("count_retired"), self._pool.acquire() as conn:
            return int(
                await conn.fetchval(
                    "SELECT count(*) FROM memory_sources WHERE corpus_id = $1 AND retired_at IS NOT NULL",
                    corpus_id,
                )
            )

    async def source_digests(self, corpus_id: str) -> list[tuple[str, dict[str, Any], str, int]]:
        async with self._guard("source_digests"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT name, metadata, encode(sha256(convert_to(text, 'UTF8')), 'hex') AS text_sha256, chunks "
                "FROM memory_sources WHERE corpus_id = $1 AND retired_at IS NULL",
                corpus_id,
            )
        return [(r["name"], _json(r["metadata"]), r["text_sha256"], r["chunks"]) for r in records]

    async def _delete_source(
        self, conn: Any, corpus_id: str, source_id: str, touched: set[str], table: str | None
    ) -> Counts:
        chunk_ids = [
            r["chunk_id"]
            for r in await conn.fetch(
                "SELECT chunk_id FROM memory_chunks WHERE corpus_id = $1 AND source_id = $2",
                corpus_id,
                source_id,
            )
        ]
        mentions, relations = await _drop_evidence(conn, corpus_id, "chunk", chunk_ids, touched)
        vectors = await _delete_vectors(conn, table, "chunk", chunk_ids)
        sources = await conn.execute(
            "DELETE FROM memory_sources WHERE corpus_id = $1 AND source_id = $2",
            corpus_id,
            source_id,
        )
        return Counts(
            sources=_count(sources),
            chunks=len(chunk_ids),
            mentions=mentions,
            relations=relations,
            vectors=vectors,
        )

    async def delete_source(self, corpus_id: str, source_id: str) -> Counts:
        async with (
            self._guard("delete_source"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _sync_lock(conn, corpus_id)
            await _language_lock(conn, corpus_id, shared=True)
            table = await _vectors_table(conn, corpus_id)
            versions = [
                r["source_id"] for r in await conn.fetch(_VERSIONS_SQL, corpus_id, source_id)
            ]
            touched: set[str] = set()
            deleted = total(
                [
                    await self._delete_source(conn, corpus_id, version, touched, table)
                    for version in versions
                ]
            )
            pruned = await _reconcile(conn, corpus_id, touched)
        return deleted.model_copy(update={"entities": pruned})

    async def forget_retired(self, corpus_id: str, *, before: datetime) -> Counts:
        async with (
            self._guard("forget_retired"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _sync_lock(conn, corpus_id)
            await _language_lock(conn, corpus_id, shared=True)
            table = await _vectors_table(conn, corpus_id)
            gone = [
                r["source_id"]
                for r in await conn.fetch(
                    "SELECT source_id FROM memory_sources WHERE corpus_id = $1 AND retired_at < $2 "
                    'ORDER BY source_id COLLATE "C" FOR UPDATE',
                    corpus_id,
                    before,
                )
            ]
            touched: set[str] = set()
            deleted = total(
                [
                    await self._delete_source(conn, corpus_id, source_id, touched, table)
                    for source_id in gone
                ]
            )
            pruned = await _reconcile(conn, corpus_id, touched)
        return deleted.model_copy(update={"entities": pruned})
