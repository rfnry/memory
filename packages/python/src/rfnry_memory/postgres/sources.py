import json
from datetime import datetime
from typing import Any

from rfnry_memory.errors import InputError, NotFoundError, StorageError
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
from rfnry_memory.postgres.sql import _CHUNKS, _facet_sql, _in_force_sql
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


def _started(write: SourceWrite, at: datetime) -> SourceWrite:
    if write.source.valid_from is not None:
        return write
    start = {"valid_from": at}
    return SourceWrite(
        write.source.model_copy(update=start),
        [c.model_copy(update=start) for c in write.chunks],
        write.vectors,
    )


_VERSIONS_SQL = (
    "WITH RECURSIVE chain(source_id) AS ("
    "SELECT source_id FROM memory_sources WHERE corpus_id = $1 AND source_id = $2 "
    "UNION SELECT s.source_id FROM memory_sources s JOIN chain c ON s.replaced_by = c.source_id "
    'WHERE s.corpus_id = $1) SELECT source_id FROM chain ORDER BY source_id COLLATE "C"'
)


def _sources_sql(
    known_at: datetime | None,
    history: bool,
    params: list[Any],
    alias: str = "",
    as_of: datetime | None = None,
) -> str:
    sql = ""
    if known_at is not None:
        params.append(known_at)
        learned = f"${len(params)}"
        sql += f" AND {alias}created_at <= {learned}"
        if not history and as_of is None:
            sql += f" AND ({alias}retired_at IS NULL OR {alias}retired_at > {learned})"
    elif not history and as_of is None:
        sql += f" AND {alias}retired_at IS NULL"
    if history:
        return sql
    return sql + _in_force_sql(as_of, known_at, params, alias)


def _closed_end(
    name: str, valid_from: datetime | None, valid_to: datetime | None, at: datetime, start: datetime
) -> datetime:
    if valid_from is not None and valid_from > at:
        return valid_from
    if valid_from is not None and start < valid_from:
        raise InputError(
            f"postgres: source {name!r} takes effect at {start.isoformat()}, before the version it replaces took "
            f"effect, at {valid_from.isoformat()}; that moment itself corrects it from the start"
        )
    if (valid_to is None or valid_to > at) and start > at:
        raise InputError(
            f"postgres: source {name!r} takes effect at {start.isoformat()}, after this write; a version in force "
            "is replaced on the day its update takes effect: write the update then"
        )
    end = start if valid_to is None else min(valid_to, start)
    return end if valid_from is None else max(valid_from, end)


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
    ) -> tuple[Counts, list[str], list[Source]] | None:
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
                    r["source_id"]: r
                    for r in await conn.fetch(
                        "SELECT source_id, digest, name, created_at, valid_from, valid_to FROM memory_sources "
                        f"WHERE corpus_id = $1 AND retired_at IS NULL{facets} "
                        'ORDER BY source_id COLLATE "C" FOR UPDATE',
                        *params,
                    )
                }
                history = await conn.fetchval(
                    "SELECT history FROM memory_corpora WHERE corpus_id = $1", corpus_id
                )
                if any(
                    source_id not in matching or matching[source_id]["digest"] != digest
                    for source_id, digest in keep.items()
                ) or (history and any(r["created_at"] > at for r in matching.values())):
                    raise _Moved
                dropped = [source_id for source_id in matching if source_id not in keep]
                if keep:
                    await conn.execute(
                        "UPDATE memory_sources SET confirmed_at = $3 WHERE corpus_id = $1 AND source_id = ANY($2::text[])",
                        corpus_id,
                        list(keep),
                        at,
                    )
                later = successors(writes)
                starts = {w.source.source_id: w.source.valid_from or at for w in writes}
                ends = [
                    _closed_end(
                        matching[source_id]["name"],
                        matching[source_id]["valid_from"],
                        matching[source_id]["valid_to"],
                        at,
                        starts.get(later.get(matching[source_id]["name"], ""), at),
                    )
                    for source_id in dropped
                ]
                touched: set[str] = set()
                retired: list[str] = []
                if history:
                    followed = {
                        later[matching[i]["name"]] for i in dropped if matching[i]["name"] in later
                    }
                    writes = [
                        _started(w, at) if w.source.source_id in followed else w for w in writes
                    ]
                    await self._retire(
                        conn,
                        corpus_id,
                        dropped,
                        at,
                        [later.get(matching[source_id]["name"]) for source_id in dropped],
                        ends,
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
        return deleted.model_copy(update={"entities": pruned}), retired, [w.source for w in writes]

    async def replace_source(
        self,
        corpus_id: str,
        source_id: str,
        write: SourceWrite,
        model: str | None,
        *,
        at: datetime,
        graph: GraphWrite | None = None,
    ) -> tuple[Counts, list[str], Source] | None:
        source, chunks, vectors = write
        if source.corpus_id != corpus_id or any(c.corpus_id != corpus_id for c in chunks):
            raise StorageError("postgres: row corpus does not match the call corpus")
        self._check_vectors(vectors)
        graph = graph or GraphWrite()
        graph.check(corpus_id, {("chunk", c.chunk_id) for c in chunks})
        try:
            async with (
                self._guard("replace_source"),
                self._pool.acquire() as conn,
                conn.transaction(isolation="read_committed"),
            ):
                await _sync_lock(conn, corpus_id)
                await _language_lock(conn, corpus_id, shared=True)
                await _pin(conn, corpus_id)
                table = await _vectors_table(conn, corpus_id)
                old = await conn.fetchrow(
                    "SELECT * FROM memory_sources WHERE corpus_id = $1 AND source_id = $2 FOR UPDATE",
                    corpus_id,
                    source_id,
                )
                if old is None:
                    raise NotFoundError(f"postgres: source {source_id!r} was not found")
                if old["replaced_by"] is not None:
                    raise InputError(
                        f"postgres: source {source_id!r} was already replaced by {old['replaced_by']!r}; replace that one"
                    )
                if old["retired_at"] is not None:
                    raise InputError(
                        f"postgres: source {source_id!r} was retired at {old['retired_at'].isoformat()}; ingest it anew instead"
                    )
                history = await conn.fetchval(
                    "SELECT history FROM memory_corpora WHERE corpus_id = $1", corpus_id
                )
                if history and old["created_at"] > at:
                    raise _Moved
                end = _closed_end(
                    source.name, old["valid_from"], old["valid_to"], at, source.valid_from or at
                )
                touched: set[str] = set()
                retired: list[str] = []
                if history:
                    write = _started(write, at)
                    await self._retire(conn, corpus_id, [source_id], at, [source.source_id], [end])
                    deleted, retired = Counts(), [source_id]
                else:
                    deleted = await self._delete_source(conn, corpus_id, source_id, touched, table)
                await self._insert_source(conn, corpus_id, write, model, table)
                await _write_graph(conn, corpus_id, graph, touched)
                pruned = await _reconcile(conn, corpus_id, touched)
        except _Moved:
            return None
        return deleted.model_copy(update={"entities": pruned}), retired, write.source

    async def _retire(
        self,
        conn: Any,
        corpus_id: str,
        source_ids: list[str],
        at: datetime,
        replaced_by: list[str | None],
        ends: list[datetime],
    ) -> None:
        if not source_ids:
            return
        await conn.execute(
            "UPDATE memory_sources s SET retired_at = $3, replaced_by = r.replaced_by, valid_to = r.ends "
            "FROM unnest($2::text[], $4::text[], $5::timestamptz[]) AS r(source_id, replaced_by, ends) "
            "WHERE s.corpus_id = $1 AND s.source_id = r.source_id",
            corpus_id,
            source_ids,
            at,
            replaced_by,
            ends,
        )
        await conn.execute(
            "UPDATE memory_chunks c SET retired_at = $3, valid_to = s.valid_to FROM memory_sources s "
            "WHERE c.corpus_id = $1 AND c.source_id = ANY($2::text[]) "
            "AND s.corpus_id = c.corpus_id AND s.source_id = c.source_id",
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
            "INSERT INTO memory_sources (corpus_id, source_id, name, metadata, text, created_at, chunks, digest, "
            "confirmed_at, valid_from, valid_to, written_to) "
            "VALUES ($1, $2, $3, $4::json, $5, $6, $7, $8, $9, $10, $11, $11)",
            corpus_id,
            source.source_id,
            source.name,
            json.dumps(source.metadata),
            source.text,
            source.created_at,
            source.chunks,
            source.digest,
            source.confirmed_at,
            source.valid_from,
            source.valid_to,
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
                source.valid_from,
                source.valid_to,
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
        columns = (
            "corpus_id, chunk_id, source_id, source_name, position, title, context, locator, text, metadata, "
            "digest, valid_from, valid_to, written_to, config"
        )
        values = (
            f"$1, $2, $3, $4, $5, $6, $7, $8::json, $9, $10::jsonb, $11, $12, $13, $13, {_CONFIG}"
        )
        if self._dimension:
            return (
                f"INSERT INTO memory_chunks ({columns}, embedding, embedding_model) "
                f"VALUES ({values}, $14::vector, $15)"
            )
        return f"INSERT INTO memory_chunks ({columns}, embedding_model) VALUES ({values}, $14)"

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
        as_of: datetime | None = None,
    ) -> tuple[list[Source], str | None]:
        params: list[Any] = [corpus_id]
        facets = _facet_sql(filters, params, "metadata::jsonb") + _sources_sql(
            known_at, history, params, as_of=as_of
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
                f"SELECT * FROM {_CHUNKS} WHERE corpus_id = $1 AND source_id = $2 ORDER BY position",
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

    async def source_digests(
        self, corpus_id: str
    ) -> list[tuple[str, dict[str, Any], str, int, datetime | None, datetime | None]]:
        async with self._guard("source_digests"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT name, metadata, encode(sha256(convert_to(text, 'UTF8')), 'hex') AS text_sha256, chunks, "
                "valid_from, written_to FROM memory_sources WHERE corpus_id = $1 AND retired_at IS NULL",
                corpus_id,
            )
        return [
            (
                r["name"],
                _json(r["metadata"]),
                r["text_sha256"],
                r["chunks"],
                r["valid_from"],
                r["written_to"],
            )
            for r in records
        ]

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
