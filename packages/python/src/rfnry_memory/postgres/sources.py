import json
from datetime import datetime
from typing import Any

from rfnry_memory.errors import InputError, NotFoundError, StorageError
from rfnry_memory.models import (
    Chunk,
    Counts,
    Resemblance,
    Source,
)
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.graph import _drop_evidence, _reconcile, _write_graph
from rfnry_memory.postgres.keys import keyed, refuse_unfit
from rfnry_memory.postgres.language import _CONFIG, _pin
from rfnry_memory.postgres.locks import _language_lock, _sync_lock
from rfnry_memory.postgres.records import (
    _chunk,
    _closings,
    _count,
    _json,
    _prior,
    _source,
    _taken,
    _vector,
)
from rfnry_memory.postgres.sql import _CHUNKS, _facet_sql, _in_force_sql, _known_closed
from rfnry_memory.postgres.vectors import _copy_vectors, _delete_vectors, _vectors_table
from rfnry_memory.rows import (
    GraphWrite,
    SourceWrite,
    after,
    chunk_digest,
    stamped,
    successors,
)


class _Moved(StorageError):
    pass


_IN_FORCE = (
    "(retired_at IS NOT NULL AND valid_to > {at}::timestamptz "
    "AND (valid_from IS NULL OR valid_to > valid_from))"
)


def _started(write: SourceWrite, at: datetime) -> SourceWrite:
    if write.source.valid_from is not None:
        return write
    start = {"valid_from": at}
    return SourceWrite(
        write.source.model_copy(update=start),
        [c.model_copy(update=start) for c in write.chunks],
        write.vectors,
        write.settings,
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
            sql += f" AND NOT {_known_closed('source', alias, learned)}"
    elif not history and as_of is None:
        sql += f" AND {alias}retired_at IS NULL"
    if history:
        return sql
    return sql + _in_force_sql(as_of, known_at, params, alias, "source")


def _withdrawn(r: Any) -> bool:
    return (
        r["retired_at"] is not None
        and r["replaced_by"] is None
        and r["valid_from"] is not None
        and r["valid_to"] == r["valid_from"]
    )


def _own_end(r: Any) -> datetime | None:
    own: datetime | None = (
        r["valid_to"]
        if r["retired_at"] is not None and r["replaced_by"] is None
        else r["written_to"]
    )
    return own


def _in_force(end: datetime | None, start: datetime | None, at: datetime) -> bool:
    return end is None or (end > at and (start is None or end > start))


def _closed_end(
    name: str,
    valid_from: datetime | None,
    own: datetime | None,
    at: datetime,
    start: datetime,
    history: bool,
) -> datetime:
    if valid_from is not None and valid_from > at:
        return valid_from
    if valid_from is not None and start < valid_from:
        raise InputError(
            f"postgres: source {name!r} takes effect at {start.isoformat()}, before the version it replaces took "
            f"effect, at {valid_from.isoformat()}; that moment itself corrects it from the start"
        )
    if not history and (own is None or own > at) and start > at:
        raise InputError(
            f"postgres: source {name!r} takes effect at {start.isoformat()}, after this write; a version in force "
            "is replaced on the day its update takes effect: write the update then"
        )
    end = start if own is None else min(own, start)
    return end if valid_from is None else max(valid_from, end)


def _retired_end(
    r: Any, follows: datetime | None, valid_to: datetime | None, at: datetime
) -> datetime:
    start: datetime | None = r["valid_from"]
    own = _own_end(r)
    if valid_to is not None:
        if start is not None and valid_to < start:
            raise InputError(
                f"postgres: source {r['source_id']!r} cannot end at {valid_to.isoformat()}, before it took effect, at {start.isoformat()}"
            )
        return valid_to
    if start is not None and start > at:
        return start
    if follows is None:
        return at if own is None else min(own, at)
    if start is not None and follows < start:
        raise InputError(
            f"postgres: the source replacing {r['source_id']!r} took effect at {follows.isoformat()}, before it did, at {start.isoformat()}; give valid_to"
        )
    return follows if own is None else min(own, follows)


_ARCHIVE_SQL = (
    "INSERT INTO memory_closings (corpus_id, kind, id, closed_at, valid_to, successor, closed_by, revised_at) "
    "SELECT corpus_id, 'source', source_id, retired_at, valid_to, replaced_by, coalesce(closed_by, '{}'), "
    "$3::timestamptz FROM memory_sources WHERE corpus_id = $1 AND source_id = ANY($2::text[]) "
    "AND retired_at IS NOT NULL"
)


_CLOSE_SQL = (
    "UPDATE memory_sources s SET valid_to = coalesce(c.ends, s.written_to), replaced_by = c.successor, "
    "retired_at = CASE WHEN c.ends IS NULL THEN NULL ELSE $3::timestamptz END, "
    "closed_by = CASE WHEN c.ends IS NULL THEN NULL ELSE $4::jsonb END, "
    "revised_at = CASE WHEN s.retired_at IS NULL THEN s.revised_at ELSE $3::timestamptz END "
    "FROM unnest($2::text[], $5::timestamptz[], $6::text[]) AS c(source_id, ends, successor) "
    "WHERE s.corpus_id = $1 AND s.source_id = c.source_id"
)


Change = tuple[Any, datetime | None, str | None]


async def _close(
    conn: Any, corpus_id: str, changes: list[Change], closed_by: dict[str, Any], at: datetime
) -> None:
    ids = [r["source_id"] for r, _, _ in changes]
    if not ids:
        return
    back = [
        r["source_id"]
        for r, end, _ in changes
        if r["retired_at"] is not None
        and not _in_force(r["valid_to"], r["valid_from"], at)
        and (end is None or _in_force(end, r["valid_from"], at))
    ]
    await refuse_unfit(conn, "memory_sources", "source_id", "source", corpus_id, back)
    await conn.execute("SELECT memory_count($1, 'chunk', $2::text[], -1)", corpus_id, ids)
    await conn.execute(_ARCHIVE_SQL, corpus_id, ids, at)
    await conn.execute(
        _CLOSE_SQL,
        corpus_id,
        ids,
        at,
        json.dumps(closed_by),
        [end for _, end, _ in changes],
        [successor for _, _, successor in changes],
    )
    await conn.execute(
        "UPDATE memory_chunks c SET retired_at = s.retired_at, valid_to = s.valid_to, "
        "revised_at = s.revised_at FROM memory_sources s WHERE c.corpus_id = $1 "
        "AND c.source_id = ANY($2::text[]) AND s.corpus_id = c.corpus_id AND s.source_id = c.source_id",
        corpus_id,
        ids,
    )
    await conn.execute("SELECT memory_count($1, 'chunk', $2::text[], 1)", corpus_id, ids)


async def _closed_by(conn: Any, corpus_id: str, ids: list[str], skip: list[str]) -> list[Any]:
    if not ids:
        return []
    return list(
        await conn.fetch(
            "SELECT * FROM memory_sources WHERE corpus_id = $1 AND replaced_by = ANY($2::text[]) "
            'AND source_id <> ALL($3::text[]) ORDER BY source_id COLLATE "C" FOR UPDATE',
            corpus_id,
            ids,
            skip,
        )
    )


async def _replacement(conn: Any, corpus_id: str, source_id: str | None) -> Any:
    seen: set[str] = set()
    while source_id is not None and source_id not in seen:
        seen.add(source_id)
        record = await conn.fetchrow(
            "SELECT * FROM memory_sources WHERE corpus_id = $1 AND source_id = $2",
            corpus_id,
            source_id,
        )
        if (
            record is None
            or record["valid_from"] is None
            or record["valid_to"] != record["valid_from"]
        ):
            return record
        source_id = record["replaced_by"]
    return None


def _moment(at: datetime, rows: list[Any]) -> datetime:
    return after(at, (r[c] for r in rows for c in ("created_at", "retired_at", "revised_at")))


def _empty(r: Any) -> bool:
    return r["valid_from"] is not None and r["valid_to"] == r["valid_from"]


async def _returned(
    conn: Any, corpus_id: str, record: Any, replacement: Any, at: datetime, by: dict[str, Any]
) -> tuple[Change, dict[str, Any]]:
    retired_at = record["retired_at"]
    while True:
        prior = await _prior(conn, corpus_id, "source", record["source_id"], retired_at)
        if prior is not None and prior["successor"] is not None:
            successor = await conn.fetchrow(
                "SELECT * FROM memory_sources WHERE corpus_id = $1 AND source_id = $2",
                corpus_id,
                prior["successor"],
            )
            if successor is None or not _empty(successor):
                return (record, prior["valid_to"], prior["successor"]), _json(prior["closed_by"])
            replacement = (
                await _replacement(conn, corpus_id, successor["replaced_by"]) or replacement
            )
            retired_at = prior["closed_at"]
            continue
        cap = record["written_to"] if prior is None else prior["valid_to"]
        if replacement is not None:
            start = replacement["valid_from"] or at
            if cap is None or start < cap:
                end = _closed_end(replacement["name"], record["valid_from"], cap, at, start, True)
                return (record, end, replacement["source_id"]), by
        if prior is None:
            return (record, None, None), by
        return (record, prior["valid_to"], None), _json(prior["closed_by"])


async def _hand_back(
    conn: Any, corpus_id: str, handed: list[tuple[Any, Any]], at: datetime, by: dict[str, Any]
) -> list[str]:
    closed: list[str] = []
    for record, replacement in sorted(
        handed, key=lambda h: (h[0]["created_at"], h[0]["source_id"]), reverse=True
    ):
        change, closer = await _returned(conn, corpus_id, record, replacement, at, by)
        await _close(conn, corpus_id, [change], closer, at)
        if change[1] is not None:
            closed.append(record["source_id"])
    return closed


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
        settings: list[dict[str, Any] | None] | None = None,
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
            await _taken(conn, corpus_id, "source", [source.source_id])
            await self._insert_sources(
                conn, corpus_id, [SourceWrite(source, chunks, vectors, settings)], model, table
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
        written_by: dict[str, Any] | None = None,
    ) -> tuple[Counts, list[str], list[Source], list[Source]] | None:
        if any(
            source.corpus_id != corpus_id or any(c.corpus_id != corpus_id for c in chunks)
            for source, chunks, *_ in writes
        ):
            raise StorageError("postgres: row corpus does not match the call corpus")
        keep = keep or {}
        for write in writes:
            self._check_vectors(write.vectors)
        graph = graph or GraphWrite()
        graph.check(corpus_id, {("chunk", c.chunk_id) for w in writes for c in w.chunks})
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
                history = bool(
                    await conn.fetchval(
                        "SELECT history FROM memory_corpora WHERE corpus_id = $1", corpus_id
                    )
                )
                params: list[Any] = [corpus_id]
                facets = _facet_sql(filters, params, cast="::jsonb")
                held = ""
                if history:
                    params.append(at)
                    held = " OR " + _IN_FORCE.format(at=f"${len(params)}")
                matching = {
                    r["source_id"]: r
                    for r in await conn.fetch(
                        f"SELECT * FROM memory_sources WHERE corpus_id = $1{facets} "
                        f'AND (retired_at IS NULL{held}) ORDER BY source_id COLLATE "C" FOR UPDATE',
                        *params,
                    )
                }
                if any(
                    source_id not in matching or matching[source_id]["digest"] != digest
                    for source_id, digest in keep.items()
                ) or (history and any(r["created_at"] > at for r in matching.values())):
                    raise _Moved
                stays = set(keep) | {i for i, r in matching.items() if r["replaced_by"] in keep}
                dropped = [source_id for source_id in matching if source_id not in stays]
                await _taken(conn, corpus_id, "source", [w.source.source_id for w in writes])
                later = successors(writes)
                touched: set[str] = set()
                retired: list[str] = []
                if history:
                    outside = await _closed_by(conn, corpus_id, dropped, list(matching))
                    moment = _moment(at, [*matching.values(), *outside])
                    if moment != at:
                        at, writes = moment, stamped(writes, moment)
                    followed = {
                        later[matching[i]["name"]] for i in dropped if matching[i]["name"] in later
                    }
                    writes = [
                        _started(w, at) if w.source.source_id in followed else w for w in writes
                    ]
                    starts = {w.source.source_id: w.source.valid_from or at for w in writes}
                    names = {w.source.source_id: w.source.name for w in writes}
                    changes: list[Change] = []
                    for source_id in dropped:
                        r = matching[source_id]
                        successor = later.get(r["name"])
                        start = starts.get(successor or "", at)
                        changes.append(
                            (
                                r,
                                _closed_end(
                                    r["name"], r["valid_from"], _own_end(r), at, start, True
                                ),
                                successor,
                            )
                        )
                    withdrawn = {
                        r["source_id"]: successor
                        for r, end, successor in changes
                        if r["valid_from"] is not None and end == r["valid_from"]
                    }
                    await _close(conn, corpus_id, changes, written_by or {}, at)
                    await self._insert_sources(conn, corpus_id, writes, model, table)
                    handed: list[tuple[Any, Any]] = [
                        (matching[i], None)
                        for i in keep
                        if matching[i]["retired_at"] is not None
                        and matching[i]["replaced_by"] in withdrawn
                    ]
                    for r in outside:
                        if r["replaced_by"] in withdrawn:
                            successor = withdrawn[r["replaced_by"]]
                            replacement = None
                            if successor is not None:
                                replacement = {
                                    "valid_from": starts[successor],
                                    "name": names[successor],
                                    "source_id": successor,
                                }
                            handed.append((r, replacement))
                    deleted = Counts()
                    retired = [r["source_id"] for r, _, _ in changes]
                    retired += await _hand_back(conn, corpus_id, handed, at, written_by or {})
                else:
                    starts = {w.source.source_id: w.source.valid_from or at for w in writes}
                    for source_id in dropped:
                        r = matching[source_id]
                        start = starts.get(later.get(r["name"], ""), at)
                        _closed_end(r["name"], r["valid_from"], _own_end(r), at, start, False)
                    deleted = await self._delete_sources(conn, corpus_id, dropped, touched, table)
                    await self._insert_sources(conn, corpus_id, writes, model, table)
                if keep:
                    await conn.execute(
                        "UPDATE memory_sources SET confirmed_at = $3, confirmed_by = $4::jsonb "
                        "WHERE corpus_id = $1 AND source_id = ANY($2::text[]) AND confirmed_at < $3",
                        corpus_id,
                        list(keep),
                        at,
                        json.dumps(written_by or {}),
                    )
                kept = {
                    r["source_id"]: _source(r)
                    for r in await conn.fetch(
                        "SELECT * FROM memory_sources WHERE corpus_id = $1 AND source_id = ANY($2::text[])",
                        corpus_id,
                        list(keep),
                    )
                }
                await _write_graph(conn, corpus_id, graph, touched)
                pruned = await _reconcile(conn, corpus_id, touched)
        except _Moved:
            return None
        return (
            deleted.model_copy(update={"entities": pruned}),
            retired,
            [w.source for w in writes],
            [kept[i] for i in keep],
        )

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
        source, chunks, vectors, _ = write
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
                if _withdrawn(old):
                    raise InputError(
                        f"postgres: source {source_id!r} was withdrawn at {old['retired_at'].isoformat()}; ingest it anew instead"
                    )
                history = await conn.fetchval(
                    "SELECT history FROM memory_corpora WHERE corpus_id = $1", corpus_id
                )
                if history and old["created_at"] > at:
                    raise _Moved
                await _taken(conn, corpus_id, "source", [source.source_id])
                touched: set[str] = set()
                retired: list[str] = []
                if history:
                    closed = await _closed_by(conn, corpus_id, [source_id], [])
                    moment = _moment(at, [old, *closed])
                    if moment != at:
                        at, [write] = moment, stamped([write], moment)
                    write = _started(write, at)
                    start = write.source.valid_from or at
                    end = _closed_end(
                        source.name, old["valid_from"], _own_end(old), at, start, True
                    )
                    await _close(
                        conn, corpus_id, [(old, end, source.source_id)], source.written_by, at
                    )
                    await self._insert_sources(conn, corpus_id, [write], model, table)
                    handed = (
                        [(r, write.source.model_dump()) for r in closed]
                        if old["valid_from"] is not None and end == old["valid_from"]
                        else []
                    )
                    deleted = Counts()
                    retired = [source_id] + await _hand_back(
                        conn, corpus_id, handed, at, source.written_by
                    )
                else:
                    _closed_end(
                        source.name,
                        old["valid_from"],
                        _own_end(old),
                        at,
                        source.valid_from or at,
                        False,
                    )
                    deleted = await self._delete_sources(
                        conn, corpus_id, [source_id], touched, table
                    )
                    await self._insert_sources(conn, corpus_id, [write], model, table)
                await _write_graph(conn, corpus_id, graph, touched)
                pruned = await _reconcile(conn, corpus_id, touched)
        except _Moved:
            return None
        return deleted.model_copy(update={"entities": pruned}), retired, write.source

    async def retire_source(
        self,
        corpus_id: str,
        source_id: str,
        *,
        at: datetime,
        replaced_by: str | None,
        valid_to: datetime | None,
        closed_by: dict[str, Any],
    ) -> Source:
        async with (
            self._guard("retire_source"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _sync_lock(conn, corpus_id)
            await _language_lock(conn, corpus_id, shared=True)
            named = [source_id] if replaced_by is None else [source_id, replaced_by]
            rows = {
                r["source_id"]: r
                for r in await conn.fetch(
                    "SELECT * FROM memory_sources WHERE corpus_id = $1 AND source_id = ANY($2::text[]) "
                    'ORDER BY source_id COLLATE "C" FOR UPDATE',
                    corpus_id,
                    named,
                )
            }
            for wanted in named:
                if wanted not in rows:
                    raise NotFoundError(f"postgres: source {wanted!r} was not found")
            old, successor = rows[source_id], rows.get(replaced_by or "")
            if not await conn.fetchval(
                "SELECT history FROM memory_corpora WHERE corpus_id = $1", corpus_id
            ):
                raise InputError(
                    f"postgres: corpus {corpus_id!r} keeps no history, so a retired document would be erased; delete_source erases it"
                )
            withdrawing = valid_to is not None and valid_to == old["valid_from"]
            if old["replaced_by"] is not None and (not withdrawing or replaced_by is not None):
                raise InputError(
                    f"postgres: source {source_id!r} was already replaced by {old['replaced_by']!r}; "
                    f"replace or retire {old['replaced_by']!r} instead"
                )
            if _withdrawn(old):
                raise InputError(
                    f"postgres: source {source_id!r} was withdrawn at {old['retired_at'].isoformat()}"
                )
            if replaced_by == source_id:
                raise InputError(f"postgres: source {source_id!r} cannot replace itself")
            if successor is not None and successor["retired_at"] is not None:
                raise InputError(
                    f"postgres: source {replaced_by!r} was retired at {successor['retired_at'].isoformat()}; name the version in force"
                )
            if old["created_at"] > at:
                raise InputError(
                    f"postgres: source {source_id!r} was written at {old['created_at'].isoformat()}, after this retirement's moment"
                )
            if (
                valid_to is None
                and replaced_by is None
                and old["retired_at"] is not None
                and old["valid_to"] <= at
            ):
                raise InputError(
                    f"postgres: source {source_id!r} was already retired at {old['retired_at'].isoformat()}, "
                    f"out of force since {old['valid_to'].isoformat()}; give valid_to to revise its end"
                )
            closed = await _closed_by(conn, corpus_id, [source_id], [])
            at = _moment(at, [old, *closed])
            end = _retired_end(
                old, None if successor is None else successor["valid_from"], valid_to, at
            )
            await _close(conn, corpus_id, [(old, end, replaced_by)], closed_by, at)
            if old["valid_from"] is not None and end == old["valid_from"]:
                replacement = successor or await _replacement(conn, corpus_id, old["replaced_by"])
                await _hand_back(conn, corpus_id, [(r, replacement) for r in closed], at, closed_by)
            record = await conn.fetchrow(
                "SELECT * FROM memory_sources WHERE corpus_id = $1 AND source_id = $2",
                corpus_id,
                source_id,
            )
        return _source(record)

    async def chunk_vectors(
        self, corpus_id: str, model: str, digests: list[str]
    ) -> dict[str, tuple[list[float], dict[str, Any] | None]]:
        if not self._dimension or not digests:
            return {}
        async with self._guard("chunk_vectors"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT DISTINCT ON (digest) digest, embedding::text AS embedding, embedding_settings "
                "FROM memory_chunks WHERE corpus_id = $1 AND embedding_model = $2 "
                "AND digest = ANY($3::text[]) AND embedding IS NOT NULL",
                corpus_id,
                model,
                digests,
            )
        return {
            r["digest"]: (
                [float(x) for x in json.loads(r["embedding"])],
                _json(r["embedding_settings"]),
            )
            for r in records
        }

    async def _insert_sources(
        self,
        conn: Any,
        corpus_id: str,
        writes: list[SourceWrite],
        model: str | None,
        table: str | None,
    ) -> None:
        if not writes:
            return
        await conn.executemany(
            "INSERT INTO memory_sources (corpus_id, source_id, name, metadata, text, created_at, chunks, digest, "
            "confirmed_at, valid_from, valid_to, written_to, written_by, confirmed_by) "
            "VALUES ($1, $2, $3, $4::json, $5, $6, $7, $8, $9, $10, $11, $11, $12::jsonb, $13::jsonb)",
            [
                (
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
                    json.dumps(source.written_by),
                    None if source.confirmed_by is None else json.dumps(source.confirmed_by),
                )
                for source, *_ in writes
            ],
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
                json.dumps(source.written_by),
                None if c.extracted_by is None else json.dumps(c.extracted_by),
                *(
                    [
                        _vector(vectors[i] if vectors is not None else None),
                        model,
                        None
                        if settings is None or settings[i] is None
                        else json.dumps(settings[i]),
                    ]
                    if self._dimension
                    else [model]
                ),
            )
            for source, chunks, vectors, settings in writes
            for i, c in enumerate(chunks)
        ]
        if rows:
            await conn.executemany(self._insert_chunk_sql(), rows)
        await conn.execute(
            "SELECT memory_count($1, 'chunk', $2::text[], 1)",
            corpus_id,
            [w.source.source_id for w in writes],
        )
        await _copy_vectors(
            conn,
            table,
            corpus_id,
            "chunk",
            [c.chunk_id for _, chunks, *_ in writes for c in chunks],
        )

    def _insert_chunk_sql(self) -> str:
        columns = (
            "corpus_id, chunk_id, source_id, source_name, position, title, context, locator, text, metadata, "
            "digest, valid_from, valid_to, written_to, written_by, extracted_by, config"
        )
        values = (
            "$1, $2, $3, $4, $5, $6, $7, $8::json, $9, $10::jsonb, $11, $12, $13, $13, $14::jsonb, "
            "$15::jsonb, " + _CONFIG
        )
        if self._dimension:
            return (
                f"INSERT INTO memory_chunks ({columns}, embedding, embedding_model, embedding_settings) "
                f"VALUES ({values}, $16::vector, $17, $18::jsonb)"
            )
        return f"INSERT INTO memory_chunks ({columns}, embedding_model) VALUES ({values}, $16)"

    async def sync_set(
        self, corpus_id: str, filters: dict[str, Any], *, at: datetime
    ) -> list[Source]:
        params: list[Any] = [corpus_id]
        facets = _facet_sql(filters, params, cast="::jsonb")
        params.append(at)
        held = _IN_FORCE.format(at=f"${len(params)}")
        async with self._guard("sync_set"), self._pool.acquire() as conn:
            records = await conn.fetch(
                f"SELECT * FROM memory_sources WHERE corpus_id = $1{facets} AND (retired_at IS NULL "
                "OR ((SELECT history FROM memory_corpora WHERE corpus_id = $1) AND "
                f'{held})) ORDER BY retired_at IS NOT NULL, created_at DESC, source_id COLLATE "C" DESC',
                *params,
            )
        return [_source(r) for r in records]

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
        now: bool = False,
    ) -> tuple[list[Source], str | None]:
        params: list[Any] = [corpus_id]
        async with self._guard("list_sources"), self._pool.acquire() as conn:
            filters = await keyed(conn, filters, as_of if now else None)
            facets = _facet_sql(filters, params, cast="::jsonb", known_at=known_at) + _sources_sql(
                known_at, history, params, as_of=as_of
            )
            offset = int(cursor) if cursor else 0
            params.extend([limit + 1, offset])
            records = await conn.fetch(
                f'SELECT * FROM memory_sources WHERE corpus_id = $1{facets} ORDER BY created_at DESC, source_id COLLATE "C" DESC '
                f"LIMIT ${len(params) - 1} OFFSET ${len(params)}",
                *params,
            )
            closings = await _closings(conn, [("source", records)], known_at)
        rows = [_source(r, known_at, closings) for r in records]
        return rows[:limit], (str(offset + limit) if len(rows) > limit else None)

    async def resembling(
        self,
        corpus_ids: list[str],
        probe: str | tuple[str, str],
        *,
        filters: dict[str, Any] | None,
        known_at: datetime | None,
        history: bool,
        as_of: datetime | None,
        limit: int,
        now: bool = False,
    ) -> list[Resemblance]:
        async with self._guard("resembling"), self._pool.acquire() as conn:
            if isinstance(probe, str):
                signature = await conn.fetchval("SELECT memory_signature($1)", probe)
            else:
                signature = await conn.fetchval(
                    "SELECT signature FROM memory_sources WHERE corpus_id = $1 AND source_id = $2",
                    *probe,
                )
            if signature is None and isinstance(probe, str):
                raise InputError("postgres: the text holds no words to compare")
            if signature is None:
                return []
            params: list[Any] = [corpus_ids, signature]
            filters = await keyed(conn, filters, as_of if now else None)
            where = _facet_sql(filters, params, cast="::jsonb", known_at=known_at) + _sources_sql(
                known_at, history, params, as_of=as_of
            )
            if not isinstance(probe, str):
                params.extend(probe)
                where += f" AND (corpus_id, source_id) <> (${len(params) - 1}, ${len(params)})"
            params.append(limit)
            records = await conn.fetch(
                "SELECT *, memory_share(signature, $2::bigint[]) AS share FROM memory_sources "
                f"WHERE corpus_id = ANY($1::text[]) AND memory_bands(signature) && memory_bands($2::bigint[]){where} "
                f'ORDER BY share DESC, corpus_id COLLATE "C", source_id COLLATE "C" LIMIT ${len(params)}',
                *params,
            )
            closings = await _closings(conn, [("source", records)], known_at)
        return [
            Resemblance(source=_source(r, known_at, closings), share=r["share"]) for r in records
        ]

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

    async def _delete_sources(
        self,
        conn: Any,
        corpus_id: str,
        source_ids: list[str],
        touched: set[str],
        table: str | None,
    ) -> Counts:
        if not source_ids:
            return Counts()
        await conn.execute(
            "SELECT memory_count($1, 'chunk', $2::text[], -1)", corpus_id, source_ids
        )
        chunk_ids = [
            r["chunk_id"]
            for r in await conn.fetch(
                "SELECT chunk_id FROM memory_chunks WHERE corpus_id = $1 AND source_id = ANY($2::text[])",
                corpus_id,
                source_ids,
            )
        ]
        mentions, relations = await _drop_evidence(conn, corpus_id, "chunk", chunk_ids, touched)
        vectors = await _delete_vectors(conn, table, "chunk", chunk_ids)
        sources = await conn.execute(
            "DELETE FROM memory_sources WHERE corpus_id = $1 AND source_id = ANY($2::text[])",
            corpus_id,
            source_ids,
        )
        closings = await conn.execute(
            "DELETE FROM memory_closings WHERE corpus_id = $1 AND kind = 'source' AND id = ANY($2::text[])",
            corpus_id,
            source_ids,
        )
        return Counts(
            sources=_count(sources),
            chunks=len(chunk_ids),
            mentions=mentions,
            relations=relations,
            vectors=vectors,
            closings=_count(closings),
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
            deleted = await self._delete_sources(conn, corpus_id, versions, touched, table)
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
                    'AND valid_to <= $2 ORDER BY source_id COLLATE "C" FOR UPDATE',
                    corpus_id,
                    before,
                )
            ]
            touched: set[str] = set()
            deleted = await self._delete_sources(conn, corpus_id, gone, touched, table)
            pruned = await _reconcile(conn, corpus_id, touched)
        return deleted.model_copy(update={"entities": pruned})
