import json
from datetime import datetime
from typing import Any

from rfnry_memory.errors import StorageError
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.records import _count, _json
from rfnry_memory.rows import RetrievalLabelRow, RetrievalRow


def _retrieval(record: Any) -> RetrievalRow:
    return RetrievalRow(
        retrieval_id=record["retrieval_id"],
        corpus_ids=list(record["corpus_ids"]),
        verb=record["verb"],
        at=record["at"],
        recorded=record["recorded"],
        duration_ms=record["duration_ms"],
        body=_json(record["body"]),
    )


def _label(record: Any) -> RetrievalLabelRow:
    return RetrievalLabelRow(
        retrieval_id=record["retrieval_id"],
        label_id=record["label_id"],
        source=record["source"],
        at=record["at"],
        body=_json(record["body"]),
    )


class Retrievals(Base):
    async def put_retrieval(self, corpus_ids: list[str], row: RetrievalRow) -> None:
        if row.corpus_ids != corpus_ids:
            raise StorageError("postgres: row corpus does not match the call corpus")
        async with self._guard("put_retrieval"), self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_retrievals (retrieval_id, corpus_ids, verb, at, recorded, duration_ms, body) "
                "VALUES ($1, $2::text[], $3, $4, $5, $6, $7::json)",
                row.retrieval_id,
                row.corpus_ids,
                row.verb,
                row.at,
                row.recorded,
                row.duration_ms,
                json.dumps(row.body),
            )

    async def get_retrieval(self, corpus_ids: list[str], retrieval_id: str) -> RetrievalRow | None:
        async with self._guard("get_retrieval"), self._pool.acquire() as conn:
            record = await conn.fetchrow(
                "SELECT * FROM memory_retrievals WHERE retrieval_id = $2 AND corpus_ids <@ $1::text[]",
                corpus_ids,
                retrieval_id,
            )
        return None if record is None else _retrieval(record)

    async def list_retrievals(
        self,
        corpus_ids: list[str],
        *,
        since: datetime | None,
        limit: int | None,
        cursor: str | None,
    ) -> tuple[list[RetrievalRow], str | None]:
        params: list[Any] = [corpus_ids]
        sql = "SELECT * FROM memory_retrievals WHERE corpus_ids <@ $1::text[]"
        if since is not None:
            params.append(since)
            sql += f" AND recorded > ${len(params)}"
        sql += ' ORDER BY recorded DESC, retrieval_id COLLATE "C" DESC'
        offset = int(cursor) if cursor else 0
        if limit is not None:
            params.extend([limit + 1, offset])
            sql += f" LIMIT ${len(params) - 1} OFFSET ${len(params)}"
        async with self._guard("list_retrievals"), self._pool.acquire() as conn:
            records = await conn.fetch(sql, *params)
        rows = [_retrieval(r) for r in records]
        if limit is None or len(rows) <= limit:
            return rows, None
        return rows[:limit], str(offset + limit)

    async def forget_retrievals(self, corpus_ids: list[str], *, before: datetime) -> int:
        async with self._guard("forget_retrievals"), self._pool.acquire() as conn:
            return _count(
                await conn.execute(
                    "DELETE FROM memory_retrievals WHERE corpus_ids <@ $1::text[] AND recorded < $2",
                    corpus_ids,
                    before,
                )
            )

    async def put_retrieval_label(
        self, corpus_ids: list[str], row: RetrievalLabelRow
    ) -> RetrievalLabelRow | None:
        async with self._guard("put_retrieval_label"), self._pool.acquire() as conn:
            held = await conn.fetchval(
                "SELECT 1 FROM memory_retrievals WHERE retrieval_id = $2 AND corpus_ids <@ $1::text[]",
                corpus_ids,
                row.retrieval_id,
            )
            if held is None:
                return None
            await conn.execute(
                "INSERT INTO memory_retrieval_labels (retrieval_id, label_id, source, at, body) "
                "VALUES ($1, $2, $3, $4, $5::json) ON CONFLICT (retrieval_id, label_id) DO NOTHING",
                row.retrieval_id,
                row.label_id,
                row.source,
                row.at,
                json.dumps(row.body),
            )
            record = await conn.fetchrow(
                "SELECT * FROM memory_retrieval_labels WHERE retrieval_id = $1 AND label_id = $2",
                row.retrieval_id,
                row.label_id,
            )
        return _label(record)

    async def retrieval_labels(
        self,
        corpus_ids: list[str],
        retrieval_ids: list[str],
        *,
        since: datetime | None = None,
    ) -> list[RetrievalLabelRow]:
        params: list[Any] = [corpus_ids, retrieval_ids]
        sql = (
            "SELECT l.* FROM memory_retrieval_labels l JOIN memory_retrievals r ON r.retrieval_id = l.retrieval_id "
            "WHERE r.corpus_ids <@ $1::text[] AND l.retrieval_id = ANY($2::text[])"
        )
        if since is not None:
            params.append(since)
            sql += f" AND l.at > ${len(params)}"
        async with self._guard("retrieval_labels"), self._pool.acquire() as conn:
            records = await conn.fetch(sql + ' ORDER BY l.at, l.label_id COLLATE "C"', *params)
        return [_label(r) for r in records]

    async def judged_retrievals(self, corpus_ids: list[str]) -> set[str]:
        async with self._guard("judged_retrievals"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT DISTINCT l.retrieval_id FROM memory_retrieval_labels l JOIN memory_retrievals r "
                "ON r.retrieval_id = l.retrieval_id WHERE r.corpus_ids <@ $1::text[] AND l.source = 'judge'",
                corpus_ids,
            )
        return {r["retrieval_id"] for r in records}
