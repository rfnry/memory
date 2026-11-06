import json
from typing import Any

from rfnry_memory.errors import ConfigurationError
from rfnry_memory.models import Chunk, Kind, Ref, Story, Window
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.locks import _language_lock
from rfnry_memory.postgres.records import _vector
from rfnry_memory.postgres.sql import _scope_sql, _table
from rfnry_memory.postgres.vectors import _KINDS, _STAGED, _page


def _no_staged_row(kind: Kind) -> str:
    table, key = _table(kind)
    return (
        "NOT (SELECT EXISTS (SELECT 1 FROM memory_staged_vectors v "
        f"WHERE v.corpus_id = {table}.corpus_id AND v.model = $2 "
        f"AND v.passage_kind = '{kind}' AND v.passage_id = {table}.{key}))"
    )


_UNSTAGED: dict[Kind, str] = {kind: _no_staged_row(kind) for kind in _KINDS}


_HOLD: dict[Kind, str] = {
    "chunk": (
        "SELECT 1 FROM memory_sources WHERE corpus_id = $1 AND source_id IN "
        "(SELECT source_id FROM memory_chunks WHERE corpus_id = $1 AND chunk_id = ANY($2::text[])) "
        'ORDER BY source_id COLLATE "C" FOR KEY SHARE'
    ),
    "story": (
        "SELECT 1 FROM memory_stories WHERE corpus_id = $1 AND story_id = ANY($2::text[]) "
        'ORDER BY story_id COLLATE "C" FOR KEY SHARE'
    ),
}


def _stage_sql(kind: Kind) -> str:
    table, key = _table(kind)
    return (
        "INSERT INTO memory_staged_vectors "
        "(corpus_id, model, passage_kind, passage_id, vector, settings) "
        f"SELECT $1, $2, '{kind}', p.{key}, given.vector::vector, $3::text::jsonb "
        "FROM unnest($4::text[], $5::text[]) AS given(id, vector) "
        f"JOIN {table} p ON p.corpus_id = $1 AND p.{key} = given.id "
        "ON CONFLICT (corpus_id, model, passage_kind, passage_id) DO UPDATE SET "
        "vector = EXCLUDED.vector, settings = EXCLUDED.settings"
    )


async def _stageable(conn: Any) -> None:
    if not await conn.fetchval(f"SELECT {_STAGED}"):
        raise ConfigurationError(
            "postgres: this schema has no table of staged vectors — run rfnry-memory upgrade, "
            "or rfnry-memory set --dimension when it holds no vectors"
        )


class Switch(Base):
    async def passages_unstaged(
        self,
        corpus_id: str,
        *,
        model: str,
        filters: dict[str, Any] | None,
        after: Ref | None,
        limit: int,
    ) -> list[Chunk | Story]:
        async with self._guard("passages_unstaged"), self._pool.acquire() as conn:
            await _stageable(conn)
            return await _page(conn, corpus_id, model, filters, after, limit, _UNSTAGED)

    async def count_unstaged(
        self, corpus_id: str, *, model: str, filters: dict[str, Any] | None
    ) -> int:
        counted = 0
        async with self._guard("count_unstaged"), self._pool.acquire() as conn:
            await _stageable(conn)
            for kind in _KINDS:
                table, _ = _table(kind)
                params: list[Any] = [corpus_id, model]
                facets = _scope_sql(kind, filters, Window(), None, None, params)
                counted += await conn.fetchval(
                    f"SELECT count(*) FROM {table} WHERE corpus_id = $1 "
                    f"AND embedding_model IS DISTINCT FROM $2 AND {_UNSTAGED[kind]}{facets}",
                    *params,
                )
        return counted

    async def stage(
        self,
        corpus_id: str,
        refs: list[Ref],
        vectors: list[list[float]],
        model: str,
        *,
        settings: dict[str, Any] | None,
    ) -> None:
        settled = None if settings is None else json.dumps(settings)
        given = list(zip(refs, vectors, strict=True))
        async with (
            self._guard("stage"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=True)
            for kind in _KINDS:
                ids = [i for (k, i), _ in given if k == kind]
                if ids:
                    await conn.execute(_HOLD[kind], corpus_id, ids)
                    await conn.execute(
                        _stage_sql(kind),
                        corpus_id,
                        model,
                        settled,
                        ids,
                        [_vector(v) for (k, _), v in given if k == kind],
                    )

    async def staged_counts(self, corpus_id: str) -> dict[str, int]:
        async with self._guard("staged_counts"), self._pool.acquire() as conn:
            if not await conn.fetchval(f"SELECT {_STAGED}"):
                return {}
            rows = await conn.fetch(
                "SELECT model, count(*) AS n FROM memory_staged_vectors WHERE corpus_id = $1 "
                "GROUP BY model",
                corpus_id,
            )
        return dict(sorted((r["model"], r["n"]) for r in rows))
