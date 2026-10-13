from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from functools import partial
from typing import Any, cast

from rfnry_memory.models import (
    Chunk,
    Kind,
    Leg,
    Ref,
    Scope,
    Story,
    Window,
)
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.language import _QUERY_CONFIG
from rfnry_memory.postgres.records import _chunk, _story, _vector
from rfnry_memory.postgres.sql import _scope_sql, _subject_kinds, _table
from rfnry_memory.postgres.vectors import _INDEXED, _vectors_name

_ANY_TERM = (
    "(SELECT string_agg(quote_literal(l), ' | ')::tsquery "
    f"FROM unnest(tsvector_to_array(to_tsvector({_QUERY_CONFIG}, $2))) AS l)"
)


_K1 = 1.5


_B = 0.75


def _vector_sql(kind: Kind, scope: str, limit: str) -> str:
    table, id_column = _table(kind)
    return (
        f"SELECT {id_column} AS id, 1 - (embedding <=> $2::vector) AS score FROM {table} "
        f"WHERE corpus_id = $1 AND embedding IS NOT NULL{scope} "
        f'ORDER BY score DESC, {id_column} COLLATE "C" LIMIT {limit}'
    )


def _nearest_sql(table: str, half: int | None, kind: Kind, scope: str, limit: str) -> str:
    passages, id_column = _table(kind)
    order = (
        "v.vector <=> $2::vector"
        if half is None
        else f"(v.vector::halfvec({half})) <=> $2::halfvec({half})"
    )
    return (
        f"SELECT v.passage_id AS id, p.score FROM {table} AS v CROSS JOIN LATERAL "
        f"(SELECT 1 - (embedding <=> $2::vector) AS score FROM {passages} "
        f"WHERE corpus_id = $1 AND {id_column} = v.passage_id AND embedding IS NOT NULL{scope} OFFSET 0) AS p "
        f"WHERE v.passage_kind = '{kind}' ORDER BY {order} LIMIT {limit}"
    )


def _bm25_sql(kind: Kind, scope: str, limit: str) -> str:
    table, id_column = _table(kind)
    return (
        f"WITH matched AS (SELECT corpus_id AS corpus, {id_column} AS id, fts FROM {table} "
        f"WHERE corpus_id = ANY($1::text[]) AND fts @@ {_ANY_TERM}{scope}), "
        "terms AS (SELECT DISTINCT l AS lexeme "
        f"FROM unnest(tsvector_to_array(to_tsvector({_QUERY_CONFIG}, $2))) AS l), "
        "sized AS (SELECT m.corpus, m.id, (SELECT sum(coalesce(array_length(u.positions, 1), 1)) "
        "FROM unnest(m.fts) AS u)::float8 AS len FROM matched AS m), "
        "freq AS (SELECT m.corpus, m.id, u.lexeme, "
        "coalesce(array_length(u.positions, 1), 1)::float8 AS tf "
        "FROM matched AS m CROSS JOIN LATERAL unnest(m.fts) AS u "
        "WHERE u.lexeme IN (SELECT lexeme FROM terms)), "
        "stats AS (SELECT count(*)::float8 AS n, avg(len) AS avgdl FROM sized), "
        "df AS (SELECT lexeme, count(*)::float8 AS df FROM freq GROUP BY lexeme) "
        f"SELECT f.corpus, f.id, sum(ln(1 + (stats.n - df.df + 0.5) / (df.df + 0.5)) * f.tf * {_K1 + 1} "
        f"/ (f.tf + {_K1} * (1 - {_B} + {_B} * sized.len / stats.avgdl))) AS score "
        "FROM freq AS f JOIN df ON df.lexeme = f.lexeme "
        "JOIN sized ON sized.corpus = f.corpus AND sized.id = f.id CROSS JOIN stats "
        "GROUP BY f.corpus, f.id "
        f'ORDER BY score DESC, f.id COLLATE "C", f.corpus COLLATE "C" LIMIT {limit}'
    )


class Search(Base):
    async def _search_kind(
        self,
        conn: Any,
        kind: Kind,
        sql: Callable[[Kind, str, str], str],
        base: list[Any],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        scope: Scope | None,
        top_k: int,
    ) -> list[Any]:
        params = list(base)
        clause = _scope_sql(kind, filters, window, subject, scope, params)
        params.append(top_k)
        return list(await conn.fetch(sql(kind, clause, f"${len(params)}"), *params))

    async def _search(
        self,
        operation: str,
        kinds: tuple[Kind, ...],
        sql: Callable[[Kind, str, str], str],
        base: list[Any],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        scope: Scope | None,
        top_k: int,
    ) -> list[tuple[Kind, Any]]:
        found: list[tuple[Kind, Any]] = []
        async with self._guard(operation), self._pool.acquire() as conn:
            for kind in _subject_kinds(kinds, subject):
                records = await self._search_kind(
                    conn, kind, sql, base, filters, window, subject, scope, top_k
                )
                found.extend((kind, r) for r in records)
        return found

    async def _search_nearest(
        self,
        conn: Any,
        kinds: tuple[Kind, ...],
        base: list[Any],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        top_k: int,
    ) -> list[tuple[Kind, Any]] | None:
        indexed = await conn.fetch(_INDEXED, _vectors_name(base[0]))
        halves = {r["kind"]: r["half"] for r in indexed}
        searched = _subject_kinds(kinds, subject)
        if not halves.keys() & set(searched):
            return None
        found: list[tuple[Kind, Any]] = []
        try:
            async with conn.transaction(isolation="read_committed", readonly=True):
                await conn.execute(
                    "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"
                )
                for kind in searched:
                    records: list[Any] = []
                    if kind in halves:
                        half = self._dimension if halves[kind] else None
                        nearest = partial(_nearest_sql, indexed[0]["vectors"], half)
                        records = await self._search_kind(
                            conn, kind, nearest, base, filters, window, subject, None, top_k
                        )
                    if len(records) < top_k:
                        records = await self._search_kind(
                            conn, kind, _vector_sql, base, filters, window, subject, None, top_k
                        )
                    found.extend((kind, r) for r in records)
        except Exception as exc:
            if getattr(exc, "sqlstate", None) != "42P01":
                raise
            return None
        return found

    async def search_text(
        self,
        corpus_ids: list[str],
        query: str,
        *,
        kinds: tuple[Kind, ...],
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        scope: Scope | None = None,
    ) -> list[tuple[str, Leg]]:
        found = await self._search(
            "search_text",
            kinds,
            _bm25_sql,
            [corpus_ids, query],
            filters,
            window,
            subject,
            scope,
            top_k,
        )
        hits: list[tuple[str, Leg]] = [
            (r["corpus"], (kind, r["id"], float(r["score"]))) for kind, r in found
        ]
        hits.sort(key=lambda hit: (-hit[1][2], hit[1][1], hit[0]))
        return hits[:top_k]

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
        scope: Scope | None = None,
    ) -> list[Leg]:
        self._check_vectors(vector)
        base = [corpus_id, _vector(vector)]
        found = None
        async with self._guard("search_vector"), self._pool.acquire() as conn:
            if scope is None:
                found = await self._search_nearest(
                    conn, kinds, base, filters, window, subject, top_k
                )
            if found is None:
                found = []
                for kind in _subject_kinds(kinds, subject):
                    records = await self._search_kind(
                        conn, kind, _vector_sql, base, filters, window, subject, scope, top_k
                    )
                    found.extend((kind, r) for r in records)
        legs: list[Leg] = [(kind, r["id"], float(r["score"])) for kind, r in found]
        legs.sort(key=lambda leg: (-leg[2], leg[1]))
        return legs[:top_k]

    async def get_passages(
        self, corpus_id: str, refs: list[Ref], *, known_at: datetime | None = None
    ) -> list[Chunk | Story]:
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
        found: dict[Ref, Chunk | Story] = {
            ("chunk", r["chunk_id"]): _chunk(r, known_at) for r in chunks
        }
        found.update({("story", r["story_id"]): _story(r, known_at) for r in stories})
        return [found[ref] for ref in refs if ref in found]

    async def passage_vector(self, corpus_id: str, passage: Ref) -> list[float] | None:
        if not self._dimension:
            return None
        kind, passage_id = passage
        table, id_column = _table(cast(Kind, kind))
        async with self._guard("passage_vector"), self._pool.acquire() as conn:
            value = await conn.fetchval(
                f"SELECT embedding::text FROM {table} WHERE corpus_id = $1 AND {id_column} = $2",
                corpus_id,
                passage_id,
            )
        return None if value is None else [float(x) for x in json.loads(value)]

    async def chunks_around(
        self,
        corpus_id: str,
        source_id: str,
        position: int,
        *,
        width: int,
        window: Window,
        filters: dict[str, Any] | None,
    ) -> list[Chunk]:
        params: list[Any] = [corpus_id, source_id, position - width, position + width]
        seen = _scope_sql("chunk", filters, window, None, None, params)
        async with self._guard("chunks_around"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_chunks WHERE corpus_id = $1 AND source_id = $2 "
                f"AND position BETWEEN $3 AND $4{seen} ORDER BY position",
                *params,
            )
        return [_chunk(r, window.known_at) for r in records]
