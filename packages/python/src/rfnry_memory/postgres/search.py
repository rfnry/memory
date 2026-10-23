import json
from collections.abc import Callable, Mapping
from datetime import datetime
from functools import partial
from typing import Any, NamedTuple, cast

from rfnry_memory.index import METHODS
from rfnry_memory.models import (
    Chunk,
    Kind,
    Leg,
    Match,
    Ref,
    Scope,
    Story,
    Window,
)
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.language import _QUERY_CONFIG
from rfnry_memory.postgres.records import _chunk, _story, _vector
from rfnry_memory.postgres.sql import _CHUNKS, _scope_sql, _subject_kinds, _table
from rfnry_memory.postgres.vectors import (
    _INDEXED,
    _PRECISIONS,
    _index_kind,
    _searched,
    _vectors_name,
)
from rfnry_memory.ranking import METHODS as RANKINGS, TextRanking

_ANY_TERM = (
    "(SELECT string_agg(quote_literal(l), ' | ')::tsquery "
    f"FROM unnest(tsvector_to_array(to_tsvector({_QUERY_CONFIG}, $2))) AS l)"
)


_ALL_TERMS = f"plainto_tsquery({_QUERY_CONFIG}, $2)"


_TERMS = f"to_tsvector({_QUERY_CONFIG}, $2)"


_MATCHES = {
    "any": _ANY_TERM,
    "all": _ALL_TERMS,
    "phrase": f"phraseto_tsquery({_QUERY_CONFIG}, $2)",
    "web": (
        f"(CASE WHEN querytree(websearch_to_tsquery({_QUERY_CONFIG}, $2)) <> 'T' "
        f"THEN websearch_to_tsquery({_QUERY_CONFIG}, $2) END)"
    ),
}


Row = Mapping[str, Any]


def _vector_sql(kind: Kind, where: str, limit: str) -> str:
    table, id_column = _table(kind)
    return (
        f"SELECT {id_column} AS id, 1 - (embedding <=> $2::vector) AS score FROM {table} "
        f"WHERE corpus_id = $1 AND embedding IS NOT NULL{where} "
        f'ORDER BY score DESC, {id_column} COLLATE "C" LIMIT {limit}'
    )


_RERANK = {"full": 1, "half": 1, "binary": 4}


class _Read(NamedTuple):
    precision: str
    dimension: int
    rerank: int


def _distance(read: _Read) -> str:
    dimension = read.dimension
    if read.precision == "half":
        return f"(v.vector::halfvec({dimension})) <=> $2::halfvec({dimension})"
    if read.precision == "binary":
        return (
            f"(binary_quantize(v.vector)::bit({dimension})) <~> "
            f"binary_quantize($2::vector)::bit({dimension})"
        )
    return "v.vector <=> $2::vector"


def _nearest_sql(table: str, read: _Read, kind: Kind, where: str, limit: str) -> str:
    passages, id_column = _table(kind)
    candidates = (
        f"SELECT v.passage_id AS id, p.score FROM {table} AS v CROSS JOIN LATERAL "
        f"(SELECT 1 - (embedding <=> $2::vector) AS score FROM {passages} "
        f"WHERE corpus_id = $1 AND {id_column} = v.passage_id AND embedding IS NOT NULL{where} OFFSET 0) AS p "
        f"WHERE v.passage_kind = '{kind}' ORDER BY {_distance(read)}"
    )
    if read.precision != "binary" and read.rerank == 1:
        return f"{candidates} LIMIT {limit}"
    return (
        f"SELECT id, score FROM ({candidates} LIMIT {limit} * {read.rerank}) AS candidates "
        f'ORDER BY score DESC, id COLLATE "C" LIMIT {limit}'
    )


def _text_sql(ranking: TextRanking, match: Match, kind: Kind, where: str, limit: str) -> str:
    table, id_column = _table(kind)
    return cast(
        str,
        RANKINGS[ranking.method].sql(
            ranking,
            table=table,
            id_column=id_column,
            matches=_MATCHES[match],
            ranks=_ALL_TERMS if match == "any" else _MATCHES[match],
            terms=_TERMS,
            where=where,
            limit=limit,
        ),
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
    ) -> list[Row]:
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
    ) -> list[tuple[Kind, Row]]:
        async with self._guard(operation), self._pool.acquire() as conn:
            return await self._search_kinds(
                conn, kinds, sql, base, filters, window, subject, scope, top_k
            )

    async def _search_kinds(
        self,
        conn: Any,
        kinds: tuple[Kind, ...],
        sql: Callable[[Kind, str, str], str],
        base: list[Any],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        scope: Scope | None,
        top_k: int,
    ) -> list[tuple[Kind, Row]]:
        found: list[tuple[Kind, Row]] = []
        for kind in _subject_kinds(kinds, subject):
            records = await self._search_kind(
                conn, kind, sql, base, filters, window, subject, scope, top_k
            )
            found.extend((kind, r) for r in records)
        return found

    async def _search_nearest(
        self,
        conn: Any,
        corpus_id: str,
        kinds: tuple[Kind, ...],
        base: list[Any],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        top_k: int,
    ) -> list[tuple[Kind, Row]] | None:
        index_table = _vectors_name(corpus_id)
        indexed = await conn.fetch(_INDEXED, index_table)
        reads: dict[str, tuple[_Read, list[tuple[str, str]]]] = {}
        for r in indexed:
            kind = _index_kind(index_table, r["name"])
            precision = _PRECISIONS.get(r["opclass"])
            if kind is None or precision is None:
                continue
            options = _searched(r["searched"])
            rerank = options.get("rerank") or _RERANK[precision]
            session = METHODS[r["method"]].session(options)
            reads[kind] = (_Read(precision, self._dimension or 0, rerank), session)
        searched = _subject_kinds(kinds, subject)
        if not reads.keys() & set(searched):
            return None
        found: list[tuple[Kind, Row]] = []
        try:
            async with conn.transaction(isolation="read_committed", readonly=True):
                for method in sorted({r["method"] for r in indexed}):
                    await conn.execute(METHODS[method].SEARCH)
                for kind in searched:
                    records: list[Row] = []
                    if kind in reads:
                        read, session = reads[kind]
                        for name, value in session:
                            await conn.execute("SELECT set_config($1, $2, true)", name, value)
                        nearest = partial(_nearest_sql, indexed[0]["index_table"], read)
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
        ranking: TextRanking,
        match: Match,
        scope: Scope | None = None,
    ) -> list[tuple[str, Leg]]:
        found = await self._search(
            "search_text",
            kinds,
            partial(_text_sql, ranking, match),
            [corpus_ids, query],
            filters,
            window,
            subject,
            scope,
            top_k,
        )
        found.sort(
            key=lambda hit: (-hit[1]["score"], -hit[1]["tie"], hit[1]["id"], hit[1]["corpus"])
        )
        hits: list[tuple[str, Leg]] = [
            (r["corpus"], (kind, r["id"], float(r["score"]))) for kind, r in found
        ]
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
                    conn, corpus_id, kinds, base, filters, window, subject, top_k
                )
            if found is None:
                found = await self._search_kinds(
                    conn, kinds, _vector_sql, base, filters, window, subject, scope, top_k
                )
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
                f"SELECT * FROM {_CHUNKS} WHERE corpus_id = $1 AND chunk_id = ANY($2::text[])",
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
                f"SELECT * FROM {_CHUNKS} WHERE corpus_id = $1 AND source_id = $2 "
                f"AND position BETWEEN $3 AND $4{seen} ORDER BY position",
                *params,
            )
        return [_chunk(r, window.known_at) for r in records]
