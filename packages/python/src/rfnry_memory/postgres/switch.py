import json
from collections import Counter
from typing import Any

from pydantic import BaseModel, TypeAdapter

from rfnry_memory.errors import ConfigurationError
from rfnry_memory.index import HNSW, IVFFlat, VectorIndex
from rfnry_memory.models import Chunk, Kind, Ref, Story, Window
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.locks import _language_lock
from rfnry_memory.postgres.records import _vector
from rfnry_memory.postgres.sql import _scope_sql, _table
from rfnry_memory.postgres.vectors import _KINDS, _STAGED, _built_index, _page


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


class CorpusVectors(BaseModel):
    corpus_id: str
    passages: int
    on_row: dict[str, int]
    staged: dict[str, int]
    staged_dimensions: dict[str, int]
    missing: dict[str, int]
    index: str | None
    to_build: str | None


def _index_text(index: HNSW | IVFFlat) -> str:
    shown = index.model_dump(exclude={"method"}, exclude_none=True)
    if shown.get("precision") == "full":
        del shown["precision"]
    settings = ", ".join(f"{name}={value}" for name, value in shown.items())
    return f"{type(index).__name__}({settings})"


def _to_build(recorded: str | None) -> str | None:
    if recorded is None:
        return None
    return _index_text(TypeAdapter(VectorIndex).validate_python(json.loads(recorded)["settings"]))


async def _on_the_row(conn: Any, corpus_id: str) -> Counter[str | None]:
    counted: Counter[str | None] = Counter()
    for kind in _KINDS:
        table, _ = _table(kind)
        for r in await conn.fetch(
            "SELECT CASE WHEN embedding IS NOT NULL THEN embedding_model END AS model, "
            f"count(*) AS n FROM {table} WHERE corpus_id = $1 GROUP BY 1",
            corpus_id,
        ):
            counted[r["model"]] += r["n"]
    return counted


async def _staged_by_model(conn: Any, corpus_id: str) -> dict[str, int]:
    rows = await conn.fetch(
        "SELECT model, count(*) AS n FROM memory_staged_vectors WHERE corpus_id = $1 GROUP BY model",
        corpus_id,
    )
    return dict(sorted((r["model"], r["n"]) for r in rows))


async def _staged_dimension(conn: Any, corpus_id: str, model: str) -> int:
    return int(
        await conn.fetchval(
            "SELECT vector_dims(vector) FROM memory_staged_vectors "
            "WHERE corpus_id = $1 AND model = $2 LIMIT 1",
            corpus_id,
            model,
        )
    )


async def _count_unstaged(
    conn: Any, corpus_id: str, model: str, filters: dict[str, Any] | None
) -> int:
    counted = 0
    for kind in _KINDS:
        table, key = _table(kind)
        params: list[Any] = [corpus_id, model]
        facets = _scope_sql(kind, filters, Window(), None, None, params)
        counted += await conn.fetchval(
            f"SELECT count(*) FROM {table} WHERE corpus_id = $1 "
            "AND embedding_model IS DISTINCT FROM $2 AND NOT EXISTS "
            "(SELECT 1 FROM memory_staged_vectors v WHERE v.corpus_id = $1 AND v.model = $2 "
            f"AND v.passage_kind = '{kind}' AND v.passage_id = {table}.{key}){facets}",
            *params,
        )
    return counted


async def _missing(
    conn: Any, corpus_id: str, model: str, on_row: Counter[str | None], staged: dict[str, int]
) -> int:
    if model not in staged:
        return on_row.total() - on_row[model]
    return await _count_unstaged(conn, corpus_id, model, None)


async def _corpus_vectors(
    conn: Any,
    corpus_id: str,
    recorded: str | None,
    on_row: Counter[str | None],
    staged: dict[str, int],
    models: list[str],
) -> CorpusVectors:
    index = await _built_index(conn, corpus_id)
    return CorpusVectors(
        corpus_id=corpus_id,
        passages=on_row.total(),
        on_row=dict(sorted((m, n) for m, n in on_row.items() if m is not None)),
        staged=staged,
        staged_dimensions={m: await _staged_dimension(conn, corpus_id, m) for m in staged},
        missing={m: await _missing(conn, corpus_id, m, on_row, staged) for m in models},
        index=None if index is None else _index_text(index),
        to_build=_to_build(recorded),
    )


async def vector_status(
    conn: Any, corpus_ids: list[str] | None, *, model: str | None = None
) -> list[CorpusVectors]:
    async with conn.transaction(isolation="repeatable_read", readonly=True):
        corpora = await conn.fetch(
            "SELECT corpus_id, vector_index::text AS recorded FROM memory_corpora "
            "WHERE $1::text[] IS NULL OR corpus_id = ANY($1::text[]) ORDER BY corpus_id",
            corpus_ids,
        )
        held = [
            (await _on_the_row(conn, r["corpus_id"]), await _staged_by_model(conn, r["corpus_id"]))
            for r in corpora
        ]
        seen = {m for on_row, staged in held for m in (*on_row, *staged) if m is not None}
        models = sorted(seen if model is None else seen | {model})
        return [
            await _corpus_vectors(conn, r["corpus_id"], r["recorded"], on_row, staged, models)
            for r, (on_row, staged) in zip(corpora, held, strict=True)
        ]


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
        async with self._guard("count_unstaged"), self._pool.acquire() as conn:
            await _stageable(conn)
            return await _count_unstaged(conn, corpus_id, model, filters)

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
            return await _staged_by_model(conn, corpus_id)
