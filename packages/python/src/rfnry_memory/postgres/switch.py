import json
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, TypeAdapter

from rfnry_memory.errors import ConfigurationError, InputError, StorageError
from rfnry_memory.index import HNSW, IVFFlat, VectorIndex
from rfnry_memory.models import Chunk, Kind, Ref, Story, Window
from rfnry_memory.postgres import migrations
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.locks import _language_lock, lock_key
from rfnry_memory.postgres.migrations import _literal, _retyped
from rfnry_memory.postgres.records import _count, _vector
from rfnry_memory.postgres.sql import _scope_sql, _table
from rfnry_memory.postgres.vectors import (
    _FULL_DIMENSIONS,
    _HALF_DIMENSIONS,
    _KINDS,
    _STAGED,
    _built_index,
    _page,
    _vectors_name,
)

_NO_NESTED_LOOPS = "SET LOCAL enable_nestloop = off"


def _held(table: str) -> str:
    return f"CASE WHEN {table}.embedding IS NOT NULL THEN {table}.embedding_model END"


def _not_held(table: str, named: str) -> str:
    return f"({_held(table)}) IS DISTINCT FROM {named}"


def _unstaged(kind: Kind) -> str:
    table, key = _table(kind)
    return (
        f"{_not_held(table, '$2')} AND NOT (SELECT EXISTS (SELECT 1 FROM memory_staged_vectors v "
        f"WHERE v.corpus_id = {table}.corpus_id AND v.model = $2 "
        f"AND v.passage_kind = '{kind}' AND v.passage_id = {table}.{key}))"
    )


_UNSTAGED: dict[Kind, str] = {kind: _unstaged(kind) for kind in _KINDS}


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


def _recorded_index(recorded: str) -> HNSW | IVFFlat:
    return TypeAdapter(VectorIndex).validate_python(json.loads(recorded)["settings"])


def _recorded_owner(recorded: str) -> str:
    return str(json.loads(recorded)["owner"])


def _to_build(recorded: str | None) -> str | None:
    return None if recorded is None else _index_text(_recorded_index(recorded))


async def _on_the_row(conn: Any, corpus_id: str) -> Counter[str | None]:
    counted: Counter[str | None] = Counter()
    for kind in _KINDS:
        table, _ = _table(kind)
        for r in await conn.fetch(
            f"SELECT {_held(table)} AS model, count(*) AS n FROM {table} "
            "WHERE corpus_id = $1 GROUP BY 1",
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


@asynccontextmanager
async def _no_nested_loops(conn: Any) -> AsyncIterator[None]:
    async with conn.transaction():
        await conn.execute(_NO_NESTED_LOOPS)
        yield


async def _count_unstaged(
    conn: Any, corpus_id: str, model: str, filters: dict[str, Any] | None
) -> int:
    counted = 0
    async with _no_nested_loops(conn):
        for kind in _KINDS:
            table, key = _table(kind)
            params: list[Any] = [corpus_id, model]
            facets = _scope_sql(kind, filters, Window(), None, None, params)
            counted += await conn.fetchval(
                f"SELECT count(*) FROM {table} WHERE corpus_id = $1 "
                f"AND {_not_held(table, '$2')} AND NOT EXISTS "
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


class SwapPlan(NamedTuple):
    steps: list[str]
    statements: list[str]
    refusals: list[str]
    corpora: list[str]


class _Dropped(NamedTuple):
    corpus_id: str | None
    table: str
    owner: str
    droppable: bool
    settings: HNSW | IVFFlat | None


class _Swap(NamedTuple):
    model: str
    held: int
    dimension: int
    corpora: list[str]
    counted: list[str] | None
    dropped: list[_Dropped]
    index_state: str


NAMED = 5


_CHANGED = "the schema changed since the plan was shown — run the command again"


_MIGRATING = "another command is changing this schema — run the command again when that ends"


_HOLD_PASSAGES = "LOCK TABLE memory_chunks, memory_stories IN SHARE ROW EXCLUSIVE MODE"


_STAGED_SIZES = (
    "SELECT corpus_id, vector_dims(vector) AS dimension, count(*) AS n "
    "FROM memory_staged_vectors WHERE model = $1 "
    "AND ($2::text[] IS NULL OR corpus_id = ANY($2::text[])) GROUP BY 1, 2"
)


_INDEX_TABLES = (
    "SELECT tablename AS name, format('%I.%I', schemaname, tablename) AS qualified, "
    "tableowner AS owner, pg_has_role(current_user, tableowner, 'USAGE') OR pg_has_role("
    "current_user, (SELECT nspowner FROM pg_namespace WHERE nspname = schemaname), 'USAGE') "
    "AS droppable FROM pg_tables WHERE schemaname = current_schema() "
    "AND starts_with(tablename, 'memory_vectors_') ORDER BY tablename"
)


_ROLES = (
    "SELECT current_user AS role, pg_get_userbyid(nspowner) AS schema_owner FROM pg_namespace "
    "WHERE nspname = current_schema()"
)


_NOT_OWNED = (
    "SELECT tablename, tableowner FROM pg_tables WHERE schemaname = current_schema() "
    "AND tablename IN ('memory_chunks', 'memory_stories') "
    "AND NOT pg_has_role(current_user, tableowner, 'USAGE') ORDER BY tablename"
)


def _named(corpora: list[str]) -> str:
    return ", ".join(corpora) if len(corpora) <= NAMED else f"{len(corpora):,} corpora"


def _lack(count: int) -> str:
    return f"{count:,} passage lacks" if count == 1 else f"{count:,} passages lack"


def _hold(count: int) -> str:
    return f"{count:,} passage holds" if count == 1 else f"{count:,} passages hold"


def _within(alias: str, corpora: list[str] | None) -> str:
    if corpora is None:
        return ""
    return f" AND {alias}.corpus_id IN ({', '.join(_literal(c) for c in corpora)})"


def _staged_for(kind: Kind, named: str) -> str:
    _, key = _table(kind)
    return (
        "EXISTS (SELECT 1 FROM memory_staged_vectors v WHERE v.corpus_id = p.corpus_id "
        f"AND v.model = {named} AND v.passage_kind = '{kind}' AND v.passage_id = p.{key})"
    )


def _replaced(kind: Kind, named: str, resized: bool) -> str:
    return "" if resized else f" AND {_staged_for(kind, named)}"


def _lacking_sql(kind: Kind, named: str, corpora: list[str] | None) -> str:
    table, _ = _table(kind)
    return (
        f"FROM {table} p WHERE {_not_held('p', named)}{_within('p', corpora)} "
        f"AND NOT {_staged_for(kind, named)}"
    )


def _unnamed_sql(kind: Kind, named: str, corpora: list[str] | None, resized: bool) -> str:
    table, _ = _table(kind)
    return (
        f"SELECT count(*) FROM {table} p WHERE p.embedding IS NOT NULL "
        f"AND p.embedding_model IS NULL{_within('p', corpora)}{_replaced(kind, named, resized)}"
    )


def _keep_sql(kind: Kind, named: str, corpora: list[str] | None, resized: bool) -> str:
    table, key = _table(kind)
    return (
        "INSERT INTO memory_staged_vectors "
        "(corpus_id, model, passage_kind, passage_id, vector, settings) "
        f"SELECT p.corpus_id, p.embedding_model, '{kind}', p.{key}, p.embedding, "
        f"p.embedding_settings FROM {table} p WHERE p.embedding IS NOT NULL "
        f"AND p.embedding_model <> {named}{_within('p', corpora)}"
        f"{_replaced(kind, named, resized)} "
        "ON CONFLICT (corpus_id, model, passage_kind, passage_id) DO UPDATE SET "
        "vector = EXCLUDED.vector, settings = EXCLUDED.settings"
    )


def _move_sql(kind: Kind, named: str, corpora: list[str] | None) -> str:
    table, key = _table(kind)
    return (
        f"UPDATE {table} p SET embedding = v.vector, embedding_model = v.model, "
        "embedding_settings = v.settings FROM memory_staged_vectors v "
        f"WHERE v.corpus_id = p.corpus_id AND v.model = {named} AND v.passage_kind = '{kind}' "
        f"AND v.passage_id = p.{key} AND {_not_held('p', named)}"
        f"{_within('p', corpora)}{_within('v', corpora)}"
    )


def _clear_sql(kind: Kind) -> str:
    table, _ = _table(kind)
    return (
        f"UPDATE {table} SET embedding_model = NULL, embedding_settings = NULL "
        "WHERE embedding IS NULL "
        "AND (embedding_model IS NOT NULL OR embedding_settings IS NOT NULL)"
    )


def _lock_sql(name: str) -> str:
    return f"SELECT pg_advisory_xact_lock({lock_key(_literal(name))})"


def _try_lock_sql(key: str, held: str) -> str:
    return "DO " + _literal(
        f"BEGIN IF NOT pg_try_advisory_xact_lock({key}) THEN "
        f"RAISE EXCEPTION USING MESSAGE = {_literal(held)}; END IF; END"
    )


def _index_state_sql(corpora: list[str] | None) -> str:
    tables = (
        "starts_with(t.relname, 'memory_vectors_')"
        if corpora is None
        else f"t.relname IN ({', '.join(_literal(_vectors_name(c)) for c in corpora)})"
    )
    return (
        "(SELECT encode(sha256(convert_to(coalesce(string_agg(concat_ws('/', t.relname, c.relname, "
        "i.indisvalid, obj_description(c.oid, 'pg_class')), ',' ORDER BY t.relname, c.relname), "
        "''), 'UTF8')), 'hex') FROM pg_class t JOIN pg_namespace n ON n.oid = t.relnamespace "
        "LEFT JOIN pg_index i ON i.indrelid = t.oid LEFT JOIN pg_class c ON c.oid = i.indexrelid "
        f"WHERE n.nspname = current_schema() AND t.relkind = 'r' AND {tables})"
    )


def _unchanged_sql(held: int, corpora: list[str] | None, index_state: str) -> str:
    return "DO " + _literal(
        f"BEGIN IF (SELECT dimension FROM memory_settings) IS DISTINCT FROM {held} "
        f"OR {_index_state_sql(corpora)} <> {_literal(index_state)} "
        f"THEN RAISE EXCEPTION USING MESSAGE = {_literal(_CHANGED)}; END IF; END"
    )


def _none_lacking_sql(model: str, corpora: list[str] | None) -> str:
    named = _literal(model)
    chunks, stories = (f"(SELECT count(*) {_lacking_sql(k, named, corpora)})" for k in _KINDS)
    fix = " — written after the plan; fill them with rfnry-memory vectors stage, or pass --allow-missing"
    return "DO " + _literal(
        f"DECLARE lacking bigint := {chunks} + {stories}; BEGIN IF lacking > 0 THEN "
        f"RAISE EXCEPTION USING MESSAGE = {_literal(f'passages lacking {model}: ')} "
        f"|| lacking || {_literal(fix)}; END IF; END"
    )


def _record_sql(corpus_id: str, settings: HNSW | IVFFlat, owner: str) -> str:
    recorded = json.dumps({"settings": settings.model_dump(), "owner": owner})
    return (
        f"UPDATE memory_corpora SET vector_index = {_literal(recorded)}::jsonb "
        f"WHERE corpus_id = {_literal(corpus_id)}"
    )


async def _lacking(conn: Any, model: str, corpora: list[str] | None) -> Counter[str]:
    named = _literal(model)
    counted: Counter[str] = Counter()
    async with _no_nested_loops(conn):
        for kind in _KINDS:
            for r in await conn.fetch(
                f"SELECT p.corpus_id, count(*) AS n {_lacking_sql(kind, named, corpora)} GROUP BY 1"
            ):
                counted[r["corpus_id"]] += r["n"]
    return counted


async def _unnamed(conn: Any, model: str, corpora: list[str] | None, resized: bool) -> int:
    named = _literal(model)
    async with _no_nested_loops(conn):
        return sum(
            [await conn.fetchval(_unnamed_sql(kind, named, corpora, resized)) for kind in _KINDS]
        )


async def _holding(conn: Any, model: str) -> int:
    counted = 0
    for kind in _KINDS:
        table, _ = _table(kind)
        counted += await conn.fetchval(
            f"SELECT count(*) FROM {table} WHERE embedding IS NOT NULL AND embedding_model = $1",
            model,
        )
    return counted


async def _index_tables(conn: Any, corpora: list[str], resized: bool) -> list[_Dropped]:
    tables = {r["name"]: r for r in await conn.fetch(_INDEX_TABLES)}
    found: list[_Dropped] = []
    for corpus_id in corpora:
        table = tables.pop(_vectors_name(corpus_id), None)
        if table is not None:
            settings = await _built_index(conn, corpus_id)
            found.append(
                _Dropped(
                    corpus_id, table["qualified"], table["owner"], table["droppable"], settings
                )
            )
    if resized:
        found.extend(
            _Dropped(None, r["qualified"], r["owner"], r["droppable"], None)
            for r in tables.values()
        )
    return found


def _dropped_step(dropped: _Dropped) -> str:
    if dropped.corpus_id is None:
        return f"{dropped.table}, the index table of a corpus that holds nothing, goes"
    if dropped.settings is None:
        return f"{dropped.corpus_id}: an index table with no finished index goes"
    return f"{dropped.corpus_id}: {_index_text(dropped.settings)} goes, owned by {dropped.owner}"


def _rebuild_steps(owners: dict[str, str], role: str, no_index: bool) -> list[str]:
    building = [] if no_index else sorted(c for c, owner in owners.items() if owner == role)
    left: dict[str, list[str]] = {}
    for corpus_id, owner in sorted(owners.items()):
        if corpus_id not in building:
            left.setdefault(owner, []).append(corpus_id)
    return [
        *(
            [
                f"the index of {_named(building)} builds again after the swap, "
                "and searches there run exactly until it ends"
            ]
            if building
            else []
        ),
        *(
            f"the index of {_named(corpora)} is left to rfnry-memory vectors index"
            f"{'' if owner == role else f', run as {owner}'}, "
            "and searches there run exactly until it is built"
            for owner, corpora in sorted(left.items())
        ),
    ]


def _unfit(indexes: dict[str, HNSW | IVFFlat], dimension: int) -> list[str]:
    if dimension <= _HALF_DIMENSIONS:
        return []
    return sorted(c for c, index in indexes.items() if index.precision != "binary")


def _halved(indexes: dict[str, HNSW | IVFFlat], dimension: int) -> list[str]:
    if not _FULL_DIMENSIONS < dimension <= _HALF_DIMENSIONS:
        return []
    return sorted(c for c, index in indexes.items() if index.precision == "full")


def _swap_steps(
    swap: _Swap,
    staged: int,
    missing: Counter[str],
    indexes: dict[str, HNSW | IVFFlat],
    rebuilds: list[str],
) -> list[str]:
    model, dimension, resized = swap.model, swap.dimension, swap.dimension != swap.held
    lacking = _lack(missing.total())
    halved = _halved(indexes, dimension)
    unfit = _unfit(indexes, dimension)
    return [
        *([f"vectors of {dimension} dimensions: the whole schema moves"] if resized else []),
        f"{staged:,} vectors of {model} go onto the rows, in {_named(swap.corpora)}",
        "every vector on a row of another model stays staged under that model"
        if resized
        else "the vectors they replace stay staged under their own models",
        *(
            [
                f"{lacking} {model}: no vector on the row, and what they hold stays staged"
                if resized
                else f"{lacking} {model} and keep what they have: absent from its vector search"
            ]
            if missing
            else []
        ),
        *(_dropped_step(dropped) for dropped in swap.dropped),
        *rebuilds,
        *(
            [
                f"the index of {_named(halved)} builds at half precision: full holds "
                f"{_FULL_DIMENSIONS} dimensions, and {model} has {dimension}"
            ]
            if halved
            else []
        ),
        *(
            [
                f"the index of {_named(unfit)} reaches {_HALF_DIMENSIONS} dimensions: "
                "rfnry-memory vectors index --precision binary builds it"
            ]
            if unfit
            else []
        ),
        f"reads and writes wait until the swap ends; after it a Memory of another model fails "
        f"check(), so the app must start on {model}"
        if resized
        else "writes wait until the swap ends; reads go on and see the old vectors; after it "
        "a Memory of another model finds nothing by vector in these corpora",
    ]


def _swap_statements(swap: _Swap, schema: str, allow_missing: bool) -> list[str]:
    resized = swap.dimension != swap.held
    named = _literal(swap.model)
    scope = None if resized else swap.corpora
    return [
        _NO_NESTED_LOOPS,
        _try_lock_sql(migrations.MIGRATING, _MIGRATING),
        _try_lock_sql(_staging_key(swap.model), _staging_runs(swap.model)),
        *(
            _try_lock_sql(
                lock_key(_literal(f"memory.vectors:{d.corpus_id}")),
                f"the vector index of {d.corpus_id} is being built or dropped — "
                "run the command again when that ends",
            )
            for d in swap.dropped
            if d.corpus_id
        ),
        *(_lock_sql(f"memory.language:{corpus_id}") for corpus_id in swap.corpora),
        _HOLD_PASSAGES,
        _unchanged_sql(swap.held, scope, swap.index_state),
        *([] if allow_missing else [_none_lacking_sql(swap.model, swap.counted)]),
        *(_keep_sql(kind, named, scope, resized) for kind in _KINDS),
        *(
            [
                *(_retyped(schema, _table(kind)[0], swap.dimension) for kind in _KINDS),
                f"UPDATE memory_settings SET dimension = {swap.dimension}",
            ]
            if resized
            else []
        ),
        *(_move_sql(kind, named, scope) for kind in _KINDS),
        *(_clear_sql(kind) for kind in _KINDS if resized),
        f"DELETE FROM memory_staged_vectors v WHERE v.model = {named}{_within('v', scope)}",
        "ANALYZE memory_chunks, memory_stories, memory_staged_vectors",
        *(
            _record_sql(d.corpus_id, d.settings, d.owner)
            for d in swap.dropped
            if d.corpus_id and d.settings
        ),
        *(f"DROP TABLE {d.table}" for d in swap.dropped),
    ]


def _role_refusals(swap: _Swap, role: str, schema_owner: str, not_owned: list[Any]) -> list[str]:
    refusals: list[str] = []
    if not_owned:
        tables = " and ".join(r["tablename"] for r in not_owned)
        owners = " or ".join(sorted({r["tableowner"] for r in not_owned}))
        refusals.append(
            f"{role} does not own {tables}, which a new dimension retypes — run this as {owners}"
        )
    refused = [d for d in swap.dropped if not d.droppable]
    if refused:
        names = _named(sorted(d.corpus_id or d.table for d in refused))
        roles = " or ".join(sorted({schema_owner, *(d.owner for d in refused)}))
        refusals.append(f"{role} cannot drop the vector index of {names} — run this as {roles}")
    return refusals


def _swap_refusals(
    swap: _Swap,
    every: set[str],
    corpus_ids: list[str] | None,
    holding: int,
    unnamed: int,
    missing: Counter[str],
    unfit: list[str],
) -> list[str]:
    model, dimension, held = swap.model, swap.dimension, swap.held
    refusals: list[str] = []
    if dimension != held and corpus_ids is not None and every - set(corpus_ids):
        refusals.append(
            f"{model} is staged at {dimension} dimensions and the schema holds {held}: "
            "a new dimension moves every corpus at once — leave out --corpus"
        )
    if holding:
        refusals.append(
            f"{_hold(holding)} {model} at {held} dimensions and it is staged at {dimension} — "
            "drop what is staged with rfnry-memory vectors forget, give the rows another name "
            "with rfnry-memory vectors rename, then stage it again"
        )
    if unnamed:
        its = "its" if unnamed == 1 else "their"
        refusals.append(
            f"{_hold(unnamed)} a vector with no model named, which this swap would lose — "
            f"set {its} embedding_model in SQL, or clear {its} embedding"
        )
    if missing:
        refusals.append(
            f"{_lack(missing.total())} a vector of {model}, in {_named(sorted(missing))} — "
            "fill them with rfnry-memory vectors stage, or pass --allow-missing to leave them "
            "as they are"
        )
    if unfit:
        refusals.append(
            f"the index of {_named(unfit)} reaches {_HALF_DIMENSIONS} dimensions and {model} has "
            f"{dimension} — pass --no-index, then rfnry-memory vectors index --precision binary "
            "builds it"
        )
    return refusals


def _refused(refusal: str) -> SwapPlan:
    return SwapPlan([], [], [refusal], [])


async def plan_swap(
    conn: Any, *, model: str, corpus_ids: list[str] | None, allow_missing: bool, no_index: bool
) -> SwapPlan:
    if max(await migrations._applied(conn), default=0) != migrations.LATEST:
        return _refused(
            "the schema is not at this rfnry-memory's version — run rfnry-memory upgrade first"
        )
    held = await conn.fetchval("SELECT dimension FROM memory_settings")
    if held is None:
        return _refused("the schema holds no vectors — run rfnry-memory set --dimension N")
    staged = await conn.fetch(_STAGED_SIZES, model, corpus_ids)
    if not staged:
        return _refused(
            f"nothing is staged for {model} — fill it with rfnry-memory vectors stage, "
            "or check the name with rfnry-memory vectors status"
        )
    sizes = sorted({r["dimension"] for r in staged})
    if len(sizes) > 1:
        return _refused(
            f"{model} is staged at {' and '.join(str(size) for size in sizes)} dimensions, and a "
            "model has one size — rfnry-memory vectors forget drops every staged vector of it, "
            "a rollback copy included, so be sure it can be embedded again, then stage it"
        )
    dimension = sizes[0]
    resized = dimension != held
    recorded = {
        r["corpus_id"]: r["recorded"]
        for r in await conn.fetch(
            "SELECT corpus_id, vector_index::text AS recorded FROM memory_corpora"
        )
    }
    corpora = sorted(recorded) if resized else sorted({r["corpus_id"] for r in staged})
    counted = None if resized else corpus_ids
    scope = None if resized else corpora
    index_state = await conn.fetchval(f"SELECT {_index_state_sql(scope)}")
    swap = _Swap(
        model,
        held,
        dimension,
        corpora,
        counted,
        await _index_tables(conn, corpora, resized),
        index_state,
    )
    missing = await _lacking(conn, model, counted)
    indexes = {
        **{c: _recorded_index(recorded[c]) for c in corpora if recorded.get(c)},
        **{d.corpus_id: d.settings for d in swap.dropped if d.corpus_id and d.settings},
    }
    owners = {
        **{c: _recorded_owner(recorded[c]) for c in corpora if recorded.get(c)},
        **{d.corpus_id: d.owner for d in swap.dropped if d.corpus_id and d.settings},
    }
    roles = await conn.fetchrow(_ROLES)
    refusals = [
        *_role_refusals(
            swap,
            roles["role"],
            roles["schema_owner"],
            await conn.fetch(_NOT_OWNED) if resized else [],
        ),
        *_swap_refusals(
            swap,
            set(recorded),
            corpus_ids,
            await _holding(conn, model) if resized else 0,
            await _unnamed(conn, model, scope, resized),
            Counter() if allow_missing else missing,
            [] if no_index else _unfit(indexes, dimension),
        ),
    ]
    if refusals:
        return SwapPlan([], [], refusals, [])
    schema = await conn.fetchval("SELECT quote_ident(current_schema())")
    return SwapPlan(
        _swap_steps(
            swap,
            sum(r["n"] for r in staged),
            missing,
            indexes,
            _rebuild_steps(owners, roles["role"], no_index),
        ),
        _swap_statements(swap, schema, allow_missing),
        [],
        corpora,
    )


class UnknownCommitError(StorageError):
    pass


async def _in_one_transaction(conn: Any, statements: list[str], noun: str) -> list[str]:
    tags: list[str] = []
    committing = False
    try:
        async with conn.transaction(isolation="read_committed"):
            for statement in statements:
                tags.append(await conn.execute(statement))
            committing = True
    except BaseException as exc:
        if committing:
            raise UnknownCommitError(
                f"postgres: the {noun} stopped during its commit, so whether it landed is "
                "unknown — rfnry-memory vectors status shows which"
            ) from exc
        if not isinstance(exc, Exception):
            raise
        if getattr(exc, "sqlstate", None) == "P0001":
            raise ConfigurationError(f"postgres: {exc}") from None
        raise StorageError(f"postgres: the {noun} rolled back, nothing changed: {exc}") from exc
    return tags


async def apply_swap(conn: Any, plan: SwapPlan) -> None:
    if plan.refusals:
        raise InputError(f"postgres: {plan.refusals[0]}")
    await _in_one_transaction(conn, plan.statements, "swap")


class IndexPlan(NamedTuple):
    builds: dict[str, HNSW | IVFFlat]
    others: dict[str, list[str]]
    unfit: list[str]
    dimension: int


def _fitted(
    index: HNSW | IVFFlat, dimension: int, precision: Literal["full", "half", "binary"] | None
) -> HNSW | IVFFlat:
    if precision is not None:
        return index.model_copy(update={"precision": precision})
    if index.precision == "full" and dimension > _FULL_DIMENSIONS:
        return index.model_copy(update={"precision": "half"})
    return index


async def plan_index(
    conn: Any, corpus_ids: list[str], precision: Literal["full", "half", "binary"] | None
) -> IndexPlan:
    role = await conn.fetchval("SELECT current_user")
    dimension = await conn.fetchval("SELECT dimension FROM memory_settings")
    builds: dict[str, HNSW | IVFFlat] = {}
    others: dict[str, list[str]] = {}
    for r in await conn.fetch(
        "SELECT corpus_id, vector_index::text AS recorded FROM memory_corpora "
        "WHERE vector_index IS NOT NULL AND corpus_id = ANY($1::text[]) ORDER BY corpus_id",
        corpus_ids,
    ):
        owner = _recorded_owner(r["recorded"])
        if owner == role:
            builds[r["corpus_id"]] = _fitted(_recorded_index(r["recorded"]), dimension, precision)
        else:
            others.setdefault(owner, []).append(r["corpus_id"])
    return IndexPlan(builds, others, _unfit(builds, dimension), dimension)


_FORGET_BATCH = 10_000


_FORGET = (
    "DELETE FROM memory_staged_vectors WHERE ctid = ANY(ARRAY("
    "SELECT ctid FROM memory_staged_vectors WHERE model = $1 "
    "AND corpus_id = ANY($2::text[]) LIMIT $3))"
)


def _staging_key(model: str) -> str:
    return lock_key(_literal(f"memory.staging:{model}"))


def _staging_runs(model: str) -> str:
    return (
        f"a stage of {model} is running, and it would go on writing under that name — "
        "stop it, then run the command again"
    )


async def hold_staging(conn: Any, model: str) -> None:
    if not await conn.fetchval(f"SELECT pg_try_advisory_lock_shared({_staging_key(model)})"):
        raise ConfigurationError(
            f"postgres: another command is changing the label {model} — "
            "run the command again when that ends"
        )


@asynccontextmanager
async def _session_lock(conn: Any, key: str, held: str) -> AsyncIterator[None]:
    if not await conn.fetchval(f"SELECT pg_try_advisory_lock({key})"):
        raise ConfigurationError(f"postgres: {held}")
    try:
        yield
    finally:
        await conn.execute(f"SELECT pg_advisory_unlock({key})")


async def staged_of(conn: Any, model: str, corpus_ids: list[str]) -> dict[str, int]:
    rows = await conn.fetch(
        "SELECT corpus_id, count(*) AS n FROM memory_staged_vectors "
        "WHERE model = $1 AND corpus_id = ANY($2::text[]) GROUP BY 1 ORDER BY 1",
        model,
        corpus_ids,
    )
    return {r["corpus_id"]: r["n"] for r in rows}


async def forget_staged(conn: Any, *, model: str, corpus_ids: list[str]) -> int:
    forgotten = 0
    async with (
        _session_lock(conn, migrations.MIGRATING, _MIGRATING),
        _session_lock(conn, _staging_key(model), _staging_runs(model)),
    ):
        while deleted := _count(await conn.execute(_FORGET, model, corpus_ids, _FORGET_BATCH)):
            forgotten += deleted
    return forgotten


def _labels(model: str, corpora: list[str]) -> list[str]:
    named = _literal(model)
    return [
        *(
            f"FROM {_table(kind)[0]} p WHERE p.embedding IS NOT NULL "
            f"AND p.embedding_model = {named}{_within('p', corpora)}"
            for kind in _KINDS
        ),
        f"FROM memory_staged_vectors p WHERE p.model = {named}{_within('p', corpora)}",
    ]


async def labeled(conn: Any, model: str, corpus_ids: list[str]) -> Counter[str]:
    counted: Counter[str] = Counter()
    if corpus_ids:
        for held in _labels(model, corpus_ids):
            for r in await conn.fetch(f"SELECT p.corpus_id, count(*) AS n {held} GROUP BY 1"):
                counted[r["corpus_id"]] += r["n"]
    return counted


def _untaken_sql(to: str, corpora: list[str]) -> str:
    taken = " OR ".join(f"EXISTS (SELECT 1 {held})" for held in _labels(to, corpora))
    refusal = f"{to} came to label vectors after the plan — run the command again"
    return "DO " + _literal(
        f"BEGIN IF {taken} THEN RAISE EXCEPTION USING MESSAGE = {_literal(refusal)}; END IF; END"
    )


def _relabel_sql(kind: Kind, model: str, to: str, corpora: list[str]) -> str:
    table, _ = _table(kind)
    return (
        f"UPDATE {table} p SET embedding_model = {_literal(to)} WHERE p.embedding IS NOT NULL "
        f"AND p.embedding_model = {_literal(model)}{_within('p', corpora)}"
    )


async def rename_model(conn: Any, *, model: str, to: str, corpus_ids: list[str]) -> int:
    corpora = sorted(corpus_ids)
    tags = await _in_one_transaction(
        conn,
        [
            _try_lock_sql(migrations.MIGRATING, _MIGRATING),
            *(_try_lock_sql(_staging_key(name), _staging_runs(name)) for name in (model, to)),
            *(_lock_sql(f"memory.language:{corpus_id}") for corpus_id in corpora),
            _HOLD_PASSAGES,
            _untaken_sql(to, corpora),
            *(_relabel_sql(kind, model, to, corpora) for kind in _KINDS),
            f"UPDATE memory_staged_vectors p SET model = {_literal(to)} "
            f"WHERE p.model = {_literal(model)}{_within('p', corpora)}",
            "ANALYZE memory_chunks, memory_stories, memory_staged_vectors",
        ],
        "rename",
    )
    return sum(_count(tag) for tag in tags if tag.startswith("UPDATE"))
