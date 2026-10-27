import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from hashlib import sha256
from typing import Any, NamedTuple

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.filters import FACET_PREFIX, WRITTEN_PREFIX, facet, validate_filters
from rfnry_memory.postgres import migrations
from rfnry_memory.postgres.migrations import KEY_TEXT_BYTES, _execute, _literal

TYPES = ("number", "text")


class _Table(NamedTuple):
    short: str
    name: str
    noun: str
    id_column: str
    current: str
    json: bool


_TABLES = (
    _Table("chunks", "memory_chunks", "chunk", "chunk_id", "retired_at", False),
    _Table("sources", "memory_sources", "source", "source_id", "retired_at", True),
    _Table("stories", "memory_stories", "story", "story_id", "closed_at", True),
)


class IndexedRange(NamedTuple):
    kind: str
    bounds: dict[str, Any]


async def keyed(
    conn: Any, filters: dict[str, Any] | None, moment: datetime | None
) -> dict[str, Any] | None:
    if moment is None or not filters:
        return filters
    ranged = [
        key
        for key, condition in filters.items()
        if isinstance(condition, dict) and key.startswith((FACET_PREFIX, WRITTEN_PREFIX))
    ]
    if not ranged:
        return filters
    rows = await conn.fetch(
        "SELECT key, type FROM memory_indexed_keys WHERE ready AND built_at <= $2 "
        "AND key = ANY($1::text[])",
        ranged,
        moment,
    )
    if not rows:
        return filters
    kinds = {r["key"]: r["type"] for r in rows}
    return {
        key: IndexedRange(kinds[key], condition) if key in kinds else condition
        for key, condition in filters.items()
    }


class KeyPlan(NamedTuple):
    steps: list[str]
    statements: list[str]
    breaches: list[str]


def tag_of(key: str) -> str:
    return sha256(key.encode("utf-8")).hexdigest()[:12]


def _path(path: str) -> str:
    return "ARRAY[" + ", ".join(_literal(part) for part in path.split(".")) + "]::text[]"


def expression(target: str, path: str, kind: str) -> str:
    if kind == "number":
        return f"memory_key_number({target}, {_path(path)})"
    return f'(memory_key_text({target}, {_path(path)}) COLLATE "C")'


def _target(table: _Table, field: str) -> str:
    return f"{field}::jsonb" if field == "metadata" and table.json else field


def _named(table: _Table, tag: str) -> tuple[str, str]:
    return f"ix_memory_key_{table.short}_{tag}", f"ck_memory_key_{table.short}_{tag}"


def _what(kind: str) -> str:
    return "a number" if kind == "number" else f"a string of at most {KEY_TEXT_BYTES} bytes"


def _checked(key: str) -> tuple[str, str]:
    if not isinstance(key, str) or not any(
        key.startswith(prefix) and len(key) > len(prefix)
        for prefix in (FACET_PREFIX, WRITTEN_PREFIX)
    ):
        raise InputError(
            f"postgres: an indexed key is a filter key, {FACET_PREFIX}<key> or {WRITTEN_PREFIX}<key>; got {key!r}"
        )
    validate_filters({key: {"gte": 0}})
    if any(not part for part in key.split(".")):
        raise InputError(f"postgres: {key!r} names an empty part")
    if any(re.fullmatch(r"-?\d+", part) for part in key.split(".")[1:]):
        raise InputError(
            f"postgres: {key!r} names a position; the parts of an indexed key name object keys"
        )
    return facet(key)


async def _state(conn: Any, tag: str) -> dict[str, tuple[bool | None, bool | None]]:
    held: dict[str, tuple[bool | None, bool | None]] = {}
    for table in _TABLES:
        index, check = _named(table, tag)
        valid = await conn.fetchval(
            "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = $1 AND n.nspname = current_schema()",
            index,
        )
        validated = await conn.fetchval(
            "SELECT k.convalidated FROM pg_constraint k JOIN pg_namespace n ON n.oid = k.connamespace "
            "WHERE k.conname = $1 AND n.nspname = current_schema()",
            check,
        )
        held[table.short] = (valid, validated)
    return held


async def _breaches(conn: Any, field: str, path: str, key: str, kind: str) -> list[str]:
    found = []
    for table in _TABLES:
        fits = f"memory_key_fits({_target(table, field)}, {_path(path)}, {_literal(kind)})"
        ended = "retired" if table.current == "retired_at" else "closed"
        for seen, fix in (
            (f"{table.current} IS NULL", "fix them before indexing it"),
            (
                f"{table.current} IS NOT NULL AND (valid_to IS NULL OR (valid_to > now() "
                "AND (valid_from IS NULL OR valid_from < valid_to)))",
                "they keep it while in force; index it once they end, or delete them",
            ),
        ):
            row = await conn.fetchrow(
                f"SELECT count(*) AS n, min(corpus_id) AS corpus FROM {table.name} "
                f"WHERE {seen} AND NOT {fits}"
            )
            if not row["n"]:
                continue
            example = await conn.fetchval(
                f"SELECT {table.id_column} FROM {table.name} WHERE corpus_id = $1 AND {seen} "
                f'AND NOT {fits} ORDER BY {table.id_column} COLLATE "C" LIMIT 1',
                row["corpus"],
            )
            held = (
                f"current {table.short}"
                if fix.startswith("fix")
                else f"{ended} {table.short} still in force"
            )
            found.append(
                f"{row['n']} {held} hold {key} as something other than {_what(kind)} — "
                f"corpus {row['corpus']!r}, {table.noun} {example!r}; {fix}"
            )
    return found


async def plan_key(conn: Any, key: str, kind: str | None) -> KeyPlan:
    field, path = _checked(key)
    if kind is not None and kind not in TYPES:
        raise InputError(f"postgres: an indexed key holds a number or text; got {kind!r}")
    if max(await migrations._applied(conn), default=0) != migrations.LATEST:
        raise ConfigurationError(
            "postgres: the schema is not at this rfnry-memory's version — run rfnry-memory upgrade first"
        )
    schema = await conn.fetchval("SELECT quote_ident(current_schema())")
    tag = tag_of(key)
    held = await conn.fetchrow("SELECT type, ready FROM memory_indexed_keys WHERE key = $1", key)
    state = await _state(conn, tag)
    if kind is None:
        present = any(v is not None or c is not None for v, c in state.values())
        if held is None and not present:
            return KeyPlan([], [], [])
        statements = [f"UPDATE memory_indexed_keys SET ready = false WHERE key = {_literal(key)}"]
        for table in _TABLES:
            index, check = _named(table, tag)
            statements.append(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}.{index}")
            statements.append(
                f"ALTER TABLE {schema}.{table.name} DROP CONSTRAINT IF EXISTS {check}"
            )
        statements.append(f"DELETE FROM memory_indexed_keys WHERE key = {_literal(key)}")
        return KeyPlan([f"drop the index of {key} and lift its rule"], statements, [])
    if held is not None and held["type"] != kind:
        raise InputError(
            f"postgres: {key} is indexed as {held['type']} — drop it first with rfnry-memory index --key {key} --drop"
        )
    if held is not None and held["ready"] and all(v and c for v, c in state.values()):
        return KeyPlan([], [], [])
    breaches = await _breaches(conn, field, path, key, kind)
    statements = [
        "INSERT INTO memory_indexed_keys (key, type, tag) "
        f"VALUES ({_literal(key)}, {_literal(kind)}, {_literal(tag)}) ON CONFLICT (key) DO NOTHING"
    ]
    for table in _TABLES:
        index, check = _named(table, tag)
        valid, validated = state[table.short]
        target = _target(table, field)
        if validated is None:
            statements.append(
                f"ALTER TABLE {schema}.{table.name} ADD CONSTRAINT {check} CHECK ({table.current} IS NOT NULL "
                f"OR memory_key_fits({target}, {_path(path)}, {_literal(kind)})) NOT VALID"
            )
        if not validated:
            statements.append(f"ALTER TABLE {schema}.{table.name} VALIDATE CONSTRAINT {check}")
        if valid is False:
            statements.append(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}.{index}")
        if not valid:
            statements.append(
                f"CREATE INDEX CONCURRENTLY {index} ON {schema}.{table.name} "
                f"(corpus_id, {expression(target, path, kind)})"
            )
    statements.append(
        f"UPDATE memory_indexed_keys SET ready = true, built_at = now() WHERE key = {_literal(key)}"
    )
    statements.extend(f"ANALYZE {schema}.{table.name}" for table in _TABLES)
    steps = [
        f"index {key} as {'a number' if kind == 'number' else 'text'} on chunks, sources and stories",
        f"refuse a write of a current row holding anything but {_what(kind)} at {key}",
    ]
    return KeyPlan(steps, statements, breaches)


async def unfit(
    conn: Any, table: str, id_column: str, corpus_id: str, ids: list[str]
) -> tuple[str, str, str] | None:
    metadata = "metadata" if table == "memory_chunks" else "metadata::jsonb"
    found = await conn.fetchrow(
        f"SELECT r.{id_column} AS id, k.key, k.type FROM {table} r CROSS JOIN memory_indexed_keys k "
        f"WHERE r.corpus_id = $1 AND r.{id_column} = ANY($2::text[]) AND NOT memory_key_fits("
        f"CASE WHEN starts_with(k.key, '{WRITTEN_PREFIX}') THEN r.written_by ELSE r.{metadata} END, "
        "string_to_array(substr(k.key, strpos(k.key, '.') + 1), '.'), k.type) "
        f'ORDER BY r.{id_column} COLLATE "C", k.key COLLATE "C" LIMIT 1',
        corpus_id,
        ids,
    )
    return None if found is None else (found["id"], found["key"], found["type"])


async def refuse_unfit(
    conn: Any, table: str, id_column: str, noun: str, corpus_id: str, ids: list[str]
) -> None:
    from rfnry_memory.postgres.base import _unfitting

    found = await unfit(conn, table, id_column, corpus_id, ids) if ids else None
    if found is not None:
        held, key, kind = found
        what, other = _unfitting(kind)
        raise InputError(
            f"postgres: {key} is indexed as {what}, so {noun} {held!r}, which holds {other} there, "
            "cannot be in force again"
        )


def ended(column: str) -> str:
    return f"(CASE WHEN {column} IS NOT NULL THEN coalesce(valid_to, 'infinity'::timestamptz) END)"


_CURRENT = (
    ("chunks_fts", "memory_chunks", "USING GIN (fts) WHERE retired_at IS NULL"),
    ("stories_fts", "memory_stories", "USING GIN (fts) WHERE closed_at IS NULL"),
    ("chunks_ended", "memory_chunks", f"(corpus_id, {ended('retired_at')})"),
    ("stories_ended", "memory_stories", f"(corpus_id, {ended('closed_at')})"),
)


CURRENT_INDEXES = [f"ix_memory_current_{name}" for name, *_ in _CURRENT]


async def _current(conn: Any) -> dict[str, bool | None]:
    return {
        name: await conn.fetchval(
            "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = $1 AND n.nspname = current_schema()",
            f"ix_memory_current_{name}",
        )
        for name, *_ in _CURRENT
    }


async def current_state(conn: Any) -> str | None:
    held = await _current(conn)
    if all(valid is None for valid in held.values()):
        return None
    return "indexed" if all(held.values()) else "not ready"


async def plan_current(conn: Any, drop: bool) -> KeyPlan:
    if max(await migrations._applied(conn), default=0) != migrations.LATEST:
        raise ConfigurationError(
            "postgres: the schema is not at this rfnry-memory's version — run rfnry-memory upgrade first"
        )
    schema = await conn.fetchval("SELECT quote_ident(current_schema())")
    held = await _current(conn)
    if drop:
        if all(valid is None for valid in held.values()):
            return KeyPlan([], [], [])
        statements = [
            f"DROP INDEX CONCURRENTLY IF EXISTS {schema}.ix_memory_current_{name}"
            for name, *_ in _CURRENT
        ]
        return KeyPlan(["drop the indexes over current rows"], statements, [])
    if all(held.values()):
        return KeyPlan([], [], [])
    statements = []
    for name, table, spec in _CURRENT:
        index = f"ix_memory_current_{name}"
        if held[name] is False:
            statements.append(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}.{index}")
        if not held[name]:
            statements.append(f"CREATE INDEX CONCURRENTLY {index} ON {schema}.{table} {spec}")
    statements.extend(f"ANALYZE {schema}.{table}" for table in ("memory_chunks", "memory_stories"))
    steps = [
        "index the keywords of current chunks and stories, and the ends of retired and closed ones, "
        "for keyword reads of the present"
    ]
    return KeyPlan(steps, statements, [])


async def apply_current(conn: Any, drop: bool, *, expected: KeyPlan) -> None:
    await _apply(conn, lambda: plan_current(conn, drop), expected)


async def apply_key(conn: Any, key: str, kind: str | None, *, expected: KeyPlan) -> None:
    await _apply(conn, lambda: plan_key(conn, key, kind), expected)


async def _apply(conn: Any, planner: Callable[[], Awaitable[KeyPlan]], expected: KeyPlan) -> None:
    lock = "hashtextextended('memory.migrate:' || current_schema(), 0)"
    await conn.execute(f"SELECT pg_advisory_lock({lock})")
    try:
        planned = await planner()
        if planned != expected:
            raise ConfigurationError(
                "postgres: the schema changed since the plan was shown — run the command again"
            )
        if planned.breaches:
            raise InputError(f"postgres: {planned.breaches[0]}")
        for statement in planned.statements:
            await _execute(conn, statement)
    finally:
        await conn.execute(f"SELECT pg_advisory_unlock({lock})")


async def indexed_keys(conn: Any) -> list[tuple[str, str, bool]]:
    rows = await conn.fetch(
        'SELECT key, type, ready FROM memory_indexed_keys ORDER BY key COLLATE "C"'
    )
    return [(r["key"], r["type"], r["ready"]) for r in rows]
