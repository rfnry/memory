import json
from datetime import datetime
from decimal import Decimal
from typing import Any

from rfnry_memory.filters import DATES, RANGE, date_bound, facet
from rfnry_memory.models import (
    Kind,
    Scope,
    Window,
)
from rfnry_memory.postgres.keys import IndexedRange, expression


def _nested(path: str, value: Any) -> dict[str, Any]:
    parts = path.split(".")
    node: Any = value
    for part in reversed(parts):
        node = {part: node}
    return dict(node)


_ROWS = {
    "story": ("memory_stories", "closed_at", "story", "story_id"),
    "source": ("memory_sources", "retired_at", "source", "source_id"),
    "chunk": ("memory_chunks", "retired_at", "source", "source_id"),
}


def _archived(row: str, alias: str, learned: str, picked: str) -> str:
    table, _, kind, id_column = _ROWS[row]
    at = alias or f"{table}."
    return (
        f"(SELECT {picked} FROM memory_closings h WHERE h.corpus_id = {at}corpus_id "
        f"AND h.kind = '{kind}' AND h.id = {at}{id_column} AND h.revised_at > {learned} "
        "ORDER BY h.revised_at LIMIT 1)"
    )


def _known_end(row: str, alias: str, learned: str) -> str:
    table, closed, _, _ = _ROWS[row]
    at = alias or f"{table}."
    held = (
        f"(CASE WHEN {at}{closed} IS NULL OR {at}{closed} <= {learned} "
        f"THEN {at}valid_to ELSE {at}written_to END)"
    )
    archived = _archived(
        row,
        alias,
        learned,
        f"CASE WHEN h.closed_at <= {learned} THEN h.valid_to ELSE {at}written_to END",
    )
    return f"(CASE WHEN {at}revised_at IS NULL OR {at}revised_at <= {learned} THEN {held} ELSE {archived} END)"


def _known_closed(row: str, alias: str, learned: str) -> str:
    table, closed, _, _ = _ROWS[row]
    at = alias or f"{table}."
    archived = _archived(row, alias, learned, f"h.closed_at <= {learned}")
    return (
        f"(CASE WHEN {at}revised_at IS NULL OR {at}revised_at <= {learned} "
        f"THEN coalesce({at}{closed} <= {learned}, false) ELSE {archived} END)"
    )


def _date_column(
    key: str, kind: Kind | None, alias: str, known_at: datetime | None, params: list[Any]
) -> str:
    if kind == "chunk" and key in ("created_at", "confirmed_at"):
        return (
            f"(SELECT src.{key} FROM memory_sources AS src WHERE src.corpus_id = {alias}corpus_id "
            f"AND src.source_id = {alias}source_id)"
        )
    column = "created_at" if kind == "story" and key == "confirmed_at" else key
    if column != "valid_to" or known_at is None:
        return f"{alias}{column}"
    params.append(known_at)
    return _known_end(kind or "source", alias, f"${len(params)}")


def _dated_sql(column: str, bounds: dict[str, Any], params: list[Any]) -> str:
    sides = []
    for op, value in bounds.items():
        params.append(date_bound(value))
        sides.append(f"{column} {RANGE[op]} ${len(params)}")
    return "(" + " AND ".join(sides) + ")"


def _facet_sql(
    filters: dict[str, Any] | None,
    params: list[Any],
    alias: str = "",
    cast: str = "",
    *,
    kind: Kind | None = None,
    known_at: datetime | None = None,
) -> str:
    if not filters:
        return ""
    clauses = []
    for key, condition in filters.items():
        if key in DATES:
            column = _date_column(key, kind, alias, known_at, params)
            clauses.append(_dated_sql(column, condition, params))
            continue
        field, path = facet(key)
        target = f"{alias}{field}{cast}" if field == "metadata" else f"{alias}{field}"
        if isinstance(condition, IndexedRange):
            clauses.append(_indexed_sql(target, path, condition, params))
            continue
        if isinstance(condition, dict):
            clauses.append(_range_sql(path, condition, params, target))
            continue
        values = condition if isinstance(condition, list) else [condition]
        alternatives = []
        for value in values:
            for shape in (value, [value]):
                params.append(json.dumps(_nested(path, shape)))
                alternatives.append(f"{target} @> ${len(params)}::jsonb")
        clauses.append("(" + " OR ".join(alternatives) + ")")
    return " AND " + " AND ".join(clauses)


def _indexed_sql(target: str, path: str, indexed: IndexedRange, params: list[Any]) -> str:
    numbers = not any(isinstance(bound, str) for bound in indexed.bounds.values())
    if numbers != (indexed.kind == "number"):
        return "FALSE"
    column = expression(target, path, indexed.kind)
    sides = []
    for op, bound in indexed.bounds.items():
        params.append(Decimal(str(bound)) if numbers else bound)
        sides.append(f"{column} {RANGE[op]} ${len(params)}::{'numeric' if numbers else 'text'}")
    return "(" + " AND ".join(sides) + ")"


def _range_sql(path: str, bounds: dict[str, Any], params: list[Any], column: str) -> str:
    keys = "".join("." + json.dumps(part) for part in path.split("."))
    test = " && ".join(f"@ {RANGE[op]} ${op}" for op in bounds)
    params.append(json.dumps(bounds))
    bound = f"${len(params)}::jsonb"
    shapes = [f"strict ${keys} ? ({test})", f"strict ${keys}[*] ? ({test})"]
    found = []
    for shape in shapes:
        params.append(shape)
        found.append(f"jsonb_path_exists({column}, ${len(params)}::jsonpath, {bound}, true)")
    return "(" + " OR ".join(found) + ")"


def _window_sql(window: Window, params: list[Any], alias: str = "") -> str:
    start, end = f"{alias}valid_from", f"{alias}valid_to"
    sql = ""
    if window.known_at is not None:
        params.append(window.known_at)
        learned = f"${len(params)}"
        sql += f" AND {alias}created_at <= {learned}"
        end = _known_end("story", alias, learned)
    if window.as_of is not None:
        params.append(window.as_of)
        moment = f"${len(params)}"
        if window.open:
            sql += (
                f" AND {alias}superseded_by IS NULL "
                f"AND ({end} IS NULL OR ({end} > {moment} AND {end} > {start}))"
            )
        elif window.known_at is not None:
            sql += f" AND {start} <= {moment} AND coalesce({end} > {moment}, true)"
        else:
            sql += f" AND {start} <= {moment} AND ({end} IS NULL OR {end} > {moment})"
    if window.since is not None:
        params.append(window.since)
        sql += f" AND {start} >= ${len(params)}"
    if window.until is not None:
        params.append(window.until)
        sql += f" AND {start} <= ${len(params)}"
    return sql


def _in_force_sql(
    as_of: datetime | None, known_at: datetime | None, params: list[Any], alias: str, row: str
) -> str:
    if as_of is None:
        return ""
    end = None
    if known_at is not None:
        params.append(known_at)
        end = _known_end(row, alias, f"${len(params)}")
    params.append(as_of)
    moment = f"${len(params)}"
    ends = (
        f"({alias}valid_to IS NULL OR {alias}valid_to > {moment})"
        if end is None
        else f"coalesce({end} > {moment}, true)"
    )
    return f" AND ({alias}valid_from IS NULL OR {alias}valid_from <= {moment}) AND {ends}"


def _chunks_sql(window: Window, params: list[Any], alias: str) -> str:
    sql = ""
    if window.known_at is not None:
        params.append(window.known_at)
        sql += (
            f" AND EXISTS (SELECT 1 FROM memory_sources src WHERE src.corpus_id = {alias}corpus_id "
            f"AND src.source_id = {alias}source_id AND src.created_at <= ${len(params)})"
        )
    return sql + _in_force_sql(window.as_of, window.known_at, params, alias, "chunk")


_CHUNKS = (
    "memory_chunks CROSS JOIN LATERAL (SELECT s.created_at, s.confirmed_at FROM memory_sources s "
    "WHERE s.corpus_id = memory_chunks.corpus_id AND s.source_id = memory_chunks.source_id) AS dated"
)


def _subject_sql(subject: str | None, params: list[Any], alias: str = "") -> str:
    if subject is None:
        return ""
    params.append(subject)
    return f" AND {alias}subject = ${len(params)}"


def _table(kind: Kind) -> tuple[str, str]:
    return ("memory_chunks", "chunk_id") if kind == "chunk" else ("memory_stories", "story_id")


def _within_sql(kind: Kind, scope: Scope | None, params: list[Any], alias: str) -> str:
    if scope is None:
        return ""
    sql = ""
    if scope.sources is not None:
        params.append(list(scope.sources))
        if kind == "chunk":
            sql += f" AND {alias}source_id = ANY(${len(params)}::text[])"
        else:
            sql += f" AND {alias}source_ids ?| ${len(params)}::text[]"
    if scope.passages is not None:
        params.append([i for k, i in scope.passages if k == kind])
        _, id_column = _table(kind)
        sql += f" AND {alias}{id_column} = ANY(${len(params)}::text[])"
    return sql


def _scope_sql(
    kind: Kind,
    filters: dict[str, Any] | None,
    window: Window,
    subject: str | None,
    scope: Scope | None,
    params: list[Any],
) -> str:
    if kind == "story":
        return (
            _facet_sql(filters, params, cast="::jsonb", kind="story", known_at=window.known_at)
            + _window_sql(window, params)
            + _subject_sql(subject, params)
            + _within_sql(kind, scope, params, "")
        )
    return (
        _facet_sql(filters, params, "memory_chunks.", kind="chunk", known_at=window.known_at)
        + _chunks_sql(window, params, "memory_chunks.")
        + _within_sql(kind, scope, params, "memory_chunks.")
    )


def _subject_kinds(kinds: tuple[Kind, ...], subject: str | None) -> tuple[Kind, ...]:
    wanted: tuple[Kind, ...] = ("chunk", "story") if subject is None else ("story",)
    return tuple(kind for kind in wanted if kind in kinds)
