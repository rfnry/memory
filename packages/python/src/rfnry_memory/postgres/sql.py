import json
from typing import Any

from rfnry_memory.filters import FACET_PREFIX
from rfnry_memory.models import (
    Kind,
    Scope,
    Window,
)


def _nested(path: str, value: Any) -> dict[str, Any]:
    parts = path.split(".")
    node: Any = value
    for part in reversed(parts):
        node = {part: node}
    return dict(node)


def _facet_sql(filters: dict[str, Any] | None, params: list[Any], column: str) -> str:
    if not filters:
        return ""
    clauses = []
    for key, condition in filters.items():
        path = key[len(FACET_PREFIX) :]
        values = condition if isinstance(condition, list) else [condition]
        alternatives = []
        for value in values:
            for shape in (value, [value]):
                params.append(json.dumps(_nested(path, shape)))
                alternatives.append(f"{column} @> ${len(params)}::jsonb")
        clauses.append("(" + " OR ".join(alternatives) + ")")
    return " AND " + " AND ".join(clauses)


def _window_sql(window: Window, params: list[Any], alias: str = "") -> str:
    start, end = f"{alias}valid_from", f"{alias}valid_to"
    sql = ""
    if window.known_at is not None:
        params.append(window.known_at)
        learned = f"${len(params)}"
        sql += f" AND {alias}created_at <= {learned}"
        end = (
            f"(CASE WHEN {alias}closed_at IS NULL OR {alias}closed_at <= {learned} "
            f"THEN {end} ELSE {alias}written_to END)"
        )
    if window.as_of is not None:
        params.append(window.as_of)
        moment = f"${len(params)}"
        if window.open:
            sql += f" AND {alias}superseded_by IS NULL AND ({end} IS NULL OR {end} > {moment})"
        else:
            sql += f" AND {start} <= {moment} AND ({end} IS NULL OR {end} > {moment})"
    if window.since is not None:
        params.append(window.since)
        sql += f" AND {start} >= ${len(params)}"
    if window.until is not None:
        params.append(window.until)
        sql += f" AND {start} <= ${len(params)}"
    return sql


def _chunks_sql(window: Window, params: list[Any], alias: str) -> str:
    sql = ""
    if window.known_at is not None:
        params.append(window.known_at)
        learned = f"${len(params)}"
        sql += (
            f" AND EXISTS (SELECT 1 FROM memory_sources src WHERE src.corpus_id = {alias}corpus_id "
            f"AND src.source_id = {alias}source_id AND src.created_at <= {learned})"
        )
        if window.as_of is not None:
            sql += f" AND ({alias}retired_at IS NULL OR {alias}retired_at > {learned})"
    elif window.as_of is not None:
        sql += f" AND {alias}retired_at IS NULL"
    return sql


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
            _facet_sql(filters, params, "metadata::jsonb")
            + _window_sql(window, params)
            + _subject_sql(subject, params)
            + _within_sql(kind, scope, params, "")
        )
    return (
        _facet_sql(filters, params, "metadata")
        + _chunks_sql(window, params, "memory_chunks.")
        + _within_sql(kind, scope, params, "memory_chunks.")
    )


def _subject_kinds(kinds: tuple[Kind, ...], subject: str | None) -> tuple[Kind, ...]:
    wanted: tuple[Kind, ...] = ("chunk", "story") if subject is None else ("story",)
    return tuple(kind for kind in wanted if kind in kinds)
