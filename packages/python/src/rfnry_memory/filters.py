import math
from copy import deepcopy
from typing import Any

from rfnry_memory.errors import InputError

RESERVED = frozenset({"corpus_id"})
FACET_PREFIX = "metadata."
WRITTEN_PREFIX = "written_by."
RANGE = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


def _number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _valid_list(values: list[Any]) -> bool:
    if all(isinstance(v, str) for v in values):
        return True
    return all(_number(v) for v in values)


def _valid_range(bounds: dict[Any, Any]) -> bool:
    if not bounds or not set(bounds) <= set(RANGE):
        return False
    values = list(bounds.values())
    return all(isinstance(v, str) for v in values) or all(_number(v) for v in values)


def validate_filters(filters: dict[str, Any] | None) -> None:
    if not filters:
        return
    for key, condition in filters.items():
        if not isinstance(key, str) or not key.strip():
            raise InputError("filters: filter keys must be non-empty strings")
        if key in RESERVED:
            raise InputError(
                f"filters: filter key {key!r} is reserved — pass it as a call argument"
            )
        if not any(key.startswith(p) and len(key) > len(p) for p in (FACET_PREFIX, WRITTEN_PREFIX)):
            raise InputError(
                f"filters: filter key {key!r} must address a facet as '{FACET_PREFIX}<key>' "
                f"or '{WRITTEN_PREFIX}<key>'"
            )
        if isinstance(condition, list):
            if not condition or not _valid_list(condition):
                raise InputError(
                    f"filters: filter {key!r} must be a non-empty list of strings or of finite numbers"
                )
            continue
        if isinstance(condition, dict):
            if not _valid_range(condition):
                raise InputError(
                    f"filters: filter {key!r} is a range of gt, gte, lt and lte, all finite numbers or all strings"
                )
            continue
        if isinstance(condition, str | bool) or _number(condition):
            continue
        raise InputError(
            f"filters: filter {key!r} must be a string, a finite number, a bool, a list of strings or of "
            "finite numbers, or a range"
        )


def _lookup(metadata: dict[str, Any], path: str) -> tuple[Any, bool]:
    current: Any = metadata
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None, False
        current = current[part]
    return current, True


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return bool(a == b)


def in_range(value: Any, bounds: dict[str, Any]) -> bool:
    texts = all(isinstance(b, str) for b in bounds.values())
    if not (isinstance(value, str) if texts else _number(value)):
        return False
    return all(
        {"gt": value > b, "gte": value >= b, "lt": value < b, "lte": value <= b}[op]
        for op, b in bounds.items()
    )


def _condition_matches(value: Any, condition: Any) -> bool:
    values = value if isinstance(value, list) else [value]
    if isinstance(condition, dict):
        return any(in_range(v, condition) for v in values)
    conditions = condition if isinstance(condition, list) else [condition]
    return any(_same(v, c) for v in values for c in conditions)


def within(value: Any, bound: Any) -> bool:
    values = value if isinstance(value, list) else [value]
    bounds = bound if isinstance(bound, list) else [bound]
    return bool(values) and all(any(_same(v, b) for b in bounds) for v in values)


def facet(key: str) -> tuple[str, str]:
    if key.startswith(WRITTEN_PREFIX):
        return "written_by", key[len(WRITTEN_PREFIX) :]
    return "metadata", key[len(FACET_PREFIX) :]


def matches_filters(
    metadata: dict[str, Any],
    filters: dict[str, Any] | None,
    written_by: dict[str, Any] | None = None,
) -> bool:
    if not filters:
        return True
    for key, condition in filters.items():
        field, path = facet(key)
        value, found = _lookup(metadata if field == "metadata" else written_by or {}, path)
        if not found or not _condition_matches(value, condition):
            return False
    return True


def stamp(metadata: dict[str, Any] | None, filters: dict[str, Any]) -> dict[str, Any]:
    stamped = deepcopy(metadata or {})
    for key, value in filters.items():
        if isinstance(value, dict):
            found, held = _lookup(stamped, key[len(FACET_PREFIX) :])
            values = found if isinstance(found, list) else [found]
            if not held or not values or not all(in_range(v, value) for v in values):
                raise InputError(
                    f"filters: metadata must give {key!r} a value inside the bound range {value!r}"
                )
            continue
        *parents, leaf = key[len(FACET_PREFIX) :].split(".")
        node: Any = stamped
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise InputError(
                    f"filters: metadata must agree with the bound filter {key!r} = {value!r}"
                )
        if leaf in node:
            if not within(node[leaf], value):
                raise InputError(
                    f"filters: metadata must agree with the bound filter {key!r} = {value!r}"
                )
            continue
        if isinstance(value, list) and len(value) > 1:
            raise InputError(
                f"filters: metadata must name one of the values bound to {key!r}: {value!r}"
            )
        node[leaf] = value[0] if isinstance(value, list) else value
    return stamped
