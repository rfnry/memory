from copy import deepcopy
from typing import Any

from rfnry_memory.errors import InputError

RESERVED = frozenset({"corpus_id"})
FACET_PREFIX = "metadata."


def _valid_list(values: list[Any]) -> bool:
    if all(isinstance(v, str) for v in values):
        return True
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


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
        if not key.startswith(FACET_PREFIX) or len(key) == len(FACET_PREFIX):
            raise InputError(
                f"filters: filter key {key!r} must address a facet as '{FACET_PREFIX}<key>'"
            )
        if isinstance(condition, list):
            if not condition or not _valid_list(condition):
                raise InputError(
                    f"filters: filter {key!r} must be a non-empty list of str or of int"
                )
            continue
        if isinstance(condition, str | int):
            continue
        raise InputError(
            f"filters: filter {key!r} must be str, int, bool or a list of str or of int"
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


def _condition_matches(value: Any, condition: Any) -> bool:
    values = value if isinstance(value, list) else [value]
    conditions = condition if isinstance(condition, list) else [condition]
    return any(_same(v, c) for v in values for c in conditions)


def within(value: Any, bound: Any) -> bool:
    values = value if isinstance(value, list) else [value]
    bounds = bound if isinstance(bound, list) else [bound]
    return bool(values) and all(any(_same(v, b) for b in bounds) for v in values)


def matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, condition in filters.items():
        value, found = _lookup(metadata, key[len(FACET_PREFIX) :])
        if not found or not _condition_matches(value, condition):
            return False
    return True


def stamp(metadata: dict[str, Any] | None, filters: dict[str, Any]) -> dict[str, Any]:
    stamped = deepcopy(metadata or {})
    for key, value in filters.items():
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
