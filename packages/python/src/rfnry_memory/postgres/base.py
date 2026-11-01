from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError, StorageError


def _unfitting(kind: str) -> tuple[str, str]:
    from rfnry_memory.postgres.migrations import KEY_TEXT_BYTES

    if kind == "number":
        return "a number", "another type or an array"
    return "text", f"another type, an array or a string over {KEY_TEXT_BYTES} bytes"


_HELD = {
    "memory_sources_pkey": ("source", "source_id"),
    "memory_stories_pkey": ("story", "story_id"),
    "memory_entities_pkey": ("entity", "entity_id"),
}


def _held(exc: Exception, corpus_id: str | None) -> InputError:
    noun, column = _HELD[getattr(exc, "constraint_name", "")]
    detail = getattr(exc, "detail", None) or ""
    key, end = f"Key (corpus_id, {column})=({corpus_id}, ", ") already exists."
    if corpus_id is not None and detail.startswith(key) and detail.endswith(end):
        held = detail[len(key) : -len(end)]
        return InputError(f"postgres: {noun} {held!r} is already held; ids must be unique")
    return InputError(f"postgres: a {noun} id this write gives is already held; ids must be unique")


class Base:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._dimension: int | None = None
        self._current = False

    @asynccontextmanager
    async def _guard(self, operation: str, corpus_id: str | None = None) -> AsyncIterator[None]:
        try:
            yield
        except StorageError, NotFoundError, ConfigurationError, InputError:
            raise
        except (
            asyncpg.PostgresError,
            asyncpg.InterfaceError,
            asyncpg.exceptions.InternalClientError,
            OSError,
            TimeoutError,
        ) as exc:
            refused = await self._refused(exc, corpus_id)
            if refused is not None:
                raise refused from exc
            raise StorageError(f"postgres: storage rejected {operation}: {exc}") from exc

    async def _refused(self, exc: Exception, corpus_id: str | None) -> InputError | None:
        name = getattr(exc, "constraint_name", None) or ""
        state = getattr(exc, "sqlstate", None)
        if state == "23505" and name in _HELD:
            return _held(exc, corpus_id)
        if state != "23514" or not name.startswith("ck_memory_key_"):
            return None
        tag = name.rsplit("_", 1)[-1]
        try:
            async with self._pool.acquire() as conn:
                held = await conn.fetchrow(
                    "SELECT key, type FROM memory_indexed_keys WHERE tag = $1", tag
                )
        except Exception:
            held = None
        if held is None:
            return InputError(f"postgres: a write holds a value an indexed key refuses ({name})")
        what, other = _unfitting(held["type"])
        return InputError(
            f"postgres: {held['key']} is indexed as {what}, so a write holding {other} there is refused"
        )

    def _dimensioned(self, asked: str) -> None:
        if not self._dimension:
            raise StorageError(
                f"postgres: {asked} but this storage has no vector dimension — call check() first"
            )

    def _check_vectors(self, vectors: object) -> None:
        if vectors is not None:
            self._dimensioned("vectors were given")
