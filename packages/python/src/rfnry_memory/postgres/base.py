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


class Base:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._dimension: int | None = None
        self._current = False

    @asynccontextmanager
    async def _guard(self, operation: str) -> AsyncIterator[None]:
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
            refused = await self._refused(exc)
            if refused is not None:
                raise refused from exc
            raise StorageError(f"postgres: storage rejected {operation}: {exc}") from exc

    async def _refused(self, exc: Exception) -> InputError | None:
        name = getattr(exc, "constraint_name", None) or ""
        if getattr(exc, "sqlstate", None) != "23514" or not name.startswith("ck_memory_key_"):
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
