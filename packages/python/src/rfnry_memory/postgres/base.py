from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError, StorageError


class Base:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._dimension: int | None = None

    @asynccontextmanager
    async def _guard(self, operation: str) -> AsyncIterator[None]:
        try:
            yield
        except StorageError, NotFoundError, ConfigurationError, InputError:
            raise
        except Exception as exc:
            raise StorageError(f"postgres: storage rejected {operation}: {exc}") from exc

    def _check_vectors(self, vectors: object) -> None:
        if vectors is not None and not self._dimension:
            raise StorageError(
                "postgres: vectors were given but this storage has no vector dimension — call check() first"
            )
