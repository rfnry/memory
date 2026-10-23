import math
from datetime import UTC, datetime

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery.holdout import HOLDOUT_PERCENT, held_out, sampled
from rfnry_memory.refinery.records import Retrieval, retrieval_of


def named(corpus_id: str | list[str]) -> list[str]:
    corpora = list(corpus_id) if isinstance(corpus_id, list | tuple) else [corpus_id]
    if not corpora or not all(isinstance(c, str) and c.strip() for c in corpora):
        raise InputError("refinery: corpus_id is required")
    return sorted(dict.fromkeys(corpora))


class MemoryRefinery:
    def __init__(self, storage: PostgresStorage, *, sample: float = 1.0) -> None:
        if not isinstance(sample, int | float) or not math.isfinite(sample) or not 0 <= sample <= 1:
            raise ConfigurationError("refinery: sample is a rate from 0 to 1")
        self.storage = storage
        self.sample = float(sample)

    def is_holdout(self, retrieval_id: str, *, holdout_percent: int = HOLDOUT_PERCENT) -> bool:
        if not 0 <= holdout_percent <= 100:
            raise InputError(f"refinery: holdout_percent must be 0-100, got {holdout_percent}")
        return held_out(retrieval_id, holdout_percent)

    def _records(self, retrieval_id: str) -> bool:
        return sampled(retrieval_id, self.sample)

    async def retrieval(self, corpus_id: str | list[str], retrieval_id: str) -> Retrieval:
        row = await self.storage.get_retrieval(named(corpus_id), retrieval_id)
        if row is None:
            raise NotFoundError(f"refinery: retrieval {retrieval_id!r} was not found")
        return retrieval_of(row)

    async def retrievals(
        self,
        corpus_id: str | list[str],
        *,
        since: datetime | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[Retrieval], str | None]:
        if limit < 1:
            raise InputError("refinery: limit must be at least 1")
        if cursor is not None and not (cursor.isdigit() and cursor.isascii()):
            raise InputError(f"refinery: cursor {cursor!r} is not one a listing returned")
        rows, following = await self.storage.list_retrievals(
            named(corpus_id), since=since, limit=limit, cursor=cursor
        )
        return [retrieval_of(row) for row in rows], following

    async def forget_recorded(self, corpus_id: str | list[str], *, before: datetime) -> int:
        moment = before if before.tzinfo is not None else before.replace(tzinfo=UTC)
        return await self.storage.forget_retrievals(named(corpus_id), before=moment)
