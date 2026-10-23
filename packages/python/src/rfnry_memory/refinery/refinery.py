import json
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError
from rfnry_memory.judge import judge as judge_hits
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery.holdout import HOLDOUT_PERCENT, held_out, sampled
from rfnry_memory.refinery.records import (
    RecordedHit,
    Relevance,
    Retrieval,
    RetrievalLabel,
    RetrievalLabeled,
    RetrievalOutcome,
    RetrievalReview,
    label_of,
    retrieval_of,
)
from rfnry_memory.rows import RetrievalLabelRow, RetrievalRow

if TYPE_CHECKING:
    from rfnry_memory.memory import Memory


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

    async def _readable(self, corpora: list[str], retrieval_id: str) -> RetrievalRow:
        row = await self.storage.get_retrieval(corpora, retrieval_id)
        if row is None:
            raise NotFoundError(f"refinery: retrieval {retrieval_id!r} was not found")
        return row

    async def _labeled(
        self,
        corpora: list[str],
        retrieval_id: str,
        source: str,
        body: dict[str, Any],
        label_id: str | None,
    ) -> RetrievalLabel:
        if label_id is not None and (not isinstance(label_id, str) or not label_id.strip()):
            raise InputError("refinery: label_id is a non-empty string")
        row = RetrievalLabelRow(
            retrieval_id=retrieval_id,
            label_id=label_id or uuid4().hex,
            source=source,
            at=datetime.now(UTC),
            body=body,
        )
        stored = await self.storage.put_retrieval_label(corpora, row)
        if stored is None:
            raise NotFoundError(f"refinery: retrieval {retrieval_id!r} was not found")
        return label_of(stored)

    async def review(
        self,
        corpus_id: str | list[str],
        retrieval_id: str,
        review: RetrievalReview,
        *,
        label_id: str | None = None,
    ) -> RetrievalLabel:
        corpora = named(corpus_id)
        _within(await self._readable(corpora, retrieval_id), review.relevance)
        return await self._labeled(
            corpora, retrieval_id, "human", {"review": review.model_dump(mode="json")}, label_id
        )

    async def outcome(
        self,
        corpus_id: str | list[str],
        retrieval_id: str,
        outcome: RetrievalOutcome,
        *,
        label_id: str | None = None,
    ) -> RetrievalLabel:
        corpora = named(corpus_id)
        _within(await self._readable(corpora, retrieval_id), outcome.cited)
        return await self._labeled(
            corpora, retrieval_id, "world", {"outcome": outcome.model_dump(mode="json")}, label_id
        )

    async def labels(self, corpus_id: str | list[str], retrieval_id: str) -> list[RetrievalLabel]:
        corpora = named(corpus_id)
        await self._readable(corpora, retrieval_id)
        return [
            label_of(row) for row in await self.storage.retrieval_labels(corpora, [retrieval_id])
        ]

    async def label(
        self, corpus_id: str | list[str], memory: Memory, *, limit: int = 500
    ) -> RetrievalLabeled:
        if memory.decision is None:
            raise ConfigurationError("refinery: label needs a memory with a decision model")
        if limit < 1:
            raise InputError("refinery: limit must be at least 1")
        corpora = named(corpus_id)
        done = await self.storage.judged_retrievals(corpora)
        rows, _ = await self.storage.list_retrievals(corpora, since=None, limit=None, cursor=None)
        pending = [
            row
            for row in reversed(rows)
            if row.retrieval_id not in done and row.verb != "resembling"
        ][:limit]
        labeled = RetrievalLabeled()
        for row in pending:
            try:
                judged = await _judged(row, memory)
            except Exception as exc:
                labeled.errors[row.retrieval_id] = f"{type(exc).__name__}: {exc}"
                continue
            body = {"judged": [r.model_dump(mode="json") for r in judged]}
            await self._labeled(corpora, row.retrieval_id, "judge", body, None)
            labeled.judged += 1
        return labeled


def _within(row: RetrievalRow, relevance: list[Relevance]) -> None:
    for graded in relevance:
        if graded.corpus_id not in row.corpus_ids:
            raise InputError(
                f"refinery: retrieval {row.retrieval_id!r} read {', '.join(row.corpus_ids)}; "
                f"a grade naming {graded.corpus_id!r} is not one of them"
            )


def _graded(hit: RecordedHit, probability: float | None) -> Relevance:
    return Relevance(
        corpus_id=hit.corpus_id,
        kind=hit.kind,
        id=hit.id,
        grade=1 if probability is not None and probability >= 0.5 else 0,
        probability=probability,
    )


async def _judged(row: RetrievalRow, memory: Memory) -> list[Relevance]:
    hits = [RecordedHit.model_validate(hit) for hit in row.body["hits"]]
    if hits and all(hit.judge is not None and hit.judge.relevant is not None for hit in hits):
        return [_graded(hit, hit.judge.relevant if hit.judge else None) for hit in hits]
    query = row.body["query"]
    if query is None:
        kind, passage_id = row.body["arguments"]["passage"]
        found = await memory.storage.get_passages(row.corpus_ids[0], [(kind, passage_id)])
        if not found:
            return []
        query = found[0].text
    fused: list[tuple[str, float, dict[str, float]]] = [
        (json.dumps([hit.corpus_id, hit.kind, hit.id]), hit.score, {}) for hit in hits
    ]
    present = await memory._hydrate(fused)
    assert memory.decision is not None
    judged, _ = await judge_hits(memory.decision, query, present, ("relevant",))
    by_id = {(hit.corpus_id, hit.kind, hit.id): hit for hit in hits}
    return [
        _graded(by_id[(hit.corpus_id, hit.kind, hit.id)], hit.judge.relevant if hit.judge else None)
        for hit in judged
    ]
