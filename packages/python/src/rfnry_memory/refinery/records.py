from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from rfnry_memory.models import Judgement, PillarReport
from rfnry_memory.rows import RetrievalRow

Verb = Literal["search", "recall", "expand", "resembling"]


class RecordedHit(BaseModel):
    corpus_id: str
    kind: Literal["chunk", "story", "source"]
    id: str
    score: float
    source_id: str | None = None
    judge: Judgement | None = None


class Retrieval(BaseModel):
    retrieval_id: str
    corpus_ids: list[str]
    verb: Verb
    at: datetime
    recorded: datetime
    duration_ms: int
    query: str | None
    arguments: dict[str, Any]
    window: dict[str, datetime | None]
    candidates: dict[str, list[RecordedHit]]
    hits: list[RecordedHit]
    pillars: dict[str, PillarReport]
    fingerprint: dict[str, Any]


def retrieval_of(row: RetrievalRow) -> Retrieval:
    return Retrieval.model_validate(
        {
            "retrieval_id": row.retrieval_id,
            "corpus_ids": row.corpus_ids,
            "verb": row.verb,
            "at": row.at,
            "recorded": row.recorded,
            "duration_ms": row.duration_ms,
            **row.body,
        }
    )
