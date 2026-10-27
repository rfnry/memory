from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, JsonValue, model_validator

from rfnry_memory.models import Judgement, PillarReport
from rfnry_memory.rows import RetrievalLabelRow, RetrievalRow

Verb = Literal["search", "recall", "expand", "resembling"]


class RecordedHit(BaseModel):
    corpus_id: str
    kind: Literal["chunk", "story", "source"]
    id: str
    score: float
    source_id: str | None = None
    source_name: str | None = None
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
    shown: list[tuple[str, Literal["chunk", "story", "source"], str]] | None = None


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


class Relevance(BaseModel):
    corpus_id: str
    kind: Literal["chunk", "story", "source"]
    id: str
    grade: int = Field(default=1, ge=0, le=3)
    probability: float | None = None
    model: str | None = None
    settings: dict[str, JsonValue] | None = None


class RetrievalReview(BaseModel):
    relevance: list[Relevance] = Field(default_factory=list)
    comment: str | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> RetrievalReview:
        if not self.relevance and not self.comment:
            raise ValueError("refinery: a review grades a passage or a document, or comments")
        return self


class RetrievalOutcome(BaseModel):
    status: Literal["succeeded", "failed"] | None = None
    cited: list[Relevance] = Field(default_factory=list)
    detail: str | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> RetrievalOutcome:
        if self.status is None and not self.cited and not self.detail:
            raise ValueError("refinery: an outcome has a status, a citation or a detail")
        return self


class RetrievalLabel(BaseModel):
    label_id: str
    retrieval_id: str
    at: datetime
    source: Literal["human", "world", "judge"]
    review: RetrievalReview | None = None
    outcome: RetrievalOutcome | None = None
    judged: list[Relevance] | None = None


class RetrievalLabeled(BaseModel):
    judged: int = 0
    skipped: int = 0
    errors: dict[str, str] = Field(default_factory=dict)


def label_of(row: RetrievalLabelRow) -> RetrievalLabel:
    return RetrievalLabel.model_validate(
        {
            "label_id": row.label_id,
            "retrieval_id": row.retrieval_id,
            "at": row.at,
            "source": row.source,
            **row.body,
        }
    )


class RetrievalReplay(BaseModel):
    recorded: Retrieval
    labels: list[RetrievalLabel]
    candidate: Retrieval | None = None
    error: str | None = None


class RetrievalScores(BaseModel):
    recall: float
    mrr: float
    ndcg: float
    measured: int


class RetrievalMetrics(BaseModel):
    k: int
    retrievals: int
    labeled: int
    errors: int
    documents: dict[str, RetrievalScores]
    passages: dict[str, RetrievalScores] | None
    latency_ms: dict[str, float]


class RetrievalExport(BaseModel):
    keys: list[str]
    rows: int
    missing: int
    watermark: datetime | None
