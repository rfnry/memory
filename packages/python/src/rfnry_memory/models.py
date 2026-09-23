from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, model_validator

from rfnry_memory.errors import ModelError

State = str | dict[str, Any] | list[Any]


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str | list[dict[str, Any]]
    tool_call_id: str | None = None
    reasoning: Any = None
    is_error: bool = False


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_creation: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)

    model_config = {"extra": "forbid", "frozen": True}

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_creation=self.cache_creation + other.cache_creation,
            cache_read=self.cache_read + other.cache_read,
        )


StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]


class Reply(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    stop_reason: StopReason = "end_turn"
    model: str | None = None
    reasoning: Any = None


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str
    options: dict[str, str]

    @model_validator(mode="after")
    def _at_least_two(self) -> Choice:
        if len(self.options) < 2:
            raise ValueError("models: a choice needs at least two options")
        return self


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: str
    levels: list[str]

    @model_validator(mode="after")
    def _at_least_two(self) -> Score:
        if len(self.levels) < 2:
            raise ValueError("models: a score needs at least two levels")
        return self


class Check(BaseModel):
    type: Literal["check"] = "check"
    instructions: str
    yes: str | None = None
    no: str | None = None


Question = Choice | Score | Check


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = 0.0


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: list[float]
    confidence: float = 0.0


class CheckAnswer(BaseModel):
    type: Literal["check"] = "check"
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = 0.0


Answer = ChoiceAnswer | ScoreAnswer | CheckAnswer


class Decided(BaseModel):
    answers: dict[str, Answer]
    usage: Usage | None = None
    model: str | None = None


def as_decided(value: object) -> Decided:
    try:
        if isinstance(value, Mapping):
            return Decided.model_validate({"answers": value}, from_attributes=True)
        return Decided.model_validate(value, from_attributes=True)
    except ValidationError as exc:
        raise ModelError(f"models: the decision model's answers do not fit the contract: {exc.errors(include_url=False)}") from exc


def as_reply(value: object) -> Reply:
    try:
        return Reply.model_validate(value, from_attributes=True)
    except ValidationError as exc:
        raise ModelError(f"models: the generation model's reply does not fit the contract: {exc.errors(include_url=False)}") from exc


def confidence(probabilities: list[float]) -> float:
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, (n * max(probabilities) - 1) / (n - 1))


def settle(questions: dict[str, Question], answers: dict[str, Answer]) -> dict[str, Answer]:
    settled: dict[str, Answer] = {}
    for qid, question in questions.items():
        answer = answers.get(qid)
        if answer is None:
            raise ModelError(f"models: no answer for question {qid!r}")
        if answer.type != question.type:
            raise ModelError(f"models: answer to {qid!r} is {answer.type}, question is {question.type}")
        if isinstance(answer, ChoiceAnswer) and isinstance(question, Choice):
            if set(answer.probabilities) != set(question.options):
                raise ModelError(f"models: probabilities for {qid!r} do not cover the options")
            if answer.choice not in question.options:
                raise ModelError(f"models: choice {answer.choice!r} is not an option of {qid!r}")
            probabilities = list(answer.probabilities.values())
        elif isinstance(answer, ScoreAnswer) and isinstance(question, Score):
            if len(answer.probabilities) != len(question.levels):
                raise ModelError(f"models: probabilities for {qid!r} do not cover the levels")
            probabilities = list(answer.probabilities)
        elif isinstance(answer, CheckAnswer):
            probabilities = [answer.probability, 1.0 - answer.probability]
        else:
            raise ModelError(f"models: answer to {qid!r} does not match its question")
        if abs(sum(probabilities) - 1.0) > 1e-3:
            raise ModelError(f"models: probabilities for {qid!r} do not sum to 1")
        settled[qid] = answer.model_copy(update={"confidence": confidence(probabilities)})
    return settled


class ToolCallLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def args(self) -> dict[str, Any]: ...


class UsageLike(Protocol):
    @property
    def input(self) -> int: ...
    @property
    def output(self) -> int: ...
    @property
    def cache_creation(self) -> int: ...
    @property
    def cache_read(self) -> int: ...


class ReplyLike(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def tool_calls(self) -> Sequence[ToolCallLike]: ...
    @property
    def usage(self) -> UsageLike | None: ...
    @property
    def stop_reason(self) -> StopReason: ...
    @property
    def model(self) -> str | None: ...
    @property
    def reasoning(self) -> Any: ...


class ChoiceAnswerLike(Protocol):
    @property
    def type(self) -> Literal["choice"]: ...
    @property
    def choice(self) -> str: ...
    @property
    def probabilities(self) -> dict[str, float]: ...


class ScoreAnswerLike(Protocol):
    @property
    def type(self) -> Literal["score"]: ...
    @property
    def score(self) -> float: ...
    @property
    def probabilities(self) -> list[float]: ...


class CheckAnswerLike(Protocol):
    @property
    def type(self) -> Literal["check"]: ...
    @property
    def probability(self) -> float: ...


AnswerLike = ChoiceAnswerLike | ScoreAnswerLike | CheckAnswerLike


class DecidedLike(Protocol):
    @property
    def answers(self) -> Mapping[str, AnswerLike]: ...
    @property
    def usage(self) -> UsageLike | None: ...
    @property
    def model(self) -> str | None: ...


@runtime_checkable
class GenerationModel(Protocol):
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ReplyLike: ...


@runtime_checkable
class DecisionModel(Protocol):
    async def decide(self, state: State, questions: dict[str, Question]) -> Mapping[str, AnswerLike] | DecidedLike: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    model: str
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


Kind = Literal["chunk", "story"]
Ref = tuple[str, str]
Leg = tuple[str, str, float]


class Section(BaseModel):
    title: str | None = None
    text: str
    locator: dict[str, Any] = Field(default_factory=dict)


class Source(BaseModel):
    corpus_id: str
    source_id: str
    name: str
    metadata: dict[str, Any]
    text: str
    created_at: datetime
    chunks: int


class Chunk(BaseModel):
    corpus_id: str
    chunk_id: str
    source_id: str
    position: int
    title: str | None
    locator: dict[str, Any]
    text: str


class Story(BaseModel):
    corpus_id: str
    story_id: str
    text: str
    subject: str | None
    occurred_at: datetime | None
    created_at: datetime
    valid_to: datetime | None
    supersedes: str | None
    source_ids: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def valid_from(self) -> datetime:
        return self.occurred_at or self.created_at

    @model_validator(mode="after")
    def _interval(self) -> Story:
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("models: valid_to must be after the story became true")
        return self


class Entity(BaseModel):
    corpus_id: str
    entity_id: str
    name: str
    key: str
    type: str
    aliases: tuple[str, ...] = ()


class Relation(BaseModel):
    corpus_id: str
    from_id: str
    to_id: str
    type: str
    evidence_kind: Kind
    evidence_id: str


class EntityIn(BaseModel):
    name: str
    type: str


class RelationIn(BaseModel):
    from_name: str
    to_name: str
    type: str
    from_type: str | None = None
    to_type: str | None = None


class Judgement(BaseModel):
    relevant: float
    evidence: float
    contradicts: float
    injection: float


class Hit(BaseModel):
    kind: Kind
    id: str
    text: str
    score: float
    provenance: dict[str, float]
    chunk: Chunk | None = None
    story: Story | None = None
    judge: Judgement | None = None


class PillarReport(BaseModel):
    returned: int
    duration_ms: int
    error: str | None = None


class Hits(BaseModel):
    hits: list[Hit]
    pillars: dict[str, PillarReport]
    usage: Usage
    duration_ms: int


class Ingested(BaseModel):
    source: Source
    chunks: int
    entities: int
    relations: int
    usage: Usage
    duration_ms: int


class Remembered(BaseModel):
    story: Story
    usage: Usage


class Deleted(BaseModel):
    sources: int = 0
    chunks: int = 0
    stories: int = 0
    entities: int = 0
    relations: int = 0
    mentions: int = 0


class Residue(BaseModel):
    sources: int
    chunks: int
    stories: int
    entities: int
    relations: int
    mentions: int

    @property
    def clean(self) -> bool:
        return not any(self.model_dump().values())


class Neighborhood(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
    passages: list[Hit]


class Window(BaseModel):
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
