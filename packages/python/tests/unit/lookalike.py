from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field, JsonValue


class Text(BaseModel):
    type: Literal["text"] = "text"
    text: str


class Image(BaseModel):
    type: Literal["image"] = "image"
    media_type: str
    data: bytes | None = None
    url: str | None = None


Part = Annotated[Text | Image, Field(discriminator="type")]
Content = str | list[Part]


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: Content = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    reasoning: Any = None
    is_error: bool = False


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_creation: int = 0
    cache_read: int = 0

    model_config = {"extra": "forbid", "frozen": True}


class Reply(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "error"] = "end_turn"
    model: str | None = None
    reasoning: Any = None


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    options: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None]


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    levels: list[str | dict[str, JsonValue] | list[JsonValue]]


class Check(BaseModel):
    type: Literal["check"] = "check"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    yes: str | dict[str, JsonValue] | list[JsonValue] | None = None
    no: str | dict[str, JsonValue] | list[JsonValue] | None = None


Question = Choice | Score | Check


class ChoiceLike(Protocol):
    @property
    def type(self) -> Literal["choice"]: ...
    @property
    def instructions(self) -> str | dict[str, JsonValue] | list[JsonValue]: ...
    @property
    def options(self) -> Mapping[str, str | dict[str, JsonValue] | list[JsonValue] | None]: ...


class ScoreLike(Protocol):
    @property
    def type(self) -> Literal["score"]: ...
    @property
    def instructions(self) -> str | dict[str, JsonValue] | list[JsonValue]: ...
    @property
    def levels(self) -> Sequence[str | dict[str, JsonValue] | list[JsonValue]]: ...


class CheckLike(Protocol):
    @property
    def type(self) -> Literal["check"]: ...
    @property
    def instructions(self) -> str | dict[str, JsonValue] | list[JsonValue]: ...
    @property
    def yes(self) -> str | dict[str, JsonValue] | list[JsonValue] | None: ...
    @property
    def no(self) -> str | dict[str, JsonValue] | list[JsonValue] | None: ...


QuestionLike = ChoiceLike | ScoreLike | CheckLike


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
    probability: float
    confidence: float = 0.0


Answer = ChoiceAnswer | ScoreAnswer | CheckAnswer


class Decided(BaseModel):
    answers: dict[str, Answer]
    usage: Usage | None = None
    model: str | None = None
