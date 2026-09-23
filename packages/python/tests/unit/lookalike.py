from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
    instructions: str
    options: dict[str, str]


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: str
    levels: list[str]


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
    probability: float
    confidence: float = 0.0


Answer = ChoiceAnswer | ScoreAnswer | CheckAnswer


class Decided(BaseModel):
    answers: dict[str, Answer]
    usage: Usage | None = None
    model: str | None = None
