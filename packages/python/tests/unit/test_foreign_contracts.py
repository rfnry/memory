from itertools import count
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from rfnry_memory.errors import ModelError
from rfnry_memory.memory import Memory
from rfnry_memory.models import EntityIn, Message, ToolSpec, Usage
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.structured import structured_call

from .fakes import HashEmbedding


class OtherUsage(BaseModel):
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_creation: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)


class OtherCheckAnswer(BaseModel):
    type: Literal["check"] = "check"
    probability: float
    confidence: float = 0.0


class OtherScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: list[float]
    confidence: float = 0.0


class OtherDecided(BaseModel):
    answers: dict[str, OtherCheckAnswer | OtherScoreAnswer]
    usage: OtherUsage | None = None
    model: str | None = None


class OtherToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class OtherReply(BaseModel):
    text: str = ""
    tool_calls: list[OtherToolCall] = Field(default_factory=list)
    usage: OtherUsage | None = None
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "error"] = "end_turn"
    model: str | None = None
    reasoning: Any = None


class OtherDecision:
    def __init__(self, answer: Any) -> None:
        self.answer = answer

    async def decide(self, state: Any, questions: Any) -> Any:
        return self.answer(state, questions)


class OtherGeneration:
    def __init__(self, *replies: OtherReply) -> None:
        self.replies = list(replies)

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Any:
        return self.replies.pop(0)


def checks(probability: float) -> dict[str, OtherCheckAnswer | OtherScoreAnswer]:
    return {
        name: OtherCheckAnswer(probability=probability)
        for name in ("relevant", "evidence", "contradicts", "injection")
    }


_CORPORA = count()


async def judged(storage: PostgresStorage, decision: Any) -> Any:
    memory = Memory(storage, embedding=HashEmbedding(), decision=decision)
    await memory.check()
    corpus_id = f"acme-{next(_CORPORA)}"
    await memory.remember(corpus_id, "Replace the pump seal yearly.")
    return await memory.search(corpus_id, "pump seal", judge=True)


async def test_the_judge_accepts_another_librarys_decided_and_plain_answers(
    storage: PostgresStorage,
) -> None:
    hits = await judged(
        storage,
        OtherDecision(lambda s, q: OtherDecided(answers=checks(0.75), usage=OtherUsage(input=7))),
    )
    assert hits.hits[0].judge is not None and hits.hits[0].judge.relevant == 0.75
    assert hits.usage == Usage(input=7 * len(hits.hits))
    hits = await judged(storage, OtherDecision(lambda s, q: checks(0.25)))
    assert (
        hits.hits[0].judge is not None
        and hits.hits[0].judge.injection == 0.25
        and hits.usage == Usage()
    )


async def test_foreign_answers_keep_strict_value_validation(storage: PostgresStorage) -> None:
    with pytest.raises(ModelError, match=r"^models: "):
        await judged(storage, OtherDecision(lambda s, q: OtherDecided(answers=checks(1.5))))
    with pytest.raises(ModelError, match=r"^models: "):
        await judged(
            storage,
            OtherDecision(
                lambda s, q: {"relevant": OtherScoreAnswer(score=1, probabilities=[1.0])}
            ),
        )
    with pytest.raises(ModelError, match=r"^models: "):
        await judged(storage, OtherDecision(lambda s, q: "yes"))


async def test_identity_accepts_another_librarys_decided(storage: PostgresStorage) -> None:
    same = OtherScoreAnswer(score=1.9, probabilities=[0.05, 0.05, 0.9])
    decision = OtherDecision(
        lambda s, q: OtherDecided(answers={"same": same}, usage=OtherUsage(input=30))
    )
    memory = Memory(storage, decision=decision, pillars={"text", "graph"})
    await memory.remember("acme", "x", entities=[EntityIn(name="Acme Corp", type="company")])
    ingested = await memory.ingest(
        "acme", name="press", text="y", entities=[EntityIn(name="ACME Corporation", type="company")]
    )
    assert (
        ingested.usage == Usage(input=30)
        and (await memory.storage.count_corpus("acme")).entities == 1
    )


class Out(BaseModel):
    names: list[str]


async def test_a_structured_call_accepts_another_librarys_reply() -> None:
    call = OtherToolCall(id="s1", name="Output", args={"names": ["a"]})
    llm = OtherGeneration(
        OtherReply(tool_calls=[call], stop_reason="tool_use", usage=OtherUsage(output=4))
    )
    out, usage, _ = await structured_call(llm, system="s", user="u", schema=Out)
    assert (out, usage) == (Out(names=["a"]), Usage(output=4))


async def test_a_reply_that_is_not_reply_shaped_is_a_model_error() -> None:
    llm = OtherGeneration(OtherReply.model_construct(tool_calls="nope"))
    with pytest.raises(ModelError, match=r"^models: "):
        await structured_call(llm, system="s", user="u", schema=Out)
