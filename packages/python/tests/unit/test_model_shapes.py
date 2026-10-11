from collections.abc import Mapping, Sequence
from types import MappingProxyType

from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    AnswerLike,
    Check,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Decided,
    DecidedLike,
    DecisionModel,
    GenerationModel,
    Message,
    QuestionLike,
    Reply,
    ReplyLike,
    Score,
    ScoreAnswer,
    State,
    ToolCall,
    ToolSpec,
    Usage,
    as_decided,
)
from rfnry_memory.storage.postgres import PostgresStorage

from . import lookalike as other
from .fakes import Fixed, Scripted


class Adapter:
    def __init__(self) -> None:
        self.replies = [
            other.Reply(
                tool_calls=[
                    other.ToolCall(
                        id="c1",
                        name="Output",
                        args={
                            "entities": [{"name": "Acme Corp", "type": "company"}],
                            "relations": [],
                        },
                    )
                ],
                stop_reason="tool_use",
                usage=other.Usage(input=3, output=1),
            ),
            other.Reply(
                tool_calls=[
                    other.ToolCall(
                        id="c2",
                        name="Output",
                        args={
                            "entities": [{"name": "ACME Corporation", "type": "company"}],
                            "relations": [],
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
        ]

    async def generate(
        self,
        system: str,
        messages: Sequence[Message | other.Message],
        tools: Sequence[ToolSpec | other.ToolSpec],
    ) -> other.Reply:
        return self.replies.pop(0)

    async def decide(
        self, state: State, questions: Mapping[str, other.QuestionLike]
    ) -> other.Decided:
        answers: dict[str, other.Answer] = {
            "same": other.ScoreAnswer(score=2, probabilities=[0.05, 0.05, 0.9])
        }
        return other.Decided(answers=answers, usage=other.Usage(input=7), model="d")


async def test_an_adapter_typed_against_lookalike_classes_runs_in_memory(
    storage: PostgresStorage,
) -> None:
    adapter = Adapter()
    memory = Memory(storage, generation=adapter, decision=adapter, relation_types=["OWNS"])
    await memory.initialize()
    first = await memory.ingest("acme", name="a", text="Acme Corp signed.")
    second = await memory.remember("acme", "ACME Corporation grew.")
    assert first.usage == Usage(input=3, output=1) and second.usage == Usage(input=7)
    [entity] = await memory.storage.entities_named("acme", "acme corporation")
    assert entity.name == "Acme Corp" and entity.aliases == ("ACME Corporation",)


def test_the_memory_classes_have_the_shapes_memory_reads() -> None:
    reply: ReplyLike = Reply(tool_calls=[ToolCall(id="c1", name="Output")], usage=Usage())
    answers: Mapping[str, AnswerLike] = {
        "a": ChoiceAnswer(choice="x", probabilities={"x": 1.0}),
        "b": ScoreAnswer(score=0, probabilities=[1.0]),
        "c": CheckAnswer(probability=1.0),
    }
    decided: DecidedLike = Decided(answers={"c": CheckAnswer(probability=1.0)})
    questions: Mapping[str, QuestionLike] = {
        "a": Choice(instructions="Kind?", options={"x": "X", "y": "Y"}),
        "b": Score(instructions="Same?", levels=["no", "yes"]),
        "c": Check(instructions="Relevant?"),
        "d": other.Check(instructions="Relevant?"),
        "e": Choice(instructions={"question": "Kind?"}, options={"x": {"what": "X"}, "y": None}),
        "f": other.Score(instructions=["Same?"], levels=[{"summary": "no"}, ["yes"]]),
        "g": other.Check(instructions={"question": "Relevant?"}, yes=["on topic"]),
    }
    generation: GenerationModel = Scripted()
    decision: DecisionModel = Fixed({})
    assert reply.tool_calls[0].name == "Output" and decided.answers and len(answers) == 3
    assert [q.type for q in questions.values()] == [
        "choice",
        "score",
        "check",
        "check",
        "choice",
        "score",
        "check",
    ]
    assert generation and decision


def test_answers_alone_are_read_from_any_mapping() -> None:
    answers = MappingProxyType({"c": other.CheckAnswer(probability=0.25)})
    assert as_decided(answers).answers["c"] == CheckAnswer(probability=0.25)
