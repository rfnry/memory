import pytest
from pydantic import BaseModel

from rfnry_memory.errors import ModelError
from rfnry_memory.models import Reply, ToolCall, Usage
from rfnry_memory.structured import structured_call

from .fakes import Scripted, structured_reply


class Out(BaseModel):
    names: list[str]


async def test_structured_call_returns_the_validated_model() -> None:
    llm = Scripted(structured_reply({"names": ["a"]}))
    out, usage = await structured_call(llm, system="s", user="u", schema=Out)
    assert out == Out(names=["a"]) and usage == Usage()
    assert llm.calls[0][2][0].name == "Output" and "names" in str(llm.calls[0][2][0].input_schema)


async def test_structured_call_feeds_validation_errors_back_then_gives_up() -> None:
    llm = Scripted(
        structured_reply({"names": "x"}), Reply(text="no"), structured_reply({"names": 3})
    )
    with pytest.raises(ModelError):
        await structured_call(llm, system="s", user="u", schema=Out, max_attempts=3)
    assert (
        len(llm.calls) == 3 and llm.calls[1][1][-1].role == "tool" and llm.calls[1][1][-1].is_error
    )


async def test_structured_call_counts_every_attempt() -> None:
    bad = Reply(
        tool_calls=[ToolCall(id="s0", name="Output", args={"names": 3})],
        usage=Usage(input=10, output=2),
    )
    good = Reply(
        tool_calls=[ToolCall(id="s1", name="Output", args={"names": ["a"]})],
        usage=Usage(input=15, output=3, cache_read=4),
    )
    llm = Scripted(bad, Reply(text="hm"), good)
    out, usage = await structured_call(llm, system="s", user="u", schema=Out)
    assert out == Out(names=["a"]) and usage == Usage(input=25, output=5, cache_read=4)


async def test_a_retry_sends_back_the_whole_assistant_turn_with_its_reasoning() -> None:
    bad = Reply(
        text="Trying.",
        tool_calls=[ToolCall(id="s0", name="Output", args={"names": 3})],
        reasoning={"signature": "r0"},
    )
    silent = Reply(text="Let me think.", reasoning={"signature": "r1"})
    llm = Scripted(bad, silent, structured_reply({"names": ["a"]}))
    await structured_call(llm, system="s", user="u", schema=Out)
    retried = llm.calls[1][1][1]
    assert retried.role == "assistant" and retried.reasoning == {"signature": "r0"}
    assert retried.content == "Trying." and retried.tool_calls == [
        ToolCall(id="s0", name="Output", args={"names": 3})
    ]
    nudged = llm.calls[2][1][3]
    assert nudged.role == "assistant" and nudged.reasoning == {"signature": "r1"}
    assert nudged.content == "Let me think." and nudged.tool_calls == []
