import pytest
from pydantic import BaseModel

from rfnry_memory.errors import ModelError
from rfnry_memory.models import Reply
from rfnry_memory.structured import structured_call

from .fakes import Scripted, structured_reply


class Out(BaseModel):
    names: list[str]


async def test_structured_call_returns_the_validated_model() -> None:
    llm = Scripted(structured_reply({"names": ["a"]}))
    out = await structured_call(llm, system="s", user="u", schema=Out)
    assert out == Out(names=["a"])
    assert llm.calls[0][2][0].name == "Output" and "names" in str(llm.calls[0][2][0].input_schema)


async def test_structured_call_feeds_validation_errors_back_then_gives_up() -> None:
    llm = Scripted(structured_reply({"names": "x"}), Reply(text="no"), structured_reply({"names": 3}))
    with pytest.raises(ModelError):
        await structured_call(llm, system="s", user="u", schema=Out, max_attempts=3)
    assert len(llm.calls) == 3 and llm.calls[1][1][-1].role == "tool"
