from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rfnry_memory.errors import ModelError
from rfnry_memory.models import (
    Check,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Message,
    Question,
    Score,
    ScoreAnswer,
    Story,
    Usage,
    confidence,
    settle,
)


def test_confidence_is_peakedness() -> None:
    assert confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert confidence([0.8, 0.2]) == pytest.approx(0.6)


def test_settle_fills_confidence_and_rejects_violations() -> None:
    questions: dict[str, Question] = {
        "same": Score(instructions="Same?", levels=["different", "related", "same"]),
        "relevant": Check(instructions="Relevant?"),
        "kind": Choice(instructions="Kind?", options={"a": "A", "b": "B"}),
    }
    settled = settle(
        questions,
        {
            "same": ScoreAnswer(score=1.7, probabilities=[0.1, 0.2, 0.7]),
            "relevant": CheckAnswer(probability=0.9),
            "kind": ChoiceAnswer(choice="b", probabilities={"a": 0.25, "b": 0.75}),
        },
    )
    assert settled["same"].confidence == pytest.approx(0.55)
    assert settled["relevant"].confidence == pytest.approx(0.8)
    assert settled["kind"].confidence == pytest.approx(0.5)
    with pytest.raises(ModelError):
        settle(questions, {})
    with pytest.raises(ModelError):
        settle({"kind": questions["kind"]}, {"kind": ChoiceAnswer(choice="z", probabilities={"a": 0.5, "b": 0.5})})


def test_usage_forbids_unknown_keys_and_adds() -> None:
    with pytest.raises(ValidationError):
        Usage.model_validate({"input_tokens": 3})
    assert (Usage(input=1) + Usage(input=2, output=3)).model_dump() == {"input": 3, "output": 3, "cache_creation": 0, "cache_read": 0}


def test_story_validity_interval() -> None:
    now = datetime.now(UTC)
    story = Story(corpus_id="c", story_id="s", text="t", subject=None, valid_from=now, created_at=now, valid_to=None, supersedes=None, source_ids=(), metadata={})
    assert story.valid_from == now
    with pytest.raises(ValidationError):
        Story(corpus_id="c", story_id="s", text="t", subject=None, valid_from=now, created_at=now, valid_to=now, supersedes=None, source_ids=(), metadata={})


def test_a_message_is_not_an_error_unless_it_says_so() -> None:
    assert Message(role="tool", tool_call_id="t", content="ok").is_error is False
    assert Message(role="tool", tool_call_id="t", content="boom", is_error=True).is_error is True


def test_a_write_reports_what_it_wrote_and_what_it_cost_and_a_read_its_pillars() -> None:
    from rfnry_memory.models import Hits, Ingested, Remembered

    assert list(Ingested.model_fields) == ["source", "usage"]
    assert list(Remembered.model_fields) == ["story", "usage"]
    assert list(Hits.model_fields) == ["hits", "pillars", "usage"]
