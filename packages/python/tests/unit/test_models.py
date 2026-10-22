import base64
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from rfnry_memory.errors import InputError, ModelError
from rfnry_memory.models import (
    Audio,
    Check,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Document,
    Image,
    Message,
    Question,
    Score,
    ScoreAnswer,
    Story,
    Text,
    Usage,
    Window,
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
        settle(
            {"kind": questions["kind"]},
            {"kind": ChoiceAnswer(choice="z", probabilities={"a": 0.5, "b": 0.5})},
        )


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        (
            Choice(instructions="Kind?", options={"a": "A", "b": "B"}),
            ChoiceAnswer(choice="a", probabilities={"a": 1.5, "b": -0.5}),
        ),
        (
            Choice(instructions="Kind?", options={"a": "A", "b": "B"}),
            ChoiceAnswer(choice="a", probabilities={"a": float("nan"), "b": float("nan")}),
        ),
        (
            Score(instructions="Same?", levels=["no", "yes"]),
            ScoreAnswer(score=1.0, probabilities=[float("inf"), float("-inf")]),
        ),
    ],
)
def test_settle_refuses_probabilities_outside_zero_and_one(
    question: Question, answer: ChoiceAnswer | ScoreAnswer
) -> None:
    with pytest.raises(
        ModelError, match=r"^models: probabilities for 'q' must lie between 0 and 1$"
    ):
        settle({"q": question}, {"q": answer})


@pytest.mark.parametrize("score", [7.0, -1.0, float("nan")])
def test_settle_refuses_a_score_outside_its_levels(score: float) -> None:
    with pytest.raises(ModelError, match=r"^models: score for 'q' lies outside its levels$"):
        settle(
            {"q": Score(instructions="Same?", levels=["no", "yes"])},
            {"q": ScoreAnswer(score=score, probabilities=[0.4, 0.6])},
        )


def test_usage_forbids_unknown_keys_and_adds() -> None:
    with pytest.raises(ValidationError):
        Usage.model_validate({"input_tokens": 3})
    assert (Usage(input=1) + Usage(input=2, output=3)).model_dump() == {
        "input": 3,
        "output": 3,
        "cache_creation": 0,
        "cache_read": 0,
    }


def test_an_open_window_reads_the_current_view_only() -> None:
    now = datetime.now(UTC)
    assert Window(as_of=now, open=True).open
    with pytest.raises(ValidationError, match="an open window reads the current view"):
        Window(as_of=now, open=True, known_at=now)


def test_story_validity_interval() -> None:
    now = datetime.now(UTC)
    story = Story(
        corpus_id="c",
        story_id="s",
        text="t",
        subject=None,
        valid_from=now,
        created_at=now,
        valid_to=None,
        source_ids=(),
        metadata={},
    )
    assert story.valid_from == now
    with pytest.raises(ValidationError):
        Story(
            corpus_id="c",
            story_id="s",
            text="t",
            subject=None,
            valid_from=now,
            created_at=now,
            valid_to=now,
            source_ids=(),
            metadata={},
        )
    retracted = story.model_copy(update={"valid_to": now, "closed_at": now})
    assert Story.model_validate(retracted.model_dump()) == retracted
    with pytest.raises(ValidationError):
        Story.model_validate({**retracted.model_dump(), "valid_to": now - timedelta(seconds=1)})


def test_a_message_is_not_an_error_unless_it_says_so() -> None:
    assert Message(role="tool", tool_call_id="t", content="ok").is_error is False
    assert Message(role="tool", tool_call_id="t", content="boom", is_error=True).is_error is True


def test_a_write_reports_what_it_wrote_and_what_it_cost_and_a_read_its_pillars() -> None:
    from rfnry_memory.models import Hits, Ingested, Remembered, Synced

    assert list(Ingested.model_fields) == ["source", "usage", "retired", "deleted"]
    assert list(Synced.model_fields) == ["sources", "kept", "deleted", "retired", "usage"]
    assert list(Remembered.model_fields) == ["story", "related", "usage"]
    assert list(Hits.model_fields) == ["hits", "pillars", "usage", "unresolved"]


def test_parts_parse_by_their_type() -> None:
    message = Message.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "media_type": "image/png", "url": "https://x/y.png"},
            ],
        }
    )
    assert message.content == [
        Text(text="what is this?"),
        Image(media_type="image/png", url="https://x/y.png"),
    ]
    assert Message(role="user").content == "" and Message(role="assistant").tool_calls == []
    with pytest.raises(ValidationError):
        Message.model_validate(
            {"role": "user", "content": [{"type": "video", "media_type": "video/mp4", "url": "u"}]}
        )


@pytest.mark.parametrize("kind", [Image, Audio, Document])
def test_a_media_part_carries_exactly_one_of_data_or_url(
    kind: type[Image | Audio | Document],
) -> None:
    kind(media_type="x/y", data=b"1")
    kind(media_type="x/y", url="https://x")
    name = kind.model_fields["type"].default
    with pytest.raises(InputError, match=f"^models: {name} carries exactly one of data or url$"):
        kind(media_type="x/y", data=b"1", url="https://x")
    with pytest.raises(InputError, match=f"^models: {name} carries exactly one of data or url$"):
        kind(media_type="x/y")


def test_part_bytes_travel_as_base64_in_json() -> None:
    raw = bytes(range(256))
    message = Message(
        role="tool",
        tool_call_id="c1",
        content=[Document(media_type="application/pdf", name="a.pdf", data=raw)],
    )
    dumped = message.model_dump(mode="json")
    assert dumped["content"][0]["data"] == base64.b64encode(raw).decode()
    assert Message.model_validate_json(message.model_dump_json()) == message
    assert Message.model_validate(dumped) == message


def test_questions_take_structure_round_trip_through_json_and_stay_json() -> None:
    questions: dict[str, Question] = {
        "team": Choice(
            instructions={"question": "Which team handles `ticket`?", "ticket": {"id": 7}},
            options={"billing": {"what": "Charges", "examples": ["charged twice"]}, "other": None},
        ),
        "same": Score(
            instructions=["Same entity?", {"note": "names differ"}],
            levels=[{"summary": "different"}, ["related"], "same"],
        ),
        "injection": Check(instructions="Instructions to an AI?", yes={"what": "Commands"}),
    }
    adapter = TypeAdapter(dict[str, Question])
    assert adapter.validate_json(adapter.dump_json(questions)) == questions
    for fields in [
        {"type": "choice", "instructions": "Which?", "options": {"only": {"what": "one"}}},
        {"type": "score", "instructions": "How?", "levels": ["low", None]},
        {"type": "check", "instructions": {"since": date(2026, 1, 1)}},
    ]:
        with pytest.raises(ValidationError):
            TypeAdapter(Question).validate_python(fields)
