import re
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Answer, CheckAnswer, QuestionLike, State
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery import MemoryRefinery, Relevance, RetrievalOutcome, RetrievalReview

from .fakes import Fixed, HashEmbedding

MANUAL = "Replace the pump seal yearly, after draining the line."
NOTES = "The pump seal on line B was replaced in March."


async def _recorded(
    storage: PostgresStorage, **options: Any
) -> tuple[Memory, MemoryRefinery, str, str]:
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery, **options)
    await memory.check()
    manual = (await memory.ingest("acme", name="manual", text=MANUAL)).source.source_id
    notes = (await memory.ingest("acme", name="notes", text=NOTES)).source.source_id
    return memory, refinery, manual, notes


async def test_a_person_and_the_world_label_a_retrieval_and_a_miss_counts(
    storage: PostgresStorage,
) -> None:
    memory, refinery, manual, notes = await _recorded(storage)
    hits = await memory.search("acme", "pump seal", kinds=("chunk",))
    assert hits.retrieval_id is not None
    review = RetrievalReview(
        relevance=[
            Relevance(corpus_id="acme", kind="chunk", id=f"{manual}-0", grade=3),
            Relevance(corpus_id="acme", kind="chunk", id=f"{notes}-0", grade=0),
            Relevance(corpus_id="acme", kind="source", id="handbook", grade=2),
        ],
        comment="The handbook answers it too.",
    )
    person = await refinery.review("acme", hits.retrieval_id, review)
    assert (person.source, person.review, person.retrieval_id) == (
        "human",
        review,
        hits.retrieval_id,
    )
    world = await refinery.outcome(
        "acme",
        hits.retrieval_id,
        RetrievalOutcome(
            status="succeeded", cited=[Relevance(corpus_id="acme", kind="chunk", id=f"{manual}-0")]
        ),
    )
    assert (
        world.source == "world" and world.outcome is not None and world.outcome.cited[0].grade == 1
    )
    labels = await refinery.labels("acme", hits.retrieval_id)
    assert [label.label_id for label in labels] == [person.label_id, world.label_id]


async def test_a_label_id_is_idempotent_within_its_retrieval(storage: PostgresStorage) -> None:
    memory, refinery, manual, _ = await _recorded(storage)
    first = (await memory.search("acme", "pump seal")).retrieval_id
    second = (await memory.search("acme", "seal")).retrieval_id
    assert first is not None and second is not None
    useful = RetrievalReview(
        relevance=[Relevance(corpus_id="acme", kind="chunk", id=f"{manual}-0")]
    )
    told = await refinery.review("acme", first, useful, label_id="message-7")
    again = await refinery.review(
        "acme", first, RetrievalReview(comment="changed my mind"), label_id="message-7"
    )
    assert again == told and len(await refinery.labels("acme", first)) == 1
    elsewhere = await refinery.review("acme", second, useful, label_id="message-7")
    assert elsewhere.retrieval_id == second


async def test_what_a_label_refuses(storage: PostgresStorage) -> None:
    memory, refinery, manual, _ = await _recorded(storage)
    retrieval_id = (await memory.search("acme", "pump seal")).retrieval_id
    assert retrieval_id is not None
    outside = RetrievalReview(relevance=[Relevance(corpus_id="ops", kind="chunk", id="x", grade=1)])
    with pytest.raises(
        InputError,
        match=re.escape(
            f"refinery: retrieval {retrieval_id!r} read acme; a grade naming 'ops' is not one of them"
        ),
    ):
        await refinery.review("acme", retrieval_id, outside)
    with pytest.raises(NotFoundError, match=r"^refinery: retrieval 'missing' was not found$"):
        await refinery.review("acme", "missing", RetrievalReview(comment="x"))
    with pytest.raises(
        NotFoundError, match=rf"^refinery: retrieval '{retrieval_id}' was not found$"
    ):
        await refinery.labels("ops", retrieval_id)
    with pytest.raises(ValidationError):
        RetrievalReview()
    with pytest.raises(ValidationError):
        RetrievalOutcome()
    with pytest.raises(ValidationError):
        Relevance(corpus_id="acme", kind="chunk", id=f"{manual}-0", grade=4)


def _relevant(state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Answer]:
    passage = str(state)
    return {
        name: CheckAnswer(probability=0.9 if "yearly" in passage else 0.2) for name in questions
    }


async def test_a_judge_grades_what_each_retrieval_returned_once(storage: PostgresStorage) -> None:
    judge = Fixed(_relevant)
    memory, refinery, manual, notes = await _recorded(storage, decision=judge)
    searched = (await memory.search("acme", "pump seal", kinds=("chunk",))).retrieval_id
    judged = (
        await memory.search("acme", "seal", kinds=("chunk",), judge=("relevant",))
    ).retrieval_id
    assert searched is not None and judged is not None
    asked = len(judge.calls)
    labeled = await refinery.label("acme", memory)
    assert (labeled.judged, labeled.errors) == (2, {})
    assert len(judge.calls) - asked == 2
    [label] = await refinery.labels("acme", searched)
    assert label.source == "judge" and label.judged is not None
    graded = {r.id: (r.grade, r.probability) for r in label.judged}
    assert graded == {f"{manual}-0": (1, 0.9), f"{notes}-0": (0, 0.2)}
    assert (await refinery.label("acme", memory)).judged == 0
    plain = Memory(storage, embedding=HashEmbedding())
    with pytest.raises(
        ConfigurationError, match=r"^refinery: label needs a memory with a decision model$"
    ):
        await refinery.label("acme", plain)


async def test_a_label_id_names_one_source(storage: PostgresStorage) -> None:
    memory, refinery, _, _ = await _recorded(storage)
    retrieval_id = (await memory.search("acme", "pump seal")).retrieval_id
    assert retrieval_id is not None
    await refinery.review("acme", retrieval_id, RetrievalReview(comment="ok"), label_id="m1")
    with pytest.raises(
        InputError,
        match=re.escape(
            f"refinery: label 'm1' of retrieval {retrieval_id!r} is a human label, not a world one"
        ),
    ):
        await refinery.outcome(
            "acme", retrieval_id, RetrievalOutcome(status="succeeded"), label_id="m1"
        )


async def test_a_judge_labels_a_retrieval_without_hits_empty_once(storage: PostgresStorage) -> None:
    judge = Fixed(_relevant)
    memory, refinery, _, _ = await _recorded(storage, decision=judge)
    empty = await memory.search("acme", "zzz", pillars={"text"})
    assert empty.hits == [] and empty.retrieval_id is not None
    labeled = await refinery.label("acme", memory)
    assert (labeled.judged, labeled.skipped, labeled.errors) == (0, 1, {})
    assert judge.calls == []
    [label] = await refinery.labels("acme", empty.retrieval_id)
    assert (label.source, label.judged) == ("judge", [])
    assert (await refinery.label("acme", memory)).skipped == 0
    measured = await refinery.measure("acme", holdout=None, sources=("judge",))
    assert measured.retrievals == 0


class Judge(Fixed):
    model = "judge-1"
    settings = {"effort": "low"}


async def test_a_judge_label_names_the_model_and_settings_of_each_grade(
    storage: PostgresStorage,
) -> None:
    memory, refinery, manual, _ = await _recorded(storage, decision=Judge(_relevant))
    reused = (
        await memory.search("acme", "seal", kinds=("chunk",), judge=("relevant",))
    ).retrieval_id
    fresh = (await memory.search("acme", "pump seal", kinds=("chunk",))).retrieval_id
    assert reused is not None and fresh is not None
    await refinery.review(
        "acme",
        fresh,
        RetrievalReview(relevance=[Relevance(corpus_id="acme", kind="chunk", id=f"{manual}-0")]),
    )
    assert (await refinery.label("acme", memory)).judged == 2
    for retrieval_id in (reused, fresh):
        labels = await refinery.labels("acme", retrieval_id)
        label = next(item for item in labels if item.source == "judge")
        assert label.judged
        assert all(r.model == "judge-1" and r.settings == {"effort": "low"} for r in label.judged)
    [human] = [label for label in await refinery.labels("acme", fresh) if label.source == "human"]
    assert human.review is not None
    assert all(r.model is None and r.settings is None for r in human.review.relevance)
