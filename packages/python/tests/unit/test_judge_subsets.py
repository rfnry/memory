from collections.abc import Mapping
from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import CheckAnswer, QuestionLike
from rfnry_memory.postgres import PostgresStorage

from .fakes import Fixed, HashEmbedding


def _relevance(state: Any, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
    text = state["passage"]["text"]
    return {"relevant": CheckAnswer(probability=0.9 if "gasket" in text else 0.2)}


async def _memory(storage: PostgresStorage, decision: Fixed) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding(), decision=decision)
    await memory.check()
    await memory.ingest("acme", name="a", text="The pump runs all day.")
    await memory.ingest("acme", name="b", text="The gasket on the pump leaks.")
    return memory


async def test_a_single_score_reranker_answers_only_relevant_and_reranks(
    storage: PostgresStorage,
) -> None:
    decision = Fixed(_relevance)
    memory = await _memory(storage, decision)
    hits = await memory.search("acme", "pump", kinds=("chunk",), judge=("relevant",))
    assert all(h.judge is not None and h.judge.evidence is None for h in hits.hits)
    assert [h.chunk.source_name for h in hits.by("relevant").hits if h.chunk] == ["b", "a"]
    assert decision.calls and all(set(q) == {"relevant"} for _, q in decision.calls)
    with pytest.raises(
        InputError,
        match=r"^models: hits were not judged for 'evidence' — ask for it with judge=True or judge=\('evidence',\)$",
    ):
        hits.by("evidence")


@pytest.mark.parametrize(
    "judge", [(), ("relevant", "relevant"), ("score",), "relevant", (["relevant"],)]
)
async def test_judge_names_known_judgements_each_once(storage: PostgresStorage, judge: Any) -> None:
    memory = await _memory(storage, Fixed(_relevance))
    with pytest.raises(
        InputError,
        match=r"^memory: judge is True, False or a tuple of relevant, evidence, contradicts and injection, each once$",
    ):
        await memory.search("acme", "pump", judge=judge)


async def test_remember_judges_related_stories_with_the_named_judgements(
    storage: PostgresStorage,
) -> None:
    decision = Fixed(lambda state, questions: {"contradicts": CheckAnswer(probability=0.8)})
    memory = await _memory(storage, decision)
    await memory.remember("acme", "The pump runs on Mondays.")
    latest = await memory.remember("acme", "The pump runs on Tuesdays.", judge=("contradicts",))
    assert [h.judge.contradicts for h in latest.related.hits if h.judge] == [0.8]
    assert all(set(q) == {"contradicts"} for _, q in decision.calls)


@pytest.mark.parametrize(
    "judge", [(), ("relevant", "relevant"), ("score",), "relevant", (["relevant"],)]
)
async def test_a_write_refuses_an_invalid_judge_before_it_writes(
    storage: PostgresStorage, judge: Any
) -> None:
    decision = Fixed(_relevance)
    memory = await _memory(storage, decision)
    held = await memory.remember("acme", "The pump runs on Mondays.", related=0)
    refused = r"^memory: judge is True, False or a tuple of relevant, evidence, contradicts and injection, each once$"
    with pytest.raises(InputError, match=refused):
        await memory.remember("acme", "The pump runs on Tuesdays.", judge=judge)
    with pytest.raises(InputError, match=refused):
        await memory.supersede(
            "acme", held.story.story_id, "The pump runs on Fridays.", judge=judge
        )
    stories, _ = await memory.list_stories("acme", history=True)
    assert [s.text for s in stories] == ["The pump runs on Mondays."]
    assert decision.calls == []


async def test_supersede_and_recall_ask_only_the_judgements_they_name(
    storage: PostgresStorage,
) -> None:
    decision = Fixed(
        lambda state, questions: {name: CheckAnswer(probability=0.7) for name in questions}
    )
    memory = await _memory(storage, decision)
    held = await memory.remember("acme", "The pump runs on Mondays.")
    await memory.remember("acme", "The pump stops on Sundays.")
    moved = await memory.supersede(
        "acme", held.story.story_id, "The pump runs on Tuesdays.", judge=("contradicts",)
    )
    assert moved.related.hits and all(
        h.judge is not None and h.judge.contradicts == 0.7 and h.judge.relevant is None
        for h in moved.related.hits
    )
    recalled = await memory.recall("acme", "pump", judge=("evidence",))
    assert recalled.hits and all(
        h.judge is not None and h.judge.evidence == 0.7 and h.judge.relevant is None
        for h in recalled.hits
    )
    assert {frozenset(q) for _, q in decision.calls} == {
        frozenset({"contradicts"}),
        frozenset({"evidence"}),
    }
