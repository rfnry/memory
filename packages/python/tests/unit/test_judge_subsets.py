from collections.abc import Mapping
from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import CheckAnswer, QuestionLike
from rfnry_memory.storage.inmemory import InMemoryStorage

from .fakes import Fixed, HashEmbedding


def _relevance(state: Any, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
    text = state["passage"]["text"]
    return {"relevant": CheckAnswer(probability=0.9 if "gasket" in text else 0.2)}


async def _memory(decision: Fixed) -> Memory:
    memory = Memory(InMemoryStorage(), embedding=HashEmbedding(), decision=decision)
    await memory.initialize()
    await memory.ingest("acme", name="a", text="The pump runs all day.")
    await memory.ingest("acme", name="b", text="The gasket on the pump leaks.")
    return memory


async def test_a_single_score_reranker_answers_only_relevant_and_reranks() -> None:
    decision = Fixed(_relevance)
    memory = await _memory(decision)
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
async def test_judge_names_known_judgements_each_once(judge: Any) -> None:
    memory = await _memory(Fixed(_relevance))
    with pytest.raises(
        InputError,
        match=r"^memory: judge is True, False or a tuple of relevant, evidence, contradicts and injection, each once$",
    ):
        await memory.search("acme", "pump", judge=judge)


async def test_remember_judges_related_stories_with_the_named_judgements() -> None:
    decision = Fixed(lambda state, questions: {"contradicts": CheckAnswer(probability=0.8)})
    memory = await _memory(decision)
    await memory.remember("acme", "The pump runs on Mondays.")
    latest = await memory.remember("acme", "The pump runs on Tuesdays.", judge=("contradicts",))
    assert [h.judge.contradicts for h in latest.related.hits if h.judge] == [0.8]
    assert all(set(q) == {"contradicts"} for _, q in decision.calls)
