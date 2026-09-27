from datetime import UTC, datetime
from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError, InputError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import EntityIn, RelationIn
from rfnry_memory.storage.inmemory import InMemoryStorage

from .fakes import HashEmbedding


async def _seeded(**kwargs: Any) -> Memory:
    memory = Memory(
        InMemoryStorage(), embedding=HashEmbedding(), chunk_chars=200, overlap_chars=20, **kwargs
    )
    await memory.initialize()
    await memory.ingest(
        "acme",
        name="manual",
        text="# Seals\n\nReplace the pump seal yearly.\n\n# Wiring\n\nGround the panel first.",
        metadata={"line": "A"},
    )
    await memory.remember(
        "acme",
        "The pump seal on P-3 was replaced in March.",
        subject="asset:P-3",
        valid_from=datetime(2026, 3, 5, tzinfo=UTC),
    )
    await memory.remember(
        "acme", "Ana owns the wiring review.", subject="user:ana", metadata={"team": "ops"}
    )
    return memory


async def test_search_fuses_text_and_vector_with_provenance() -> None:
    memory = await _seeded()
    hits = await memory.search("acme", "pump seal")
    assert (
        hits.hits
        and hits.hits[0].kind in ("chunk", "story")
        and "seal" in hits.hits[0].text.lower()
    )
    assert {"text", "vector"} <= set(hits.pillars) and hits.pillars["text"].error is None
    top = hits.hits[0]
    assert "text" in top.provenance and 0 < top.provenance["text"] <= 1.0
    assert (top.chunk is not None) != (top.story is not None)
    assert hits.usage.input == 0


async def test_kinds_pillars_and_filters_narrow_the_search() -> None:
    memory = await _seeded()
    stories = await memory.search("acme", "pump seal", kinds=("story",))
    assert all(h.kind == "story" for h in stories.hits) and stories.hits
    text_only = await memory.search("acme", "pump seal", pillars={"text"})
    assert set(text_only.pillars) == {"text"} and all(
        set(h.provenance) == {"text"} for h in text_only.hits
    )
    faceted = await memory.search("acme", "wiring", filters={"metadata.team": "ops"})
    assert [h.kind for h in faceted.hits] == ["story"]
    with pytest.raises(InputError):
        await memory.search("acme", "x", filters={"corpus_id": "acme"})
    with pytest.raises(ConfigurationError):
        await memory.search("acme", "x", pillars={"graph"})
    assert (await memory.search("acme", "   ")).hits == []


async def test_the_text_pillar_finds_a_word_in_another_form_and_nothing_for_stop_words() -> None:
    memory = await _seeded()
    sealing = await memory.search("acme", "sealing", pillars={"text"})
    assert sealing.hits and all("seal" in h.text.lower() for h in sealing.hits)
    assert (await memory.search("acme", "what is the", pillars={"text"})).hits == []


async def test_recall_applies_time_and_subject() -> None:
    memory = await _seeded()
    hits = await memory.recall("acme", "seal", subject="asset:P-3")
    assert [h.kind for h in hits.hits] == ["story"] and hits.hits[0].story is not None
    before = await memory.recall("acme", "seal", as_of=datetime(2026, 1, 1, tzinfo=UTC))
    assert before.hits == []
    after = await memory.recall(
        "acme", "seal", since=datetime(2026, 3, 1, tzinfo=UTC), pillars={"text"}
    )
    assert len(after.hits) == 1


async def test_graph_leg_finds_by_name_and_stands_down_under_a_facet() -> None:
    memory = await _seeded(pillars={"text", "vector", "graph"})
    await memory.ingest(
        "acme",
        name="sheet",
        text="Valve V-101 feeds pump P-3 through line L-7.",
        entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump")],
        relations=[RelationIn(from_name="V-101", to_name="P-3", type="CONNECTS_TO")],
    )
    hits = await memory.search("acme", "what does V-101 feed?")
    assert "graph" in hits.pillars and hits.pillars["graph"].returned >= 1
    graph_hits = [h for h in hits.hits if "graph" in h.provenance]
    assert graph_hits and all(h.provenance["graph"] == 1.0 for h in graph_hits)
    faceted = await memory.search("acme", "what does V-101 feed?", filters={"metadata.line": "A"})
    assert "graph" not in faceted.pillars
    hood = await memory.neighbors("acme", "V-101", hops=1)
    assert [e.name for e in hood.entities] == ["V-101", "P-3"] and hood.relations[
        0
    ].type == "CONNECTS_TO"
    assert hood.passages and hood.passages[0].kind == "chunk"
    with pytest.raises(ConfigurationError):
        await (await _seeded()).neighbors("acme", "V-101")


async def test_a_failing_pillar_is_reported_and_the_rest_answer() -> None:
    memory = await _seeded()

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise StorageError("vector index unavailable")

    memory.storage.search_vector = broken  # type: ignore[method-assign]
    hits = await memory.search("acme", "pump seal")
    assert hits.hits and hits.pillars["vector"].error == "StorageError: vector index unavailable"
    assert all("vector" not in h.provenance for h in hits.hits)


async def test_judge_needs_a_decision_model() -> None:
    memory = await _seeded()
    with pytest.raises(ConfigurationError):
        await memory.search("acme", "pump seal", judge=True)


async def test_reads_refuse_arguments_that_cannot_mean_anything() -> None:
    memory = await _seeded()
    with pytest.raises(
        InputError,
        match=r"^memory: unknown kinds: \['document'\]; choose from \['chunk', 'story'\]$",
    ):
        await memory.search("acme", "seal", kinds=("chunk", "document"))  # type: ignore[arg-type]
    for top_k in (0, -1):
        with pytest.raises(InputError, match=r"^memory: top_k must be at least 1$"):
            await memory.search("acme", "seal", top_k=top_k)
        with pytest.raises(InputError, match=r"^memory: top_k must be at least 1$"):
            await memory.recall("acme", "seal", top_k=top_k)
    for limit in (0, -5):
        with pytest.raises(InputError, match=r"^memory: limit must be at least 1$"):
            await memory.list_sources("acme", limit=limit)
        with pytest.raises(InputError, match=r"^memory: limit must be at least 1$"):
            await memory.list_stories("acme", limit=limit)


class _Short(HashEmbedding):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [v[:7] for v in await super().embed(texts)]


class _Fewer(HashEmbedding):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return (await super().embed(texts))[:-1]


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (_Short(), "a vector of 7 dimensions; it declares 8"),
        (_Fewer(), "0 vectors for 1 texts"),
    ],
)
async def test_an_embedding_model_that_breaks_its_contract_fails_its_pillar(
    broken: HashEmbedding, message: str
) -> None:
    memory = await _seeded()
    hits = await Memory(memory.storage, embedding=broken).search("acme", "pump seal")
    assert hits.hits and all("vector" not in h.provenance for h in hits.hits)
    assert (
        hits.pillars["vector"].error
        == f"ModelError: memory: the embedding model returned {message}"
    )
