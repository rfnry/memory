from datetime import UTC, datetime
from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError, InputError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import EntityIn, RelationIn
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding


async def _seeded(storage: PostgresStorage, **kwargs: Any) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding(), chunk_chars=200, overlap_chars=20, **kwargs)
    await memory.check()
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


async def test_search_fuses_text_and_vector_with_provenance(storage: PostgresStorage) -> None:
    memory = await _seeded(storage)
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


async def test_kinds_pillars_and_filters_narrow_the_search(storage: PostgresStorage) -> None:
    memory = await _seeded(storage)
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


async def test_a_blank_query_is_refused(storage: PostgresStorage) -> None:
    memory = await _seeded(storage)
    refused = r"^memory: a search needs a query$"
    for query in ("", "   ", "\t\n"):
        with pytest.raises(InputError, match=refused):
            await memory.search("acme", query)
        with pytest.raises(InputError, match=refused):
            await memory.recall("acme", query)
    with pytest.raises(InputError, match=r"^memory: top_k must be at least 1$"):
        await memory.search("acme", " ", top_k=0)
    tools = MemoryTools(memory, corpus_id="acme")
    for name in ("search", "recall"):
        with pytest.raises(InputError, match=refused):
            await tools.dispatch(name, {"query": " "})


async def test_the_text_pillar_finds_a_word_in_another_form_and_nothing_for_stop_words(
    storage: PostgresStorage,
) -> None:
    memory = await _seeded(storage)
    sealing = await memory.search("acme", "sealing", pillars={"text"})
    assert sealing.hits and all("seal" in h.text.lower() for h in sealing.hits)
    assert (await memory.search("acme", "what is the", pillars={"text"})).hits == []


async def test_recall_applies_time_and_subject(storage: PostgresStorage) -> None:
    memory = await _seeded(storage)
    hits = await memory.recall("acme", "seal", subject="asset:P-3")
    assert [h.kind for h in hits.hits] == ["story"] and hits.hits[0].story is not None
    before = await memory.recall("acme", "seal", as_of=datetime(2026, 1, 1, tzinfo=UTC))
    assert before.hits == []
    after = await memory.recall(
        "acme", "seal", since=datetime(2026, 3, 1, tzinfo=UTC), pillars={"text"}
    )
    assert len(after.hits) == 1


async def test_graph_leg_finds_by_name_and_stands_down_under_a_facet(
    storage: PostgresStorage,
) -> None:
    memory = await _seeded(storage, pillars={"text", "vector", "graph"})
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
        await (await _seeded(storage)).neighbors("acme", "V-101")


async def test_a_failing_pillar_is_reported_and_the_rest_answer(storage: PostgresStorage) -> None:
    memory = await _seeded(storage)

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise StorageError("vector index unavailable")

    memory.storage.search_vector = broken  # type: ignore[method-assign]
    hits = await memory.search("acme", "pump seal")
    assert hits.hits and hits.pillars["vector"].error == "StorageError: vector index unavailable"
    assert all("vector" not in h.provenance for h in hits.hits)


class _VendorDown(HashEmbedding):
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        if purpose == "query":
            raise RuntimeError("vendor down")
        return await super().embed(texts, purpose=purpose)


class _Refusing(HashEmbedding):
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        if purpose == "query":
            raise InputError("fake: refused")
        return await super().embed(texts, purpose=purpose)


async def test_a_vendor_error_drops_the_vector_pillar_and_the_text_pillar_answers(
    storage: PostgresStorage,
) -> None:
    memory = await _seeded(storage)
    hits = await Memory(memory.storage, embedding=_VendorDown()).search("acme", "pump seal")
    assert hits.hits and hits.pillars["vector"].error == "RuntimeError: vendor down"
    assert hits.pillars["text"].error is None


async def test_a_raw_error_inside_the_text_pillar_raises(storage: PostgresStorage) -> None:
    memory = await _seeded(storage)

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("library bug")

    memory.storage.search_text = broken  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="library bug"):
        await memory.search("acme", "pump seal", pillars={"text"})


async def test_a_raw_error_inside_the_graph_pillar_raises(storage: PostgresStorage) -> None:
    memory = await _seeded(storage, pillars={"text", "vector", "graph"})

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("library bug")

    memory.storage.entities_named_in = broken  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="library bug"):
        await memory.search("acme", "pump seal", pillars={"graph"})


async def test_a_library_refusal_inside_a_pillar_still_raises(storage: PostgresStorage) -> None:
    memory = await _seeded(storage)
    with pytest.raises(InputError, match="fake: refused"):
        await Memory(memory.storage, embedding=_Refusing()).search("acme", "refund")


async def test_judge_needs_a_decision_model(storage: PostgresStorage) -> None:
    memory = await _seeded(storage)
    with pytest.raises(ConfigurationError):
        await memory.search("acme", "pump seal", judge=True)


async def test_reads_refuse_arguments_that_cannot_mean_anything(storage: PostgresStorage) -> None:
    memory = await _seeded(storage)
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
    when: dict[str, Any] = {"history": True, "as_of": datetime(2026, 1, 20, tzinfo=UTC)}
    refused = r"^memory: history=True reads every moment; it takes no as_of$"
    with pytest.raises(InputError, match=refused):
        await memory.search("acme", "seal", **when)
    with pytest.raises(InputError, match=refused):
        await memory.recall("acme", "seal", **when)
    with pytest.raises(InputError, match=refused):
        await memory.list_stories("acme", **when)
    graph = Memory(storage, pillars={"text", "graph"})
    await graph.check()
    with pytest.raises(InputError, match=refused):
        await graph.neighbors("acme", "P-3", **when)
    with pytest.raises(ConfigurationError, match=r"^memory: neighbors needs the graph pillar$"):
        await memory.neighbors("acme", "P-3", **when)


class _Short(HashEmbedding):
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [v[:7] for v in await super().embed(texts, purpose=purpose)]


class _Fewer(HashEmbedding):
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return (await super().embed(texts, purpose=purpose))[:-1]


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (_Short(), "a vector of 7 dimensions; it declares 8"),
        (_Fewer(), "0 vectors for 1 texts"),
    ],
)
async def test_an_embedding_model_that_breaks_its_contract_fails_its_pillar(
    storage: PostgresStorage, broken: HashEmbedding, message: str
) -> None:
    memory = await _seeded(storage)
    hits = await Memory(memory.storage, embedding=broken).search("acme", "pump seal")
    assert hits.hits and all("vector" not in h.provenance for h in hits.hits)
    assert (
        hits.pillars["vector"].error
        == f"ModelError: memory: the embedding model returned {message}"
    )


async def test_a_query_is_embedded_as_a_query_and_what_is_indexed_as_documents(
    storage: PostgresStorage,
) -> None:
    embedding = HashEmbedding()
    memory = Memory(storage, embedding=embedding)
    await memory.check()
    await memory.ingest("acme", name="manual", text="Replace the seal yearly.")
    story = (await memory.remember("acme", "Ana owns the review.")).story
    await memory.supersede("acme", story.story_id, "Bo owns the review.")
    assert embedding.purposes == ["document", "document", "document"]
    embedding.purposes.clear()
    await memory.search("acme", "seal")
    await memory.recall(["acme"], "review")
    assert embedding.purposes == ["query", "query"]
    renamed = HashEmbedding()
    renamed.model = "hash-8-v2"
    await Memory(memory.storage, embedding=renamed).reembed("acme")
    assert renamed.purposes and set(renamed.purposes) == {"document"}
