from typing import Any

import pytest

from rfnry_memory.errors import InputError, NotFoundError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding


async def _memory(embedding: HashEmbedding | None = None) -> Memory:
    memory = Memory(InMemoryStorage(), embedding=embedding or HashEmbedding())
    await memory.initialize()
    for corpus, line in (("hr", "A"), ("ops", "B"), ("finance", "A")):
        await memory.ingest(
            corpus,
            name=f"{corpus} manual",
            text=f"The {corpus} pump seal is replaced yearly.",
            metadata={"line": line},
        )
        await memory.remember(
            corpus, f"The {corpus} pump seal was replaced in March.", metadata={"line": line}
        )
    return memory


async def test_a_search_reads_every_corpus_it_names_and_no_other() -> None:
    embedding = HashEmbedding()
    memory = await _memory(embedding)
    embedding.calls.clear()
    hits = await memory.search(["hr", "ops"], "pump seal replaced")
    assert len(embedding.calls) == 1
    assert {h.corpus_id for h in hits.hits} == {"hr", "ops"}
    assert all("finance" not in h.text for h in hits.hits)
    assert all(h.corpus_id == (h.chunk or h.story).corpus_id for h in hits.hits)  # type: ignore[union-attr]
    assert len({(h.corpus_id, h.id) for h in hits.hits}) == len(hits.hits) == 4
    recalled = await memory.recall(["ops", "finance"], "pump seal replaced")
    assert sorted(h.corpus_id for h in recalled.hits) == ["finance", "ops"]
    one = await memory.search("hr", "pump seal replaced")
    assert one.hits == (await memory.search(["hr", "hr"], "pump seal replaced")).hits
    assert {h.corpus_id for h in one.hits} == {"hr"}


async def test_filters_hold_in_every_corpus_and_the_corpus_list_is_required() -> None:
    memory = await _memory()
    hits = await memory.search(
        ["hr", "ops", "finance"], "pump seal", filters={"metadata.line": "A"}
    )
    assert {h.corpus_id for h in hits.hits} == {"hr", "finance"}
    for named in ([], [" "], ["hr", ""]):
        with pytest.raises(InputError, match=r"^memory: corpus_id is required$"):
            await memory.search(named, "pump seal")


async def test_a_corpus_that_fails_a_pillar_fails_it_for_the_search() -> None:
    memory = await _memory()
    storage = memory.storage
    original = storage.search_text

    async def failing(corpus_ids: list[str], *args: Any, **kwargs: Any) -> Any:
        if "ops" in corpus_ids:
            raise StorageError("inmemory: the text index is down")
        return await original(corpus_ids, *args, **kwargs)

    storage.search_text = failing  # type: ignore[method-assign]
    hits = await memory.search(["hr", "ops"], "pump seal replaced")
    assert hits.pillars["text"].error == "StorageError: inmemory: the text index is down"
    assert hits.pillars["vector"].error is None and {h.corpus_id for h in hits.hits} == {
        "hr",
        "ops",
    }


async def test_tools_bound_to_several_corpora_read_them_and_narrow_never_widen() -> None:
    memory = await _memory()
    tools = MemoryTools(memory, corpus_id=["hr", "ops"])
    assert set(tools.names) == {"search", "recall", "read_source", "list_sources", "list_stories"}
    schemas = {d["name"]: d["input_schema"] for d in tools.definitions()}
    assert schemas["read_source"]["properties"]["corpus_id"]["enum"] == ["hr", "ops"]
    assert "corpus_id" in schemas["read_source"]["required"]
    assert "corpus_id" not in schemas["search"].get("required", [])
    found = await tools.dispatch("search", {"query": "pump seal replaced"})
    assert isinstance(found, dict) and {h["corpus_id"] for h in found["hits"]} == {"hr", "ops"}
    narrowed = await tools.dispatch("search", {"query": "pump seal", "corpus_id": "ops"})
    assert isinstance(narrowed, dict) and {h["corpus_id"] for h in narrowed["hits"]} == {"ops"}
    for widened in ("finance", ["ops", "finance"], []):
        with pytest.raises(
            InputError, match=r"^tools: corpus_id can name only the corpora you may read: hr, ops$"
        ):
            await tools.dispatch("search", {"query": "pump seal", "corpus_id": widened})
    chunk = next(h for h in found["hits"] if h["kind"] == "chunk" and h["corpus_id"] == "ops")
    read = await tools.dispatch(
        "read_source", {"source_id": chunk["source_id"], "corpus_id": "ops"}
    )
    assert isinstance(read, dict) and "ops pump seal" in read["text"]
    with pytest.raises(InputError, match=r"^tools: corpus_id is required: name one of hr, ops$"):
        await tools.dispatch("read_source", {"source_id": chunk["source_id"]})
    with pytest.raises(NotFoundError):
        await tools.dispatch("read_source", {"source_id": chunk["source_id"], "corpus_id": "hr"})
    listed = await tools.dispatch("list_sources", {"corpus_id": "hr"})
    assert isinstance(listed, dict) and [s["name"] for s in listed["sources"]] == ["hr manual"]


async def test_tools_bound_to_one_corpus_keep_their_shape() -> None:
    memory = await _memory()
    tools = MemoryTools(memory, corpus_id="hr")
    assert all("corpus_id" not in d["input_schema"]["properties"] for d in tools.definitions())
    found = await tools.dispatch("search", {"query": "pump seal"})
    assert isinstance(found, dict) and all("corpus_id" not in h for h in found["hits"])
    narrowed = await tools.dispatch("search", {"query": "pump seal", "corpus_id": "hr"})
    assert isinstance(narrowed, dict) and narrowed["hits"] == found["hits"]
    with pytest.raises(
        InputError, match=r"^tools: corpus_id can name only the corpora you may read: hr$"
    ):
        await tools.dispatch("search", {"query": "pump seal", "corpus_id": "ops"})
    assert MemoryTools(memory, corpus_id=["hr"]).names == tools.names


async def test_bound_filters_hold_in_every_bound_corpus() -> None:
    memory = await _memory()
    tools = MemoryTools(memory, corpus_id=["hr", "ops", "finance"], filters={"metadata.line": "A"})
    found = await tools.dispatch("search", {"query": "pump seal"})
    assert isinstance(found, dict) and {h["corpus_id"] for h in found["hits"]} == {"hr", "finance"}
