from typing import Any

import pytest

from rfnry_memory.errors import InputError, NotFoundError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.storage.postgres import PostgresStorage
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding


async def _memory(storage: PostgresStorage, embedding: HashEmbedding | None = None) -> Memory:
    memory = Memory(storage, embedding=embedding or HashEmbedding())
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


async def test_a_search_reads_every_corpus_it_names_and_no_other(storage: PostgresStorage) -> None:
    embedding = HashEmbedding()
    memory = await _memory(storage, embedding)
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


async def test_filters_hold_in_every_corpus_and_the_corpus_list_is_required(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    hits = await memory.search(
        ["hr", "ops", "finance"], "pump seal", filters={"metadata.line": "A"}
    )
    assert {h.corpus_id for h in hits.hits} == {"hr", "finance"}
    for named in ([], [" "], ["hr", ""]):
        with pytest.raises(InputError, match=r"^memory: corpus_id is required$"):
            await memory.search(named, "pump seal")


async def test_a_corpus_that_fails_a_pillar_fails_it_for_the_search(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    original = storage.search_text

    async def failing(corpus_ids: list[str], *args: Any, **kwargs: Any) -> Any:
        if "ops" in corpus_ids:
            raise StorageError("postgres: the text index is down")
        return await original(corpus_ids, *args, **kwargs)

    storage.search_text = failing  # type: ignore[method-assign]
    hits = await memory.search(["hr", "ops"], "pump seal replaced")
    assert hits.pillars["text"].error == "StorageError: postgres: the text index is down"
    assert hits.pillars["vector"].error is None and {h.corpus_id for h in hits.hits} == {
        "hr",
        "ops",
    }


async def test_tools_bound_to_several_corpora_read_them_and_narrow_never_widen(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    tools = MemoryTools(memory, corpus_id=["hr", "ops"])
    assert set(tools.names) == {
        "search",
        "recall",
        "expand",
        "read_source",
        "list_sources",
        "list_stories",
        "describe",
    }
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


async def test_tools_bound_to_one_corpus_keep_their_shape(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
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


async def test_bound_filters_hold_in_every_bound_corpus(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    tools = MemoryTools(memory, corpus_id=["hr", "ops", "finance"], filters={"metadata.line": "A"})
    found = await tools.dispatch("search", {"query": "pump seal"})
    assert isinstance(found, dict) and {h["corpus_id"] for h in found["hits"]} == {"hr", "finance"}


MANUAL = [
    "gasket and pump assembly",
    "gasket material",
    "gasket storage",
    "gasket supplier",
    "gasket inspection",
    "gasket sizes",
    "pump curve",
    "pump wiring",
]
NOTES = ["pump noise", "pump restart", "pump alarm", "pump service", "gasket ordered"]


async def _filed(memory: Memory, corpus: str, prefix: str, texts: list[str]) -> None:
    for i, text in enumerate(texts):
        await memory.ingest(corpus, name=f"{prefix}{i}", text=text)


async def test_keyword_search_counts_corpora_of_one_language_as_one(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.initialize()
    await _filed(memory, "whole", "m", MANUAL)
    await _filed(memory, "whole", "n", NOTES)
    await _filed(memory, "manual", "m", MANUAL)
    await _filed(memory, "notes", "n", NOTES)

    async def ranked(corpora: str | list[str]) -> list[tuple[float, str]]:
        hits = await memory.search(corpora, "gasket pump", pillars={"text"}, top_k=20)
        return sorted((round(h.provenance["text"], 9), h.text) for h in hits.hits)

    assert await ranked(["notes", "manual"]) == await ranked("whole")
    first = await memory.search(["notes", "manual"], "gasket pump", pillars={"text"})
    assert (first.hits[0].corpus_id, first.hits[0].text) == ("manual", "gasket and pump assembly")


async def test_corpora_that_share_a_configuration_pool_however_it_was_named(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    for corpus, name in (("whole", "french"), ("manual", "French"), ("notes", "pg_catalog.french")):
        await memory.set_language(corpus, name)
    await _filed(memory, "whole", "m", MANUAL)
    await _filed(memory, "whole", "n", NOTES)
    await _filed(memory, "manual", "m", MANUAL)
    await _filed(memory, "notes", "n", NOTES)
    searched: list[list[str]] = []
    original = storage.search_text

    async def recorded(corpus_ids: list[str], *args: Any, **kwargs: Any) -> Any:
        searched.append(list(corpus_ids))
        return await original(corpus_ids, *args, **kwargs)

    storage.search_text = recorded  # type: ignore[method-assign]

    async def ranked(corpora: str | list[str]) -> list[tuple[float, str]]:
        hits = await memory.search(corpora, "gasket pump", pillars={"text"}, top_k=20)
        return sorted((round(h.provenance["text"], 9), h.text) for h in hits.hits)

    assert await ranked(["notes", "manual"]) == await ranked("whole")
    assert searched == [["notes", "manual"], ["whole"]]


async def test_keyword_lists_of_different_languages_take_turns_by_rank(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.initialize()
    await memory.set_language("es", "spanish")
    await _filed(
        memory, "en", "e", ["Part 4471 ships in boxes of ten.", "Part 4471 is discontinued."]
    )
    await _filed(
        memory, "es", "s", ["La pieza 4471 llega en cajas.", "La pieza 4471 está descontinuada."]
    )
    hits = await memory.search(["en", "es"], "4471", pillars={"text"})
    assert [h.corpus_id for h in hits.hits] == ["en", "es", "en", "es"]
    assert [h.provenance["text"] for h in hits.hits[:2]] == [1.0, 1.0]
    flipped = await memory.search(["es", "en"], "4471", pillars={"text"})
    assert [h.corpus_id for h in flipped.hits] == ["es", "en", "es", "en"]
    english = await memory.search(["en", "es"], "boxes", pillars={"text"})
    assert [h.corpus_id for h in english.hits] == ["en"]
    assert english.hits[0].provenance["text"] == 1.0 and english.pillars["text"].error is None


async def test_corpora_of_one_language_pool_while_another_language_takes_turns(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.initialize()
    await memory.set_language("es", "spanish")
    await _filed(memory, "en1", "a", ["Part 4471 ships."])
    await _filed(memory, "en2", "b", ["Part 4471 ships in boxes of ten with a packing note."])
    await _filed(
        memory, "es", "s", ["La pieza 4471 llega.", "La pieza 4471 llega en cajas grandes."]
    )
    hits = await memory.search(["en2", "es", "en1"], "4471", pillars={"text"})
    assert [h.corpus_id for h in hits.hits] == ["en1", "es", "en2", "es"]
    assert hits.hits[0].provenance["text"] == 1.0
    assert hits.hits[2].provenance["text"] < 1.0


async def test_only_a_search_of_several_corpora_looks_up_their_languages(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    looked: list[str] = []
    original = storage.get_language

    async def counting(corpus_id: str) -> Any:
        looked.append(corpus_id)
        return await original(corpus_id)

    storage.get_language = counting  # type: ignore[method-assign]
    await memory.search("hr", "pump seal")
    assert looked == []
    await memory.search(["hr", "ops"], "pump seal", pillars={"vector"})
    assert looked == []
    await memory.search(["hr", "ops"], "pump seal")
    assert sorted(looked) == ["hr", "ops"]


async def test_a_failing_language_lookup_fails_only_the_keyword_pillar(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)

    async def failing(corpus_id: str) -> Any:
        raise StorageError("postgres: the languages are down")

    memory.storage.get_language = failing  # type: ignore[method-assign]
    hits = await memory.search(["hr", "ops"], "pump seal replaced")
    assert hits.pillars["text"].error == "StorageError: postgres: the languages are down"
    assert hits.pillars["vector"].error is None
    assert {h.corpus_id for h in hits.hits} == {"hr", "ops"}


async def test_a_corpus_never_written_beside_others_breaks_nothing(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    hits = await memory.search(["hr", "ghost"], "pump seal replaced")
    assert hits.pillars["text"].error is None and hits.pillars["text"].returned > 0
    assert {h.corpus_id for h in hits.hits} == {"hr"}
