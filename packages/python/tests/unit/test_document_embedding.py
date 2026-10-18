from typing import Any

import pytest

from rfnry_memory import DocumentEmbeddingModel
from rfnry_memory.errors import ModelError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Section, SourceIn
from rfnry_memory.postgres import PostgresStorage

from .fakes import DocumentHashEmbedding, HashEmbedding


async def test_a_document_aware_model_embeds_each_source_in_one_call_without_reuse(
    storage: PostgresStorage,
) -> None:
    embedding = DocumentHashEmbedding()
    assert isinstance(embedding, DocumentEmbeddingModel)
    assert not isinstance(HashEmbedding(), DocumentEmbeddingModel)
    memory = Memory(storage, embedding=embedding)
    await memory.check()
    for name in ("a", "b"):
        await memory.ingest(
            "acme",
            name=name,
            sections=[Section(text="Shared paragraph."), Section(text=f"Only in {name}.")],
        )
    assert embedding.documents == [
        ["Shared paragraph.", "Only in a."],
        ["Shared paragraph.", "Only in b."],
    ]
    assert embedding.calls == []


async def test_a_document_aware_model_that_returns_the_wrong_count_raises(
    storage: PostgresStorage,
) -> None:
    class Short(DocumentHashEmbedding):
        async def embed_in_document(self, texts: list[str]) -> list[list[float]]:
            return (await super().embed_in_document(texts))[:-1]

    memory = Memory(storage, embedding=Short())
    await memory.check()
    with pytest.raises(
        ModelError, match=r"^memory: the embedding model returned 1 vectors for 2 texts$"
    ):
        await memory.ingest("acme", name="a", sections=[Section(text="One."), Section(text="Two.")])
    assert (await memory.list_sources("acme"))[0] == []


async def test_reembed_moves_a_corpus_to_a_document_aware_model_source_by_source(
    storage: PostgresStorage,
) -> None:
    plain = Memory(storage, embedding=HashEmbedding())
    await plain.check()
    await plain.ingest("acme", name="a", sections=[Section(text="One."), Section(text="Two.")])
    await plain.remember("acme", "The pump runs.")
    aware = DocumentHashEmbedding()
    memory = Memory(storage, embedding=aware)
    await memory.check()
    assert (await memory.fingerprint("acme")).stale == 3
    assert await memory.reembed("acme") == 3
    assert aware.documents == [["One.", "Two."]]
    assert aware.calls == [["The pump runs."]]
    assert (await memory.fingerprint("acme")).stale == 0


async def test_a_sync_with_a_document_aware_model_embeds_each_written_source_in_one_call(
    storage: PostgresStorage,
) -> None:
    embedding = DocumentHashEmbedding()
    memory = Memory(storage, embedding=embedding)
    await memory.check()
    doc = {"metadata.doc": "manual"}

    def manual(name: str, last: str) -> SourceIn:
        return SourceIn(
            name=name,
            sections=[Section(text="Shared paragraph."), Section(text=last)],
            metadata={"doc": "manual"},
        )

    await memory.sync(
        "acme", filters=doc, sources=[manual("a", "Only in a."), manual("b", "Only in b.")]
    )
    assert embedding.documents == [
        ["Shared paragraph.", "Only in a."],
        ["Shared paragraph.", "Only in b."],
    ]
    embedding.documents.clear()
    again = await memory.sync(
        "acme", filters=doc, sources=[manual("a", "Only in a."), manual("b", "Now in b.")]
    )
    assert len(again.kept) == 1 and embedding.documents == [["Shared paragraph.", "Now in b."]]
    assert embedding.calls == []


async def test_a_reembed_whose_document_model_returns_the_wrong_count_raises_and_writes_nothing(
    storage: PostgresStorage,
) -> None:
    class Short(DocumentHashEmbedding):
        async def embed_in_document(self, texts: list[str]) -> list[list[float]]:
            return (await super().embed_in_document(texts))[:-1]

    plain = Memory(storage, embedding=HashEmbedding())
    await plain.check()
    await plain.ingest("acme", name="a", sections=[Section(text="One."), Section(text="Two.")])
    memory = Memory(storage, embedding=Short())
    with pytest.raises(
        ModelError, match=r"^memory: the embedding model returned 1 vectors for 2 texts$"
    ):
        await memory.reembed("acme")
    assert (await memory.fingerprint("acme")).stale == 2


async def test_reembed_takes_a_source_whole_when_only_some_of_its_chunks_are_stale(
    storage: PostgresStorage, pool: Any
) -> None:
    plain = Memory(storage, embedding=HashEmbedding())
    await plain.check()
    for name in ("a", "b"):
        await plain.ingest(
            "acme", name=name, sections=[Section(text=f"{name} one."), Section(text=f"{name} two.")]
        )
    aware = DocumentHashEmbedding()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_chunks SET embedding_model = $1 WHERE corpus_id = 'acme' AND text = 'a one.'",
            aware.model,
        )
    memory = Memory(storage, embedding=aware)
    assert (await memory.fingerprint("acme")).stale == 3
    assert await memory.reembed("acme") == 4
    assert sorted(aware.documents) == [["a one.", "a two."], ["b one.", "b two."]]
    assert (await memory.fingerprint("acme")).stale == 0
