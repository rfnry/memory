import pytest

from rfnry_memory import DocumentEmbeddingModel
from rfnry_memory.errors import ModelError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Section
from rfnry_memory.storage.inmemory import InMemoryStorage

from .fakes import DocumentHashEmbedding, HashEmbedding


async def test_a_document_aware_model_embeds_each_source_in_one_call_without_reuse() -> None:
    embedding = DocumentHashEmbedding()
    assert isinstance(embedding, DocumentEmbeddingModel)
    assert not isinstance(HashEmbedding(), DocumentEmbeddingModel)
    memory = Memory(InMemoryStorage(), embedding=embedding)
    await memory.initialize()
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


async def test_a_document_aware_model_that_returns_the_wrong_count_raises() -> None:
    class Short(DocumentHashEmbedding):
        async def embed_in_document(self, texts: list[str]) -> list[list[float]]:
            return (await super().embed_in_document(texts))[:-1]

    memory = Memory(InMemoryStorage(), embedding=Short())
    await memory.initialize()
    with pytest.raises(
        ModelError, match=r"^memory: the embedding model returned 1 vectors for 2 texts$"
    ):
        await memory.ingest("acme", name="a", sections=[Section(text="One."), Section(text="Two.")])
    assert (await memory.list_sources("acme"))[0] == []


async def test_reembed_moves_a_corpus_to_a_document_aware_model_source_by_source() -> None:
    storage = InMemoryStorage()
    plain = Memory(storage, embedding=HashEmbedding())
    await plain.initialize()
    await plain.ingest("acme", name="a", sections=[Section(text="One."), Section(text="Two.")])
    await plain.remember("acme", "The pump runs.")
    aware = DocumentHashEmbedding()
    memory = Memory(storage, embedding=aware)
    await memory.initialize()
    assert (await memory.fingerprint("acme")).stale == 3
    assert await memory.reembed("acme") == 3
    assert aware.documents == [["One.", "Two."]]
    assert aware.calls == [["The pump runs."]]
    assert (await memory.fingerprint("acme")).stale == 0
