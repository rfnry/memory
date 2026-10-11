import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Counts
from rfnry_memory.storage.postgres import PostgresStorage


async def _memory(storage: PostgresStorage) -> Memory:
    memory = Memory(storage)
    await memory.initialize()
    await memory.ingest(
        "acme", name="manual", text="La información de las válvulas está en el manual."
    )
    await memory.remember("acme", "Las bombas funcionan mejor después del cambio.")
    return memory


async def _found(memory: Memory, query: str) -> list[str]:
    hits = await memory.search("acme", query, pillars={"text"})
    return [h.kind for h in hits.hits]


async def test_a_corpus_reads_its_own_language_and_folds_accents(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    assert await _found(memory, "informacion valvula") == []
    assert await memory.set_language("acme", "spanish") == 2
    assert await _found(memory, "informacion valvula") == ["chunk"]
    assert await _found(memory, "bomba funcionando") == ["story"]
    assert await memory.set_language("acme", "spanish") == 0
    assert await memory.set_language("acme", "simple") == 2
    assert await _found(memory, "valvula bomba") == []
    assert await _found(memory, "VALVULAS") == ["chunk"]


async def test_a_language_belongs_to_one_corpus_and_goes_with_it(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    await memory.ingest("other", name="manual", text="La información de las válvulas.")
    await memory.set_language("acme", "portuguese")
    assert (await memory.fingerprint("acme")).language == "portuguese"
    assert (await memory.fingerprint("other")).language == "english"
    assert (await memory.verify_absent("acme")).corpora == 1
    assert (await memory.verify_absent("other")).corpora == 0
    deleted = await memory.delete_corpus("acme")
    assert deleted.corpora == 1 and (await memory.verify_absent("acme")) == Counts()
    assert (await memory.fingerprint("acme")).language == "english"


async def test_an_unknown_language_is_refused(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    with pytest.raises(
        InputError,
        match=r"^memory: unknown language 'klingon'; the languages are english, spanish, portuguese, simple$",
    ):
        await memory.set_language("acme", "klingon")  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: corpus_id is required$"):
        await memory.set_language(" ", "spanish")
    assert (await memory.fingerprint("acme")).language == "english"
