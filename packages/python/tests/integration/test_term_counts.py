from typing import Any

from unit.fakes import HashEmbedding

from rfnry_memory.memory import Memory
from rfnry_memory.models import Window
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.ranking import BM25


async def _score(storage: PostgresStorage, corpus: str, query: str) -> list[float]:
    legs = await storage.search_text(
        [corpus],
        query,
        kinds=("chunk",),
        top_k=10,
        filters=None,
        window=Window(),
        subject=None,
        ranking=BM25(statistics="corpus"),
        match="any",
    )
    return sorted(score for _, (_, _, score) in legs)


async def test_the_counts_of_a_never_reach_b_and_erase_with_a(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    for corpus in ("tenant-a", "tenant-b"):
        await memory.set_statistics(corpus, True)
        await memory.ingest(corpus, name="the", text="Replace the pump seal.")
    before = await _score(storage, "tenant-b", "seal")
    for n in range(6):
        await memory.ingest("tenant-a", name="the", text=f"Seal {n} of the zircon pump.")
    assert await _score(storage, "tenant-b", "seal") == before
    async with pool.acquire() as conn:
        leaked = await conn.fetchval(
            "SELECT count(*) FROM memory_term_counts WHERE corpus_id = 'tenant-b' AND lexeme = 'zircon'"
        )
        mixed = await conn.fetchval(
            "SELECT passages FROM memory_term_totals WHERE corpus_id = 'tenant-b' AND kind = 'chunk'"
        )
    assert (leaked, mixed) == (0, 1)
    held = (await memory.verify_absent("tenant-a")).statistics
    erased = await memory.delete_corpus("tenant-a")
    assert erased.statistics == held > 0 and (await memory.verify_absent("tenant-a")).clean
    assert await _score(storage, "tenant-b", "seal") == before
