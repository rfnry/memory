import asyncio
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn, Window
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.ranking import BM25

from .fakes import HashEmbedding

CORPUS = BM25(statistics="corpus")


async def _recount(
    pool: Any, corpus: str
) -> tuple[dict[str, tuple[int, int]], Counter[tuple[str, str]]]:
    totals: dict[str, tuple[int, int]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    async with pool.acquire() as conn:
        for kind, sql in (
            (
                "chunk",
                "SELECT fts, fts_length FROM memory_chunks WHERE corpus_id = $1 AND retired_at IS NULL",
            ),
            (
                "story",
                "SELECT fts, fts_length FROM memory_stories WHERE corpus_id = $1 AND closed_at IS NULL",
            ),
        ):
            rows = await conn.fetch(
                f"SELECT tsvector_to_array(p.fts) AS words, p.fts_length FROM ({sql}) AS p", corpus
            )
            totals[kind] = (len(rows), sum(r["fts_length"] for r in rows))
            for r in rows:
                counts.update((kind, word) for word in r["words"])
    return totals, counts


async def _held(
    pool: Any, corpus: str
) -> tuple[dict[str, tuple[int, int]], Counter[tuple[str, str]]]:
    async with pool.acquire() as conn:
        totals = {
            r["kind"]: (r["passages"], r["length"])
            for r in await conn.fetch(
                "SELECT kind, passages, length FROM memory_term_totals WHERE corpus_id = $1", corpus
            )
        }
        counts = Counter(
            {
                (r["kind"], r["lexeme"]): r["passages"]
                for r in await conn.fetch(
                    "SELECT kind, lexeme, passages FROM memory_term_counts WHERE corpus_id = $1",
                    corpus,
                )
            }
        )
    return totals, counts


async def _exact(pool: Any, corpus: str) -> None:
    assert await _held(pool, corpus) == await _recount(pool, corpus)


async def test_the_counts_follow_every_write_as_a_recount_would(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    assert await memory.set_statistics("acme", True) == 0
    await _exact(pool, "acme")
    pump = await memory.ingest("acme", name="the", text="Replace the pump seal A-1001.")
    await _exact(pool, "acme")
    parts = [SourceIn(name="the", text=f"Valve {n} seal.", metadata={"set": "v"}) for n in range(3)]
    await memory.sync("acme", filters={"metadata.set": "v"}, sources=parts)
    await memory.sync("acme", filters={"metadata.set": "v"}, sources=parts[:2])
    await _exact(pool, "acme")
    replaced = await memory.ingest(
        "acme", name="the", text="Replace the pump seal yearly.", replaces=pump.source.source_id
    )
    await _exact(pool, "acme")
    await memory.delete_source("acme", replaced.source.source_id)
    await _exact(pool, "acme")
    told = await memory.remember("acme", "Ana owns the seal plan.")
    later = await memory.supersede("acme", told.story.story_id, "Ben owns the seal plan.")
    await _exact(pool, "acme")
    other = await memory.remember("acme", "The pump was serviced.")
    await memory.end_story("acme", other.story.story_id)
    retracted = await memory.remember("acme", "The valve was replaced.")
    await memory.end_story("acme", retracted.story.story_id, valid_to=retracted.story.valid_from)
    await _exact(pool, "acme")
    await memory.delete_story("acme", later.story.story_id)
    await _exact(pool, "acme")
    assert await memory.set_language("acme", "simple") > 0
    await _exact(pool, "acme")
    await memory.reembed("acme")
    await _exact(pool, "acme")
    totals, _ = await _held(pool, "acme")
    assert totals["chunk"][0] == 2 and totals["story"][0] == 0


async def test_retired_versions_leave_the_counts_and_forgetting_them_changes_nothing(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", True)
    await memory.set_statistics("acme", True)
    first = await memory.ingest(
        "acme", name="the", text="Spec of the pump, first.", metadata={"doc": "s"}
    )
    await memory.sync(
        "acme",
        filters={"metadata.doc": "s"},
        sources=[SourceIn(name="the", text="Spec of the pump, second.", metadata={"doc": "s"})],
    )
    await _exact(pool, "acme")
    third = await memory.ingest("acme", name="the", text="Spec of the valve.")
    await memory.retire_source("acme", third.source.source_id)
    await _exact(pool, "acme")
    before = await _held(pool, "acme")
    await memory.forget_retired("acme", before=datetime.now(UTC) + timedelta(days=1))
    assert await _held(pool, "acme") == before
    assert first.source.source_id


async def test_a_reopened_story_counts_again_and_a_revised_closing_changes_nothing(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_statistics("acme", True)
    told = await memory.remember("acme", "Ana owns the seal plan.")
    later = await memory.supersede(
        "acme",
        told.story.story_id,
        "Ben owns the seal plan.",
        valid_from=datetime.now(UTC) + timedelta(days=3),
    )
    await _exact(pool, "acme")
    await memory.end_story("acme", later.story.story_id, valid_to=later.story.valid_from)
    await _exact(pool, "acme")
    totals, counts = await _held(pool, "acme")
    assert totals["story"][0] == 1 and counts["story", "ana"] == 1 and counts["story", "ben"] == 0
    for days in (10, 20):
        await memory.end_story(
            "acme", told.story.story_id, valid_to=datetime.now(UTC) + timedelta(days=days)
        )
        await _exact(pool, "acme")
    assert (await _held(pool, "acme"))[0]["story"][0] == 0


async def test_a_version_reopened_by_a_withdrawal_counts_again(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", True)
    await memory.set_statistics("acme", True)
    first = await memory.ingest("acme", name="the", text="Spec of the pump, first.")
    second = await memory.ingest(
        "acme",
        name="the",
        text="Spec of the valve, second.",
        valid_from=datetime.now(UTC) + timedelta(days=3),
        replaces=first.source.source_id,
    )
    await _exact(pool, "acme")
    await memory.retire_source("acme", second.source.source_id)
    await _exact(pool, "acme")
    _, counts = await _held(pool, "acme")
    assert (counts["chunk", "pump"], counts["chunk", "valv"]) == (1, 0)
    for days in (10, 20):
        await memory.retire_source(
            "acme", first.source.source_id, valid_to=datetime.now(UTC) + timedelta(days=days)
        )
        await _exact(pool, "acme")


async def test_bm25_with_corpus_statistics_scores_by_the_whole_corpus(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    for name, text in (
        ("d1", "pump seal"),
        ("d2", "pump pump"),
        ("d3", "valve"),
        ("d4", "seal seal valve"),
    ):
        await memory.ingest("acme", name="the", text=text, metadata={"name": name})
    await memory.set_statistics("acme", True)
    legs = await storage.search_text(
        ["acme"],
        "seal",
        kinds=("chunk",),
        top_k=10,
        filters=None,
        window=Window(),
        subject=None,
        ranking=CORPUS,
        match="any",
    )
    by_text = {}
    for _, (_, chunk_id, score) in legs:
        [chunk] = await storage.get_passages("acme", [("chunk", chunk_id)])
        by_text[chunk.text] = score
    idf = math.log(1 + (4 - 2 + 0.5) / (2 + 0.5))

    def bm25(tf: int, length: int) -> float:
        return idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * length / 2))

    assert by_text == pytest.approx({"seal seal valve": bm25(2, 3), "pump seal": bm25(1, 2)})


async def test_under_all_a_rare_word_outweighs_a_common_one(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.ingest("acme", name="the", text="pump pump seal")
    await memory.ingest("acme", name="the", text="pump seal seal")
    for n in range(8):
        await memory.ingest("acme", name="the", text=f"pump {n}")
    await memory.set_statistics("acme", True)

    async def ranked(ranking: BM25) -> list[tuple[str, float]]:
        legs = await storage.search_text(
            ["acme"],
            "pump seal",
            kinds=("chunk",),
            top_k=10,
            filters=None,
            window=Window(),
            subject=None,
            ranking=ranking,
            match="all",
        )
        found = []
        for _, (_, chunk_id, score) in legs:
            [chunk] = await storage.get_passages("acme", [("chunk", chunk_id)])
            found.append((chunk.text, score))
        return found

    read = await ranked(BM25())
    assert read[0][1] == pytest.approx(read[1][1])
    corpus = await ranked(CORPUS)
    assert [text for text, _ in corpus] == ["pump seal seal", "pump pump seal"]
    assert corpus[0][1] > corpus[1][1]


async def test_with_corpus_statistics_a_filter_leaves_the_scores_as_they_were(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.ingest("acme", name="the", text="seal of line a", metadata={"line": "a"})
    for n in range(4):
        await memory.ingest("acme", name="the", text=f"seal {n} of line b", metadata={"line": "b"})
    await memory.set_statistics("acme", True)

    async def score(ranking: BM25, filters: dict[str, Any] | None) -> float:
        legs = await storage.search_text(
            ["acme"],
            "seal",
            kinds=("chunk",),
            top_k=10,
            filters=filters,
            window=Window(),
            subject=None,
            ranking=ranking,
            match="any",
        )
        for _, (_, chunk_id, value) in legs:
            [chunk] = await storage.get_passages("acme", [("chunk", chunk_id)])
            if chunk.text == "seal of line a":
                return value
        raise AssertionError("the line a passage was not found")

    only_a = {"metadata.line": "a"}
    assert await score(CORPUS, only_a) == pytest.approx(await score(CORPUS, None))
    assert await score(BM25(), only_a) != pytest.approx(await score(BM25(), None))


async def test_a_corpus_that_keeps_no_statistics_is_refused(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.ingest("acme", name="the", text="Replace the pump seal.")
    refused = (
        r"^postgres: corpus 'acme' keeps no statistics — call set_statistics\('acme', True\) "
        r"to rank it with statistics='corpus'$"
    )
    with pytest.raises(ConfigurationError, match=refused):
        await memory.search("acme", "seal", text=CORPUS)
    counted = Memory(storage, embedding=HashEmbedding(), text=CORPUS)
    with pytest.raises(ConfigurationError, match=refused):
        await counted.search("acme", "seal")
    await memory.set_statistics("acme", True)
    assert (await counted.search("acme", "seal")).hits


async def test_set_statistics_counts_again_and_forgets(storage: PostgresStorage, pool: Any) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.ingest("acme", name="the", text="Replace the pump seal.")
    await memory.remember("acme", "Ana owns the seal plan.")
    assert not (await memory.fingerprint("acme")).statistics
    assert await memory.set_statistics("acme", True) == 2
    assert await memory.set_statistics("acme", True) == 2
    assert (await memory.fingerprint("acme")).statistics
    await _exact(pool, "acme")
    assert (await memory.verify_absent("acme")).statistics > 0
    assert await memory.set_statistics("acme", False) == 0
    assert await _held(pool, "acme") == ({}, Counter())
    with pytest.raises(InputError, match=r"^memory: keep is True or False$"):
        await memory.set_statistics("acme", 1)  # type: ignore[arg-type]
    await memory.set_statistics("acme", True)
    erased = await memory.delete_corpus("acme")
    assert erased.statistics > 0 and (await memory.verify_absent("acme")).clean


async def test_stress_concurrent_writes_into_a_counted_corpus_keep_the_counts_exact(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_statistics("acme", True)
    told = [await memory.remember("acme", f"Plan {n} for the seal.") for n in range(4)]
    writes = [
        *(
            memory.ingest("acme", name="the", text=f"Pump {n} seal and valve {n}.")
            for n in range(12)
        ),
        *(memory.remember("acme", f"Note {n} on the pump seal.") for n in range(6)),
        *(
            memory.supersede("acme", t.story.story_id, f"Plan {n} for the valve.")
            for n, t in enumerate(told)
        ),
    ]
    await asyncio.wait_for(asyncio.gather(*writes), timeout=60)
    await _exact(pool, "acme")
    totals, _ = await _held(pool, "acme")
    assert totals == {"chunk": (12, totals["chunk"][1]), "story": (10, totals["story"][1])}


async def test_stress_ending_and_deleting_one_story_at_once_never_deadlocks(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_statistics("acme", True)
    for n in range(10):
        told = await memory.remember("acme", f"Plan {n} for the seal.")
        story_id = told.story.story_id
        later = (
            memory.supersede("acme", story_id, f"Plan {n} for the valve.")
            if n % 2
            else memory.end_story("acme", story_id)
        )
        outcomes = await asyncio.wait_for(
            asyncio.gather(later, memory.delete_story("acme", story_id), return_exceptions=True),
            timeout=30,
        )
        failures = [o for o in outcomes if isinstance(o, BaseException)]
        assert all(type(f).__name__ == "NotFoundError" for f in failures), failures
    await _exact(pool, "acme")
