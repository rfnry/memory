from datetime import UTC, datetime

import pytest
from database import Schemas, provision

from rfnry_memory.errors import InputError
from rfnry_memory.fusion import RRF, RelativeScore
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery import (
    MemoryRefinery,
    Relevance,
    RetrievalReview,
    refinery as refinery_module,
)
from rfnry_memory.refinery.metrics import ndcg_at, percentile, recall_at, reciprocal_rank, unique

from .fakes import HashEmbedding

JAN, FEB, MAR = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3))
MANUAL = "Replace the pump seal yearly, after draining the line."
NOTES = "Pump seal notes from March: the seal on line B leaked."
PANEL = "Panel wiring guide for the control cabinet."


def test_the_measures_score_known_rankings_by_hand() -> None:
    a, b, c, d = (("x", "source", name) for name in "abcd")
    grades = {a: 3, c: 1, d: 0}
    ranked = [b, c, a, d]
    assert recall_at(ranked, grades, 2) == 0.5
    assert recall_at(ranked, grades, 3) == 1.0
    assert reciprocal_rank(ranked, grades) == 0.5
    assert ndcg_at(ranked, grades, 3) == pytest.approx(4.130929753571457 / 7.630929753571457)
    assert recall_at(ranked, {d: 0}, 3) == 0.0 and ndcg_at(ranked, {}, 3) == 0.0
    assert (percentile([5, 1, 3, 2, 4], 0.5), percentile([5, 1, 3, 2, 4], 0.95)) == (3, 5)
    assert unique([a, b, a, c]) == [a, b, c]


async def _filed(memory: Memory) -> dict[str, str]:
    ids = {}
    for name, text in (("manual", MANUAL), ("notes", NOTES), ("panel", PANEL)):
        ids[name] = (await memory.ingest("acme", name=name, text=text)).source.source_id
    return ids


async def test_measure_scores_the_recordings_as_served_on_their_labels(
    storage: PostgresStorage,
) -> None:
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery)
    await memory.check()
    ids = await _filed(memory)
    labeled = (await memory.search("acme", "pump seal", kinds=("chunk",))).retrieval_id
    await memory.search("acme", "wiring", kinds=("chunk",))
    judged = (await memory.search("acme", "seal", kinds=("chunk",))).retrieval_id
    assert labeled is not None and judged is not None
    grades = [
        Relevance(corpus_id="acme", kind="source", id=ids["manual"], grade=3),
        Relevance(corpus_id="acme", kind="source", id=ids["panel"], grade=1),
    ]
    await refinery.review("acme", labeled, RetrievalReview(relevance=grades))
    recording = await refinery.retrieval("acme", labeled)
    filed = {("acme", "source", g.id): g.grade for g in grades}
    served = unique([("acme", "source", h.source_id or "") for h in recording.hits])
    metrics = await refinery.measure("acme", k=2, holdout=None)
    assert (metrics.retrievals, metrics.labeled, metrics.errors, metrics.k) == (1, 1, 0, 2)
    assert set(metrics.documents) == {"search", "search.text", "search.vector"}
    fused = metrics.documents["search"]
    assert fused.recall == pytest.approx(recall_at(served, filed, 2))
    assert fused.mrr == pytest.approx(reciprocal_rank(served, filed))
    assert fused.ndcg == pytest.approx(ndcg_at(served, filed, 2))
    assert metrics.passages == {}
    assert set(metrics.latency_ms) == {"p50", "p95"}
    every = await refinery.measure("acme", holdout=None, labeled=False)
    assert (every.retrievals, every.labeled) == (3, 1)
    first = Relevance(corpus_id="acme", kind="chunk", id=recording.hits[0].id, grade=2)
    await refinery.review("acme", labeled, RetrievalReview(relevance=[first]))
    passages = (await refinery.measure("acme", k=2, holdout=None)).passages
    assert passages is not None and passages["search"].measured == 1
    assert passages["search"].mrr == 1.0


async def test_replay_reads_at_the_recorded_moment_unless_told_now(
    storage: PostgresStorage,
) -> None:
    clock = [JAN]
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery, clock=lambda: clock[0])
    await memory.check()
    ids = await _filed(memory)
    clock[0] = FEB
    retrieval_id = (await memory.search("acme", "pump seal", fusion=RRF(k=10))).retrieval_id
    assert retrieval_id is not None
    useful = [Relevance(corpus_id="acme", kind="source", id=ids["manual"], grade=2)]
    await refinery.review("acme", retrieval_id, RetrievalReview(relevance=useful))
    clock[0] = MAR
    later = (await memory.ingest("acme", name="new manual", text=MANUAL + " New seal kit.")).source
    candidate = Memory(storage, embedding=HashEmbedding(), fusion=RelativeScore())
    [then] = [pair async for pair in refinery.replay("acme", candidate, holdout=None)]
    assert then.error is None and then.candidate is not None
    assert later.source_id not in {h.source_id for h in then.candidate.hits}
    assert then.candidate.window["known_at"] == FEB
    assert then.candidate.fingerprint["fusion"]["method"] == "relative_score"
    assert then.candidate.arguments["fusion"] is None
    assert [label.source for label in then.labels] == ["human"]
    [now] = [pair async for pair in refinery.replay("acme", candidate, holdout=None, moment="now")]
    assert now.candidate is not None
    assert later.source_id in {h.source_id for h in now.candidate.hits}
    assert len((await refinery.retrievals("acme"))[0]) == 1


async def test_a_candidate_on_other_data_matches_documents_by_name(
    storage: PostgresStorage, schemas: Schemas
) -> None:
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery)
    await memory.check()
    ids = await _filed(memory)
    retrieval_id = (await memory.search("acme", "pump seal yearly", kinds=("chunk",))).retrieval_id
    assert retrieval_id is not None
    useful = [Relevance(corpus_id="acme", kind="source", id=ids["manual"], grade=2)]
    await refinery.review("acme", retrieval_id, RetrievalReview(relevance=useful))
    other = await provision(await schemas.pool(await schemas.create()))
    candidate = Memory(other, embedding=HashEmbedding(), chunk_chars=30, overlap_chars=5)
    await candidate.check()
    await _filed(candidate)
    metrics = await refinery.measure("acme", candidate, holdout=None, moment="now")
    assert metrics.errors == 0 and metrics.passages is None
    assert {name: s.recall for name, s in metrics.documents.items()} == {
        "search": 1.0,
        "search.text": 1.0,
        "search.vector": 1.0,
    }


async def test_replay_selects_the_holdout_and_reports_what_it_cannot_run(
    storage: PostgresStorage,
) -> None:
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery)
    await memory.check()
    ids = await _filed(memory)
    recorded = []
    for n in range(12):
        hits = await memory.search("acme", f"pump seal {n}", pillars={"vector"})
        assert hits.retrieval_id is not None
        recorded.append(hits.retrieval_id)
        useful = [Relevance(corpus_id="acme", kind="source", id=ids["manual"])]
        await refinery.review("acme", hits.retrieval_id, RetrievalReview(relevance=useful))
    held = {r for r in recorded if refinery.is_holdout(r)}
    chosen = {p.recorded.retrieval_id async for p in refinery.replay("acme", memory)}
    assert chosen == held
    rest = {p.recorded.retrieval_id async for p in refinery.replay("acme", memory, holdout=False)}
    assert rest == set(recorded) - held
    keyword_only = Memory(storage)
    failed = [p async for p in refinery.replay("acme", keyword_only, holdout=None, limit=2)]
    assert [p.error for p in failed] == [
        "InputError: refinery: none of the recording's pillars is configured on the candidate"
    ] * 2
    assert (await refinery.measure("acme", keyword_only, holdout=None)).errors == 12
    with pytest.raises(InputError, match=r"^refinery: moment is recorded or now$"):
        [p async for p in refinery.replay("acme", memory, moment="later")]  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^refinery: sources are among human, world and judge$"):
        await refinery.measure("acme", sources=("people",))  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^refinery: k must be at least 1$"):
        await refinery.measure("acme", k=0)


async def test_each_verb_is_scored_apart_and_selection_pages_past_live_calls(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refinery_module, "PAGE", 2)
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery)
    await memory.check()
    ids = await _filed(memory)
    useful = [Relevance(corpus_id="acme", kind="source", id=ids["manual"], grade=2)]
    labeled = []
    for n in range(4):
        retrieval_id = (
            await memory.search("acme", f"pump seal {n}", kinds=("chunk",))
        ).retrieval_id
        assert retrieval_id is not None
        labeled.append(retrieval_id)
    near = await memory.expand("acme", ("chunk", f"{ids['notes']}-0"), by="vector")
    assert near.retrieval_id is not None
    labeled.append(near.retrieval_id)
    for retrieval_id in labeled:
        await refinery.review("acme", retrieval_id, RetrievalReview(relevance=useful))
    metrics = await refinery.measure("acme", holdout=None)
    assert (metrics.retrievals, metrics.labeled) == (5, 5)
    assert set(metrics.documents) == {
        "search",
        "search.text",
        "search.vector",
        "expand",
        "expand.vector",
    }
    assert (metrics.documents["search"].measured, metrics.documents["expand"].measured) == (4, 1)
    replayed = []
    async for pair in refinery.replay("acme", memory, holdout=None):
        assert pair.error is None
        replayed.append(pair.recorded.retrieval_id)
        assert (await memory.search("acme", "wiring")).retrieval_id is not None
    assert sorted(replayed) == sorted(labeled) and len(replayed) == 5
    assert len((await refinery.retrievals("acme"))[0]) == 10
    with pytest.raises(InputError, match=r"^refinery: holdout_percent must be 0-100, got -1$"):
        await refinery.measure("acme", holdout_percent=-1)
    none_held = await refinery.measure("acme", holdout=True, holdout_percent=0)
    assert none_held.retrievals == 0
