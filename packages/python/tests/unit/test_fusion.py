import asyncio

import pytest
from pydantic import TypeAdapter, ValidationError

from rfnry_memory import BM25, RRF, CorpusFingerprint, CoverDensity, Fusion, Memory, RelativeScore
from rfnry_memory.errors import InputError
from rfnry_memory.fusion import relative_score, rrf
from rfnry_memory.postgres import PostgresStorage

from .fakes import HashEmbedding


def test_rank_fusion_orders_by_reciprocal_rank_and_normalizes_provenance() -> None:
    legs = [
        ("text", [("a", 10.0), ("b", 5.0)]),
        ("vector", [("b", 0.9), ("c", 0.8)]),
    ]
    fused = rrf.fuse(RRF(), legs)
    keys = [key for key, _, _ in fused]
    assert keys[0] == "b" and set(keys) == {"a", "b", "c"}
    scores = {key: score for key, score, _ in fused}
    assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)
    provenance = {key: prov for key, _, prov in fused}
    assert provenance["a"] == {"text": 1.0} and provenance["b"] == {"text": 0.5, "vector": 1.0}
    assert "text" not in provenance["c"]


def test_k_sets_how_much_the_top_of_each_list_counts() -> None:
    legs = [
        ("text", [("x", 4.0), ("p", 3.0), ("q", 2.0), ("y", 1.0)]),
        ("vector", [("r", 0.9), ("s", 0.8), ("t", 0.7), ("y", 0.6)]),
    ]
    assert rrf.fuse(RRF(k=60), legs)[0][0] == "y"
    assert rrf.fuse(RRF(k=1), legs)[0][0] == "x"


def test_weights_scale_a_pillar() -> None:
    legs = [("text", [("a", 1.0)]), ("graph", [("b", 1.0)])]
    assert [key for key, _, _ in rrf.fuse(RRF(), legs)] == ["a", "b"]
    assert [key for key, _, _ in rrf.fuse(RRF(weights={"graph": 3.0}), legs)] == ["b", "a"]
    assert [
        key for key, _, _ in relative_score.fuse(RelativeScore(weights={"graph": 3.0}), legs)
    ] == [
        "b",
        "a",
    ]


def test_relative_score_scales_each_pillar_by_its_own_range_and_adds_them() -> None:
    legs = [
        ("text", [("a", 12.0), ("b", 7.0), ("c", 2.0)]),
        ("vector", [("c", 0.9), ("a", 0.5), ("d", 0.1)]),
    ]
    fused = relative_score.fuse(RelativeScore(), legs)
    scores = {key: score for key, score, _ in fused}
    assert scores == pytest.approx({"a": 1.0 + 0.5, "b": 0.5, "c": 0.0 + 1.0, "d": 0.0})
    assert [key for key, _, _ in fused] == ["a", "c", "b", "d"]
    provenance = {key: prov for key, _, prov in fused}
    assert provenance["c"] == pytest.approx({"text": 2.0 / 12.0, "vector": 1.0})


def test_relative_score_reads_equal_scores_and_a_single_hit_as_the_top() -> None:
    fused = relative_score.fuse(
        RelativeScore(),
        [("text", [("a", 4.0), ("b", 4.0)]), ("vector", [("c", 0.3)]), ("graph", [])],
    )
    assert {key: score for key, score, _ in fused} == {"a": 1.0, "b": 1.0, "c": 1.0}
    assert [key for key, _, _ in fused] == ["a", "b", "c"]


def test_fusion_settings_refuse_what_they_cannot_mean() -> None:
    for bad in (
        {"k": 0},
        {"k": 1.5},
        {"window": 0},
        {"weights": {"sonar": 1.0}},
        {"weights": {"text": 0.0}},
        {"weights": {"text": -1.0}},
        {"weights": {"text": float("inf")}},
        {"boost": 2},
    ):
        with pytest.raises(ValidationError):
            RRF(**bad)
    with pytest.raises(ValidationError):
        RelativeScore(k=60)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        RRF().k = 10


def test_a_fusion_round_trips_through_json() -> None:
    adapter: TypeAdapter[Fusion] = TypeAdapter(Fusion)
    for setting in (
        RRF(k=20, weights={"vector": 2.0}, window=6),
        RelativeScore(weights={"text": 0.3}),
    ):
        assert adapter.validate_json(setting.model_dump_json()) == setting


async def test_the_default_fusion_is_reciprocal_rank_at_k_60(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    assert memory.fusion == RRF(k=60, weights={}, window=4)
    assert repr(memory.fusion) == "RRF(method='rrf', k=60, weights={}, window=4)"
    with pytest.raises(TypeError):
        Memory(storage, weights={"text": 2.0})  # type: ignore[call-arg]


async def test_a_call_can_fuse_its_own_way_and_leaves_the_memory_as_it_was(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    for i in range(6):
        await memory.ingest("a", name=f"note {i}", text=f"Pump seal note {i}.")
    wide = await memory.search("a", "pump seal", kinds=("chunk",), top_k=1)
    assert wide.pillars["text"].returned == 4
    narrow = await memory.search(
        "a", "pump seal", kinds=("chunk",), top_k=1, fusion=RelativeScore(window=2)
    )
    assert narrow.pillars["text"].returned == 2 and len(narrow.hits) == 1
    assert memory.fusion == RRF()
    with pytest.raises(
        InputError, match=r"^memory: fusion is RRF\(\.\.\.\) or RelativeScore\(\.\.\.\)$"
    ):
        await memory.search("a", "pump seal", fusion="rrf")  # type: ignore[arg-type]


async def test_the_fingerprint_reports_the_fusion(storage: PostgresStorage) -> None:
    await Memory(storage).ingest("a", name="manual", text="Check the valve.")
    assert (await Memory(storage).fingerprint("a")).fusion == RRF()
    chosen = Memory(storage, fusion=RelativeScore(weights={"text": 2.0}))
    assert (await chosen.fingerprint("a")).fusion == RelativeScore(weights={"text": 2.0})


async def test_concurrent_calls_keep_their_own_settings(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    for i in range(6):
        await memory.ingest("a", name=f"note {i}", text=f"Pump seal note {i}.")
    chosen: list[tuple[BM25 | CoverDensity | None, RRF | RelativeScore | None]] = [
        (BM25(b=0), RRF(window=1)),
        (CoverDensity(), RelativeScore(window=2)),
        (None, None),
    ]

    async def one(
        text: BM25 | CoverDensity | None, fusion: RRF | RelativeScore | None
    ) -> list[str]:
        hits = await memory.search(
            "a", "pump seal", kinds=("chunk",), top_k=1, text=text, fusion=fusion
        )
        return [str(hits.pillars["text"].returned), *(hit.id for hit in hits.hits)]

    alone = [await one(text, fusion) for text, fusion in chosen]
    together = await asyncio.gather(*(one(text, fusion) for text, fusion in chosen))
    assert list(together) == alone
    assert [found[0] for found in alone] == ["1", "2", "4"]
    assert memory.text == BM25() and memory.fusion == RRF()


async def test_a_fingerprint_kept_as_json_reads_back_its_settings(storage: PostgresStorage) -> None:
    memory = Memory(
        storage,
        text=CoverDensity(length="log"),
        fusion=RelativeScore(weights={"text": 2.0}, window=3),
    )
    await memory.ingest("a", name="manual", text="Check the valve.")
    kept = (await memory.fingerprint("a")).model_dump_json()
    read = CorpusFingerprint.model_validate_json(kept)
    assert (read.text, read.fusion) == (memory.text, memory.fusion)


def test_a_fusion_setting_hashes_like_the_other_settings() -> None:
    first = RRF(weights={"text": 2.0, "vector": 1.0})
    second = RRF(weights={"vector": 1.0, "text": 2.0})
    assert first == second and hash(first) == hash(second)
    assert len({RRF(), RelativeScore(), BM25(), CoverDensity(), first, second}) == 5
    assert {RelativeScore(window=2): "narrow"}[RelativeScore(window=2)] == "narrow"
