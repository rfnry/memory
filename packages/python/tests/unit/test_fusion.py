import pytest

from rfnry_memory.fusion import fuse_by_rank


def test_rank_fusion_orders_by_reciprocal_rank_and_normalizes_provenance() -> None:
    legs = [
        ("text", 1.0, [("a", 10.0), ("b", 5.0)]),
        ("vector", 1.0, [("b", 0.9), ("c", 0.8)]),
    ]
    fused = fuse_by_rank(legs, k=60)
    keys = [key for key, _, _ in fused]
    assert keys[0] == "b" and set(keys) == {"a", "b", "c"}
    scores = {key: score for key, score, _ in fused}
    assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)
    provenance = {key: prov for key, _, prov in fused}
    assert provenance["a"] == {"text": 1.0} and provenance["b"] == {"text": 0.5, "vector": 1.0}
    assert "text" not in provenance["c"]


def test_weights_scale_a_pillar() -> None:
    fused = fuse_by_rank([("text", 1.0, [("a", 1.0)]), ("graph", 3.0, [("b", 1.0)])])
    assert [key for key, _, _ in fused] == ["b", "a"]
