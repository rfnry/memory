from __future__ import annotations

from collections.abc import Sequence


def fuse_by_rank(
    legs: Sequence[tuple[str, float, Sequence[tuple[str, float]]]],
    *,
    k: int = 60,
) -> list[tuple[str, float, dict[str, float]]]:
    scores: dict[str, float] = {}
    provenance: dict[str, dict[str, float]] = {}
    pillar_max: dict[str, float] = {}
    for pillar, weight, ranked in legs:
        for rank, (key, raw) in enumerate(ranked):
            scores.setdefault(key, 0.0)
            provenance.setdefault(key, {})[pillar] = raw
            pillar_max[pillar] = max(pillar_max.get(pillar, 0.0), raw)
            scores[key] += weight / (k + rank + 1)
    for per_pillar in provenance.values():
        for pillar, raw in per_pillar.items():
            top = pillar_max[pillar]
            per_pillar[pillar] = raw / top if top else 0.0
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [(key, scores[key], provenance[key]) for key in ordered]
