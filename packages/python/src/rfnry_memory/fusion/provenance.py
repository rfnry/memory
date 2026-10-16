from __future__ import annotations

from collections.abc import Sequence


def provenance(
    legs: Sequence[tuple[str, Sequence[tuple[str, float]]]],
) -> dict[str, dict[str, float]]:
    found: dict[str, dict[str, float]] = {}
    for pillar, ranked in legs:
        top = max((raw for _, raw in ranked), default=0.0)
        for key, raw in ranked:
            found.setdefault(key, {})[pillar] = raw / top if top > 0 else 0.0
    return found
