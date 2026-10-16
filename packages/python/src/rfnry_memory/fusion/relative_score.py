from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from rfnry_memory.fusion.provenance import provenance


class RelativeScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["relative_score"] = "relative_score"
    weights: dict[
        Literal["text", "vector", "graph"], Annotated[float, Field(gt=0, allow_inf_nan=False)]
    ] = Field(default_factory=dict)
    window: int = Field(default=4, ge=1)

    def __hash__(self) -> int:
        return hash((self.method, tuple(sorted(self.weights.items())), self.window))


def fuse(
    settings: RelativeScore, legs: Sequence[tuple[str, Sequence[tuple[str, float]]]]
) -> list[tuple[str, float, dict[str, float]]]:
    scores: dict[str, float] = {}
    for pillar, ranked in legs:
        if not ranked:
            continue
        weight = dict[str, float](settings.weights.items()).get(pillar, 1.0)
        low, high = min(raw for _, raw in ranked), max(raw for _, raw in ranked)
        for key, raw in ranked:
            scaled = (raw - low) / (high - low) if high > low else 1.0
            scores[key] = scores.get(key, 0.0) + weight * scaled
    marks = provenance(legs)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [(key, scores[key], marks[key]) for key in ordered]
