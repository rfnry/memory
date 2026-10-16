from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from rfnry_memory.fusion.provenance import provenance


class RRF(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["rrf"] = "rrf"
    k: int = Field(default=60, ge=1)
    weights: dict[
        Literal["text", "vector", "graph"], Annotated[float, Field(gt=0, allow_inf_nan=False)]
    ] = Field(default_factory=dict)
    window: int = Field(default=4, ge=1)

    def __hash__(self) -> int:
        return hash((self.method, self.k, tuple(sorted(self.weights.items())), self.window))


def fuse(
    settings: RRF, legs: Sequence[tuple[str, Sequence[tuple[str, float]]]]
) -> list[tuple[str, float, dict[str, float]]]:
    scores: dict[str, float] = {}
    for pillar, ranked in legs:
        weight = dict[str, float](settings.weights.items()).get(pillar, 1.0)
        for rank, (key, _) in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + weight / (settings.k + rank + 1)
    marks = provenance(legs)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [(key, scores[key], marks[key]) for key in ordered]
