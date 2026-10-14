from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

SEARCH = "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"


class HNSW(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["hnsw"] = "hnsw"
    m: int | None = None
    ef_construction: int | None = None
    precision: Literal["full", "half"] | None = None


def clause(kind: str, settings: HNSW, dimension: int) -> str:
    column = (
        f"(vector::halfvec({dimension})) halfvec_cosine_ops"
        if settings.precision == "half"
        else "vector vector_cosine_ops"
    )
    options = ", ".join(
        f"{name} = {value}"
        for name, value in (("m", settings.m), ("ef_construction", settings.ef_construction))
        if value is not None
    )
    return f"USING hnsw ({column}) {f'WITH ({options}) ' if options else ''}WHERE passage_kind = '{kind}'"


def read(options: list[str], half: bool) -> HNSW:
    values = dict(option.split("=", 1) for option in options)
    return HNSW(
        m=int(values["m"]) if "m" in values else None,
        ef_construction=int(values["ef_construction"]) if "ef_construction" in values else None,
        precision="half" if half else "full",
    )
