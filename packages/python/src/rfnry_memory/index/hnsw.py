from __future__ import annotations

from rfnry_memory.models import VectorIndex

SEARCH = "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"


def clause(kind: str, settings: VectorIndex, dimension: int) -> str:
    column = (
        "vector vector_cosine_ops"
        if settings.precision == "full"
        else f"(vector::halfvec({dimension})) halfvec_cosine_ops"
    )
    return (
        f"USING hnsw ({column}) WITH (m = {settings.m}, ef_construction = {settings.ef_construction}) "
        f"WHERE passage_kind = '{kind}'"
    )


def read(options: list[str], half: bool) -> VectorIndex:
    values = dict(option.split("=", 1) for option in options)
    return VectorIndex(
        m=int(values["m"]),
        ef_construction=int(values["ef_construction"]),
        precision="half" if half else "full",
    )
