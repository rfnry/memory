from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SEARCH = "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"

SEARCHED = ("rerank", "ef_search")


class HNSW(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["hnsw"] = "hnsw"
    m: int | None = None
    ef_construction: int | None = None
    precision: Literal["full", "half", "binary"] | None = None
    rerank: int | None = Field(default=None, ge=1)
    ef_search: int | None = Field(default=None, ge=1, le=1000)


def column(precision: str, dimension: int) -> str:
    if precision == "half":
        return f"(vector::halfvec({dimension})) halfvec_cosine_ops"
    if precision == "binary":
        return f"(binary_quantize(vector)::bit({dimension})) bit_hamming_ops"
    return "vector vector_cosine_ops"


def clause(kind: str, settings: HNSW, dimension: int) -> str:
    options = ", ".join(
        f"{name} = {value}"
        for name, value in (("m", settings.m), ("ef_construction", settings.ef_construction))
        if value is not None
    )
    return (
        f"USING hnsw ({column(settings.precision or 'full', dimension)}) "
        f"{f'WITH ({options}) ' if options else ''}WHERE passage_kind = '{kind}'"
    )


def session(settings: dict[str, Any]) -> list[tuple[str, str]]:
    ef_search = settings.get("ef_search")
    return [] if ef_search is None else [("hnsw.ef_search", str(ef_search))]


def read(options: list[str], precision: str, searched: dict[str, Any]) -> HNSW:
    values = dict(option.split("=", 1) for option in options)
    return HNSW.model_validate(
        {
            "m": int(values["m"]) if "m" in values else None,
            "ef_construction": int(values["ef_construction"])
            if "ef_construction" in values
            else None,
            "precision": precision,
            **{name: searched[name] for name in SEARCHED if name in searched},
        }
    )
