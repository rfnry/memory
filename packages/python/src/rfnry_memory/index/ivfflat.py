from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

SEARCH = "SELECT set_config('ivfflat.iterative_scan', 'relaxed_order', true)"


class IVFFlat(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["ivfflat"] = "ivfflat"
    lists: int | None = None
    precision: Literal["full", "half"] | None = None


def clause(kind: str, settings: IVFFlat, dimension: int) -> str:
    column = (
        f"(vector::halfvec({dimension})) halfvec_cosine_ops"
        if settings.precision == "half"
        else "vector vector_cosine_ops"
    )
    options = f"WITH (lists = {settings.lists}) " if settings.lists is not None else ""
    return f"USING ivfflat ({column}) {options}WHERE passage_kind = '{kind}'"


def read(options: list[str], half: bool) -> IVFFlat:
    values = dict(option.split("=", 1) for option in options)
    return IVFFlat(
        lists=int(values["lists"]) if "lists" in values else None,
        precision="half" if half else "full",
    )
