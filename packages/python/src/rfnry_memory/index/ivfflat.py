from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rfnry_memory.index.hnsw import column

SEARCH = "SELECT set_config('ivfflat.iterative_scan', 'relaxed_order', true)"

SEARCHED = ("rerank", "probes")


class IVFFlat(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["ivfflat"] = "ivfflat"
    lists: int | None = None
    precision: Literal["full", "half", "binary"] | None = None
    rerank: int | None = Field(default=None, ge=1)
    probes: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _probes_within_lists(self) -> IVFFlat:
        if self.probes is not None and self.lists is not None and self.probes > self.lists:
            raise ValueError("index: probes reads at most as many lists as the index has")
        return self


def clause(kind: str, settings: IVFFlat, dimension: int) -> str:
    options = f"WITH (lists = {settings.lists}) " if settings.lists is not None else ""
    return (
        f"USING ivfflat ({column(settings.precision or 'full', dimension)}) "
        f"{options}WHERE passage_kind = '{kind}'"
    )


def session(settings: dict[str, Any]) -> list[tuple[str, str]]:
    probes = settings.get("probes")
    return [] if probes is None else [("ivfflat.probes", str(probes))]


def read(options: list[str], precision: str, searched: dict[str, Any]) -> IVFFlat:
    values = dict(option.split("=", 1) for option in options)
    return IVFFlat.model_validate(
        {
            "lists": int(values["lists"]) if "lists" in values else None,
            "precision": precision,
            **{name: searched[name] for name in SEARCHED if name in searched},
        }
    )
