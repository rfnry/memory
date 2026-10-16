from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

_NORMALIZATION = {"ignore": 0, "log": 1, "linear": 2}


class CoverDensity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["cover_density"] = "cover_density"
    length: Literal["ignore", "log", "linear"] = "ignore"


def sql(
    settings: CoverDensity,
    *,
    table: str,
    id_column: str,
    matches: str,
    ranks: str,
    terms: str,
    scope: str,
    limit: str,
) -> str:
    return (
        f"SELECT corpus_id AS corpus, {id_column} AS id, "
        f"ts_rank_cd(fts, {ranks}, {_NORMALIZATION[settings.length]})::float8 AS score "
        f"FROM {table} WHERE corpus_id = ANY($1::text[]) AND fts @@ {matches}{scope} "
        f'ORDER BY score DESC, {id_column} COLLATE "C", corpus_id COLLATE "C" LIMIT {limit}'
    )
