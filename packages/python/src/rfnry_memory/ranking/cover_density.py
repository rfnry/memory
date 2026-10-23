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
    where: str,
    limit: str,
    kind: str,
) -> str:
    length = _NORMALIZATION[settings.length]
    return (
        f"SELECT corpus_id AS corpus, {id_column} AS id, "
        f"ts_rank_cd(fts, {ranks}, {length})::float8 AS score, "
        f"ts_rank_cd(fts, {matches}, {length})::float8 AS tie "
        f"FROM {table} WHERE corpus_id = ANY($1::text[]) AND fts @@ {matches}{where} "
        f'ORDER BY score DESC, tie DESC, {id_column} COLLATE "C", corpus_id COLLATE "C" '
        f"LIMIT {limit}"
    )
