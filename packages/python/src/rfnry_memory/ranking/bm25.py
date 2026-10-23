from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BM25(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["bm25"] = "bm25"
    k1: float = Field(default=1.2, ge=0, allow_inf_nan=False)
    b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    statistics: Literal["read", "corpus"] = "read"


def sql(
    settings: BM25,
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
    k1, b = settings.k1, settings.b
    stats = "stats AS (SELECT count(*)::float8 AS n, avg(len) AS avgdl FROM sized), "
    df = "df AS (SELECT lexeme, count(*)::float8 AS df FROM freq GROUP BY lexeme) "
    if settings.statistics == "corpus":
        stats = (
            "stats AS (SELECT greatest(coalesce(sum(passages), 0), 1)::float8 AS n, "
            "coalesce(nullif(sum(length)::float8 / nullif(sum(passages), 0), 0), 1) AS avgdl "
            f"FROM memory_term_totals WHERE corpus_id = ANY($1::text[]) AND kind = '{kind}'), "
        )
        df = (
            "df AS (SELECT t.lexeme, greatest(coalesce((SELECT sum(c.passages) FROM memory_term_counts c "
            f"WHERE c.corpus_id = ANY($1::text[]) AND c.kind = '{kind}' AND c.lexeme = t.lexeme), 0), 1)"
            "::float8 AS df FROM terms AS t) "
        )
    return (
        f"WITH matched AS (SELECT corpus_id AS corpus, {id_column} AS id, fts, fts_length FROM {table} "
        f"WHERE corpus_id = ANY($1::text[]) AND fts @@ {matches}{where}), "
        f"terms AS (SELECT DISTINCT l AS lexeme FROM unnest(tsvector_to_array({terms})) AS l), "
        "sized AS (SELECT m.corpus, m.id, m.fts_length::float8 AS len FROM matched AS m), "
        "freq AS (SELECT m.corpus, m.id, u.lexeme, "
        "coalesce(array_length(u.positions, 1), 1)::float8 AS tf "
        "FROM matched AS m CROSS JOIN LATERAL unnest(m.fts) AS u "
        "WHERE u.lexeme IN (SELECT lexeme FROM terms)), "
        f"{stats}{df}"
        f"SELECT f.corpus, f.id, sum(ln(1 + (stats.n - df.df + 0.5) / (df.df + 0.5)) * f.tf * {k1 + 1} "
        f"/ (f.tf + {k1} * (1 - {b} + {b} * sized.len / stats.avgdl))) AS score, 0::float8 AS tie "
        "FROM freq AS f JOIN df ON df.lexeme = f.lexeme "
        "JOIN sized ON sized.corpus = f.corpus AND sized.id = f.id CROSS JOIN stats "
        "GROUP BY f.corpus, f.id "
        f'ORDER BY score DESC, f.id COLLATE "C", f.corpus COLLATE "C" LIMIT {limit}'
    )
