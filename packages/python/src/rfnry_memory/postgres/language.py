from __future__ import annotations

from typing import Any

_DEFAULT = "(SELECT language, config FROM memory_settings)"


_CONFIG = "(SELECT config FROM memory_corpora WHERE corpus_id = $1)"


_QUERY_CONFIG = (
    "coalesce((SELECT config FROM memory_corpora WHERE corpus_id = ($1::text[])[1]), "
    f"(SELECT config FROM {_DEFAULT} AS d))::regconfig"
)


_LANGUAGE = (
    "SELECT coalesce((SELECT language FROM memory_corpora WHERE corpus_id = $1), "
    f"(SELECT language FROM {_DEFAULT} AS d))"
)


_PIN = (
    "INSERT INTO memory_corpora (corpus_id, language, config) "
    f"SELECT $1, language, config FROM {_DEFAULT} AS d ON CONFLICT (corpus_id) DO NOTHING"
)


_NOT_A_NAME = {"42601", "42602", "42704", "0A000", "22003", "3F000"}


async def _pin(conn: Any, corpus_id: str) -> None:
    await conn.execute(_PIN, corpus_id)


async def _configuration(conn: Any, language: str) -> tuple[str, str, bool] | None:
    if language == "-" or language.isdigit():
        return None
    try:
        found = await conn.fetchrow(
            "SELECT $1::regconfig::text AS language, n.nspname = 'pg_catalog' AS shipped, "
            "format('%I.%I', n.nspname, c.cfgname) AS config FROM pg_ts_config c "
            "JOIN pg_namespace n ON n.oid = c.cfgnamespace WHERE c.oid = $1::regconfig",
            language,
        )
    except Exception as exc:
        if getattr(exc, "sqlstate", None) in _NOT_A_NAME:
            return None
        raise
    if found is None:
        return None
    return found["language"], found["config"], found["shipped"]
