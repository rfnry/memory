from __future__ import annotations

from typing import Any

_ENGLISH = "format('%I.%I', current_schema(), 'memory_english')"


_CONFIG = f"coalesce((SELECT config FROM memory_corpora WHERE corpus_id = $1), {_ENGLISH})"


_QUERY_CONFIG = (
    "coalesce((SELECT config FROM memory_corpora WHERE corpus_id = ($1::text[])[1]), "
    f"{_ENGLISH})::regconfig"
)


_NOT_A_NAME = {"42601", "42602", "42704", "0A000", "22003", "3F000"}


_FOLDING = """SELECT (SELECT format('%I.unaccent', n.nspname) FROM pg_extension e
        JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'unaccent') AS unaccent,
    string_agg(format('%I.%I', dn.nspname, d.dictname), ', ' ORDER BY m.mapseqno) AS dictionaries,
    format('pg_catalog.%I', $1::text) AS base
FROM pg_ts_config c
JOIN LATERAL ts_token_type(c.cfgparser) t ON t.alias = 'word'
JOIN pg_ts_config_map m ON m.mapcfg = c.oid AND m.maptokentype = t.tokid
JOIN pg_ts_dict d ON d.oid = m.mapdict
JOIN pg_namespace dn ON dn.oid = d.dictnamespace
WHERE c.cfgnamespace = 'pg_catalog'::regnamespace AND c.cfgname = $1"""


async def _folded(conn: Any, name: str) -> str:
    copy = f"memory_{name}"
    held = (
        "SELECT format('%I.%I', current_schema(), $1::text) AS config, EXISTS (SELECT 1 FROM pg_ts_config "
        "WHERE cfgname = $1 AND cfgnamespace = current_schema()::regnamespace) AS held"
    )
    found = await conn.fetchrow(held, copy)
    if found["held"]:
        return str(found["config"])
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", f"memory.config:{copy}"
    )
    if (await conn.fetchrow(held, copy))["held"]:
        return str(found["config"])
    folding = await conn.fetchrow(_FOLDING, name)
    await conn.execute(
        f"CREATE TEXT SEARCH CONFIGURATION {found['config']} (COPY = {folding['base']})"
    )
    await conn.execute(
        f"ALTER TEXT SEARCH CONFIGURATION {found['config']} ALTER MAPPING FOR hword, hword_part, word "
        f"WITH {folding['unaccent']}, {folding['dictionaries']}"
    )
    return str(found["config"])


async def _configuration(conn: Any, language: str) -> tuple[str, str, bool] | None:
    if language == "-" or language.isdigit():
        return None
    try:
        found = await conn.fetchrow(
            "SELECT $1::regconfig::text AS language, n.nspname, c.cfgname, "
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
    if found["nspname"] != "pg_catalog":
        return found["language"], found["config"], False
    async with conn.transaction(isolation="read_committed"):
        return found["language"], await _folded(conn, found["cfgname"]), True
