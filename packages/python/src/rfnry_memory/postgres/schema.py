from __future__ import annotations

from typing import Any

from rfnry_memory.errors import ConfigurationError
from rfnry_memory.postgres import migrations
from rfnry_memory.postgres.base import Base

_STATE = (
    "SELECT current_schema() AS schema, current_setting('search_path') AS path, "
    "to_regclass(format('%I.memory_migrations', current_schema())) IS NOT NULL AS migrated, "
    "to_regclass(format('%I.memory_settings', current_schema())) IS NOT NULL AS settled, "
    "EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = current_schema() "
    "AND starts_with(tablename, 'memory_')) AS tables, "
    "to_regtype('vector') IS NOT NULL AS vector, "
    "to_regprocedure('similarity(text, text)') IS NOT NULL AS trigram, "
    "EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') AS trigram_installed"
)


async def _problems(
    conn: Any, *, dimension: int | None, graph: bool
) -> tuple[list[str], int | None]:
    state = await conn.fetchrow(_STATE)
    schema, path = state["schema"], state["path"]
    if schema is None:
        return [f"no schema on search_path {path!r} exists — run rfnry-memory create"], None
    if not state["migrated"] or not state["settled"]:
        if state["tables"]:
            return [
                f"the memory tables in {schema} predate migrations — create a new schema with rfnry-memory create"
            ], None
        return [f"no memory schema in {schema} — run rfnry-memory create --schema {schema}"], None
    version = int(await conn.fetchval("SELECT max(version) FROM memory_migrations") or 0)
    held = await conn.fetchval("SELECT dimension FROM memory_settings")
    problems: list[str] = []
    latest = migrations.LATEST
    if version < latest:
        problems.append(
            f"{schema} is at version {version}, this rfnry-memory needs {latest} — run rfnry-memory upgrade --schema {schema}"
        )
    elif version > latest:
        problems.append(
            f"{schema} is at version {version}, this rfnry-memory knows {latest} — upgrade rfnry-memory"
        )
    if dimension is not None and held != dimension:
        holds = "no vectors" if held is None else f"vectors of {held} dimensions"
        problems.append(
            f"{schema} holds {holds}; this memory embeds {dimension} — run rfnry-memory set --dimension {dimension} --schema {schema}, or use another schema"
        )
    if held is not None and not state["vector"]:
        problems.append(
            f"the vector extension is not reachable from search_path {path!r} — add the schema that holds it"
        )
    if graph and not state["trigram"]:
        problems.append(
            f"the pg_trgm extension is not reachable from search_path {path!r} — add the schema that holds it"
            if state["trigram_installed"]
            else f"the graph pillar needs pg_trgm — run rfnry-memory set --graph --schema {schema}"
        )
    return problems, held


class Schema(Base):
    async def check(self, *, dimension: int | None, graph: bool) -> None:
        async with self._guard("check"), self._pool.acquire() as conn:
            problems, held = await _problems(conn, dimension=dimension, graph=graph)
        if problems:
            raise ConfigurationError(f"postgres: {'; '.join(problems)}")
        self._dimension = held
