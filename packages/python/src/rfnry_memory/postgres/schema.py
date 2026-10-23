import re
from typing import Any

from rfnry_memory.errors import ConfigurationError
from rfnry_memory.postgres import migrations
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.keys import CURRENT_INDEXES

_STATE = (
    "SELECT current_schema() AS schema, current_setting('search_path') AS path, current_user AS role, "
    "to_regclass(format('%I.memory_migrations', current_schema())) IS NOT NULL AS migrated, "
    "to_regclass(format('%I.memory_settings', current_schema())) IS NOT NULL AS settled, "
    "EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = current_schema() "
    "AND starts_with(tablename, 'memory_')) AS tables, "
    "to_regtype('vector') IS NOT NULL AS vector, "
    "to_regprocedure('similarity(text, text)') IS NOT NULL AS trigram, "
    "EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') AS trigram_installed"
)


_NAMES = re.compile(r'"((?:[^"]|"")*)"|([^,\s]+)')


def _first(path: str, role: str, schema: str | None) -> str | None:
    for quoted, bare in _NAMES.findall(path):
        name: str = quoted.replace('""', '"') if quoted else bare.lower()
        if name != "$user":
            return name
        if role == schema:
            return role
    return None


async def schema_problems(
    conn: Any, *, dimension: int | None, graph: bool
) -> tuple[list[str], int | None]:
    state = await conn.fetchrow(_STATE)
    schema, path = state["schema"], state["path"]
    first = _first(path, state["role"], schema)
    if first is not None and first != schema:
        return [
            f"there is no schema {first} on search_path {path!r} — run rfnry-memory create --schema {first}"
        ], None
    if schema is None:
        return [f"no schema on search_path {path!r} exists — run rfnry-memory create"], None
    if not state["migrated"] or not state["settled"]:
        if state["tables"]:
            return [
                f"the memory tables in {schema} predate migrations — create a new schema with rfnry-memory create"
            ], None
        if migrations._shared(schema):
            return [
                f"no memory schema on search_path {path!r} — run rfnry-memory create and put its schema first on the pool's search_path"
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
            problems, held = await schema_problems(conn, dimension=dimension, graph=graph)
        if problems:
            raise ConfigurationError(f"postgres: {'; '.join(problems)}")
        self._dimension = held
        async with self._guard("check"), self._pool.acquire() as conn:
            built = await conn.fetchval(
                "SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relname = ANY($1::text[]) AND n.nspname = current_schema() AND i.indisvalid",
                CURRENT_INDEXES,
            )
        self._current = built == len(CURRENT_INDEXES)
