from typing import Any


def lock_key(name: str) -> str:
    return f"hashtextextended(current_schema() || ':' || {name}, 0)"


async def _lock(conn: Any, name: str, *, shared: bool = False) -> None:
    lock = "pg_advisory_xact_lock_shared" if shared else "pg_advisory_xact_lock"
    await conn.execute(f"SELECT {lock}({lock_key('$1')})", name)


async def _language_lock(conn: Any, corpus_id: str, *, shared: bool) -> None:
    await _lock(conn, f"memory.language:{corpus_id}", shared=shared)


async def _sync_lock(conn: Any, corpus_id: str) -> None:
    await _lock(conn, f"memory.sync:{corpus_id}")


async def _graph_lock(conn: Any, corpus_id: str) -> None:
    await _lock(conn, f"memory.graph:{corpus_id}")


async def _closings_lock(conn: Any, corpus_id: str) -> None:
    await _lock(conn, f"memory.closings:{corpus_id}")
