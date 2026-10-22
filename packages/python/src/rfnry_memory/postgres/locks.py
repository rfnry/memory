from typing import Any


async def _language_lock(conn: Any, corpus_id: str, *, shared: bool) -> None:
    lock = "pg_advisory_xact_lock_shared" if shared else "pg_advisory_xact_lock"
    await conn.execute(
        f"SELECT {lock}(hashtextextended(current_schema() || ':' || $1, 0))",
        f"memory.language:{corpus_id}",
    )


async def _sync_lock(conn: Any, corpus_id: str) -> None:
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || ':' || $1, 0))",
        f"memory.sync:{corpus_id}",
    )
