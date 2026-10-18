from typing import Any

import asyncpg

from rfnry_memory.models import (
    Counts,
    Language,
)
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.language import _LANGUAGE, _configuration, _pin
from rfnry_memory.postgres.locks import _language_lock, _sync_lock
from rfnry_memory.postgres.records import _count
from rfnry_memory.postgres.vectors import _vectors_lock, _vectors_table


async def _held_vectors(conn: Any, corpus_id: str, counts: dict[str, int]) -> str | None:
    index_table = await _vectors_table(conn, corpus_id)
    if index_table is None:
        return None
    try:
        counts["vectors"] = int(await conn.fetchval(f"SELECT count(*) FROM {index_table}"))
    except asyncpg.UndefinedTableError:
        return None
    counts["corpora"] += 1
    return index_table


class Corpus(Base):
    async def delete_corpus(self, corpus_id: str) -> Counts:
        async with (
            self._guard("delete_corpus"),
            self._pool.acquire() as conn,
            _vectors_lock(conn, corpus_id),
            conn.transaction(isolation="read_committed"),
        ):
            await _sync_lock(conn, corpus_id)
            await _language_lock(conn, corpus_id, shared=False)
            counts = {}
            for field, table in (
                ("mentions", "memory_mentions"),
                ("relations", "memory_relations"),
                ("entities", "memory_entities"),
                ("stories", "memory_stories"),
                ("chunks", "memory_chunks"),
                ("sources", "memory_sources"),
                ("corpora", "memory_corpora"),
            ):
                counts[field] = _count(
                    await conn.execute(f"DELETE FROM {table} WHERE corpus_id = $1", corpus_id)
                )
            index_table = await _held_vectors(conn, corpus_id, counts)
            if index_table is not None:
                await conn.execute(f"DROP TABLE {index_table}")
        return Counts(**counts)

    async def count_corpus(self, corpus_id: str) -> Counts:
        async with self._guard("count_corpus"), self._pool.acquire() as conn:
            counts = {}
            for field, table in (
                ("sources", "memory_sources"),
                ("chunks", "memory_chunks"),
                ("stories", "memory_stories"),
                ("entities", "memory_entities"),
                ("relations", "memory_relations"),
                ("mentions", "memory_mentions"),
                ("corpora", "memory_corpora"),
            ):
                counts[field] = int(
                    await conn.fetchval(
                        f"SELECT count(*) FROM {table} WHERE corpus_id = $1", corpus_id
                    )
                )
            await _held_vectors(conn, corpus_id, counts)
        return Counts(**counts)

    async def text_configs(self, corpus_ids: list[str]) -> list[str]:
        async with self._guard("text_configs"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT c.corpus_id, coalesce(m.config, (SELECT config FROM memory_settings)) AS config "
                "FROM unnest($1::text[]) AS c(corpus_id) "
                "LEFT JOIN memory_corpora m ON m.corpus_id = c.corpus_id",
                list(corpus_ids),
            )
        configs = {r["corpus_id"]: str(r["config"]) for r in records}
        return [configs[c] for c in corpus_ids]

    async def get_language(self, corpus_id: str) -> Language:
        async with self._guard("get_language"), self._pool.acquire() as conn:
            language = await conn.fetchval(_LANGUAGE, corpus_id)
        return str(language)

    async def set_language(self, corpus_id: str, language: Language) -> int | None:
        async with self._guard("set_language"), self._pool.acquire() as conn:
            found = await _configuration(conn, language)
            if found is None:
                return None
            name, config, shipped = found
            stale = " AND config <> $2" if shipped else ""
            async with conn.transaction(isolation="read_committed"):
                await _language_lock(conn, corpus_id, shared=False)
                await conn.execute(
                    "INSERT INTO memory_corpora (corpus_id, language, config) VALUES ($1, $2, $3) "
                    "ON CONFLICT (corpus_id) DO UPDATE SET language = EXCLUDED.language, config = EXCLUDED.config",
                    corpus_id,
                    name,
                    config,
                )
                changed = 0
                for table in ("memory_chunks", "memory_stories"):
                    changed += _count(
                        await conn.execute(
                            f"UPDATE {table} SET config = $2 WHERE corpus_id = $1{stale}",
                            corpus_id,
                            config,
                        )
                    )
                await conn.execute(
                    f"UPDATE memory_entities SET config = $2 WHERE corpus_id = $1{stale}",
                    corpus_id,
                    config,
                )
        return changed

    async def get_history(self, corpus_id: str) -> bool:
        async with self._guard("get_history"), self._pool.acquire() as conn:
            history = await conn.fetchval(
                "SELECT history FROM memory_corpora WHERE corpus_id = $1", corpus_id
            )
        return bool(history)

    async def set_history(self, corpus_id: str, keep: bool) -> None:
        async with (
            self._guard("set_history"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _sync_lock(conn, corpus_id)
            await _pin(conn, corpus_id)
            await conn.execute(
                "UPDATE memory_corpora SET history = $2 WHERE corpus_id = $1", corpus_id, keep
            )
