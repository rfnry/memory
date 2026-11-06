import asyncio
from typing import Any

import pytest
from database import EMPTY, Schemas, Spawned, provision
from unit.fakes import HashEmbedding

from rfnry_memory.errors import ConfigurationError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Chunk, Section, Story
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.postgres.switch import _UNSTAGED
from rfnry_memory.postgres.vectors import _page_sql

NEW = "new-model"


def _keys(passages: list[Chunk | Story]) -> list[tuple[str, str]]:
    return [
        ("chunk", p.chunk_id) if isinstance(p, Chunk) else ("story", p.story_id) for p in passages
    ]


async def _filled(storage: PostgresStorage, corpus_id: str = "a") -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.ingest(
        corpus_id,
        name="manual",
        sections=[Section(text=f"Line {i} runs.") for i in range(3)],
        metadata={"line": "A"},
    )
    await memory.ingest(corpus_id, name="sheet", text="Valve notes.", metadata={"line": "B"})
    await memory.remember(corpus_id, "The pump runs.", metadata={"line": "A"})
    await memory.remember(corpus_id, "The valve leaks.", metadata={"line": "B"})
    return memory


async def _page(
    storage: PostgresStorage,
    *,
    filters: dict[str, Any] | None = None,
    after: tuple[str, str] | None = None,
    limit: int = 100,
    corpus_id: str = "a",
) -> list[tuple[str, str]]:
    return _keys(
        await storage.passages_unstaged(
            corpus_id, model=NEW, filters=filters, after=after, limit=limit
        )
    )


async def _vectors(pool: Any, corpus_id: str = "a") -> dict[tuple[str, str, str], str]:
    async with pool.acquire() as conn:
        return {
            (r["model"], r["passage_kind"], r["passage_id"]): r["vector"]
            for r in await conn.fetch(
                "SELECT model, passage_kind, passage_id, vector::text AS vector "
                "FROM memory_staged_vectors WHERE corpus_id = $1",
                corpus_id,
            )
        }


async def _waiting_on_a_lock(admin: Any, count: int) -> None:
    for _ in range(600):
        waiting = await admin.fetchval(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
            "AND datname = current_database() AND pid <> pg_backend_pid()"
        )
        if waiting >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"tests: {count} sessions never waited on a lock")


async def test_a_page_reads_chunks_then_stories_each_in_key_order(
    storage: PostgresStorage,
) -> None:
    await _filled(storage)
    keys = await _page(storage)
    assert [kind for kind, _ in keys] == ["chunk"] * 4 + ["story"] * 2
    async with storage._pool.acquire() as conn:
        for kind, table, column in (
            ("chunk", "memory_chunks", "chunk_id"),
            ("story", "memory_stories", "story_id"),
        ):
            ordered = [
                r[column]
                for r in await conn.fetch(f"SELECT {column} FROM {table} ORDER BY {column}")
            ]
            assert [i for k, i in keys if k == kind] == ordered


async def test_pages_walk_every_passage_once_from_the_key_they_are_given(
    storage: PostgresStorage,
) -> None:
    await _filled(storage)
    everything = await _page(storage)
    walked: list[tuple[str, str]] = []
    after: tuple[str, str] | None = None
    while page := await _page(storage, after=after, limit=2):
        assert len(page) <= 2
        walked.extend(page)
        after = page[-1]
    assert walked == everything
    assert await _page(storage, after=everything[3]) == everything[4:]
    assert await _page(storage, after=everything[-1]) == []


async def test_a_page_passes_what_holds_the_model_on_the_row_or_staged(
    storage: PostgresStorage,
) -> None:
    await _filled(storage)
    [first, second, *rest] = await _page(storage)
    await storage.stage("a", [first], [[0.5] * 4], NEW, settings=None)
    await storage.set_vectors("a", [second], [[0.5] * 8], NEW)
    await storage.stage("a", [rest[0]], [[0.5] * 4], "another-model", settings=None)
    assert await _page(storage) == rest
    assert await storage.count_unstaged("a", model=NEW, filters=None) == len(rest)


async def test_filters_narrow_a_page_and_its_count_for_chunks_and_stories(
    storage: PostgresStorage,
) -> None:
    await _filled(storage)
    keys = await _page(storage, filters={"metadata.line": "A"})
    assert [kind for kind, _ in keys] == ["chunk"] * 3 + ["story"]
    dated = {"created_at": {"gte": "2000-01-01T00:00:00+00:00"}}
    assert await _page(storage, filters=dated) == await _page(storage)
    assert await storage.count_unstaged("a", model=NEW, filters={"metadata.line": "A"}) == 4
    assert await storage.count_unstaged("a", model=NEW, filters=None) == 6
    counted = await storage.count_vectors("a", filters={"metadata.line": "A"})
    assert counted == {"hash-8": 4}


async def _many(pool: Any, sources: int = 400, each: int = 100) -> str:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) "
            "VALUES ('a', 'english', 'english')"
        )
        await conn.execute(
            "INSERT INTO memory_sources (corpus_id, source_id, name, text, created_at, "
            "confirmed_at, chunks) SELECT 'a', md5(s::text), 'n', 'x', now(), now(), $2 "
            "FROM generate_series(1, $1::int) s",
            sources,
            each,
        )
        await conn.execute(
            "INSERT INTO memory_chunks (corpus_id, chunk_id, source_id, position, source_name, "
            "text, digest, embedding, embedding_model, config, metadata) "
            "SELECT 'a', md5(s::text) || '-' || p, md5(s::text), p, 'n', 'pump', md5(p::text), "
            "'[1,2,3,4,5,6,7,8]', 'hash-8', 'english', "
            "jsonb_build_object('line', CASE WHEN s % 10 = 0 THEN 'A' ELSE 'B' END) "
            "FROM generate_series(1, $1::int) s, generate_series(0, $2::int - 1) p",
            sources,
            each,
        )
        middle = await conn.fetchval(
            "SELECT chunk_id FROM memory_chunks ORDER BY chunk_id OFFSET $1 LIMIT 1",
            sources * each // 2,
        )
        await conn.execute(
            "INSERT INTO memory_staged_vectors (corpus_id, model, passage_kind, passage_id, vector) "
            "SELECT 'a', $1, 'chunk', chunk_id, '[1,2,3]' FROM memory_chunks WHERE chunk_id < $2",
            NEW,
            middle,
        )
        await conn.execute(
            "ANALYZE memory_chunks; ANALYZE memory_sources; ANALYZE memory_staged_vectors"
        )
    return str(middle)


def _literal(value: Any) -> str:
    return str(value) if isinstance(value, int) else "'" + str(value).replace("'", "''") + "'"


async def _plan(conn: Any, sql: str, params: list[Any], mode: str) -> str:
    types = ", ".join("bigint" if isinstance(p, int) else "text" for p in params)
    await conn.execute(f"SET plan_cache_mode = {mode}")
    await conn.execute(f"PREPARE shown ({types}) AS {sql}")
    try:
        shown = ", ".join(_literal(p) for p in params)
        return "\n".join(r[0] for r in await conn.fetch(f"EXPLAIN EXECUTE shown({shown})"))
    finally:
        await conn.execute("DEALLOCATE shown")
        await conn.execute("RESET plan_cache_mode")


async def test_a_page_walks_the_primary_key_whatever_the_plan(
    storage: PostgresStorage, pool: Any
) -> None:
    middle = await _many(pool)
    async with pool.acquire() as conn:
        for mode in ("force_custom_plan", "force_generic_plan"):
            for filters in (None, {"metadata.line": "A"}):
                sql, params = _page_sql("chunk", "a", NEW, filters, middle, 256, _UNSTAGED["chunk"])
                plan = await _plan(conn, sql, params, mode)
                assert "memory_chunks_pkey" in plan, (mode, filters, plan)
                assert "Seq Scan on memory_chunks" not in plan, (mode, filters, plan)
                assert "ix_memory_chunks_metadata" not in plan, (mode, filters, plan)
                assert "Anti" not in plan, (mode, filters, plan)


async def test_a_page_walks_the_primary_key_over_a_table_full_of_dead_rows(
    storage: PostgresStorage, pool: Any
) -> None:
    await _many(pool)
    async with pool.acquire() as conn:
        await conn.execute("UPDATE memory_chunks SET position = position + 1")
        await conn.execute("UPDATE memory_chunks SET position = position + 1")
        await conn.execute(EMPTY)
    middle = await _many(pool)
    async with pool.acquire() as conn:
        for mode in ("force_custom_plan", "force_generic_plan"):
            sql, params = _page_sql("chunk", "a", NEW, None, middle, 256, _UNSTAGED["chunk"])
            plan = await _plan(conn, sql, params, mode)
            assert "memory_chunks_pkey" in plan, (mode, plan)
            assert "Seq Scan on memory_chunks" not in plan, (mode, plan)


async def test_stage_writes_on_its_key_and_replaces_what_the_key_held(
    storage: PostgresStorage, pool: Any
) -> None:
    await _filled(storage)
    [first, *_] = await _page(storage)
    await storage.stage("a", [first], [[0.25, 0.5]], NEW, settings={"dimensions": 2})
    await storage.stage("a", [first], [[0.5, 0.25, 1.0]], NEW, settings=None)
    await storage.stage("a", [first], [[0.75]], "another-model", settings=None)
    assert await _vectors(pool) == {
        (NEW, *first): "[0.5,0.25,1]",
        ("another-model", *first): "[0.75]",
    }
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval("SELECT settings FROM memory_staged_vectors WHERE model = $1", NEW)
            is None
        )


async def test_stage_records_the_settings_the_model_declares(
    storage: PostgresStorage, pool: Any
) -> None:
    await _filled(storage)
    [first, *_] = await _page(storage)
    await storage.stage("a", [first], [[0.25]], NEW, settings={"dimensions": 1})
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT settings::text FROM memory_staged_vectors WHERE model = $1", NEW
            )
            == '{"dimensions": 1}'
        )


async def test_stage_names_only_passages_of_its_corpus(storage: PostgresStorage, pool: Any) -> None:
    await _filled(storage)
    keys = await _page(storage)
    await storage.stage("b", keys, [[0.5]] * len(keys), NEW, settings=None)
    await storage.stage("a", [("chunk", "ghost")], [[0.5]], NEW, settings=None)
    assert await _vectors(pool, "b") == {}
    assert await _vectors(pool) == {}


async def test_staged_counts_are_by_model_sorted_and_stay_in_their_corpus(
    storage: PostgresStorage,
) -> None:
    await _filled(storage)
    await _filled(storage, "b")
    keys = await _page(storage)
    await storage.stage("a", keys[:2], [[0.5]] * 2, "zeta", settings=None)
    await storage.stage("a", keys[:1], [[0.5]], "alpha", settings=None)
    assert list((await storage.staged_counts("a")).items()) == [("alpha", 1), ("zeta", 2)]
    assert await storage.staged_counts("b") == {}


async def test_a_stage_waits_out_a_delete_and_stages_nothing_of_it(
    storage: PostgresStorage, pool: Any, spawn: Spawned
) -> None:
    memory = await _filled(storage)
    chunks = [key for key in await _page(storage) if key[0] == "chunk"]
    [source, _] = (await memory.list_sources("a"))[0]
    gone = [("chunk", c.chunk_id) for c in await storage.source_chunks("a", source.source_id)]
    assert set(gone) <= set(chunks)
    async with pool.acquire() as admin:
        deleting = admin.transaction()
        await deleting.start()
        await admin.execute(
            "DELETE FROM memory_sources WHERE corpus_id = 'a' AND source_id = $1",
            source.source_id,
        )
        staging = spawn(storage.stage("a", gone, [[0.5]] * len(gone), NEW, settings=None))
        async with pool.acquire() as watcher:
            await _waiting_on_a_lock(watcher, 1)
        assert not staging.done()
        await deleting.commit()
    await asyncio.wait_for(staging, 10)
    assert await _vectors(pool) == {}


async def test_a_stage_lets_a_delete_that_rolled_back_stand(
    storage: PostgresStorage, pool: Any, spawn: Spawned
) -> None:
    memory = await _filled(storage)
    [source, _] = (await memory.list_sources("a"))[0]
    kept = [("chunk", c.chunk_id) for c in await storage.source_chunks("a", source.source_id)]
    async with pool.acquire() as admin:
        deleting = admin.transaction()
        await deleting.start()
        await admin.execute(
            "DELETE FROM memory_sources WHERE corpus_id = 'a' AND source_id = $1",
            source.source_id,
        )
        staging = spawn(storage.stage("a", kept, [[0.5]] * len(kept), NEW, settings=None))
        async with pool.acquire() as watcher:
            await _waiting_on_a_lock(watcher, 1)
        await deleting.rollback()
    await asyncio.wait_for(staging, 10)
    assert len(await _vectors(pool)) == len(kept)


@pytest.mark.parametrize("kind", ["chunk", "story"])
async def test_a_delete_during_a_stage_takes_the_staged_vector_with_it(
    storage: PostgresStorage, pool: Any, spawn: Spawned, kind: str
) -> None:
    memory = await _filled(storage)
    [source, _] = (await memory.list_sources("a"))[0]
    stories = [key for key in await _page(storage) if key[0] == "story"]
    chunks = [("chunk", c.chunk_id) for c in await storage.source_chunks("a", source.source_id)]
    keys = chunks if kind == "chunk" else stories[:1]
    async with pool.acquire() as admin, pool.acquire() as watcher:
        held = admin.transaction()
        await held.start()
        await admin.execute("LOCK TABLE memory_staged_vectors IN EXCLUSIVE MODE")
        staging = spawn(storage.stage("a", keys, [[0.5]] * len(keys), NEW, settings=None))
        await _waiting_on_a_lock(watcher, 1)
        deleting = spawn(
            memory.delete_source("a", source.source_id)
            if kind == "chunk"
            else memory.delete_story("a", keys[0][1])
        )
        await _waiting_on_a_lock(watcher, 2)
        await held.commit()
    await asyncio.wait_for(staging, 10)
    await asyncio.wait_for(deleting, 10)
    assert await _vectors(pool) == {}


async def test_delete_corpus_waits_for_a_stage_and_clears_it(
    storage: PostgresStorage, pool: Any, spawn: Spawned
) -> None:
    memory = await _filled(storage)
    keys = await _page(storage)
    async with pool.acquire() as admin, pool.acquire() as watcher:
        held = admin.transaction()
        await held.start()
        await admin.execute("LOCK TABLE memory_staged_vectors IN EXCLUSIVE MODE")
        staging = spawn(storage.stage("a", keys, [[0.5]] * len(keys), NEW, settings=None))
        await _waiting_on_a_lock(watcher, 1)
        deleting = spawn(memory.delete_corpus("a"))
        await _waiting_on_a_lock(watcher, 2)
        await held.commit()
    await asyncio.wait_for(staging, 10)
    await asyncio.wait_for(deleting, 10)
    assert (await memory.verify_absent("a")).clean


async def test_a_schema_without_vectors_says_what_gives_it_a_staging_table(
    schemas: Schemas,
) -> None:
    name = await schemas.create()
    pool = await schemas.pool(name)
    storage = await provision(pool, dimension=None)
    message = "^postgres: this schema has no table of staged vectors — run rfnry-memory"
    with pytest.raises(ConfigurationError, match=message):
        await storage.passages_unstaged("a", model=NEW, filters=None, after=None, limit=1)
    with pytest.raises(ConfigurationError, match=message):
        await storage.count_unstaged("a", model=NEW, filters=None)
    assert await storage.staged_counts("a") == {}
    assert (await Memory(storage).fingerprint("a")).staged == {}
