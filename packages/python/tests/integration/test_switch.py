import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from database import EMPTY, Schemas, Spawned, executed_plan, provision, read_once
from unit.fakes import HashEmbedding

from rfnry_memory.errors import ConfigurationError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Chunk, Kind, Section, Story, Window
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.postgres.sql import _scope_sql, _table
from rfnry_memory.postgres.switch import _UNSTAGED
from rfnry_memory.postgres.vectors import _EVERY, _page_sql

NEW = "new-model"
PAGE = 256
DATED = {"created_at": {"gte": "2000-01-01T00:00:00+00:00"}}


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
    assert await _page(storage, filters=DATED) == await _page(storage)
    assert await storage.count_unstaged("a", model=NEW, filters={"metadata.line": "A"}) == 4
    assert await storage.count_unstaged("a", model=NEW, filters=None) == 6
    counted = await storage.count_vectors("a", filters={"metadata.line": "A"})
    assert counted == {"hash-8": 4}


async def _many(pool: Any, sources: int = 400, each: int = 100) -> dict[Kind, str]:
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
        await conn.execute(
            "INSERT INTO memory_stories (corpus_id, story_id, text, text_hash, valid_from, "
            "created_at, embedding, embedding_model, config, metadata) "
            "SELECT 'a', md5(s::text), 'pump', md5(s::text), now(), now(), "
            "'[1,2,3,4,5,6,7,8]', 'hash-8', 'english', "
            "json_build_object('line', CASE WHEN s / 100 % 10 = 0 THEN 'A' ELSE 'B' END) "
            "FROM generate_series(1, $1::int) s",
            sources * each,
        )
        middles: dict[Kind, str] = {}
        for kind in ("chunk", "story"):
            table, key = _table(kind)
            middles[kind] = await conn.fetchval(
                f"SELECT {key} FROM {table} ORDER BY {key} OFFSET $1 LIMIT 1",
                sources * each // 2,
            )
            await conn.execute(
                "INSERT INTO memory_staged_vectors (corpus_id, model, passage_kind, passage_id, "
                f"vector) SELECT 'a', $1, '{kind}', {key}, '[1,2,3]' FROM {table} WHERE {key} < $2",
                NEW,
                middles[kind],
            )
        await conn.execute(
            "ANALYZE memory_chunks; ANALYZE memory_stories; ANALYZE memory_sources; "
            "ANALYZE memory_staged_vectors"
        )
    return middles


def _nodes(plan: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield plan
    for child in plan.get("Plans", []):
        if child["Parent Relationship"] not in ("SubPlan", "InitPlan"):
            yield from _nodes(child)


def _rows_read_along_the_key(plan: dict[str, Any], table: str) -> int:
    limit = next(node for node in _nodes(plan) if node["Node Type"] == "Limit")
    assert limit["Plan Rows"] <= PAGE, plan
    walk = list(_nodes(limit))[1:]
    assert {node["Node Type"] for node in walk} <= {"Subquery Scan", "Index Scan"}, plan
    (scan,) = [node for node in walk if node["Node Type"] == "Index Scan"]
    assert scan["Index Name"] == f"{table}_pkey", plan
    assert scan["Scan Direction"] == "Forward", plan
    return int(scan["Actual Rows"] + scan.get("Rows Removed by Filter", 0))


async def _every_page_walks_the_key(conn: Any, middles: dict[Kind, str]) -> None:
    for kind, middle in middles.items():
        table, _ = _table(kind)
        for wanted in (_UNSTAGED[kind], _EVERY[kind]):
            for mode in ("force_custom_plan", "force_generic_plan"):
                for filters in (None, {"metadata.line": "A"}, DATED):
                    sql, params = _page_sql(kind, "a", NEW, filters, middle, PAGE, wanted)
                    read = _rows_read_along_the_key(
                        await executed_plan(conn, sql, params, mode), table
                    )
                    assert read < 10000, (kind, wanted, mode, filters, read)
                    sql, params = _page_sql(kind, "a", NEW, filters, None, PAGE, wanted)
                    _rows_read_along_the_key(await executed_plan(conn, sql, params, mode), table)


async def _untouched(schemas: Schemas) -> Any:
    pool = await schemas.pool(await schemas.create())
    await provision(pool)
    return pool


async def test_a_page_walks_the_primary_key_whatever_the_plan(schemas: Schemas) -> None:
    pool = await _untouched(schemas)
    middles = await _many(pool)
    async with pool.acquire() as conn:
        await _every_page_walks_the_key(conn, middles)


async def test_a_page_walks_the_primary_key_over_a_table_full_of_dead_rows(
    schemas: Schemas,
) -> None:
    pool = await _untouched(schemas)
    await _many(pool)
    async with pool.acquire() as conn:
        await conn.execute("UPDATE memory_chunks SET position = position + 1")
        await conn.execute("UPDATE memory_chunks SET position = position + 1")
        await conn.execute(EMPTY)
    middles = await _many(pool)
    async with pool.acquire() as conn:
        await _every_page_walks_the_key(conn, middles)


async def test_a_page_walks_the_primary_key_while_half_the_rows_hold_the_model(
    schemas: Schemas,
) -> None:
    pool = await _untouched(schemas)
    middles = await _many(pool)
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memory_staged_vectors")
        for kind, middle in middles.items():
            table, key = _table(kind)
            await conn.execute(
                f"UPDATE {table} SET embedding_model = $1 WHERE {key} < $2", NEW, middle
            )
        await conn.execute(
            "ANALYZE memory_chunks; ANALYZE memory_stories; ANALYZE memory_staged_vectors"
        )
        await _every_page_walks_the_key(conn, middles)


@pytest.mark.parametrize("kind", ["chunk", "story"])
@pytest.mark.parametrize("filters", [{"metadata.line": "A"}, {"metadata.page": {"gte": 1}}])
async def test_a_filter_value_is_read_once_a_page_and_not_once_a_row(
    kind: Kind, filters: dict[str, Any], storage: PostgresStorage, pool: Any
) -> None:
    sql, params = _page_sql(kind, "a", NEW, filters, None, PAGE, _EVERY[kind])
    async with pool.acquire() as conn:
        assert read_once(await executed_plan(conn, sql, params, "force_generic_plan")) == 2


@pytest.mark.parametrize("kind", ["chunk", "story"])
@pytest.mark.parametrize("filters", [{"metadata.line": "A"}, {"metadata.page": {"gte": 1}}])
def test_only_a_walk_reads_a_filter_value_apart_from_the_rows(
    kind: Kind, filters: dict[str, Any]
) -> None:
    window = Window()
    assert "(SELECT" not in _scope_sql(kind, filters, window, None, None, [])
    apart = _scope_sql(kind, filters, window, None, None, [], read_once=True)
    assert "(SELECT $1::text::jsonb)" in apart


async def _walked(
    conn: Any, kind: Kind, wanted: str, filters: dict[str, Any] | None, mode: str
) -> list[str]:
    _, key = _table(kind)
    keys: list[str] = []
    after = None
    await conn.execute(f"SET plan_cache_mode = {mode}")
    try:
        while True:
            sql, params = _page_sql(kind, "a", NEW, filters, after, 300, wanted)
            rows = await conn.fetch(sql, *params)
            if not rows:
                return keys
            after = rows[-1][key]
            keys.extend(r[key] for r in rows)
    finally:
        await conn.execute("RESET plan_cache_mode")


@pytest.mark.parametrize("kind", ["chunk", "story"])
@pytest.mark.parametrize("mode", ["force_custom_plan", "force_generic_plan"])
async def test_a_walk_of_many_pages_reads_the_same_rows_as_a_plain_read(
    kind: Kind, mode: str, schemas: Schemas
) -> None:
    pool = await _untouched(schemas)
    middle = (await _many(pool, sources=30))[kind]
    table, key = _table(kind)
    async with pool.acquire() as conn:
        for wanted, start in ((_UNSTAGED[kind], middle), (_EVERY[kind], "")):
            for filters, line in ((None, "{}"), ({"metadata.line": "A"}, '{"line": "A"}')):
                plain = await conn.fetch(
                    f"SELECT {key} FROM {table} WHERE {key} >= $1 "
                    f"AND metadata::jsonb @> $2::text::jsonb ORDER BY {key}",
                    start,
                    line,
                )
                assert plain
                assert await _walked(conn, kind, wanted, filters, mode) == [r[key] for r in plain]


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
    assert await storage.stage("b", keys, [[0.5]] * len(keys), NEW, settings=None) == 0
    assert await storage.stage("a", [("chunk", "ghost")], [[0.5]], NEW, settings=None) == 0
    assert await _vectors(pool, "b") == {}
    assert await _vectors(pool) == {}
    assert await storage.stage("a", keys[:1], [[0.5]], NEW, settings=None) == 1


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
