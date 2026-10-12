import asyncio
import re
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import pytest
from database import Schemas
from unit.fakes import HashEmbedding

from rfnry_memory.errors import ConfigurationError, InputError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Counts, EntityIn, Section, SourceIn, VectorIndex
from rfnry_memory.storage.postgres import PostgresStorage, _vectors_ddl

DEFAULT = VectorIndex(m=16, ef_construction=64, precision="full")


class Wide(HashEmbedding):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [[(i % 5 + 1) / 5 for i in range(self.dimension)] for _ in texts]


class Salted(HashEmbedding):
    model = "hash-8-salted"

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return await super().embed([f"salted {text}" for text in texts], purpose=purpose)


def _table(corpus_id: str) -> str:
    return f"memory_vectors_{sha256(corpus_id.encode('utf-8')).hexdigest()[:24]}"


async def _indexes(pool: Any, corpus_id: str) -> dict[str, tuple[int, bool]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT c.relname, c.oid, i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE i.indrelid = to_regclass(format('%I.%I', current_schema(), $1::text)) AND NOT i.indisprimary",
            _table(corpus_id),
        )
    return {r["relname"]: (r["oid"], r["indisvalid"]) for r in rows}


async def _rows(pool: Any, corpus_id: str) -> list[tuple[str, str]] | None:
    async with pool.acquire() as conn:
        table = await conn.fetchval(
            "SELECT format('%I.%I', current_schema(), $1::text) "
            "WHERE to_regclass(format('%I.%I', current_schema(), $1::text)) IS NOT NULL",
            _table(corpus_id),
        )
        if table is None:
            return None
        rows = await conn.fetch(f"SELECT passage_kind, passage_id FROM {table}")
    return sorted((r["passage_kind"], r["passage_id"]) for r in rows)


async def _embedded(pool: Any, corpus_id: str) -> list[tuple[str, str]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT 'chunk' AS kind, chunk_id AS id FROM memory_chunks WHERE corpus_id = $1 AND embedding IS NOT NULL "
            "UNION ALL SELECT 'story', story_id FROM memory_stories WHERE corpus_id = $1 AND embedding IS NOT NULL",
            corpus_id,
        )
    return sorted((r["kind"], r["id"]) for r in rows)


async def _vectors(pool: Any, corpus_id: str) -> list[tuple[str, str, str]] | None:
    async with pool.acquire() as conn:
        table = await conn.fetchval(
            "SELECT format('%I.%I', current_schema(), $1::text) "
            "WHERE to_regclass(format('%I.%I', current_schema(), $1::text)) IS NOT NULL",
            _table(corpus_id),
        )
        if table is None:
            return None
        rows = await conn.fetch(
            f"SELECT passage_kind, passage_id, vector::text AS vector FROM {table}"
        )
    return sorted((r["passage_kind"], r["passage_id"], r["vector"]) for r in rows)


async def _in_step(pool: Any, corpus_id: str) -> list[tuple[str, str, str]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT 'chunk' AS kind, chunk_id AS id, embedding::text AS vector FROM memory_chunks "
            "WHERE corpus_id = $1 AND embedding IS NOT NULL UNION ALL "
            "SELECT 'story', story_id, embedding::text FROM memory_stories "
            "WHERE corpus_id = $1 AND embedding IS NOT NULL",
            corpus_id,
        )
    passages = sorted((r["kind"], r["id"], r["vector"]) for r in rows)
    assert await _vectors(pool, corpus_id) == passages
    return passages


async def _manuals(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.ingest(
        "manuals",
        name="manual",
        sections=[Section(text="Replace the pump seal yearly."), Section(text="Ground the panel.")],
    )
    await memory.remember("manuals", "Ana owns the seal plan.")
    plain = Memory(storage, embedding=HashEmbedding(), pillars={"text"})
    await plain.ingest("manuals", name="notes", text="Notes kept without a vector.")
    await memory.ingest("other", name="manual", text="Another tenant's manual.")
    return memory


async def test_a_corpus_switches_to_an_index_of_its_own_and_back(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", True) == 3
    assert (await memory.fingerprint("manuals")).vector_index == DEFAULT
    held = await _rows(pool, "manuals")
    assert held == await _embedded(pool, "manuals") and held is not None and len(held) == 3
    hashed = _table("manuals")
    indexes = await _indexes(pool, "manuals")
    assert sorted(re.sub(r"_[0-9a-f]{8}$", "", name) for name in indexes) == [
        f"ix_{hashed}_chunk",
        f"ix_{hashed}_story",
    ]
    assert all(valid for _, valid in indexes.values())
    assert (await memory.fingerprint("other")).vector_index is None
    assert await _rows(pool, "other") is None
    assert await memory.set_vector_index("manuals", False) == 0
    assert (await memory.fingerprint("manuals")).vector_index is None
    assert await _rows(pool, "manuals") is None
    assert await memory.set_vector_index("manuals", False) == 0


async def test_a_corpus_that_holds_nothing_yet_can_be_switched_on(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    assert await memory.set_vector_index("empty", True) == 0
    assert (await memory.fingerprint("empty")).vector_index == DEFAULT
    assert await _rows(pool, "empty") == []


async def test_a_corpus_id_that_is_no_sql_name_is_indexed_like_any_other(
    storage: PostgresStorage, pool: Any
) -> None:
    corpus = 'acme\'s "manuals" ü ' + "x" * 180
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.ingest(corpus, name="manual", text="Replace the pump seal yearly.")
    assert await memory.set_vector_index(corpus, True) == 1
    assert (await memory.fingerprint(corpus)).vector_index == DEFAULT
    assert await _rows(pool, corpus) == await _embedded(pool, corpus)
    assert len(await _indexes(pool, corpus)) == 2
    assert await memory.set_vector_index(corpus, False) == 0
    assert await _rows(pool, corpus) is None


async def test_the_same_values_change_nothing_and_new_ones_replace_the_old_indexes(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", True) == 3
    first = await _indexes(pool, "manuals")
    assert await memory.set_vector_index("manuals", True, m=16, ef_construction=64) == 3
    assert await memory.set_vector_index("manuals", True, precision="full") == 3
    assert await _indexes(pool, "manuals") == first
    assert await memory.set_vector_index("manuals", True, m=32) == 3
    retuned = await _indexes(pool, "manuals")
    assert len(retuned) == 2 and not set(retuned) & set(first)
    assert all(valid for _, valid in retuned.values())
    assert (await memory.fingerprint("manuals")).vector_index == VectorIndex(
        m=32, ef_construction=64, precision="full"
    )
    assert await memory.set_vector_index("manuals", True, m=32, precision="half") == 3
    halved = await _indexes(pool, "manuals")
    assert len(halved) == 2 and not set(halved) & set(retuned)
    assert (await memory.fingerprint("manuals")).vector_index == VectorIndex(
        m=32, ef_construction=64, precision="half"
    )


async def test_a_build_that_never_finished_reports_none_until_the_next_call_replaces_it(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    await memory.set_vector_index("manuals", True)
    built = await _indexes(pool, "manuals")
    chunk = next(name for name in built if "_chunk_" in name)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE pg_index SET indisvalid = false WHERE indexrelid = $1", built[chunk][0]
        )
    assert (await memory.fingerprint("manuals")).vector_index is None
    assert await memory.set_vector_index("manuals", True) == 3
    rebuilt = await _indexes(pool, "manuals")
    assert set(rebuilt) == set(built) and rebuilt[chunk][0] != built[chunk][0]
    assert all(valid for _, valid in rebuilt.values())
    assert (await memory.fingerprint("manuals")).vector_index == DEFAULT


async def test_the_dimension_decides_the_precision_up_to_what_pgvector_indexes(
    schemas: Schemas,
) -> None:
    async def wide(dimension: int) -> Memory:
        pool = await schemas.pool(await schemas.create())
        memory = Memory(PostgresStorage(pool), embedding=Wide(dimension))
        await memory.initialize()
        await memory.ingest("a", name="manual", text="Replace the pump seal yearly.")
        return memory

    full = await wide(2000)
    assert await full.set_vector_index("a", True) == 1
    assert (await full.fingerprint("a")).vector_index == DEFAULT
    half = await wide(2001)
    with pytest.raises(
        InputError,
        match="^"
        + re.escape(
            "postgres: full precision indexes up to 2000 dimensions; this memory embeds 2001 — pass precision='half'"
        )
        + "$",
    ):
        await half.set_vector_index("a", True, precision="full")
    assert (await half.fingerprint("a")).vector_index is None
    assert await half.set_vector_index("a", True) == 1
    assert (await half.fingerprint("a")).vector_index == VectorIndex(
        m=16, ef_construction=64, precision="half"
    )
    beyond = await wide(4001)
    refused = "^postgres: a vector index reaches 4000 dimensions at half precision; this memory embeds 4001$"
    with pytest.raises(ConfigurationError, match=refused):
        await beyond.set_vector_index("a", True)
    with pytest.raises(ConfigurationError, match=refused):
        await beyond.set_vector_index("a", True, precision="half")
    assert (await beyond.fingerprint("a")).vector_index is None


async def test_a_value_pgvector_refuses_is_an_input_error_and_builds_nothing(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    out_of_bounds = (
        "^"
        + re.escape(
            'postgres: value 1 out of bounds for option "m" — Valid values are between "2" and "100".'
        )
        + "$"
    )
    with pytest.raises(InputError, match=out_of_bounds):
        await memory.set_vector_index("manuals", True, m=1)
    with pytest.raises(
        InputError,
        match=r"^postgres: ef_construction must be greater than or equal to 2 \* m$",
    ):
        await memory.set_vector_index("manuals", True, ef_construction=16)
    assert await _rows(pool, "manuals") is None
    assert await memory.set_vector_index("manuals", True) == 3
    built = await _indexes(pool, "manuals")
    with pytest.raises(InputError, match=out_of_bounds):
        await memory.set_vector_index("manuals", True, m=1)
    assert await _indexes(pool, "manuals") == built
    assert (await memory.fingerprint("manuals")).vector_index == DEFAULT


async def test_pgvector_older_than_0_8_0_is_a_configuration_error(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def old(conn: Any) -> str:
        return "0.7.4"

    monkeypatch.setattr("rfnry_memory.storage.postgres._pgvector", old)
    memory = await _manuals(storage)
    with pytest.raises(
        ConfigurationError,
        match=r"^postgres: a vector index needs pgvector 0\.8\.0 or later; this database has 0\.7\.4$",
    ):
        await memory.set_vector_index("manuals", True)
    assert await _rows(pool, "manuals") is None


async def test_set_vector_index_checks_its_arguments_and_needs_an_embedding_model(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    with pytest.raises(InputError, match=r"^memory: on is True or False$"):
        await memory.set_vector_index("manuals", 1)  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: m is an int$"):
        await memory.set_vector_index("manuals", True, m=True)
    with pytest.raises(InputError, match=r"^memory: m is an int$"):
        await memory.set_vector_index("manuals", True, m="16")  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: ef_construction is an int$"):
        await memory.set_vector_index("manuals", True, ef_construction=64.0)  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: precision is 'full', 'half' or None$"):
        await memory.set_vector_index("manuals", True, precision="quarter")  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: corpus_id is required$"):
        await memory.set_vector_index(" ", True)
    unembedded = Memory(storage)
    for on in (True, False):
        with pytest.raises(
            ConfigurationError, match=r"^memory: a vector index needs an embedding model$"
        ):
            await unembedded.set_vector_index("manuals", on)
    with pytest.raises(
        ConfigurationError,
        match=r"^postgres: a vector index needs the embedding column — initialize this storage with a dimension$",
    ):
        await PostgresStorage(pool).set_vector_index(
            "manuals", True, m=16, ef_construction=64, precision=None
        )
    assert await _rows(pool, "manuals") is None


async def test_two_services_switching_one_corpus_run_one_after_the_other(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    built = await asyncio.gather(
        memory.set_vector_index("manuals", True),
        memory.set_vector_index("manuals", True, m=24, ef_construction=96),
    )
    assert list(built) == [3, 3]
    indexes = await _indexes(pool, "manuals")
    assert len(indexes) == 2 and all(valid for _, valid in indexes.values())
    assert (await memory.fingerprint("manuals")).vector_index in (
        DEFAULT,
        VectorIndex(m=24, ef_construction=96, precision="full"),
    )
    switched = await asyncio.gather(
        memory.set_vector_index("manuals", True, m=8, ef_construction=32),
        memory.set_vector_index("manuals", False),
    )
    assert list(switched) == [3, 0]
    indexed = (await memory.fingerprint("manuals")).vector_index
    assert indexed in (None, VectorIndex(m=8, ef_construction=32, precision="full"))
    assert (await _rows(pool, "manuals") is None) == (indexed is None)


async def test_every_write_keeps_an_indexed_corpus_in_step(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    assert await memory.set_vector_index("manuals", True) == 0
    manual = await memory.ingest(
        "manuals",
        name="manual",
        sections=[Section(text="Replace the pump seal yearly."), Section(text="Ground the panel.")],
    )
    assert len(await _in_step(pool, "manuals")) == 2
    plain = Memory(storage, embedding=HashEmbedding(), pillars={"text"})
    await plain.ingest("manuals", name="notes", text="Notes kept without a vector.")
    assert len(await _in_step(pool, "manuals")) == 2
    returns, doc = {"metadata.doc": "returns"}, {"doc": "returns"}
    await memory.sync(
        "manuals",
        filters=returns,
        sources=[
            SourceIn(name="returns", text="Returns within 30 days.", metadata=doc),
            SourceIn(name="refunds", text="Refunds within a week.", metadata=doc),
        ],
    )
    assert len(await _in_step(pool, "manuals")) == 4
    replaced = await memory.sync(
        "manuals",
        filters=returns,
        sources=[SourceIn(name="returns", text="Returns within 60 days.", metadata=doc)],
    )
    assert replaced.deleted.vectors == 2 and len(await _in_step(pool, "manuals")) == 3
    await memory.set_history("manuals", True)
    retiring = await memory.sync(
        "manuals",
        filters=returns,
        sources=[SourceIn(name="returns", text="Returns within 90 days.", metadata=doc)],
    )
    held = await _in_step(pool, "manuals")
    assert retiring.deleted.clean and len(held) == 4
    assert ("chunk", f"{retiring.retired[0]}-0") in [(kind, id) for kind, id, _ in held]
    emptied = await memory.sync("manuals", filters=returns, sources=[])
    assert len(emptied.retired) == 1 and len(await _in_step(pool, "manuals")) == 4
    forgotten = await memory.forget_retired("manuals", before=datetime(2100, 1, 1, tzinfo=UTC))
    assert forgotten.vectors == 2 and len(await _in_step(pool, "manuals")) == 2
    erased = await memory.delete_source("manuals", manual.source.source_id)
    assert erased.vectors == 2 and await _in_step(pool, "manuals") == []
    story = (await memory.remember("manuals", "Ana owns the seal plan.")).story
    assert len(await _in_step(pool, "manuals")) == 1
    again = await memory.remember("manuals", "Ana owns the seal plan.")
    assert again.story == story and len(await _in_step(pool, "manuals")) == 1
    newer = (await memory.supersede("manuals", story.story_id, "Bo owns the seal plan.")).story
    assert [(kind, id) for kind, id, _ in await _in_step(pool, "manuals")] == sorted(
        [("story", story.story_id), ("story", newer.story_id)]
    )
    ended = await memory.remember(
        "manuals",
        "The panel was grounded.",
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_to=datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert len(await _in_step(pool, "manuals")) == 3
    dropped = await memory.delete_story("manuals", ended.story.story_id)
    assert dropped.vectors == 1 and len(await _in_step(pool, "manuals")) == 2
    await memory.ingest("manuals", name="manual", text="Replace the pump seal yearly.")
    before = await _in_step(pool, "manuals")
    assert await Memory(storage, embedding=Salted()).reembed("manuals") == 4
    after = await _in_step(pool, "manuals")
    assert len(before) == 3 and len(after) == 4 and not set(before) & set(after)


async def test_a_passage_written_again_under_its_id_replaces_its_row_in_the_table(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", True) == 3
    [manual] = [s for s in (await memory.list_sources("manuals"))[0] if s.name == "manual"]
    chunks = await storage.source_chunks("manuals", manual.source_id)
    await storage.put_source("manuals", manual, chunks, [[0.5] * 8, [0.25] * 8], "hash-8")
    held = await _in_step(pool, "manuals")
    assert [vector for kind, _, vector in held if kind == "chunk"] == [
        "[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]",
        "[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]",
    ]
    await storage.put_source("manuals", manual, chunks, None, None)
    assert [kind for kind, _, _ in await _in_step(pool, "manuals")] == ["story"]
    [story] = (await memory.list_stories("manuals"))[0]
    await storage.put_story("manuals", story, None, None)
    assert await _in_step(pool, "manuals") == []
    await storage.put_story("manuals", story, [0.5] * 8, "hash-8")
    assert await _in_step(pool, "manuals") == [
        ("story", story.story_id, "[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]")
    ]


async def test_a_write_that_fails_after_reaching_the_table_leaves_it_as_it_was(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", True) == 3
    before = await _in_step(pool, "manuals")
    [manual] = [s for s in (await memory.list_sources("manuals"))[0] if s.name == "manual"]
    [story] = (await memory.list_stories("manuals"))[0]

    async def fail(self: PostgresStorage, conn: Any, corpus_id: str, touched: set[str]) -> int:
        held = await conn.fetchval(f"SELECT count(*) FROM {_table(corpus_id)}")
        raise RuntimeError(f"the table held {held}")

    monkeypatch.setattr(PostgresStorage, "_reconcile", fail)
    valve = SourceIn(name="valve", text="Check the valve.", metadata={"doc": "valve"})
    for write, held in (
        (lambda: memory.ingest("manuals", name="valve", text="Check the valve."), 4),
        (lambda: memory.sync("manuals", filters={"metadata.doc": "valve"}, sources=[valve]), 4),
        (lambda: memory.remember("manuals", "Bo owns the valve."), 4),
        (lambda: memory.supersede("manuals", story.story_id, "Bo owns the seal plan."), 4),
        (lambda: memory.delete_source("manuals", manual.source_id), 1),
        (lambda: memory.delete_story("manuals", story.story_id), 2),
    ):
        with pytest.raises(StorageError, match=f"the table held {held}$"):
            await write()
        assert await _in_step(pool, "manuals") == before


async def test_writes_running_while_a_corpus_builds_all_reach_its_table(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)

    async def write(writer: int) -> None:
        for i in range(8):
            await memory.ingest("manuals", name=f"note {writer}", text=f"Pump {writer}-{i} is due.")
            await memory.remember("manuals", f"Ana checked pump {writer}-{i}.")

    writing = asyncio.gather(*(write(writer) for writer in range(4)))
    built = await memory.set_vector_index("manuals", True)
    await writing
    held = await _in_step(pool, "manuals")
    assert len(held) == 3 + 4 * 8 * 2 and 3 <= built <= len(held)


_WAITERS = (
    "SELECT count(*) FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
    "WHERE a.application_name = current_setting('application_name') "
    "AND l.locktype = 'advisory' AND NOT l.granted "
    "AND ((l.classid::bigint << 32) | l.objid::bigint) = hashtextextended($1, 0)"
)


async def test_a_write_that_waits_out_the_copy_reaches_the_table(schemas: Schemas) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema, application_name=schema)
    memory = Memory(PostgresStorage(pool), embedding=HashEmbedding())
    await memory.initialize()
    writer = Memory(
        PostgresStorage(await schemas.pool(schema, application_name=schema)),
        embedding=HashEmbedding(),
    )
    await writer.initialize()
    await writer.ingest("manuals", name="manual", text="Replace the pump seal yearly.")
    key = "memory.language:manuals"

    async def waiting(admin: Any, count: int) -> None:
        for _ in range(200):
            await admin.execute("SELECT pg_stat_clear_snapshot()")
            if await admin.fetchval(_WAITERS, key) == count:
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f"{count} waiters never queued")

    async with pool.acquire() as admin, admin.transaction():
        await admin.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", key)
        building = asyncio.create_task(memory.set_vector_index("manuals", True))
        await waiting(admin, 1)
        writing = asyncio.create_task(
            writer.ingest("manuals", name="notes", text="Ground the panel.")
        )
        await waiting(admin, 2)
    assert await asyncio.wait_for(building, 10) in (1, 2)
    await asyncio.wait_for(writing, 10)
    assert len(await _in_step(pool, "manuals")) == 2


async def test_deleting_an_indexed_corpus_drops_its_table_and_counts_what_it_held(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    await memory.set_history("manuals", True)
    assert await memory.set_vector_index("manuals", True) == 3
    assert await memory.set_vector_index("other", True) == 1
    live = await memory.verify_absent("manuals")
    assert (live.sources, live.chunks, live.stories, live.vectors, live.corpora) == (2, 3, 1, 3, 2)
    assert await memory.delete_corpus("manuals") == live
    assert await _rows(pool, "manuals") is None
    assert (await memory.verify_absent("manuals")).clean
    assert (await memory.verify_absent("other")).vectors == len(await _in_step(pool, "other")) == 1
    async with pool.acquire() as conn:
        schema = await conn.fetchval("SELECT quote_ident(current_schema())")
        await conn.execute(_vectors_ddl(f"{schema}.{_table('manuals')}", 8))
    assert await memory.verify_absent("manuals") == Counts(corpora=1)
    assert await memory.delete_corpus("manuals") == Counts(corpora=1)
    assert (await memory.verify_absent("manuals")).clean


async def test_deleting_a_corpus_while_it_builds_waits_for_the_build(schemas: Schemas) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema, application_name=schema)
    memory = Memory(PostgresStorage(pool), embedding=HashEmbedding())
    await memory.initialize()
    await memory.ingest("manuals", name="manual", text="Replace the pump seal yearly.")
    async with pool.acquire() as admin, admin.transaction(isolation="repeatable_read"):
        await admin.execute("SELECT 1")
        building = asyncio.create_task(memory.set_vector_index("manuals", True))
        for _ in range(200):
            await admin.execute("SELECT pg_stat_clear_snapshot()")
            if await admin.fetchval(
                "SELECT count(*) FROM pg_stat_progress_create_index p JOIN pg_stat_activity a "
                "ON a.pid = p.pid WHERE a.application_name = current_setting('application_name') "
                "AND p.phase = 'waiting for old snapshots'"
            ):
                break
            await asyncio.sleep(0.05)
        deleting = asyncio.create_task(memory.delete_corpus("manuals"))
        await asyncio.sleep(0.3)
        assert not building.done() and not deleting.done()
    assert await asyncio.wait_for(building, 10) == 1
    assert (await asyncio.wait_for(deleting, 10)).vectors == 1
    assert (await memory.verify_absent("manuals")).clean


async def test_deleting_a_corpus_waits_for_its_writes_in_flight(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.ingest(
        "manuals",
        name="manual",
        text="Valve V-1 feeds pump P-1.",
        entities=[EntityIn(name="P-1", type="pump")],
    )
    assert await memory.set_vector_index("manuals", True) == 1
    async with pool.acquire() as writer, writer.transaction():
        await writer.execute(
            "SELECT pg_advisory_xact_lock_shared(hashtextextended($1, 0))",
            "memory.language:manuals",
        )
        await writer.execute(
            f"INSERT INTO {_table('manuals')} VALUES ('chunk', 'late', $1)", "[1,1,1,1,1,1,1,1]"
        )
        deleting = asyncio.create_task(memory.delete_corpus("manuals"))
        await asyncio.sleep(0.3)
        await writer.fetch("SELECT * FROM memory_entities WHERE corpus_id = 'manuals' FOR UPDATE")
        assert not deleting.done()
    assert (await asyncio.wait_for(deleting, 10)).vectors == 2
    assert (await memory.verify_absent("manuals")).clean


async def test_a_corpus_id_that_is_no_sql_name_is_written_like_any_other(
    storage: PostgresStorage, pool: Any
) -> None:
    corpus = 'acme\'s "manuals" ü ' + "x" * 180
    memory = Memory(storage, embedding=HashEmbedding())
    assert await memory.set_vector_index(corpus, True) == 0
    manual = await memory.ingest(corpus, name="manual", text="Replace the pump seal yearly.")
    await memory.remember(corpus, "Ana owns the seal plan.")
    assert len(await _in_step(pool, corpus)) == 2
    assert (await memory.delete_source(corpus, manual.source.source_id)).vectors == 1
    assert [kind for kind, _, _ in await _in_step(pool, corpus)] == ["story"]
    assert (await memory.verify_absent(corpus)).vectors == 1
    assert (await memory.delete_corpus(corpus)).vectors == 1
    assert (await memory.verify_absent(corpus)).clean and await _rows(pool, corpus) is None


async def test_writes_reach_only_the_table_of_their_own_corpus(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    for corpus in ("a", "b", "c"):
        await memory.ingest(corpus, name="manual", text=f"{corpus} replaces the pump seal yearly.")
    assert await memory.set_vector_index("a", True) == 1
    assert await memory.set_vector_index("c", True) == 1
    a, c = await _in_step(pool, "a"), await _in_step(pool, "c")
    notes = await memory.ingest("b", name="notes", text="Notes on b's pump.")
    story = (await memory.remember("b", "Bo owns b's pump.")).story
    await memory.supersede("b", story.story_id, "Cy owns b's pump.")
    await memory.set_history("b", True)
    for text in ("First returns policy.", "Second returns policy."):
        await memory.sync(
            "b",
            filters={"metadata.doc": "returns"},
            sources=[SourceIn(name="returns", text=text, metadata={"doc": "returns"})],
        )
    await memory.forget_retired("b", before=datetime(2100, 1, 1, tzinfo=UTC))
    await memory.delete_source("b", notes.source.source_id)
    await memory.delete_story("b", story.story_id)
    await Memory(storage, embedding=Salted()).reembed("b")
    await memory.delete_corpus("b")
    assert await _in_step(pool, "a") == a and await _in_step(pool, "c") == c
    await memory.ingest("a", name="notes", text="Notes on a's pump.")
    await memory.remember("a", "Ana owns a's pump.")
    await Memory(storage, embedding=Salted()).reembed("a")
    assert len(await _in_step(pool, "a")) == 3 and await _in_step(pool, "c") == c
    async with pool.acquire() as conn:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() "
            "AND starts_with(tablename, 'memory_vectors_') ORDER BY tablename"
        )
    assert [t["tablename"] for t in tables] == sorted([_table("a"), _table("c")])
