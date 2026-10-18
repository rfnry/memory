import asyncio
import json
import re
from collections.abc import Callable
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import asyncpg
import pytest
from database import Schemas, provision
from pydantic import TypeAdapter, ValidationError
from unit.fakes import HashEmbedding

from rfnry_memory.errors import ConfigurationError, InputError, StorageError
from rfnry_memory.index import HNSW, IVFFlat, VectorIndex, hnsw
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    CorpusFingerprint,
    Counts,
    EntityIn,
    Kind,
    Leg,
    Section,
    SourceIn,
    Window,
)
from rfnry_memory.postgres import PostgresStorage, corpus as postgres_corpus
from rfnry_memory.postgres.search import _nearest_sql, _vector_sql
from rfnry_memory.postgres.vectors import _vectors_ddl
from rfnry_memory.tools import MemoryTools

DEFAULT = HNSW(precision="full")


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


async def _pumps(memory: Memory, corpus_id: str, count: int) -> list[str]:
    pumps = await memory.ingest(
        corpus_id,
        name="pumps",
        sections=[Section(text=f"Pump {i} needs a new seal.") for i in range(count)],
    )
    chunks = await memory.storage.source_chunks(corpus_id, pumps.source.source_id)
    return [c.chunk_id for c in chunks]


async def _nearest(
    storage: PostgresStorage,
    corpus_id: str,
    top_k: int,
    *,
    kinds: tuple[Kind, ...] = ("chunk",),
    window: Window | None = None,
    subject: str | None = None,
    filters: dict[str, Any] | None = None,
    query: str = "pump seal",
) -> list[Leg]:
    [vector] = await HashEmbedding().embed([query], purpose="query")
    return await storage.search_vector(
        corpus_id,
        vector,
        kinds=kinds,
        top_k=top_k,
        filters=filters,
        window=window or Window(),
        subject=subject,
    )


async def _scans(pool: Any, corpus_id: str) -> dict[str, int]:
    async with AsyncExitStack() as stack:
        held = [await stack.enter_async_context(pool.acquire()) for _ in range(pool.get_size())]
        for conn in held:
            await conn.execute("SELECT pg_stat_force_next_flush()")
        rows = await held[0].fetch(
            "SELECT indexrelname, idx_scan FROM pg_stat_user_indexes WHERE schemaname = current_schema() "
            "AND relname = $1 AND starts_with(indexrelname, 'ix_')",
            _table(corpus_id),
        )
    return {r["indexrelname"].split("_")[4]: r["idx_scan"] for r in rows}


def _queries(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Kind]]:
    ran: list[tuple[str, Kind]] = []

    def nearest(table: str, half: int | None, kind: Kind, scope: str, limit: str) -> str:
        ran.append(("nearest" if half is None else "half", kind))
        return _nearest_sql(table, half, kind, scope, limit)

    def exact(kind: Kind, scope: str, limit: str) -> str:
        ran.append(("exact", kind))
        return _vector_sql(kind, scope, limit)

    monkeypatch.setattr("rfnry_memory.postgres.search._nearest_sql", nearest)
    monkeypatch.setattr("rfnry_memory.postgres.search._vector_sql", exact)
    return ran


def _transactions(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    opened: list[bool] = []
    transaction = asyncpg.connection.Connection.transaction

    def spy(self: Any, **options: Any) -> Any:
        opened.append(bool(options.get("readonly")))
        return transaction(self, **options)

    monkeypatch.setattr(asyncpg.connection.Connection, "transaction", spy)
    return opened


async def test_a_corpus_switches_to_an_index_of_its_own_and_back(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", HNSW()) == 3
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
    assert await memory.set_vector_index("manuals", None) == 0
    assert (await memory.fingerprint("manuals")).vector_index is None
    assert await _rows(pool, "manuals") is None
    assert await memory.set_vector_index("manuals", None) == 0


async def test_a_corpus_that_holds_nothing_yet_can_be_switched_on(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    assert await memory.set_vector_index("empty", HNSW()) == 0
    assert (await memory.fingerprint("empty")).vector_index == DEFAULT
    assert await _rows(pool, "empty") == []


async def test_a_corpus_id_that_is_no_sql_name_is_indexed_like_any_other(
    storage: PostgresStorage, pool: Any
) -> None:
    corpus = 'acme\'s "manuals" ü ' + "x" * 180
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.ingest(corpus, name="manual", text="Replace the pump seal yearly.")
    assert await memory.set_vector_index(corpus, HNSW()) == 1
    assert (await memory.fingerprint(corpus)).vector_index == DEFAULT
    assert await _rows(pool, corpus) == await _embedded(pool, corpus)
    assert len(await _indexes(pool, corpus)) == 2
    assert await memory.set_vector_index(corpus, None) == 0
    assert await _rows(pool, corpus) is None


async def test_the_same_values_change_nothing_and_new_ones_replace_the_old_indexes(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    first = await _indexes(pool, "manuals")
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    assert await memory.set_vector_index("manuals", HNSW(precision="full")) == 3
    assert await _indexes(pool, "manuals") == first
    assert await memory.set_vector_index("manuals", HNSW(m=32)) == 3
    retuned = await _indexes(pool, "manuals")
    assert len(retuned) == 2 and not set(retuned) & set(first)
    assert all(valid for _, valid in retuned.values())
    assert (await memory.fingerprint("manuals")).vector_index == HNSW(m=32, precision="full")
    assert await memory.set_vector_index("manuals", HNSW(m=32, precision="half")) == 3
    halved = await _indexes(pool, "manuals")
    assert len(halved) == 2 and not set(halved) & set(retuned)
    assert (await memory.fingerprint("manuals")).vector_index == HNSW(m=32, precision="half")


async def test_a_re_tune_waiting_to_drop_the_old_index_leaves_searches_running(
    schemas: Schemas,
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema, application_name=schema)
    await provision(pool)
    storage = PostgresStorage(pool)
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await _pumps(memory, "manuals", 50)
    assert await memory.set_vector_index("manuals", HNSW()) == 50
    async with pool.acquire() as holder, holder.transaction():
        await holder.execute(f"SELECT count(*) FROM {_table('manuals')}")
        retuning = asyncio.create_task(memory.set_vector_index("manuals", HNSW(m=24)))
        for _ in range(200):
            if await pool.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name = "
                "current_setting('application_name') AND starts_with(query, 'DROP INDEX') "
                "AND wait_event_type = 'Lock'"
            ):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("the re-tune never waited to drop the old index")
        assert len(await asyncio.wait_for(_nearest(storage, "manuals", 10), 5)) == 10
        assert not retuning.done()
    assert await asyncio.wait_for(retuning, 10) == 50
    assert (await memory.fingerprint("manuals")).vector_index == HNSW(m=24, precision="full")


async def test_a_build_that_never_finished_reports_none_until_the_next_call_replaces_it(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _manuals(storage)
    chunks = await _nearest(storage, "manuals", 10)
    both = await _nearest(storage, "manuals", 10, kinds=("chunk", "story"))
    await memory.set_vector_index("manuals", HNSW())
    built = await _indexes(pool, "manuals")
    chunk = next(name for name in built if "_chunk_" in name)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE pg_index SET indisvalid = false WHERE indexrelid = $1", built[chunk][0]
        )
    assert (await memory.fingerprint("manuals")).vector_index is None
    ran, opened = _queries(monkeypatch), _transactions(monkeypatch)
    assert await _nearest(storage, "manuals", 10) == chunks
    assert ran == [("exact", "chunk")] and opened == []
    assert await _nearest(storage, "manuals", 10, kinds=("chunk", "story")) == both
    assert ran[1:] == [("exact", "chunk"), ("nearest", "story"), ("exact", "story")]
    assert opened == [True]
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    rebuilt = await _indexes(pool, "manuals")
    assert set(rebuilt) == set(built) and rebuilt[chunk][0] != built[chunk][0]
    assert all(valid for _, valid in rebuilt.values())
    assert (await memory.fingerprint("manuals")).vector_index == DEFAULT


async def test_the_dimension_decides_the_precision_up_to_what_pgvector_indexes(
    schemas: Schemas,
) -> None:
    async def wide(dimension: int) -> Memory:
        pool = await schemas.pool(await schemas.create())
        await provision(pool, dimension=dimension)
        memory = Memory(PostgresStorage(pool), embedding=Wide(dimension))
        await memory.check()
        await memory.ingest("a", name="manual", text="Replace the pump seal yearly.")
        return memory

    full = await wide(2000)
    assert await full.set_vector_index("a", HNSW()) == 1
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
        await half.set_vector_index("a", HNSW(precision="full"))
    assert (await half.fingerprint("a")).vector_index is None
    assert await half.set_vector_index("a", HNSW()) == 1
    assert (await half.fingerprint("a")).vector_index == HNSW(precision="half")
    beyond = await wide(4001)
    refused = "^postgres: a vector index reaches 4000 dimensions at half precision; this memory embeds 4001$"
    with pytest.raises(ConfigurationError, match=refused):
        await beyond.set_vector_index("a", HNSW())
    with pytest.raises(ConfigurationError, match=refused):
        await beyond.set_vector_index("a", HNSW(precision="half"))
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
        await memory.set_vector_index("manuals", HNSW(m=1))
    with pytest.raises(
        InputError,
        match=r"^postgres: ef_construction must be greater than or equal to 2 \* m$",
    ):
        await memory.set_vector_index("manuals", HNSW(ef_construction=16))
    assert await _rows(pool, "manuals") is None
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    built = await _indexes(pool, "manuals")
    with pytest.raises(InputError, match=out_of_bounds):
        await memory.set_vector_index("manuals", HNSW(m=1))
    assert await _indexes(pool, "manuals") == built
    assert (await memory.fingerprint("manuals")).vector_index == DEFAULT


async def test_pgvector_older_than_0_8_0_is_a_configuration_error(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def old(conn: Any) -> str:
        return "0.7.4"

    monkeypatch.setattr("rfnry_memory.postgres.vectors._pgvector", old)
    memory = await _manuals(storage)
    with pytest.raises(
        ConfigurationError,
        match=r"^postgres: a vector index needs pgvector 0\.8\.0 or later; this database has 0\.7\.4$",
    ):
        await memory.set_vector_index("manuals", HNSW())
    assert await _rows(pool, "manuals") is None


async def test_set_vector_index_checks_its_arguments_and_needs_an_embedding_model(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    named = r"^memory: index is HNSW\(\.\.\.\), IVFFlat\(\.\.\.\) or None$"
    with pytest.raises(InputError, match=named):
        await memory.set_vector_index("manuals", True)  # type: ignore[arg-type]
    with pytest.raises(InputError, match=named):
        await memory.set_vector_index("manuals", {"method": "hnsw"})  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: rebuild is True or False$"):
        await memory.set_vector_index("manuals", HNSW(), rebuild=1)  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^memory: rebuild needs an index method$"):
        await memory.set_vector_index("manuals", None, rebuild=True)
    for wrong in ({"m": True}, {"m": "16"}, {"ef_construction": 64.0}, {"precision": "quarter"}):
        with pytest.raises(ValidationError):
            HNSW.model_validate(wrong)
    for other in ({"lists": "100"}, {"m": 16}, {"method": "hnsw"}):
        with pytest.raises(ValidationError):
            IVFFlat.model_validate(other)
    with pytest.raises(InputError, match=r"^memory: corpus_id is required$"):
        await memory.set_vector_index(" ", HNSW())
    unembedded = Memory(storage)
    for index in (HNSW(), IVFFlat(), None):
        with pytest.raises(
            ConfigurationError, match=r"^memory: a vector index needs an embedding model$"
        ):
            await unembedded.set_vector_index("manuals", index)
    with pytest.raises(
        ConfigurationError,
        match=r"^postgres: a vector index needs the embedding column — run rfnry-memory set --dimension, then check\(\) again$",
    ):
        await PostgresStorage(pool).set_vector_index("manuals", HNSW(), rebuild=False)
    assert await _rows(pool, "manuals") is None


async def test_two_services_switching_one_corpus_run_one_after_the_other(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    built = await asyncio.gather(
        memory.set_vector_index("manuals", HNSW()),
        memory.set_vector_index("manuals", HNSW(m=24, ef_construction=96)),
    )
    assert list(built) == [3, 3]
    indexes = await _indexes(pool, "manuals")
    assert len(indexes) == 2 and all(valid for _, valid in indexes.values())
    assert (await memory.fingerprint("manuals")).vector_index in (
        DEFAULT,
        HNSW(m=24, ef_construction=96, precision="full"),
    )
    switched = await asyncio.gather(
        memory.set_vector_index("manuals", HNSW(m=8, ef_construction=32)),
        memory.set_vector_index("manuals", None),
    )
    assert list(switched) == [3, 0]
    indexed = (await memory.fingerprint("manuals")).vector_index
    assert indexed in (None, HNSW(m=8, ef_construction=32, precision="full"))
    assert (await _rows(pool, "manuals") is None) == (indexed is None)


async def test_every_write_keeps_an_indexed_corpus_in_step(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    assert await memory.set_vector_index("manuals", HNSW()) == 0
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
    assert await memory.set_vector_index("manuals", HNSW()) == 3
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
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    before = await _in_step(pool, "manuals")
    [manual] = [s for s in (await memory.list_sources("manuals"))[0] if s.name == "manual"]
    [story] = (await memory.list_stories("manuals"))[0]

    async def fail(conn: Any, corpus_id: str, touched: set[str]) -> int:
        held = await conn.fetchval(f"SELECT count(*) FROM {_table(corpus_id)}")
        raise RuntimeError(f"the table held {held}")

    monkeypatch.setattr("rfnry_memory.postgres.sources._reconcile", fail)
    monkeypatch.setattr("rfnry_memory.postgres.stories._reconcile", fail)
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
    built = await memory.set_vector_index("manuals", HNSW())
    await writing
    held = await _in_step(pool, "manuals")
    assert len(held) == 3 + 4 * 8 * 2 and 3 <= built <= len(held)


_WAITERS = (
    "SELECT count(*) FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
    "WHERE a.application_name = current_setting('application_name') "
    "AND l.locktype = 'advisory' AND NOT l.granted "
    "AND ((l.classid::bigint << 32) | l.objid::bigint) = hashtextextended($1, 0)"
)


async def _waiting(admin: Any, key: str, count: int) -> None:
    for _ in range(200):
        await admin.execute("SELECT pg_stat_clear_snapshot()")
        if await admin.fetchval(_WAITERS, key) == count:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"{count} waiters never queued")


async def test_a_write_that_waits_out_the_copy_reaches_the_table(schemas: Schemas) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema, application_name=schema)
    await provision(pool)
    memory = Memory(PostgresStorage(pool), embedding=HashEmbedding())
    await memory.check()
    writer = Memory(
        PostgresStorage(await schemas.pool(schema, application_name=schema)),
        embedding=HashEmbedding(),
    )
    await writer.check()
    await writer.ingest("manuals", name="manual", text="Replace the pump seal yearly.")
    key = "memory.language:manuals"
    async with pool.acquire() as admin, admin.transaction():
        await admin.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", key)
        building = asyncio.create_task(memory.set_vector_index("manuals", HNSW()))
        await _waiting(admin, key, 1)
        writing = asyncio.create_task(
            writer.ingest("manuals", name="notes", text="Ground the panel.")
        )
        await _waiting(admin, key, 2)
    assert await asyncio.wait_for(building, 10) in (1, 2)
    await asyncio.wait_for(writing, 10)
    assert len(await _in_step(pool, "manuals")) == 2


async def test_a_re_embedding_that_waits_out_the_copy_reaches_the_table_at_repeatable_read(
    schemas: Schemas,
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema, application_name=schema)
    await provision(pool)
    memory = Memory(PostgresStorage(pool), embedding=HashEmbedding())
    await memory.check()
    await memory.ingest("manuals", name="manual", text="Replace the pump seal yearly.")
    repeatable = await schemas.pool(
        schema, application_name=schema, default_transaction_isolation="repeatable read"
    )
    salted = Memory(PostgresStorage(repeatable), embedding=Salted())
    await salted.check()
    key = "memory.language:manuals"
    async with pool.acquire() as admin, admin.transaction():
        await admin.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", key)
        building = asyncio.create_task(memory.set_vector_index("manuals", HNSW()))
        await _waiting(admin, key, 1)
        reembedding = asyncio.create_task(salted.reembed("manuals"))
        await _waiting(admin, key, 2)
    assert await asyncio.wait_for(building, 10) == 1
    assert await asyncio.wait_for(reembedding, 10) == 1
    assert (await salted.fingerprint("manuals")).stale == 0
    assert len(await _in_step(pool, "manuals")) == 1


async def test_deleting_a_corpus_at_repeatable_read_deletes_what_a_write_it_waited_for_wrote(
    schemas: Schemas,
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema, application_name=schema)
    await provision(pool)
    memory = Memory(PostgresStorage(pool), embedding=HashEmbedding())
    await memory.check()
    await memory.ingest("manuals", name="manual", text="Replace the pump seal yearly.")
    assert await memory.set_vector_index("manuals", HNSW()) == 1
    repeatable = await schemas.pool(
        schema, application_name=schema, default_transaction_isolation="repeatable read"
    )
    deleter = Memory(PostgresStorage(repeatable), embedding=HashEmbedding())
    await deleter.check()
    key = "memory.language:manuals"
    async with pool.acquire() as admin, admin.transaction():
        await admin.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", key)
        writing = asyncio.create_task(
            memory.ingest("manuals", name="notes", text="Ground the panel.")
        )
        await _waiting(admin, key, 1)
        deleting = asyncio.create_task(deleter.delete_corpus("manuals"))
        await _waiting(admin, key, 2)
    await asyncio.wait_for(writing, 10)
    deleted = await asyncio.wait_for(deleting, 10)
    assert (deleted.sources, deleted.chunks, deleted.vectors) == (2, 2, 2)
    assert (await memory.verify_absent("manuals")).clean


async def test_deleting_an_indexed_corpus_drops_its_table_and_counts_what_it_held(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    await memory.set_history("manuals", True)
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    assert await memory.set_vector_index("other", HNSW()) == 1
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
    await provision(pool)
    memory = Memory(PostgresStorage(pool), embedding=HashEmbedding())
    await memory.check()
    await memory.ingest("manuals", name="manual", text="Replace the pump seal yearly.")
    async with pool.acquire() as admin, admin.transaction(isolation="repeatable_read"):
        await admin.execute("SELECT 1")
        building = asyncio.create_task(memory.set_vector_index("manuals", HNSW()))
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
    assert await memory.set_vector_index("manuals", HNSW()) == 1
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
    assert await memory.set_vector_index(corpus, HNSW()) == 0
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
    assert await memory.set_vector_index("a", HNSW()) == 1
    assert await memory.set_vector_index("c", HNSW()) == 1
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


async def test_a_search_of_an_indexed_corpus_reads_its_index(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    pumps = await _pumps(memory, "manuals", 50)
    await memory.remember("manuals", "Ana owns the seal plan.")
    assert await memory.set_vector_index("manuals", HNSW()) == 51
    ran = _queries(monkeypatch)
    before = await _scans(pool, "manuals")
    hits = await memory.search(
        "manuals", "pump seal", kinds=("chunk",), pillars={"vector"}, top_k=3
    )
    after = await _scans(pool, "manuals")
    assert len(hits.hits) == 3 and {h.id for h in hits.hits} <= set(pumps)
    assert ran == [("nearest", "chunk")]
    assert after["chunk"] > before["chunk"] and after["story"] == before["story"]


async def test_a_search_of_an_unindexed_corpus_returns_what_it_returned_before(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _manuals(storage)
    before = await memory.search("manuals", "pump seal", pillars={"vector"})
    assert await memory.set_vector_index("other", HNSW()) == 1
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    assert await memory.set_vector_index("manuals", None) == 0
    ran = _queries(monkeypatch)
    after = await memory.search("manuals", "pump seal", pillars={"vector"})
    assert len(before.hits) == 3 and after.hits == before.hits
    assert ran == [("exact", "chunk"), ("exact", "story")]
    assert await _scans(pool, "other") == {"chunk": 0, "story": 0}


async def test_a_filter_that_matches_fewer_than_top_k_passages_returns_them_all(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.ingest(
        "manuals",
        name="pumps",
        sections=[Section(text=f"Pump {i} needs a new seal.") for i in range(50)],
        metadata={"doc": "pumps"},
    )
    valves = await memory.ingest(
        "manuals",
        name="valves",
        sections=[Section(text=f"Valve {i} needs a new seal.") for i in range(3)],
        metadata={"doc": "valves"},
    )
    assert await memory.set_vector_index("manuals", HNSW()) == 53
    ran = _queries(monkeypatch)
    hits = await memory.search(
        "manuals",
        "pump seal",
        kinds=("chunk",),
        pillars={"vector"},
        filters={"metadata.doc": "valves"},
    )
    chunks = await storage.source_chunks("manuals", valves.source.source_id)
    assert sorted(h.id for h in hits.hits) == sorted(c.chunk_id for c in chunks)
    assert ran == [("nearest", "chunk"), ("exact", "chunk")]


async def test_a_scoped_search_of_an_indexed_corpus_reads_every_passage_in_its_scope(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    manual = await memory.ingest(
        "manuals",
        name="manual",
        sections=[Section(text="Replace the pump seal yearly."), Section(text="Ground the panel.")],
    )
    await memory.ingest("manuals", name="notes", text="The pump seal leaked in March.")
    await memory.remember(
        "manuals",
        "Apollo's pump seal was ordered.",
        entities=[EntityIn(name="Apollo", type="project")],
        source_ids=(manual.source.source_id,),
    )
    await memory.remember(
        "manuals",
        "Borealis needs a pump seal.",
        entities=[EntityIn(name="Borealis", type="project")],
    )

    async def scoped() -> list[list[str]]:
        cited = await memory.search(
            "manuals", "pump seal", pillars={"vector"}, sources=[manual.source.source_id]
        )
        about = await memory.search(
            "manuals", "pump seal", pillars={"vector"}, around=["Apollo"], hops=0
        )
        return [[h.id for h in cited.hits], [h.id for h in about.hits]]

    before = await scoped()
    assert [len(ids) for ids in before] == [3, 1]
    assert await memory.set_vector_index("manuals", HNSW()) == 5
    ran = _queries(monkeypatch)
    scans = await _scans(pool, "manuals")
    assert await scoped() == before
    assert ran == [("exact", "chunk"), ("exact", "story")] * 2
    assert await _scans(pool, "manuals") == scans


async def test_a_search_for_more_than_ef_search_passages_returns_them_all(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    pumps = await _pumps(memory, "manuals", 300)
    assert await memory.set_vector_index("manuals", HNSW()) == 300
    ran = _queries(monkeypatch)
    before = await _scans(pool, "manuals")
    found = await _nearest(storage, "manuals", 60)
    assert len(found) == 60 and {passage_id for _, passage_id, _ in found} <= set(pumps)
    expanded = await memory.expand(
        "manuals", ("chunk", pumps[0]), by="vector", top_k=60, kinds=("chunk",)
    )
    assert len(expanded.hits) == 60 and pumps[0] not in {h.id for h in expanded.hits}
    assert ran == [("nearest", "chunk")] * 2
    assert (await _scans(pool, "manuals"))["chunk"] >= before["chunk"] + 2


async def test_a_search_racing_the_switch_off_returns_the_exact_result(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    searching = PostgresStorage._search_kind
    dropped: list[Kind] = []

    async def racing(
        self: PostgresStorage,
        conn: Any,
        kind: Kind,
        sql: Callable[[Kind, str, str], str],
        *rest: Any,
    ) -> list[Any]:
        if sql is not _vector_sql and not dropped:
            dropped.append(kind)
            assert await memory.set_vector_index("manuals", None) == 0
        return await searching(self, conn, kind, sql, *rest)

    monkeypatch.setattr(PostgresStorage, "_search_kind", racing)
    raced = await _nearest(storage, "manuals", 10, kinds=("chunk", "story"))
    assert dropped == ["chunk"] and await _rows(pool, "manuals") is None
    assert len(raced) == 3
    assert raced == await _nearest(storage, "manuals", 10, kinds=("chunk", "story"))


async def test_an_indexed_corpus_with_history_finds_a_retired_chunk_through_its_table(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 1, 10, tzinfo=UTC))
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.set_history("manuals", True)
    await memory.ingest("manuals", name="manual", text="Replace the pump seal yearly.")
    returns, doc = {"metadata.doc": "returns"}, {"doc": "returns"}
    await memory.sync(
        "manuals",
        filters=returns,
        sources=[SourceIn(name="returns", text="Returns within 30 days.", metadata=doc)],
    )
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 2, 10, tzinfo=UTC))
    synced = await memory.sync(
        "manuals",
        filters=returns,
        sources=[SourceIn(name="returns", text="Returns within 60 days.", metadata=doc)],
    )
    retired = f"{synced.retired[0]}-0"
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    ran = _queries(monkeypatch)
    told = datetime(2026, 1, 20, tzinfo=UTC)

    async def found(window: Window, top_k: int) -> set[str]:
        legs = await _nearest(storage, "manuals", top_k, window=window, query="returns")
        return {passage_id for _, passage_id, _ in legs}

    assert retired in await found(Window(as_of=told, known_at=told), 2)
    assert retired in await found(Window(), 3)
    assert retired not in await found(Window(as_of=datetime(2026, 3, 1, tzinfo=UTC)), 2)
    assert ran == [("nearest", "chunk")] * 3
    known = await memory.search("manuals", "returns", pillars={"vector"}, known_at=told)
    every = await memory.search("manuals", "returns", pillars={"vector"}, history=True)
    assert retired in {h.id for h in known.hits} & {h.id for h in every.hits}


async def test_a_corpus_switched_on_empty_finds_what_is_written_next_through_its_index(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    assert await memory.set_vector_index("manuals", HNSW()) == 0
    pumps = await _pumps(memory, "manuals", 50)
    ran = _queries(monkeypatch)

    async def found() -> set[str]:
        hits = await memory.search(
            "manuals", "pump seal", kinds=("chunk",), pillars={"vector"}, top_k=3
        )
        return {h.id for h in hits.hits}

    assert len(await found()) == 3 and await found() <= set(pumps)
    async with pool.acquire() as conn:
        await conn.execute(f"ANALYZE {_table('manuals')}")
    before = await _scans(pool, "manuals")
    assert len(await found()) == 3 and await found() <= set(pumps)
    assert ran == [("nearest", "chunk")] * 4
    assert (await _scans(pool, "manuals"))["chunk"] >= before["chunk"] + 2


async def test_a_corpus_id_that_is_no_sql_name_is_searched_through_its_index(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = 'acme\'s "manuals" ü ' + "x" * 180
    memory = Memory(storage, embedding=HashEmbedding())
    pumps = await _pumps(memory, corpus, 50)
    assert await memory.set_vector_index(corpus, HNSW()) == 50
    ran = _queries(monkeypatch)
    before = await _scans(pool, corpus)
    found = await _nearest(storage, corpus, 10)
    assert len(found) == 10 and {passage_id for _, passage_id, _ in found} <= set(pumps)
    assert ran == [("nearest", "chunk")]
    assert (await _scans(pool, corpus))["chunk"] > before["chunk"]


async def test_a_half_precision_index_is_searched_and_scores_at_full_precision(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await _pumps(memory, "manuals", 50)
    exact = {passage_id: score for _, passage_id, score in await _nearest(storage, "manuals", 50)}
    assert len(exact) == 50
    assert await memory.set_vector_index("manuals", HNSW(precision="half")) == 50
    ran = _queries(monkeypatch)
    before = await _scans(pool, "manuals")
    halved = {passage_id: score for _, passage_id, score in await _nearest(storage, "manuals", 10)}
    assert len(halved) == 10 and halved.items() <= exact.items()
    assert ran == [("half", "chunk")]
    assert (await _scans(pool, "manuals"))["chunk"] > before["chunk"]


async def test_a_half_re_tune_that_never_finished_leaves_searches_on_the_full_index(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await _pumps(memory, "manuals", 50)
    assert await memory.set_vector_index("manuals", HNSW()) == 50
    half = HNSW(precision="half")
    tuned = sha256(json.dumps(half.model_dump(), sort_keys=True).encode()).hexdigest()[:8]
    unfinished = f"ix_{_table('manuals')}_chunk_{tuned}"
    async with pool.acquire() as conn:
        await conn.execute(
            f"CREATE INDEX {unfinished} ON {_table('manuals')} {hnsw.clause('chunk', half, 8)}"
        )
        await conn.execute(
            "UPDATE pg_index SET indisvalid = false WHERE indexrelid = $1::text::regclass",
            unfinished,
        )
    assert (await memory.fingerprint("manuals")).vector_index == DEFAULT
    ran = _queries(monkeypatch)
    assert len(await _nearest(storage, "manuals", 10)) == 10
    assert ran == [("nearest", "chunk")]
    assert await memory.set_vector_index("manuals", HNSW(precision="half")) == 50
    assert (await memory.fingerprint("manuals")).vector_index == half
    assert len(await _nearest(storage, "manuals", 10)) == 10
    assert ran[1:] == [("half", "chunk")]


async def test_an_indexed_corpus_applies_every_window_subject_and_filter_as_before(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    jan, feb, now = (datetime(2026, month, 1, tzinfo=UTC) for month in (1, 2, 4))
    ops = {"team": "ops"}
    launch = await memory.remember(
        "acme", "Apollo goes live in March.", subject="project:apollo", valid_from=jan, metadata=ops
    )
    await memory.supersede(
        "acme", launch.story.story_id, "Apollo goes live in April.", valid_from=feb
    )
    await memory.remember(
        "acme", "Borealis is on hold.", subject="project:borealis", valid_from=jan, metadata=ops
    )
    await memory.remember("acme", "The panel was grounded.", valid_from=jan, valid_to=feb)
    await memory.remember("acme", "Ana owns the seal plan.", valid_from=feb)
    cases: list[tuple[Window, str | None, dict[str, Any] | None]] = [
        (Window(as_of=now), None, None),
        (Window(as_of=datetime(2026, 1, 15, tzinfo=UTC)), None, None),
        (Window(), None, None),
        (Window(as_of=now, open=True), None, None),
        (Window(as_of=now, since=feb), None, None),
        (Window(), "project:apollo", None),
        (Window(as_of=now), None, {"metadata.team": "ops"}),
    ]

    async def found(
        window: Window, subject: str | None, filters: dict[str, Any] | None, top_k: int
    ) -> set[str]:
        legs = await _nearest(
            storage,
            "acme",
            top_k,
            kinds=("story",),
            window=window,
            subject=subject,
            filters=filters,
            query="Apollo",
        )
        return {passage_id for _, passage_id, _ in legs}

    exact = [await found(*case, 100) for case in cases]
    assert [len(held) for held in exact] == [3, 3, 5, 3, 2, 2, 2]
    assert await memory.set_vector_index("acme", HNSW()) == 5
    ran = _queries(monkeypatch)
    assert [await found(*case, len(held)) for case, held in zip(cases, exact, strict=True)] == exact
    assert ran == [("nearest", "story")] * len(cases)


def test_the_scope_descriptions_say_a_scoped_search_reads_every_passage(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    definitions = MemoryTools(memory, corpus_id="acme").definitions()
    for definition in definitions:
        if definition["name"] in ("search", "recall"):
            properties = definition["input_schema"]["properties"]
            for argument in ("sources", "around"):
                assert (
                    "a scoped search reads every passage in its scope"
                    in properties[argument]["description"]
                )
    described = json.dumps(definitions).lower()
    assert not any(word in described for word in ("index", "hnsw", "approximate"))


async def test_a_corpus_switches_methods_building_the_new_index_before_dropping_the_old(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    first = await _indexes(pool, "manuals")
    assert await memory.set_vector_index("manuals", IVFFlat(lists=2)) == 3
    switched = await _indexes(pool, "manuals")
    assert len(switched) == 2 and not set(switched) & set(first)
    assert all(valid for _, valid in switched.values())
    async with pool.acquire() as conn:
        methods = await conn.fetch(
            "SELECT DISTINCT a.amname FROM pg_class c JOIN pg_am a ON a.oid = c.relam "
            "WHERE c.relname = ANY($1::text[])",
            list(switched),
        )
    assert [r["amname"] for r in methods] == ["ivfflat"]
    assert (await memory.fingerprint("manuals")).vector_index == IVFFlat(lists=2, precision="full")
    assert len(await _nearest(storage, "manuals", 2)) == 2
    assert await memory.set_vector_index("manuals", None) == 0
    assert await _rows(pool, "manuals") is None


async def test_rebuild_builds_the_same_settings_again_and_without_it_nothing_is_built(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", IVFFlat(lists=2)) == 3
    built = await _indexes(pool, "manuals")
    assert await memory.set_vector_index("manuals", IVFFlat(lists=2)) == 3
    assert await _indexes(pool, "manuals") == built
    assert await memory.set_vector_index("manuals", IVFFlat(lists=2), rebuild=True) == 3
    rebuilt = await _indexes(pool, "manuals")
    assert set(rebuilt) == set(built)
    assert all(rebuilt[name][0] != built[name][0] and rebuilt[name][1] for name in built)
    assert (await memory.fingerprint("manuals")).vector_index == IVFFlat(lists=2, precision="full")


async def test_a_setting_saved_as_json_builds_the_index_it_names(
    storage: PostgresStorage,
) -> None:
    memory = await _manuals(storage)
    flag = '{"method": "ivfflat", "lists": 2}'
    assert (
        await memory.set_vector_index("manuals", TypeAdapter(VectorIndex).validate_json(flag)) == 3
    )
    fingerprint = await memory.fingerprint("manuals")
    assert fingerprint.vector_index == IVFFlat(lists=2, precision="full")
    assert CorpusFingerprint.model_validate_json(fingerprint.model_dump_json()) == fingerprint
    with pytest.raises(ValidationError):
        TypeAdapter(VectorIndex).validate_json('{"method": "diskann"}')


async def test_pgvector_judges_the_options_of_either_method(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _manuals(storage)
    with pytest.raises(InputError, match=r"^postgres: .*lists"):
        await memory.set_vector_index("manuals", IVFFlat(lists=0))
    with pytest.raises(InputError, match=r"^postgres: .*\bm\b"):
        await memory.set_vector_index("manuals", HNSW(m=1))
    assert await _rows(pool, "manuals") is None


async def test_a_filtered_ivfflat_search_reads_on_until_it_has_as_many_as_asked(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    for doc in ("pumps", "valves"):
        await memory.ingest(
            "manuals",
            name=doc,
            sections=[Section(text=f"{doc} {i} needs a new seal.") for i in range(100)],
            metadata={"doc": doc},
        )
    assert await memory.set_vector_index("manuals", IVFFlat(lists=20)) == 200
    ran = _queries(monkeypatch)
    found = await _nearest(storage, "manuals", 10, filters={"metadata.doc": "valves"})
    assert len(found) == 10
    assert ran == [("nearest", "chunk")]


async def test_a_re_tune_whose_second_kind_fails_keeps_the_old_settings_whole(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", HNSW()) == 3
    first = await _indexes(pool, "manuals")
    clause = hnsw.clause

    def broken(kind: str, settings: HNSW, dimension: int) -> str:
        if kind == "story" and settings.m == 24:
            return "USING hnsw (no_such_column vector_cosine_ops)"
        return clause(kind, settings, dimension)

    monkeypatch.setattr(hnsw, "clause", broken)
    with pytest.raises(StorageError):
        await memory.set_vector_index("manuals", HNSW(m=24))
    assert (await memory.fingerprint("manuals")).vector_index == DEFAULT
    left = await _indexes(pool, "manuals")
    assert all(left[name][1] for name in first)
    assert len(await _nearest(storage, "manuals", 2, kinds=("chunk", "story"))) == 2
    monkeypatch.setattr(hnsw, "clause", clause)
    assert await memory.set_vector_index("manuals", HNSW(m=24)) == 3
    assert (await memory.fingerprint("manuals")).vector_index == HNSW(m=24, precision="full")
    assert len(await _indexes(pool, "manuals")) == 2


async def test_counting_a_corpus_whose_table_is_dropped_meanwhile_counts_it_gone(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _manuals(storage)
    assert await memory.set_vector_index("manuals", HNSW()) == 3

    async def dropped(conn: Any, corpus_id: str) -> str:
        return "memory_vectors_dropped_meanwhile"

    monkeypatch.setattr(postgres_corpus, "_vectors_table", dropped)
    counts = await memory.verify_absent("manuals")
    assert counts.vectors == 0 and counts.corpora == 1 and counts.sources == 2
