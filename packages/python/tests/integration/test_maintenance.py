from datetime import UTC, datetime
from typing import Any

from database import Schemas, provision, run_cli
from unit.fakes import HashEmbedding
from unit.test_docs_examples import sections

from rfnry_memory.index import HNSW
from rfnry_memory.memory import Memory
from rfnry_memory.models import EntityIn, RelationIn, SourceIn


async def _run(*argv: str) -> int:
    return await run_cli(*argv)


async def _filled(schemas: Schemas) -> tuple[str, Memory]:
    name = await schemas.create()
    storage = await provision(await schemas.pool(name))
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    for n in range(12):
        await memory.ingest(
            "acme",
            name=f"manual {n}",
            text=f"Pump P-{n} seal ERR-{n:03d} replaced on line A. The seal kit SK_{n} fits.",
            metadata={"line": "A", "rev": n},
            entities=[
                EntityIn(name=f"Pump P-{n}", type="machine"),
                EntityIn(name="Line A", type="line"),
            ],
            relations=[RelationIn(from_name=f"Pump P-{n}", to_name="Line A", type="on")],
        )
        await memory.remember("acme", f"Ana prefers slot {n} for ERR-{n:03d}.", subject=f"user:{n}")
    await memory.set_vector_index("acme", HNSW(precision="binary"))
    assert (
        await _run("index", "--schema", name, "--key", "metadata.line", "--type", "text", "--yes")
        == 0
    )
    assert (
        await _run("index", "--schema", name, "--key", "metadata.rev", "--type", "number", "--yes")
        == 0
    )
    assert await _run("index", "--schema", name, "--current", "--yes") == 0
    return name, memory


async def _answers(memory: Memory) -> list[list[str]]:
    return [
        [h.id for h in (await memory.search("acme", query)).hits]
        for query in ("seal ERR-007", "Ana slot", "Pump P-3 line A")
    ]


async def _invalid(pool: Any, name: str) -> int:
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                "SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND NOT i.indisvalid",
                name,
            )
        )


async def test_a_schema_is_rebuilt_and_maintained_without_itself_on_the_search_path(
    schemas: Schemas,
) -> None:
    name, memory = await _filled(schemas)
    before = await _answers(memory)
    bare = await schemas.pool(public=False)
    async with bare.acquire() as conn:
        assert await conn.fetchval("SHOW search_path") == ""
        for table in ("memory_chunks", "memory_stories", "memory_entities"):
            await conn.execute(f"UPDATE {name}.{table} SET config = config")
        await conn.execute(f"UPDATE {name}.memory_sources SET text = text")
        await conn.execute(f"REINDEX SCHEMA {name}")
        await conn.execute(
            f"VACUUM (ANALYZE) {name}.memory_sources, {name}.memory_chunks, "
            f"{name}.memory_stories, {name}.memory_entities"
        )
    assert await _invalid(bare, name) == 0
    assert await _answers(memory) == before


STORED = {
    "chunks": "SELECT chunk_id, fts::text, fts_length FROM memory_chunks ORDER BY 1",
    "stories": "SELECT story_id, fts::text, fts_length FROM memory_stories ORDER BY 1",
    "entities": "SELECT entity_id, spotted::text FROM memory_entities ORDER BY 1",
    "counts": "SELECT kind, lexeme, passages FROM memory_term_counts ORDER BY 1, 2",
    "totals": "SELECT kind, passages, length FROM memory_term_totals ORDER BY 1",
}


async def _stored(conn: Any) -> dict[str, list[tuple[Any, ...]]]:
    return {name: [tuple(r) for r in await conn.fetch(sql)] for name, sql in STORED.items()}


async def test_the_documented_recompute_brings_every_stored_value_up_to_date(
    schemas: Schemas,
) -> None:
    [recompute] = [
        block
        for blocks in sections().values()
        for language, block in blocks
        if language == "sql" and "memory_recount" in block
    ]
    name = await schemas.create()
    pool = await schemas.pool(name)
    memory = Memory(await provision(pool), embedding=HashEmbedding(), pillars={"text", "graph"})
    async with pool.acquire() as conn:
        await conn.execute("CREATE TEXT SEARCH CONFIGURATION probe (COPY = pg_catalog.english)")
    await memory.set_language("acme", "probe")
    await memory.set_statistics("acme", True)
    for n in range(6):
        await memory.ingest(
            "acme",
            name=f"manual {n}",
            text=f"The pumps were replaced on lines {n} and {n + 1}.",
            entities=[EntityIn(name=f"Running Pumps {n}", type="machine")],
        )
        await memory.remember("acme", f"Ana keeps the pumps running on line {n}.")
    async with pool.acquire() as conn:
        await conn.execute(
            "ALTER TEXT SEARCH CONFIGURATION probe ALTER MAPPING FOR asciiword WITH simple"
        )
        stale = await _stored(conn)
        fresh_run = conn.transaction()
        await fresh_run.start()
        for table in ("memory_chunks", "memory_stories", "memory_entities"):
            await conn.execute(f"UPDATE {table} SET config = config")
        await conn.execute("SELECT memory_recount('acme')")
        fresh = await _stored(conn)
        await fresh_run.rollback()
        assert all(stale[part] != fresh[part] for part in STORED)
        assert (await memory.search("acme", "were")).hits == []
        await conn.execute(recompute)
        assert await _stored(conn) == fresh
    assert len((await memory.search("acme", "were")).hits) == 6


async def _staged(pool: Any, corpus: str) -> list[tuple[str, str]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT passage_kind, passage_id FROM memory_staged_vectors WHERE corpus_id = $1 "
            "ORDER BY 1, 2",
            corpus,
        )
    return [(r["passage_kind"], r["passage_id"]) for r in rows]


async def _stage_all(pool: Any, corpus: str) -> list[tuple[str, str]]:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_staged_vectors (corpus_id, model, passage_kind, passage_id, vector) "
            "SELECT corpus_id, 'next-model', 'chunk', chunk_id, '[1,2,3,4,5,6,7,8]' "
            "FROM memory_chunks WHERE corpus_id = $1",
            corpus,
        )
        await conn.execute(
            "INSERT INTO memory_staged_vectors (corpus_id, model, passage_kind, passage_id, vector) "
            "SELECT corpus_id, 'next-model', 'story', story_id, '[1,2,3,4,5,6,7,8]' "
            "FROM memory_stories WHERE corpus_id = $1",
            corpus,
        )
    return await _staged(pool, corpus)


async def _memory(schemas: Schemas) -> tuple[Any, Memory]:
    pool = await schemas.pool(await schemas.create())
    storage = await provision(pool)
    return pool, Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector"})


async def test_deleting_a_source_deletes_its_staged_vectors(schemas: Schemas) -> None:
    pool, memory = await _memory(schemas)
    first = await memory.ingest("a", name="one", text="The seal is good.")
    second = await memory.ingest("a", name="two", text="The valve is good.")
    await memory.ingest("b", name="one", text="The seal is good.")
    held = await _stage_all(pool, "a")
    assert len(held) == 2
    await _stage_all(pool, "b")
    deleted = await memory.delete_source("a", first.source.source_id)
    assert deleted.staged == 1
    assert await _staged(pool, "a") == [p for p in held if p[1].startswith(second.source.source_id)]
    assert len(await _staged(pool, "b")) == 1


async def test_forgetting_retired_sources_deletes_their_staged_vectors(schemas: Schemas) -> None:
    pool, memory = await _memory(schemas)
    await memory.set_history("a", True)
    old = await memory.ingest("a", name="doc", text="The seal is good.")
    new = await memory.ingest(
        "a", name="doc", text="The seal is worn.", replaces=old.source.source_id
    )
    held = await _stage_all(pool, "a")
    assert len(held) == 2
    forgotten = await memory.forget_retired("a", before=datetime(2100, 1, 1, tzinfo=UTC))
    assert forgotten.staged == 1
    assert [p[1] for p in await _staged(pool, "a")] == [
        p[1] for p in held if p[1].startswith(new.source.source_id)
    ]


async def test_a_sync_that_deletes_deletes_the_staged_vectors_of_what_it_removed(
    schemas: Schemas,
) -> None:
    pool, memory = await _memory(schemas)
    filters = {"metadata.set": "x"}
    await memory.sync(
        "a",
        filters=filters,
        sources=[
            SourceIn(name="one", text="The seal is good.", metadata={"set": "x"}),
            SourceIn(name="two", text="The valve is good.", metadata={"set": "x"}),
        ],
    )
    assert len(await _stage_all(pool, "a")) == 2
    synced = await memory.sync(
        "a",
        filters=filters,
        sources=[SourceIn(name="one", text="The seal is good.", metadata={"set": "x"})],
    )
    assert synced.deleted.staged == 1
    assert len(await _staged(pool, "a")) == 1
    await memory.sync("a", filters=filters, sources=[])
    assert await _staged(pool, "a") == []


async def test_deleting_a_story_deletes_its_staged_vectors(schemas: Schemas) -> None:
    pool, memory = await _memory(schemas)
    kept = await memory.remember("a", "The seal is good.")
    gone = await memory.remember("a", "The valve is good.")
    await _stage_all(pool, "a")
    deleted = await memory.delete_story("a", gone.story.story_id)
    assert deleted.staged == 1
    assert await _staged(pool, "a") == [("story", kept.story.story_id)]


async def test_a_schema_without_the_staging_table_deletes_as_it_always_did(
    schemas: Schemas,
) -> None:
    pool = await schemas.pool(await schemas.create())
    memory = Memory(await provision(pool, dimension=None), pillars={"text"})
    ingested = await memory.ingest("a", name="one", text="The seal is good.")
    remembered = await memory.remember("a", "The valve is good.")
    assert (await memory.delete_source("a", ingested.source.source_id)).staged == 0
    assert (await memory.delete_story("a", remembered.story.story_id)).staged == 0
    assert (await memory.delete_corpus("a")).staged == 0
    assert (await memory.verify_absent("a")).clean
