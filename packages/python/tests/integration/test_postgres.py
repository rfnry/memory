import os
from uuid import uuid4

import pytest
from unit.fakes import HashEmbedding

from rfnry_memory.errors import ConfigurationError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn, Window
from rfnry_memory.storage.inmemory import lexemes
from rfnry_memory.storage.postgres import PostgresStorage

pytestmark = pytest.mark.postgres


async def test_memory_without_an_embedding_model_writes_and_reads_on_postgres() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    schema = f"nodim_{uuid4().hex[:12]}"
    admin = await asyncpg.connect(url)
    await admin.execute(f"CREATE SCHEMA {schema}")
    pool = await asyncpg.create_pool(url, server_settings={"search_path": f"{schema}, public"})
    try:
        memory = Memory(PostgresStorage(pool))
        await memory.initialize()
        async with pool.acquire() as conn:
            columns = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = $1 AND table_name = 'memory_chunks'",
                schema,
            )
        assert "embedding" not in {r["column_name"] for r in columns}
        ingested = await memory.ingest(
            "a", name="manual", text="The pump seal on P-3 must be replaced every year."
        )
        assert ingested.source.chunks == 1
        story = (await memory.remember("a", "The seal on pump P-3 was replaced in March.")).story
        assert (
            await memory.remember("a", "the seal on pump  P-3 was replaced in march.")
        ).story == story
        hits = await memory.search("a", "pump seal")
        assert set(hits.pillars) == {"text"} and hits.pillars["text"].error is None
        with pytest.raises(
            StorageError,
            match=r"^postgres: vectors were given but this storage has no vector dimension — call initialize\(\) first$",
        ):
            await memory.storage.search_vector(
                "a",
                [0.0] * 8,
                kinds=("chunk",),
                top_k=1,
                filters=None,
                window=Window(),
                subject=None,
            )
        assert {h.kind for h in hits.hits} == {"chunk", "story"}
        assert (await memory.search("b", "pump seal")).hits == []
    finally:
        await pool.close()
        await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        await admin.close()


async def test_initialize_refuses_tables_made_for_another_dimension() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    admin = await asyncpg.connect(url)
    schemas = [f"dims_{uuid4().hex[:12]}" for _ in range(2)]
    pools = []
    try:
        for schema in schemas:
            await admin.execute(f"CREATE SCHEMA {schema}")
            pools.append(
                await asyncpg.create_pool(url, server_settings={"search_path": f"{schema}, public"})
            )
        eight, bare = pools
        await PostgresStorage(eight).initialize(dimension=8)
        unused = await admin.fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname = $1 AND (indexdef ILIKE '%hnsw%' OR indexdef ILIKE '%trgm%')",
            schemas[0],
        )
        assert unused == []
        await PostgresStorage(eight).initialize(dimension=8)
        await PostgresStorage(eight).initialize(dimension=None)
        with pytest.raises(
            ConfigurationError,
            match=r"^postgres: tables were created for dimension 8; this memory embeds 1024 — re-provision or use another schema$",
        ):
            await PostgresStorage(eight).initialize(dimension=1024)
        await PostgresStorage(bare).initialize(dimension=None)
        with pytest.raises(
            ConfigurationError,
            match=r"^postgres: tables were created without a vector column; this memory embeds 1024 — re-provision or use another schema$",
        ):
            await PostgresStorage(bare).initialize(dimension=1024)
    finally:
        for pool in pools:
            await pool.close()
        for schema in schemas:
            await admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await admin.close()


async def test_initialize_checks_the_tables_it_would_create_not_others_on_the_search_path() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    admin = await asyncpg.connect(url)
    below, above = f"below_{uuid4().hex[:12]}", f"above_{uuid4().hex[:12]}"
    pools = []
    try:
        for schema in (below, above):
            await admin.execute(f"CREATE SCHEMA {schema}")
        pools.append(
            await asyncpg.create_pool(url, server_settings={"search_path": f"{below}, public"})
        )
        await PostgresStorage(pools[0]).initialize(dimension=8)
        pools.append(
            await asyncpg.create_pool(
                url, server_settings={"search_path": f"{above}, {below}, public"}
            )
        )
        await PostgresStorage(pools[1]).initialize(dimension=16)
        created = await admin.fetch(
            "SELECT table_schema FROM information_schema.tables WHERE table_name = 'memory_stories' AND table_schema = ANY($1::text[])",
            [below, above],
        )
        assert sorted(r["table_schema"] for r in created) == sorted([below, above])
    finally:
        for pool in pools:
            await pool.close()
        for schema in (below, above):
            await admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await admin.close()


async def test_a_storage_that_was_not_initialized_refuses_vectors_rather_than_drop_them() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    schema = f"uninit_{uuid4().hex[:12]}"
    admin = await asyncpg.connect(url)
    await admin.execute(f"CREATE SCHEMA {schema}")
    pool = await asyncpg.create_pool(url, server_settings={"search_path": f"{schema}, public"})
    try:
        first = Memory(PostgresStorage(pool), embedding=HashEmbedding())
        await first.initialize()
        story = (await first.remember("a", "Ana owns the pump.")).story
        kept = (
            await first.ingest("a", name="v0", text="version zero", metadata={"file": "a"})
        ).source
        restarted = Memory(PostgresStorage(pool), embedding=HashEmbedding())
        first_write = r"^postgres: vectors were given but this storage has no vector dimension — call initialize\(\) first$"
        with pytest.raises(StorageError, match=first_write):
            await restarted.ingest("a", name="manual", text="pump seal yearly")
        with pytest.raises(StorageError, match=first_write):
            await restarted.remember("a", "Bo owns the pump.")
        with pytest.raises(StorageError, match=first_write):
            await restarted.supersede("a", story.story_id, "Cy owns the pump.")
        with pytest.raises(StorageError, match=first_write):
            await restarted.replace(
                "a",
                filters={"metadata.file": "a"},
                sources=[SourceIn(name="v1", text="version one", metadata={"file": "a"})],
            )
        assert await restarted.list_sources("a") == ([kept], None)
        hits = await restarted.search("a", "pump", pillars={"vector"})
        assert hits.pillars["vector"].error == "StorageError: " + first_write[1:-1].replace(
            "\\", ""
        )
        renamed = HashEmbedding()
        renamed.model = "hash-8-v2"
        with pytest.raises(StorageError, match=first_write):
            await Memory(PostgresStorage(pool), embedding=renamed).reembed("a")
        residue = await restarted.verify_absent("a")
        assert (residue.sources, residue.stories) == (1, 1)
        plain = PostgresStorage(pool)
        await plain.initialize(dimension=None)
        await Memory(plain, embedding=HashEmbedding()).ingest(
            "a", name="m", text="pump seal yearly"
        )
        hits = await first.search("a", "pump seal yearly", pillars={"vector"})
        assert hits.pillars["vector"].error is None and hits.hits[0].kind == "chunk"
        assert await first.reembed("a") == 0
    finally:
        await pool.close()
        await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        await admin.close()


ORDINARY = [
    "The seals were replaced; the pump is running again.",
    "How often are the seals replaced?",
    "Replacing, replaced, replaces, replacement: organizations organize universally.",
    "What is it that they are doing there?",
    "the of and to in",
    "Pump P-3 and valve V-101 moved 3.14 metres in 2026, at -5 degrees, +7 on 1,000 runs.",
    "ABC123 mp3s 5th 1st 007 12:30",
    "MiXeD CaSe WORDS Running RAN runs",
    "Punctuation!!! (brackets) [square] {curly}; colons: dots... \"x\" 'y' don't don’t O'Brien",
    "Café naïve Éclair ÉCOLE straße STRASSE Ünïcödé İstanbul ΣΟΦΟΣ",
    "x² ½ and underscores_between words",
    "Pump manual > Seals > Replacing the P-3 seal Replace the seal every twelve months.",
    "?!",
    "",
]


async def test_the_in_memory_analyser_reads_text_as_postgres_english_does() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    conn = await asyncpg.connect(url)
    try:
        for text in ORDINARY:
            postgres = await conn.fetchval(
                "SELECT tsvector_to_array(to_tsvector('english', $1))", text
            )
            assert set(lexemes(text)) == set(postgres), text
    finally:
        await conn.close()
