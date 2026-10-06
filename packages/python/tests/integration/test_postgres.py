import asyncio
import os
from datetime import UTC, datetime
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
        fingerprint = await memory.fingerprint("a")
        assert (fingerprint.embedding, fingerprint.embedded, fingerprint.stale) == (None, {}, 0)
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
            await restarted.sync(
                "a",
                filters={"metadata.file": "a"},
                sources=[SourceIn(name="v1", text="version one", metadata={"file": "a"})],
            )
        assert await restarted.list_sources("a") == ([kept], None)
        held = await first.fingerprint("a")
        assert (held.embedded, held.stale) == ({"hash-8": 2}, 0)
        assert await restarted.fingerprint("a") == held
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


FOLDED = {
    "spanish": [
        "La información de las válvulas está en el manual; la informacion sin tildes también.",
        "Cambiar los sellos cada año. Las bombas funcionan mejor; ¿qué más? Sí, PEQUEÑAS ACCIONES.",
        "Canción, acción, niños, pingüino, el 3 de mayo a las 12:30, válvula V-101.",
    ],
    "portuguese": [
        "A informação das válvulas está no manual; a informacao sem acentos também.",
        "Trocar as vedações todo ano. As bombas funcionam melhor; não, mãe, pão, CORAÇÕES.",
        "Ações à vista, você está aí? Às 12:30 a válvula V-101 abriu.",
    ],
    "simple": [
        "The seals were replaced; la información está aquí, a informação também.",
        "Café naïve ÉCOLE niños ações the of and 3.14 -5",
        "Straße Æble Øl Łódź Œuvre ﬁnal Þing ðe đak",
    ],
}


@pytest.mark.parametrize("language", sorted(FOLDED))
async def test_the_in_memory_analyser_reads_text_as_the_postgres_configuration_does(
    language: str,
) -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(url)
    try:
        await PostgresStorage(pool).initialize(dimension=8)
        async with pool.acquire() as conn:
            for text in FOLDED[language]:
                postgres = await conn.fetchval(
                    "SELECT tsvector_to_array(to_tsvector($1::regconfig, $2))",
                    f"memory_{language}",
                    text,
                )
                assert set(lexemes(text, language)) == set(postgres), text  # type: ignore[arg-type]
    finally:
        await pool.close()


async def test_no_column_has_a_reg_type_so_the_database_can_be_upgraded() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(url)
    try:
        await PostgresStorage(pool).initialize(dimension=8)
        async with pool.acquire() as conn:
            typed = await conn.fetch(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name LIKE 'memory%' AND data_type LIKE 'reg%'"
            )
        assert typed == []
    finally:
        await pool.close()


async def test_a_write_that_waited_for_a_language_change_lands_under_it_at_repeatable_read() -> (
    None
):
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(
        url, server_settings={"default_transaction_isolation": "repeatable read"}
    )
    holder = await asyncpg.connect(url)
    corpus = f"rr-{uuid4().hex[:8]}"
    storage = PostgresStorage(pool)
    try:
        await storage.initialize(dimension=8)
        memory = Memory(storage)
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", f"memory.language:{corpus}"
            )
            await holder.execute(
                "INSERT INTO memory_corpora (corpus_id, language) VALUES ($1, 'spanish')", corpus
            )
            writing = asyncio.create_task(
                memory.ingest(corpus, name="m", text="La información de las válvulas.")
            )
            remembering = asyncio.create_task(memory.remember(corpus, "Las bombas funcionan."))
            await asyncio.sleep(0.5)
            assert not writing.done() and not remembering.done()
        await asyncio.wait_for(asyncio.gather(writing, remembering), 10)
        async with pool.acquire() as conn:
            languages = await conn.fetch(
                "SELECT language FROM memory_chunks WHERE corpus_id = $1 "
                "UNION ALL SELECT language FROM memory_stories WHERE corpus_id = $1",
                corpus,
            )
        assert [r["language"] for r in languages] == ["spanish", "spanish"]
        assert [
            h.kind for h in (await memory.search(corpus, "informacion", pillars={"text"})).hits
        ] == ["chunk"]
    finally:
        await storage.delete_corpus(corpus)
        await holder.close()
        await pool.close()


_OLD_LAYOUT = [
    "ALTER TABLE memory_stories DROP COLUMN superseded_by, DROP COLUMN closed_at, DROP COLUMN written_to",
    "ALTER TABLE memory_stories ALTER COLUMN supersedes DROP DEFAULT",
    "ALTER TABLE memory_stories ALTER COLUMN supersedes DROP NOT NULL",
    "ALTER TABLE memory_stories ALTER COLUMN supersedes TYPE TEXT USING supersedes->>0",
    "CREATE UNIQUE INDEX ux_memory_stories_open_text ON memory_stories (corpus_id, text_hash) WHERE valid_to IS NULL",
]
_UPGRADE_WAITERS = (
    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted "
    "AND ((classid::bigint << 32) | objid::bigint) = hashtextextended('memory.upgrade', 0)"
)
_LEGACY = (
    "INSERT INTO memory_stories (corpus_id, story_id, text, text_hash, subject, valid_from, valid_to, "
    "created_at, supersedes, source_ids, metadata) VALUES ($1, $2, $3, md5($3), NULL, $4, $5, $6, $7, '[]', '{}')"
)


async def test_initialize_upgrades_an_old_database_once_even_when_two_start_together() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    schema = f"upgrade_{uuid4().hex[:12]}"
    admin = await asyncpg.connect(url)
    await admin.execute(f"CREATE SCHEMA {schema}")
    pool = await asyncpg.create_pool(url, server_settings={"search_path": f"{schema}, public"})
    jan1, jan5, jan10 = (datetime(2026, 1, d, tzinfo=UTC) for d in (1, 5, 10))
    feb1, feb2 = datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 2, 2, tzinfo=UTC)
    try:
        await PostgresStorage(pool).initialize(dimension=8)
        async with pool.acquire() as conn:
            for statement in _OLD_LAYOUT:
                await conn.execute(statement)
            await conn.execute(
                _LEGACY, "a", "s1", "The go-live is February 20.", jan10, feb1, jan10, None
            )
            await conn.execute(
                _LEGACY, "a", "s2", "The go-live moved to March 3.", feb1, None, feb2, "s1"
            )
            await conn.execute(_LEGACY, "a", "s3", "Ana prefers Tuesday.", jan1, None, jan1, None)
            await conn.execute(_LEGACY, "a", "s4", "The fair ran.", jan1, jan5, jan1, None)
            await conn.execute(_LEGACY, "a", "s5", "The plant opened.", jan1, None, jan1, "gone")
            await conn.execute(_LEGACY, "a", "s6", "The old rate applied.", jan1, feb1, jan1, None)
        racers = [PostgresStorage(pool), PostgresStorage(pool)]
        async with admin.transaction():
            await admin.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "memory.upgrade"
            )
            starting = [asyncio.create_task(s.initialize(dimension=8)) for s in racers]
            waiting = 0
            for _ in range(200):
                waiting = await admin.fetchval(_UPGRADE_WAITERS)
                if waiting == 2:
                    break
                await asyncio.sleep(0.05)
            assert waiting == 2 and not any(task.done() for task in starting)
        await asyncio.wait_for(asyncio.gather(*starting), 10)
        await racers[0].initialize(dimension=8)
        s1, s2, s3, s4, s5, s6 = await racers[0].get_stories(
            "a", ["s1", "s2", "s3", "s4", "s5", "s6"]
        )
        assert (s1.valid_to, s1.superseded_by, s1.closed_at, s1.supersedes) == (
            feb1,
            "s2",
            feb2,
            (),
        )
        assert (s2.supersedes, s2.superseded_by, s2.closed_at) == (("s1",), None, None)
        assert (s3.supersedes, s3.superseded_by, s3.valid_to) == ((), None, None)
        assert (s4.valid_to, s4.superseded_by) == (jan5, None)
        assert (s5.supersedes, s5.superseded_by) == (("gone",), None)
        assert (s6.valid_to, s6.superseded_by, s6.closed_at) == (feb1, None, None)
        async with pool.acquire() as conn:
            written = dict(
                await conn.fetch(
                    "SELECT story_id, written_to FROM memory_stories ORDER BY story_id"
                )
            )
        assert written == {
            "s1": None,
            "s2": None,
            "s3": None,
            "s4": jan5,
            "s5": None,
            "s6": feb1,
        }
        async with pool.acquire() as conn:
            assert not await conn.fetchval(
                "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 AND indexname = 'ux_memory_stories_open_text'",
                schema,
            )
    finally:
        await pool.close()
        await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        await admin.close()


async def test_the_upgrade_drops_the_old_index_only_in_its_own_schema() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    first, second = f"first_{uuid4().hex[:12]}", f"second_{uuid4().hex[:12]}"
    admin = await asyncpg.connect(url)
    await admin.execute(f"CREATE SCHEMA {first}")
    await admin.execute(f"CREATE SCHEMA {second}")
    await admin.execute(f"CREATE TABLE {second}.other (a TEXT)")
    await admin.execute(f"CREATE UNIQUE INDEX ux_memory_stories_open_text ON {second}.other (a)")
    pool = await asyncpg.create_pool(
        url, server_settings={"search_path": f"{first}, {second}, public"}
    )
    try:
        await PostgresStorage(pool).initialize(dimension=8)
        assert await admin.fetchval(
            "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 AND indexname = 'ux_memory_stories_open_text'",
            second,
        )
    finally:
        await pool.close()
        await admin.execute(f"DROP SCHEMA {first} CASCADE")
        await admin.execute(f"DROP SCHEMA {second} CASCADE")
        await admin.close()


_OLD_SOURCES = [
    "ALTER TABLE memory_corpora DROP COLUMN history",
    "ALTER TABLE memory_sources DROP COLUMN retired_at, DROP COLUMN replaced_by",
    "ALTER TABLE memory_chunks DROP COLUMN retired_at",
]


async def test_initialize_adds_the_history_columns_and_reads_old_sources_as_current() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    schema = f"history_{uuid4().hex[:12]}"
    admin = await asyncpg.connect(url)
    await admin.execute(f"CREATE SCHEMA {schema}")
    pool = await asyncpg.create_pool(url, server_settings={"search_path": f"{schema}, public"})
    try:
        await PostgresStorage(pool).initialize(dimension=8)
        async with pool.acquire() as conn:
            for statement in _OLD_SOURCES:
                await conn.execute(statement)
            await conn.execute(
                "INSERT INTO memory_sources (corpus_id, source_id, name, metadata, text, created_at, chunks) "
                "VALUES ('a', 's1', 'manual', '{}', 'Ground first.', $1, 0)",
                datetime(2026, 1, 1, tzinfo=UTC),
            )
            await conn.execute(
                "INSERT INTO memory_corpora (corpus_id, language) VALUES ('a', 'english')"
            )
        racers = [PostgresStorage(pool), PostgresStorage(pool)]
        await asyncio.gather(*(s.initialize(dimension=8) for s in racers))
        source = await racers[0].get_source("a", "s1")
        assert source is not None and (source.retired_at, source.replaced_by) == (None, None)
        async with pool.acquire() as conn:
            columns = {
                (r["table_name"], r["column_name"])
                for r in await conn.fetch(
                    "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = $1",
                    schema,
                )
            }
            history = await conn.fetchval(
                "SELECT history FROM memory_corpora WHERE corpus_id = 'a'"
            )
        assert {
            ("memory_corpora", "history"),
            ("memory_sources", "retired_at"),
            ("memory_sources", "replaced_by"),
            ("memory_chunks", "retired_at"),
        } <= columns
        assert history is False
    finally:
        await pool.close()
        await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        await admin.close()


async def test_set_history_waits_for_a_running_sync() -> None:
    url = os.environ.get("RFNRY_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RFNRY_TEST_POSTGRES_URL is not set")
    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(url)
    holder = await asyncpg.connect(url)
    corpus = f"hist-{uuid4().hex[:8]}"
    storage = PostgresStorage(pool)
    try:
        await storage.initialize(dimension=8)
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", f"memory.sync:{corpus}"
            )
            switching = asyncio.create_task(storage.set_history(corpus, True))
            await asyncio.sleep(0.5)
            assert not switching.done() and await storage.get_history(corpus) is False
        await asyncio.wait_for(switching, 10)
        assert await storage.get_history(corpus) is True
    finally:
        await storage.delete_corpus(corpus)
        await holder.close()
        await pool.close()
