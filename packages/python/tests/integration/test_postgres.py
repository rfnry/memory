import asyncio
import os
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest
from database import Schemas
from unit.fakes import HashEmbedding

from rfnry_memory.errors import ConfigurationError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn, Window
from rfnry_memory.storage.postgres import PostgresStorage


async def test_memory_without_an_embedding_model_writes_and_reads_on_postgres(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 3, 1, tzinfo=UTC))
    schema = await schemas.create()
    pool = await schemas.pool(schema)
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


async def test_initialize_refuses_tables_made_for_another_dimension(schemas: Schemas) -> None:
    first, second = await schemas.create(), await schemas.create()
    eight, bare = await schemas.pool(first), await schemas.pool(second)
    await PostgresStorage(eight).initialize(dimension=8)
    async with eight.acquire() as conn:
        unused = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname = $1 AND (indexdef ILIKE '%hnsw%' OR indexdef ILIKE '%trgm%')",
            first,
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


async def test_initialize_checks_the_tables_it_would_create_not_others_on_the_search_path(
    schemas: Schemas,
) -> None:
    below, above = await schemas.create(), await schemas.create()
    lower = await schemas.pool(below)
    await PostgresStorage(lower).initialize(dimension=8)
    await PostgresStorage(await schemas.pool(above, below)).initialize(dimension=16)
    async with lower.acquire() as conn:
        created = await conn.fetch(
            "SELECT table_schema FROM information_schema.tables WHERE table_name = 'memory_stories' AND table_schema = ANY($1::text[])",
            [below, above],
        )
    assert sorted(r["table_schema"] for r in created) == sorted([below, above])


async def test_a_storage_that_was_not_initialized_refuses_vectors_rather_than_drop_them(
    pool: Any,
) -> None:
    first = Memory(PostgresStorage(pool), embedding=HashEmbedding())
    await first.initialize()
    story = (await first.remember("a", "Ana owns the pump.")).story
    kept = (await first.ingest("a", name="v0", text="version zero", metadata={"file": "a"})).source
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
    assert hits.pillars["vector"].error == "StorageError: " + first_write[1:-1].replace("\\", "")
    renamed = HashEmbedding()
    renamed.model = "hash-8-v2"
    with pytest.raises(StorageError, match=first_write):
        await Memory(PostgresStorage(pool), embedding=renamed).reembed("a")
    residue = await restarted.verify_absent("a")
    assert (residue.sources, residue.stories) == (1, 1)
    plain = PostgresStorage(pool)
    await plain.initialize(dimension=None)
    await Memory(plain, embedding=HashEmbedding()).ingest("a", name="m", text="pump seal yearly")
    hits = await first.search("a", "pump seal yearly", pillars={"vector"})
    assert hits.pillars["vector"].error is None and hits.hits[0].kind == "chunk"
    assert await first.reembed("a") == 0


async def test_no_column_has_a_reg_type_so_the_database_can_be_upgraded(pool: Any) -> None:
    async with pool.acquire() as conn:
        typed = await conn.fetch(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name LIKE 'memory%' AND data_type LIKE 'reg%'"
        )
    assert typed == []


async def test_a_write_that_waited_for_a_language_change_lands_under_it_at_repeatable_read(
    pool: Any, schemas: Schemas
) -> None:
    repeatable = await schemas.pool(schemas.run, default_transaction_isolation="repeatable read")
    storage = PostgresStorage(repeatable)
    await storage.initialize(dimension=8)
    await storage.set_language("es", "spanish")
    memory = Memory(storage)
    async with pool.acquire() as holder, holder.transaction():
        await holder.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "memory.language:rr"
        )
        await holder.execute(
            "INSERT INTO memory_corpora (corpus_id, language, config) "
            "SELECT 'rr', language, config FROM memory_corpora WHERE corpus_id = 'es'"
        )
        writing = asyncio.create_task(
            memory.ingest("rr", name="m", text="La información de las válvulas.")
        )
        remembering = asyncio.create_task(memory.remember("rr", "Las bombas funcionan."))
        await asyncio.sleep(0.5)
        assert not writing.done() and not remembering.done()
    await asyncio.wait_for(asyncio.gather(writing, remembering), 10)
    async with pool.acquire() as conn:
        configs = await conn.fetch(
            "SELECT config FROM memory_chunks WHERE corpus_id = 'rr' "
            "UNION ALL SELECT config FROM memory_stories WHERE corpus_id = 'rr'"
        )
    assert [r["config"] for r in configs] == [f"{schemas.run}.memory_spanish"] * 2
    assert [h.kind for h in (await memory.search("rr", "funcionando", pillars={"text"})).hits] == [
        "story"
    ]


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
    "created_at, supersedes, source_ids, metadata, config) VALUES ($1, $2, $3, md5($3), NULL, $4, $5, $6, $7, "
    "'[]', '{}', format('%I.%I', current_schema(), 'memory_english'))"
)


async def test_initialize_upgrades_an_old_database_once_even_when_two_start_together(
    schemas: Schemas,
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    jan1, jan5, jan10 = (datetime(2026, 1, d, tzinfo=UTC) for d in (1, 5, 10))
    feb1, feb2 = datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 2, 2, tzinfo=UTC)
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
    async with pool.acquire() as admin, admin.transaction():
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
    s1, s2, s3, s4, s5, s6 = await racers[0].get_stories("a", ["s1", "s2", "s3", "s4", "s5", "s6"])
    assert (s1.valid_to, s1.superseded_by, s1.closed_at, s1.supersedes) == (feb1, "s2", feb2, ())
    assert (s2.supersedes, s2.superseded_by, s2.closed_at) == (("s1",), None, None)
    assert (s3.supersedes, s3.superseded_by, s3.valid_to) == ((), None, None)
    assert (s4.valid_to, s4.superseded_by) == (jan5, None)
    assert (s5.supersedes, s5.superseded_by) == (("gone",), None)
    assert (s6.valid_to, s6.superseded_by, s6.closed_at) == (feb1, None, None)
    async with pool.acquire() as conn:
        written = dict(
            await conn.fetch("SELECT story_id, written_to FROM memory_stories ORDER BY story_id")
        )
        indexed = await conn.fetchval(
            "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 AND indexname = 'ux_memory_stories_open_text'",
            schema,
        )
    assert written == {"s1": None, "s2": None, "s3": None, "s4": jan5, "s5": None, "s6": feb1}
    assert not indexed


async def test_the_upgrade_drops_the_old_index_only_in_its_own_schema(schemas: Schemas) -> None:
    first, second = await schemas.create(), await schemas.create()
    other = await schemas.pool(second)
    async with other.acquire() as conn:
        await conn.execute("CREATE TABLE other (a TEXT)")
        await conn.execute("CREATE UNIQUE INDEX ux_memory_stories_open_text ON other (a)")
    await PostgresStorage(await schemas.pool(first, second)).initialize(dimension=8)
    async with other.acquire() as conn:
        assert await conn.fetchval(
            "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 AND indexname = 'ux_memory_stories_open_text'",
            second,
        )


_OLD_SOURCES = [
    "ALTER TABLE memory_corpora DROP COLUMN history",
    "ALTER TABLE memory_sources DROP COLUMN retired_at, DROP COLUMN replaced_by",
    "ALTER TABLE memory_chunks DROP COLUMN retired_at",
]


async def test_initialize_adds_the_history_columns_and_reads_old_sources_as_current(
    schemas: Schemas,
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
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
        history = await conn.fetchval("SELECT history FROM memory_corpora WHERE corpus_id = 'a'")
    assert {
        ("memory_corpora", "history"),
        ("memory_sources", "retired_at"),
        ("memory_sources", "replaced_by"),
        ("memory_chunks", "retired_at"),
    } <= columns
    assert history is False


async def test_set_history_waits_for_a_running_sync(storage: PostgresStorage) -> None:
    holder = await asyncpg.connect(os.environ["RFNRY_TEST_POSTGRES_URL"])
    try:
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "memory.sync:hist"
            )
            switching = asyncio.create_task(storage.set_history("hist", True))
            await asyncio.sleep(0.5)
            assert not switching.done() and await storage.get_history("hist") is False
        await asyncio.wait_for(switching, 10)
    finally:
        await holder.close()
    assert await storage.get_history("hist") is True


async def test_erasing_waits_for_a_running_sync(storage: PostgresStorage) -> None:
    holder = await asyncpg.connect(os.environ["RFNRY_TEST_POSTGRES_URL"])
    try:
        async with holder.transaction():
            await holder.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "memory.sync:erase"
            )
            deleting = asyncio.create_task(storage.delete_source("erase", "ghost"))
            forgetting = asyncio.create_task(
                storage.forget_retired("erase", before=datetime(2030, 1, 1, tzinfo=UTC))
            )
            dropping = asyncio.create_task(storage.delete_corpus("erase"))
            await asyncio.sleep(0.5)
            assert not (deleting.done() or forgetting.done() or dropping.done())
        await asyncio.wait_for(asyncio.gather(deleting, forgetting, dropping), 10)
    finally:
        await holder.close()


async def test_neighbors_skips_a_relation_whose_chunk_is_gone(
    pool: Any, storage: PostgresStorage
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_relations (corpus_id, from_id, to_id, type, evidence_kind, evidence_id) "
            "VALUES ('orphan', 'e1', 'e2', 'FEEDS', 'chunk', 'ghost-0')"
        )
    assert await storage.neighbors("orphan", ["e1"], relation_types=None, window=Window()) == []


_OLD_CHUNKS = [
    "ALTER TABLE memory_chunks DROP COLUMN fts",
    "ALTER TABLE memory_chunks DROP COLUMN context",
    "ALTER TABLE memory_chunks RENAME COLUMN config TO language",
    "ALTER TABLE memory_chunks ALTER COLUMN language SET DEFAULT 'english'",
    "ALTER TABLE memory_chunks ADD COLUMN fts tsvector GENERATED ALWAYS AS "
    "(to_tsvector('english'::regconfig, coalesce(title, '') || ' ' || text)) STORED",
]


async def test_initialize_rebuilds_the_keyword_column_and_finds_old_chunks_by_name(
    schemas: Schemas,
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    await PostgresStorage(pool).initialize(dimension=None)
    async with pool.acquire() as conn:
        for statement in _OLD_CHUNKS:
            await conn.execute(statement)
        await conn.execute(
            "INSERT INTO memory_sources (corpus_id, source_id, name, metadata, text, created_at, chunks) "
            "VALUES ('a', 's1', 'Gasket catalogue', '{}', 'Part 4471 ships.', $1, 1)",
            datetime(2026, 1, 1, tzinfo=UTC),
        )
        await conn.execute(
            "INSERT INTO memory_chunks (corpus_id, chunk_id, source_id, position, source_name, title, text, digest) "
            "VALUES ('a', 's1-0', 's1', 0, 'Gasket catalogue', NULL, 'Part 4471 ships.', 'd')"
        )
    racers = [PostgresStorage(pool), PostgresStorage(pool)]
    await asyncio.gather(*(s.initialize(dimension=None) for s in racers))
    legs = await racers[0].search_text(
        ["a"],
        "catalogue",
        kinds=("chunk",),
        top_k=10,
        filters=None,
        window=Window(as_of=datetime(2026, 2, 1, tzinfo=UTC)),
        subject=None,
    )
    assert [(c, i) for c, (_, i, _) in legs] == [("a", "s1-0")]
    relfilenode = (
        "SELECT relfilenode FROM pg_class "
        "WHERE oid = to_regclass(format('%I.%I', $1::text, 'memory_chunks'))"
    )
    async with pool.acquire() as conn:
        expression = await conn.fetchval(
            "SELECT generation_expression FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = 'memory_chunks' AND column_name = 'fts'",
            schema,
        )
        indexed = await conn.fetchval(
            "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 AND indexname = 'ix_memory_chunks_fts'",
            schema,
        )
        before = await conn.fetchval(relfilenode, schema)
    assert "context" in expression and indexed == 1
    await PostgresStorage(pool).initialize(dimension=None)
    async with pool.acquire() as conn:
        after = await conn.fetchval(relfilenode, schema)
    assert before is not None and after == before


async def test_initialize_indexes_the_aliases_of_older_tables_and_finds_entities_by_them(
    schemas: Schemas,
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    await PostgresStorage(pool).initialize(dimension=None)
    async with pool.acquire() as conn:
        await conn.execute("DROP INDEX ix_memory_entities_alias_keys")
        await conn.execute("DROP FUNCTION memory_alias_keys(jsonb)")
        await conn.execute(
            "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases) "
            "VALUES ('a', 'e1', 'p-3', 'P-3', 'pump', '[\"Pump  THREE \"]')"
        )
    racers = [PostgresStorage(pool), PostgresStorage(pool)]
    await asyncio.gather(*(s.initialize(dimension=None) for s in racers))
    async with pool.acquire() as conn:
        indexed = await conn.fetchval(
            "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 AND indexname = 'ix_memory_entities_alias_keys'",
            schema,
        )
    assert indexed == 1
    assert [e.entity_id for e in await racers[0].entities_named("a", "pump three")] == ["e1"]
    assert [e.entity_id for e in await racers[1].entities_named("a", " P-3")] == ["e1"]
    assert await racers[0].entities_named("b", "pump three") == []


async def test_initialize_takes_no_lock_on_entities_once_their_aliases_are_indexed(
    schemas: Schemas,
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema, lock_timeout="2s")
    writer = await schemas.pool(schema)
    await PostgresStorage(pool).initialize(dimension=None)
    async with writer.acquire() as conn, conn.transaction():
        await conn.execute(
            "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type) "
            "VALUES ('a', 'e1', 'p-3', 'P-3', 'pump')"
        )
        await PostgresStorage(pool).initialize(dimension=None)
    assert [e.entity_id for e in await PostgresStorage(pool).entities_named("a", "P-3")] == ["e1"]


async def test_a_built_in_language_gets_its_folding_copy_the_first_time_a_corpus_uses_it(
    schemas: Schemas,
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    storage = PostgresStorage(pool)
    await storage.initialize(dimension=None)
    copies = (
        "SELECT cfgname FROM pg_ts_config WHERE cfgnamespace = $1::regnamespace ORDER BY cfgname"
    )
    async with pool.acquire() as conn:
        assert [r["cfgname"] for r in await conn.fetch(copies, schema)] == ["memory_english"]
    assert list(
        await asyncio.gather(
            storage.set_language("a", "french"), storage.set_language("b", "French")
        )
    ) == [0, 0]
    async with pool.acquire() as conn:
        assert [r["cfgname"] for r in await conn.fetch(copies, schema)] == [
            "memory_english",
            "memory_french",
        ]
        assert (
            await conn.fetchval(
                "SELECT to_tsvector(format('%I.memory_french', $1::text)::regconfig, 'l''école')::text",
                schema,
            )
            == "'ecol':2"
        )
        assert dict(
            await conn.fetch("SELECT corpus_id, config FROM memory_corpora ORDER BY corpus_id")
        ) == {"a": f"{schema}.memory_french", "b": f"{schema}.memory_french"}
    assert [await storage.get_language(c) for c in ("a", "b")] == ["french", "french"]


def _keyword(text: str) -> str:
    cases = " ".join(
        f"WHEN '{name}' THEN to_tsvector('memory_{name}'::regconfig, {text})"
        for name in ("spanish", "portuguese", "simple")
    )
    return f"CASE language {cases} ELSE to_tsvector('english'::regconfig, {text}) END"


_SEARCHED = "coalesce(source_name, '') || ' ' || coalesce(title, '') || ' ' || coalesce(context, '') || ' ' || text"
_LANGUAGE_COLUMNS = [
    "ALTER TABLE memory_corpora DROP COLUMN config",
    *(
        statement
        for table, text in (("memory_chunks", _SEARCHED), ("memory_stories", "text"))
        for statement in (
            f"ALTER TABLE {table} DROP COLUMN fts",
            f"ALTER TABLE {table} RENAME COLUMN config TO language",
            f"ALTER TABLE {table} ALTER COLUMN language SET DEFAULT 'english'",
            f"ALTER TABLE {table} ADD COLUMN fts tsvector GENERATED ALWAYS AS ({_keyword(text)}) STORED",
            f"CREATE INDEX ix_{table}_fts ON {table} USING GIN (fts)",
        )
    ),
    "DROP FUNCTION memory_tsvector(text, text)",
    "DROP TEXT SEARCH CONFIGURATION memory_english",
]


async def test_initialize_moves_each_passage_onto_its_configuration_once(
    schemas: Schemas,
) -> None:
    schema = await schemas.create()
    pool = await schemas.pool(schema)
    await PostgresStorage(pool).initialize(dimension=None)
    jan1 = datetime(2026, 1, 1, tzinfo=UTC)
    async with pool.acquire() as conn:
        unaccent = await conn.fetchval(
            "SELECT format('%I.unaccent', n.nspname) FROM pg_extension e "
            "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'unaccent'"
        )
        for name, dictionary in (
            ("spanish", "spanish_stem"),
            ("portuguese", "portuguese_stem"),
            ("simple", "simple"),
        ):
            await conn.execute(
                f"CREATE TEXT SEARCH CONFIGURATION memory_{name} (COPY = pg_catalog.{name})"
            )
            await conn.execute(
                f"ALTER TEXT SEARCH CONFIGURATION memory_{name} ALTER MAPPING FOR hword, hword_part, "
                f"word WITH {unaccent}, pg_catalog.{dictionary}"
            )
        for statement in _LANGUAGE_COLUMNS:
            await conn.execute(statement)
        await conn.execute(
            "INSERT INTO memory_corpora (corpus_id, language) VALUES ('es', 'spanish')"
        )
        for corpus, language, text in (
            ("es", "spanish", "Las bombas funcionan en el café."),
            ("en", "english", "The pumps run in the café."),
        ):
            await conn.execute(
                "INSERT INTO memory_sources (corpus_id, source_id, name, metadata, text, created_at, chunks) "
                "VALUES ($1, 's1', 'note', '{}', $2, $3, 1)",
                corpus,
                text,
                jan1,
            )
            await conn.execute(
                "INSERT INTO memory_chunks (corpus_id, chunk_id, source_id, position, source_name, text, digest, language) "
                "VALUES ($1, 's1-0', 's1', 0, 'note', $2, 'd', $3)",
                corpus,
                text,
                language,
            )
            await conn.execute(
                "INSERT INTO memory_stories (corpus_id, story_id, text, text_hash, valid_from, created_at, language) "
                "VALUES ($1, 't1', $2, md5($2), $3, $3, $4)",
                corpus,
                text,
                jan1,
                language,
            )
    racers = [PostgresStorage(pool), PostgresStorage(pool)]
    await asyncio.gather(*(s.initialize(dimension=None) for s in racers))
    async with pool.acquire() as conn:
        columns = {
            (r["table_name"], r["column_name"])
            for r in await conn.fetch(
                "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = $1",
                schema,
            )
        }
        configs = await conn.fetch(
            "SELECT corpus_id, config FROM memory_chunks UNION ALL "
            "SELECT corpus_id, config FROM memory_stories UNION ALL "
            "SELECT corpus_id, config FROM memory_corpora ORDER BY 1, 2"
        )
    assert {
        ("memory_corpora", "config"),
        ("memory_chunks", "config"),
        ("memory_stories", "config"),
    } <= columns
    assert not {("memory_chunks", "language"), ("memory_stories", "language")} & columns
    assert [tuple(r) for r in configs] == [
        ("en", f"{schema}.memory_english"),
        ("en", f"{schema}.memory_english"),
        ("es", f"{schema}.memory_spanish"),
        ("es", f"{schema}.memory_spanish"),
        ("es", f"{schema}.memory_spanish"),
    ]
    assert [await racers[0].get_language(c) for c in ("es", "en")] == ["spanish", "english"]
    window = Window(as_of=datetime(2026, 2, 1, tzinfo=UTC))
    for corpus, query in (("es", "funcionando"), ("en", "cafe")):
        found = await racers[1].search_text(
            [corpus],
            query,
            kinds=("chunk", "story"),
            top_k=10,
            filters=None,
            window=window,
            subject=None,
        )
        assert sorted(leg[0] for _, leg in found) == ["chunk", "story"], corpus
    relfilenodes = (
        "SELECT relname, relfilenode FROM pg_class WHERE relnamespace = $1::regnamespace "
        "AND relname IN ('memory_chunks', 'memory_stories') ORDER BY relname"
    )
    async with pool.acquire() as conn:
        before = await conn.fetch(relfilenodes, schema)
    await PostgresStorage(pool).initialize(dimension=None)
    async with pool.acquire() as conn:
        assert await conn.fetch(relfilenodes, schema) == before
