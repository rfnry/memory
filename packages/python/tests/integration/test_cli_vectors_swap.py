import asyncio
import os
import signal
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest
from database import URL, WAIT, Database, Schemas, Spawned, provision, run_cli
from unit.fakes import HashEmbedding

from integration.cli_vectors_support import (
    NEW,
    ODD_CORPUS,
    ODD_MODEL,
    OLD,
    PUMP,
    WIDE,
    LostAtCommit,
    Odd,
    Turned,
    Wide,
    _adapter,
    _failing_after,
    _filled,
    _filling,
    _interrupted,
    _own,
    _record,
    _rows,
    _running,
    _staged,
    _swap,
    _vectors,
    _waiting,
)
from rfnry_memory import cli, cli_vectors
from rfnry_memory.errors import ConfigurationError, InputError, StorageError
from rfnry_memory.index import HNSW
from rfnry_memory.memory import Memory
from rfnry_memory.models import Counts
from rfnry_memory.postgres import PostgresStorage, migrations
from rfnry_memory.postgres.switch import SwapPlan, apply_swap, plan_swap, vector_status
from rfnry_memory.postgres.vectors import (
    _vector_indexes,
    _vectors_lock,
    _vectors_name,
    _vectors_table,
)


async def _plan(conn: Any, model: str = NEW, *, allow_missing: bool = False) -> SwapPlan:
    return await plan_swap(
        conn, model=model, corpus_ids=None, allow_missing=allow_missing, no_index=False
    )


async def test_swap_puts_the_new_vectors_on_the_rows_and_keeps_the_old_staged(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    old = await _filled(storage)
    new = await _filling(storage)
    before, filled = await _rows(pool), await _staged(pool)
    assert (await old.search("acme", PUMP, pillars={"vector"})).hits
    assert not (await new.search("acme", PUMP, pillars={"vector"})).hits
    assert await _swap(schema) == 0
    assert await _rows(pool) == {ref: (NEW, *filled[(NEW, *ref)]) for ref in before}
    assert await _staged(pool) == {
        (OLD, *ref): (vector, settings) for ref, (_, vector, settings) in before.items()
    }
    assert (await new.search("acme", PUMP, pillars={"vector"})).hits
    assert not (await old.search("acme", PUMP, pillars={"vector"})).hits


async def test_swap_back_restores_the_old_vectors_exactly(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    await _filling(storage)
    before, filled = await _rows(pool), await _staged(pool)
    assert await _swap(schema) == 0
    moved = await _rows(pool)
    assert moved != before
    assert await _swap(schema, OLD) == 0
    assert await _rows(pool) == before
    assert await _staged(pool) == filled
    assert await _swap(schema) == 0
    assert await _rows(pool) == moved


async def test_swap_refuses_while_a_passage_lacks_the_model_and_says_how_many(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage, limit=4)
    before, filled = await _rows(pool), await _staged(pool)
    assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        "cli: 2 passages lack a vector of new-model, in acme — fill them with "
        "rfnry-memory vectors stage, or pass --allow-missing to leave them as they are\n"
    )
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


async def test_swap_with_allow_missing_leaves_those_passages_as_they_are(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    new = await _filling(storage, limit=4)
    before, filled = await _rows(pool), await _staged(pool)
    moving = {(kind, passage) for _, kind, passage in filled}
    assert await _swap(schema, NEW, "--allow-missing") == 0
    assert "  2 passages lack new-model and keep what they have" in capsys.readouterr().out
    after = await _rows(pool)
    assert {ref: after[ref] for ref in moving} == {
        ref: (NEW, *filled[(NEW, *ref)]) for ref in moving
    }
    assert {ref: after[ref] for ref in before.keys() - moving} == {
        ref: before[ref] for ref in before.keys() - moving
    }
    assert await _staged(pool) == {(OLD, *ref): before[ref][1:] for ref in moving}
    assert await _vectors("status", schema, "--model", NEW) == 1
    assert "new-model: 2 missing, in acme" in capsys.readouterr().out
    assert len((await new.search("acme", PUMP, pillars={"vector"})).hits) == 4
    assert len((await old.search("acme", PUMP, pillars={"vector"})).hits) == 2


@pytest.mark.parametrize("way", ["allow-missing", "stage"])
async def test_rollback_after_new_writes_needs_allow_missing_or_a_stage(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    capsys: pytest.CaptureFixture[str],
    way: str,
) -> None:
    old = await _filled(storage)
    new = await _filling(storage)
    before = await _rows(pool)
    assert await _swap(schema) == 0
    told = await new.remember("acme", "The fan of acme hums.")
    written = ("story", told.story.story_id)
    assert await _swap(schema, OLD) == 1
    assert capsys.readouterr().err.startswith("cli: 1 passage lacks a vector of hash-8")
    if way == "stage":
        await old.stage_vectors("acme")
        assert await _swap(schema, OLD) == 0
        after = await _rows(pool)
        assert after.pop(written)[0] == OLD
        assert after == before
    else:
        assert await _swap(schema, OLD, "--allow-missing") == 0
        after = await _rows(pool)
        assert after.pop(written)[0] == NEW
        assert after == before


async def test_a_new_dimension_moves_the_whole_schema_and_refuses_part_of_it(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage, "acme")
    await old.set_vector_index("acme", HNSW(m=8))
    await _filled(storage, "hr")
    wide = Memory(storage, embedding=Wide())
    await wide.stage_vectors("acme")
    await wide.stage_vectors("hr")
    before = {corpus_id: await _rows(pool, corpus_id) for corpus_id in ("acme", "hr")}
    assert await _swap(name, WIDE, "--corpus", "acme") == 1
    assert capsys.readouterr().err == (
        "cli: wide-model is staged at 16 dimensions and the schema holds 8: a new dimension moves "
        "every corpus at once — leave out --corpus\n"
    )
    assert await _rows(pool) == before["acme"]
    assert await _swap(name, WIDE, "--corpus", "acme", "--corpus", "hr") == 0
    out = capsys.readouterr().out
    assert "  vectors of 16 dimensions: the whole schema moves" in out
    assert "  reads and writes wait until the swap ends" in out
    fresh = await schemas.pool(name)
    wide = Memory(PostgresStorage(fresh), embedding=Wide())
    await wide.check()
    with pytest.raises(ConfigurationError, match="holds vectors of 16 dimensions"):
        await Memory(PostgresStorage(fresh), embedding=HashEmbedding()).check()
    assert (await wide.search("acme", PUMP, pillars={"vector"})).hits
    async with fresh.acquire() as conn:
        assert await conn.fetchval("SELECT dimension FROM memory_settings") == 16
    assert await _record(fresh) is None
    assert await wide.storage.get_vector_index("acme") == HNSW(m=8, precision="full")
    for corpus_id in ("acme", "hr"):
        assert {model for model, *_ in (await _rows(fresh, corpus_id)).values()} == {WIDE}
        assert await _staged(fresh, corpus_id) == {
            (OLD, *ref): (vector, settings)
            for ref, (_, vector, settings) in before[corpus_id].items()
        }


async def test_swap_back_from_a_new_dimension_restores_the_old_vectors_exactly(
    schemas: Schemas,
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    before, filled = await _rows(pool), await _staged(pool)
    assert await _swap(name, WIDE) == 0
    assert await _swap(name, OLD) == 0
    fresh = await schemas.pool(name)
    assert await _rows(fresh) == before
    assert await _staged(fresh) == filled
    await Memory(PostgresStorage(fresh), embedding=HashEmbedding()).check()


async def test_a_new_dimension_with_allow_missing_clears_what_a_passage_cannot_keep(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme", limit=4)
    before = await _rows(pool)
    assert await _swap(name, WIDE, "--allow-missing") == 0
    assert (
        "  2 passages lack wide-model: no vector on the row, and what they hold stays staged"
    ) in capsys.readouterr().out
    fresh = await schemas.pool(name)
    after = await _rows(fresh)
    left = [row for row in after.values() if row[0] != WIDE]
    assert left == [(None, None, None)] * 2
    assert await _staged(fresh) == {
        (OLD, *ref): (vector, settings) for ref, (_, vector, settings) in before.items()
    }
    assert await _vectors("status", name, "--model", WIDE) == 1
    assert "wide-model: 2 missing, in acme" in capsys.readouterr().out
    assert await _vectors("status", name, "--model", OLD) == 0


async def test_a_new_dimension_leaves_the_staged_vectors_of_a_third_model(
    schemas: Schemas,
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)
    await _filling(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    third = {key: held for key, held in (await _staged(pool)).items() if key[0] == NEW}
    assert await _swap(name, WIDE) == 0
    kept = await _staged(await schemas.pool(name))
    assert {key: held for key, held in kept.items() if key[0] == NEW} == third
    assert {model for model, *_ in kept} == {OLD, NEW}


async def test_a_new_dimension_refuses_rows_holding_the_model_at_the_old_one(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)

    class Grown(Wide):
        model = OLD

    await Memory(storage, embedding=Turned()).reembed("acme", limit=2)
    await Memory(storage, embedding=Grown()).stage_vectors("acme")
    before = await _rows(pool)
    assert await _swap(name, OLD, "--allow-missing") == 1
    assert capsys.readouterr().err == (
        "cli: 4 passages hold hash-8 at 8 dimensions and it is staged at 16 — drop what is staged "
        "with rfnry-memory vectors forget, give the rows another name with rfnry-memory vectors "
        "rename, then stage it again\n"
    )
    assert await _rows(pool) == before


async def test_a_model_staged_at_two_sizes_is_refused(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_staged_vectors SET vector = '[1,2]' WHERE passage_id = "
            "(SELECT min(passage_id) FROM memory_staged_vectors)"
        )
    assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        "cli: new-model is staged at 2 and 8 dimensions, and a model has one size — "
        "rfnry-memory vectors forget drops every staged vector of it, a rollback copy included, "
        "so be sure it can be embedded again, then stage it\n"
    )


async def test_a_row_that_already_holds_the_model_stays_and_its_staged_copy_goes(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    new = await _filling(storage)
    await new.reembed("acme", limit=2)
    async with pool.acquire() as conn:
        await conn.execute("UPDATE memory_staged_vectors SET vector = '[0,0,0,0,0,0,0,1]'")
    before = await _rows(pool)
    kept = {ref: row for ref, row in before.items() if row[0] == NEW}
    assert len(kept) == 2
    assert await _swap(schema) == 0
    after = await _rows(pool)
    assert {ref: after[ref] for ref in kept} == kept
    assert {key[0] for key in await _staged(pool)} == {OLD}
    assert {(kind, passage) for _, kind, passage in await _staged(pool)} == before.keys() - kept


@pytest.mark.parametrize("embedding", [Turned, Wide])
async def test_a_vector_with_no_model_named_is_refused_rather_than_lost(
    schemas: Schemas, capsys: pytest.CaptureFixture[str], embedding: type[HashEmbedding]
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)
    await Memory(storage, embedding=embedding()).stage_vectors("acme")
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_chunks SET embedding_model = NULL WHERE chunk_id = "
            "(SELECT min(chunk_id) FROM memory_chunks)"
        )
    before = await _rows(pool)
    assert await _swap(name, embedding.model, "--allow-missing") == 1
    assert capsys.readouterr().err == (
        "cli: 1 passage holds a vector with no model named, which this swap would lose — "
        "set its embedding_model in SQL, or clear its embedding\n"
    )
    assert await _rows(pool) == before


async def test_a_vector_with_no_model_named_outside_the_swap_stops_nothing(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    await _filling(storage, limit=4)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_stories SET embedding_model = NULL WHERE story_id = "
            "(SELECT min(story_id) FROM memory_stories WHERE story_id NOT IN "
            "(SELECT passage_id FROM memory_staged_vectors))"
        )
    before = await _rows(pool)
    assert await _swap(schema, NEW, "--allow-missing") == 0
    after = await _rows(pool)
    assert [row for row in after.values() if row[0] is None] == [
        row for row in before.values() if row[0] is None
    ]


async def test_a_label_with_no_vector_counts_as_missing_until_it_is_staged(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_chunks SET embedding = NULL WHERE chunk_id = "
            "(SELECT min(chunk_id) FROM memory_chunks)"
        )
    assert await _vectors("status", schema, "--model", OLD) == 1
    assert "hash-8: 1 missing, in acme" in capsys.readouterr().out
    assert await storage.count_unstaged("acme", model=OLD, filters=None) == 1
    assert (await old.stage_vectors("acme")).embedded == 1
    assert await storage.count_unstaged("acme", model=OLD, filters=None) == 0
    assert await _vectors("status", schema, "--model", OLD) == 0
    assert await _swap(schema, OLD) == 0
    assert all(vector is not None for _, vector, _ in (await _rows(pool)).values())


async def test_swap_moves_retired_and_closed_passages_alike(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", True)
    source = await memory.ingest("acme", name="manual", text="The old manual.")
    await memory.retire_source("acme", source.source.source_id)
    told = await memory.remember("acme", "The pump runs.")
    await memory.end_story("acme", told.story.story_id)
    await _filling(storage)
    assert await _swap(schema) == 0
    assert [model for model, *_ in (await _rows(pool)).values()] == [NEW, NEW]


async def test_a_passage_with_no_vector_takes_the_staged_one(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await Memory(storage).ingest("acme", name="manual", text="Written with no model.")
    await _filling(storage)
    assert await _swap(schema) == 0
    [(model, vector, _)] = (await _rows(pool)).values()
    assert model == NEW and vector is not None
    assert await _staged(pool) == {}


async def test_a_passage_deleted_after_staging_does_not_stop_a_swap(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    old = await _filled(storage)
    source = await old.ingest("acme", name="gone", text="A sheet deleted before the swap.")
    story = await old.remember("acme", "A story deleted before the swap.")
    await _filling(storage)
    await old.delete_source("acme", source.source.source_id)
    await old.delete_story("acme", story.story.story_id)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_staged_vectors (corpus_id, model, passage_kind, passage_id, vector) "
            "VALUES ('acme', $1, 'chunk', 'a passage nothing holds', '[1,0,0,0,0,0,0,0]')",
            NEW,
        )
    before = await _rows(pool)
    assert await _swap(schema) == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}
    assert await _staged(pool) == {
        (OLD, *ref): (vector, settings) for ref, (_, vector, settings) in before.items()
    }
    await old.delete_corpus("acme")
    assert await old.verify_absent("acme") == Counts()


async def test_a_passage_deleted_after_a_swap_takes_its_old_vector_with_it(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    old = await _filled(storage)
    source = await old.ingest("acme", name="gone", text="A sheet deleted after the swap.")
    await _filling(storage)
    assert await _swap(schema) == 0
    staged = len(await _staged(pool))
    assert (await old.delete_source("acme", source.source.source_id)).staged == 1
    assert len(await _staged(pool)) == staged - 1


async def test_swap_of_one_corpus_leaves_the_others_alone(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    acme = await _filled(storage, "acme")
    hr = await _filled(storage, "hr")
    for memory, corpus_id in ((acme, "acme"), (hr, "hr")):
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    before, filled = await _rows(pool, "hr"), await _staged(pool, "hr")
    assert await _swap(schema, NEW, "--corpus", "acme", "--no-index") == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}
    assert await _rows(pool, "hr") == before
    assert await _staged(pool, "hr") == filled
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "acme") is None
        assert await _vectors_table(conn, "hr") is not None
        assert (
            await conn.fetchval("SELECT vector_index FROM memory_corpora WHERE corpus_id = 'hr'")
            is None
        )
    assert (await hr.search("hr", "The pump of hr runs.", pillars={"vector"})).hits


async def test_every_statement_of_a_swap_of_one_corpus_names_it(
    storage: PostgresStorage, pool: Any
) -> None:
    acme = await _filled(storage, "acme")
    await acme.set_vector_index("acme", HNSW(m=8))
    await _filled(storage, "hr")
    await _filling(storage, "acme")
    await _filling(storage, "hr")
    async with pool.acquire() as conn:
        plan = await plan_swap(
            conn, model=NEW, corpus_ids=["acme"], allow_missing=False, no_index=False
        )
    moving = [s for s in plan.statements if s.startswith(("INSERT", "UPDATE", "DELETE"))]
    assert len(moving) == 6
    assert all(
        "corpus_id IN ('acme')" in statement or "corpus_id = 'acme'" in statement
        for statement in moving
    )
    assert not any("'hr'" in statement for statement in plan.statements)


async def test_swap_leaves_the_index_of_a_corpus_with_nothing_staged(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage, "acme")
    hr = await _filled(storage, "hr")
    await hr.set_vector_index("hr", HNSW(m=8))
    await _filling(storage, "acme")
    assert await _swap(schema, NEW, "--allow-missing") == 0
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "hr") is not None


async def test_swap_drops_the_index_table_and_records_its_settings_and_owner(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8, ef_construction=32))
    new = await _filling(storage)
    assert await _swap(schema, NEW, "--no-index") == 0
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "acme") is None
        owner = await conn.fetchval("SELECT current_user")
    settings = HNSW(m=8, ef_construction=32, precision="full").model_dump()
    assert await _record(pool) == {"settings": settings, "owner": owner}
    assert (
        f"  acme: HNSW(m=8, ef_construction=32) goes, owned by {owner}\n"
    ) in capsys.readouterr().out
    assert (await new.search("acme", PUMP, pillars={"vector"})).hits
    assert await _vectors("status", schema) == 0
    assert "index to build: HNSW(m=8, ef_construction=32)" in capsys.readouterr().out


async def test_a_swap_before_the_index_is_built_keeps_what_was_recorded(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    assert await _swap(schema, NEW, "--no-index") == 0
    recorded = await _record(pool)
    assert recorded is not None
    assert await _swap(schema, OLD, "--no-index") == 0
    assert await _record(pool) == recorded


async def test_swap_with_no_staged_vectors_says_so(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        "cli: nothing is staged for new-model — fill it with rfnry-memory vectors stage, "
        "or check the name with rfnry-memory vectors status\n"
    )
    await _filled(storage, "hr")
    await _filling(storage, "hr")
    assert await _swap(schema, NEW, "--corpus", "acme") == 1
    assert "cli: nothing is staged for new-model" in capsys.readouterr().err


async def test_swap_before_upgrade_is_refused(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    now = migrations.LATEST
    monkeypatch.setattr(migrations, "LATEST", now + 1)
    assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        f"cli: {schema} is at version {now}, this rfnry-memory needs {now + 1} — "
        f"run rfnry-memory upgrade --schema {schema}\n"
    )
    async with pool.acquire() as conn:
        plan = await _plan(conn)
    assert plan == SwapPlan(
        [],
        [],
        ["the schema is not at this rfnry-memory's version — run rfnry-memory upgrade first"],
        [],
    )
    assert await _rows(pool) == before


async def test_a_schema_with_no_vectors_has_nothing_to_swap(schemas: Schemas) -> None:
    name = await schemas.create()
    pool = await schemas.pool(name)
    await provision(pool, dimension=None)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
    assert plan.refusals == ["the schema holds no vectors — run rfnry-memory set --dimension N"]


async def test_a_refused_plan_runs_nothing(storage: PostgresStorage, pool: Any) -> None:
    await _filled(storage)
    before = await _rows(pool)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        assert plan.refusals and not plan.statements
        with pytest.raises(InputError, match="^postgres: nothing is staged for new-model"):
            await apply_swap(conn, plan)
    assert await _rows(pool) == before


async def test_swap_sql_prints_and_runs_nothing(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    assert await _vectors("swap", schema, "--model", NEW, "--sql") == 0
    printed = capsys.readouterr().out
    lines = printed.splitlines()
    assert lines[:2] == [
        f'SET search_path TO "{schema}", public;',
        "BEGIN ISOLATION LEVEL READ COMMITTED;",
    ]
    assert lines[-1] == "COMMIT;"
    assert "LOCK TABLE memory_chunks, memory_stories IN SHARE ROW EXCLUSIVE MODE;" in lines
    assert os.environ[URL] not in printed
    assert f":{urlsplit(os.environ[URL]).password}@" not in printed
    assert await _rows(pool) == before
    async with pool.acquire() as conn:
        await conn.execute(printed)
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}


async def test_swap_sql_prints_refusals_rather_than_statements(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    assert await _vectors("swap", schema, "--model", NEW, "--sql") == 1
    printed = capsys.readouterr()
    assert printed.out == ""
    assert printed.err.startswith("cli: nothing is staged for new-model")


async def test_swap_asks_unless_yes(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert await _vectors("swap", schema, "--model", NEW) == 0
    out = capsys.readouterr().out
    assert "  6 vectors of new-model go onto the rows, in acme" in out
    assert "  writes wait until the swap ends; reads go on and see the old vectors" in out
    assert out.endswith("nothing was run\n")
    assert await _rows(pool) == before
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert await _vectors("swap", schema, "--model", NEW) == 0
    assert capsys.readouterr().out.endswith(
        f"schema {schema}: new-model is on the rows\n"
        f"next: rfnry-memory vectors status --model new-model --schema {schema}\n"
    )
    assert await _rows(pool) != before

    def unasked(prompt: str) -> str:
        raise AssertionError("tests: --yes asks nothing")

    monkeypatch.setattr("builtins.input", unasked)
    assert await _swap(schema, OLD) == 0


async def test_stale_statistics_cannot_turn_a_swap_back_into_a_crawl(
    storage: PostgresStorage, pool: Any
) -> None:
    told = await (await _filled(storage)).ingest("acme", name="bulk", text="Bulk.")
    vector = "[" + ",".join(["0.5"] * 8) + "]"
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_chunks (corpus_id, chunk_id, source_id, position, source_name, "
            "text, digest, config, embedding, embedding_model) SELECT 'acme', 'bulk-' || g, $1, g, "
            "'bulk', 'bulk', md5(g::text), 'english', $2::text::vector, $3 "
            "FROM generate_series(1, 30000) g",
            told.source.source_id,
            vector,
            OLD,
        )
        await conn.execute(
            "INSERT INTO memory_staged_vectors (corpus_id, model, passage_kind, passage_id, vector) "
            "SELECT 'acme', $1, 'chunk', chunk_id, embedding FROM memory_chunks "
            "WHERE corpus_id = 'acme'",
            NEW,
        )
        await conn.execute("ANALYZE memory_chunks, memory_staged_vectors")
        await conn.execute("UPDATE memory_chunks SET embedding_model = $1", NEW)
        await conn.execute("UPDATE memory_staged_vectors SET model = $1", OLD)
        async with asyncio.timeout(WAIT):
            await vector_status(conn, None)
            await apply_swap(conn, await _plan(conn, OLD))
    assert {model for model, *_ in (await _rows(pool)).values()} == {OLD}


async def test_a_swap_leaves_the_statistics_true(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as conn:
        await conn.execute("ANALYZE memory_chunks, memory_stories, memory_staged_vectors")
        assert await _swap(schema) == 0
        common = (
            "SELECT most_common_vals::text FROM pg_stats WHERE schemaname = $1 "
            "AND tablename = $2 AND attname = $3"
        )
        assert NEW in await conn.fetchval(common, schema, "memory_chunks", "embedding_model")
        assert NEW in await conn.fetchval(common, schema, "memory_stories", "embedding_model")
        assert OLD in await conn.fetchval(common, schema, "memory_staged_vectors", "model")


async def test_a_write_during_a_swap_waits_for_it(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    database: Database,
    spawn: Spawned,
    capsys: pytest.CaptureFixture[str],
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        async with conn.transaction(isolation="read_committed"):
            await apply_swap(conn, plan)
            writing = spawn(old.ingest("acme", name="late", text="A line written late."))
            await _waiting(database.admin, 1)
            assert not writing.done()
            assert (await old.search("acme", PUMP, pillars={"vector"})).hits
            assert await _rows(pool) == before
    async with asyncio.timeout(WAIT):
        await writing
    after = await _rows(pool)
    assert len(after) == len(before) + 1
    assert [model for model, *_ in after.values()].count(OLD) == 1
    assert await _vectors("status", schema, "--model", NEW) == 1
    assert "new-model: 1 missing, in acme" in capsys.readouterr().out


async def test_a_write_of_the_old_size_during_a_new_dimension_fails_out_loud(
    schemas: Schemas, database: Database, spawn: Spawned
) -> None:
    _, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    async with pool.acquire() as conn:
        plan = await _plan(conn, WIDE)
        async with conn.transaction(isolation="read_committed"):
            await apply_swap(conn, plan)
            writing = spawn(old.ingest("acme", name="late", text="A line written late."))
            await _waiting(database.admin, 1)
            assert not writing.done()
    async with asyncio.timeout(WAIT):
        with pytest.raises(StorageError, match="expected 16 dimensions, not 8"):
            await writing
    assert {model for model, *_ in (await _rows(pool)).values()} == {WIDE}


async def test_a_swap_of_one_corpus_holds_the_writes_of_the_others_too(
    storage: PostgresStorage, pool: Any, database: Database, spawn: Spawned
) -> None:
    await _filled(storage, "acme")
    hr = await _filled(storage, "hr")
    await _filling(storage, "acme")
    async with pool.acquire() as conn:
        plan = await plan_swap(
            conn, model=NEW, corpus_ids=["acme"], allow_missing=False, no_index=False
        )
        async with conn.transaction(isolation="read_committed"):
            await apply_swap(conn, plan)
            writing = spawn(hr.ingest("hr", name="late", text="A line written late."))
            await _waiting(database.admin, 1)
            assert not writing.done()
    async with asyncio.timeout(WAIT):
        await writing


async def test_a_write_after_the_plan_stops_the_swap_and_says_how_many(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        await old.remember("acme", "A story written after the plan.")
        before, filled = await _rows(pool), await _staged(pool)
        with pytest.raises(ConfigurationError, match="^postgres: passages lacking new-model: 1 — "):
            await apply_swap(conn, plan)
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


async def test_a_write_after_the_plan_goes_through_with_allow_missing(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as conn:
        plan = await _plan(conn, allow_missing=True)
        told = await old.remember("acme", "A story written after the plan.")
        await apply_swap(conn, plan)
    assert (await _rows(pool))[("story", told.story.story_id)][0] == OLD


async def test_an_index_built_after_the_plan_stops_the_swap(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        await old.set_vector_index("acme", HNSW(m=8))
        with pytest.raises(ConfigurationError, match="changed since the plan was shown"):
            await apply_swap(conn, plan)
    assert await _rows(pool) == before


async def test_an_index_tuned_again_after_the_plan_stops_the_swap(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        await old.set_vector_index("acme", HNSW(m=16))
        with pytest.raises(ConfigurationError, match="changed since the plan was shown"):
            await apply_swap(conn, plan)
    assert await _rows(pool) == before


async def test_an_index_finished_after_the_plan_stops_the_swap(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as conn:
        [story] = [
            i.name for i in await _vector_indexes(conn, _vectors_name("acme")) if i.kind == "story"
        ]
        await conn.execute(f"DROP INDEX {story}")
        plan = await _plan(conn)
        await old.set_vector_index("acme", HNSW(m=8))
        with pytest.raises(ConfigurationError, match="changed since the plan was shown"):
            await apply_swap(conn, plan)
        assert await _vectors_table(conn, "acme") is not None
    assert await _rows(pool) == before


async def test_a_new_dimension_planned_twice_moves_once(schemas: Schemas) -> None:
    _, pool, storage = await _own(schemas)
    await _filled(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    async with pool.acquire() as conn:
        plan = await _plan(conn, WIDE)
        await apply_swap(conn, plan)
        moved = await _rows(pool)
        with pytest.raises(ConfigurationError, match="changed since the plan was shown"):
            await apply_swap(conn, plan)
    assert await _rows(pool) == moved


async def test_a_source_being_deleted_finishes_before_a_swap_starts(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    database: Database,
    spawn: Spawned,
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    source = await old.ingest("acme", name="gone", text="A sheet deleted beside the swap.")
    await _filling(storage)
    async with pool.acquire() as held, held.transaction():
        await held.execute(
            "SELECT 1 FROM memory_sources WHERE corpus_id = 'acme' AND source_id = $1 FOR UPDATE",
            source.source.source_id,
        )
        deleting = spawn(old.delete_source("acme", source.source.source_id))
        await _waiting(database.admin, 1)
        swapping = spawn(_swap(schema))
        await _waiting(database.admin, 2)
    async with asyncio.timeout(WAIT):
        await deleting
        assert await swapping == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}


async def test_a_story_being_deleted_finishes_before_a_swap_starts(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    database: Database,
    spawn: Spawned,
) -> None:
    old = await _filled(storage)
    told = await old.remember("acme", "A story deleted beside the swap.")
    await _filling(storage)
    async with pool.acquire() as held, held.transaction():
        await held.execute(
            "SELECT 1 FROM memory_staged_vectors WHERE corpus_id = 'acme' AND passage_id = $1 "
            "FOR UPDATE",
            told.story.story_id,
        )
        deleting = spawn(old.delete_story("acme", told.story.story_id))
        await _waiting(database.admin, 1)
        swapping = spawn(_swap(schema))
        await _waiting(database.admin, 2)
    async with asyncio.timeout(WAIT):
        await deleting
        assert await swapping == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}


async def test_a_batch_being_staged_finishes_before_a_swap_starts(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    database: Database,
    spawn: Spawned,
) -> None:
    await _filled(storage)
    await _filling(storage)
    story_id = next(passage for _, kind, passage in await _staged(pool) if kind == "story")
    async with pool.acquire() as held, held.transaction():
        await held.execute(
            "SELECT 1 FROM memory_staged_vectors WHERE corpus_id = 'acme' AND passage_id = $1 "
            "FOR UPDATE",
            story_id,
        )
        staging = spawn(
            storage.stage("acme", [("story", story_id)], [[0.5] * 8], NEW, settings=None)
        )
        await _waiting(database.admin, 1)
        swapping = spawn(_swap(schema))
        await _waiting(database.admin, 2)
    async with asyncio.timeout(WAIT):
        await staging
        assert await swapping == 0
    assert (await _rows(pool))[("story", story_id)][1] == "[" + ",".join(["0.5"] * 8) + "]"


async def test_deleting_another_corpus_beside_a_swap_ends_with_both_done(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    database: Database,
    spawn: Spawned,
) -> None:
    await _filled(storage, "acme")
    hr = await _filled(storage, "hr")
    await _filling(storage, "acme")
    async with pool.acquire() as held, held.transaction():
        await held.execute("LOCK TABLE memory_stories IN SHARE ROW EXCLUSIVE MODE")
        deleting = spawn(hr.delete_corpus("hr"))
        await _waiting(database.admin, 1)
        swapping = spawn(_swap(schema, NEW, "--corpus", "acme"))
        await _waiting(database.admin, 2)
    async with asyncio.timeout(WAIT):
        await deleting
        assert await swapping == 0
    assert await hr.verify_absent("hr") == Counts()


async def test_a_swap_stops_while_an_index_is_being_built(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as building, _vectors_lock(building, "acme"):
        assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        "postgres: the vector index of acme is being built or dropped — "
        "run the command again when that ends\n"
    )
    assert await _rows(pool) == before
    assert await _swap(schema) == 0


async def test_a_swap_stops_while_another_command_changes_the_schema(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as held, held.transaction():
        await held.execute(migrations.LOCK)
        assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        "postgres: another command is changing this schema — run the command again when that ends\n"
    )
    assert await _rows(pool) == before


async def test_an_interrupted_swap_changes_nothing(
    storage: PostgresStorage, schema: str, pool: Any, database: Database
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before, filled = await _rows(pool), await _staged(pool)
    async with pool.acquire() as reading, reading.transaction():
        table = await _vectors_table(reading, "acme")
        await reading.fetchval(f"SELECT count(*) FROM {table}")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "rfnry_memory.cli",
            "vectors",
            "swap",
            "--url",
            os.environ[URL],
            "--schema",
            schema,
            "--model",
            NEW,
            "--yes",
            env={**os.environ, "PYTHONPATH": str(Path(cli.__file__).parents[1])},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await _waiting(database.admin, 1)
        process.send_signal(signal.SIGINT)
        async with asyncio.timeout(WAIT):
            _, left = await process.communicate()
    assert process.returncode == 130
    assert left.decode() == "cli: stopped — the swap rolled back, nothing moved\n"
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


async def test_a_swap_that_fails_part_way_changes_nothing(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before, filled = await _rows(pool), await _staged(pool)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        table = await _vectors_table(conn, "acme")
        await conn.execute(f"ALTER TABLE {table} RENAME TO memory_vectors_elsewhere")
        broken = plan._replace(statements=[s for s in plan.statements if not s.startswith("DO ")])
        with pytest.raises(StorageError, match="^postgres: the swap rolled back, nothing changed"):
            await apply_swap(conn, broken)
        await conn.execute(
            f"ALTER TABLE memory_vectors_elsewhere RENAME TO {_vectors_name('acme')}"
        )
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


@pytest.mark.parametrize(
    "lost", [ConnectionResetError("tests: the connection went"), asyncio.CancelledError()]
)
async def test_a_stop_during_the_commit_does_not_claim_nothing_changed(
    lost: BaseException,
) -> None:
    with pytest.raises(
        StorageError,
        match="^postgres: the swap stopped during its commit, so whether it landed is unknown — "
        "rfnry-memory vectors status shows which",
    ):
        await apply_swap(LostAtCommit(lost), SwapPlan([], ["SELECT 1"], [], []))


async def test_a_connection_lost_before_the_commit_says_nothing_changed() -> None:
    conn = await asyncpg.connect(os.environ[URL])
    plan = SwapPlan([], ["SELECT pg_terminate_backend(pg_backend_pid())"], [], [])
    with pytest.raises(StorageError, match="^postgres: the swap rolled back, nothing changed"):
        await apply_swap(conn, plan)


@asynccontextmanager
async def _role(schemas: Schemas, schema: str, pool: Any) -> AsyncIterator[str]:
    role = f"{schemas.run}_role_{uuid4().hex[:8]}"
    async with pool.acquire() as conn:
        await conn.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'role'")
        await conn.execute(f"GRANT USAGE, CREATE ON SCHEMA {schema} TO {role}")
        await conn.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema} TO {role}"
        )
    try:
        yield role
    finally:
        async with pool.acquire() as conn:
            await conn.execute(f"REASSIGN OWNED BY {role} TO CURRENT_USER")
            await conn.execute(f"DROP OWNED BY {role}")
            await conn.execute(f"DROP ROLE {role}")


def _url_of(role: str) -> str:
    parts = urlsplit(os.environ[URL])
    return urlunsplit(parts._replace(netloc=f"{role}:role@{parts.hostname}:{parts.port}"))


async def _vectors_as(role: str, action: str, schema: str, *argv: str) -> int:
    given = ["vectors", action, "--url", _url_of(role), "--schema", schema, *argv]
    return await asyncio.to_thread(cli.main, given)


async def _swap_as(role: str, schema: str, model: str = NEW, *argv: str) -> int:
    return await _vectors_as(role, "swap", schema, "--model", model, "--yes", *argv)


async def _owner(conn: Any, corpus_id: str = "acme") -> str:
    return str(
        await conn.fetchval(
            "SELECT tableowner FROM pg_tables WHERE schemaname = current_schema() "
            "AND tablename = $1",
            _vectors_name(corpus_id),
        )
    )


class SplitRoles(NamedTuple):
    schema: str
    administrator: str
    tenant: str
    admin: Any
    served: Any
    storage: PostgresStorage


@asynccontextmanager
async def _split_roles(schemas: Schemas, database: Database) -> AsyncIterator[SplitRoles]:
    name = f"{schemas.run}_{uuid4().hex[:8]}"
    administrator, tenant = f"{name}_administrator", f"{name}_tenant"
    here = await database.admin.fetchval("SELECT quote_ident(current_database())")
    await database.admin.execute(f"CREATE ROLE {administrator} LOGIN CREATEROLE PASSWORD 'role'")
    await database.admin.execute(f"GRANT CREATE ON DATABASE {here} TO {administrator}")
    schemas._made.append(name)
    path = {"search_path": f"{name}, public"}
    admin = await asyncpg.connect(_url_of(administrator), server_settings=path)
    await admin.execute(f"CREATE ROLE {tenant} LOGIN PASSWORD 'role'")
    made = ["create", "--url", _url_of(administrator), "--schema", name, "--dimension", "8"]
    assert await asyncio.to_thread(cli.main, made) == 0
    await admin.execute(f"GRANT USAGE, CREATE ON SCHEMA {name} TO {tenant}")
    await admin.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {name} TO {tenant}"
    )
    await admin.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {name} "
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {tenant}"
    )
    served = await asyncpg.create_pool(_url_of(tenant), server_settings=path)
    try:
        yield SplitRoles(name, administrator, tenant, admin, served, PostgresStorage(served))
    finally:
        await served.close()
        await admin.close()
        for role in (tenant, administrator):
            await database.admin.execute(f"REASSIGN OWNED BY {role} TO CURRENT_USER")
            await database.admin.execute(f"DROP OWNED BY {role}")
            await database.admin.execute(f"DROP ROLE {role}")


async def _app_built(split: SplitRoles) -> bool:
    table = await _vectors_table(split.admin, "acme")
    readable = await split.admin.fetchval("SELECT has_table_privilege($1, 'SELECT')", table)
    return not readable and await _owner(split.admin) == split.tenant


async def test_a_role_that_cannot_alter_is_refused_before_anything_moves(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    before, filled = await _rows(pool), await _staged(pool)
    async with pool.acquire() as conn:
        owner = await conn.fetchval("SELECT current_user")
    async with _role(schemas, name, pool) as role:
        assert await _swap_as(role, name, WIDE) == 1
        assert capsys.readouterr().err == (
            f"cli: {role} does not own memory_chunks and memory_stories, which a new dimension "
            f"retypes — run this as {owner}\n"
        )
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


async def test_a_role_that_cannot_drop_an_index_is_refused_before_anything_moves(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before, filled = await _rows(pool), await _staged(pool)
    async with pool.acquire() as conn:
        owner = await conn.fetchval("SELECT current_user")
    async with _role(schemas, name, pool) as role:
        assert await _swap_as(role, name) == 1
        assert capsys.readouterr().err == (
            f"cli: {role} cannot drop the vector index of acme — run this as {owner}\n"
        )
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


async def test_the_role_that_owns_an_index_may_swap_without_owning_the_schema(
    schemas: Schemas,
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    async with _role(schemas, name, pool) as role, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {role}")
        assert await _swap_as(role, name) == 0
        assert await _owner(conn) == role


async def test_the_schema_owner_may_drop_a_table_another_role_owns(
    schemas: Schemas, database: Database
) -> None:
    async with _split_roles(schemas, database) as split:
        old = await _filled(split.storage)
        await old.set_vector_index("acme", HNSW(m=8))
        await _filling(split.storage)
        assert await _app_built(split)
        assert await _swap_as(split.administrator, split.schema, NEW, "--no-index") == 0
        assert await _vectors_table(split.admin, "acme") is None
        recorded = await _record(split.served)
    assert recorded is not None and recorded["owner"] == split.tenant


async def test_a_non_superuser_administrator_switches_an_index_the_app_built(
    schemas: Schemas, database: Database, capsys: pytest.CaptureFixture[str]
) -> None:
    async with _split_roles(schemas, database) as split:
        name, administrator, tenant = split.schema, split.administrator, split.tenant
        old = await _filled(split.storage)
        await old.set_vector_index("acme", HNSW(m=8))
        assert await _app_built(split)
        checked = ["check", "--url", _url_of(administrator), "--schema", name]
        assert await asyncio.to_thread(cli.main, checked) == 0
        assert await _vectors_as(administrator, "status", name) == 0
        assert "HNSW(m=8)" in capsys.readouterr().out
        await Memory(split.storage, embedding=Wide()).stage_vectors("acme")
        assert await _swap_as(tenant, name, WIDE, "--no-index") == 1
        assert capsys.readouterr().err == (
            f"cli: {tenant} does not own memory_chunks and memory_stories, which a new dimension "
            f"retypes — run this as {administrator}\n"
        )
        assert await _swap_as(administrator, name, WIDE, "--no-index") == 0
        assert capsys.readouterr().out.endswith(_next_as(tenant, name))
        assert await _vectors_as(tenant, "index", name) == 0
        assert await _owner(split.admin) == tenant
        await split.served.expire_connections()
        wide = Memory(PostgresStorage(split.served), embedding=Wide())
        await wide.check()
        assert (await wide.search("acme", PUMP, pillars={"vector"})).hits
        assert await wide.storage.get_vector_index("acme") == HNSW(m=8, precision="full")
        assert (await wide.delete_corpus("acme")).vectors == 6
        assert await _vectors_table(split.admin, "acme") is None
        assert (await wide.verify_absent("acme")).clean


async def test_a_schema_owner_still_hears_of_any_other_table_it_cannot_read(
    schemas: Schemas, database: Database, capsys: pytest.CaptureFixture[str]
) -> None:
    async with _split_roles(schemas, database) as split:
        await database.admin.execute(
            f"ALTER TABLE {split.schema}.memory_sources OWNER TO {split.tenant}"
        )
        assert await _vectors_as(split.administrator, "status", split.schema) == 1
        assert capsys.readouterr().err.startswith(
            f"cli: {split.administrator} lacks SELECT, INSERT, UPDATE, DELETE on memory_sources — "
        )


async def test_names_that_are_no_sql_swap_like_any_other(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage, ODD_CORPUS)
    await old.set_vector_index(ODD_CORPUS, HNSW(m=8))
    await Memory(storage, embedding=Odd()).stage_vectors(ODD_CORPUS)
    before = await _rows(pool, ODD_CORPUS)
    assert await _swap(schema, ODD_MODEL, "--corpus", ODD_CORPUS) == 0
    assert {model for model, *_ in (await _rows(pool, ODD_CORPUS)).values()} == {ODD_MODEL}
    capsys.readouterr()
    assert await _vectors("swap", schema, "--model", OLD, "--corpus", ODD_CORPUS, "--sql") == 0
    async with pool.acquire() as conn:
        await conn.execute(capsys.readouterr().out)
    assert await _rows(pool, ODD_CORPUS) == before


async def test_a_schema_name_that_needs_quoting_swaps_to_a_new_dimension(
    schemas: Schemas,
) -> None:
    name = f"{schemas.run}_{uuid4().hex[:8]} Odd-Name"
    assert await run_cli("create", "--schema", name, "--dimension", "8") == 0
    quoted = '"' + name + '"'
    schemas._made.append(quoted)
    storage = PostgresStorage(await schemas.pool(quoted))
    memory = await _filled(storage)
    await memory.set_vector_index("acme", HNSW(m=8))
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    assert await _swap(name, WIDE) == 0
    fresh = PostgresStorage(await schemas.pool(quoted))
    wide = Memory(fresh, embedding=Wide())
    await wide.check()
    assert (await wide.search("acme", PUMP, pillars={"vector"})).hits


BUILDING = "an index takes real time at scale; searches run exactly until it is built"


HELD = "the swap held and new-model is on the rows; "


LEFT_TO_INDEX = "the index of acme is left to rfnry-memory vectors index"


UNTIL_BUILT = "and searches there run exactly until it is built"


class Broad(HashEmbedding):
    model = "broad-model"
    dimension = 2008

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [vector * 251 for vector in await super().embed(texts, purpose=purpose)]


class Vast(HashEmbedding):
    model = "vast-model"
    dimension = 4008

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [vector * 501 for vector in await super().embed(texts, purpose=purpose)]


def _only(owner: str, schema: str) -> str:
    return (
        f"cli: only {owner} builds the index in acme, so its table keeps its owner — "
        f"run rfnry-memory vectors index --url URL --corpus acme --schema {schema} as {owner}\n"
    )


def _next_as(owner: str, schema: str) -> str:
    return (
        f"next: rfnry-memory vectors index --url URL --corpus acme --schema {schema} as {owner}\n"
    )


def _unbuilt(schema: str, held: str = "") -> str:
    return (
        f"{held}the index in acme is still to build, its searches run exactly, and "
        f"rfnry-memory vectors index --corpus acme --schema {schema} finishes it\n"
    )


async def test_swap_rebuilds_the_index_it_dropped(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    tuned = HNSW(m=8, ef_construction=32, precision="full", ef_search=40)
    await old.set_vector_index("acme", tuned)
    new = await _filling(storage)
    assert await _swap(schema) == 0
    out = capsys.readouterr().out
    assert (
        "  the index of acme builds again after the swap, and searches there run exactly "
        "until it ends\n"
    ) in out
    assert out.endswith(
        f"schema {schema}: new-model is on the rows\n"
        f"schema {schema}: {BUILDING}\n"
        "acme: building HNSW(m=8, ef_construction=32, ef_search=40)\n"
        "acme: built, 6 vectors\n"
        f"next: rfnry-memory vectors status --model new-model --schema {schema}\n"
    )
    assert await _record(pool) is None
    assert await storage.get_vector_index("acme") == tuned
    async with pool.acquire() as conn:
        table = await _vectors_table(conn, "acme")
        indexed = {
            (r["passage_kind"], r["passage_id"]): r["vector"]
            for r in await conn.fetch(
                f"SELECT passage_kind, passage_id, vector::text AS vector FROM {table}"
            )
        }
    assert indexed == {ref: vector for ref, (_, vector, _) in (await _rows(pool)).items()}
    assert (await new.search("acme", PUMP, pillars={"vector"})).hits
    assert await _vectors("status", schema) == 0
    shown = capsys.readouterr().out
    assert "HNSW(m=8, ef_construction=32, ef_search=40)" in shown
    assert "index to build" not in shown


async def test_swap_no_index_leaves_it_to_vectors_index(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    assert await _swap(schema, NEW, "--no-index") == 0
    out = capsys.readouterr().out
    assert f"  {LEFT_TO_INDEX}, {UNTIL_BUILT}\n" in out
    assert out.endswith(
        f"schema {schema}: new-model is on the rows\n"
        f"next: rfnry-memory vectors index --corpus acme --schema {schema}\n"
    )
    assert await storage.get_vector_index("acme") is None
    assert await _record(pool) is not None
    assert await _vectors("index", schema) == 0
    assert capsys.readouterr().out == (
        f"schema {schema}: {BUILDING}\nacme: building HNSW(m=8)\nacme: built, 6 vectors\n"
    )
    assert await _record(pool) is None
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="full")
    assert await _vectors("index", schema) == 0
    assert capsys.readouterr().out == f"schema {schema}: no index to build\n"


async def test_swap_builds_what_it_moved_and_no_other_corpus(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    assert await _swap(schema, NEW, "--corpus", "hr", "--no-index") == 0
    assert await _swap(schema) == 0
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="full")
    assert await storage.get_vector_index("hr") is None
    assert await _record(pool, "hr") is not None


async def test_a_swap_builds_what_an_earlier_swap_left_to_build(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    assert await _swap(schema, NEW, "--no-index") == 0
    capsys.readouterr()
    assert await _swap(schema, OLD) == 0
    assert (
        "  the index of acme builds again after the swap, and searches there run exactly "
        "until it ends\n"
    ) in capsys.readouterr().out
    assert await _record(pool) is None
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="full")


async def test_an_index_keeps_its_owner(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    async with _role(schemas, name, pool) as app, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {app}")
        assert await _swap(name) == 0
        told = capsys.readouterr()
        assert f"  {LEFT_TO_INDEX}, run as {app}, {UNTIL_BUILT}\n" in told.out
        assert told.out.endswith(f"schema {name}: new-model is on the rows\n{_next_as(app, name)}")
        assert BUILDING not in told.out
        assert told.err == ""
        assert await _vectors_table(conn, "acme") is None
        recorded = await _record(pool)
        assert recorded is not None and recorded["owner"] == app
        assert await _vectors("index", name) == 1
        assert capsys.readouterr().err == _only(app, name)
        assert await _vectors_table(conn, "acme") is None
        assert await _vectors_as(app, "index", name) == 0
        assert await _owner(conn) == app
        assert await _record(pool) is None
        served = await asyncpg.create_pool(
            _url_of(app), server_settings={"search_path": f"{name}, public"}
        )
        try:
            deleted = await Memory(PostgresStorage(served), embedding=Turned()).delete_corpus(
                "acme"
            )
        finally:
            await served.close()
        assert deleted.vectors == 6
        assert await _vectors_table(conn, "acme") is None


async def test_a_role_builds_what_it_owns_and_names_the_rest(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    async with _role(schemas, name, pool) as app, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {app}")
        await conn.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA {name} "
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {app}"
        )
        assert await _swap(name, NEW, "--no-index") == 0
        capsys.readouterr()
        assert await _vectors("index", name) == 1
        told = capsys.readouterr()
        assert told.err == _only(app, name)
        assert "hr: built, 6 vectors\n" in told.out
        assert await _vectors_table(conn, "acme") is None
        assert await _record(pool, "hr") is None
        assert await _vectors_as(app, "index", name) == 0
        assert await _owner(conn) == app
        assert await _owner(conn, "hr") == await conn.fetchval("SELECT current_user")


async def test_a_swap_builds_what_its_role_owns_and_names_the_owner_of_the_rest(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    async with _role(schemas, name, pool) as app, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {app}")
        assert await _swap(name) == 0
        told = capsys.readouterr()
        assert told.err == ""
        assert told.out.endswith(f"hr: built, 6 vectors\n{_next_as(app, name)}")
        assert await _vectors_table(conn, "acme") is None


async def test_a_build_that_fails_beside_an_owners_index_names_both(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)

    async def refused(*_: Any, **__: Any) -> int:
        raise StorageError("postgres: tests: the build failed")

    monkeypatch.setattr(PostgresStorage, "set_vector_index", refused)
    async with _role(schemas, name, pool) as app, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {app}")
        assert await _swap(name) == 3
        told = capsys.readouterr()
        assert told.err == (
            "cli: stopped by StorageError: postgres: tests: the build failed — "
            f"{HELD}the index in hr is still to build, its searches run exactly, and "
            f"rfnry-memory vectors index --corpus hr --schema {name} finishes it\n"
        )
        assert told.out.endswith(_next_as(app, name))


@pytest.mark.parametrize(
    "stop", [ConnectionResetError("tests: the connection went"), asyncio.CancelledError()]
)
async def test_a_stop_during_the_swaps_commit_exits_4_and_names_status(
    storage: PostgresStorage,
    schema: str,
    stop: BaseException,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    await _filling(storage)

    async def lost(_: Any, plan: SwapPlan) -> None:
        await apply_swap(LostAtCommit(stop), plan)

    monkeypatch.setattr(cli_vectors, "apply_swap", lost)
    assert await _swap(schema) == 4
    told = capsys.readouterr()
    assert told.err == (
        "postgres: the swap stopped during its commit, so whether it landed is unknown — "
        "rfnry-memory vectors status shows which\n"
    )
    assert "next:" not in told.out


async def test_a_full_index_moved_past_2000_dimensions_builds_as_half(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await Memory(storage, embedding=Broad()).stage_vectors("acme")
    assert await _swap(name, "broad-model") == 0
    out = capsys.readouterr().out
    assert (
        "  the index of acme builds at half precision: full holds 2000 dimensions, "
        "and broad-model has 2008\n"
    ) in out
    assert "acme: building HNSW(m=8, precision=half)\n" in out
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="half")
    assert await _record(pool) is None


async def test_past_4000_dimensions_swap_says_to_pass_binary_before_it_starts(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await Memory(storage, embedding=Vast()).stage_vectors("acme")
    before, filled = await _rows(pool), await _staged(pool)
    assert await _swap(name, "vast-model") == 1
    assert capsys.readouterr().err == (
        "cli: the index of acme reaches 4000 dimensions and vast-model has 4008 — pass --no-index, "
        "then rfnry-memory vectors index --precision binary builds it\n"
    )
    assert await _rows(pool) == before
    assert await _staged(pool) == filled
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="full")
    assert await _swap(name, "vast-model", "--no-index") == 0
    out = capsys.readouterr().out
    assert (
        "  the index of acme reaches 4000 dimensions: "
        "rfnry-memory vectors index --precision binary builds it\n"
    ) in out
    assert out.endswith(
        f"next: rfnry-memory vectors index --precision binary --corpus acme --schema {name}\n"
    )
    assert await _vectors("index", name) == 1
    assert capsys.readouterr().err == (
        "cli: the index in acme reaches 4000 dimensions and this schema holds 4008 — "
        f"run rfnry-memory vectors index --precision binary --corpus acme --schema {name}\n"
    )
    assert await _record(pool) is not None
    assert await _vectors("index", name, "--precision", "binary") == 0
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="binary")
    assert await _record(pool) is None
    vast = Memory(PostgresStorage(await schemas.pool(name)), embedding=Vast())
    await vast.check()
    assert (await vast.search("acme", PUMP, pillars={"vector"})).hits


async def test_precision_overrides_what_was_recorded(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    assert await _swap(schema, NEW, "--no-index") == 0
    assert await _vectors("index", schema, "--precision", "half") == 0
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="half")


async def test_a_failed_rebuild_says_the_swap_held_and_what_finishes_it(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    new = await _filling(storage)
    async with _role(schemas, name, pool) as app, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {app}")
        await conn.execute(f"REVOKE CREATE ON SCHEMA {name} FROM {app}")
        assert await _swap_as(app, name) == 3
        told = capsys.readouterr()
        assert f"schema {name}: new-model is on the rows\n" in told.out
        assert "next:" not in told.out
        assert told.err.startswith("cli: stopped by StorageError: postgres: storage rejected ")
        assert told.err.endswith(f" — {_unbuilt(name, HELD)}")
        assert ":role@" not in told.out + told.err
        assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}
        assert (await new.search("acme", PUMP, pillars={"vector"})).hits
        recorded = await _record(pool)
        assert recorded is not None and recorded["owner"] == app
        assert await _vectors_as(app, "index", name) == 1
        assert capsys.readouterr().err.endswith(f" — {_unbuilt(name)}")
        await conn.execute(f"GRANT CREATE ON SCHEMA {name} TO {app}")
        assert await _vectors_as(app, "index", name) == 0
        assert await _owner(conn) == app
    assert await _record(pool) is None


async def test_an_interrupted_build_says_the_swap_held_and_running_index_finishes_it(
    storage: PostgresStorage, schema: str, pool: Any, database: Database
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    swap = ("vectors", "swap", "--schema", schema, "--model", NEW, "--yes")
    async with pool.acquire() as reading, reading.transaction(isolation="repeatable_read"):
        await reading.fetchval("SELECT count(*) FROM memory_chunks")
        code, left = await _interrupted(database, *swap)
    assert code == 3
    assert left == f"cli: stopped — {_unbuilt(schema, HELD)}"
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}
    assert await _record(pool) is not None
    assert await storage.get_vector_index("acme") is None
    async with pool.acquire() as reading, reading.transaction(isolation="repeatable_read"):
        await reading.fetchval("SELECT count(*) FROM memory_chunks")
        code, left = await _interrupted(database, "vectors", "index", "--schema", schema)
    assert code == 130
    assert left == f"cli: stopped — {_unbuilt(schema)}"
    assert await _record(pool) is not None
    assert await _vectors("index", schema) == 0
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="full")
    assert await _record(pool) is None


async def test_a_running_stage_of_the_model_stops_a_swap(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    rows = await _rows(pool)
    adapter = _adapter(tmp_path, monkeypatch, _failing_after(1, "await asyncio.sleep(3600)"))
    process = await _running(
        "vectors",
        "stage",
        "--schema",
        schema,
        "--embedding",
        f"{adapter}:Failing",
        "--batch",
        "2",
        cwd=tmp_path,
    )
    assert process.stderr is not None
    try:
        async with asyncio.timeout(WAIT):
            while b"acme: 2 of 6" not in await process.stderr.readline():
                pass
        assert await _swap(schema, NEW, "--allow-missing") == 1
        assert capsys.readouterr().err == (
            "postgres: a stage of new-model is running, and it would go on writing under that "
            "name — stop it, then run the command again\n"
        )
        assert await _rows(pool) == rows
        assert await storage.staged_counts("acme") == {NEW: 2}
    finally:
        process.send_signal(signal.SIGINT)
        async with asyncio.timeout(WAIT):
            await process.communicate()
    assert await _swap(schema, NEW, "--allow-missing") == 0
    assert await storage.staged_counts("acme") == {OLD: 2}
