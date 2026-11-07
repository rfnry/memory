import asyncio
import os
import signal
from pathlib import Path
from typing import Any

import pytest
from database import WAIT, Database, Ran, Schemas, Spawned
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
    _forget,
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
from rfnry_memory import cli_vectors
from rfnry_memory.errors import ConfigurationError, NotFoundError, StorageError
from rfnry_memory.index import HNSW
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import PostgresStorage, migrations, switch
from rfnry_memory.postgres.switch import _staging_key, forget_staged, hold_staging, rename_model
from rfnry_memory.postgres.vectors import _vectors_table

RENAMED = "renamed-model"


async def _rename(schema: str, model: str, to: str, *argv: str) -> Ran:
    return await _vectors("rename", schema, "--model", model, "--to", to, "--yes", *argv)


class Renamed(HashEmbedding):
    model = RENAMED


async def test_vectors_index_stays_in_the_corpora_it_names(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    assert await _swap(schema, NEW, "--no-index") == 0
    assert await _vectors("index", schema, "--corpus", "hr") == 0
    assert await storage.get_vector_index("hr") == HNSW(m=8, precision="full")
    assert await _record(pool, "hr") is None
    assert await storage.get_vector_index("acme") is None
    assert await _record(pool, "acme") is not None


async def _two_left_to_build(storage: PostgresStorage, schema: str) -> None:
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    assert await _swap(schema, NEW, "--no-index") == 0


async def test_a_corpus_deleted_between_the_read_and_the_build_gets_no_table_back(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _two_left_to_build(storage, schema)
    capsys.readouterr()
    read = switch.plan_index

    async def then_deleted(conn: Any, corpus_ids: list[str], precision: Any) -> Any:
        plan = await read(conn, corpus_ids, precision)
        if corpus_ids == ["hr"]:
            await PostgresStorage(conn).delete_corpus("hr")
        return plan

    monkeypatch.setattr(cli_vectors, "plan_index", then_deleted)
    assert await _vectors("index", schema) == 0
    assert capsys.readouterr().out.endswith("hr: nothing recorded to build any more\n")
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "hr") is None
        assert await _vectors_table(conn, "acme") is not None


async def test_a_build_of_what_is_recorded_takes_the_record_under_its_own_lock(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    with pytest.raises(NotFoundError, match=r"^postgres: "):
        await storage.set_vector_index("acme", HNSW(m=8), rebuild=False, only_recorded=True)
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "acme") is None


async def test_a_corpus_deleted_while_the_queue_waits_is_not_built_again(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _two_left_to_build(storage, schema)
    capsys.readouterr()
    build = PostgresStorage.set_vector_index

    async def then_deleted(self: PostgresStorage, corpus_id: str, index: Any, **given: bool) -> int:
        built = await build(self, corpus_id, index, **given)
        if corpus_id == "acme":
            await Memory(self, embedding=Turned()).delete_corpus("hr")
        return built

    monkeypatch.setattr(PostgresStorage, "set_vector_index", then_deleted)
    assert await _vectors("index", schema) == 0
    assert capsys.readouterr().out.endswith(
        "acme: built, 6 vectors\nhr: nothing recorded to build any more\n"
    )
    assert (await Memory(storage, embedding=Turned()).verify_absent("hr")).clean
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "hr") is None


async def test_an_index_the_app_built_while_the_queue_waits_is_left_as_it_is(
    storage: PostgresStorage, schema: str, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _two_left_to_build(storage, schema)
    build = PostgresStorage.set_vector_index

    async def then_tuned(self: PostgresStorage, corpus_id: str, index: Any, **given: bool) -> int:
        built = await build(self, corpus_id, index, **given)
        if corpus_id == "acme":
            await build(self, "hr", HNSW(m=16), rebuild=False)
        return built

    monkeypatch.setattr(PostgresStorage, "set_vector_index", then_tuned)
    assert await _vectors("index", schema) == 0
    assert await storage.get_vector_index("hr") == HNSW(m=16, precision="full")
    assert await _record(pool, "hr") is None


async def test_a_failed_forget_says_what_stays_deleted_and_what_finishes_it(
    storage: PostgresStorage,
    schema: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    await _filling(storage)

    async def lost(conn: Any, *, model: str, corpus_ids: list[str]) -> int:
        raise OSError(f"lost {os.environ['RFNRY_TEST_POSTGRES_URL']}")

    monkeypatch.setattr(cli_vectors, "forget_staged", lost)
    assert await _forget(schema, NEW) == 1
    assert capsys.readouterr().err == (
        "cli: stopped by OSError: lost <url> — "
        "what was deleted stays deleted; running it again finishes\n"
    )


class Addressed(HashEmbedding):
    model = "hf://org/embed-v2"


async def test_a_corpus_and_a_model_named_like_addresses_go_through_every_action(
    storage: PostgresStorage, schema: str
) -> None:
    tenant = "tenant:ops@acme.example"
    model = Addressed.model
    served = Memory(storage, embedding=HashEmbedding())
    await served.check()
    await served.remember(tenant, "A story of the tenant.")
    await Memory(storage, embedding=Addressed(), pillars={"vector"}).stage_vectors(tenant)
    scope = ("--corpus", tenant)
    assert await _vectors("status", schema, "--model", model, *scope) == 0
    assert await _swap(schema, model, *scope) == 0
    assert await _swap(schema, OLD, *scope) == 0
    assert await storage.staged_counts(tenant) == {model: 1}
    assert await _rename(schema, model, "hf://org/embed-v3", *scope) == 0
    assert await _forget(schema, "hf://org/embed-v3", *scope) == 0
    assert await storage.staged_counts(tenant) == {}


async def _one_passage(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.remember("acme", "The only story.")


async def test_one_vector_reads_in_the_singular(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _one_passage(storage)
    assert await _rename(schema, OLD, RENAMED) == 0
    printed = capsys.readouterr().out
    assert "  1 vector labeled hash-8 takes the label renamed-model, in acme;" in printed
    assert f"schema {schema}: 1 vector of hash-8 is labeled renamed-model\n" in printed
    await _filling(storage)
    assert await _forget(schema, NEW) == 0
    assert "  1 staged vector of new-model goes, in acme\n" in capsys.readouterr().out
    await _filling(storage)
    assert await _swap(schema) == 0
    assert "  1 vector of new-model goes onto the rows, in acme\n" in capsys.readouterr().out


async def test_forget_deletes_a_models_staged_rows_in_scope(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    for corpus_id in ("acme", "hr"):
        await _filled(storage, corpus_id)
        await _filling(storage, corpus_id)
        await Memory(storage, embedding=Wide()).stage_vectors(corpus_id)
    rows = await _rows(pool)
    assert await _forget(schema, NEW, "--corpus", "acme") == 0
    assert capsys.readouterr().out == (
        "  6 staged vectors of new-model go, in acme\n"
        f"schema {schema}: deleting in batches of 10,000, each committed\n"
        f"schema {schema}: 6 staged vectors of new-model deleted\n"
        "a rollback to new-model now means staging it again\n"
    )
    assert await storage.staged_counts("acme") == {WIDE: 6}
    assert await storage.staged_counts("hr") == {NEW: 6, WIDE: 6}
    assert await _rows(pool) == rows
    assert await _forget(schema, NEW) == 0
    assert await storage.staged_counts("hr") == {WIDE: 6}


async def test_forget_with_nothing_staged_does_nothing(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    rows = await _rows(pool)
    assert await _forget(schema, NEW) == 0
    assert capsys.readouterr().out == f"schema {schema}: nothing staged for new-model\n"
    assert await _rows(pool) == rows


async def test_forget_asks_unless_yes(
    storage: PostgresStorage,
    schema: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    await _filling(storage)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert await _vectors("forget", schema, "--model", NEW) == 0
    assert capsys.readouterr().out == (
        "  6 staged vectors of new-model go, in acme\n"
        "what goes is gone: moving to new-model after this means staging it again\n"
        "nothing was run\n"
    )
    assert await storage.staged_counts("acme") == {NEW: 6}
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert await _vectors("forget", schema, "--model", NEW) == 0
    assert await storage.staged_counts("acme") == {}


async def test_after_forget_a_rollback_needs_the_old_model_staged_again(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    assert await _swap(schema) == 0
    assert await _forget(schema, OLD) == 0
    assert await _swap(schema, OLD) == 1
    assert "nothing is staged for hash-8" in capsys.readouterr().err
    await old.stage_vectors("acme")
    assert await _swap(schema, OLD) == 0


class AfterEachDelete:
    def __init__(self, conn: Any, after: Any) -> None:
        self.conn = conn
        self.after = after

    async def fetchval(self, statement: str, *args: Any) -> Any:
        return await self.conn.fetchval(statement, *args)

    async def execute(self, statement: str, *args: Any) -> str:
        done = str(await self.conn.execute(statement, *args))
        if statement.startswith("DELETE"):
            await self.after()
        return done


async def test_forget_commits_each_batch(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _filled(storage)
    await _filling(storage)
    monkeypatch.setattr(switch, "_FORGET_BATCH", 2)
    left: list[int] = []

    async def seen_from_another_connection() -> None:
        left.append((await storage.staged_counts("acme")).get(NEW, 0))

    async with pool.acquire() as conn:
        watched = AfterEachDelete(conn, seen_from_another_connection)
        assert await forget_staged(watched, model=NEW, corpus_ids=["acme"]) == 6
    assert left == [4, 2, 0, 0]


async def test_nothing_moves_a_model_while_forget_runs(
    storage: PostgresStorage, schema: str, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _filled(storage)
    await _filling(storage)
    monkeypatch.setattr(switch, "_FORGET_BATCH", 2)
    told: list[Ran] = []

    async def try_to_move() -> None:
        if not told:
            told.append(await _swap(schema, NEW, "--allow-missing"))
            told.append(await _rename(schema, OLD, RENAMED))
            told.append(await _forget(schema, NEW))

    async with pool.acquire() as conn:
        watched = AfterEachDelete(conn, try_to_move)
        assert await forget_staged(watched, model=NEW, corpus_ids=["acme"]) == 6
    assert [int(ran) for ran in told] == [1, 1, 1]
    assert all("another command is changing this schema" in ran.printed for ran in told)
    assert {model for model, *_ in (await _rows(pool)).values()} == {OLD}


async def test_forget_stops_while_another_command_changes_the_schema(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as held, held.transaction():
        await held.execute(migrations.LOCK)
        assert await _forget(schema, NEW) == 1
    assert capsys.readouterr().err == (
        "postgres: another command is changing this schema — run the command again when that ends\n"
    )
    assert await storage.staged_counts("acme") == {NEW: 6}


async def test_an_interrupted_forget_says_what_stays(
    storage: PostgresStorage, schema: str, pool: Any, database: Database
) -> None:
    await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as held, held.transaction():
        await held.execute("SELECT 1 FROM memory_staged_vectors FOR UPDATE")
        code, left = await _interrupted(
            database, "vectors", "forget", "--schema", schema, "--model", NEW, "--yes"
        )
    assert code == 130
    assert left == "cli: stopped — what was deleted stays deleted; running it again finishes\n"
    assert await _forget(schema, NEW) == 0
    assert await storage.staged_counts("acme") == {}


async def _held_both_ways(storage: PostgresStorage, corpus_id: str = "acme") -> Memory:
    old = await _filled(storage, corpus_id)
    await Memory(storage, embedding=Turned()).reembed(corpus_id, limit=3)
    await old.stage_vectors(corpus_id)
    return old


async def test_rename_changes_the_label_and_no_vector(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _held_both_ways(storage)
    rows, staged = await _rows(pool), await _staged(pool)
    assert {model for model, *_ in rows.values()} == {OLD, NEW}
    assert {model for model, *_ in staged} == {OLD}
    assert await _rename(schema, OLD, RENAMED) == 0
    assert capsys.readouterr().out == (
        "  6 vectors labeled hash-8 take the label renamed-model, in acme; no vector changes\n"
        "  writes wait until the rename ends; reads go on; after it a Memory of hash-8 finds "
        "nothing by vector there\n"
        f"schema {schema}: 6 vectors of hash-8 are labeled renamed-model\n"
        f"next: rfnry-memory vectors status --model renamed-model --schema {schema}\n"
    )
    assert await _rows(pool) == {
        ref: (RENAMED if model == OLD else model, vector, settings)
        for ref, (model, vector, settings) in rows.items()
    }
    assert await _staged(pool) == {
        (RENAMED, kind, passage): held for (_, kind, passage), held in staged.items()
    }
    renamed = Memory(storage, embedding=Renamed())
    assert (await renamed.search("acme", PUMP, pillars={"vector"})).hits
    assert not (await old.search("acme", PUMP, pillars={"vector"})).hits


async def test_rename_refuses_a_name_already_in_scope(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage, "acme")
    await _filled(storage, "hr")
    await _filling(storage, "hr")
    held = {c: (await _rows(pool, c), await _staged(pool, c)) for c in ("acme", "hr")}
    assert await _rename(schema, OLD, NEW) == 1
    assert capsys.readouterr().err == (
        "cli: new-model already labels 6 staged vectors, in hr — give another name, or drop them "
        f"with rfnry-memory vectors forget --model new-model --schema {schema}\n"
    )
    assert await _rename(schema, NEW, OLD) == 1
    assert capsys.readouterr().err == _on_rows("12 vectors on rows, in acme, hr", NEW, OLD, schema)
    assert {c: (await _rows(pool, c), await _staged(pool, c)) for c in held} == held
    assert await _rename(schema, OLD, NEW, "--corpus", "acme") == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}
    assert await _rows(pool, "hr") == held["hr"][0]
    assert await _staged(pool, "hr") == held["hr"][1]


def _on_rows(held: str, model: str, to: str, schema: str, *scope: str, aside: str = "aside") -> str:
    named = "".join(f" {flag}" for flag in scope)
    return (
        f"cli: {to} already labels {held}, and vectors forget clears only what is staged — "
        f"give another name, or hold writes and run rfnry-memory vectors rename --model {to} "
        f"--to {aside}{named} --schema {schema}, then rfnry-memory vectors rename --model {model} "
        f"--to {to}{named} --schema {schema}, then reembed under the new adapter, which writes "
        "only the rows set aside\n"
    )


async def test_rename_counts_rows_apart_from_staged_and_keeps_to_its_scope(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _held_both_ways(storage)
    assert await _rename(schema, NEW, OLD, "--corpus", "acme") == 1
    assert capsys.readouterr().err == _on_rows(
        "3 vectors on rows and 3 staged, in acme", NEW, OLD, schema, "--corpus", "acme"
    )


async def _deployed_first(storage: PostgresStorage) -> Memory:
    await _filled(storage)
    deployed = Memory(storage, embedding=Renamed())
    await deployed.remember("acme", "Written after the deploy.")
    return deployed


async def test_a_rename_after_the_deploy_goes_through_a_name_set_aside(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    deployed = await _deployed_first(storage)
    before = await _rows(pool)
    assert await _rename(schema, OLD, RENAMED) == 1
    assert capsys.readouterr().err == _on_rows("1 vector on rows, in acme", OLD, RENAMED, schema)
    assert await _forget(schema, RENAMED) == 0
    assert capsys.readouterr().out == f"schema {schema}: nothing staged for {RENAMED}\n"
    assert await _rename(schema, OLD, RENAMED) == 1
    assert await _rename(schema, RENAMED, "aside") == 0
    assert await _rename(schema, OLD, RENAMED) == 0
    assert await deployed.reembed("acme") == 1
    assert await _rows(pool) == {
        ref: (RENAMED, vector, settings) for ref, (_, vector, settings) in before.items()
    }
    assert (await deployed.search("acme", PUMP, pillars={"vector"})).hits


async def test_a_write_between_the_two_renames_refuses_the_second(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    deployed = await _deployed_first(storage)
    assert await _rename(schema, RENAMED, "aside") == 0
    await deployed.remember("acme", "Written between the two.")
    before = await _rows(pool)
    capsys.readouterr()
    assert await _rename(schema, OLD, RENAMED) == 1
    assert capsys.readouterr().err == _on_rows(
        "1 vector on rows, in acme", OLD, RENAMED, schema, aside="aside-2"
    )
    assert await _rows(pool) == before
    assert await _rename(schema, RENAMED, "aside-2") == 0
    assert await _rename(schema, OLD, RENAMED) == 0


async def test_the_way_out_of_a_taken_name_passes_every_label_taken_in_its_scope(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    deployed = await _deployed_first(storage)
    await _filled(storage, "hr")
    assert await _rename(schema, RENAMED, "aside", "--corpus", "acme") == 0
    await deployed.remember("acme", "Written between the two.")
    assert await _rename(schema, RENAMED, "aside-2", "--corpus", "acme") == 0
    await deployed.remember("acme", "Written after the second.")
    capsys.readouterr()
    assert await _rename(schema, OLD, RENAMED, "--corpus", "acme") == 1
    assert capsys.readouterr().err == _on_rows(
        "1 vector on rows, in acme", OLD, RENAMED, schema, "--corpus", "acme", aside="aside-3"
    )
    assert await _rename(schema, OLD, "aside", "--corpus", "acme") == 1
    assert capsys.readouterr().err == _on_rows(
        "1 vector on rows, in acme", OLD, "aside", schema, "--corpus", "acme", aside="aside-3"
    )
    assert await _rename(schema, OLD, "aside-3", "--corpus", "hr") == 0
    capsys.readouterr()
    assert await _rename(schema, OLD, RENAMED, "--corpus", "acme") == 1
    assert capsys.readouterr().err == _on_rows(
        "1 vector on rows, in acme", OLD, RENAMED, schema, "--corpus", "acme", aside="aside-3"
    )
    assert await _rename(schema, OLD, RENAMED) == 1
    assert capsys.readouterr().err == _on_rows(
        "1 vector on rows, in acme", OLD, RENAMED, schema, aside="aside-4"
    )


async def test_rename_stays_in_the_corpora_it_names(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _held_both_ways(storage, "acme")
    await _held_both_ways(storage, "hr")
    rows, staged = await _rows(pool, "hr"), await _staged(pool, "hr")
    assert await _rename(schema, OLD, RENAMED, "--corpus", "acme") == 0
    assert await _rows(pool, "hr") == rows
    assert await _staged(pool, "hr") == staged
    assert {model for model, *_ in await _staged(pool)} == {RENAMED}
    assert await _rename(schema, OLD, RENAMED, "--corpus", "hr") == 0
    assert {model for model, *_ in await _staged(pool, "hr")} == {RENAMED}


async def test_rename_of_a_name_nothing_holds_says_so(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    assert await _rename(schema, "typo", RENAMED) == 1
    assert capsys.readouterr().err == (
        "cli: no vector here is labeled typo — check the name with "
        f"rfnry-memory vectors status --schema {schema}\n"
    )


async def test_a_schema_with_no_corpora_has_nothing_to_index_forget_or_rename(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await _vectors("index", schema) == 0
    assert capsys.readouterr().out == f"schema {schema}: no index to build\n"
    assert await _forget(schema, NEW) == 0
    assert capsys.readouterr().out == f"schema {schema}: nothing staged for new-model\n"
    assert await _rename(schema, NEW, RENAMED) == 1


async def test_rename_asks_unless_yes(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    rows = await _rows(pool)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert await _vectors("rename", schema, "--model", OLD, "--to", RENAMED) == 0
    assert capsys.readouterr().out.endswith(
        "one transaction: every label moves or none does, and a stop changes nothing\n"
        "nothing was run\n"
    )
    assert await _rows(pool) == rows
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert await _vectors("rename", schema, "--model", OLD, "--to", RENAMED) == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {RENAMED}


async def test_a_running_stage_of_either_name_stops_a_rename_and_a_forget(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
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
    refused = (
        "postgres: a stage of new-model is running, and it would go on writing under that name — "
        "stop it, then run the command again\n"
    )
    try:
        async with asyncio.timeout(WAIT):
            while b"acme: 2 of 6" not in await process.stderr.readline():
                pass
        assert await _rename(schema, NEW, RENAMED) == 1
        assert capsys.readouterr().err == refused
        assert await _forget(schema, NEW) == 1
        assert capsys.readouterr().err == refused
        assert await storage.staged_counts("acme") == {NEW: 2}
    finally:
        process.send_signal(signal.SIGINT)
        async with asyncio.timeout(WAIT):
            await process.communicate()
    assert await _rename(schema, NEW, RENAMED) == 0
    assert await storage.staged_counts("acme") == {RENAMED: 2}


async def test_a_rename_to_a_name_a_stage_has_yet_to_write_is_refused(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    rows = await _rows(pool)
    async with pool.acquire() as staging:
        await hold_staging(staging, NEW)
        try:
            assert await _rename(schema, OLD, NEW) == 1
        finally:
            await staging.execute("SELECT pg_advisory_unlock_all()")
    assert capsys.readouterr().err == (
        "postgres: a stage of new-model is running, and it would go on writing under that name — "
        "stop it, then run the command again\n"
    )
    assert await _rows(pool) == rows


async def test_a_stage_of_a_name_being_swapped_renamed_or_forgotten_is_refused(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch)
    async with pool.acquire() as held, held.transaction():
        await held.execute(f"SELECT pg_advisory_xact_lock({_staging_key(NEW)})")
        assert await _vectors("stage", schema, "--embedding", f"{adapter}:New") == 1
    assert capsys.readouterr().err == (
        "postgres: a swap, a forget or a rename is at work on new-model — "
        "run the command again when that ends\n"
    )
    assert await storage.staged_counts("acme") == {}
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:New") == 0


async def test_a_batch_being_staged_is_renamed_with_the_rest(
    storage: PostgresStorage, schema: str, pool: Any, database: Database, spawn: Spawned
) -> None:
    await _filled(storage)
    await _filling(storage, limit=4)
    staged = await _staged(pool)
    chunk_id = next(passage for _, kind, passage in staged if kind == "chunk")
    story_id = next(passage for kind, passage in await _rows(pool) if kind == "story")
    batch = [("chunk", chunk_id), ("story", story_id)]
    async with pool.acquire() as held, held.transaction():
        await held.execute(
            "SELECT 1 FROM memory_staged_vectors WHERE corpus_id = 'acme' AND passage_id = $1 "
            "FOR UPDATE",
            chunk_id,
        )
        staging = spawn(storage.stage("acme", batch, [[0.5] * 8] * 2, NEW, settings=None))
        await _waiting(database.admin, 1)
        renaming = spawn(_rename(schema, NEW, RENAMED))
        await _waiting(database.admin, 2)
    async with asyncio.timeout(WAIT):
        await staging
        assert await renaming == 0
    assert await storage.staged_counts("acme") == {RENAMED: 5}


async def test_a_run_of_your_own_code_that_outlives_a_rename_is_seen_and_can_be_cleared(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    new = await _filling(storage, limit=4)
    assert await _rename(schema, NEW, RENAMED) == 0
    await new.stage_vectors("acme")
    assert await storage.staged_counts("acme") == {NEW: 6, RENAMED: 4}
    assert await _vectors("status", schema) == 0
    capsys.readouterr()
    assert await _rename(schema, NEW, RENAMED) == 1
    assert "rfnry-memory vectors forget" in capsys.readouterr().err
    assert await _forget(schema, NEW) == 0
    assert await storage.staged_counts("acme") == {RENAMED: 4}


async def test_a_read_during_a_rename_sees_every_old_label_and_a_write_waits(
    storage: PostgresStorage, schema: str, pool: Any, database: Database, spawn: Spawned
) -> None:
    old = await _held_both_ways(storage)
    rows, staged = await _rows(pool), await _staged(pool)
    async with pool.acquire() as held, held.transaction():
        await held.execute("SELECT 1 FROM memory_staged_vectors FOR UPDATE")
        renaming = spawn(_rename(schema, OLD, RENAMED))
        await _waiting(database.admin, 1)
        writing = spawn(old.ingest("acme", name="late", text="A line written late."))
        await _waiting(database.admin, 2)
        assert not writing.done()
        assert await _rows(pool) == rows
        assert await _staged(pool) == staged
    async with asyncio.timeout(WAIT):
        assert await renaming == 0
        await writing
    assert {model for model, *_ in await _staged(pool)} == {RENAMED}


async def test_a_rename_of_one_corpus_holds_the_writes_of_the_others_too(
    storage: PostgresStorage, schema: str, pool: Any, database: Database, spawn: Spawned
) -> None:
    await _held_both_ways(storage)
    hr = await _filled(storage, "hr")
    async with pool.acquire() as held, held.transaction():
        await held.execute("SELECT 1 FROM memory_staged_vectors FOR UPDATE")
        renaming = spawn(_rename(schema, OLD, RENAMED, "--corpus", "acme"))
        await _waiting(database.admin, 1)
        writing = spawn(hr.ingest("hr", name="late", text="A line written late."))
        await _waiting(database.admin, 2)
        assert not writing.done()
    async with asyncio.timeout(WAIT):
        assert await renaming == 0
        await writing


async def test_a_label_with_no_vector_is_neither_moved_nor_a_name_taken(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    stories = sorted(passage for kind, passage in await _rows(pool) if kind == "story")
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_stories SET embedding = NULL WHERE story_id = $1", stories[0]
        )
        await conn.execute(
            "UPDATE memory_stories SET embedding = NULL, embedding_model = $2 WHERE story_id = $1",
            stories[1],
            RENAMED,
        )
    assert await _rename(schema, OLD, RENAMED) == 0
    assert "4 vectors of hash-8 are labeled renamed-model" in capsys.readouterr().out
    assert (await _rows(pool))[("story", stories[0])][0] == OLD


async def test_a_rename_that_fails_part_way_changes_nothing(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    await _held_both_ways(storage)
    rows, staged = await _rows(pool), await _staged(pool)
    async with pool.acquire() as conn:
        await conn.execute(
            "CREATE FUNCTION refused() RETURNS trigger LANGUAGE plpgsql AS "
            "$$ BEGIN RAISE EXCEPTION 'tests: refused' USING ERRCODE = 'XX000'; END $$"
        )
        await conn.execute(
            "CREATE TRIGGER refusing BEFORE UPDATE ON memory_staged_vectors "
            "FOR EACH ROW EXECUTE FUNCTION refused()"
        )
    assert await _rename(name, OLD, RENAMED) == 1
    assert capsys.readouterr().err == (
        "postgres: the rename rolled back, nothing changed: tests: refused\n"
    )
    assert await _rows(pool) == rows
    assert await _staged(pool) == staged


async def test_a_name_taken_after_the_plan_stops_the_rename(
    storage: PostgresStorage, pool: Any
) -> None:
    await _filled(storage)
    await _filling(storage)
    rows, staged = await _rows(pool), await _staged(pool)
    async with pool.acquire() as conn:
        with pytest.raises(
            ConfigurationError,
            match="^postgres: new-model came to label vectors after the plan — "
            "run the command again$",
        ):
            await rename_model(conn, model=OLD, to=NEW, corpus_ids=["acme"])
    assert await _rows(pool) == rows
    assert await _staged(pool) == staged


async def test_a_rename_stops_while_another_command_changes_the_schema(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    rows = await _rows(pool)
    async with pool.acquire() as held, held.transaction():
        await held.execute(migrations.LOCK)
        assert await _rename(schema, OLD, RENAMED) == 1
    assert capsys.readouterr().err == (
        "postgres: another command is changing this schema — run the command again when that ends\n"
    )
    assert await _rows(pool) == rows


async def test_an_interrupted_rename_changes_nothing(
    storage: PostgresStorage, schema: str, pool: Any, database: Database
) -> None:
    await _held_both_ways(storage)
    rows, staged = await _rows(pool), await _staged(pool)
    async with pool.acquire() as held, held.transaction():
        await held.execute("SELECT 1 FROM memory_staged_vectors FOR UPDATE")
        code, left = await _interrupted(
            database,
            "vectors",
            "rename",
            "--schema",
            schema,
            "--model",
            OLD,
            "--to",
            RENAMED,
            "--yes",
        )
    assert code == 130
    assert left == "cli: stopped — the rename rolled back, nothing changed\n"
    assert await _rows(pool) == rows
    assert await _staged(pool) == staged


@pytest.mark.parametrize(
    "lost", [ConnectionResetError("tests: the connection went"), asyncio.CancelledError()]
)
async def test_a_stop_during_a_renames_commit_does_not_claim_nothing_changed(
    lost: BaseException,
) -> None:
    with pytest.raises(
        StorageError,
        match="^postgres: the rename stopped during its commit, so whether it landed is unknown — "
        "rfnry-memory vectors status shows which",
    ):
        await rename_model(LostAtCommit(lost), model=OLD, to=NEW, corpus_ids=["acme"])


@pytest.mark.parametrize(
    "stop", [ConnectionResetError("tests: the connection went"), asyncio.CancelledError()]
)
async def test_a_stop_during_a_renames_commit_exits_4_and_names_status(
    storage: PostgresStorage,
    schema: str,
    stop: BaseException,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)

    async def lost(_: Any, *, model: str, to: str, corpus_ids: list[str]) -> int:
        return await rename_model(LostAtCommit(stop), model=model, to=to, corpus_ids=corpus_ids)

    monkeypatch.setattr(cli_vectors, "rename_model", lost)
    assert await _rename(schema, OLD, RENAMED) == 4
    told = capsys.readouterr()
    assert told.err == (
        "postgres: the rename stopped during its commit, so whether it landed is unknown — "
        "rfnry-memory vectors status shows which\n"
    )
    assert "next:" not in told.out


async def test_a_rename_leaves_the_statistics_true(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    async with pool.acquire() as conn:
        await conn.execute("ANALYZE memory_chunks, memory_stories")
    assert await _rename(schema, OLD, RENAMED) == 0
    async with pool.acquire() as conn:
        held = await conn.fetch(
            "SELECT most_common_vals::text AS vals FROM pg_stats "
            "WHERE schemaname = current_schema() AND tablename IN "
            "('memory_chunks', 'memory_stories') AND attname = 'embedding_model'"
        )
    assert len(held) == 2 and all(RENAMED in r["vals"] for r in held)


ODD_TO = "it's $$ 100% \\ other"


async def test_names_that_are_no_sql_forget_and_rename_like_any_other(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage, ODD_CORPUS)
    await Memory(storage, embedding=Odd()).stage_vectors(ODD_CORPUS)
    assert await _rename(schema, ODD_MODEL, ODD_TO, "--corpus", ODD_CORPUS) == 0
    assert await storage.staged_counts(ODD_CORPUS) == {ODD_TO: 6}
    assert await _forget(schema, ODD_TO, "--corpus", ODD_CORPUS) == 0
    assert await storage.staged_counts(ODD_CORPUS) == {}
