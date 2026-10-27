import copy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from database import Schemas, provision, run_cli
from unit.fakes import HashEmbedding

from rfnry_memory.memory import Memory, _window
from rfnry_memory.models import SourceIn
from rfnry_memory.postgres import keys
from rfnry_memory.postgres.search import _text_sql
from rfnry_memory.postgres.sql import _scope_sql
from rfnry_memory.ranking import BM25

NAMES = ("chunks_fts", "stories_fts", "chunks_ended", "stories_ended")


async def _run(*argv: str) -> int:
    return await run_cli(*argv)


async def _schema(schemas: Schemas) -> tuple[str, Any, Memory]:
    name = await schemas.create()
    pool = await schemas.pool(name)
    storage = await provision(pool)
    return name, pool, Memory(storage, embedding=HashEmbedding())


async def _held(pool: Any) -> dict[str, bool | None]:
    async with pool.acquire() as conn:
        return await keys._current(conn)


async def test_index_current_builds_four_indexes_and_check_lists_them(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, memory = await _schema(schemas)
    await memory.ingest("acme", name="pump", text="Replace the pump seal.")
    assert await _run("index", "--schema", name, "--current", "--yes") == 0
    out = capsys.readouterr().out
    assert "  index the keywords of current chunks and stories, and the ends of retired" in out
    assert f"schema {name}: current rows indexed" in out
    assert await _held(pool) == {n: True for n in NAMES}
    assert await _run("index", "--schema", name, "--current", "--yes") == 0
    assert f"schema {name}: nothing to do" in capsys.readouterr().out
    assert await _run("check", "--schema", name) == 0
    assert "\n  current rows: indexed\n" in capsys.readouterr().out


async def test_index_current_with_sql_prints_its_statements_and_builds_nothing(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, _ = await _schema(schemas)
    assert await _run("index", "--schema", name, "--current", "--sql") == 0
    out = capsys.readouterr().out
    assert "USING GIN (fts) WHERE retired_at IS NULL;" in out
    assert (
        "(corpus_id, (CASE WHEN closed_at IS NOT NULL THEN coalesce(valid_to, "
        "'infinity'::timestamptz) END));" in out
    )
    assert await _held(pool) == {n: None for n in NAMES}


async def test_index_current_drop_removes_them_and_a_partial_build_is_finished(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, _ = await _schema(schemas)
    async with pool.acquire() as conn:
        plan = await keys.plan_current(conn, False)
        for statement in plan.statements[:2]:
            await conn.execute(statement)
    assert await _run("check", "--schema", name) == 0
    assert (
        f"  current rows: not ready — run rfnry-memory index --current --schema {name}"
        in capsys.readouterr().out
    )
    assert await _run("index", "--schema", name, "--current", "--yes") == 0
    assert await _held(pool) == {n: True for n in NAMES}
    assert await _run("index", "--schema", name, "--current", "--drop", "--yes") == 0
    assert f"schema {name}: current rows no longer indexed" in capsys.readouterr().out
    assert await _held(pool) == {n: None for n in NAMES}
    assert await _run("index", "--schema", name, "--current", "--drop", "--yes") == 0
    assert f"schema {name}: nothing to do" in capsys.readouterr().out
    assert await _run("check", "--schema", name) == 0
    assert "current rows" not in capsys.readouterr().out


async def test_index_refuses_current_with_a_type_and_a_key_without_one(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, _, _ = await _schema(schemas)
    assert await _run("index", "--schema", name, "--current", "--type", "number") == 1
    assert "cli: --current indexes the current rows; it takes no --type" in capsys.readouterr().err
    assert await _run("index", "--schema", name, "--key", "metadata.price") == 1
    assert "cli: --key takes --type number or --type text, or --drop" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        await _run("index", "--schema", name, "--current", "--key", "metadata.price")


async def _history(memory: Memory, versions: int, width: int = 1) -> None:
    await memory.set_history("acme", True)
    for n in range(versions):
        await memory.sync(
            "acme",
            filters={"metadata.doc": "pump"},
            sources=[
                SourceIn(
                    name=f"pump {i}",
                    text=f"Replace the pump seal {i}, revision {n}.",
                    metadata={"doc": "pump"},
                )
                for i in range(width)
            ],
        )


async def _bulk(pool: Any, chunks: int, stories: int) -> None:
    async with pool.acquire() as conn:
        for table, id_column, ended, copies in (
            ("memory_chunks", "chunk_id", "retired_at", chunks),
            ("memory_stories", "story_id", "closed_at", stories),
        ):
            columns = [
                r["column_name"]
                for r in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema = "
                    "current_schema() AND table_name = $1 AND is_generated = 'NEVER' "
                    "ORDER BY ordinal_position",
                    table,
                )
            ]
            chosen = ", ".join(f"{c} || '-copy-' || g" if c == id_column else c for c in columns)
            await conn.execute(
                f"INSERT INTO {table} ({', '.join(columns)}) SELECT {chosen} FROM {table} "
                f"CROSS JOIN generate_series(1, $1) AS g WHERE {ended} IS NOT NULL",
                copies,
            )


async def test_a_present_keyword_read_of_a_history_heavy_corpus_reads_both_partial_indexes(
    schemas: Schemas,
) -> None:
    name, pool, memory = await _schema(schemas)
    await _history(memory, 3, width=100)
    for n in range(5):
        told = await memory.remember("acme", f"Plan {n} for the pump seal.")
        await memory.supersede("acme", told.story.story_id, f"Plan {n} for the pump seal, again.")
    await _bulk(pool, chunks=40, stories=2000)
    assert await _run("index", "--schema", name, "--current", "--yes") == 0

    async def plan(kind: Any, history: bool, generic: bool) -> str:
        window = _window(None, None, None, history, now=datetime.now(UTC))
        params: list[Any] = [["acme"], "pump seal"]
        moment = None
        if window.now:
            params.append(window.as_of)
            moment = f"${len(params)}"
        where = _scope_sql(kind, None, window, None, None, params)
        params.append(10)
        sql = _text_sql(BM25(), "any", moment, kind, where, f"${len(params)}")
        async with pool.acquire() as conn, conn.transaction():
            if generic:
                await conn.execute("SET LOCAL plan_cache_mode = force_generic_plan")
            rows = await conn.fetch(f"EXPLAIN {sql}", *params)
        return "\n".join(r[0] for r in rows)

    for kind, table in (("chunk", "chunks"), ("story", "stories")):
        present = await plan(kind, False, False)
        assert f"ix_memory_current_{table}_fts" in present, present
        assert f"ix_memory_current_{table}_ended" in present, present
        assert f"ix_memory_{table}_fts" not in present, present
        assert f"ix_memory_current_{table}_fts" in await plan(kind, False, True)
        assert "ix_memory_current_" not in await plan(kind, True, False)


async def test_reads_of_the_present_keep_their_answers_with_the_current_rows_indexes(
    schemas: Schemas,
) -> None:
    name, pool, memory = await _schema(schemas)
    await _history(memory, 5)
    told = await memory.remember("acme", "Ana owns the pump seal plan.", metadata={"team": "a"})
    later = datetime.now(UTC) + timedelta(days=3)
    await memory.supersede(
        "acme", told.story.story_id, "Ben owns the pump seal plan.", valid_from=later
    )
    await memory.remember("acme", "The seal kit is in stock.", metadata={"team": "b"})
    frozen = datetime.now(UTC) + timedelta(days=30)
    still = Memory(memory.storage, embedding=HashEmbedding(), clock=lambda: frozen)
    await still.set_history("lag", True)
    kept = await still.ingest(
        "lag", name="valve", text="The valve seal, first.", metadata={"doc": "v"}
    )
    await still.sync(
        "lag",
        filters={"metadata.doc": "v"},
        sources=[SourceIn(name="valve", text="The valve seal, second.", metadata={"doc": "v"})],
    )

    async def answers() -> list[Any]:
        found: list[Any] = []
        for filters in (None, {"metadata.doc": "pump"}, {"metadata.team": "b"}):
            for history in (False, True):
                hits = await memory.search(
                    "acme",
                    "pump seal",
                    filters=filters,
                    history=history,
                    pillars={"text"},
                    top_k=50,
                )
                found.append(sorted((h.kind, h.id, round(h.score, 9)) for h in hits.hits))
        lagging = await still.search("lag", "valve seal", pillars={"text"}, top_k=50)
        found.append(sorted(h.id for h in lagging.hits))
        return found

    before = await answers()
    assert any(told.story.story_id == i for _, i, _ in before[0])
    assert before[-1] == [f"{kept.source.source_id}-0"]
    assert await _run("index", "--schema", name, "--current", "--yes") == 0
    await memory.storage.check(dimension=8, graph=True)
    assert memory.storage._current
    assert await answers() == before


async def test_a_reopened_or_revised_story_reads_through_the_current_rows_indexes_as_without(
    schemas: Schemas,
) -> None:
    name, pool, memory = await _schema(schemas)
    assert await _run("index", "--schema", name, "--current", "--yes") == 0
    await memory.storage.check(dimension=8, graph=True)
    plain = copy.copy(memory.storage)
    plain._current = False
    without = Memory(plain, embedding=HashEmbedding())
    now = datetime.now(UTC)
    told = await memory.remember(
        "acme", "Ana owns the pump seal plan.", valid_from=now - timedelta(days=2)
    )
    later = await memory.supersede(
        "acme",
        told.story.story_id,
        "Ben owns the pump seal plan.",
        valid_from=now + timedelta(days=3),
    )

    async def found() -> list[str]:
        answers = []
        for reader in (memory, without):
            hits = await reader.search("acme", "pump seal plan", pillars={"text"}, top_k=50)
            answers.append(sorted(h.id for h in hits.hits))
        assert answers[0] == answers[1]
        return answers[0]

    assert await found() == [told.story.story_id]
    await memory.end_story("acme", later.story.story_id, valid_to=later.story.valid_from)
    assert await found() == [told.story.story_id]
    await memory.end_story("acme", told.story.story_id, valid_to=now + timedelta(days=5))
    assert await found() == [told.story.story_id]
    await memory.end_story("acme", told.story.story_id, valid_to=now - timedelta(days=1))
    assert await found() == []
    await memory.end_story("acme", told.story.story_id, valid_to=now + timedelta(days=1))
    assert await found() == [told.story.story_id]
