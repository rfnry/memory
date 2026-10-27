import base64
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from database import Schemas, provision, run_cli
from unit.fakes import HashEmbedding

from rfnry_memory import memory as memory_module
from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn
from rfnry_memory.postgres import keys, migrations, sql as sql_module

TABLES = ("chunks", "sources", "stories")


async def _run(*argv: str) -> int:
    return await run_cli(*argv)


async def _schema(schemas: Schemas) -> tuple[str, Any, Memory]:
    name = await schemas.create()
    pool = await schemas.pool(name)
    storage = await provision(pool)
    return name, pool, Memory(storage, embedding=HashEmbedding())


async def _objects(pool: Any, key: str) -> dict[str, tuple[bool | None, bool | None]]:
    async with pool.acquire() as conn:
        return await keys._state(conn, keys.tag_of(key))


async def _held(pool: Any) -> list[tuple[str, str, bool]]:
    async with pool.acquire() as conn:
        return await keys.indexed_keys(conn)


async def test_index_builds_a_key_s_indexes_and_check_lists_it(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, memory = await _schema(schemas)
    await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": 120})
    await memory.remember("acme", "Ana owns the seal plan.", metadata={"price": 3.5})
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "  index metadata.price as a number on chunks, sources and stories\n" in out
    assert f"schema {name}: metadata.price is indexed as number" in out
    assert await _objects(pool, "metadata.price") == {table: (True, True) for table in TABLES}
    assert await _held(pool) == [("metadata.price", "number", True)]
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    assert f"schema {name}: nothing to do" in capsys.readouterr().out
    assert await _run("check", "--schema", name) == 0
    assert "\n  metadata.price: number\n" in capsys.readouterr().out


async def test_index_with_sql_prints_its_statements_and_builds_nothing(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, _ = await _schema(schemas)
    assert (
        await _run("index", "--schema", name, "--key", "metadata.rev", "--type", "text", "--sql")
        == 0
    )
    out = capsys.readouterr().out
    assert "CREATE INDEX CONCURRENTLY ix_memory_key_chunks_" in out
    assert "memory_key_text(metadata, ARRAY['rev']::text[]) COLLATE \"C\"" in out
    assert "memory_key_text(metadata::jsonb, ARRAY['rev']::text[])" in out
    assert await _objects(pool, "metadata.rev") == {table: (None, None) for table in TABLES}
    assert await _held(pool) == []


async def test_index_refuses_current_rows_that_break_the_rule_and_ignores_the_past(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, memory = await _schema(schemas)
    await memory.set_history("acme", True)
    old = await memory.ingest(
        "acme", name="pump", text="Pump seal v1.", metadata={"doc": "pump", "price": "120"}
    )
    await memory.sync(
        "acme",
        filters={"metadata.doc": "pump"},
        sources=[
            SourceIn(
                name="pump",
                text="Pump seal v2.",
                metadata={"doc": "pump", "price": 99},
                valid_from=datetime.now(UTC) - timedelta(days=1),
            )
        ],
    )
    bad = await memory.ingest("ops", name="valve", text="Valve.", metadata={"price": [1, 2]})
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 1
    )
    err = capsys.readouterr().err
    assert old.source.source_id not in err
    assert (
        f"cli: 1 current sources hold metadata.price as something other than a number — "
        f"corpus 'ops', source {bad.source.source_id!r}; fix them before indexing it"
    ) in err
    assert (
        "cli: 1 current chunks hold metadata.price as something other than a number — corpus 'ops'"
        in err
    )
    assert await _objects(pool, "metadata.price") == {table: (None, None) for table in TABLES}
    await memory.delete_source("ops", bad.source.source_id)
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    long = "x" * 2001
    await memory.ingest("acme", name="rev", text="Rev.", metadata={"rev": long})
    assert (
        await _run("index", "--schema", name, "--key", "metadata.rev", "--type", "text", "--yes")
        == 1
    )
    assert "as something other than a string of at most 2000 bytes" in capsys.readouterr().err


async def test_a_write_holding_another_type_at_an_indexed_key_is_refused(
    schemas: Schemas,
) -> None:
    name, pool, memory = await _schema(schemas)
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    assert (
        await _run("index", "--schema", name, "--key", "written_by.run", "--type", "text", "--yes")
        == 0
    )
    number = (
        "^postgres: metadata.price is indexed as a number, so a write holding another type or an array "
        "there is refused$"
    )
    for value in ("120", [1, 2], True, {"amount": 1}):
        with pytest.raises(InputError, match=number):
            await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": value})
        with pytest.raises(InputError, match=number):
            await memory.remember("acme", "Ana owns the seal plan.", metadata={"price": value})
    assert (await memory.list_sources("acme"))[0] == []
    text = (
        "^postgres: written_by.run is indexed as text, so a write holding another type, an array or a "
        "string over 2000 bytes there is refused$"
    )
    for run in (5, "r" * 2001):
        with pytest.raises(InputError, match=text):
            await memory.remember("acme", "Ana owns the seal plan.", written_by={"run": run})
    await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": 120})
    await memory.ingest("acme", name="valve", text="Valve.", metadata={"price": None})
    story = await memory.remember("acme", "Ana owns the seal plan.", written_by={"run": "r1"})
    await memory.end_story("acme", story.story.story_id)
    assert len((await memory.list_sources("acme"))[0]) == 2


async def test_a_story_held_in_force_again_must_fit_an_indexed_key(schemas: Schemas) -> None:
    name, pool, memory = await _schema(schemas)
    now = datetime.now(UTC)
    told = await memory.remember(
        "acme",
        "Ana owns the seal plan.",
        metadata={"price": "high"},
        valid_from=now - timedelta(days=10),
    )
    later = await memory.supersede(
        "acme",
        told.story.story_id,
        "Ben owns the seal plan.",
        metadata={"price": 1},
        valid_from=now - timedelta(days=5),
    )
    ended = await memory.remember(
        "acme", "The kit is late.", metadata={"price": [1, 2]}, valid_from=now - timedelta(days=10)
    )
    await memory.end_story("acme", ended.story.story_id, valid_to=now - timedelta(days=5))
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    for story_id, attempt in (
        (
            told.story.story_id,
            memory.end_story("acme", later.story.story_id, valid_to=later.story.valid_from),
        ),
        (
            ended.story.story_id,
            memory.end_story("acme", ended.story.story_id, valid_to=now + timedelta(days=5)),
        ),
    ):
        with pytest.raises(
            InputError,
            match=rf"^postgres: metadata.price is indexed as a number, so story '{story_id}', which holds another type or an array there, cannot be in force again$",
        ):
            await attempt
    [kept] = await memory.get_stories("acme", [told.story.story_id])
    assert kept.superseded_by == later.story.story_id
    revised = await memory.end_story("acme", ended.story.story_id, valid_to=now - timedelta(days=4))
    assert revised.revised_at is not None


async def test_a_version_held_in_force_again_must_fit_an_indexed_key(schemas: Schemas) -> None:
    name, pool, memory = await _schema(schemas)
    await memory.set_history("acme", True)
    now = datetime.now(UTC)
    first = await memory.ingest(
        "acme",
        name="pump",
        text="Pump seal.",
        metadata={"price": "high"},
        valid_from=now - timedelta(days=10),
    )
    second = await memory.ingest(
        "acme",
        name="pump",
        text="Pump seal, again.",
        metadata={"price": 1},
        valid_from=now - timedelta(days=5),
        replaces=first.source.source_id,
    )
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    with pytest.raises(
        InputError,
        match=rf"^postgres: metadata.price is indexed as a number, so source '{first.source.source_id}', which holds another type or an array there, cannot be in force again$",
    ):
        await memory.retire_source(
            "acme", second.source.source_id, valid_to=second.source.valid_from
        )
    held = await memory.get_source("acme", first.source.source_id)
    assert held is not None and held.replaced_by == second.source.source_id


async def test_drop_removes_the_indexes_and_lifts_the_rule(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, memory = await _schema(schemas)
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    assert await _run("index", "--schema", name, "--key", "metadata.price", "--drop", "--yes") == 0
    assert f"schema {name}: metadata.price is no longer indexed" in capsys.readouterr().out
    assert await _objects(pool, "metadata.price") == {table: (None, None) for table in TABLES}
    assert await _held(pool) == []
    await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": "120"})
    assert await _run("index", "--schema", name, "--key", "metadata.price", "--drop", "--yes") == 0
    assert f"schema {name}: nothing to do" in capsys.readouterr().out


async def test_an_interrupted_build_is_listed_not_ready_and_finished_by_running_again(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, memory = await _schema(schemas)
    async with pool.acquire() as conn:
        plan = await keys.plan_key(conn, "metadata.price", "number")
        for statement in plan.statements[:3]:
            await conn.execute(statement)
    assert await _held(pool) == [("metadata.price", "number", False)]
    assert await _run("check", "--schema", name) == 0
    assert (
        f"  metadata.price: number, not ready — run rfnry-memory index --key metadata.price "
        f"--type number --schema {name}"
    ) in capsys.readouterr().out
    with pytest.raises(InputError, match="^postgres: metadata.price is indexed as a number"):
        await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": "120"})
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    assert await _objects(pool, "metadata.price") == {table: (True, True) for table in TABLES}
    assert await _held(pool) == [("metadata.price", "number", True)]


async def test_index_refuses_a_bad_key_a_type_change_and_an_old_schema(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, _ = await _schema(schemas)
    assert await _run("index", "--schema", name, "--key", "price", "--type", "number") == 1
    assert (
        "postgres: an indexed key is a filter key, metadata.<key> or written_by.<key>; got 'price'"
        in capsys.readouterr().err
    )
    assert await _run("index", "--schema", name, "--key", "metadata.a..b", "--type", "number") == 1
    assert "postgres: 'metadata.a..b' names an empty part" in capsys.readouterr().err
    for position in ("metadata.tags.0", "written_by.runs.-1"):
        assert await _run("index", "--schema", name, "--key", position, "--type", "number") == 1
        assert (
            f"postgres: {position!r} names a position; the parts of an indexed key name object keys"
            in capsys.readouterr().err
        )
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    assert (
        await _run("index", "--schema", name, "--key", "metadata.price", "--type", "text", "--yes")
        == 1
    )
    assert (
        "postgres: metadata.price is indexed as number — drop it first with rfnry-memory index "
        "--key metadata.price --drop"
    ) in capsys.readouterr().err
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memory_migrations WHERE version = $1", migrations.LATEST)
    assert await _run("index", "--schema", name, "--key", "metadata.size", "--type", "number") == 1
    assert "run rfnry-memory upgrade first" in capsys.readouterr().err


async def test_a_write_refused_by_a_key_dropped_meanwhile_still_says_why(
    schemas: Schemas,
) -> None:
    _, pool, memory = await _schema(schemas)
    async with pool.acquire() as conn:
        plan = await keys.plan_key(conn, "metadata.price", "number")
        for statement in plan.statements[:3]:
            await conn.execute(statement)
        await conn.execute("DELETE FROM memory_indexed_keys")
    with pytest.raises(
        InputError,
        match=r"^postgres: a write holds a value an indexed key refuses \(ck_memory_key_",
    ):
        await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": "120"})


RANGES: list[dict[str, Any]] = [
    {"metadata.price": {"gte": 100}},
    {"metadata.price": {"gt": 10, "lte": 120}},
    {"metadata.price": {"lt": 0.5}},
    {"metadata.price": {"gte": "100"}},
    {"metadata.rev": {"gte": "a"}},
    {"metadata.rev": {"lt": "b", "gte": "B"}},
    {"metadata.rev": {"gte": 1}},
    {"written_by.run": {"gte": "r2"}},
]


async def _answers(memory: Memory) -> list[Any]:
    answers: list[Any] = []
    for filters in RANGES:
        searched = await memory.search("acme", "seal", filters=filters, top_k=50)
        vector = await memory.search("acme", "seal", filters=filters, pillars={"vector"}, top_k=50)
        sources, _ = await memory.list_sources("acme", filters=filters)
        stories, _ = await memory.list_stories("acme", filters=filters)
        answers.append(
            (
                sorted(h.id for h in searched.hits),
                sorted(h.id for h in vector.hits),
                sorted(s.source_id for s in sources),
                sorted(s.story_id for s in stories),
            )
        )
    return answers


async def test_a_range_on_an_indexed_key_answers_what_it_answered_before(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, _, memory = await _schema(schemas)
    values: list[tuple[Any, Any]] = [
        (120, "apple"),
        (99.5, "Banana"),
        (0.25, "cherry"),
        (None, "b"),
        (-3, None),
    ]
    for n, (price, rev) in enumerate(values):
        metadata = {
            k: v for k, v in (("price", price), ("rev", rev)) if v is not None or k == "price"
        }
        await memory.ingest("acme", name=f"seal {n}", text=f"Seal note {n}.", metadata=metadata)
        await memory.remember(
            "acme", f"Seal story {n}.", metadata=metadata, written_by={"run": f"r{n}"}
        )
    await memory.ingest("acme", name="bare", text="Seal without metadata.")
    before = await _answers(memory)
    assert any(any(found) for found in before)
    for key, kind in (
        ("metadata.price", "number"),
        ("metadata.rev", "text"),
        ("written_by.run", "text"),
    ):
        assert await _run("index", "--schema", name, "--key", key, "--type", kind, "--yes") == 0
    typed: list[str] = []
    indexed_sql = sql_module._indexed_sql

    def spy(target: str, path: str, indexed: keys.IndexedRange, params: list[Any]) -> str:
        typed.append(target)
        return indexed_sql(target, path, indexed, params)

    monkeypatch.setattr(sql_module, "_indexed_sql", spy)
    assert await _answers(memory) == before
    assert before[3] == ([], [], [], []) and before[6] == ([], [], [], [])
    assert {"memory_chunks.metadata", "metadata::jsonb", "written_by"} <= set(typed)


async def test_a_selective_range_on_an_indexed_key_reads_its_index(schemas: Schemas) -> None:
    name, pool, memory = await _schema(schemas)
    plain = Memory(memory.storage, pillars={"text"})
    parts = [
        SourceIn(name=f"part {n}", text=f"Part {n}.", metadata={"set": "parts", "price": n})
        for n in range(400)
    ]
    await plain.sync("acme", filters={"metadata.set": "parts"}, sources=parts)
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    async with pool.acquire() as conn:
        filters = await keys.keyed(
            conn, {"metadata.price": {"gte": 397}}, datetime.now(UTC) + timedelta(minutes=1)
        )
        params: list[Any] = ["acme"]
        clause = sql_module._facet_sql(filters, params, cast="::jsonb")
        plan = "\n".join(
            r[0]
            for r in await conn.fetch(
                f"EXPLAIN SELECT source_id FROM memory_sources WHERE corpus_id = $1{clause}",
                *params,
            )
        )
    assert "ix_memory_key_sources_" in plan
    listed, _ = await memory.list_sources("acme", filters={"metadata.price": {"gte": 397}})
    assert sorted(s.name for s in listed) == ["part 397", "part 398", "part 399"]


async def test_an_indexed_key_never_reads_another_tenant_s_rows(schemas: Schemas) -> None:
    name, _, memory = await _schema(schemas)
    for corpus in ("tenant-a", "tenant-b"):
        await memory.ingest(corpus, name="pump", text="Pump seal.", metadata={"price": 120})
        await memory.remember(corpus, "Seal story.", metadata={"price": 120})
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    ranged = {"metadata.price": {"gte": 100}}
    hits = await memory.search("tenant-b", "seal", filters=ranged)
    assert hits.hits and {h.corpus_id for h in hits.hits} == {"tenant-b"}
    sources, _ = await memory.list_sources("tenant-b", filters=ranged)
    assert [s.corpus_id for s in sources] == ["tenant-b"]
    await memory.delete_corpus("tenant-a")
    assert (await memory.verify_absent("tenant-a")).clean
    assert len((await memory.list_sources("tenant-b", filters=ranged))[0]) == 1


async def test_a_read_of_the_past_keeps_the_json_path_test(schemas: Schemas) -> None:
    name, pool, memory = await _schema(schemas)
    await memory.set_history("acme", True)
    await memory.ingest(
        "acme", name="pump", text="Pump seal v1.", metadata={"doc": "pump", "price": [100, 1]}
    )
    await memory.sync(
        "acme",
        filters={"metadata.doc": "pump"},
        sources=[
            SourceIn(
                name="pump",
                text="Pump seal v2.",
                metadata={"doc": "pump", "price": 5},
                valid_from=memory_module._now() - timedelta(days=1),
            )
        ],
    )
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    ranged = {"metadata.price": {"gte": 100}}
    assert (await memory.list_sources("acme", filters=ranged))[0] == []
    past, _ = await memory.list_sources("acme", filters=ranged, history=True)
    assert [s.text for s in past] == ["Pump seal v1."]
    async with pool.acquire() as conn:
        assert await keys.keyed(conn, ranged, None) == ranged
        assert await keys.keyed(conn, ranged, datetime(2020, 1, 1, tzinfo=UTC)) == ranged
        assert await keys.keyed(conn, ranged, datetime.now(UTC) + timedelta(minutes=1)) == {
            "metadata.price": keys.IndexedRange("number", {"gte": 100})
        }


async def test_listing_and_resembling_read_the_present_through_the_index(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, _, memory = await _schema(schemas)
    await memory.ingest(
        "acme", name="pump", text="Replace the pump seal yearly.", metadata={"price": 120}
    )
    await memory.ingest(
        "acme", name="valve", text="Replace the valve seal yearly.", metadata={"price": 5}
    )
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    typed: list[str] = []
    indexed_sql = sql_module._indexed_sql

    def spy(target: str, path: str, indexed: keys.IndexedRange, params: list[Any]) -> str:
        typed.append(target)
        return indexed_sql(target, path, indexed, params)

    monkeypatch.setattr(sql_module, "_indexed_sql", spy)
    ranged = {"metadata.price": {"gte": 100}}
    listed, _ = await memory.list_sources("acme", filters=ranged)
    assert [s.name for s in listed] == ["pump"] and typed == ["metadata::jsonb"]
    found = await memory.resembling("acme", text="Replace the pump seal yearly.", filters=ranged)
    assert [m.source.name for m in found.matches] == ["pump"] and len(typed) == 2
    await memory.list_sources("acme", filters=ranged, history=True)
    assert len(typed) == 2


async def test_a_text_key_indexes_past_a_long_string_kept_in_history(schemas: Schemas) -> None:
    name, _, memory = await _schema(schemas)
    await memory.set_history("acme", True)
    long = "z" + base64.b64encode(os.urandom(3000)).decode()
    await memory.ingest("acme", name="spec", text="Spec v1.", metadata={"doc": "spec", "rev": long})
    await memory.sync(
        "acme",
        filters={"metadata.doc": "spec"},
        sources=[
            SourceIn(
                name="spec",
                text="Spec v2.",
                metadata={"doc": "spec", "rev": "b"},
                valid_from=memory_module._now() - timedelta(days=1),
            )
        ],
    )
    assert (
        await _run("index", "--schema", name, "--key", "metadata.rev", "--type", "text", "--yes")
        == 0
    )
    listed, _ = await memory.list_sources("acme", filters={"metadata.rev": {"gte": "a"}})
    assert [s.text for s in listed] == ["Spec v2."]
    past, _ = await memory.list_sources(
        "acme", filters={"metadata.rev": {"gte": "A"}}, history=True
    )
    assert sorted(s.text for s in past) == ["Spec v1.", "Spec v2."]


async def test_index_counts_a_story_superseded_in_advance_that_is_still_in_force(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, _, memory = await _schema(schemas)
    told = await memory.remember("acme", "Ana owns the seal plan.", metadata={"price": [1, 2]})
    later = datetime.now(UTC) + timedelta(days=3)
    await memory.supersede(
        "acme", told.story.story_id, "Ben owns the seal plan.", valid_from=later, metadata={}
    )
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 1
    )
    assert (
        f"cli: 1 closed stories still in force hold metadata.price as something other than a number "
        f"— corpus 'acme', story {told.story.story_id!r}; they keep it while in force; index it "
        "once they end, or delete them"
    ) in capsys.readouterr().err


async def test_a_version_replaced_before_it_began_never_blocks_an_index(schemas: Schemas) -> None:
    name, _, memory = await _schema(schemas)
    await memory.set_history("acme", True)
    start = datetime.now(UTC) + timedelta(days=10)
    await memory.ingest(
        "acme",
        name="spec",
        text="Spec, planned.",
        metadata={"doc": "s", "price": [1, 2]},
        valid_from=start,
    )
    await memory.sync(
        "acme",
        filters={"metadata.doc": "s"},
        sources=[SourceIn(name="spec", text="Spec, now.", metadata={"doc": "s", "price": 5})],
    )
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )


async def test_a_read_at_a_moment_before_the_index_was_built_keeps_the_json_path_test(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, _, memory = await _schema(schemas)
    await memory.ingest("acme", name="pump", text="Pump seal.", metadata={"price": 120})
    assert (
        await _run(
            "index", "--schema", name, "--key", "metadata.price", "--type", "number", "--yes"
        )
        == 0
    )
    typed: list[str] = []
    indexed_sql = sql_module._indexed_sql

    def spy(target: str, path: str, indexed: keys.IndexedRange, params: list[Any]) -> str:
        typed.append(target)
        return indexed_sql(target, path, indexed, params)

    monkeypatch.setattr(sql_module, "_indexed_sql", spy)
    past = Memory(
        memory.storage, embedding=HashEmbedding(), clock=lambda: datetime(2020, 1, 1, tzinfo=UTC)
    )
    ranged = {"metadata.price": {"gte": 100}}
    await past.list_sources("acme", filters=ranged)
    assert typed == []
    await memory.list_sources("acme", filters=ranged)
    assert typed == ["metadata::jsonb"]
