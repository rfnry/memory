import re
from datetime import UTC, datetime, timedelta

import pytest

from rfnry_memory.errors import InputError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding

JAN, FEB, MAR = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3))


async def _memory(storage: PostgresStorage, history: bool = True) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", history)
    return memory


async def test_a_document_retires_in_favour_of_one_already_filed(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    old = (await memory.ingest("acme", name="Returns 2025", text="Returns within 30 days.")).source
    new = (await memory.ingest("acme", name="Returns 2026", text="Returns within 60 days.")).source
    retired = await memory.retire_source(
        "acme", old.source_id, replaced_by=new.source_id, written_by={"user": "ana"}
    )
    assert retired.retired_at is not None and retired.retired_at > old.created_at
    assert (retired.replaced_by, retired.closed_by) == (new.source_id, {"user": "ana"})
    assert retired.valid_to == retired.retired_at
    assert await memory.get_source("acme", old.source_id) == retired
    assert [c.retired_at for c in await storage.source_chunks("acme", old.source_id)] == [
        retired.retired_at
    ]
    current, _ = await memory.list_sources("acme")
    assert [s.source_id for s in current] == [new.source_id]
    every, _ = await memory.list_sources("acme", history=True)
    assert {s.source_id for s in every} == {old.source_id, new.source_id}
    then, _ = await memory.list_sources("acme", as_of=retired.retired_at - timedelta(seconds=1))
    assert {s.source_id for s in then} == {old.source_id, new.source_id}
    hits = await memory.search("acme", "returns within 30 days", kinds=("chunk",))
    assert old.source_id not in {h.chunk.source_id for h in hits.hits if h.chunk}
    alone = await memory.retire_source("acme", new.source_id)
    assert (alone.replaced_by, alone.closed_by) == (None, {})
    await memory.forget_retired("acme", before=datetime.now(UTC) + timedelta(days=1))
    assert await memory.get_source("acme", old.source_id) is None


async def test_a_retired_document_ends_where_the_rules_of_a_replace_say(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    now = datetime.now(UTC)

    async def filed(name: str, **validity: datetime) -> str:
        return (await memory.ingest("acme", name=name, text=name, **validity)).source.source_id  # type: ignore[arg-type]

    successor = await filed("successor", valid_from=FEB)
    ended = await memory.retire_source(
        "acme", await filed("old", valid_from=JAN), replaced_by=successor
    )
    assert ended.valid_to == FEB
    given = await memory.retire_source("acme", await filed("given", valid_from=JAN), valid_to=MAR)
    assert given.valid_to == MAR
    naive = await memory.retire_source(
        "acme", await filed("naive", valid_from=JAN), valid_to=datetime(2026, 3, 1)
    )
    assert naive.valid_to == MAR
    never = await memory.retire_source("acme", await filed("never", valid_from=JAN), valid_to=JAN)
    assert never.valid_to == never.valid_from == JAN
    planned = now + timedelta(days=30)
    withdrawn = await memory.retire_source("acme", await filed("planned", valid_from=planned))
    assert withdrawn.valid_to == withdrawn.valid_from == planned
    lapsed = await memory.retire_source("acme", await filed("lapsed", valid_from=JAN, valid_to=FEB))
    assert lapsed.valid_to == FEB


async def test_what_retire_source_refuses(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    now = datetime.now(UTC)

    async def filed(name: str, **validity: datetime) -> str:
        return (await memory.ingest("acme", name=name, text=name, **validity)).source.source_id  # type: ignore[arg-type]

    old, new = await filed("old", valid_from=FEB), await filed("new")
    gone = await memory.retire_source("acme", await filed("gone"))
    assert gone.retired_at is not None
    refused: list[tuple[str, dict[str, object], str]] = [
        (
            gone.source_id,
            {},
            f"memory: source {gone.source_id!r} was already retired at {gone.retired_at.isoformat()}",
        ),
        (old, {"replaced_by": old}, f"memory: source {old!r} cannot replace itself"),
        (
            old,
            {"replaced_by": gone.source_id},
            f"memory: source {gone.source_id!r} was retired at {gone.retired_at.isoformat()}; "
            "name the version in force",
        ),
        (
            old,
            {"valid_to": now + timedelta(days=1)},
            f"memory: source {old!r} cannot end after this write; a document's planned end is "
            "written with it",
        ),
        (
            old,
            {"valid_to": JAN},
            f"memory: source {old!r} cannot end at {JAN.isoformat()}, before it took effect, "
            f"at {FEB.isoformat()}",
        ),
        (
            old,
            {"replaced_by": await filed("early", valid_from=JAN)},
            f"memory: the source replacing {old!r} took effect at {JAN.isoformat()}, before it "
            f"did, at {FEB.isoformat()}; give valid_to",
        ),
        (
            old,
            {"replaced_by": await filed("later", valid_from=now + timedelta(days=30))},
            f"memory: the source replacing {old!r} takes effect after this write; retire {old!r} "
            "on the day it does",
        ),
    ]
    for source_id, options, message in refused:
        with pytest.raises(InputError, match=f"^{re.escape(message)}$"):
            await memory.retire_source("acme", source_id, **options)  # type: ignore[arg-type]
    with pytest.raises(NotFoundError, match=r"^memory: source 'missing' was not found$"):
        await memory.retire_source("acme", "missing")
    with pytest.raises(NotFoundError, match=r"^memory: source 'missing' was not found$"):
        await memory.retire_source("acme", old, replaced_by="missing")
    await memory.set_history("acme", False)
    with pytest.raises(
        InputError,
        match=re.escape(
            "memory: corpus 'acme' keeps no history, so a retired document would be erased; "
            "delete_source erases it"
        ),
    ):
        await memory.retire_source("acme", new)
    assert (await memory.get_source("acme", new)) is not None


async def test_a_retirement_comes_after_the_write_it_closes_whatever_the_clock_says(
    storage: PostgresStorage,
) -> None:
    clock = [MAR]
    memory = Memory(storage, embedding=HashEmbedding(), clock=lambda: clock[0])
    await memory.check()
    await memory.set_history("acme", True)
    filed = (await memory.ingest("acme", name="Returns", text="Returns within 30 days.")).source
    assert filed.created_at == MAR
    clock[0] = JAN
    retired = await memory.retire_source("acme", filed.source_id)
    assert retired.retired_at == MAR + timedelta(microseconds=1)
    assert retired.valid_to == MAR + timedelta(microseconds=1)


async def test_the_retire_source_tool_retires_inside_the_view_as_the_host_declares(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    librarian = {"agent": "librarian"}
    tools = MemoryTools(
        memory, corpus_id="acme", filters={"metadata.line": "A"}, written_by=librarian
    )
    definitions = {d["name"]: d for d in tools.definitions()}
    retire = definitions["retire_source"]
    assert set(retire["input_schema"]["properties"]) == {"source_id", "replaced_by", "valid_to"}
    assert "ingest with replaces" in retire["description"]
    assert "resembling" in definitions["ingest"]["description"]
    old = await tools.dispatch("ingest", {"name": "Returns", "text": "Returns within 30 days."})
    new = await tools.dispatch("ingest", {"name": "Returns", "text": "Returns within 60 days."})
    assert isinstance(old, dict) and isinstance(new, dict)
    outside = (
        await memory.ingest("acme", name="B", text="Line B returns.", metadata={"line": "B"})
    ).source
    for options in (
        {"source_id": outside.source_id},
        {"source_id": new["source"]["source_id"], "replaced_by": outside.source_id},
    ):
        with pytest.raises(
            NotFoundError, match=rf"^tools: source '{outside.source_id}' was not found$"
        ):
            await tools.dispatch("retire_source", options)
    retired = await tools.dispatch(
        "retire_source",
        {"source_id": old["source"]["source_id"], "replaced_by": new["source"]["source_id"]},
    )
    assert isinstance(retired, dict) and "text" not in retired["source"]
    row = retired["source"]
    assert (row["replaced_by"], row["closed_by"]) == (new["source"]["source_id"], librarian)
    assert row["retired_at"] is not None
