from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Chunk, EntityIn, Source, SourceIn, Synced
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.rows import known_chunk, known_source

from .fakes import HashEmbedding


def _on(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def _version(
    source_id: str,
    text: str,
    created: int,
    retired_at: datetime | None = None,
    replaced_by: str | None = None,
) -> Source:
    return Source(
        corpus_id="a",
        source_id=source_id,
        name="Returns policy",
        metadata={},
        text=text,
        created_at=_on(created),
        chunks=1,
        retired_at=retired_at,
        replaced_by=replaced_by,
    )


def test_a_source_shows_itself_retired_only_once_it_was() -> None:
    old = _version("v1", "30 days", 1, retired_at=_on(10), replaced_by="v2")
    then = known_source(old, _on(5))
    assert (then.retired_at, then.replaced_by) == (None, None)
    assert known_source(old, None) == old and known_source(old, _on(10)) == old


def test_a_chunk_shows_its_source_retired_only_once_it_was() -> None:
    chunk = Chunk(
        corpus_id="a",
        chunk_id="v1-0",
        source_id="v1",
        source_name="Returns policy",
        position=0,
        title=None,
        locator={},
        text="30 days",
        metadata={},
        retired_at=_on(10),
    )
    assert known_chunk(chunk, _on(5)).retired_at is None
    assert known_chunk(chunk, _on(10)) == chunk and known_chunk(chunk, None) == chunk
    current = chunk.model_copy(update={"retired_at": None})
    assert known_chunk(current, _on(5)) == current and known_chunk(current, None) == current


RETURNS = {"metadata.doc": "returns"}


def _policy(text: str) -> SourceIn:
    return SourceIn(name="Returns policy", text=text, metadata={"doc": "returns"})


async def test_sync_retires_under_history_and_hands_over_at_one_moment(
    storage: PostgresStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    with pytest.raises(InputError, match=r"^memory: keep is True or False$"):
        await memory.set_history("acme", "yes")  # type: ignore[arg-type]
    await memory.set_history("acme", True)
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(10))
    first = await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 30 days.")])
    assert first.retired == [] and first.sources[0].created_at == _on(10)
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(20))
    second = await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 60 days.")])
    old, new = first.sources[0], second.sources[0]
    assert second.retired == [old.source_id] and second.deleted.clean
    retired = await memory.get_source("acme", old.source_id)
    assert retired is not None and (retired.retired_at, retired.replaced_by) == (
        _on(20),
        new.source_id,
    )
    assert new.created_at == _on(20)
    at_handover, _ = await memory.list_sources("acme", known_at=_on(20))
    before, _ = await memory.list_sources("acme", known_at=datetime(2026, 1, 15))
    every, _ = await memory.list_sources("acme", history=True)
    assert [s.source_id for s in at_handover] == [new.source_id]
    assert [s.source_id for s in before] == [old.source_id] and before[0].retired_at is None
    assert {s.source_id for s in every} == {old.source_id, new.source_id}
    assert [s.source_id for s in (await memory.list_sources("acme"))[0]] == [new.source_id]
    again = await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 30 days.")])
    assert again.kept == [] and again.sources[0].source_id != old.source_id


async def test_a_sync_overtaken_by_a_newer_one_takes_a_later_moment(
    storage: PostgresStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", True)
    clock = [_on(5)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    first = await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 30 days.")])
    original = memory.storage.sync_sources
    racing: list[Synced] = []
    entered = [False]

    async def overtaken(*args: Any, **kwargs: Any) -> Any:
        if not entered[0]:
            entered[0] = True
            clock[0] = _on(15)
            memory.storage.sync_sources = original  # type: ignore[method-assign]
            racing.append(
                await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 60 days.")])
            )
            memory.storage.sync_sources = overtaken  # type: ignore[method-assign]
            clock[0] = _on(20)
        return await original(*args, **kwargs)

    clock[0] = _on(10)
    memory.storage.sync_sources = overtaken  # type: ignore[method-assign]
    a = await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 90 days.")])
    v1, b, mine = first.sources[0], racing[0].sources[0], a.sources[0]
    assert (v1.created_at, b.created_at, mine.created_at) == (_on(5), _on(15), _on(20))
    stored_mine = await memory.get_source("acme", mine.source_id)
    assert stored_mine is not None and (stored_mine.retired_at, stored_mine.replaced_by) == (
        None,
        None,
    )
    stored_b, stored_v1 = (
        await memory.get_source("acme", b.source_id),
        await memory.get_source("acme", v1.source_id),
    )
    assert stored_b is not None and (stored_b.retired_at, stored_b.replaced_by) == (
        _on(20),
        mine.source_id,
    )
    assert stored_v1 is not None and (stored_v1.retired_at, stored_v1.replaced_by) == (
        _on(15),
        b.source_id,
    )

    async def at(day: int) -> list[str]:
        rows, _ = await memory.list_sources("acme", known_at=_on(day))
        return [s.source_id for s in rows]

    assert await at(12) == [v1.source_id]
    assert await at(15) == [b.source_id]
    assert await at(20) == [mine.source_id]
    every, _ = await memory.list_sources("acme", history=True)
    assert all(s.retired_at is None or s.created_at <= s.retired_at for s in every)


@pytest.mark.parametrize("history", [False, True])
async def test_a_sync_behind_the_clock_of_the_set_stamps_after_it_only_where_it_keeps_history(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch, history: bool
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", history)
    clock = [_on(20)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    first = await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 30 days.")])
    clock[0] = _on(10)
    second = await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 60 days.")])
    old, new = first.sources[0], second.sources[0]
    stored = await memory.get_source("acme", old.source_id)
    if history:
        assert new.created_at == _on(20) + timedelta(microseconds=1)
        assert second.retired == [old.source_id]
        assert stored is not None and (stored.retired_at, stored.replaced_by) == (
            new.created_at,
            new.source_id,
        )
        assert stored.retired_at is not None and stored.retired_at > stored.created_at
    else:
        assert new.created_at == _on(10)
        assert second.retired == [] and stored is None


async def test_forget_retired_erases_old_versions_and_settles_their_entities(
    storage: PostgresStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    await memory.set_history("acme", True)
    outlet = EntityIn(name="Outlet", type="store")
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(10))
    await memory.sync(
        "acme",
        filters=RETURNS,
        sources=[
            SourceIn(
                name="Returns policy",
                text="Covers the Outlet store.",
                metadata={"doc": "returns"},
                entities=[outlet],
            )
        ],
    )
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(20))
    await memory.sync("acme", filters=RETURNS, sources=[_policy("Covers the Main store.")])
    assert await memory.storage.entities_named("acme", "Outlet") != []
    forgot = await memory.forget_retired("acme", before=datetime(2026, 1, 25))
    assert (forgot.sources, forgot.entities) == (1, 1)
    assert await memory.storage.entities_named("acme", "Outlet") == []
    assert (await memory.list_sources("acme", history=True))[0][0].text == "Covers the Main store."


async def test_deleting_a_source_erases_every_version_that_led_to_it_branches_included(
    storage: PostgresStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", True)
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(10))
    first = await memory.sync(
        "acme", filters=RETURNS, sources=[_policy("Within 30 days."), _policy("Within 45 days.")]
    )
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(20))
    second = await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 60 days.")])
    newest = second.sources[0]
    assert sorted(second.retired) == sorted(s.source_id for s in first.sources)
    for old in first.sources:
        stored = await memory.get_source("acme", old.source_id)
        assert stored is not None and stored.replaced_by == newest.source_id
    deleted = await memory.delete_source("acme", newest.source_id)
    assert deleted.sources == 3
    assert (await memory.list_sources("acme", history=True))[0] == []


async def test_forget_retired_still_erases_once_history_is_turned_off(
    storage: PostgresStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", True)
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(10))
    first = await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 30 days.")])
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: _on(20))
    second = await memory.sync("acme", filters=RETURNS, sources=[_policy("Within 60 days.")])
    await memory.set_history("acme", False)
    forgot = await memory.forget_retired("acme", before=_on(25))
    assert forgot.sources == 1
    assert await memory.get_source("acme", first.sources[0].source_id) is None
    every, _ = await memory.list_sources("acme", history=True)
    assert [s.source_id for s in every] == [second.sources[0].source_id]
