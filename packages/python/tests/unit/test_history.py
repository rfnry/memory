from datetime import UTC, datetime

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Chunk, Source, SourceIn
from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.storage.storage import known_chunk, known_source, source_seen

from .fakes import HashEmbedding


def _on(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def _version(source_id: str, text: str, created: int, **kw: object) -> Source:
    return Source(
        corpus_id="a",
        source_id=source_id,
        name="Returns policy",
        metadata={},
        text=text,
        created_at=_on(created),
        chunks=1,
        **kw,  # type: ignore[arg-type]
    )


def test_a_source_is_read_current_as_known_then_or_in_every_version() -> None:
    old = _version("v1", "30 days", 1, retired_at=_on(10), replaced_by="v2")
    new = _version("v2", "60 days", 10)
    assert source_seen(old, known_at=None, history=False) is None
    assert source_seen(new, known_at=None, history=False) == new
    assert source_seen(old, known_at=None, history=True) == old
    then = source_seen(old, known_at=_on(5), history=False)
    assert then is not None and (then.retired_at, then.replaced_by) == (None, None)
    assert source_seen(new, known_at=_on(5), history=False) is None
    assert source_seen(new, known_at=_on(5), history=True) is None
    assert source_seen(old, known_at=_on(20), history=True) == old
    assert known_source(old, None) == old and known_source(old, _on(10)) == old


def test_versions_hand_over_at_one_moment() -> None:
    old = _version("v1", "30 days", 1, retired_at=_on(10), replaced_by="v2")
    new = _version("v2", "60 days", 10)
    seen = [source_seen(s, known_at=_on(10), history=False) for s in (old, new)]
    assert seen == [None, new]


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


RETURNS = {"metadata.doc": "returns"}


def _policy(text: str) -> SourceIn:
    return SourceIn(name="Returns policy", text=text, metadata={"doc": "returns"})


async def test_sync_retires_under_history_and_hands_over_at_one_moment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = Memory(InMemoryStorage(), embedding=HashEmbedding())
    await memory.initialize()
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
