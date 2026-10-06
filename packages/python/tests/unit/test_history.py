from datetime import UTC, datetime

from rfnry_memory.models import Chunk, Source
from rfnry_memory.storage.storage import known_chunk, known_source, source_seen


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
