from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import EntityIn, Section, SourceIn
from rfnry_memory.postgres import PostgresStorage

from .fakes import HashEmbedding


class Clock:
    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at


def _sequence(prefix: str) -> Callable[[], str]:
    numbers = count(1)
    return lambda: f"{prefix}{next(numbers)}"


async def test_every_stamp_is_the_clock_the_memory_is_given(storage: PostgresStorage) -> None:
    clock = Clock(datetime(2026, 1, 10))
    memory = Memory(storage, embedding=HashEmbedding(), clock=clock)
    await memory.check()
    jan10 = datetime(2026, 1, 10, tzinfo=UTC)
    told = (await memory.remember("acme", "Ana owns the review.")).story
    assert told.valid_from == told.created_at == jan10
    source = (await memory.ingest("acme", name="manual", text="Ground first.")).source
    assert source.created_at == source.confirmed_at == jan10
    assert (await memory.list_stories("acme"))[0] == [told]
    clock.at = datetime(2026, 2, 1, tzinfo=UTC)
    later = (await memory.supersede("acme", told.story_id, "Bo owns the review.")).story
    [closed] = await memory.get_stories("acme", [told.story_id])
    assert later.created_at == later.valid_from == closed.closed_at == closed.valid_to == clock.at
    clock.at = datetime(2026, 1, 1, tzinfo=UTC)
    feb5 = datetime(2026, 2, 5, tzinfo=UTC)
    last = (
        await memory.supersede("acme", later.story_id, "Cy owns the review.", valid_from=feb5)
    ).story
    [behind] = await memory.get_stories("acme", [later.story_id])
    assert last.created_at == behind.closed_at == later.created_at + timedelta(microseconds=1)
    assert (await memory.list_stories("acme", as_of=feb5))[0] == [last]


async def test_a_clock_that_returns_no_datetime_is_refused(storage: PostgresStorage) -> None:
    memory = Memory(storage, clock=lambda: "now")  # type: ignore[arg-type,return-value]
    with pytest.raises(
        ConfigurationError, match=r"^memory: the clock returned 'now', not a datetime$"
    ):
        await memory.remember("acme", "Ana owns the review.")


async def test_every_new_record_is_named_by_the_ids_the_memory_is_given(
    storage: PostgresStorage,
) -> None:
    memory = Memory(
        storage,
        embedding=HashEmbedding(),
        pillars={"text", "vector", "graph"},
        ids=_sequence("n"),
    )
    await memory.check()
    story = (
        await memory.remember(
            "acme",
            "Ana owns P-3.",
            entities=[EntityIn(name="Ana", type="person"), EntityIn(name="P-3", type="pump")],
        )
    ).story
    source = (
        await memory.ingest("acme", name="m", sections=[Section(text="a"), Section(text="b")])
    ).source
    entities = {
        e.entity_id
        for name in ("Ana", "P-3")
        for e in (await memory.neighbors("acme", name)).entities
    }
    named = {story.story_id, source.source_id, *entities}
    assert named == {"n1", "n2", "n3", "n4"}
    chunks = await storage.source_chunks("acme", source.source_id)
    assert [c.chunk_id for c in chunks] == [f"{source.source_id}-0", f"{source.source_id}-1"]


async def test_an_id_the_corpus_holds_is_refused_and_nothing_is_written(
    storage: PostgresStorage,
) -> None:
    memory = Memory(
        storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"}, ids=lambda: "same"
    )
    await memory.check()
    told = (await memory.remember("acme", "Ana owns the review.")).story
    with pytest.raises(
        InputError, match=r"^postgres: story 'same' is already held; ids must be unique$"
    ):
        await memory.remember("acme", "Bo owns the review.")
    with pytest.raises(
        InputError, match=r"^postgres: story 'same' is already held; ids must be unique$"
    ):
        await memory.supersede("acme", told.story_id, "Bo owns the review.")
    await memory.ingest("acme", name="manual", text="Ground first.")
    with pytest.raises(
        InputError, match=r"^postgres: source 'same' is already held; ids must be unique$"
    ):
        await memory.ingest("acme", name="notes", text="Check the seal.")
    with pytest.raises(InputError, match=r"entity 'same' is already held; ids must be unique$"):
        await memory.ingest(
            "other",
            name="sheet",
            text="P-3 and V-101.",
            entities=[EntityIn(name="P-3", type="pump"), EntityIn(name="V-101", type="valve")],
        )
    assert [s.text for s in (await memory.list_stories("acme"))[0]] == ["Ana owns the review."]
    assert [s.name for s in (await memory.list_sources("acme"))[0]] == ["manual"]
    assert (await memory.list_sources("other"))[0] == []


@pytest.mark.parametrize("made", ["", "a\x00b", 3])
async def test_an_id_that_is_no_name_is_refused(storage: PostgresStorage, made: object) -> None:
    memory = Memory(storage, ids=lambda: made)  # type: ignore[arg-type,return-value]
    with pytest.raises(InputError, match=r"^memory: the ids source returned .+, not a name$"):
        await memory.remember("acme", "Ana owns the review.")
    assert (await memory.list_stories("acme"))[0] == []


async def test_a_clock_that_does_not_advance_never_turns_a_default_into_a_correction(
    storage: PostgresStorage,
) -> None:
    frozen = datetime(2026, 1, 10, tzinfo=UTC)
    memory = Memory(storage, embedding=HashEmbedding(), clock=lambda: frozen)
    await memory.check()
    told = (await memory.remember("acme", "Ana owns the review.")).story
    later = (await memory.supersede("acme", told.story_id, "Bo owns the review.")).story
    [closed] = await memory.get_stories("acme", [told.story_id])
    tick = timedelta(microseconds=1)
    assert later.valid_from == closed.valid_to == frozen + tick
    assert (await memory.list_stories("acme", as_of=frozen))[0] == [closed]
    fresh = (await memory.remember("acme", "Cy owns the audit.")).story
    ended = await memory.end_story("acme", fresh.story_id)
    assert ended.valid_from == frozen and ended.valid_to == frozen + tick


async def test_a_confirmation_never_moves_back_and_ids_are_checked_before_anything_goes(
    storage: PostgresStorage,
) -> None:
    clock = Clock(datetime(2026, 1, 10, tzinfo=UTC))
    memory = Memory(storage, embedding=HashEmbedding(), clock=clock, ids=_sequence("s"))
    await memory.check()
    doc = {"metadata.doc": "returns"}
    policy = SourceIn(name="Returns policy", text="30 days.", metadata={"doc": "returns"})
    [first] = (await memory.sync("acme", filters=doc, sources=[policy])).sources
    clock.at = datetime(2026, 1, 1, tzinfo=UTC)
    [kept] = (await memory.sync("acme", filters=doc, sources=[policy])).sources
    assert kept.confirmed_at == first.confirmed_at == first.created_at
    assert await memory.get_source("acme", first.source_id) == kept
    again = Memory(storage, embedding=HashEmbedding(), ids=lambda: first.source_id)
    changed = policy.model_copy(update={"text": "60 days."})
    with pytest.raises(
        InputError,
        match=rf"^postgres: source '{first.source_id}' is already held; ids must be unique$",
    ):
        await again.sync("acme", filters=doc, sources=[changed])
    assert await memory.get_source("acme", first.source_id) == kept
    twins = Memory(storage, embedding=HashEmbedding(), ids=lambda: "twin")
    with pytest.raises(
        InputError, match=r"^postgres: source 'twin' is given twice; ids must be unique$"
    ):
        await twins.sync(
            "acme",
            filters={"metadata.doc": "pair"},
            sources=[
                SourceIn(name="A", text="a", metadata={"doc": "pair"}),
                SourceIn(name="B", text="b", metadata={"doc": "pair"}),
            ],
        )
