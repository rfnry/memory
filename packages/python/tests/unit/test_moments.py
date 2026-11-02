import copy
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Counts, SourceIn
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery import MemoryRefinery
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding

LATE = datetime(9999, 12, 31, 23, 30, tzinfo=timezone(timedelta(hours=-1)))
EARLY = datetime(1, 1, 1, 0, 30, tzinfo=timezone(timedelta(hours=1)))


class Counted:
    def __init__(self, pool: Any) -> None:
        self._pool, self.acquired = pool, 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)

    def acquire(self) -> Any:
        self.acquired += 1
        return self._pool.acquire()


class Bucket:
    def __init__(self) -> None:
        self.keys: list[str] = []

    async def put(self, key: str, data: bytes) -> None:
        self.keys.append(key)


def _at(moment: datetime) -> Callable[[], datetime]:
    return lambda: moment


async def _drained(stream: AsyncIterator[Any]) -> None:
    async for _ in stream:
        pass


Call = Callable[[Memory, MemoryRefinery, MemoryTools, Bucket], Awaitable[Any]]

CASES: list[tuple[str, datetime, Call]] = [
    ("memory: valid_from", LATE, lambda m, r, t, b: m.remember("acme", "Pump.", valid_from=LATE)),
    ("memory: valid_to", LATE, lambda m, r, t, b: m.remember("acme", "Pump.", valid_to=LATE)),
    (
        "memory: valid_from",
        EARLY,
        lambda m, r, t, b: m.supersede("acme", "missing", "Pump.", valid_from=EARLY),
    ),
    (
        "memory: valid_to",
        LATE,
        lambda m, r, t, b: m.supersede("acme", "missing", "Pump.", valid_to=LATE),
    ),
    ("memory: valid_to", LATE, lambda m, r, t, b: m.end_story("acme", "missing", valid_to=LATE)),
    (
        "memory: valid_to",
        EARLY,
        lambda m, r, t, b: m.retire_source("acme", "missing", valid_to=EARLY),
    ),
    ("memory: before", EARLY, lambda m, r, t, b: m.forget_retired("acme", before=EARLY)),
    (
        "models: valid_from of source 'Pump'",
        LATE,
        lambda m, r, t, b: m.ingest("acme", name="Pump", text="Pumps run.", valid_from=LATE),
    ),
    (
        "models: valid_to of source 'Pump'",
        EARLY,
        lambda m, r, t, b: m.ingest("acme", name="Pump", text="Pumps run.", valid_to=EARLY),
    ),
    ("memory: as_of", LATE, lambda m, r, t, b: m.list_sources("acme", as_of=LATE)),
    ("memory: known_at", EARLY, lambda m, r, t, b: m.list_sources("acme", known_at=EARLY)),
    ("memory: as_of", LATE, lambda m, r, t, b: m.list_stories("acme", as_of=LATE)),
    ("memory: since", EARLY, lambda m, r, t, b: m.list_stories("acme", since=EARLY)),
    ("memory: until", LATE, lambda m, r, t, b: m.list_stories("acme", until=LATE)),
    ("memory: as_of", LATE, lambda m, r, t, b: m.search("acme", "pump", as_of=LATE)),
    ("memory: known_at", LATE, lambda m, r, t, b: m.recall("acme", "pump", known_at=LATE)),
    ("memory: since", EARLY, lambda m, r, t, b: m.recall("acme", "pump", since=EARLY)),
    ("memory: until", LATE, lambda m, r, t, b: m.recall("acme", "pump", until=LATE)),
    (
        "memory: as_of",
        LATE,
        lambda m, r, t, b: m.expand("acme", ("story", "missing"), by="vector", as_of=LATE),
    ),
    ("memory: known_at", EARLY, lambda m, r, t, b: m.neighbors("acme", "Pump", known_at=EARLY)),
    (
        "memory: as_of",
        LATE,
        lambda m, r, t, b: m.resembling("acme", source_id="missing", as_of=LATE),
    ),
    (
        "filters: filter 'valid_from' gte",
        LATE,
        lambda m, r, t, b: m.search("acme", "pump", filters={"valid_from": {"gte": LATE}}),
    ),
    (
        "filters: filter 'created_at' lt",
        EARLY,
        lambda m, r, t, b: m.list_stories(
            "acme", filters={"created_at": {"lt": EARLY.isoformat()}}
        ),
    ),
    ("refinery: since", LATE, lambda m, r, t, b: r.retrievals("acme", since=LATE)),
    ("refinery: before", EARLY, lambda m, r, t, b: r.forget_recorded("acme", before=EARLY)),
    ("refinery: since", LATE, lambda m, r, t, b: _drained(r.replay("acme", m, since=LATE))),
    ("refinery: since", EARLY, lambda m, r, t, b: r.measure("acme", since=EARLY)),
    ("refinery: since", LATE, lambda m, r, t, b: r.export("acme", b, since=LATE)),
    (
        "memory: as_of",
        LATE,
        lambda m, r, t, b: t.dispatch("list_stories", {"as_of": LATE.isoformat()}),
    ),
    (
        "memory: valid_from",
        LATE,
        lambda m, r, t, b: t.dispatch(
            "remember", {"text": "Pump ten is red.", "valid_from": LATE.isoformat()}
        ),
    ),
    (
        "models: valid_to of source 'Pump'",
        EARLY,
        lambda m, r, t, b: t.dispatch(
            "ingest", {"name": "Pump", "text": "Pumps run.", "valid_to": EARLY.isoformat()}
        ),
    ),
]


@pytest.mark.parametrize(
    ("refused", "moment", "call"),
    CASES,
    ids=[f"{n}-{refused.split(': ')[1]}" for n, (refused, _, _) in enumerate(CASES)],
)
async def test_a_moment_outside_years_1_to_9999_in_utc_is_refused_before_anything_runs(
    storage: PostgresStorage, refused: str, moment: datetime, call: Call
) -> None:
    pool = Counted(storage._pool)
    counted = copy.copy(storage)
    counted._pool = pool
    embedding = HashEmbedding()
    memory = Memory(counted, embedding=embedding, pillars={"text", "vector", "graph"})
    await memory.check()
    refinery = MemoryRefinery(counted)
    tools = MemoryTools(memory, corpus_id="acme")
    bucket = Bucket()
    pool.acquired = 0
    message = f"{refused} falls outside years 1 to 9999 in UTC: {moment.isoformat()}"
    with pytest.raises(InputError, match=f"^{re.escape(message)}$"):
        await call(memory, refinery, tools, bucket)
    assert (pool.acquired, embedding.calls, bucket.keys) == (0, [], [])
    assert await storage.count_corpus("acme") == Counts()


async def test_a_source_or_a_clock_outside_years_1_to_9999_in_utc_is_refused(
    storage: PostgresStorage,
) -> None:
    refused = f"of source 'Pump' falls outside years 1 to 9999 in UTC: {LATE.isoformat()}"
    with pytest.raises(InputError, match=f"^{re.escape(f'models: valid_from {refused}')}$"):
        SourceIn(name="Pump", text="Pumps run.", valid_from=LATE)
    with pytest.raises(InputError, match=f"^{re.escape(f'models: valid_to {refused}')}$"):
        SourceIn(name="Pump", text="Pumps run.", valid_to=LATE)
    for moment in (LATE, EARLY):
        memory = Memory(storage, clock=_at(moment))
        refused = f"memory: the clock returned {moment.isoformat()}, outside years 1 to 9999 in UTC"
        for call in (
            memory.remember("acme", "Pump eleven is red."),
            memory.list_stories("acme"),
        ):
            with pytest.raises(ConfigurationError, match=f"^{re.escape(refused)}$"):
                await call
    assert await storage.count_corpus("acme") == Counts()


async def test_the_ends_of_time_and_moments_just_inside_them_are_held(
    storage: PostgresStorage,
) -> None:
    first, last = datetime.min.replace(tzinfo=UTC), datetime.max.replace(tzinfo=UTC)
    inside = datetime(9999, 12, 31, 22, 30, tzinfo=timezone(timedelta(hours=-1)))
    memory = Memory(storage)
    await memory.check()
    told = await memory.remember("acme", "Pump twelve is red.", valid_from=first, valid_to=inside)
    assert (told.story.valid_from, told.story.valid_to) == (first, inside)
    stories, _ = await memory.list_stories(
        "acme", as_of=inside - timedelta(hours=1), known_at=last, since=first, until=last
    )
    assert [s.story_id for s in stories] == [told.story.story_id]
    stories, _ = await memory.list_stories(
        "acme", history=True, filters={"valid_to": {"lte": last}}
    )
    assert [s.story_id for s in stories] == [told.story.story_id]
    assert await MemoryRefinery(storage).retrievals("acme", since=first) == ([], None)
