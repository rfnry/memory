import base64
import json
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery import MemoryRefinery
from rfnry_memory.rows import RetrievalRow
from rfnry_memory.tools import MemoryTools

JAN = datetime(2026, 1, 1, tzinfo=UTC)
ODD = [
    "a,b|c",
    'it\'s "quoted"',
    "back\\slash/and/slash",
    "==+/padded==",
    "naïve 漢字 ✓",
    "x" * 399 + "!",
    '["a", "json"]',
]

Listing = Callable[..., Awaitable[tuple[list[Any], str | None]]]


def _ids(rows: list[Any]) -> list[str]:
    return [
        getattr(r, "source_id", None) or getattr(r, "story_id", None) or r.retrieval_id
        for r in rows
    ]


async def _walk(listing: Listing, limit: int, **options: Any) -> list[str]:
    seen: list[str] = []
    cursor: str | None = None
    for _ in range(1000):
        page, cursor = await listing(limit=limit, cursor=cursor, **options)
        seen.extend(_ids(page))
        if cursor is None:
            return seen
    raise AssertionError("tests: the listing never ended")


def _named(names: list[str]) -> Callable[[], str]:
    made: Iterator[str] = iter(names)
    return lambda: next(made, None) or uuid4().hex


async def _stories(storage: PostgresStorage, corpus: str, ids: list[str], **kw: Any) -> Memory:
    memory = Memory(storage, ids=_named(ids), **kw)
    await memory.check()
    for n, story_id in enumerate(ids):
        await memory.remember(
            corpus, f"Story {story_id} holds.", valid_from=JAN + timedelta(days=n), related=0
        )
    return memory


async def _recordings(storage: PostgresStorage, ids: list[str], at: list[datetime]) -> None:
    for retrieval_id, moment in zip(ids, at, strict=True):
        await storage.put_retrieval(
            ["acme"],
            RetrievalRow(
                retrieval_id=retrieval_id,
                corpus_ids=["acme"],
                verb="search",
                at=moment,
                recorded=moment,
                duration_ms=1,
                body={
                    "query": retrieval_id,
                    "arguments": {},
                    "window": {"as_of": None, "known_at": None},
                    "candidates": {},
                    "hits": [],
                    "pillars": {},
                    "fingerprint": {},
                },
            ),
        )


async def test_a_story_written_or_deleted_between_two_pages_repeats_or_skips_no_row(
    storage: PostgresStorage,
) -> None:
    memory = await _stories(storage, "acme", ["s1", "s2", "s3", "s4", "s5", "new"])
    first, cursor = await memory.list_stories("acme", limit=2)
    assert _ids(first) == ["new", "s5"]
    await memory.remember("acme", "The newest story holds.", valid_from=JAN + timedelta(days=9))
    second, cursor = await memory.list_stories("acme", limit=2, cursor=cursor)
    assert _ids(second) == ["s4", "s3"]
    await memory.delete_story("acme", "s4")
    await memory.delete_story("acme", "new")
    third, cursor = await memory.list_stories("acme", limit=2, cursor=cursor)
    assert (_ids(third), cursor) == (["s2", "s1"], None)


async def test_a_source_written_or_deleted_between_two_pages_repeats_or_skips_no_row(
    storage: PostgresStorage,
) -> None:
    names = ["d1", "d2", "d3", "d4", "d5", "d6"]
    memory = Memory(storage, ids=_named(names))
    await memory.check()
    for name in names[:5]:
        await memory.ingest("acme", name=name, text=f"Document {name}.")
    first, cursor = await memory.list_sources("acme", limit=2)
    assert _ids(first) == ["d5", "d4"]
    await memory.ingest("acme", name="d6", text="Document d6.")
    second, cursor = await memory.list_sources("acme", limit=2, cursor=cursor)
    assert _ids(second) == ["d3", "d2"]
    await memory.delete_source("acme", "d5")
    await memory.delete_source("acme", "d3")
    third, cursor = await memory.list_sources("acme", limit=2, cursor=cursor)
    assert (_ids(third), cursor) == (["d1"], None)


async def test_a_recording_written_between_two_pages_repeats_no_row(
    storage: PostgresStorage,
) -> None:
    refinery = MemoryRefinery(storage)
    await _recordings(
        storage, ["r1", "r2", "r3", "r4"], [JAN + timedelta(minutes=m) for m in (1, 2, 3, 4)]
    )
    first, cursor = await refinery.retrievals("acme", limit=2)
    assert _ids(first) == ["r4", "r3"]
    await _recordings(storage, ["r5"], [JAN + timedelta(minutes=5)])
    second, cursor = await refinery.retrievals("acme", limit=2, cursor=cursor)
    assert (_ids(second), cursor) == (["r2", "r1"], None)


async def test_rows_tied_on_their_moment_page_once_each_in_order(
    storage: PostgresStorage,
) -> None:
    ties = [f"t{n}" for n in range(7)]
    frozen = Memory(storage, ids=_named(ties * 2), clock=lambda: JAN)
    await frozen.check()
    for name in ties:
        await frozen.ingest("acme", name=name, text=f"Document {name}.")
        await frozen.remember("acme", f"Story {name} holds.", valid_from=JAN, related=0)
    await _recordings(storage, ties, [JAN] * len(ties))
    tied = sorted(ties, reverse=True)
    for limit in (1, 2, 3, 7, 8):
        assert await _walk(frozen.list_sources, limit, corpus_id="acme") == tied
        assert await _walk(frozen.list_stories, limit, corpus_id="acme") == tied
        assert await _walk(MemoryRefinery(storage).retrievals, limit, corpus_id="acme") == tied


async def test_an_id_of_any_shape_round_trips_through_the_cursor(storage: PostgresStorage) -> None:
    memory = await _stories(storage, "acme", ODD)
    sources = Memory(storage, ids=_named(ODD))
    for name in ODD:
        await sources.ingest("acme", name="same", text=f"Document {name}.")
    await _recordings(storage, ODD, [JAN] * len(ODD))
    assert await _walk(memory.list_stories, 1, corpus_id="acme") == list(reversed(ODD))
    assert await _walk(memory.list_sources, 1, corpus_id="acme") == list(reversed(ODD))
    ordered = sorted(ODD, key=lambda i: i.encode("utf-8"), reverse=True)
    assert await _walk(MemoryRefinery(storage).retrievals, 1, corpus_id="acme") == ordered


def _encoded(value: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


async def test_a_cursor_no_listing_returned_is_refused(storage: PostgresStorage) -> None:
    memory = await _stories(storage, "acme", ["s1", "s2", "s3"])
    await _recordings(storage, ["r1", "r2", "r3"], [JAN, JAN, JAN])
    _, given = await memory.list_stories("acme", limit=1)
    assert given is not None
    refinery = MemoryRefinery(storage)
    wrong = [
        "1",
        "100",
        "garbage!",
        given[:-4],
        given + "A",
        _encoded({"at": JAN.isoformat(), "id": "s1"}),
        _encoded([JAN.isoformat()]),
        _encoded([JAN.isoformat(), "s1", "extra"]),
        _encoded([1, "s1"]),
        _encoded(["not a moment", "s1"]),
        _encoded(["2026-01-01T00:00:00", "s1"]),
        _encoded([JAN.isoformat(), 1]),
        given + "=",
        " " + given,
        "",
    ]
    for cursor in wrong:
        message = rf"^memory: cursor {cursor!r} is not one a listing returned$".replace("+", r"\+")
        with pytest.raises(InputError, match=message):
            await memory.list_stories("acme", limit=1, cursor=cursor)
        with pytest.raises(InputError, match=message):
            await memory.list_sources("acme", limit=1, cursor=cursor)
        with pytest.raises(InputError, match=message.replace("memory:", "refinery:")):
            await refinery.retrievals("acme", limit=1, cursor=cursor)
        with pytest.raises(InputError, match=message.replace("memory:", "postgres:")):
            await storage.list_sources("acme", filters=None, limit=1, cursor=cursor)
        with pytest.raises(InputError, match=message.replace("memory:", "postgres:")):
            await storage.list_retrievals(["acme"], since=None, limit=1, cursor=cursor)
    tools = MemoryTools(memory, corpus_id="acme")
    for name in ("list_stories", "list_sources"):
        with pytest.raises(
            InputError, match="^memory: cursor '100' is not one a listing returned$"
        ):
            await tools.dispatch(name, {"cursor": "100"})


async def test_a_cursor_carried_to_another_corpus_reads_only_that_corpus(
    storage: PostgresStorage,
) -> None:
    memory = await _stories(storage, "a", ["a1", "a2", "a3", "a4"])
    other = await _stories(storage, "b", ["b1", "b2", "b3", "b4"])
    _, cursor = await memory.list_stories("a", limit=2)
    carried, _ = await other.list_stories("b", limit=10, cursor=cursor)
    assert _ids(carried) == ["b2", "b1"]
    held, _ = await memory.list_stories("a", limit=10, cursor=cursor)
    assert _ids(held) == ["a2", "a1"]


async def test_every_window_pages_as_one_page_reads(storage: PostgresStorage) -> None:
    names = [f"v{n}" for n in range(6)]
    memory = Memory(storage, ids=_named([i for n in names for i in (f"s-{n}", f"d-{n}")]))
    await memory.check()
    await memory.set_history("acme", True)
    told = []
    for n, name in enumerate(names):
        told.append(
            (
                await memory.remember(
                    "acme", f"Story {name} holds.", valid_from=JAN + timedelta(days=n), related=0
                )
            ).story
        )
        await memory.ingest("acme", name=name, text=f"Document {name}.")
    middle = told[3].created_at
    for story in told[:3]:
        await memory.end_story("acme", story.story_id, valid_to=JAN + timedelta(days=10))
    for retired in ("d-v0", "d-v1"):
        await memory.retire_source("acme", retired, valid_to=JAN + timedelta(days=20))
    windows: list[dict[str, Any]] = [
        {},
        {"history": True},
        {"as_of": JAN + timedelta(days=4)},
        {"known_at": middle},
        {"as_of": JAN + timedelta(days=30), "known_at": middle},
    ]
    for window in windows:
        for listing in (memory.list_stories, memory.list_sources):
            whole, _ = await listing("acme", limit=100, **window)
            for limit in (1, 2, 4):
                assert await _walk(listing, limit, corpus_id="acme", **window) == _ids(whole), (
                    listing.__name__,
                    window,
                    limit,
                )


async def test_the_tools_page_through_listings_with_the_cursor_they_return(
    storage: PostgresStorage,
) -> None:
    names = [f"n{n}" for n in range(5)]
    memory = await _stories(storage, "acme", names)
    for name in names:
        await memory.ingest("acme", name=name, text=f"Document {name}.")
    tools = MemoryTools(memory, corpus_id="acme")
    for name, key, field in (
        ("list_stories", "stories", "story_id"),
        ("list_sources", "sources", "source_id"),
    ):
        seen: list[str] = []
        args: dict[str, Any] = {"limit": 2}
        for _ in range(100):
            result = await tools.dispatch(name, args)
            assert isinstance(result, dict)
            seen.extend(row[field] for row in result[key])
            if result["cursor"] is None:
                break
            args = {"limit": 2, "cursor": result["cursor"]}
        else:
            raise AssertionError("tests: the listing never ended")
        whole = (
            await memory.list_stories("acme")
            if key == "stories"
            else await memory.list_sources("acme")
        )[0]
        assert seen == _ids(whole)
