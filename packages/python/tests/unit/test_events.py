import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from rfnry_memory import Chunked, Embedded, Event, Extracted
from rfnry_memory.memory import Memory
from rfnry_memory.models import Section, SourceIn, Usage
from rfnry_memory.postgres import PostgresStorage

from .fakes import HashEmbedding, Scripted, structured_reply


class Kept:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)

    def names(self) -> list[str]:
        return [type(e).__name__ for e in self.events]

    def still(self) -> list[Event]:
        return [
            e.model_copy(update={"duration_ms": 0}) if hasattr(e, "duration_ms") else e
            for e in self.events
        ]


class Halt(Exception):
    pass


def _pump(spent: Usage) -> Any:
    reply = structured_reply({"entities": [{"name": "P-3", "type": "pump"}], "relations": []})
    return reply.model_copy(update={"usage": spent})


async def test_ingest_reports_chunking_embedding_and_each_chunks_graph_step(
    storage: PostgresStorage,
) -> None:
    spent = Usage(input=7, output=3)
    memory = Memory(
        storage,
        embedding=HashEmbedding(),
        generation=Scripted(_pump(spent), _pump(spent)),
        relation_types=["FEEDS"],
        chunk_chars=40,
        overlap_chars=0,
    )
    await memory.check()
    kept = Kept()
    ingested = await memory.ingest(
        "acme",
        name="manual",
        sections=[
            Section(text="Pump P-3 feeds the line."),
            Section(text="Pump P-3 runs at night."),
        ],
        on_event=kept,
    )
    assert kept.still() == [
        Chunked(corpus_id="acme", name="manual", source=1, sources=1, chunks=2),
        Embedded(corpus_id="acme", name="manual", done=2, total=2, computed=2, duration_ms=0),
        Extracted(
            corpus_id="acme",
            name="manual",
            chunk=1,
            chunks=2,
            entities=1,
            relations=0,
            usage=spent,
            duration_ms=0,
        ),
        Extracted(
            corpus_id="acme",
            name="manual",
            chunk=2,
            chunks=2,
            entities=1,
            relations=0,
            usage=spent,
            duration_ms=0,
        ),
    ]
    assert ingested.usage == spent + spent
    assert all(e.duration_ms >= 0 for e in kept.events if isinstance(e, Embedded | Extracted))


async def test_sync_sends_every_chunked_first_and_nothing_for_a_kept_source(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), chunk_chars=200, overlap_chars=0)
    await memory.check()
    one = SourceIn(name="one", text="Alpha.", metadata={"set": "a"})
    first = await memory.sync("acme", filters={"metadata.set": "a"}, sources=[one])
    kept = Kept()
    synced = await memory.sync(
        "acme",
        filters={"metadata.set": "a"},
        sources=[
            one,
            SourceIn(name="two", text="Beta.", metadata={"set": "a"}),
            SourceIn(name="three", text="Gamma.", metadata={"set": "a"}),
        ],
        on_event=kept,
    )
    assert synced.kept == [first.sources[0].source_id]
    assert kept.names() == ["Chunked", "Chunked", "Embedded", "Embedded"]
    chunked = [e for e in kept.events if isinstance(e, Chunked)]
    assert [(e.name, e.source, e.sources, e.chunks) for e in chunked] == [
        ("two", 1, 2, 1),
        ("three", 2, 2, 1),
    ]


async def test_a_sync_that_prepares_again_sends_its_events_again(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    written = memory.storage.sync_sources
    calls = 0

    async def moved(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return None if calls == 1 else await written(*args, **kwargs)

    monkeypatch.setattr(memory.storage, "sync_sources", moved)
    kept = Kept()
    source = SourceIn(name="one", text="Alpha.", metadata={"set": "a"})
    await memory.sync("acme", filters={"metadata.set": "a"}, sources=[source], on_event=kept)
    assert kept.names() == ["Chunked", "Embedded", "Chunked", "Embedded"]


async def test_reused_vectors_are_not_counted_as_computed(storage: PostgresStorage) -> None:
    embedding = HashEmbedding()
    memory = Memory(storage, embedding=embedding, chunk_chars=40, overlap_chars=0)
    await memory.check()
    await memory.ingest("acme", name="first", text="Alpha.")
    kept = Kept()
    await memory.ingest(
        "acme",
        name="second",
        sections=[Section(text="Alpha."), Section(text="Beta.")],
        on_event=kept,
    )
    [embedded] = [e for e in kept.events if isinstance(e, Embedded)]
    assert (embedded.done, embedded.total, embedded.computed) == (2, 2, 1)
    assert embedding.calls[-1] == ["Beta."]
    again = Kept()
    await memory.ingest("acme", name="third", text="Alpha.", on_event=again)
    [held] = [e for e in again.events if isinstance(e, Embedded)]
    assert (held.computed, held.duration_ms) == (0, 0)


async def test_a_source_with_no_chunks_sends_only_its_chunked(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    kept = Kept()
    await memory.ingest("acme", name="empty", sections=[], on_event=kept)
    assert kept.events == [Chunked(corpus_id="acme", name="empty", source=1, sources=1, chunks=0)]


async def test_a_handler_that_raises_stops_the_call_and_nothing_is_written(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()

    async def halt(event: Event) -> None:
        if isinstance(event, Extracted):
            raise Halt

    with pytest.raises(Halt):
        await memory.ingest("acme", name="m", text="Pump P-3 feeds the line.", on_event=halt)
    with pytest.raises(Halt):
        await memory.sync(
            "acme",
            filters={"metadata.set": "a"},
            sources=[SourceIn(name="one", text="Alpha.", metadata={"set": "a"})],
            on_event=halt,
        )
    counts = await memory.storage.count_corpus("acme")
    assert (counts.sources, counts.chunks, counts.entities) == (0, 0, 0)


async def test_a_handler_that_raises_on_a_retried_sync_writes_nothing(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    written = memory.storage.sync_sources
    calls = 0

    async def moved(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return None if calls == 1 else await written(*args, **kwargs)

    monkeypatch.setattr(memory.storage, "sync_sources", moved)
    embedded = 0

    async def halt(event: Event) -> None:
        nonlocal embedded
        if isinstance(event, Embedded):
            embedded += 1
            if embedded == 2:
                raise Halt

    source = SourceIn(name="one", text="Alpha.", metadata={"set": "a"})
    with pytest.raises(Halt):
        await memory.sync("acme", filters={"metadata.set": "a"}, sources=[source], on_event=halt)
    assert calls == 1
    counts = await memory.storage.count_corpus("acme")
    assert (counts.sources, counts.chunks) == (0, 0)


async def test_cancelling_the_call_while_its_handler_waits_writes_nothing(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    waiting = asyncio.Event()

    async def stuck(event: Event) -> None:
        if isinstance(event, Embedded):
            waiting.set()
            await asyncio.sleep(3600)

    call = asyncio.create_task(memory.ingest("acme", name="m", text="Alpha.", on_event=stuck))
    await asyncio.wait_for(waiting.wait(), timeout=10)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    counts = await memory.storage.count_corpus("acme")
    assert (counts.sources, counts.chunks) == (0, 0)


async def test_without_a_handler_the_stages_still_run_unobserved(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    ingested = await memory.ingest("acme", name="m", text="Alpha.")
    assert ingested.source.name == "m"


def test_events_are_frozen_and_closed() -> None:
    chunked = Chunked(corpus_id="acme", name="m", source=1, sources=1, chunks=1)
    with pytest.raises(ValidationError):
        chunked.chunks = 2
    with pytest.raises(ValidationError):
        Chunked.model_validate(
            {"corpus_id": "acme", "name": "m", "source": 1, "sources": 1, "chunks": 1, "x": 1}
        )
