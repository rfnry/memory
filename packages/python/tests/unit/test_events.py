import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from rfnry_memory import Chunked, Embedded, Event, Extracted, Judged, Ranked, Retrieved
from rfnry_memory.errors import InputError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    CheckAnswer,
    Decided,
    EntityIn,
    QuestionLike,
    Section,
    SourceIn,
    State,
    Usage,
)
from rfnry_memory.postgres import PostgresStorage

from .fakes import DocumentHashEmbedding, Fixed, HashEmbedding, Scripted, structured_reply


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


class NewerHash(HashEmbedding):
    model = "hash-8-v2"


class NewerDocumentHash(DocumentHashEmbedding):
    model = "hash-document-8-v2"


async def test_reembed_counts_its_batches_up_to_the_stale_total(storage: PostgresStorage) -> None:
    old = Memory(storage, embedding=HashEmbedding(), chunk_chars=40, overlap_chars=0)
    await old.check()
    await old.ingest(
        "acme", name="many", sections=[Section(text=f"Line {i} runs.") for i in range(299)]
    )
    await old.remember("acme", "The line runs nights.", related=0)
    memory = Memory(storage, embedding=NewerHash(), chunk_chars=40, overlap_chars=0)
    kept = Kept()
    assert await memory.reembed("acme", on_event=kept) == 300
    assert [
        (e.name, e.done, e.total, e.computed) for e in kept.events if isinstance(e, Embedded)
    ] == [
        (None, 256, 300, 256),
        (None, 300, 300, 44),
    ]
    assert kept.names() == ["Embedded", "Embedded"]


async def test_reembed_whole_documents_names_each_and_grows_its_total(
    storage: PostgresStorage,
) -> None:
    old = Memory(storage, embedding=DocumentHashEmbedding(), chunk_chars=40, overlap_chars=0)
    await old.check()
    ingested = await old.ingest(
        "acme", name="sheet", sections=[Section(text="Feed line."), Section(text="Return line.")]
    )
    memory = Memory(storage, embedding=NewerDocumentHash(), chunk_chars=40, overlap_chars=0)
    [first, _] = await storage.source_chunks("acme", ingested.source.source_id)
    await storage.set_vectors(
        "acme", [("chunk", first.chunk_id)], [[0.5] * 8], "hash-document-8-v2"
    )
    kept = Kept()
    assert await memory.reembed("acme", on_event=kept) == 2
    assert [
        (e.name, e.done, e.total, e.computed) for e in kept.events if isinstance(e, Embedded)
    ] == [("sheet", 2, 2, 2)]


async def test_remember_and_supersede_report_the_storys_embedding_and_graph_step(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    kept = Kept()
    story = (
        await memory.remember(
            "acme",
            "Pump P-3 feeds the line.",
            entities=[EntityIn(name="P-3", type="pump")],
            related=0,
            on_event=kept,
        )
    ).story
    expected = [
        Embedded(corpus_id="acme", name=None, done=1, total=1, computed=1, duration_ms=0),
        Extracted(
            corpus_id="acme",
            name=None,
            chunk=1,
            chunks=1,
            entities=1,
            relations=0,
            usage=Usage(),
            duration_ms=0,
        ),
    ]
    assert kept.still() == expected
    again = Kept()
    await memory.supersede(
        "acme",
        story.story_id,
        "Pump P-3 feeds the night line.",
        entities=[EntityIn(name="P-3", type="pump")],
        related=0,
        on_event=again,
    )
    assert again.still() == expected
    repeat = Kept()
    await memory.remember("acme", "pump p-3 feeds the night line.", related=0, on_event=repeat)
    assert repeat.events == []


def _deciding(spent: Usage) -> Fixed:
    def answer(state: State, questions: Mapping[str, QuestionLike]) -> Decided:
        return Decided(
            answers={name: CheckAnswer(probability=0.5) for name in questions}, usage=spent
        )

    return Fixed(answer)


async def _filled(storage: PostgresStorage, **settings: Any) -> Memory:
    memory = Memory(
        storage,
        embedding=HashEmbedding(),
        pillars={"text", "vector", "graph"},
        chunk_chars=40,
        overlap_chars=0,
        **settings,
    )
    await memory.check()
    await memory.ingest(
        "acme",
        name="manual",
        sections=[
            Section(text="Replace the pump seal yearly."),
            Section(text="The pump seal leaks at night."),
            Section(text="Check the seal on pump P-3."),
        ],
    )
    return memory


async def test_search_reports_each_pillar_then_the_ranked_hits_then_each_judgement(
    storage: PostgresStorage,
) -> None:
    spent = Usage(input=2, output=1)
    memory = await _filled(storage, decision=_deciding(spent))
    kept = Kept()
    found = await memory.search("acme", "pump seal", judge=True, on_event=kept)
    count = len(found.hits)
    assert count == 3
    assert kept.names() == ["Retrieved"] * 3 + ["Ranked"] + ["Judged"] * count
    retrieved = [e for e in kept.events if isinstance(e, Retrieved)]
    assert sorted(e.pillar for e in retrieved) == ["graph", "text", "vector"]
    assert {e.pillar: e.report for e in retrieved} == found.pillars
    assert all(e.corpus_ids == ["acme"] for e in retrieved)
    [ranked] = [e for e in kept.events if isinstance(e, Ranked)]
    assert ranked.hits == [h.model_copy(update={"judge": None}) for h in found.hits]
    judged = [e for e in kept.events if isinstance(e, Judged)]
    assert [e.done for e in judged] == list(range(1, count + 1))
    assert {e.total for e in judged} == {count}
    assert sorted(e.hit.id for e in judged) == sorted(h.id for h in found.hits)
    assert {e.hit.id: e.hit for e in judged} == {h.id: h for h in found.hits}
    total = Usage()
    for event in judged:
        total = total + event.usage
    assert total == found.usage == Usage(input=2 * count, output=count)


async def test_recall_reports_like_search(storage: PostgresStorage) -> None:
    memory = await _filled(storage)
    await memory.remember("acme", "The pump seal was replaced in March.", related=0)
    kept = Kept()
    found = await memory.recall("acme", "pump seal", on_event=kept)
    assert kept.names() == ["Retrieved"] * 3 + ["Ranked"]
    [ranked] = [e for e in kept.events if isinstance(e, Ranked)]
    assert ranked.hits == found.hits


async def test_a_failed_pillar_is_reported_with_its_error(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _filled(storage)

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise StorageError("postgres: the text index is gone")

    monkeypatch.setattr(memory.storage, "search_text", broken)
    kept = Kept()
    found = await memory.search("acme", "pump seal", on_event=kept)
    [text] = [e for e in kept.events if isinstance(e, Retrieved) and e.pillar == "text"]
    assert text.report == found.pillars["text"]
    assert (text.report.error or "").startswith("StorageError: ")


async def test_the_handler_never_runs_beside_itself(storage: PostgresStorage) -> None:
    memory = await _filled(storage, decision=_deciding(Usage()))
    active, most = 0, 0

    async def slow(event: Event) -> None:
        nonlocal active, most
        active += 1
        most = max(most, active)
        await asyncio.sleep(0.01)
        active -= 1

    await memory.search("acme", "pump seal", judge=True, on_event=slow)
    assert most == 1


async def test_a_handler_may_search_itself_and_serve_several_calls(
    storage: PostgresStorage,
) -> None:
    memory = await _filled(storage)
    inner: list[int] = []

    async def curious(event: Event) -> None:
        if isinstance(event, Ranked) and not inner:
            inner.append(len((await memory.search("acme", "seal")).hits))

    await asyncio.wait_for(memory.search("acme", "pump seal", on_event=curious), timeout=10)
    assert inner == [3]
    kept = Kept()
    both = await asyncio.wait_for(
        asyncio.gather(
            memory.search("acme", "pump seal", on_event=kept),
            memory.search("acme", "night leaks", on_event=kept),
        ),
        timeout=10,
    )
    assert len(both) == 2 and kept.names().count("Ranked") == 2


async def test_remember_reports_its_related_search(storage: PostgresStorage) -> None:
    memory = await _filled(storage)
    await memory.remember("acme", "The pump seal was replaced in March.", related=0)
    kept = Kept()
    remembered = await memory.remember(
        "acme", "The pump seal leaks again.", related=3, on_event=kept
    )
    assert kept.names() == ["Embedded", "Extracted"] + ["Retrieved"] * 3 + ["Ranked"]
    [ranked] = [e for e in kept.events if isinstance(e, Ranked)]
    assert ranked.hits == remembered.related.hits
    assert remembered.story.story_id not in [h.id for h in ranked.hits]
    repeat = Kept()
    again = await memory.remember("acme", "the pump seal leaks again.", related=3, on_event=repeat)
    assert again.story == remembered.story
    assert repeat.names() == ["Retrieved"] * 3 + ["Ranked"]


async def test_a_blank_query_sends_nothing(storage: PostgresStorage) -> None:
    memory = await _filled(storage)
    kept = Kept()
    with pytest.raises(InputError, match=r"^memory: a search needs a query$"):
        await memory.search("acme", "  ", on_event=kept)
    assert kept.events == []
