from datetime import UTC, datetime
from typing import Any

import pytest

from rfnry_memory import structured
from rfnry_memory.errors import InputError, StorageError
from rfnry_memory.events import Event, Extracted
from rfnry_memory.graph import extraction
from rfnry_memory.graph.extraction import extraction_key
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    Counts,
    EntityIn,
    RelationIn,
    Reply,
    Section,
    Source,
    SourceIn,
    Usage,
    Window,
)
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.rows import chunk_digest, name_key
from rfnry_memory.tools import MemoryTools

from .fakes import DocumentHashEmbedding, HashEmbedding, Scripted, structured_reply

LINE_A = {"metadata.file": "line-a.L5X"}
PLC = {"file": "line-a.L5X", "kind": "plc"}


def _plc(name: str, **kwargs: Any) -> SourceIn:
    return SourceIn(name=name, metadata=PLC, **kwargs)


class _Failing(HashEmbedding):
    def __init__(self) -> None:
        super().__init__()
        self.fail_on: str | None = None

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        if self.fail_on is not None and any(self.fail_on in t for t in texts):
            raise RuntimeError("the embedding service is down")
        return await super().embed(texts, purpose=purpose)


async def _memory(storage: PostgresStorage, embedding: HashEmbedding | None = None) -> Memory:
    memory = Memory(
        storage,
        embedding=embedding or HashEmbedding(),
        pillars={"text", "vector", "graph"},
    )
    await memory.check()
    return memory


async def _seed(memory: Memory) -> list[str]:
    old = [
        await memory.ingest(
            "acme", name=f"old {i}", text=f"Rung {i} drives Motor_{i}.", metadata=PLC
        )
        for i in range(2)
    ]
    await memory.ingest(
        "acme", name="other file", text="Line B conveyor.", metadata={"file": "line-b.L5X"}
    )
    await memory.ingest("other", name="other corpus", text="Rung 0 drives Motor_0.", metadata=PLC)
    return [o.source.source_id for o in old]


async def _names(
    memory: Memory, corpus_id: str = "acme", filters: dict[str, Any] | None = None
) -> list[str]:
    sources, _ = await memory.list_sources(corpus_id, filters=filters or LINE_A)
    return sorted(s.name for s in sources)


async def test_sync_swaps_the_matching_set(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    old_ids = await _seed(memory)
    synced = await memory.sync(
        "acme",
        filters=LINE_A,
        sources=[
            _plc(
                "Routine MainRoutine",
                sections=[
                    Section(title="Rung 0", text="XIC(Start) OTE(Motor_1)", locator={"rung": 0}),
                    Section(title="Rung 1", text="XIC(Stop) OTU(Motor_1)", locator={"rung": 1}),
                ],
            ),
            _plc("Tags", text="Motor_1 BOOL"),
        ],
    )
    assert await _names(memory) == ["Routine MainRoutine", "Tags"]
    listed, _ = await memory.list_sources("acme", filters=LINE_A)
    assert {s.source_id for s in listed} == {s.source_id for s in synced.sources}
    assert not {s.source_id for s in synced.sources} & set(old_ids)
    assert synced.deleted == Counts(sources=2, chunks=2) and synced.usage == Usage()
    assert [s.chunks for s in synced.sources] == [2, 1]
    assert [await memory.get_source("acme", i) for i in old_ids] == [None, None]
    routine = synced.sources[0]
    assert await memory.read_source("acme", routine.source_id, locator={"rung": 1}) == (
        "XIC(Stop) OTU(Motor_1)"
    )
    assert (await memory.verify_absent("acme")).chunks == 4
    hits = await memory.search("acme", "Rung 0 drives Motor_0", kinds=("chunk",))
    assert not {h.chunk.source_id for h in hits.hits if h.chunk} & set(old_ids)
    assert await _names(memory, filters={"metadata.file": "line-b.L5X"}) == ["other file"]
    assert await _names(memory, "other") == ["other corpus"]


async def test_a_model_that_fails_before_the_write_leaves_the_old_set_whole(
    storage: PostgresStorage,
) -> None:
    embedding = _Failing()
    memory = await _memory(storage, embedding)
    old_ids = await _seed(memory)
    before = await memory.verify_absent("acme")
    embedding.fail_on = "BOOM"
    with pytest.raises(RuntimeError, match="the embedding service is down"):
        await memory.sync(
            "acme",
            filters=LINE_A,
            sources=[_plc("Tags", text="Motor_1 BOOL"), _plc("Routine", text="BOOM")],
        )
    assert await memory.verify_absent("acme") == before
    listed, _ = await memory.list_sources("acme", filters=LINE_A)
    assert sorted(s.source_id for s in listed) == sorted(old_ids)


async def test_a_source_outside_the_filters_is_refused_before_any_model_runs(
    storage: PostgresStorage,
) -> None:
    embedding = HashEmbedding()
    memory = await _memory(storage, embedding)
    await _seed(memory)
    before, calls = await memory.verify_absent("acme"), len(embedding.calls)
    for stray in (
        SourceIn(name="Tags", text="Motor_1 BOOL", metadata={"file": "line-b.L5X"}),
        SourceIn(name="Tags", text="Motor_1 BOOL"),
    ):
        with pytest.raises(
            InputError, match=r"^memory: source 'Tags' does not match the filters it syncs$"
        ):
            await memory.sync(
                "acme", filters=LINE_A, sources=[_plc("Routine", text="XIC(Start)"), stray]
            )
    assert len(embedding.calls) == calls and await memory.verify_absent("acme") == before


@pytest.mark.parametrize("filters", [{}, None])
async def test_sync_without_filters_is_refused(
    storage: PostgresStorage, filters: dict[str, Any] | None
) -> None:
    memory = await _memory(storage)
    await _seed(memory)
    before = await memory.verify_absent("acme")
    with pytest.raises(
        InputError,
        match=r"^memory: sync needs filters naming the set it syncs — delete_corpus empties a corpus$",
    ):
        await memory.sync("acme", filters=filters, sources=[])  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^filters: filter key 'corpus_id' is reserved"):
        await memory.sync("acme", filters={"corpus_id": "other"}, sources=[])
    assert await memory.verify_absent("acme") == before


async def test_sync_with_no_sources_deletes_the_set(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    await _seed(memory)
    synced = await memory.sync("acme", filters=LINE_A, sources=[])
    assert (synced.sources, synced.deleted, synced.usage) == (
        [],
        Counts(sources=2, chunks=2),
        Usage(),
    )
    assert await _names(memory) == []
    assert await _names(memory, filters={"metadata.file": "line-b.L5X"}) == ["other file"]


def _held(source: Source) -> dict[str, Any]:
    return source.model_dump(exclude={"confirmed_at"})


async def test_a_retry_with_the_same_input_keeps_the_set_as_it_is(storage: PostgresStorage) -> None:
    embedding = HashEmbedding()
    memory = await _memory(storage, embedding)
    await _seed(memory)
    sources = [_plc("Routine", text="XIC(Start) OTE(Motor_1)"), _plc("Tags", text="Motor_1 BOOL")]
    first = await memory.sync("acme", filters=LINE_A, sources=sources)
    calls = len(embedding.calls)
    fingerprint = await memory.fingerprint("acme")
    second = await memory.sync("acme", filters=LINE_A, sources=sources)
    assert [_held(s) for s in second.sources] == [_held(s) for s in first.sources]
    pairs = zip(first.sources, second.sources, strict=True)
    assert all(b.confirmed_at > a.confirmed_at for a, b in pairs)
    assert second.kept == [s.source_id for s in first.sources]
    assert second.deleted.clean and second.usage == Usage()
    assert len(embedding.calls) == calls
    assert await _names(memory) == ["Routine", "Tags"]
    assert await memory.fingerprint("acme") == fingerprint


async def test_a_sync_rewrites_only_what_changed(storage: PostgresStorage) -> None:
    embedding = HashEmbedding()
    memory = await _memory(storage, embedding)
    pages = [
        Section(title=f"Page {i}", text=f"Page {i} of the manual.", locator={"page": i})
        for i in range(3)
    ]
    manual = _plc("Manual", sections=pages)
    tags = _plc("Tags", text="Motor_1 BOOL")
    first = await memory.sync("acme", filters=LINE_A, sources=[manual, tags])
    story = (
        await memory.remember(
            "acme", "Tags were audited.", source_ids=(first.sources[1].source_id,)
        )
    ).story
    embedding.calls.clear()
    redacted = _plc(
        "Manual",
        sections=[
            pages[0],
            Section(title="Page 1", text="[redacted]", locator={"page": 1}),
            pages[2],
        ],
    )
    second = await memory.sync(
        "acme",
        filters=LINE_A,
        sources=[redacted, tags, _plc("Notes", text="Page 2 of the manual.")],
    )
    new_manual, kept_tags, notes = second.sources
    assert second.kept == [first.sources[1].source_id]
    assert _held(kept_tags) == _held(first.sources[1])
    assert new_manual.source_id != first.sources[0].source_id
    assert second.deleted == Counts(sources=1, chunks=3)
    assert embedding.calls == [["[redacted]"]]
    assert notes.chunks == 1
    assert await memory.get_stories("acme", [story.story_id]) == [story]
    assert await _names(memory) == ["Manual", "Notes", "Tags"]
    assert (await memory.fingerprint("acme")).stale == 0
    assert (
        await memory.read_source("acme", new_manual.source_id, locator={"page": 1}) == "[redacted]"
    )


async def test_a_change_of_chunking_or_pillars_rewrites_every_source(
    storage: PostgresStorage,
) -> None:
    sources = [_plc("Tags", text="Motor_1 BOOL")]
    memory = Memory(storage, embedding=HashEmbedding(), chunk_chars=300, overlap_chars=30)
    await memory.check()
    first = await memory.sync("acme", filters=LINE_A, sources=sources)
    for changed in (
        Memory(storage, embedding=HashEmbedding(), chunk_chars=400, overlap_chars=30),
        Memory(
            storage, embedding=HashEmbedding(), chunk_chars=300, overlap_chars=30, pillars={"text"}
        ),
    ):
        again = await changed.sync("acme", filters=LINE_A, sources=sources)
        assert again.kept == [] and again.deleted.sources == 1
        assert again.sources[0].source_id != first.sources[0].source_id


async def test_a_list_that_repeats_a_source_keeps_one_and_writes_the_other(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    tags = _plc("Tags", text="Motor_1 BOOL")
    first = await memory.sync("acme", filters=LINE_A, sources=[tags])
    second = await memory.sync("acme", filters=LINE_A, sources=[tags, tags])
    assert second.kept == [first.sources[0].source_id]
    assert (
        _held(second.sources[0]) == _held(first.sources[0])
        and second.sources[1].source_id != first.sources[0].source_id
    )
    assert second.deleted.clean and await _names(memory) == ["Tags", "Tags"]


async def test_a_set_that_moves_during_the_sync_is_written_whole(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    tags = _plc("Tags", text="Motor_1 BOOL")
    first = await memory.sync("acme", filters=LINE_A, sources=[tags])
    original = storage.sync_sources
    raced: list[bool] = []

    async def racing(*args: Any, **kwargs: Any) -> Any:
        if not raced:
            raced.append(True)
            await original("acme", LINE_A, [], None, at=datetime.now(UTC))
        return await original(*args, **kwargs)

    storage.sync_sources = racing  # type: ignore[method-assign]
    second = await memory.sync("acme", filters=LINE_A, sources=[tags, _plc("Notes", text="n")])
    assert second.kept == [] and second.deleted.clean
    assert second.sources[0].source_id != first.sources[0].source_id
    assert await _names(memory) == ["Notes", "Tags"]


async def test_a_set_that_keeps_moving_stops_after_one_retry(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    tags = _plc("Tags", text="Motor_1 BOOL")
    await memory.sync("acme", filters=LINE_A, sources=[tags])
    calls: list[bool] = []

    async def refusing(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        return None

    memory.storage.sync_sources = refusing  # type: ignore[method-assign]
    with pytest.raises(
        StorageError, match=r"^memory: the set moved under the sync twice; sync again$"
    ):
        await memory.sync("acme", filters=LINE_A, sources=[tags, _plc("Notes", text="n")])
    assert len(calls) == 2


async def test_a_chunk_the_corpus_already_holds_is_not_embedded_again(
    storage: PostgresStorage,
) -> None:
    embedding = HashEmbedding()
    memory = await _memory(storage, embedding)
    await memory.ingest(
        "acme", name="a", sections=[Section(text="Shared paragraph."), Section(text="Only in a.")]
    )
    await memory.ingest("other", name="c", text="Only in the other corpus.")
    embedding.calls.clear()
    await memory.ingest(
        "acme",
        name="b",
        sections=[Section(text="Shared paragraph."), Section(text="Only in the other corpus.")],
    )
    assert embedding.calls == [["Only in the other corpus."]]
    renamed = HashEmbedding()
    renamed.model = "hash-8-v2"
    switched = Memory(memory.storage, embedding=renamed, pillars={"text", "vector", "graph"})
    await switched.ingest("acme", name="d", text="Shared paragraph.")
    assert renamed.calls == [["Shared paragraph."]]
    held = await memory.storage.chunk_vectors("acme", "hash-8", [chunk_digest("Shared paragraph.")])
    [vector] = await HashEmbedding().embed(["Shared paragraph."])
    assert list(held) == [chunk_digest("Shared paragraph.")]
    reused, settings = held[chunk_digest("Shared paragraph.")]
    assert reused == pytest.approx(vector) and settings is None


async def test_sync_swaps_the_graph_evidence_and_keeps_only_entities_still_named(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    old = await memory.ingest(
        "acme",
        name="old routine",
        text="Rung 0: V-101 feeds Motor_1.",
        metadata=PLC,
        entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="Motor_1", type="tag")],
        relations=[RelationIn(from_name="V-101", to_name="Motor_1", type="FEEDS")],
    )
    [valve] = (await memory.storage.entities_named("acme", ["V-101"]))[name_key("V-101")]
    [motor] = (await memory.storage.entities_named("acme", ["Motor_1"]))[name_key("Motor_1")]
    synced = await memory.sync(
        "acme",
        filters=LINE_A,
        sources=[
            _plc(
                "Routine",
                text="Rung 0: Motor_1 fills T-9.",
                entities=[EntityIn(name="Motor_1", type="tag"), EntityIn(name="T-9", type="tank")],
                relations=[RelationIn(from_name="Motor_1", to_name="T-9", type="FILLS")],
            ),
            _plc(
                "Tags",
                text="T-9 REAL; P-3 BOOL",
                entities=[EntityIn(name="T-9", type="tank")],
                relations=[RelationIn(from_name="P-3", to_name="T-9", type="DRAINS")],
            ),
            _plc(
                "Pumps", text="P-3 drains the tank.", entities=[EntityIn(name="P-3", type="pump")]
            ),
        ],
    )
    routine, tags, _ = synced.sources
    assert synced.deleted == Counts(sources=1, chunks=1, relations=1, mentions=2, entities=1)
    assert (
        await memory.storage.neighbors(
            "acme", [valve.entity_id], relation_types=None, window=Window()
        )
        == []
    )
    assert (await memory.storage.entities_named("acme", ["V-101"]))[name_key("V-101")] == []
    assert (await memory.storage.entities_named("acme", ["Motor_1"]))[name_key("Motor_1")] == [
        motor
    ]
    [tank] = (await memory.storage.entities_named("acme", ["T-9"]))[name_key("T-9")]
    refs = await memory.storage.mentions(
        "acme", [tank.entity_id], kinds=("chunk",), filters=None, window=Window(), subject=None
    )
    assert sorted(ref for _, ref in refs) == sorted(
        [("chunk", f"{routine.source_id}-0"), ("chunk", f"{tags.source_id}-0")]
    )
    [pump] = (await memory.storage.entities_named("acme", ["P-3"]))[name_key("P-3")]
    assert sorted(
        (r.from_id, r.type, r.evidence_id)
        for r in await memory.storage.neighbors(
            "acme", [tank.entity_id], relation_types=None, window=Window()
        )
    ) == sorted(
        [
            (motor.entity_id, "FILLS", f"{routine.source_id}-0"),
            (pump.entity_id, "DRAINS", f"{tags.source_id}-0"),
        ]
    )
    residue = await memory.verify_absent("acme")
    assert (residue.sources, residue.entities, residue.relations) == (3, 3, 2)
    assert old.source.source_id not in {s.source_id for s in synced.sources}


async def test_a_relation_no_new_source_or_stored_entity_names_is_refused(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await _seed(memory)
    before = await memory.verify_absent("acme")
    with pytest.raises(
        InputError, match=r"^memory: relation 'Ghost' -> 'T-9' names an unknown entity$"
    ):
        await memory.sync(
            "acme",
            filters=LINE_A,
            sources=[
                _plc("Tags", text="T-9 REAL", entities=[EntityIn(name="T-9", type="tank")]),
                _plc(
                    "Routine",
                    text="x",
                    relations=[RelationIn(from_name="Ghost", to_name="T-9", type="X")],
                ),
            ],
        )
    assert await memory.verify_absent("acme") == before


async def test_stories_keep_the_source_ids_they_named(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    old_ids = await _seed(memory)
    story = (await memory.remember("acme", "Motor_0 was rewired.", source_ids=(old_ids[0],))).story
    await memory.sync("acme", filters=LINE_A, sources=[_plc("Tags", text="Motor_1 BOOL")])
    assert await memory.get_stories("acme", [story.story_id]) == [story]
    assert (await memory.verify_absent("acme")).stories == 1


@pytest.mark.parametrize("corpus_id", ["", "  "])
async def test_sync_needs_a_corpus_like_every_verb(
    storage: PostgresStorage, corpus_id: str
) -> None:
    memory = await _memory(storage)
    await _seed(memory)
    before = await memory.verify_absent("acme")
    with pytest.raises(InputError, match=r"^memory: corpus_id is required$"):
        await memory.sync(corpus_id, filters=LINE_A, sources=[_plc("Tags", text="x")])
    assert await memory.verify_absent("acme") == before


def test_a_source_takes_exactly_one_of_text_or_sections() -> None:
    message = r"^models: source 'x' takes exactly one of text or sections$"
    with pytest.raises(InputError, match=message):
        SourceIn(name="x")
    with pytest.raises(InputError, match=message):
        SourceIn(name="x", text="a", sections=[Section(text="b")])
    assert SourceIn(name="x", sections=[]).sections == []


def test_the_tools_do_not_offer_sync(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    tools = MemoryTools(memory, corpus_id="acme")
    assert "sync" not in tools and "sync" not in tools.names


async def test_sources_of_one_write_that_share_a_chunk_embed_it_once(
    storage: PostgresStorage,
) -> None:
    embedding = HashEmbedding()
    memory = await _memory(storage, embedding)
    shared = Section(title="Safety", text="Lock out the line before any work.")
    await memory.sync(
        "acme",
        filters=LINE_A,
        sources=[
            _plc(f"Routine {i}", sections=[shared, Section(text=f"Rung {i} only.")])
            for i in range(3)
        ],
    )
    texts = [t for call in embedding.calls for t in call]
    assert texts.count(shared.text) == 1 and len(texts) == 4


class Reader(Scripted):
    model = "reader-1"


class TunedReader(Reader):
    settings = {"effort": "low"}


def _read(*names: str) -> Reply:
    return structured_reply(
        {"entities": [{"name": n, "type": "part"} for n in names], "relations": []}
    )


def _doc(sections: list[Section]) -> SourceIn:
    return SourceIn(name="manual", sections=sections, metadata={"file": "line-a.L5X"})


def _reading(storage: PostgresStorage, llm: Scripted) -> Memory:
    return Memory(storage, generation=llm, relation_types=["PART_OF"], pillars={"text", "graph"})


SEAL = Section(text="The seal is good.")
VALVE = Section(text="The valve is good.")
PUMP = Section(text="The pump is good.")
GASKET = Section(text="The gasket is good.")


async def test_a_changed_source_reads_only_its_changed_chunks(storage: PostgresStorage) -> None:
    llm = Reader(_read("Seal"), _read("Valve"), _read("Pump"))
    memory = Memory(storage, generation=llm, relation_types=["PART_OF"], pillars={"text", "graph"})
    one, two = Section(text="The seal is good."), Section(text="The valve is good.")
    await memory.sync("acme", filters=LINE_A, sources=[_doc([one, two])])
    assert len(llm.calls) == 2
    await memory.sync(
        "acme", filters=LINE_A, sources=[_doc([one, Section(text="The pump is good.")])]
    )
    assert len(llm.calls) == 3


async def test_a_reused_chunk_keeps_its_entities_and_who_read_it(storage: PostgresStorage) -> None:
    named = _read("Seal").model_copy(update={"model": "reader-1-march"})
    llm = Reader(named, _read("Valve"), _read("Pump"))
    memory = _reading(storage, llm)
    old = await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])])
    held, _ = await storage.source_chunks("acme", old.sources[0].source_id)
    new = await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, PUMP])])
    reused, fresh = await storage.source_chunks("acme", new.sources[0].source_id)
    assert held.extracted_by == {"model": "reader-1-march", "settings": None}
    assert reused.extracted_by == held.extracted_by
    assert fresh.extracted_by == {"model": "reader-1", "settings": None}
    hood = await memory.neighbors("acme", "Seal")
    assert [e.name for e in hood.entities] == ["Seal"]
    assert [p.id for p in hood.passages] == [reused.chunk_id]


async def test_a_new_model_or_vocabulary_reads_again(storage: PostgresStorage) -> None:
    llm = Reader(_read("Seal"))
    await _reading(storage, llm).sync("acme", filters=LINE_A, sources=[_doc([SEAL])])
    assert len(llm.calls) == 1
    renamed = Reader(_read("Seal"), _read("Valve"))
    renamed.model = "reader-2"
    await _reading(storage, renamed).sync("acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])])
    assert len(renamed.calls) == 2
    worded = Reader(_read("Seal"), _read("Valve"))
    worded.model = "reader-2"
    wider = Memory(
        storage,
        generation=worded,
        relation_types=["PART_OF", "FEEDS"],
        pillars={"text", "graph"},
    )
    await wider.sync("acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])])
    assert len(worded.calls) == 2


def test_the_key_of_a_reading_names_the_reader_and_what_it_was_told(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = Reader()
    key = extraction_key(plain, ["PART_OF", "FEEDS"])
    assert key is not None and len(key) == 64
    assert key == extraction_key(Reader(), ["FEEDS", "PART_OF"])
    assert key != extraction_key(TunedReader(), ["PART_OF", "FEEDS"])
    assert key != extraction_key(plain, ["PART_OF"])
    assert extraction_key(Scripted(), ["PART_OF"]) is None
    monkeypatch.setattr(extraction, "SYSTEM", "You read closely.")
    reworded = extraction_key(plain, ["PART_OF", "FEEDS"])
    assert key != reworded
    monkeypatch.undo()
    monkeypatch.setattr(structured, "OUTPUT_TOOL", "Emit")
    assert key != extraction_key(plain, ["PART_OF", "FEEDS"])
    monkeypatch.undo()
    monkeypatch.setattr(structured, "OUTPUT_DESCRIPTION", "Say it.")
    assert key != extraction_key(plain, ["PART_OF", "FEEDS"])
    monkeypatch.undo()
    monkeypatch.setattr(extraction, "SYSTEM", "You read closely.")

    class Wider(extraction.Extraction):
        confidence: float = 0.0

    monkeypatch.setattr(extraction, "Extraction", Wider)
    assert reworded != extraction_key(plain, ["PART_OF", "FEEDS"])


async def test_an_adapter_with_no_model_always_reads(storage: PostgresStorage, pool: Any) -> None:
    llm = Scripted(*(_read("Seal") for _ in range(5)))
    memory = _reading(storage, llm)
    await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])])
    assert len(llm.calls) == 2
    await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, SEAL, VALVE])])
    assert len(llm.calls) == 5
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT extraction, extraction_key FROM memory_chunks")
    assert len(rows) == 3
    assert all(r["extraction"] is not None and r["extraction_key"] is None for r in rows)


async def test_reuse_stays_inside_its_corpus(storage: PostgresStorage) -> None:
    llm = Reader(_read("Seal"), _read("Seal"))
    memory = _reading(storage, llm)
    await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL])])
    await memory.sync("other", filters=LINE_A, sources=[_doc([SEAL])])
    assert len(llm.calls) == 2


async def test_repeated_text_in_one_call_is_read_once(storage: PostgresStorage) -> None:
    llm = Reader(_read("Seal"), _read("Valve"))
    memory = _reading(storage, llm)
    copied = SourceIn(name="copy", sections=[SEAL], metadata={"file": "line-a.L5X"})
    synced = await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, VALVE, SEAL]), copied])
    assert len(llm.calls) == 2
    for source in synced.sources:
        chunks = await storage.source_chunks("acme", source.source_id)
        assert all(c.extracted_by is not None for c in chunks)
    hood = await memory.neighbors("acme", "Seal")
    assert len(hood.passages) == 3


async def test_a_reused_chunk_reports_no_usage(storage: PostgresStorage) -> None:
    paid = Usage(input=10, output=2)
    llm = Reader(_read("Seal"), _read("Pump").model_copy(update={"usage": paid}))
    memory = _reading(storage, llm)
    await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL])])
    seen: list[Extracted] = []

    async def keep(event: Event) -> None:
        if isinstance(event, Extracted):
            seen.append(event)

    synced = await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, PUMP])], on_event=keep)
    reused, fresh = seen
    assert reused.reused is True and reused.usage == Usage() and reused.entities == 1
    assert (reused.model, reused.settings) == ("reader-1", None)
    assert fresh.reused is False and fresh.usage == paid
    assert synced.usage == paid


async def test_caller_entities_still_land_on_a_reused_chunk(storage: PostgresStorage) -> None:
    llm = Reader(_read("Seal"))
    memory = _reading(storage, llm)
    await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL])])
    told = _doc([SEAL]).model_copy(update={"entities": [EntityIn(name="good", type="state")]})
    synced = await memory.sync("acme", filters=LINE_A, sources=[told])
    assert len(llm.calls) == 1
    [chunk] = await storage.source_chunks("acme", synced.sources[0].source_id)
    for name in ("good", "Seal"):
        assert [p.id for p in (await memory.neighbors("acme", name)).passages] == [chunk.chunk_id]


async def test_a_reading_carries_across_every_version_after_it(storage: PostgresStorage) -> None:
    llm = Reader(_read("Seal"), _read("Valve"), _read("Pump"), _read("Gasket"))
    memory = _reading(storage, llm)
    await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])])
    await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, PUMP])])
    synced = await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, GASKET])])
    assert len(llm.calls) == 4
    first, _ = await storage.source_chunks("acme", synced.sources[0].source_id)
    assert first.extracted_by == {"model": "reader-1", "settings": None}


async def test_ingest_and_what_replaces_it_read_only_what_changed(
    storage: PostgresStorage,
) -> None:
    llm = Reader(_read("Seal"), _read("Valve"), _read("Pump"))
    memory = _reading(storage, llm)
    old = await memory.ingest("acme", name="manual", sections=[SEAL, VALVE])
    new = await memory.ingest(
        "acme", name="manual", sections=[SEAL, PUMP], replaces=old.source.source_id
    )
    assert len(llm.calls) == 3
    await memory.ingest("acme", name="manual", sections=[PUMP], replaces=new.source.source_id)
    assert len(llm.calls) == 3


async def test_a_corpus_keeping_history_reads_from_its_retired_versions(
    storage: PostgresStorage,
) -> None:
    llm = Reader(_read("Seal"), _read("Valve"), _read("Pump"), _read("Gasket"))
    memory = _reading(storage, llm)
    await memory.set_history("acme", True)
    await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])])
    await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, PUMP])])
    assert len(llm.calls) == 3
    synced = await memory.sync("acme", filters=LINE_A, sources=[_doc([VALVE, PUMP])])
    assert len(llm.calls) == 3
    replacing = await memory.ingest(
        "acme", name="manual", sections=[GASKET], replaces=synced.sources[0].source_id
    )
    assert len(llm.calls) == 4
    await memory.ingest(
        "acme", name="manual", sections=[GASKET, SEAL], replaces=replacing.source.source_id
    )
    assert len(llm.calls) == 4


async def _ids(memory: Memory) -> list[str]:
    listed, _ = await memory.list_sources("acme", filters=LINE_A)
    return [s.source_id for s in listed]


async def test_keep_false_writes_every_source_anew_in_one_step(storage: PostgresStorage) -> None:
    memory = await _memory(storage, HashEmbedding())
    sources = [_plc("Routine", text="XIC(Start) OTE(Motor_1)"), _plc("Tags", text="Motor_1 BOOL")]
    first = await memory.sync("acme", filters=LINE_A, sources=sources)
    again = await memory.sync("acme", filters=LINE_A, sources=sources, keep=False)
    assert again.kept == []
    assert {s.source_id for s in again.sources}.isdisjoint(s.source_id for s in first.sources)
    assert sorted(await _ids(memory)) == sorted(s.source_id for s in again.sources)
    assert len(again.sources) == 2


async def test_keep_false_under_a_new_extractor_reads_every_chunk(storage: PostgresStorage) -> None:
    first = Reader(_read("Seal"), _read("Valve"))
    await _reading(storage, first).sync("acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])])
    held = await _ids(_reading(storage, first))
    failing = Reader(_read("Seal"))
    failing.model = "reader-2"
    with pytest.raises(AssertionError):
        await _reading(storage, failing).sync(
            "acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])], keep=False
        )
    assert await _ids(_reading(storage, first)) == held
    second = Reader(_read("Seal"), _read("Valve"))
    second.model = "reader-2"
    synced = await _reading(storage, second).sync(
        "acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])], keep=False
    )
    assert len(second.calls) == 2
    assert synced.kept == [] and await _ids(_reading(storage, second)) != held


async def test_keep_false_under_the_same_extractor_reads_nothing(storage: PostgresStorage) -> None:
    llm = Reader(_read("Seal"), _read("Valve"))
    memory = _reading(storage, llm)
    await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])])
    synced = await memory.sync("acme", filters=LINE_A, sources=[_doc([SEAL, VALVE])], keep=False)
    assert len(llm.calls) == 2
    assert synced.kept == [] and synced.usage == Usage()


async def test_keep_false_with_history_makes_a_version_of_each_source(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage, HashEmbedding())
    await memory.set_history("acme", True)
    sources = [_plc("Routine", text="XIC(Start) OTE(Motor_1)"), _plc("Tags", text="Motor_1 BOOL")]
    first = await memory.sync("acme", filters=LINE_A, sources=sources)
    again = await memory.sync("acme", filters=LINE_A, sources=sources, keep=False)
    assert sorted(again.retired) == sorted(s.source_id for s in first.sources)
    assert again.kept == []
    retired, _ = await memory.list_sources("acme", filters=LINE_A, history=True)
    assert len(retired) == 4


async def test_keep_false_with_a_document_model_embeds_every_chunk(
    storage: PostgresStorage,
) -> None:
    embedding = DocumentHashEmbedding()
    memory = await _memory(storage, embedding)
    sources = [_plc("Routine", text="XIC(Start) OTE(Motor_1)"), _plc("Tags", text="Motor_1 BOOL")]
    await memory.sync("acme", filters=LINE_A, sources=sources)
    assert len(embedding.documents) == 2
    await memory.sync("acme", filters=LINE_A, sources=sources, keep=False)
    assert len(embedding.documents) == 4


async def test_keep_must_be_a_bool(storage: PostgresStorage) -> None:
    memory = await _memory(storage, HashEmbedding())
    with pytest.raises(InputError):
        await memory.sync("acme", filters=LINE_A, sources=[], keep="no")  # type: ignore[arg-type]
