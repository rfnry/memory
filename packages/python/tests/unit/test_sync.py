from datetime import UTC, datetime
from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    Counts,
    EntityIn,
    RelationIn,
    Section,
    SourceIn,
    Usage,
    Window,
)
from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.storage.storage import chunk_digest
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding

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


async def _memory(embedding: HashEmbedding | None = None) -> Memory:
    memory = Memory(
        InMemoryStorage(),
        embedding=embedding or HashEmbedding(),
        pillars={"text", "vector", "graph"},
    )
    await memory.initialize()
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


async def test_sync_swaps_the_matching_set() -> None:
    memory = await _memory()
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


async def test_a_model_that_fails_before_the_write_leaves_the_old_set_whole() -> None:
    embedding = _Failing()
    memory = await _memory(embedding)
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


async def test_a_source_outside_the_filters_is_refused_before_any_model_runs() -> None:
    embedding = HashEmbedding()
    memory = await _memory(embedding)
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
async def test_sync_without_filters_is_refused(filters: dict[str, Any] | None) -> None:
    memory = await _memory()
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


async def test_sync_with_no_sources_deletes_the_set() -> None:
    memory = await _memory()
    await _seed(memory)
    synced = await memory.sync("acme", filters=LINE_A, sources=[])
    assert (synced.sources, synced.deleted, synced.usage) == (
        [],
        Counts(sources=2, chunks=2),
        Usage(),
    )
    assert await _names(memory) == []
    assert await _names(memory, filters={"metadata.file": "line-b.L5X"}) == ["other file"]


async def test_a_retry_with_the_same_input_keeps_the_set_as_it_is() -> None:
    embedding = HashEmbedding()
    memory = await _memory(embedding)
    await _seed(memory)
    sources = [_plc("Routine", text="XIC(Start) OTE(Motor_1)"), _plc("Tags", text="Motor_1 BOOL")]
    first = await memory.sync("acme", filters=LINE_A, sources=sources)
    calls = len(embedding.calls)
    fingerprint = await memory.fingerprint("acme")
    second = await memory.sync("acme", filters=LINE_A, sources=sources)
    assert second.sources == first.sources
    assert second.kept == [s.source_id for s in first.sources]
    assert second.deleted.clean and second.usage == Usage()
    assert len(embedding.calls) == calls
    assert await _names(memory) == ["Routine", "Tags"]
    assert await memory.fingerprint("acme") == fingerprint


async def test_a_sync_rewrites_only_what_changed() -> None:
    embedding = HashEmbedding()
    memory = await _memory(embedding)
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
    assert second.kept == [first.sources[1].source_id] and kept_tags == first.sources[1]
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


async def test_a_change_of_chunking_or_pillars_rewrites_every_source() -> None:
    storage = InMemoryStorage()
    sources = [_plc("Tags", text="Motor_1 BOOL")]
    memory = Memory(storage, embedding=HashEmbedding(), chunk_chars=300, overlap_chars=30)
    await memory.initialize()
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


async def test_a_list_that_repeats_a_source_keeps_one_and_writes_the_other() -> None:
    memory = await _memory()
    tags = _plc("Tags", text="Motor_1 BOOL")
    first = await memory.sync("acme", filters=LINE_A, sources=[tags])
    second = await memory.sync("acme", filters=LINE_A, sources=[tags, tags])
    assert second.kept == [first.sources[0].source_id]
    assert (
        second.sources[0] == first.sources[0]
        and second.sources[1].source_id != first.sources[0].source_id
    )
    assert second.deleted.clean and await _names(memory) == ["Tags", "Tags"]


async def test_a_set_that_moves_during_the_sync_is_written_whole() -> None:
    memory = await _memory()
    tags = _plc("Tags", text="Motor_1 BOOL")
    first = await memory.sync("acme", filters=LINE_A, sources=[tags])
    storage = memory.storage
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


async def test_a_chunk_the_corpus_already_holds_is_not_embedded_again() -> None:
    embedding = HashEmbedding()
    memory = await _memory(embedding)
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
    assert held == {
        chunk_digest("Shared paragraph."): (await HashEmbedding().embed(["Shared paragraph."]))[0]
    }


async def test_sync_swaps_the_graph_evidence_and_keeps_only_entities_still_named() -> None:
    memory = await _memory()
    old = await memory.ingest(
        "acme",
        name="old routine",
        text="Rung 0: V-101 feeds Motor_1.",
        metadata=PLC,
        entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="Motor_1", type="tag")],
        relations=[RelationIn(from_name="V-101", to_name="Motor_1", type="FEEDS")],
    )
    [valve] = await memory.storage.entities_named("acme", "V-101")
    [motor] = await memory.storage.entities_named("acme", "Motor_1")
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
    assert await memory.storage.entities_named("acme", "V-101") == []
    assert await memory.storage.entities_named("acme", "Motor_1") == [motor]
    [tank] = await memory.storage.entities_named("acme", "T-9")
    refs = await memory.storage.mentions(
        "acme", [tank.entity_id], kinds=("chunk",), filters=None, window=Window(), subject=None
    )
    assert sorted(refs) == sorted(
        [("chunk", f"{routine.source_id}-0"), ("chunk", f"{tags.source_id}-0")]
    )
    [pump] = await memory.storage.entities_named("acme", "P-3")
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


async def test_a_relation_no_new_source_or_stored_entity_names_is_refused() -> None:
    memory = await _memory()
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


async def test_stories_keep_the_source_ids_they_named() -> None:
    memory = await _memory()
    old_ids = await _seed(memory)
    story = (await memory.remember("acme", "Motor_0 was rewired.", source_ids=(old_ids[0],))).story
    await memory.sync("acme", filters=LINE_A, sources=[_plc("Tags", text="Motor_1 BOOL")])
    assert await memory.get_stories("acme", [story.story_id]) == [story]
    assert (await memory.verify_absent("acme")).stories == 1


@pytest.mark.parametrize("corpus_id", ["", "  "])
async def test_sync_needs_a_corpus_like_every_verb(corpus_id: str) -> None:
    memory = await _memory()
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


def test_the_tools_do_not_offer_sync() -> None:
    memory = Memory(InMemoryStorage(), pillars={"text", "graph"})
    tools = MemoryTools(memory, corpus_id="acme")
    assert "sync" not in tools and "sync" not in tools.names


async def test_sources_of_one_write_that_share_a_chunk_embed_it_once() -> None:
    embedding = HashEmbedding()
    memory = await _memory(embedding)
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
