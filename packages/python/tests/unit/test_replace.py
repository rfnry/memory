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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail_on is not None and any(self.fail_on in t for t in texts):
            raise RuntimeError("the embedding service is down")
        return await super().embed(texts)


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


async def test_replace_swaps_the_matching_set() -> None:
    memory = await _memory()
    old_ids = await _seed(memory)
    replaced = await memory.replace(
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
    assert {s.source_id for s in listed} == {s.source_id for s in replaced.sources}
    assert not {s.source_id for s in replaced.sources} & set(old_ids)
    assert replaced.deleted == Counts(sources=2, chunks=2) and replaced.usage == Usage()
    assert [s.chunks for s in replaced.sources] == [2, 1]
    assert [await memory.get_source("acme", i) for i in old_ids] == [None, None]
    routine = replaced.sources[0]
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
        await memory.replace(
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
            InputError, match=r"^memory: source 'Tags' does not match the filters it replaces$"
        ):
            await memory.replace(
                "acme", filters=LINE_A, sources=[_plc("Routine", text="XIC(Start)"), stray]
            )
    assert len(embedding.calls) == calls and await memory.verify_absent("acme") == before


@pytest.mark.parametrize("filters", [{}, None])
async def test_replace_without_filters_is_refused(filters: dict[str, Any] | None) -> None:
    memory = await _memory()
    await _seed(memory)
    before = await memory.verify_absent("acme")
    with pytest.raises(
        InputError,
        match=r"^memory: replace needs filters naming the set it replaces — delete_corpus empties a corpus$",
    ):
        await memory.replace("acme", filters=filters, sources=[])  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"^filters: filter key 'corpus_id' is reserved"):
        await memory.replace("acme", filters={"corpus_id": "other"}, sources=[])
    assert await memory.verify_absent("acme") == before


async def test_replace_with_no_sources_deletes_the_set() -> None:
    memory = await _memory()
    await _seed(memory)
    replaced = await memory.replace("acme", filters=LINE_A, sources=[])
    assert (replaced.sources, replaced.deleted, replaced.usage) == (
        [],
        Counts(sources=2, chunks=2),
        Usage(),
    )
    assert await _names(memory) == []
    assert await _names(memory, filters={"metadata.file": "line-b.L5X"}) == ["other file"]


async def test_a_retry_with_the_same_input_leaves_one_set() -> None:
    memory = await _memory()
    await _seed(memory)
    sources = [_plc("Routine", text="XIC(Start) OTE(Motor_1)"), _plc("Tags", text="Motor_1 BOOL")]
    first = await memory.replace("acme", filters=LINE_A, sources=sources)
    second = await memory.replace("acme", filters=LINE_A, sources=sources)
    assert await _names(memory) == ["Routine", "Tags"]
    assert second.deleted == Counts(sources=2, chunks=2)
    listed, _ = await memory.list_sources("acme", filters=LINE_A)
    assert {s.source_id for s in listed} == {s.source_id for s in second.sources}
    assert not {s.source_id for s in first.sources} & {s.source_id for s in second.sources}


async def test_replace_swaps_the_graph_evidence_and_keeps_the_entities() -> None:
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
    replaced = await memory.replace(
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
    routine, tags, _ = replaced.sources
    assert replaced.deleted == Counts(sources=1, chunks=1, relations=1, mentions=2)
    assert await memory.storage.neighbors("acme", [valve.entity_id], relation_types=None) == []
    assert await memory.storage.entities_named("acme", "V-101") == [valve]
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
        for r in await memory.storage.neighbors("acme", [tank.entity_id], relation_types=None)
    ) == sorted(
        [
            (motor.entity_id, "FILLS", f"{routine.source_id}-0"),
            (pump.entity_id, "DRAINS", f"{tags.source_id}-0"),
        ]
    )
    residue = await memory.verify_absent("acme")
    assert (residue.sources, residue.entities, residue.relations) == (3, 4, 2)
    assert old.source.source_id not in {s.source_id for s in replaced.sources}


async def test_a_relation_no_new_source_or_stored_entity_names_is_refused() -> None:
    memory = await _memory()
    await _seed(memory)
    before = await memory.verify_absent("acme")
    with pytest.raises(
        InputError, match=r"^memory: relation 'Ghost' -> 'T-9' names an unknown entity$"
    ):
        await memory.replace(
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
    await memory.replace("acme", filters=LINE_A, sources=[_plc("Tags", text="Motor_1 BOOL")])
    assert await memory.get_stories("acme", [story.story_id]) == [story]
    assert (await memory.verify_absent("acme")).stories == 1


@pytest.mark.parametrize("corpus_id", ["", "  "])
async def test_replace_needs_a_corpus_like_every_verb(corpus_id: str) -> None:
    memory = await _memory()
    await _seed(memory)
    before = await memory.verify_absent("acme")
    with pytest.raises(InputError, match=r"^memory: corpus_id is required$"):
        await memory.replace(corpus_id, filters=LINE_A, sources=[_plc("Tags", text="x")])
    assert await memory.verify_absent("acme") == before


def test_a_source_takes_exactly_one_of_text_or_sections() -> None:
    message = r"^models: source 'x' takes exactly one of text or sections$"
    with pytest.raises(InputError, match=message):
        SourceIn(name="x")
    with pytest.raises(InputError, match=message):
        SourceIn(name="x", text="a", sections=[Section(text="b")])
    assert SourceIn(name="x", sections=[]).sections == []


def test_the_tools_do_not_offer_replace() -> None:
    memory = Memory(InMemoryStorage(), pillars={"text", "graph"})
    tools = MemoryTools(memory, corpus_id="acme")
    assert "replace" not in tools and "replace" not in tools.names
