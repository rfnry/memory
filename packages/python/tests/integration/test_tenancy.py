from datetime import UTC, datetime

import pytest
from unit.fakes import HashEmbedding

from rfnry_memory.errors import MemoryException, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Entity, EntityIn, RelationIn, SourceIn, Story, Window
from rfnry_memory.storage.storage import GraphWrite, Storage
from rfnry_memory.tools import MemoryTools

MARKER = "MARKER-ALPHA-SECRET"


async def _seed(memory: Memory, corpus: str, marker: str) -> tuple[str, str]:
    ingested = await memory.ingest(
        corpus,
        name="doc",
        text=f"The {marker} valve V-{marker} feeds pump P-1.",
        metadata={"line": marker},
        entities=[EntityIn(name=f"V-{marker}", type="valve"), EntityIn(name="P-1", type="pump")],
        relations=[RelationIn(from_name=f"V-{marker}", to_name="P-1", type="CONNECTS_TO")],
    )
    remembered = await memory.remember(
        corpus,
        f"{marker} story about the valve",
        subject=f"asset:{marker}",
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return ingested.source.source_id, remembered.story.story_id


def _leaks(payload: object) -> bool:
    return MARKER in str(payload)


async def test_nothing_of_a_is_reachable_from_b(storage: Storage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    source_id, story_id = await _seed(memory, "tenant-a", MARKER)
    await _seed(memory, "tenant-b", "beta")
    before = await memory.verify_absent("tenant-a")
    fingerprint = await memory.fingerprint("tenant-a")
    for query in (MARKER, "valve", "story"):
        assert not _leaks(await memory.search("tenant-b", query))
        assert not _leaks(await memory.recall("tenant-b", query))
    assert not _leaks(await memory.list_sources("tenant-b"))
    assert not _leaks(await memory.list_stories("tenant-b"))
    assert not _leaks(await memory.neighbors("tenant-b", "P-1", hops=3))
    assert await storage.entities_named("tenant-b", f"v-{MARKER.lower()}") == []
    assert await storage.entities_named_in("tenant-b", f"is V-{MARKER} open") == []
    assert await memory.get_source("tenant-b", source_id) is None
    assert await memory.get_stories("tenant-b", [story_id]) == []
    for call in (
        lambda: memory.read_source("tenant-b", source_id),
        lambda: memory.delete_source("tenant-b", source_id),
        lambda: memory.delete_story("tenant-b", story_id),
        lambda: memory.supersede("tenant-b", story_id, "x"),
    ):
        with pytest.raises(NotFoundError):
            await call()
    assert (
        await memory.sync("tenant-b", filters={"metadata.line": MARKER}, sources=[])
    ).deleted.clean
    renamed = HashEmbedding()
    renamed.model = "hash-8-v2"
    assert await Memory(storage, embedding=renamed).reembed("tenant-b") == 2
    stale = await storage.passages_embedded_with("tenant-a", not_model="hash-8-v2", limit=10)
    assert len(stale) == 2
    tools = MemoryTools(memory, corpus_id="tenant-b")
    for name, ids in (
        ("read_source", {"source_id": source_id}),
        ("delete_story", {"story_id": story_id}),
        ("supersede", {"story_id": story_id, "text": "x"}),
    ):
        with pytest.raises(NotFoundError):
            await tools.dispatch(name, ids)
    for definition in tools.definitions():
        name = definition["name"]
        args = {
            "query": MARKER,
            "name": "x",
            "text": "x",
            "story_id": story_id,
            "source_id": source_id,
            "entity": "P-1",
            "subject": "asset:beta",
        }
        wanted = {k: v for k, v in args.items() if k in definition["input_schema"]["properties"]}
        try:
            answered: object = await tools.dispatch(name, wanted)
        except MemoryException as error:
            answered = str(error)
        assert not _leaks(answered), name
    written = await memory.sync(
        "tenant-b",
        filters={"metadata.line": MARKER},
        sources=[SourceIn(name="b's own", text="b writes here", metadata={"line": MARKER})],
    )
    assert written.deleted.clean and [s.corpus_id for s in written.sources] == ["tenant-b"]
    assert [s.source_id for s in (await memory.list_sources("tenant-a"))[0]] == [source_id]
    assert await memory.verify_absent("tenant-a") == before
    assert await memory.fingerprint("tenant-a") == fingerprint


async def test_erasure_is_provable_and_falsifiable(storage: Storage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await _seed(memory, "tenant-a", MARKER)
    assert not (await memory.verify_absent("tenant-a")).clean
    deleted = await memory.delete_corpus("tenant-a")
    assert (
        deleted.sources == 1
        and deleted.stories == 1
        and deleted.entities == 2
        and deleted.relations == 1
    )
    assert (await memory.verify_absent("tenant-a")).clean
    orphan = Entity(corpus_id="tenant-a", entity_id="orphan", name="o", key="o", type="t")
    await storage.put_story(
        "tenant-a",
        Story(
            corpus_id="tenant-a",
            story_id="late",
            text="late",
            subject=None,
            valid_from=datetime(2026, 1, 1, tzinfo=UTC),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            valid_to=None,
            supersedes=None,
            source_ids=(),
            metadata={},
        ),
        None,
        None,
        graph=GraphWrite(entities=[orphan]),
    )
    assert not (await memory.verify_absent("tenant-a")).clean
    assert (
        await storage.mentions(
            "tenant-a",
            ["orphan"],
            kinds=("chunk", "story"),
            filters=None,
            window=Window(),
            subject=None,
        )
        == []
    )
