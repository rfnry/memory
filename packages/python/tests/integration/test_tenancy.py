from datetime import UTC, datetime
from typing import Any

import pytest
from unit.fakes import Fixed, HashEmbedding, Scripted, structured_reply

from rfnry_memory.errors import MemoryException, NotFoundError
from rfnry_memory.index import HNSW
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    Counts,
    Decided,
    Entity,
    EntityIn,
    RelationIn,
    Scope,
    ScoreAnswer,
    SourceIn,
    Story,
    Window,
)
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.ranking import BM25
from rfnry_memory.rows import GraphWrite, name_key
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


async def test_nothing_of_a_is_reachable_from_b(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    source_id, story_id = await _seed(memory, "tenant-a", MARKER)
    assert await memory.set_vector_index("tenant-a", HNSW()) == 2
    await _seed(memory, "tenant-b", "beta")
    before = await memory.verify_absent("tenant-a")
    assert (before.vectors, before.corpora) == (2, 2)
    fingerprint = await memory.fingerprint("tenant-a")
    for query in (MARKER, "valve", "story"):
        assert not _leaks(await memory.search("tenant-b", query))
        assert not _leaks(await memory.recall("tenant-b", query))
    assert not _leaks(await memory.list_sources("tenant-b"))
    assert not _leaks(await memory.list_stories("tenant-b"))
    assert not _leaks(await memory.neighbors("tenant-b", "P-1", hops=3))
    named = await storage.entities_named("tenant-b", [f"V-{MARKER}", "p-1", "P-1 "])
    assert named[name_key(f"V-{MARKER}")] == []
    assert [e.corpus_id for e in named["p-1"]] == ["tenant-b"]
    assert not _leaks(list(named.values()))
    for query in ("P-1", f"V-{MARKER}"):
        similar = await storage.similar_entities("tenant-b", query, limit=50)
        assert all(e.corpus_id == "tenant-b" for e in similar) and not _leaks(similar)
    assert [e.name for e in await storage.similar_entities("tenant-b", "P-1", limit=50)] == ["P-1"]
    for spot in ("words", "exact", "substring"):
        assert await storage.entities_named_in("tenant-b", f"is V-{MARKER} open", spot=spot) == []
    assert await memory.get_source("tenant-b", source_id) is None
    assert await memory.get_stories("tenant-b", [story_id]) == []
    for call in (
        lambda: memory.read_source("tenant-b", source_id),
        lambda: memory.delete_source("tenant-b", source_id),
        lambda: memory.delete_story("tenant-b", story_id),
        lambda: memory.supersede("tenant-b", story_id, "x"),
        lambda: memory.end_story("tenant-b", story_id),
        lambda: memory.ingest("tenant-b", name="x", text="x", replaces=source_id),
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
    await _seed(memory, "tenant-c", "gamma")
    for query in (MARKER, "valve", "story"):
        assert not _leaks(await memory.search(["tenant-b", "tenant-c"], query))
        assert not _leaks(await memory.recall(["tenant-c", "tenant-b"], query))
    several = MemoryTools(memory, corpus_id=["tenant-b", "tenant-c"])
    assert not _leaks(await several.dispatch("search", {"query": MARKER}))
    with pytest.raises(MemoryException):
        await several.dispatch("search", {"query": MARKER, "corpus_id": "tenant-a"})
    with pytest.raises(MemoryException):
        await several.dispatch("read_source", {"source_id": source_id, "corpus_id": "tenant-a"})
    for corpus in ("tenant-b", "tenant-c"):
        with pytest.raises(NotFoundError):
            await several.dispatch("read_source", {"source_id": source_id, "corpus_id": corpus})
    tools = MemoryTools(memory, corpus_id="tenant-b")
    for name, ids in (
        ("read_source", {"source_id": source_id}),
        ("delete_story", {"story_id": story_id}),
        ("supersede", {"story_id": story_id, "text": "x"}),
        ("end_story", {"story_id": story_id}),
        ("ingest", {"name": "x", "text": "x", "replaces": source_id}),
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
            "story_ids": [story_id],
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


async def test_erasure_is_provable_and_falsifiable(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await _seed(memory, "tenant-a", MARKER)
    assert await memory.set_vector_index("tenant-a", HNSW()) == 2
    assert not (await memory.verify_absent("tenant-a")).clean
    deleted = await memory.delete_corpus("tenant-a")
    assert (
        deleted.sources == 1
        and deleted.stories == 1
        and deleted.entities == 2
        and deleted.relations == 1
        and deleted.vectors == 2
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


async def test_retired_versions_of_a_never_reach_b_and_erase_with_a(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    marked = {"metadata.doc": MARKER}
    await memory.set_history("tenant-a", True)
    for text in (f"{MARKER} first.", f"{MARKER} second."):
        synced = await memory.sync(
            "tenant-a",
            filters=marked,
            sources=[SourceIn(name="doc", text=text, metadata={"doc": MARKER})],
        )
    old_id = synced.retired[0]
    await _seed(memory, "tenant-b", "beta")
    assert _leaks(await memory.list_sources("tenant-a", history=True))
    assert _leaks(await memory.search("tenant-a", MARKER, history=True))
    assert not _leaks(await memory.list_sources("tenant-b", history=True))
    assert not _leaks(await memory.search("tenant-b", MARKER, history=True))
    assert await memory.get_source("tenant-b", old_id) is None
    assert (await memory.forget_retired("tenant-b", before=datetime(2100, 1, 1, tzinfo=UTC))).clean
    with pytest.raises(NotFoundError):
        await memory.delete_source("tenant-b", old_id)
    assert await memory.get_source("tenant-a", old_id) is not None
    await memory.delete_corpus("tenant-a")
    assert (await memory.verify_absent("tenant-a")).clean


async def test_the_reads_a_model_drives_read_no_other_corpus(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    source_id, story_id = await _seed(memory, "tenant-a", MARKER)
    await _seed(memory, "tenant-b", "beta")
    chunk, story = ("chunk", f"{source_id}-0"), ("story", story_id)
    assert await storage.passage_vector("tenant-a", chunk) is not None
    for ref in (chunk, story):
        assert await storage.passage_vector("tenant-b", ref) is None
    assert await storage.passage_entities("tenant-b", [chunk, story]) == []
    assert (
        await storage.chunks_around(
            "tenant-b", source_id, 0, width=5, window=Window(), filters=None
        )
        == []
    )
    assert await storage.graph_types("tenant-b") == ({"pump": 1, "valve": 1}, {"CONNECTS_TO": 1})
    [valve] = (await storage.entities_named("tenant-a", [f"V-{MARKER}"]))[name_key(f"V-{MARKER}")]
    inside = Scope(sources=(source_id,), passages=(chunk, story))
    assert (
        await storage.mentions(
            "tenant-b",
            [valve.entity_id],
            kinds=("chunk", "story"),
            filters=None,
            window=Window(),
            subject=None,
            scope=inside,
        )
        == []
    )
    assert (
        await storage.search_text(
            ["tenant-b"],
            MARKER,
            kinds=("chunk", "story"),
            top_k=10,
            filters=None,
            window=Window(),
            subject=None,
            ranking=BM25(),
            match="any",
            scope=inside,
        )
        == []
    )


async def test_what_built_a_graph_is_read_and_erased_with_its_corpus(
    storage: PostgresStorage, pool: Any
) -> None:
    said = structured_reply(
        {"entities": [{"name": f"V-{MARKER} unit", "type": "valve"}], "relations": []}
    )
    deciding = Fixed(
        lambda state, questions: Decided(
            answers={"same": ScoreAnswer(score=1.9, probabilities=[0.05, 0.05, 0.9])},
            model=f"decider-{MARKER}",
        )
    )

    class Marked(Scripted):
        model = f"gen-{MARKER}"

    marked = Memory(
        storage,
        generation=Marked(said, said),
        decision=deciding,
        relation_types=["CONNECTS_TO"],
    )
    await _seed(marked, "tenant-a", MARKER)
    plain = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await _seed(plain, "tenant-b", "beta")
    [valve] = (await storage.entities_named("tenant-a", [f"V-{MARKER}"]))[name_key(f"V-{MARKER}")]
    assert valve.aliased_by and _leaks(valve.aliased_by)
    assert not _leaks(await plain.neighbors("tenant-b", "P-1", hops=3))
    assert not _leaks(await plain.search("tenant-b", "valve story"))
    assert not _leaks(await plain.list_stories("tenant-b"))
    names = "SELECT count(*) FROM memory_entity_names WHERE corpus_id = 'tenant-a'"
    async with pool.acquire() as conn:
        assert await conn.fetchval(names) > 0
    await marked.delete_corpus("tenant-a")
    assert (await marked.verify_absent("tenant-a")).clean
    async with pool.acquire() as conn:
        assert await conn.fetchval(names) == 0
        held = await conn.fetchval(
            "SELECT count(*) FROM (SELECT extracted_by::text AS t FROM memory_chunks UNION ALL "
            "SELECT extracted_by::text FROM memory_stories UNION ALL "
            "SELECT aliased_by::text || coalesce(resolved_by::text, '') FROM memory_entities) AS p "
            "WHERE t LIKE '%' || $1 || '%'",
            MARKER,
        )
    assert held == 0


async def test_the_closings_of_a_never_reach_b_and_erase_with_a(storage: PostgresStorage) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]

    def numbered() -> Any:
        counter = iter(range(1, 100))
        return lambda: f"s-{next(counter)}"

    held = {
        corpus: Memory(storage, embedding=HashEmbedding(), clock=lambda: clock[0], ids=numbered())
        for corpus in ("tenant-a", "tenant-b")
    }
    ends = {
        "tenant-a": (datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 1, 25, tzinfo=UTC)),
        "tenant-b": (datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC)),
    }
    for corpus, memory in held.items():
        word = MARKER if corpus == "tenant-a" else "beta"
        told = await memory.remember(
            corpus, f"{word} holds.", valid_from=datetime(2026, 1, 5, tzinfo=UTC)
        )
        assert told.story.story_id == "s-1"
    for moment, step in (
        (datetime(2026, 1, 20, tzinfo=UTC), 0),
        (datetime(2026, 1, 30, tzinfo=UTC), 1),
    ):
        clock[0] = moment
        for corpus, memory in held.items():
            await memory.end_story(corpus, "s-1", valid_to=ends[corpus][step])
    clock[0] = datetime(2026, 2, 5, tzinfo=UTC)
    believed = datetime(2026, 1, 25, tzinfo=UTC)
    for corpus, memory in held.items():
        [then], _ = await memory.list_stories(corpus, known_at=believed, history=True)
        assert then.valid_to == ends[corpus][0], corpus
        hits = await memory.recall(
            corpus, "holds", known_at=believed, as_of=datetime(2026, 1, 28, tzinfo=UTC)
        )
        assert [h.story.valid_to for h in hits.hits if h.story] == [ends[corpus][0]], corpus
    b = held["tenant-b"]
    assert not _leaks(await b.list_stories("tenant-b", known_at=believed, history=True))
    assert (await b.verify_absent("tenant-a")).closings == 1
    assert (await b.verify_absent("tenant-b")).closings == 1
    deleted = await b.delete_story("tenant-b", "s-1")
    assert (deleted.stories, deleted.closings) == (1, 1)
    assert await b.verify_absent("tenant-b") == Counts(corpora=1)
    erased = await held["tenant-a"].delete_corpus("tenant-a")
    assert (erased.stories, erased.closings) == (1, 1)
    assert (await b.verify_absent("tenant-a")).clean
