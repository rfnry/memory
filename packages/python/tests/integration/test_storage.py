from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from rfnry_memory.errors import InputError, NotFoundError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Chunk, Entity, Leg, Relation, Source, Story, Window
from rfnry_memory.storage.storage import Storage, text_hash

NOW = datetime(2026, 3, 1, tzinfo=UTC)


def ids(legs: list[Leg]) -> list[str]:
    return [i for _, i, _ in legs]


def _vec(text: str) -> list[float]:
    return [b / 255 for b in sha256(text.lower().encode()).digest()[:8]]


def _source(corpus: str, source_id: str, texts: list[str], *, metadata: dict[str, object] | None = None) -> tuple[Source, list[Chunk], list[list[float]]]:
    source = Source(corpus_id=corpus, source_id=source_id, name=source_id, metadata=metadata or {}, text="\n".join(texts), created_at=NOW, chunks=len(texts))
    chunks = [Chunk(corpus_id=corpus, chunk_id=f"{source_id}-{i}", source_id=source_id, position=i, title=None, locator={"page": i}, text=t) for i, t in enumerate(texts)]
    return source, chunks, [_vec(t) for t in texts]


def _story(corpus: str, story_id: str, text: str, *, subject: str | None = None, occurred_at: datetime | None = None, valid_to: datetime | None = None, metadata: dict[str, object] | None = None) -> Story:
    return Story(corpus_id=corpus, story_id=story_id, text=text, subject=subject, occurred_at=occurred_at, created_at=NOW, valid_to=valid_to, supersedes=None, source_ids=(), metadata=metadata or {})


async def test_sources_roundtrip_list_and_delete(storage: Storage) -> None:
    source, chunks, vectors = _source("a", "s1", ["pump seal replacement", "wiring of the panel"], metadata={"line": "A"})
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    assert await storage.get_source("a", "s1") == source
    assert await storage.get_source("b", "s1") is None
    assert [c.chunk_id for c in await storage.source_chunks("a", "s1")] == ["s1-0", "s1-1"]
    listed, cursor = await storage.list_sources("a", filters={"metadata.line": "A"}, limit=10, cursor=None)
    assert [s.source_id for s in listed] == ["s1"] and cursor is None
    listed, _ = await storage.list_sources("a", filters={"metadata.line": "B"}, limit=10, cursor=None)
    assert listed == []
    deleted = await storage.delete_source("a", "s1")
    assert deleted.sources == 1 and deleted.chunks == 2
    assert await storage.get_source("a", "s1") is None and await storage.source_chunks("a", "s1") == []


async def test_list_sources_pages_with_a_cursor(storage: Storage) -> None:
    for i in range(5):
        source, chunks, vectors = _source("a", f"s{i}", [f"text {i}"])
        await storage.put_source("a", source.model_copy(update={"created_at": NOW + timedelta(minutes=i)}), chunks, vectors, "hash-8")
    first, cursor = await storage.list_sources("a", filters=None, limit=2, cursor=None)
    assert [s.source_id for s in first] == ["s4", "s3"] and cursor is not None
    second, cursor = await storage.list_sources("a", filters=None, limit=2, cursor=cursor)
    assert [s.source_id for s in second] == ["s2", "s1"]
    third, cursor = await storage.list_sources("a", filters=None, limit=2, cursor=cursor)
    assert [s.source_id for s in third] == ["s0"] and cursor is None


async def test_text_search_ranks_by_term_match_and_honours_kind_and_filters(storage: Storage) -> None:
    source, chunks, vectors = _source("a", "s1", ["the pump seal must be replaced yearly", "wiring diagram of the control panel"], metadata={"line": "A"})
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    await storage.put_story("a", _story("a", "st1", "the seal on pump P-3 was replaced", metadata={"line": "B"}), _vec("seal"), "hash-8")
    legs = await storage.search_text("a", "pump seal", kinds=("chunk", "story"), top_k=10, filters=None, window=Window())
    assert [(k, i) for k, i, _ in legs][:2] == [("chunk", "s1-0"), ("story", "st1")] or [(k, i) for k, i, _ in legs][:2] == [("story", "st1"), ("chunk", "s1-0")]
    assert all(score > 0 for _, _, score in legs) and ("chunk", "s1-1") not in [(k, i) for k, i, _ in legs]
    only_chunks = await storage.search_text("a", "pump seal", kinds=("chunk",), top_k=10, filters=None, window=Window())
    assert [i for _, i, _ in only_chunks] == ["s1-0"]
    faceted = await storage.search_text("a", "pump seal", kinds=("chunk", "story"), top_k=10, filters={"metadata.line": "B"}, window=Window())
    assert [i for _, i, _ in faceted] == ["st1"]
    assert await storage.search_text("b", "pump seal", kinds=("chunk", "story"), top_k=10, filters=None, window=Window()) == []


async def test_vector_search_orders_by_cosine(storage: Storage) -> None:
    source, chunks, vectors = _source("a", "s1", ["alpha", "beta", "gamma"])
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    legs = await storage.search_vector("a", _vec("beta"), kinds=("chunk",), top_k=2, filters=None, window=Window())
    assert legs[0][1] == "s1-1" and legs[0][2] == pytest.approx(1.0) and len(legs) == 2


async def test_story_windows_apply_inside_search(storage: Storage) -> None:
    jan, feb, mar = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3))
    await storage.put_story("a", _story("a", "old", "go-live is february twenty", subject="apollo", occurred_at=jan, valid_to=feb), _vec("go-live"), "hash-8")
    await storage.put_story("a", _story("a", "new", "go-live is march three", subject="apollo", occurred_at=feb), _vec("go-live"), "hash-8")
    assert set(ids(await storage.search_text("a", "go-live", kinds=("story",), top_k=10, filters=None, window=Window()))) == {"old", "new"}
    assert ids(await storage.search_text("a", "go-live", kinds=("story",), top_k=10, filters=None, window=Window(as_of=jan + timedelta(days=10)))) == ["old"]
    assert ids(await storage.search_text("a", "go-live", kinds=("story",), top_k=10, filters=None, window=Window(as_of=mar))) == ["new"]
    assert ids(await storage.search_vector("a", _vec("go-live"), kinds=("story",), top_k=10, filters=None, window=Window(since=feb))) == ["new"]
    assert ids(await storage.search_vector("a", _vec("go-live"), kinds=("story",), top_k=10, filters=None, window=Window(until=jan))) == ["old"]
    listed, _ = await storage.list_stories("a", subject="apollo", window=Window(as_of=mar), filters=None, limit=10, cursor=None)
    assert [s.story_id for s in listed] == ["new"]


async def test_supersede_is_one_step_and_hash_finds_only_open_stories(storage: Storage) -> None:
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    old = _story("a", "old", "The date is Feb 20.", occurred_at=jan)
    await storage.put_story("a", old, None, None)
    assert (await storage.find_open_story("a", text_hash(old.text))) == old
    new = _story("a", "new", "The date is Mar 3.", occurred_at=feb).model_copy(update={"supersedes": "old"})
    closed, created = await storage.supersede_story("a", "old", feb, new, None, None)
    assert closed.valid_to == feb and created.supersedes == "old"
    assert await storage.find_open_story("a", text_hash(old.text)) is None
    assert await storage.find_open_story("a", text_hash(new.text)) == created
    with pytest.raises(NotFoundError):
        await storage.supersede_story("a", "ghost", feb, new.model_copy(update={"story_id": "x"}), None, None)
    with pytest.raises(StorageError):
        await storage.supersede_story("a", "old", feb, new.model_copy(update={"story_id": "y"}), None, None)


async def test_graph_entities_relations_mentions_and_traversal(storage: Storage) -> None:
    source, chunks, vectors = _source("a", "s1", ["valve V-101 feeds pump P-3"])
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    v = Entity(corpus_id="a", entity_id="e1", name="V-101", key="v-101", type="valve")
    p = Entity(corpus_id="a", entity_id="e2", name="P-3", key="p-3", type="pump")
    t = Entity(corpus_id="a", entity_id="e3", name="Tank T-9", key="tank t-9", type="tank")
    for e in (v, p, t):
        await storage.put_entity("a", e)
    await storage.put_relations("a", [Relation(corpus_id="a", from_id="e1", to_id="e2", type="CONNECTS_TO", evidence_kind="chunk", evidence_id="s1-0"), Relation(corpus_id="a", from_id="e2", to_id="e3", type="FEEDS", evidence_kind="chunk", evidence_id="s1-0")])
    await storage.put_mentions("a", ["e1", "e2"], "chunk", "s1-0")
    assert await storage.find_entity("a", "p-3", "pump") == p and await storage.find_entity("a", "p-3", "valve") is None
    assert (await storage.get_entity("a", "p-3")) == p and (await storage.get_entity("a", "P-3")) == p
    assert [e.entity_id for e in await storage.entities_by_id("a", ["e3", "ghost", "e1"])] == ["e3", "e1"]
    assert [e.entity_id for e in await storage.entities_named_in("a", "is V-101 open?")] == ["e1"]
    assert [e.entity_id for e in await storage.similar_entities("a", "tank T9", limit=5)][:1] == ["e3"]
    hop = await storage.neighbors("a", ["e1"], relation_types=None)
    assert [(r.from_id, r.to_id) for r in hop] == [("e1", "e2")]
    assert [(r.type) for r in await storage.neighbors("a", ["e2"], relation_types=["FEEDS"])] == ["FEEDS"]
    assert await storage.mentions("a", ["e1", "e3"], kinds=("chunk", "story"), filters=None, window=Window()) == [("chunk", "s1-0")]
    await storage.add_alias("a", "e2", "Pump three")
    assert (await storage.get_entity("a", "pump three")) is not None
    deleted = await storage.delete_source("a", "s1")
    assert deleted.mentions == 2 and deleted.relations == 2
    assert await storage.neighbors("a", ["e1"], relation_types=None) == []


async def test_passages_hydrate_in_order_and_reembed_helpers(storage: Storage) -> None:
    source, chunks, vectors = _source("a", "s1", ["one", "two"])
    await storage.put_source("a", source, chunks, vectors, "old-model")
    await storage.put_story("a", _story("a", "st", "three"), _vec("three"), "old-model")
    passages = await storage.get_passages("a", [("story", "st"), ("chunk", "s1-1"), ("chunk", "ghost")])
    assert [p.story_id if isinstance(p, Story) else p.chunk_id for p in passages] == ["st", "s1-1"]
    stale = await storage.passages_embedded_with("a", not_model="new-model", limit=10)
    assert len(stale) == 3
    await storage.set_vectors("a", [("chunk", "s1-0"), ("chunk", "s1-1"), ("story", "st")], [_vec("x")] * 3, "new-model")
    assert await storage.passages_embedded_with("a", not_model="new-model", limit=10) == []


async def test_delete_corpus_is_total_and_counted(storage: Storage) -> None:
    source, chunks, vectors = _source("a", "s1", ["one"])
    await storage.put_source("a", source, chunks, vectors, "m")
    await storage.put_story("a", _story("a", "st", "two"), None, None)
    await storage.put_entity("a", Entity(corpus_id="a", entity_id="e1", name="x", key="x", type="t"))
    await storage.put_mentions("a", ["e1"], "story", "st")
    other, other_chunks, other_vectors = _source("b", "s2", ["keep"])
    await storage.put_source("b", other, other_chunks, other_vectors, "m")
    residue = await storage.count_corpus("a")
    assert (residue.sources, residue.chunks, residue.stories, residue.entities, residue.mentions) == (1, 1, 1, 1, 1) and not residue.clean
    deleted = await storage.delete_corpus("a")
    assert (deleted.sources, deleted.chunks, deleted.stories, deleted.entities, deleted.mentions) == (1, 1, 1, 1, 1)
    assert (await storage.count_corpus("a")).clean and not (await storage.count_corpus("b")).clean


async def test_aliases_match_by_normalized_name(storage: Storage) -> None:
    await storage.put_entity("a", Entity(corpus_id="a", entity_id="e1", name="P-3", key="p-3", type="pump"))
    await storage.add_alias("a", "e1", "\t Pump \n  Three ")
    assert (await storage.get_entity("a", "pump three")) is not None
    assert [e.entity_id for e in await storage.entities_named_in("a", "where is PUMP three now")] == ["e1"]


async def test_metadata_and_locators_round_trip_exactly(storage: Storage) -> None:
    metadata: dict[str, object] = {"zeta": 1, "a": {"refund": "r", "no": "n", "billing": "b"}, "big": 1e16, "tiny": 1.5e-7}
    source, chunks, vectors = _source("a", "s1", ["one"], metadata=metadata)
    chunks = [c.model_copy(update={"locator": metadata}) for c in chunks]
    await storage.put_source("a", source, chunks, vectors, "m")
    await storage.put_story("a", _story("a", "st", "two", metadata=metadata), None, None)
    got = await storage.get_source("a", "s1")
    listed, _ = await storage.list_sources("a", filters={"metadata.zeta": 1}, limit=10, cursor=None)
    stories, _ = await storage.list_stories("a", subject=None, window=Window(), filters={"metadata.a.no": "n"}, limit=10, cursor=None)
    passages = await storage.get_passages("a", [("chunk", "s1-0"), ("story", "st")])
    assert got is not None and len(listed) == 1 and len(stories) == 1 and len(passages) == 2
    stored = [got.metadata, listed[0].metadata, stories[0].metadata, (await storage.source_chunks("a", "s1"))[0].locator,
              (await storage.get_stories("a", ["st"]))[0].metadata]
    stored += [p.locator if isinstance(p, Chunk) else p.metadata for p in passages]
    for value in stored:
        assert value == metadata
        assert list(value) == list(metadata) and list(value["a"]) == ["refund", "no", "billing"]
        assert type(value["big"]) is float


async def test_ties_order_by_code_point_whatever_the_collation(storage: Storage) -> None:
    tied = ["ab", "B", "a-b", "_z", "a"]
    for tie in tied:
        source, chunks, vectors = _source("a", tie, [f"text {tie}"])
        await storage.put_source("a", source, chunks, vectors, "m")
        await storage.put_story("a", _story("a", tie, f"story {tie}", occurred_at=NOW), None, None)
        await storage.put_entity("a", Entity(corpus_id="a", entity_id=tie, name="valve", key="valve", type=f"type {tie}"))
    await storage.put_mentions("a", ["ab"], "story", "ab")
    for tie in tied:
        await storage.put_mentions("a", ["ab"], "story", tie)
    sources, _ = await storage.list_sources("a", filters=None, limit=10, cursor=None)
    assert [s.source_id for s in sources] == sorted(tied, reverse=True)
    stories, _ = await storage.list_stories("a", subject=None, window=Window(), filters=None, limit=10, cursor=None)
    assert [s.story_id for s in stories] == sorted(tied, reverse=True)
    assert [e.entity_id for e in await storage.entities_named_in("a", "the valve")] == sorted(tied)
    assert [e.entity_id for e in await storage.similar_entities("a", "valve", limit=10)] == sorted(tied)
    refs = await storage.mentions("a", ["ab"], kinds=("story",), filters=None, window=Window())
    assert refs == [("story", tie) for tie in sorted(tied)]


async def test_text_with_a_nul_character_is_refused_on_every_storage(storage: Storage) -> None:
    memory = Memory(storage)
    with pytest.raises(InputError):
        await memory.ingest("a", name="manual", text="pump\x00seal")
    with pytest.raises(InputError):
        await memory.ingest("a", name="manual", text="pump seal", metadata={"line": "A\x00"})
    with pytest.raises(InputError):
        await memory.remember("a", "the seal\x00 was replaced")
    with pytest.raises(InputError):
        await memory.remember("a", "the seal was replaced", subject="P\x003")
    with pytest.raises(InputError):
        await memory.search("a", "pump\x00seal")
    with pytest.raises(InputError):
        await memory.get_source("a\x00", "s1")
    assert (await storage.count_corpus("a")).clean
