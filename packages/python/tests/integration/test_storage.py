import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest
from unit.fakes import HashEmbedding

from rfnry_memory.errors import InputError, NotFoundError, StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    Chunk,
    Entity,
    EntityIn,
    Hits,
    Kind,
    Leg,
    Relation,
    RelationIn,
    Source,
    Story,
    Window,
)
from rfnry_memory.storage.storage import GraphWrite, Storage, text_hash
from rfnry_memory.tools import MemoryTools

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
    legs = await storage.search_text("a", "pump seal", kinds=("chunk", "story"), top_k=10, filters=None, window=Window(), subject=None)
    assert [(k, i) for k, i, _ in legs][:2] == [("chunk", "s1-0"), ("story", "st1")] or [(k, i) for k, i, _ in legs][:2] == [("story", "st1"), ("chunk", "s1-0")]
    assert all(score > 0 for _, _, score in legs) and ("chunk", "s1-1") not in [(k, i) for k, i, _ in legs]
    only_chunks = await storage.search_text("a", "pump seal", kinds=("chunk",), top_k=10, filters=None, window=Window(), subject=None)
    assert [i for _, i, _ in only_chunks] == ["s1-0"]
    faceted = await storage.search_text("a", "pump seal", kinds=("chunk", "story"), top_k=10, filters={"metadata.line": "B"}, window=Window(), subject=None)
    assert [i for _, i, _ in faceted] == ["st1"]
    assert await storage.search_text("b", "pump seal", kinds=("chunk", "story"), top_k=10, filters=None, window=Window(), subject=None) == []


async def test_vector_search_orders_by_cosine(storage: Storage) -> None:
    source, chunks, vectors = _source("a", "s1", ["alpha", "beta", "gamma"])
    await storage.put_source("a", source, chunks, vectors, "hash-8")
    legs = await storage.search_vector("a", _vec("beta"), kinds=("chunk",), top_k=2, filters=None, window=Window(), subject=None)
    assert legs[0][1] == "s1-1" and legs[0][2] == pytest.approx(1.0) and len(legs) == 2


async def test_story_windows_apply_inside_search(storage: Storage) -> None:
    jan, feb, mar = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3))
    await storage.put_story("a", _story("a", "old", "go-live is february twenty", subject="apollo", occurred_at=jan, valid_to=feb), _vec("go-live"), "hash-8")
    await storage.put_story("a", _story("a", "new", "go-live is march three", subject="apollo", occurred_at=feb), _vec("go-live"), "hash-8")
    assert set(ids(await storage.search_text("a", "go-live", kinds=("story",), top_k=10, filters=None, window=Window(), subject=None))) == {"old", "new"}
    assert ids(await storage.search_text("a", "go-live", kinds=("story",), top_k=10, filters=None, window=Window(as_of=jan + timedelta(days=10)), subject=None)) == ["old"]
    assert ids(await storage.search_text("a", "go-live", kinds=("story",), top_k=10, filters=None, window=Window(as_of=mar), subject=None)) == ["new"]
    assert ids(await storage.search_vector("a", _vec("go-live"), kinds=("story",), top_k=10, filters=None, window=Window(since=feb), subject=None)) == ["new"]
    assert ids(await storage.search_vector("a", _vec("go-live"), kinds=("story",), top_k=10, filters=None, window=Window(until=jan), subject=None)) == ["old"]
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
    v = Entity(corpus_id="a", entity_id="e1", name="V-101", key="v-101", type="valve")
    p = Entity(corpus_id="a", entity_id="e2", name="P-3", key="p-3", type="pump")
    t = Entity(corpus_id="a", entity_id="e3", name="Tank T-9", key="tank t-9", type="tank")
    relations = [Relation(corpus_id="a", from_id="e1", to_id="e2", type="CONNECTS_TO", evidence_kind="chunk", evidence_id="s1-0"),
                 Relation(corpus_id="a", from_id="e2", to_id="e3", type="FEEDS", evidence_kind="chunk", evidence_id="s1-0")]
    graph = GraphWrite(entities=[v, p, t], relations=relations, mentions=[("e1", "chunk", "s1-0"), ("e2", "chunk", "s1-0")])
    await storage.put_source("a", source, chunks, vectors, "hash-8", graph=graph)
    assert await storage.find_entity("a", "p-3", "pump") == p and await storage.find_entity("a", "p-3", "valve") is None
    assert (await storage.entities_named("a", "p-3")) == [p] and (await storage.entities_named("a", "P-3")) == [p]
    assert [e.entity_id for e in await storage.entities_by_id("a", ["e3", "ghost", "e1"])] == ["e3", "e1"]
    assert [e.entity_id for e in await storage.entities_named_in("a", "is V-101 open?")] == ["e1"]
    assert [e.entity_id for e in await storage.similar_entities("a", "tank T9", limit=5)][:1] == ["e3"]
    hop = await storage.neighbors("a", ["e1"], relation_types=None)
    assert [(r.from_id, r.to_id) for r in hop] == [("e1", "e2")]
    assert [(r.type) for r in await storage.neighbors("a", ["e2"], relation_types=["FEEDS"])] == ["FEEDS"]
    assert await storage.mentions("a", ["e1", "e3"], kinds=("chunk", "story"), filters=None, window=Window(), subject=None) == [("chunk", "s1-0")]
    await storage.put_story("a", _story("a", "st", "P-3 is also called pump three"), None, None, graph=GraphWrite(aliases=[("e2", "Pump three")]))
    assert (await storage.entities_named("a", "pump three")) == [p.model_copy(update={"aliases": ("Pump three",)})]
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
    await storage.put_story("a", _story("a", "st", "two"), None, None,
                            graph=GraphWrite(entities=[Entity(corpus_id="a", entity_id="e1", name="x", key="x", type="t")], mentions=[("e1", "story", "st")]))
    other, other_chunks, other_vectors = _source("b", "s2", ["keep"])
    await storage.put_source("b", other, other_chunks, other_vectors, "m")
    residue = await storage.count_corpus("a")
    assert (residue.sources, residue.chunks, residue.stories, residue.entities, residue.mentions) == (1, 1, 1, 1, 1) and not residue.clean
    deleted = await storage.delete_corpus("a")
    assert (deleted.sources, deleted.chunks, deleted.stories, deleted.entities, deleted.mentions) == (1, 1, 1, 1, 1)
    assert (await storage.count_corpus("a")).clean and not (await storage.count_corpus("b")).clean


async def test_aliases_match_by_normalized_name(storage: Storage) -> None:
    entity = Entity(corpus_id="a", entity_id="e1", name="P-3", key="p-3", type="pump", aliases=("\t Pump \n  Three ",))
    await storage.put_story("a", _story("a", "st", "P-3"), None, None, graph=GraphWrite(entities=[entity]))
    assert (await storage.entities_named("a", "pump three")) != []
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
        entity = Entity(corpus_id="a", entity_id=tie, name="valve", key="valve", type=f"type {tie}")
        await storage.put_story("a", _story("a", tie, f"story {tie}", occurred_at=NOW), None, None,
                                graph=GraphWrite(entities=[entity], mentions=[("ab", "story", tie)]))
    sources, _ = await storage.list_sources("a", filters=None, limit=10, cursor=None)
    assert [s.source_id for s in sources] == sorted(tied, reverse=True)
    stories, _ = await storage.list_stories("a", subject=None, window=Window(), filters=None, limit=10, cursor=None)
    assert [s.story_id for s in stories] == sorted(tied, reverse=True)
    assert [e.entity_id for e in await storage.entities_named_in("a", "the valve")] == sorted(tied)
    assert [e.entity_id for e in await storage.similar_entities("a", "valve", limit=10)] == sorted(tied)
    refs = await storage.mentions("a", ["ab"], kinds=("story",), filters=None, window=Window(), subject=None)
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


def _hit_ids(hits: Hits) -> set[str]:
    return {h.id for h in hits.hits}


async def test_story_reads_answer_with_what_is_true_now_unless_history_is_asked(storage: Storage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    apollo = [EntityIn(name="Apollo", type="project")]
    jan, feb = datetime(2026, 1, 10, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    closed = (await memory.remember("a", "The go-live is February 20.", subject="project:apollo", occurred_at=jan, metadata={"team": "ops"}, entities=apollo)).story
    current = (await memory.supersede("a", closed.story_id, "The go-live moved to March 3.", effective_at=feb)).story
    later = (await memory.remember("a", "The go-live rehearsal is planned.", subject="project:apollo", occurred_at=datetime.now(UTC) + timedelta(days=30), entities=apollo)).story
    lead = (await memory.remember("a", "Ana leads Apollo.", subject="user:ana", entities=apollo)).story
    everything = {closed.story_id, current.story_id, later.story_id}

    assert _hit_ids(await memory.recall("a", "go-live")) == {current.story_id}
    assert _hit_ids(await memory.recall("a", "go-live", history=True)) == everything
    assert _hit_ids(await memory.recall("a", "go-live", as_of=jan + timedelta(days=10))) == {closed.story_id}
    with pytest.raises(InputError):
        await memory.recall("a", "go-live", history=True, as_of=jan)
    assert _hit_ids(await memory.recall("a", "go-live", since=feb)) == {current.story_id}
    assert _hit_ids(await memory.recall("a", "go-live", since=feb, history=True)) == {current.story_id, later.story_id}
    assert _hit_ids(await memory.recall("a", "go-live", until=feb, history=True)) == {closed.story_id, current.story_id}
    assert _hit_ids(await memory.recall("a", "go-live", subject="project:apollo")) == {current.story_id}
    assert _hit_ids(await memory.recall("a", "go-live", filters={"metadata.team": "ops"}, history=True)) == {closed.story_id, current.story_id}
    assert _hit_ids(await memory.recall("a", "go-live", filters={"metadata.team": "ops"})) == {current.story_id}

    assert _hit_ids(await memory.search("a", "go-live", kinds=("story",))) == {current.story_id}
    assert _hit_ids(await memory.search("a", "go-live", kinds=("story",), history=True)) == everything

    listed, _ = await memory.list_stories("a", subject="project:apollo")
    assert [s.story_id for s in listed] == [current.story_id]
    listed, _ = await memory.list_stories("a", subject="project:apollo", history=True)
    assert [s.story_id for s in listed] == [later.story_id, current.story_id, closed.story_id]
    listed, _ = await memory.list_stories("a", as_of=jan + timedelta(days=10))
    assert [s.story_id for s in listed] == [closed.story_id]
    with pytest.raises(InputError):
        await memory.list_stories("a", history=True, as_of=jan)

    assert {h.id for h in (await memory.neighbors("a", "Apollo")).passages} == {lead.story_id}
    assert {h.id for h in (await memory.neighbors("a", "Apollo", history=True)).passages} == {closed.story_id, later.story_id, lead.story_id}


async def test_a_superseding_story_is_written_to_the_graph_like_a_remembered_one(storage: Storage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    feb = datetime(2026, 2, 1, tzinfo=UTC)
    closed = (await memory.remember("a", "Ana owns the pump.", occurred_at=datetime(2026, 1, 10, tzinfo=UTC),
                                    entities=[EntityIn(name="Ana", type="person"), EntityIn(name="P-3", type="pump")],
                                    relations=[RelationIn(from_name="Ana", to_name="P-3", type="OWNS")])).story
    current = (await memory.supersede("a", closed.story_id, "Bo owns the pump.", effective_at=feb,
                                      entities=[EntityIn(name="Bo", type="person"), EntityIn(name="P-3", type="pump")],
                                      relations=[RelationIn(from_name="Bo", to_name="P-3", type="OWNS")])).story

    hood = await memory.neighbors("a", "P-3")
    assert {e.name for e in hood.entities} == {"P-3", "Ana", "Bo"}
    assert {h.id for h in hood.passages} == {current.story_id}
    assert {h.id for h in (await memory.neighbors("a", "P-3", history=True)).passages} == {closed.story_id, current.story_id}
    assert {h.id for h in (await memory.neighbors("a", "Bo")).passages} == {current.story_id}
    assert _hit_ids(await memory.recall("a", "Bo", pillars={"graph"})) == {current.story_id}
    with pytest.raises(InputError):
        await memory.supersede("a", current.story_id, "Cy owns the pump.", relations=[RelationIn(from_name="Cy", to_name="P-3", type="OWNS")])
    assert [s.story_id for s in (await memory.list_stories("a"))[0]] == [current.story_id]


def _bad_graph(corpus: str, kind: Kind, passage_id: str) -> GraphWrite:
    return GraphWrite(
        entities=[Entity(corpus_id=corpus, entity_id="e1", name="P-3", key="p-3", type="pump")],
        relations=[Relation(corpus_id=corpus, from_id="e1", to_id="ghost", type="FEEDS", evidence_kind=kind, evidence_id=passage_id)],
        mentions=[("e1", kind, passage_id)],
    )


async def test_a_write_whose_graph_is_refused_leaves_nothing_behind(storage: Storage) -> None:
    source, chunks, vectors = _source("a", "s1", ["pump P-3 feeds tank T-9"])
    with pytest.raises(StorageError):
        await storage.put_source("a", source, chunks, vectors, "m", graph=_bad_graph("a", "chunk", "s1-0"))
    with pytest.raises(StorageError):
        await storage.put_story("a", _story("a", "st", "pump P-3 feeds tank T-9"), _vec("p-3"), "m", graph=_bad_graph("a", "story", "st"))
    elsewhere = GraphWrite(entities=[Entity(corpus_id="a", entity_id="e1", name="P-3", key="p-3", type="pump")], mentions=[("e1", "story", "other")])
    with pytest.raises(StorageError):
        await storage.put_story("a", _story("a", "st", "pump P-3"), None, None, graph=elsewhere)
    assert (await storage.count_corpus("a")).clean
    assert await storage.search_vector("a", _vec("p-3"), kinds=("chunk", "story"), top_k=10, filters=None, window=Window(), subject=None) == []

    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    old = _story("a", "old", "P-3 feeds tank T-8", occurred_at=jan)
    await storage.put_story("a", old, None, None)
    new = _story("a", "new", "P-3 feeds tank T-9", occurred_at=feb).model_copy(update={"supersedes": "old"})
    with pytest.raises(StorageError):
        await storage.supersede_story("a", "old", feb, new, _vec("t-9"), "m", graph=_bad_graph("a", "story", "new"))
    assert await storage.get_stories("a", ["old", "new"]) == [old]
    residue = await storage.count_corpus("a")
    assert (residue.stories, residue.entities, residue.relations, residue.mentions) == (1, 0, 0, 0)


async def test_a_write_carries_its_graph_in_the_same_step(storage: Storage) -> None:
    source, chunks, vectors = _source("a", "s1", ["pump P-3 feeds tank T-9", "tank T-9 holds water"])
    pump = Entity(corpus_id="a", entity_id="e1", name="P-3", key="p-3", type="pump")
    tank = Entity(corpus_id="a", entity_id="e2", name="T-9", key="t-9", type="tank")
    feeds = Relation(corpus_id="a", from_id="e1", to_id="e2", type="FEEDS", evidence_kind="chunk", evidence_id="s1-0")
    await storage.put_source("a", source, chunks, vectors, "m", graph=GraphWrite(
        entities=[pump, tank], relations=[feeds], mentions=[("e1", "chunk", "s1-0"), ("e2", "chunk", "s1-0"), ("e2", "chunk", "s1-1")]))
    residue = await storage.count_corpus("a")
    assert (residue.sources, residue.chunks, residue.entities, residue.relations, residue.mentions) == (1, 2, 2, 1, 3)
    assert ids(await storage.search_vector("a", _vec("tank T-9 holds water"), kinds=("chunk",), top_k=1, filters=None, window=Window(), subject=None)) == ["s1-1"]

    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    old = _story("a", "old", "Ana runs P-3", occurred_at=jan)
    await storage.put_story("a", old, _vec("ana"), "m", graph=GraphWrite(
        entities=[Entity(corpus_id="a", entity_id="e3", name="Ana", key="ana", type="person")],
        relations=[Relation(corpus_id="a", from_id="e3", to_id="e1", type="RUNS", evidence_kind="story", evidence_id="old")],
        mentions=[("e3", "story", "old"), ("e1", "story", "old")]))
    new = _story("a", "new", "Bo runs P-3", occurred_at=feb).model_copy(update={"supersedes": "old"})
    closed, created = await storage.supersede_story("a", "old", feb, new, _vec("bo"), "m", graph=GraphWrite(
        entities=[Entity(corpus_id="a", entity_id="e4", name="Bo", key="bo", type="person")],
        aliases=[("e1", "Pump three")],
        relations=[Relation(corpus_id="a", from_id="e4", to_id="e1", type="RUNS", evidence_kind="story", evidence_id="new")],
        mentions=[("e4", "story", "new"), ("e1", "story", "new")]))
    assert closed.valid_to == feb and created == new
    assert await storage.mentions("a", ["e1"], kinds=("chunk", "story"), filters=None, window=Window(as_of=feb), subject=None) == [("chunk", "s1-0"), ("story", "new")]
    assert [(r.from_id, r.type) for r in await storage.neighbors("a", ["e4"], relation_types=None)] == [("e4", "RUNS")]
    assert (await storage.entities_named("a", "pump three")) == [pump.model_copy(update={"aliases": ("Pump three",)})]
    residue = await storage.count_corpus("a")
    assert (residue.stories, residue.entities, residue.relations, residue.mentions) == (2, 4, 3, 7)


async def test_an_entity_created_by_a_concurrent_writer_is_joined_not_duplicated(storage: Storage) -> None:
    first = Entity(corpus_id="a", entity_id="first", name="P-3", key="p-3", type="pump")
    await storage.put_story("a", _story("a", "st1", "P-3 hums"), None, None, graph=GraphWrite(entities=[first], mentions=[("first", "story", "st1")]))
    late = Entity(corpus_id="a", entity_id="late", name="P-3", key="p-3", type="pump", aliases=("Pump three",))
    tank = Entity(corpus_id="a", entity_id="tank", name="T-9", key="t-9", type="tank")
    await storage.put_story("a", _story("a", "st2", "pump three feeds T-9"), None, None, graph=GraphWrite(
        entities=[late, tank],
        relations=[Relation(corpus_id="a", from_id="late", to_id="tank", type="FEEDS", evidence_kind="story", evidence_id="st2")],
        mentions=[("late", "story", "st2"), ("tank", "story", "st2")]))
    assert (await storage.count_corpus("a")).entities == 2
    assert await storage.entities_by_id("a", ["late"]) == []
    assert (await storage.entities_named("a", "pump three")) == [first.model_copy(update={"aliases": ("Pump three",)})]
    assert await storage.mentions("a", ["first"], kinds=("story",), filters=None, window=Window(), subject=None) == [("story", "st1"), ("story", "st2")]
    assert [(r.from_id, r.to_id) for r in await storage.neighbors("a", ["first"], relation_types=None)] == [("first", "tank")]

    racers = [Entity(corpus_id="a", entity_id=f"racer{i}", name="V-7", key="v-7", type="valve", aliases=(f"valve {i}",)) for i in range(4)]
    await asyncio.gather(*(
        storage.put_story("a", _story("a", f"race{i}", f"V-7 report {i}"), None, None,
                          graph=GraphWrite(entities=[racer, tank], mentions=[(racer.entity_id, "story", f"race{i}")]))
        for i, racer in enumerate(racers)))
    [joined] = await storage.entities_named("a", "v-7")
    assert sorted(joined.aliases) == [f"valve {i}" for i in range(4)]
    assert (await storage.count_corpus("a")).entities == 3
    assert len(await storage.mentions("a", [joined.entity_id], kinds=("story",), filters=None, window=Window(), subject=None)) == 4


async def test_memory_writes_graph_and_passage_as_one_step(storage: Storage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    facts = [EntityIn(name="Ana", type="person"), EntityIn(name="P-3", type="pump")]
    ingested = await memory.ingest("a", name="sheet", text="Ana runs P-3.", entities=facts,
                                   relations=[RelationIn(from_name="Ana", to_name="P-3", type="RUNS")])
    story = (await memory.remember("a", "P-3 was serviced.", entities=[EntityIn(name="p-3", type="pump")])).story
    current = (await memory.supersede("a", story.story_id, "P-3 was replaced.", entities=[EntityIn(name="P-3", type="pump")],
                                      relations=[RelationIn(from_name="Ana", to_name="P-3", type="RUNS")])).story
    residue = await storage.count_corpus("a")
    assert (residue.sources, residue.stories, residue.entities, residue.relations, residue.mentions) == (1, 2, 2, 2, 4)
    hood = await memory.neighbors("a", "P-3")
    assert {h.id for h in hood.passages} == {f"{ingested.source.source_id}-0", current.story_id}


async def test_subject_narrows_inside_every_leg_not_after_the_fetch(storage: Storage) -> None:
    go_live = Entity(corpus_id="a", entity_id="go", name="go-live", key="go-live", type="event")
    for i in range(60):
        story = _story("a", f"other-{i:02d}", f"go-live go-live go-live {i}", subject=f"project:{i}")
        await storage.put_story("a", story, _vec("go-live"), "hash-8",
                                graph=GraphWrite(entities=[go_live] if i == 0 else [], mentions=[("go", "story", story.story_id)]))
    await storage.put_story("a", _story("a", "zz-apollo", "the apollo go-live moved after a long review of the plan", subject="project:apollo"),
                            _vec("unrelated"), "hash-8", graph=GraphWrite(mentions=[("go", "story", "zz-apollo")]))
    source, chunks, vectors = _source("a", "s1", ["go-live"])
    await storage.put_source("a", source, chunks, vectors, "hash-8", graph=GraphWrite(mentions=[("go", "chunk", "s1-0")]))
    both: tuple[Kind, ...] = ("chunk", "story")
    assert ids(await storage.search_text("a", "go-live", kinds=both, top_k=5, filters=None, window=Window(), subject="project:apollo")) == ["zz-apollo"]
    assert ids(await storage.search_vector("a", _vec("go-live"), kinds=both, top_k=5, filters=None, window=Window(), subject="project:apollo")) == ["zz-apollo"]
    assert await storage.mentions("a", ["go"], kinds=both, filters=None, window=Window(), subject="project:apollo") == [("story", "zz-apollo")]
    assert "zz-apollo" not in ids(await storage.search_text("a", "go-live", kinds=both, top_k=5, filters=None, window=Window(), subject=None))
    assert len(await storage.mentions("a", ["go"], kinds=both, filters=None, window=Window(), subject=None)) == 62


async def test_recall_by_subject_finds_the_subject_story_however_many_others_outrank_it(storage: Storage) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    go_live = [EntityIn(name="go-live", type="event")]
    for i in range(60):
        await memory.remember("a", f"go-live go-live go-live {i}", subject=f"project:{i}", entities=go_live)
    apollo = (await memory.remember("a", "the apollo go-live moved after a long review of the plan", subject="project:apollo", entities=go_live)).story
    for pillars in ({"text"}, {"vector"}, {"graph"}, {"text", "vector", "graph"}):
        hits = await memory.recall("a", "go-live", top_k=5, subject="project:apollo", pillars=pillars)
        assert [h.id for h in hits.hits] == [apollo.story_id], pillars


async def test_bound_tools_recall_what_they_remember(storage: Storage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.remember("a", "Line B owns the wiring review.", metadata={"line": "B"})
    tools = MemoryTools(memory, corpus_id="a", filters={"metadata.line": "A", "metadata.site.zone": 3})
    remembered = await tools.dispatch("remember", {"text": "Ana owns the wiring review."})
    assert remembered["story"]["metadata"] == {"line": "A", "site": {"zone": 3}}
    recalled = await tools.dispatch("recall", {"query": "wiring review"})
    assert [h["id"] for h in recalled["hits"]] == [remembered["story"]["story_id"]]
    hidden = {"error": {"type": "InputError", "message": "tools: this text already exists outside your view"}}
    assert await tools.dispatch("remember", {"text": "line b owns the  wiring review."}) == hidden
    assert await tools.dispatch("supersede", {"story_id": remembered["story"]["story_id"], "text": "Line B owns the wiring review."}) == hidden


async def test_neighbors_asks_for_a_type_when_a_name_names_several(storage: Storage) -> None:
    memory = Memory(storage, pillars={"text", "graph"})
    await memory.ingest("a", name="bom", text="Part P-3 is a gasket.", entities=[EntityIn(name="P-3", type="part")])
    facts = [EntityIn(name="V-1", type="valve"), EntityIn(name="P-3", type="pump")]
    feeds = RelationIn(from_name="V-1", to_name="P-3", type="FEEDS")
    with pytest.raises(InputError, match=r"^memory: relation 'V-1' -> 'P-3' names 'P-3', which has several types: \['part', 'pump'\]"):
        await memory.ingest("a", name="sheet", text="Valve V-1 feeds pump P-3.", entities=facts, relations=[feeds])
    assert (await storage.count_corpus("a")).sources == 1 and [e.type for e in await storage.entities_named("a", "p-3")] == ["part"]
    await memory.ingest("a", name="sheet", text="Valve V-1 feeds pump P-3.", entities=facts,
                        relations=[feeds.model_copy(update={"to_type": "pump"})])
    assert sorted(e.type for e in await storage.entities_named("a", "p-3")) == ["part", "pump"]
    assert await storage.entities_named("b", "p-3") == []
    with pytest.raises(InputError, match=r"^memory: entity 'P-3' names entities of several types: \['part', 'pump'\]"):
        await memory.neighbors("a", "P-3")
    assert {(e.name, e.type) for e in (await memory.neighbors("a", "P-3", entity_type="Pump")).entities} == {("P-3", "pump"), ("V-1", "valve")}
    assert [(e.name, e.type) for e in (await memory.neighbors("a", "P-3", entity_type="part")).entities] == [("P-3", "part")]
    with pytest.raises(NotFoundError, match=r"^memory: entity 'P-3' of type 'valve' was not found"):
        await memory.neighbors("a", "P-3", entity_type="valve")
    assert {e.name for e in (await memory.neighbors("a", "V-1")).entities} == {"V-1", "P-3"}
