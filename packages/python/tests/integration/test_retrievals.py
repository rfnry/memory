from datetime import UTC, datetime, timedelta

import pytest
from unit.fakes import HashEmbedding

from rfnry_memory.errors import StorageError
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery import MemoryRefinery, RetrievalReview
from rfnry_memory.rows import RetrievalLabelRow, RetrievalRow

T0 = datetime(2026, 3, 1, tzinfo=UTC)


def _row(
    retrieval_id: str, corpus_ids: list[str], minute: int, verb: str = "search"
) -> RetrievalRow:
    moment = T0 + timedelta(minutes=minute)
    return RetrievalRow(
        retrieval_id=retrieval_id,
        corpus_ids=corpus_ids,
        verb=verb,
        at=moment,
        recorded=moment,
        duration_ms=minute,
        body={"query": retrieval_id, "hits": []},
    )


def _label(retrieval_id: str, label_id: str, source: str, minute: int) -> RetrievalLabelRow:
    return RetrievalLabelRow(
        retrieval_id=retrieval_id,
        label_id=label_id,
        source=source,
        at=T0 + timedelta(minutes=minute),
        body={"review": {"relevance": [], "comment": label_id}},
    )


async def test_every_retrieval_storage_method(storage: PostgresStorage) -> None:
    for retrieval_id, minute, verb in (
        ("r1", 1, "search"),
        ("r2", 2, "resembling"),
        ("r3", 3, "recall"),
    ):
        await storage.put_retrieval(["acme"], _row(retrieval_id, ["acme"], minute, verb))
    with pytest.raises(
        StorageError, match=r"^postgres: row corpus does not match the call corpus$"
    ):
        await storage.put_retrieval(["ops"], _row("r4", ["acme"], 4))
    got = await storage.get_retrieval(["acme"], "r1")
    assert got is not None and (got.verb, got.body["query"], got.recorded) == (
        "search",
        "r1",
        T0 + timedelta(minutes=1),
    )
    page, cursor = await storage.list_retrievals(["acme"], since=None, limit=2, cursor=None)
    assert [r.retrieval_id for r in page] == ["r3", "r2"] and cursor == "2"
    rest, last = await storage.list_retrievals(["acme"], since=None, limit=2, cursor=cursor)
    assert [r.retrieval_id for r in rest] == ["r1"] and last is None
    recent, _ = await storage.list_retrievals(
        ["acme"], since=T0 + timedelta(minutes=1), limit=None, cursor=None
    )
    assert [r.retrieval_id for r in recent] == ["r3", "r2"]
    first = await storage.scan_retrievals(
        ["acme"], since=None, relabeled=False, after=None, limit=2
    )
    assert [r.retrieval_id for r in first] == ["r1", "r2"]
    after = (first[-1].recorded, first[-1].retrieval_id)
    following = await storage.scan_retrievals(
        ["acme"], since=None, relabeled=False, after=after, limit=2
    )
    assert [r.retrieval_id for r in following] == ["r3"]
    stored = await storage.put_retrieval_label(["acme"], _label("r1", "m1", "human", 10))
    assert stored is not None and stored.body["review"]["comment"] == "m1"
    again = await storage.put_retrieval_label(["acme"], _label("r1", "m1", "world", 11))
    assert again is not None and (again.source, again.at) == ("human", T0 + timedelta(minutes=10))
    assert await storage.put_retrieval_label(["acme"], _label("missing", "m2", "human", 10)) is None
    later = T0 + timedelta(minutes=5)
    quiet = await storage.scan_retrievals(
        ["acme"], since=later, relabeled=False, after=None, limit=10
    )
    relabeled = await storage.scan_retrievals(
        ["acme"], since=later, relabeled=True, after=None, limit=10
    )
    assert quiet == [] and [r.retrieval_id for r in relabeled] == ["r1"]
    await storage.put_retrieval_label(["acme"], _label("r3", "j1", "judge", 12))
    unjudged = await storage.unjudged_retrievals(["acme"], limit=10)
    assert [r.retrieval_id for r in unjudged] == ["r1"]
    labels = await storage.retrieval_labels(["acme"], ["r1", "r3"])
    assert [(label.retrieval_id, label.label_id) for label in labels] == [
        ("r1", "m1"),
        ("r3", "j1"),
    ]
    since = await storage.retrieval_labels(["acme"], ["r1", "r3"], since=T0 + timedelta(minutes=11))
    assert [label.label_id for label in since] == ["j1"]
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    source = (await memory.ingest("acme", name="manual", text="Replace the seal.")).source
    assert await storage.source_names("acme", [source.source_id, "gone"]) == {
        source.source_id: "manual"
    }
    assert await storage.source_names("ops", [source.source_id]) == {}
    assert await storage.forget_retrievals(["acme"], before=T0 + timedelta(minutes=2)) == 1
    assert await storage.get_retrieval(["acme"], "r1") is None
    assert await storage.retrieval_labels(["acme"], ["r1"]) == []


async def test_no_recording_of_a_is_reachable_from_b(storage: PostgresStorage) -> None:
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery)
    await memory.check()
    for corpus in ("tenant-a", "tenant-b"):
        await memory.ingest(corpus, name="manual", text="Replace the pump seal yearly.")
    alone = (await memory.search("tenant-a", "pump seal")).retrieval_id
    both = (await memory.search(["tenant-a", "tenant-b"], "pump seal")).retrieval_id
    own = (await memory.search("tenant-b", "pump seal")).retrieval_id
    assert alone is not None and both is not None and own is not None
    await refinery.review(["tenant-a", "tenant-b"], both, RetrievalReview(comment="seen"))
    await refinery.review("tenant-a", alone, RetrievalReview(comment="seen"))
    b = ["tenant-b"]
    for retrieval_id in (alone, both):
        assert await storage.get_retrieval(b, retrieval_id) is None
        assert await storage.put_retrieval_label(b, _label(retrieval_id, "x", "human", 1)) is None
        assert await storage.retrieval_labels(b, [retrieval_id]) == []
    listed, _ = await storage.list_retrievals(b, since=None, limit=None, cursor=None)
    scanned = await storage.scan_retrievals(b, since=None, relabeled=True, after=None, limit=100)
    unjudged = await storage.unjudged_retrievals(b, limit=100)
    assert {r.retrieval_id for r in listed + scanned + unjudged} == {own}
    assert await storage.forget_retrievals(b, before=datetime.now(UTC) + timedelta(days=1)) == 1
    assert (await memory.verify_absent("tenant-a")).retrievals == 2
    erased = await memory.delete_corpus("tenant-a")
    assert erased.retrievals == 2 and (await memory.verify_absent("tenant-a")).clean
    assert await storage.get_retrieval(["tenant-a", "tenant-b"], both) is None
