import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError, NotFoundError
from rfnry_memory.events import Event, Recorded
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery import MemoryRefinery
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding

MANUAL = "Replace the pump seal yearly, after draining the line."


async def _memory(storage: PostgresStorage, **options: Any) -> tuple[Memory, MemoryRefinery]:
    refinery = MemoryRefinery(storage, **options)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery)
    await memory.check()
    return memory, refinery


async def test_a_search_records_its_candidates_and_its_hits_by_id_never_by_text(
    storage: PostgresStorage,
) -> None:
    memory, refinery = await _memory(storage)
    filed = (await memory.ingest("acme", name="manual", text=MANUAL)).source
    await memory.remember("acme", "Ana replaced the pump seal.")
    heard: list[Event] = []

    async def hear(event: Event) -> None:
        heard.append(event)

    began = datetime.now(UTC)
    hits = await memory.search("acme", "pump seal", top_k=5, on_event=hear)
    assert hits.retrieval_id is not None
    [recorded] = [e for e in heard if isinstance(e, Recorded)]
    assert (recorded.corpus_ids, recorded.retrieval_id, recorded.error) == (
        ["acme"],
        hits.retrieval_id,
        None,
    )
    got = await refinery.retrieval("acme", hits.retrieval_id)
    assert (got.verb, got.query, got.corpus_ids) == ("search", "pump seal", ["acme"])
    assert got.arguments["top_k"] == 5 and got.arguments["kinds"] == ["chunk", "story"]
    assert set(got.candidates) == {"text", "vector"}
    assert [(h.kind, h.id) for h in got.hits] == [(h.kind, h.id) for h in hits.hits]
    assert {h.source_id for h in got.hits if h.kind == "chunk"} == {filed.source_id}
    assert got.fingerprint["embedding"] == HashEmbedding().model
    assert got.fingerprint["chunk_chars"] == memory.chunk_chars
    assert got.at >= began and got.recorded >= got.at and got.duration_ms >= 0
    assert got.pillars["text"].returned == len(got.candidates["text"])
    assert MANUAL not in json.dumps(got.model_dump(mode="json"))


async def test_recall_expand_and_resembling_are_recorded_and_the_tools_carry_the_id(
    storage: PostgresStorage,
) -> None:
    memory, refinery = await _memory(storage)
    filed = (await memory.ingest("acme", name="manual", text=MANUAL)).source
    await memory.remember("acme", "Ana replaced the pump seal.")
    recalled = await memory.recall("acme", "pump seal", known_at=datetime.now(UTC))
    assert recalled.retrieval_id is not None
    got = await refinery.retrieval("acme", recalled.retrieval_id)
    assert got.verb == "recall" and got.window["known_at"] is not None
    around = await memory.expand("acme", ("chunk", f"{filed.source_id}-0"), by="vector")
    assert around.retrieval_id is not None
    expanded = await refinery.retrieval("acme", around.retrieval_id)
    assert (expanded.verb, expanded.query, list(expanded.candidates)) == (
        "expand",
        None,
        ["vector"],
    )
    assert expanded.arguments["passage"] == ["chunk", f"{filed.source_id}-0"]
    resembled = await memory.resembling("acme", text=MANUAL)
    assert resembled.matches[0].source.source_id == filed.source_id
    assert resembled.retrieval_id is not None
    alike = await refinery.retrieval("acme", resembled.retrieval_id)
    assert [(h.kind, h.id, h.score) for h in alike.hits] == [("source", filed.source_id, 1.0)]
    tools = MemoryTools(memory, corpus_id="acme")
    for name, args in (
        ("search", {"query": "pump seal"}),
        ("recall", {"query": "pump seal"}),
        ("expand", {"kind": "chunk", "id": f"{filed.source_id}-0", "by": "document"}),
        ("resembling", {"text": MANUAL}),
    ):
        result = await tools.dispatch(name, args)
        assert isinstance(result, dict) and isinstance(result["retrieval_id"], str), name
    listed, _ = await refinery.retrievals("acme")
    assert len(listed) == 7 and listed[0].verb == "resembling"


async def test_nothing_is_recorded_without_a_refinery_or_outside_the_sample(
    storage: PostgresStorage,
) -> None:
    plain = Memory(storage, embedding=HashEmbedding())
    await plain.check()
    await plain.ingest("acme", name="manual", text=MANUAL)
    assert (await plain.search("acme", "pump seal")).retrieval_id is None
    memory, refinery = await _memory(storage, sample=0.0)
    assert (await memory.search("acme", "pump seal")).retrieval_id is None
    assert await refinery.retrievals("acme") == ([], None)
    for rate in (-0.1, 1.5, float("nan")):
        with pytest.raises(ConfigurationError, match=r"^refinery: sample is a rate from 0 to 1$"):
            MemoryRefinery(storage, sample=rate)


async def test_a_recording_that_fails_never_fails_the_call(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, _ = await _memory(storage)
    await memory.ingest("acme", name="manual", text=MANUAL)

    async def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(storage, "put_retrieval", broken)
    heard: list[Event] = []

    async def hear(event: Event) -> None:
        heard.append(event)

    hits = await memory.search("acme", "pump seal", on_event=hear)
    assert hits.hits and hits.retrieval_id is None
    [recorded] = [e for e in heard if isinstance(e, Recorded)]
    assert (recorded.retrieval_id, recorded.error) == (None, "RuntimeError: disk full")


async def test_a_recording_is_read_only_by_a_call_naming_every_corpus_it_read(
    storage: PostgresStorage,
) -> None:
    memory, refinery = await _memory(storage)
    for corpus in ("hr", "ops"):
        await memory.ingest(corpus, name="manual", text=MANUAL)
    both = (await memory.search(["hr", "ops"], "pump seal")).retrieval_id
    alone = (await memory.search("hr", "pump seal")).retrieval_id
    assert both is not None and alone is not None
    with pytest.raises(NotFoundError, match=rf"^refinery: retrieval '{both}' was not found$"):
        await refinery.retrieval("hr", both)
    assert (await refinery.retrieval(["ops", "hr"], both)).corpus_ids == ["hr", "ops"]
    assert [r.retrieval_id for r in (await refinery.retrievals("hr"))[0]] == [alone]
    assert len((await refinery.retrievals(["hr", "ops"]))[0]) == 2
    later = datetime.now(UTC) + timedelta(days=1)
    assert await refinery.forget_recorded("ops", before=later) == 0
    erased = await memory.delete_corpus("ops")
    assert erased.retrievals == 1 and (await memory.verify_absent("ops")).clean
    assert [r.retrieval_id for r in (await refinery.retrievals(["hr", "ops"]))[0]] == [alone]
    assert await refinery.forget_recorded("hr", before=later) == 1
    assert await refinery.retrievals("hr") == ([], None)


async def test_the_holdout_follows_engines_rule(storage: PostgresStorage) -> None:
    refinery = MemoryRefinery(storage)
    held = [refinery.is_holdout(f"r{n}") for n in range(1000)]
    assert 150 < sum(held) < 250
    assert refinery.is_holdout("r1") == refinery.is_holdout("r1")
    assert not any(refinery.is_holdout(f"r{n}", holdout_percent=0) for n in range(100))
    assert all(refinery.is_holdout(f"r{n}", holdout_percent=100) for n in range(100))
