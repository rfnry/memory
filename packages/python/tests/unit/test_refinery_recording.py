import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError, InputError, ModelError, NotFoundError
from rfnry_memory.events import Event, Recorded
from rfnry_memory.memory import Memory
from rfnry_memory.models import CheckAnswer
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery import MemoryRefinery
from rfnry_memory.tools import MemoryTools

from .fakes import Fixed, HashEmbedding

MANUAL = "Replace the pump seal yearly, after draining the line."


class QueryOutage(HashEmbedding):
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        if purpose == "query":
            raise ModelError("fake: embedding down")
        return await super().embed(texts, purpose=purpose)


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


async def test_a_failed_pillar_is_recorded_with_its_error(storage: PostgresStorage) -> None:
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=QueryOutage(), refinery=refinery)
    await memory.check()
    await memory.ingest("acme", name="manual", text=MANUAL)
    retrieval_id = (await memory.search("acme", "pump seal")).retrieval_id
    assert retrieval_id is not None
    recording = await refinery.retrieval("acme", retrieval_id)
    assert recording.pillars["vector"].error == "ModelError: fake: embedding down"
    assert recording.hits and recording.candidates.get("vector", []) == []


async def test_a_handler_that_raises_on_recorded_fails_the_call_and_keeps_the_row(
    storage: PostgresStorage,
) -> None:
    memory, refinery = await _memory(storage)
    await memory.ingest("acme", name="manual", text=MANUAL)

    async def hear(event: Event) -> None:
        if isinstance(event, Recorded):
            raise RuntimeError("handler broke")

    with pytest.raises(RuntimeError, match=r"^handler broke$"):
        await memory.search("acme", "pump seal", on_event=hear)
    [kept], _ = await refinery.retrievals("acme")
    assert kept.verb == "search"


async def test_kinds_and_dated_filters_survive_the_recording_and_the_replay(
    storage: PostgresStorage,
) -> None:
    memory, refinery = await _memory(storage)
    await memory.ingest("acme", name="manual", text=MANUAL)
    await memory.remember("acme", "The pump seal was replaced in March.")
    since = datetime(2020, 1, 1, tzinfo=UTC)
    served = await memory.search(
        "acme", "pump seal", kinds=("story",), filters={"created_at": {"gte": since}}
    )
    assert served.retrieval_id is not None and [h.kind for h in served.hits] == ["story"]
    recording = await refinery.retrieval("acme", served.retrieval_id)
    assert recording.arguments["kinds"] == ["story"]
    assert recording.arguments["filters"] == {"created_at": {"gte": since.isoformat()}}
    [replayed] = [p async for p in refinery.replay("acme", memory, holdout=None, labeled=False)]
    assert replayed.error is None and replayed.candidate is not None
    assert [h.id for h in replayed.candidate.hits] == [h.id for h in served.hits]


async def test_a_listing_reads_a_naive_since_as_utc_and_refuses_a_forged_cursor(
    storage: PostgresStorage,
) -> None:
    memory, refinery = await _memory(storage)
    await memory.ingest("acme", name="manual", text=MANUAL)
    await memory.search("acme", "pump seal")
    yesterday = datetime.now(UTC) - timedelta(days=1)
    assert len((await refinery.retrievals("acme", since=yesterday.replace(tzinfo=None)))[0]) == 1
    for cursor in (str(2**63), "-1", "x"):
        with pytest.raises(
            InputError, match=r"^refinery: cursor .* is not one a listing returned$"
        ):
            await refinery.retrievals("acme", cursor=cursor)


class Truncating(HashEmbedding):
    settings = {"truncate": "end"}


class Judge(Fixed):
    model = "judge-1"
    settings = {"effort": "low"}


async def test_a_recorded_hit_s_judgement_names_its_model_and_settings(
    storage: PostgresStorage,
) -> None:
    refinery = MemoryRefinery(storage)
    judge = Judge(
        lambda state, questions: {name: CheckAnswer(probability=0.9) for name in questions}
    )
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery, decision=judge)
    await memory.check()
    await memory.ingest("acme", name="manual", text=MANUAL)
    hits = await memory.search("acme", "pump seal", judge=True)
    assert hits.retrieval_id is not None
    recording = await refinery.retrieval("acme", hits.retrieval_id)
    assert recording.hits
    assert all(
        hit.judge is not None
        and hit.judge.model == "judge-1"
        and hit.judge.settings == {"effort": "low"}
        for hit in recording.hits
    )


async def test_the_recorded_fingerprint_names_the_embedding_s_settings(
    storage: PostgresStorage,
) -> None:
    async def fingerprint(**options: Any) -> dict[str, Any]:
        refinery = MemoryRefinery(storage)
        memory = Memory(storage, refinery=refinery, **options)
        await memory.check()
        await memory.ingest("acme", name="manual", text=MANUAL)
        hits = await memory.search("acme", "pump seal")
        assert hits.retrieval_id is not None
        return (await refinery.retrieval("acme", hits.retrieval_id)).fingerprint

    assert (await fingerprint(embedding=Truncating()))["embedding_settings"] == {"truncate": "end"}
    assert (await fingerprint(embedding=HashEmbedding()))["embedding_settings"] is None
    assert (await fingerprint(embedding=Truncating(), pillars={"text"}))[
        "embedding_settings"
    ] is None
