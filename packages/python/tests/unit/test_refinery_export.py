import json
from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import CheckAnswer
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery import (
    MemoryRefinery,
    Relevance,
    RetrievalOutcome,
    RetrievalReview,
    refinery as refinery_module,
)

from .fakes import Fixed, HashEmbedding

MANUAL = "Replace the pump seal yearly, after draining the line."
NOTES = "Pump seal notes from March: the seal on line B leaked."
PANEL = "Panel wiring guide for the control cabinet."


class Folder:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.files[key] = data

    def rows(self, key: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.files[key].decode().splitlines()]


async def test_an_export_writes_positives_and_near_misses_with_their_text(
    storage: PostgresStorage,
) -> None:
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery)
    await memory.check()
    ids = {}
    for name, text in (("manual", MANUAL), ("notes", NOTES), ("panel", PANEL)):
        ids[name] = (await memory.ingest("acme", name=name, text=text)).source.source_id
    labeled = (await memory.search("acme", "pump seal", kinds=("chunk",), top_k=2)).retrieval_id
    plain = (await memory.search("acme", "wiring", kinds=("chunk",))).retrieval_id
    assert labeled is not None and plain is not None
    recording = await refinery.retrieval("acme", labeled)
    returned = {h.id for h in recording.hits}
    wrong = next(h.id for h in recording.hits if h.id != f"{ids['manual']}-0")
    review = RetrievalReview(
        relevance=[
            Relevance(corpus_id="acme", kind="chunk", id=f"{ids['manual']}-0", grade=3),
            Relevance(corpus_id="acme", kind="chunk", id=wrong, grade=0),
            Relevance(corpus_id="acme", kind="source", id=ids["panel"], grade=1),
        ]
    )
    await refinery.review("acme", labeled, review)
    await refinery.outcome(
        "acme",
        labeled,
        RetrievalOutcome(cited=[Relevance(corpus_id="acme", kind="chunk", id=wrong)]),
    )
    folder = Folder()
    exported = await refinery.export("acme", folder)
    [key] = exported.keys
    assert key.startswith("memory/acme/retrievals/") and key.endswith(".jsonl")
    assert (exported.rows, exported.missing) == (2, 0)
    rows = {row["retrieval_id"]: row for row in folder.rows(key)}
    row = rows[labeled]
    assert row["holdout"] == refinery.is_holdout(labeled)
    assert row["query"] == "pump seal" and row["verb"] == "search"
    assert {tuple(h) for h in row["hits"]} == {("acme", "chunk", i) for i in returned}
    positives = {(p["kind"], p["id"]): p for p in row["positives"]}
    assert positives[("chunk", f"{ids['manual']}-0")]["text"] == MANUAL
    assert positives[("chunk", f"{ids['manual']}-0")]["source"] == "human"
    assert positives[("source", ids["panel"])]["text"] == PANEL
    texts = {
        f"{ids[name]}-0": text
        for name, text in (("manual", MANUAL), ("notes", NOTES), ("panel", PANEL))
    }
    assert [(n["id"], n["grade"], n["text"]) for n in row["negatives"]] == [
        (wrong, 0, texts[wrong])
    ]
    assert row["answered"] is True
    assert (rows[plain]["positives"], rows[plain]["answered"]) == ([], False)
    await memory.delete_source("acme", ids["panel"])
    again = await refinery.export("acme", Folder())
    assert again.rows == 2 and again.missing == 1 + (wrong == f"{ids['panel']}-0")


async def test_an_export_takes_what_was_recorded_or_labeled_since(
    storage: PostgresStorage,
) -> None:
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery)
    await memory.check()
    await memory.ingest("hr", name="manual", text=MANUAL)
    await memory.ingest("ops/b", name="manual", text=MANUAL)
    first = (await memory.search("hr", "pump seal")).retrieval_id
    assert first is not None
    folder = Folder()
    done = await refinery.export("hr", folder)
    assert done.rows == 1 and done.watermark is not None
    quiet = await refinery.export("hr", folder, since=done.watermark)
    assert (quiet.keys, quiet.rows, quiet.watermark) == ([], 0, done.watermark)
    naive = await refinery.export("hr", folder, since=done.watermark.replace(tzinfo=None))
    assert (naive.rows, naive.watermark) == (0, done.watermark)
    await refinery.review("hr", first, RetrievalReview(comment="late review"))
    relabeled = await refinery.export("hr", folder, since=done.watermark)
    assert relabeled.rows == 1 and relabeled.watermark is not None
    assert relabeled.watermark > done.watermark
    await memory.search(["hr", "ops/b"], "pump seal")
    both = await refinery.export(["ops/b", "hr"], folder)
    [key] = both.keys
    assert key.startswith("memory/hr+ops%2Fb/retrievals/") and both.rows == 2
    with pytest.raises(InputError, match=r"^refinery: holdout_percent must be 0-100, got 101$"):
        await refinery.export("hr", folder, holdout_percent=101)


async def test_an_export_pages_through_every_recording(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refinery_module, "PAGE", 2)
    refinery = MemoryRefinery(storage)
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery)
    await memory.check()
    await memory.ingest("acme", name="manual", text=MANUAL)
    recorded = set()
    for n in range(5):
        retrieval_id = (await memory.search("acme", f"pump seal {n}")).retrieval_id
        assert retrieval_id is not None
        recorded.add(retrieval_id)
    folder = Folder()
    exported = await refinery.export("acme", folder)
    [key] = exported.keys
    assert exported.rows == 5 and {row["retrieval_id"] for row in folder.rows(key)} == recorded


class Judge(Fixed):
    model = "judge-1"
    settings = {"effort": "low"}


async def test_an_exported_row_s_judge_label_carries_the_model_and_settings(
    storage: PostgresStorage,
) -> None:
    refinery = MemoryRefinery(storage)
    judge = Judge(
        lambda state, questions: {name: CheckAnswer(probability=0.9) for name in questions}
    )
    memory = Memory(storage, embedding=HashEmbedding(), refinery=refinery, decision=judge)
    await memory.check()
    await memory.ingest("acme", name="manual", text=MANUAL)
    retrieval_id = (await memory.search("acme", "pump seal", kinds=("chunk",))).retrieval_id
    assert retrieval_id is not None
    assert (await refinery.label("acme", memory)).judged == 1
    folder = Folder()
    [key] = (await refinery.export("acme", folder)).keys
    [row] = folder.rows(key)
    judged = next(label for label in row["labels"] if label["source"] == "judge")
    assert judged["judged"][0]["model"] == "judge-1"
    assert judged["judged"][0]["settings"] == {"effort": "low"}
