import random
import re
from datetime import timedelta

import pytest

from rfnry_memory.errors import InputError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn
from rfnry_memory.postgres import PostgresStorage

from .fakes import HashEmbedding

WORDS = [
    "pump",
    "seal",
    "valve",
    "line",
    "tank",
    "feed",
    "return",
    "replace",
    "inspect",
    "motor",
    "panel",
    "wiring",
    "order",
    "refund",
    "delivery",
    "customer",
    "invoice",
    "contract",
    "clause",
    "notice",
    "period",
    "renewal",
    "warranty",
    "repair",
    "part",
    "supplier",
    "shipment",
    "damage",
    "claim",
    "policy",
    "days",
    "receipt",
    "store",
    "credit",
    "exchange",
    "label",
    "box",
]


def _paragraphs(seed: int, count: int = 12) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(40)) for _ in range(count)]


def _text(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs)


POLICY = _text(_paragraphs(1))
EDITION = _text(_paragraphs(2, 1) + _paragraphs(1)[1:])
UNRELATED = _text(_paragraphs(3))


async def _memory(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    return memory


async def test_a_renamed_copy_shares_everything_and_an_edition_most(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    first = (await memory.ingest("acme", name="Returns policy 2025", text=POLICY)).source
    renamed = (await memory.ingest("acme", name="RP-final-v2.pdf", text=POLICY)).source
    edition = (await memory.ingest("acme", name="Returns 2026", text=EDITION)).source
    await memory.ingest("acme", name="Seal manual", text=UNRELATED)
    found = await memory.resembling("acme", text=POLICY)
    assert [m.source.source_id for m in found] == [
        *sorted([first.source_id, renamed.source_id]),
        edition.source_id,
    ]
    assert [m.share for m in found[:2]] == [1.0, 1.0]
    assert 0.8 < found[2].share < 1.0
    assert found[2].source == edition


async def test_a_stored_document_finds_its_versions_in_the_corpora_named(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    legal = (await memory.ingest("legal", name="Returns policy", text=POLICY)).source
    sales = (await memory.ingest("sales", name="Returns FAQ", text=EDITION)).source
    await memory.ingest("support", name="Returns", text=POLICY)
    found = await memory.resembling(
        ["sales", "legal"], source_id=legal.source_id, source_corpus="legal"
    )
    assert [(m.source.corpus_id, m.source.source_id) for m in found] == [("sales", sales.source_id)]
    assert await memory.resembling("legal", source_id=legal.source_id) == []


async def test_the_documents_read_follow_list_sources(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    await memory.set_history("acme", True)
    doc = {"metadata.doc": "returns"}
    first = SourceIn(name="Returns", text=POLICY, metadata={"doc": "returns"})
    [v1] = (await memory.sync("acme", filters=doc, sources=[first], written_by={"run": 1})).sources
    second = first.model_copy(update={"text": EDITION})
    [v2] = (await memory.sync("acme", filters=doc, sources=[second], written_by={"run": 2})).sources
    copy = (await memory.ingest("acme", name="Copy", text=POLICY, metadata={"doc": "copy"})).source

    async def ids(**options: object) -> set[str]:
        return {m.source.source_id for m in await memory.resembling("acme", text=POLICY, **options)}  # type: ignore[arg-type]

    assert await ids() == {v2.source_id, copy.source_id}
    assert await ids(history=True) == {v1.source_id, v2.source_id, copy.source_id}
    assert await ids(known_at=v2.created_at - timedelta(microseconds=1)) == {v1.source_id}
    assert await ids(history=True, filters=doc) == {v1.source_id, v2.source_id}
    assert await ids(history=True, filters={"written_by.run": 1}) == {v1.source_id}
    retired = await memory.resembling(
        "acme", text=POLICY, history=True, filters={"written_by.run": 1}
    )
    assert retired[0].source.retired_at is not None and retired[0].share == 1.0
    assert len(await memory.resembling("acme", text=POLICY, history=True, limit=1)) == 1


async def test_what_resembling_refuses(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    held = (await memory.ingest("acme", name="Returns", text=POLICY)).source
    one = "memory: resembling takes exactly one of text= or source_id="
    refused: list[tuple[dict[str, object], str]] = [
        ({}, one),
        ({"text": POLICY, "source_id": held.source_id}, one),
        ({"text": "   "}, "memory: the text holds no words to compare"),
        ({"text": "-- !! --"}, "memory: the text holds no words to compare"),
        ({"text": POLICY, "limit": 0}, "memory: limit must be at least 1"),
    ]
    for options, message in refused:
        with pytest.raises(InputError, match=f"^{re.escape(message)}$"):
            await memory.resembling("acme", **options)  # type: ignore[arg-type]
    with pytest.raises(
        InputError,
        match=r"^memory: name the corpus that holds source '\w+' in source_corpus$",
    ):
        await memory.resembling(["acme", "legal"], source_id=held.source_id)
    with pytest.raises(
        InputError, match=r"^memory: source_corpus 'sales' is not one of the corpora named$"
    ):
        await memory.resembling(["acme", "legal"], source_id=held.source_id, source_corpus="sales")
    with pytest.raises(NotFoundError, match=r"^memory: source 'missing' was not found$"):
        await memory.resembling("acme", source_id="missing")
