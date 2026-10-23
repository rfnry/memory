import random
import re
from datetime import timedelta

import pytest

from rfnry_memory.errors import InputError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.tools import MemoryTools

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
    assert [m.source.source_id for m in found.matches] == [
        *sorted([first.source_id, renamed.source_id]),
        edition.source_id,
    ]
    assert [m.share for m in found.matches[:2]] == [1.0, 1.0]
    assert 0.8 < found.matches[2].share < 1.0
    assert found.matches[2].source == edition


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
    assert [(m.source.corpus_id, m.source.source_id) for m in found.matches] == [
        ("sales", sales.source_id)
    ]
    assert (await memory.resembling("legal", source_id=legal.source_id)).matches == []


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
        found = await memory.resembling("acme", text=POLICY, **options)  # type: ignore[arg-type]
        return {m.source.source_id for m in found.matches}

    assert await ids() == {v2.source_id, copy.source_id}
    assert await ids(history=True) == {v1.source_id, v2.source_id, copy.source_id}
    assert await ids(known_at=v2.created_at - timedelta(microseconds=1)) == {v1.source_id}
    assert await ids(history=True, filters=doc) == {v1.source_id, v2.source_id}
    assert await ids(history=True, filters={"written_by.run": 1}) == {v1.source_id}
    retired = await memory.resembling(
        "acme", text=POLICY, history=True, filters={"written_by.run": 1}
    )
    assert retired.matches[0].source.retired_at is not None and retired.matches[0].share == 1.0
    assert len((await memory.resembling("acme", text=POLICY, history=True, limit=1)).matches) == 1


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
    with pytest.raises(InputError, match=r"^memory: source_corpus goes with source_id$"):
        await memory.resembling("acme", text=POLICY, source_corpus="acme")


async def test_a_stored_probe_is_read_whatever_moment_the_documents_are_read_at(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    first = (await memory.ingest("acme", name="Returns", text=POLICY)).source
    copy = (await memory.ingest("acme", name="Returns copy", text=POLICY)).source
    found = await memory.resembling(
        "acme", source_id=copy.source_id, known_at=copy.created_at - timedelta(microseconds=1)
    )
    assert [(m.source.source_id, m.share) for m in found.matches] == [(first.source_id, 1.0)]


async def test_the_resembling_tool_finds_versions_inside_the_view(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    first = (
        await memory.ingest("acme", name="Returns", text=POLICY, metadata={"line": "A"})
    ).source
    edition = (
        await memory.ingest("acme", name="Returns 2026", text=EDITION, metadata={"line": "A"})
    ).source
    elsewhere = (
        await memory.ingest("acme", name="Returns B", text=POLICY, metadata={"line": "B"})
    ).source
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    [definition] = [d for d in tools.definitions() if d["name"] == "resembling"]
    assert set(definition["input_schema"]["properties"]) == {
        "text",
        "source_id",
        "filters",
        "as_of",
        "known_at",
        "history",
        "limit",
    }
    assert "before ingest" in definition["description"]
    assert "words, not meaning" in definition["description"]
    found = await tools.dispatch("resembling", {"text": POLICY})
    assert isinstance(found, dict)
    assert [(r["source_id"], r["share"]) for r in found["matches"]][0] == (first.source_id, 1.0)
    assert [r["source_id"] for r in found["matches"]] == [first.source_id, edition.source_id]
    assert set(found["matches"][0]) == {
        "source_id",
        "name",
        "share",
        "metadata",
        "written_by",
        "created_at",
        "confirmed_at",
    }
    assert found["matches"][1]["share"] == round(found["matches"][1]["share"], 2)
    versions = await tools.dispatch("resembling", {"source_id": first.source_id})
    assert isinstance(versions, dict)
    assert [r["source_id"] for r in versions["matches"]] == [edition.source_id]
    with pytest.raises(
        NotFoundError, match=rf"^tools: source '{elsewhere.source_id}' was not found$"
    ):
        await tools.dispatch("resembling", {"source_id": elsewhere.source_id})
    with pytest.raises(InputError, match=r"^tools: filter 'metadata.line' can narrow"):
        await tools.dispatch("resembling", {"text": POLICY, "filters": {"metadata.line": "B"}})


async def test_the_resembling_tool_reads_the_corpora_it_may(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    legal = (await memory.ingest("legal", name="Returns policy", text=POLICY)).source
    sales = (await memory.ingest("sales", name="Returns FAQ", text=EDITION)).source
    await memory.ingest("support", name="Returns", text=POLICY)
    tools = MemoryTools(memory, corpus_id=["legal", "sales"])
    assert "retire_source" not in tools.names
    [definition] = [d for d in tools.definitions() if d["name"] == "resembling"]
    properties = definition["input_schema"]["properties"]
    assert properties["source_corpus"]["enum"] == ["legal", "sales"]
    assert "anyOf" in properties["corpus_id"]
    found = await tools.dispatch("resembling", {"text": POLICY})
    assert isinstance(found, dict)
    assert [(r["corpus_id"], r["source_id"]) for r in found["matches"]] == [
        ("legal", legal.source_id),
        ("sales", sales.source_id),
    ]
    with pytest.raises(
        InputError,
        match=r"^tools: source_corpus is required: name the corpus that holds source_id, one of legal, sales$",
    ):
        await tools.dispatch("resembling", {"source_id": legal.source_id})
    versions = await tools.dispatch(
        "resembling", {"source_id": legal.source_id, "source_corpus": "legal"}
    )
    assert isinstance(versions, dict)
    assert [(r["corpus_id"], r["source_id"]) for r in versions["matches"]] == [
        ("sales", sales.source_id)
    ]
