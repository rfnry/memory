from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.filters import matches_filters
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn
from rfnry_memory.postgres import PostgresStorage

from .fakes import HashEmbedding

ROWS: list[dict[str, Any]] = [
    {"doc": "a", "confidence": 0.95, "signed": "2024-03-01", "scores": [1, 5]},
    {"doc": "b", "confidence": 0.6, "signed": "2023-11-20", "scores": [7]},
    {"doc": "c", "confidence": 1, "signed": "2025-02-01"},
    {"doc": "d", "confidence": "high", "signed": 2024},
    {"doc": "e", "owner": {"ext": 318}, "flag": True},
]

RANGES: list[dict[str, Any]] = [
    {"metadata.confidence": {"gte": 0.8}},
    {"metadata.confidence": {"gt": 0.6, "lt": 1}},
    {"metadata.signed": {"gte": "2024-01-01", "lt": "2025-01-01"}},
    {"metadata.scores": {"gt": 4, "lt": 6}},
    {"metadata.owner.ext": {"lte": 318}},
    {"metadata.flag": {"gte": 0}},
    {"metadata.confidence": 1.0},
    {"metadata.confidence": [0.6, 0.95]},
    {"metadata.confidence": {"gte": 0.5}, "metadata.signed": {"lt": "2024"}},
]


def _expected(filters: dict[str, Any]) -> list[str]:
    return sorted(row["doc"] for row in ROWS if matches_filters(row, filters))


async def test_every_read_selects_by_range_as_python_matches(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    for row in ROWS:
        await memory.ingest("acme", name=row["doc"], text=f"Valve note {row['doc']}.", metadata=row)
        await memory.remember("acme", f"Valve fact {row['doc']}.", metadata=row)
    for filters in RANGES:
        expected = _expected(filters)
        listed, _ = await memory.list_sources("acme", filters=filters)
        stories, _ = await memory.list_stories("acme", filters=filters)
        found = await memory.search("acme", "valve", filters=filters, top_k=20)
        assert sorted(s.name for s in listed) == expected, filters
        assert sorted(s.metadata["doc"] for s in stories) == expected, filters
        chunks = sorted({h.chunk.metadata["doc"] for h in found.hits if h.chunk is not None})
        told = sorted({h.story.metadata["doc"] for h in found.hits if h.story is not None})
        assert chunks == told == expected, filters
    assert _expected(RANGES[0]) == ["a", "c"] and _expected(RANGES[2]) == ["a"]


async def test_a_sync_takes_a_range_and_its_sources_must_fall_in_it(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    revs = {"metadata.rev": {"gte": 3}}
    synced = await memory.sync(
        "acme",
        filters=revs,
        sources=[SourceIn(name="Manual", text="Rev three.", metadata={"rev": 3})],
    )
    assert [s.metadata for s in synced.sources] == [{"rev": 3}]
    with pytest.raises(
        InputError, match=r"^memory: source 'Older' does not match the filters it syncs$"
    ):
        await memory.sync(
            "acme",
            filters=revs,
            sources=[SourceIn(name="Older", text="Rev two.", metadata={"rev": 2})],
        )
    old = await memory.ingest("acme", name="Old manual", text="Rev two.", metadata={"rev": 2})
    await memory.sync("acme", filters=revs, sources=[])
    left, _ = await memory.list_sources("acme")
    assert [s.source_id for s in left] == [old.source.source_id]
