import re
from datetime import UTC, datetime
from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import SourceIn
from rfnry_memory.postgres import PostgresStorage

from .fakes import HashEmbedding

JAN, FEB, MAR, APR, MAY = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3, 4, 5))
DEC = datetime(2026, 12, 1, tzinfo=UTC)


async def _memory(storage: PostgresStorage, clock: list[datetime]) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding(), clock=lambda: clock[0])
    await memory.check()
    return memory


def _doc(name: str, text: str, doc: str, valid_to: datetime | None = None) -> SourceIn:
    return SourceIn(name=name, text=text, metadata={"doc": doc}, valid_to=valid_to)


async def test_documents_and_their_passages_filter_by_their_own_dates(
    storage: PostgresStorage,
) -> None:
    clock = [JAN]
    memory = await _memory(storage, clock)
    old = _doc("Returns 2025", "Returns within 30 days.", "a")
    [kept] = (await memory.sync("acme", filters={"metadata.doc": "a"}, sources=[old])).sources
    clock[0] = MAR
    new = _doc("Returns 2026", "Returns within 60 days.", "b", valid_to=DEC)
    [filed] = (await memory.sync("acme", filters={"metadata.doc": "b"}, sources=[new])).sources
    clock[0] = MAY
    await memory.sync("acme", filters={"metadata.doc": "a"}, sources=[old])

    async def names(filters: dict[str, Any]) -> list[str]:
        listed, _ = await memory.list_sources("acme", filters=filters)
        return sorted(s.name for s in listed)

    assert await names({"created_at": {"lt": FEB}}) == ["Returns 2025"]
    assert await names({"created_at": {"lt": datetime(2026, 2, 1)}}) == ["Returns 2025"]
    assert await names({"confirmed_at": {"lt": "2026-04-01T00:00:00Z"}}) == ["Returns 2026"]
    assert await names({"confirmed_at": {"gte": APR}}) == ["Returns 2025"]
    assert await names({"valid_to": {"lt": "2027-01-01"}}) == ["Returns 2026"]
    assert await names({"valid_from": {"lt": DEC}}) == []
    assert await names({"created_at": {"gte": JAN}, "metadata.doc": "b"}) == ["Returns 2026"]
    hits = await memory.search(
        "acme", "returns within days", kinds=("chunk",), filters={"confirmed_at": {"lt": APR}}
    )
    assert {h.chunk.source_id for h in hits.hits if h.chunk} == {filed.source_id}
    alike = await memory.resembling(
        "acme", text="Returns within 30 days.", filters={"created_at": {"lt": FEB}}
    )
    assert [m.source.source_id for m in alike] == [kept.source_id]


async def test_stories_filter_by_their_own_dates(storage: PostgresStorage) -> None:
    clock = [JAN]
    memory = await _memory(storage, clock)
    await memory.remember("acme", "Ana owns the review.")
    clock[0] = MAR
    await memory.remember("acme", "Bo owns the audit.", valid_to=DEC)

    async def texts(filters: dict[str, Any]) -> list[str]:
        listed, _ = await memory.list_stories("acme", filters=filters)
        return sorted(s.text for s in listed)

    assert await texts({"created_at": {"lt": FEB}}) == ["Ana owns the review."]
    assert await texts({"confirmed_at": {"lt": FEB}}) == ["Ana owns the review."]
    assert await texts({"valid_from": {"gte": FEB}}) == ["Bo owns the audit."]
    assert await texts({"valid_to": {"lte": DEC}}) == ["Bo owns the audit."]
    found = await memory.recall("acme", "owns", filters={"valid_from": {"lt": FEB}})
    assert [h.text for h in found.hits] == ["Ana owns the review."]


async def test_a_date_filter_reads_dates_as_the_rows_show_them(storage: PostgresStorage) -> None:
    clock = [JAN]
    memory = await _memory(storage, clock)
    await memory.set_history("acme", True)
    doc = {"metadata.doc": "a"}
    await memory.sync("acme", filters=doc, sources=[_doc("Returns", "30 days.", "a")])
    clock[0] = MAR
    await memory.sync("acme", filters=doc, sources=[_doc("Returns", "60 days.", "a")])
    ending = {"valid_to": {"lt": DEC}}
    every, _ = await memory.list_sources("acme", filters=ending, history=True)
    assert [s.text for s in every] == ["30 days."]
    then, _ = await memory.list_sources("acme", filters=ending, known_at=FEB)
    assert then == []
    with pytest.raises(
        InputError,
        match=re.escape(
            "memory: confirmed_at keeps no history, so a filter on it cannot be read as memory knew things at known_at"
        ),
    ):
        await memory.list_sources("acme", filters={"confirmed_at": {"lt": DEC}}, known_at=FEB)


async def test_what_a_date_filter_refuses(storage: PostgresStorage) -> None:
    memory = await _memory(storage, [JAN])
    ranged = "filters: filter 'created_at' is a range of gt, gte, lt and lte over datetimes or ISO 8601 strings"
    for condition in (
        "2026-01-01",
        ["2026-01-01"],
        {"lt": 5},
        {"lt": "yesterday"},
        {"lt": JAN, "gt": 1},
        {},
    ):
        with pytest.raises(InputError, match=f"^{re.escape(ranged)}$"):
            await memory.list_sources("acme", filters={"created_at": condition})
    with pytest.raises(
        InputError,
        match=r"^memory: a sync names its set by metadata; 'created_at' is not one$",
    ):
        await memory.sync("acme", filters={"created_at": {"lt": JAN}}, sources=[])
