from typing import Any

from rfnry_memory.memory import Memory
from rfnry_memory.models import Match
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.ranking import CoverDensity

from .fakes import HashEmbedding

A = "Replace part A-1001 before the pump starts."
B = "Replace part B-1001 before the pump starts."


async def _memory(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    return memory


async def _names(memory: Memory, query: str, match: Match, **options: Any) -> list[str]:
    hits = await memory.search(
        "acme", query, kinds=("chunk",), pillars={"text"}, match=match, **options
    )
    return [h.chunk.source_name for h in hits.hits if h.chunk is not None]


async def test_memory_codes_keeps_each_code_whole_once_per_time_it_appears(
    storage: PostgresStorage, pool: Any
) -> None:
    async with pool.acquire() as conn:
        held = await conn.fetchval(
            "SELECT memory_codes($1)::text",
            "A-1001, a-1001 and MAX_SPEED on 2026-10-23: an e-mail follow-up about ACME#42.",
        )
        prose = await conn.fetchval(
            "SELECT memory_codes($1)::text", "An e-mail follow-up, co-operate."
        )
    assert held == "'2026-10-23':1 'a-1001':1,2 'acme#42':1 'max_speed':1"
    assert prose == ""


async def test_a_code_is_needed_whole_under_all_and_phrase_and_ranks_first_otherwise(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.ingest("acme", name="a", text=A)
    await memory.ingest("acme", name="b", text=B)
    assert await _names(memory, "A-1001", "all") == ["a"]
    assert await _names(memory, "part A-1001", "phrase") == ["a"]
    assert await _names(memory, "A-1001", "any") == ["a", "b"]
    assert await _names(memory, "A-1001", "web") == ["a", "b"]
    assert await _names(memory, "A-1001", "all", text=CoverDensity()) == ["a"]
    assert await _names(memory, "C-1001", "all") == []


async def test_codes_joined_by_underscores_colons_and_digits_stay_whole(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.ingest("acme", name="one", text="Tag Line3_PLC:Motor1_Run trips the motor.")
    await memory.ingest("acme", name="two", text="Tag Line3_PLC:Motor2_Run trips the motor.")
    await memory.ingest("acme", name="eth", text="The 1756-EN2T module links the rack.")
    await memory.ingest("acme", name="near", text="The 1756 rack and an EN2T module.")
    assert await _names(memory, "Line3_PLC:Motor1_Run", "all") == ["one"]
    assert await _names(memory, "1756-EN2T", "all") == ["eth"]
    assert await _names(memory, "1756-EN2T", "phrase") == ["eth"]
    assert (await _names(memory, "1756-EN2T", "any"))[0] == "eth"


async def test_prose_with_hyphens_reads_as_it_did(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    await memory.ingest("acme", name="mail", text="Send the e-mail follow-up today.")
    assert await _names(memory, "e-mails follow-up", "all") == ["mail"]
    assert await _names(memory, "the e-mails", "phrase") == ["mail"]
    assert await _names(memory, "co-operation", "all") == []


async def test_a_code_counts_toward_matching_not_toward_length(
    storage: PostgresStorage, pool: Any
) -> None:
    memory = await _memory(storage)
    await memory.ingest("acme", name="a", text="A-1001 A-1001 seal")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT fts_length, fts::text AS fts FROM memory_chunks WHERE corpus_id = 'acme'"
        )
    assert row["fts_length"] == 3
    assert "'a-1001':" in row["fts"]


async def test_a_query_with_a_code_spots_the_names_it_spotted_before(
    storage: PostgresStorage,
) -> None:
    from rfnry_memory.models import EntityIn

    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "graph"})
    await memory.check()
    await memory.ingest(
        "acme",
        name="alarms",
        text="Alarm A-203 trips the heater.",
        entities=[EntityIn(name="A-203", type="alarm")],
    )
    spotted = await memory.storage.entities_named_in("acme", "is B-203 open?", spot="words")
    assert [e.name for e in spotted] == ["A-203"]
