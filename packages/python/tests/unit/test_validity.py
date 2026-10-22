from datetime import UTC, datetime, timedelta

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Section, SourceIn
from rfnry_memory.postgres import PostgresStorage

from .fakes import HashEmbedding

MARCH = datetime(2024, 3, 1, tzinfo=UTC)
LAPSES = datetime(2027, 2, 28, tzinfo=UTC)


async def _memory(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    return memory


async def test_a_source_carries_the_validity_it_is_given_and_its_chunks_carry_its_dates(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    contract = (
        await memory.ingest(
            "acme",
            name="Supply contract",
            text="Acme supplies the feed valves.",
            valid_from=MARCH.replace(tzinfo=None),
            valid_to=LAPSES,
        )
    ).source
    assert (contract.valid_from, contract.valid_to) == (MARCH, LAPSES)
    assert contract.confirmed_at == contract.created_at
    assert await memory.get_source("acme", contract.source_id) == contract
    [chunk] = await storage.source_chunks("acme", contract.source_id)
    assert (chunk.created_at, chunk.confirmed_at, chunk.valid_from, chunk.valid_to) == (
        contract.created_at,
        contract.created_at,
        MARCH,
        LAPSES,
    )
    found = await memory.search("acme", "feed valves")
    [hit] = [h for h in found.hits if h.chunk is not None]
    assert hit.chunk == chunk
    plain = (await memory.ingest("acme", name="Manual", text="Ground first.")).source
    assert (plain.valid_from, plain.valid_to, plain.confirmed_at) == (None, None, plain.created_at)


async def test_a_source_ends_after_it_takes_effect(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    for valid_to in (MARCH, MARCH - timedelta(days=1)):
        with pytest.raises(
            InputError,
            match=r"^models: source 'Contract' must end after it takes effect: valid_to after valid_from$",
        ):
            await memory.ingest(
                "acme", name="Contract", text="x", valid_from=MARCH, valid_to=valid_to
            )
        with pytest.raises(InputError, match=r"^models: source 'Contract' must end after it"):
            SourceIn(name="Contract", text="x", valid_from=MARCH, valid_to=valid_to)
    assert (await memory.list_sources("acme"))[0] == []


def test_a_source_without_validity_keeps_the_digest_it_had(storage: PostgresStorage) -> None:
    memory = Memory(storage, pillars={"text"})
    policy = SourceIn(
        name="Returns policy",
        text="Returns are accepted within 30 days.",
        metadata={"doc": "returns"},
    )
    sheet = SourceIn(name="Sheet", sections=[Section(title="A", text="x", locator={"p": 1})])
    assert memory._digest_of(policy) == (
        "cbe00ca8ed58605fa8d6d99a7dc74ce3718945c23d581eae86f23754fd944575"
    )
    assert memory._digest_of(sheet) == (
        "c447181d73fd6da05075f5666697642bcd8e9092c04d18a5a27dc70947b489c5"
    )
    dated = policy.model_copy(update={"valid_from": MARCH})
    assert memory._digest_of(dated) != memory._digest_of(policy)
