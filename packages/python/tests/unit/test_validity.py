from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from rfnry_memory.errors import InputError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Counts, EntityIn, Section, Source, SourceIn
from rfnry_memory.postgres import PostgresStorage

from .fakes import HashEmbedding

MARCH = datetime(2024, 3, 1, tzinfo=UTC)
LAPSES = datetime(2027, 2, 28, tzinfo=UTC)
THIRTY = "Returns are accepted within 30 days."
SIXTY = "Returns are accepted within 60 days."


class Counting(HashEmbedding):
    embeds = 0

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        Counting.embeds += 1
        return await super().embed(texts, purpose=purpose)


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


def _names(sources: list[Source]) -> list[str]:
    return sorted(s.name for s in sources)


async def _texts(memory: Memory, **window: Any) -> list[str]:
    hits = await memory.search("acme", "returns policy accepted", top_k=20, **window)
    return sorted(h.text for h in hits.hits if h.chunk is not None)


async def test_every_read_answers_with_the_documents_in_force_at_its_moment(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    soon = datetime.now(UTC) + timedelta(days=30)
    for name, valid_from, valid_to in (
        ("contract", MARCH, LAPSES),
        ("expired", datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)),
        ("scheduled", soon, None),
        ("plain", None, None),
    ):
        await memory.ingest(
            "acme",
            name=name,
            text=f"The {name} returns policy: returns are accepted.",
            valid_from=valid_from,
            valid_to=valid_to,
        )
    then = datetime(2020, 6, 1, tzinfo=UTC)
    assert _names((await memory.list_sources("acme"))[0]) == ["contract", "plain"]
    assert _names((await memory.list_sources("acme", as_of=then))[0]) == ["expired", "plain"]
    later = soon + timedelta(days=1)
    assert _names((await memory.list_sources("acme", as_of=later))[0]) == [
        "contract",
        "plain",
        "scheduled",
    ]
    assert len((await memory.list_sources("acme", history=True))[0]) == 4
    assert [t.split()[1] for t in await _texts(memory)] == ["contract", "plain"]
    assert [t.split()[1] for t in await _texts(memory, as_of=then)] == ["expired", "plain"]
    assert len(await _texts(memory, history=True)) == 4
    with pytest.raises(InputError, match=r"^memory: history=True reads every moment"):
        await memory.list_sources("acme", history=True, as_of=then)


REFUSED = (
    r"^memory: source 'Returns policy' takes effect at .+, after this write; a version in force "
    r"is replaced on the day its update takes effect: write the update then$"
)


async def test_an_update_that_starts_after_its_write_waits_for_its_day_and_a_plan_withdrawn(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=Counting())
    await memory.check()
    doc = {"metadata.doc": "returns"}
    source = SourceIn(name="Returns policy", text=THIRTY, metadata={"doc": "returns"})
    later = datetime.now(UTC) + timedelta(days=30)
    scheduled = source.model_copy(update={"text": SIXTY, "valid_from": later})
    plain = (await memory.sync("plain", filters=doc, sources=[source])).sources[0]
    embedded = Counting.embeds
    with pytest.raises(InputError, match=REFUSED):
        await memory.sync("plain", filters=doc, sources=[scheduled])
    assert Counting.embeds == embedded
    assert await memory.get_source("plain", plain.source_id) == plain
    await memory.set_history("acme", True)
    first = (await memory.sync("acme", filters=doc, sources=[source])).sources[0]
    waiting = await memory.sync("acme", filters=doc, sources=[scheduled])
    assert waiting.retired == [first.source_id]
    [old] = await _versions(memory, first.source_id)
    assert (old.valid_to, old.replaced_by) == (later, waiting.sources[0].source_id)
    assert await _texts(memory) == [THIRTY]
    assert await _texts(memory, as_of=later + timedelta(days=1)) == [SIXTY]
    plan = {"metadata.doc": "fair"}
    fair = SourceIn(
        name="Fair rules",
        text="The fair opens at nine.",
        metadata={"doc": "fair"},
        valid_from=later,
    )
    planned = (await memory.sync("acme", filters=plan, sources=[fair])).sources[0]
    moved = (
        await memory.sync(
            "acme",
            filters=plan,
            sources=[
                fair.model_copy(
                    update={"text": "The fair opens at ten.", "valid_from": later + timedelta(7)}
                )
            ],
        )
    ).sources[0]
    [withdrawn] = await _versions(memory, planned.source_id)
    assert withdrawn.valid_from == withdrawn.valid_to == later
    assert withdrawn.replaced_by == moved.source_id and moved.valid_from == later + timedelta(7)
    assert (await memory.list_sources("acme", filters=plan, as_of=later + timedelta(1)))[0] == []
    [opens] = (await memory.list_sources("acme", filters=plan, as_of=later + timedelta(8)))[0]
    assert opens.source_id == moved.source_id


async def test_an_update_with_no_dates_takes_over_when_it_is_written(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.set_history("acme", True)
    doc = {"metadata.doc": "returns"}
    source = SourceIn(name="Returns policy", text=THIRTY, metadata={"doc": "returns"})
    first = (await memory.sync("acme", filters=doc, sources=[source])).sources[0]
    second = (
        await memory.sync("acme", filters=doc, sources=[source.model_copy(update={"text": SIXTY})])
    ).sources[0]
    [old] = await _versions(memory, first.source_id)
    assert old.valid_to is not None and old.valid_to == old.retired_at
    assert old.valid_to == second.valid_from == second.created_at and first.valid_from is None
    before = old.valid_to - timedelta(microseconds=1)
    assert await _texts(memory, as_of=before) == [THIRTY]
    assert await _texts(memory) == [SIXTY]
    gone = await memory.sync("acme", filters=doc, sources=[])
    [ended] = await _versions(memory, second.source_id)
    assert gone.retired == [second.source_id] and ended.valid_to is not None
    assert ended.valid_to == ended.retired_at and await _texts(memory) == []
    assert await _texts(memory, as_of=ended.valid_to - timedelta(microseconds=1)) == [SIXTY]


async def test_an_update_reaches_back_to_the_start_of_what_it_replaces_and_no_further(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.set_history("acme", True)
    doc = {"metadata.doc": "returns"}
    source = SourceIn(
        name="Returns policy", text=THIRTY, metadata={"doc": "returns"}, valid_from=MARCH
    )
    first = (await memory.sync("acme", filters=doc, sources=[source])).sources[0]
    january = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(
        InputError,
        match=r"^memory: source 'Returns policy' takes effect at 2024-01-01T00:00:00\+00:00, before the version "
        r"it replaces took effect, at 2024-03-01T00:00:00\+00:00; that moment itself corrects it from the start$",
    ):
        await memory.sync(
            "acme",
            filters=doc,
            sources=[source.model_copy(update={"text": SIXTY, "valid_from": january})],
        )
    second = (
        await memory.sync("acme", filters=doc, sources=[source.model_copy(update={"text": SIXTY})])
    ).sources[0]
    [old] = await _versions(memory, first.source_id)
    assert old.valid_from == old.valid_to == MARCH and second.valid_from == MARCH
    assert await _texts(memory, as_of=datetime(2024, 6, 1, tzinfo=UTC)) == [SIXTY]
    third = (
        await memory.sync(
            "acme",
            filters=doc,
            sources=[source.model_copy(update={"text": "Ninety.", "valid_from": None})],
        )
    ).sources[0]
    assert third.valid_from is not None and third.valid_from > MARCH
    with pytest.raises(InputError, match=r"before the version it replaces took effect"):
        await memory.ingest(
            "acme", name="Returns policy", text="x", valid_from=MARCH, replaces=third.source_id
        )
    for moment in (MARCH, datetime(2024, 6, 1, tzinfo=UTC), datetime.now(UTC)):
        assert len(await _texts(memory, as_of=moment)) == 1


async def test_twins_of_a_name_are_each_checked_before_any_model_runs(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=Counting())
    await memory.check()
    held = {"doc": "twins"}
    await memory.ingest(
        "acme",
        name="A",
        text="The old A.",
        metadata=held,
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        valid_to=datetime(2021, 1, 1, tzinfo=UTC),
    )
    await memory.ingest("acme", name="A", text="The current A.", metadata=held)
    embedded = Counting.embeds
    later = datetime.now(UTC) + timedelta(days=30)
    with pytest.raises(InputError, match=r"^memory: source 'A' takes effect at "):
        await memory.sync(
            "acme",
            filters={"metadata.doc": "twins"},
            sources=[SourceIn(name="A", text="The next A.", metadata=held, valid_from=later)],
        )
    assert Counting.embeds == embedded


async def test_an_expired_version_can_be_followed_by_one_that_starts_later(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.set_history("acme", True)
    doc = {"metadata.doc": "fair"}
    lapsed = datetime(2021, 1, 1, tzinfo=UTC)
    old = SourceIn(
        name="Fair rules",
        text="The fair opens at nine.",
        metadata={"doc": "fair"},
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        valid_to=lapsed,
    )
    first = (await memory.sync("acme", filters=doc, sources=[old])).sources[0]
    later = datetime.now(UTC) + timedelta(days=30)
    nxt = old.model_copy(
        update={"text": "The fair opens at ten.", "valid_from": later, "valid_to": None}
    )
    second = (await memory.sync("acme", filters=doc, sources=[nxt])).sources[0]
    [closed] = await _versions(memory, first.source_id)
    assert closed.valid_to == lapsed and second.valid_from == later


def test_a_source_reads_its_validity_in_utc_so_one_instant_has_one_digest(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"text"})
    plus_two = timezone(timedelta(hours=2))
    shifted = SourceIn(
        name="Contract", text="x", valid_from=datetime(2024, 3, 1, 2, tzinfo=plus_two)
    )
    plain = SourceIn(name="Contract", text="x", valid_from=MARCH)
    assert shifted.valid_from == MARCH and shifted.valid_from.tzinfo == UTC
    assert memory._digest_of(shifted) == memory._digest_of(plain)


async def test_graph_evidence_and_expand_follow_the_document_in_force(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    await memory.set_history("acme", True)
    pump = [EntityIn(name="P-3", type="pump")]
    old = (
        await memory.ingest(
            "acme",
            name="P-3 notes",
            sections=[Section(text="P-3 leaks at the seal."), Section(text="Check the gasket.")],
            entities=pump,
        )
    ).source
    new = (
        await memory.ingest(
            "acme",
            name="P-3 notes",
            text="P-3 runs quietly.",
            entities=pump,
            replaces=old.source_id,
        )
    ).source
    before = new.created_at - timedelta(microseconds=1)
    now_hood = await memory.neighbors("acme", "P-3")
    then_hood = await memory.neighbors("acme", "P-3", as_of=before)
    assert {h.text for h in now_hood.passages} == {"P-3 runs quietly."}
    assert {h.text for h in then_hood.passages} == {"P-3 leaks at the seal."}
    start = ("chunk", f"{old.source_id}-0")
    beside = await memory.expand("acme", start, by="document", as_of=before)
    assert [h.text for h in beside.hits] == ["Check the gasket."]
    assert (await memory.expand("acme", start, by="document")).hits == []


async def test_without_history_a_sync_erases_what_it_replaces_and_sequences_nothing(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    doc = {"metadata.doc": "returns"}
    source = SourceIn(name="Returns policy", text=THIRTY, metadata={"doc": "returns"})
    await memory.sync("acme", filters=doc, sources=[source])
    plain = (
        await memory.sync("acme", filters=doc, sources=[source.model_copy(update={"text": SIXTY})])
    ).sources[0]
    assert plain.valid_from is None
    assert await _texts(memory, as_of=datetime(2020, 1, 1, tzinfo=UTC)) == [SIXTY]
    takes_effect = datetime.now(UTC) + timedelta(days=30)
    scheduled = source.model_copy(update={"text": "Returns: 90 days.", "valid_from": takes_effect})
    with pytest.raises(InputError, match=REFUSED):
        await memory.sync("acme", filters=doc, sources=[scheduled])
    assert await _texts(memory) == [SIXTY]


async def _versions(memory: Memory, *source_ids: str) -> list[Source]:
    return [s for s in [await memory.get_source("acme", i) for i in source_ids] if s is not None]


async def test_ingest_replaces_a_document_it_names_in_valid_time_where_history_is_kept(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.set_history("acme", True)
    old = (await memory.ingest("acme", name="Returns policy", text=THIRTY)).source
    replacing = await memory.ingest(
        "acme", name="Returns policy, 2026", text=SIXTY, replaces=old.source_id
    )
    new = replacing.source
    assert replacing.retired == [old.source_id] and replacing.deleted == Counts()
    [closed] = await _versions(memory, old.source_id)
    assert closed.replaced_by == new.source_id and closed.retired_at == new.created_at
    assert closed.valid_to == new.valid_from == new.created_at
    assert await _texts(memory) == [SIXTY]
    assert await _texts(memory, as_of=new.created_at - timedelta(microseconds=1)) == [THIRTY]
    with pytest.raises(
        InputError,
        match=rf"^memory: source '{old.source_id}' was already replaced by '{new.source_id}'; replace that one$",
    ):
        await memory.ingest("acme", name="x", text="x", replaces=old.source_id)
    with pytest.raises(NotFoundError, match=r"^memory: source 'missing' was not found$"):
        await memory.ingest("acme", name="x", text="x", replaces="missing")
    with pytest.raises(NotFoundError):
        await memory.ingest("other", name="x", text="x", replaces=new.source_id)
    later = datetime.now(UTC) + timedelta(days=30)
    scheduled = await memory.ingest(
        "acme",
        name="Returns policy, 2027",
        text="Returns: 90 days.",
        valid_from=later,
        replaces=new.source_id,
    )
    [waiting] = await _versions(memory, new.source_id)
    assert (waiting.valid_to, waiting.replaced_by) == (later, scheduled.source.source_id)
    assert await _texts(memory) == [SIXTY]
    await memory.sync("acme", filters={"metadata.doc": "gone"}, sources=[])
    removed = (
        await memory.ingest("acme", name="Gone", text="Gone soon.", metadata={"doc": "gone"})
    ).source
    dropped = await memory.sync("acme", filters={"metadata.doc": "gone"}, sources=[])
    assert dropped.retired == [removed.source_id]
    [ended] = await _versions(memory, removed.source_id)
    back = await memory.ingest("acme", name="Gone", text="Back.", replaces=removed.source_id)
    [kept] = await _versions(memory, removed.source_id)
    assert (kept.valid_to, kept.replaced_by) == (ended.valid_to, back.source.source_id)
    assert kept.revised_at == kept.retired_at == back.source.created_at


async def test_ingest_replaces_by_erasing_where_no_history_is_kept(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    old = (await memory.ingest("acme", name="Returns policy", text=THIRTY)).source
    replacing = await memory.ingest(
        "acme", name="Returns policy", text=SIXTY, replaces=old.source_id
    )
    assert replacing.retired == [] and replacing.deleted.sources == 1
    assert replacing.deleted.chunks == 1 and await memory.get_source("acme", old.source_id) is None
    assert replacing.source.valid_from is None
    assert await _texts(memory, as_of=datetime(2020, 1, 1, tzinfo=UTC)) == [SIXTY]


async def test_a_sync_that_keeps_a_document_confirms_it(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    doc = {"metadata.doc": "returns"}
    source = SourceIn(name="Returns policy", text=THIRTY, metadata={"doc": "returns"})
    first = (await memory.sync("acme", filters=doc, sources=[source])).sources[0]
    again = await memory.sync("acme", filters=doc, sources=[source])
    [kept] = again.sources
    assert again.kept == [first.source_id] and kept.created_at == first.created_at
    assert kept.confirmed_at > first.confirmed_at
    assert await memory.get_source("acme", first.source_id) == kept
    [hit] = [h for h in (await memory.search("acme", "returns accepted")).hits if h.chunk]
    assert hit.chunk is not None and hit.chunk.created_at == first.created_at
    assert hit.chunk.confirmed_at == kept.confirmed_at


async def test_the_validity_given_is_part_of_the_fingerprint(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    doc = {"metadata.doc": "returns"}
    source = SourceIn(name="Returns policy", text=THIRTY, metadata={"doc": "returns"})
    await memory.sync("acme", filters=doc, sources=[source])
    plain = (await memory.fingerprint("acme")).content
    dated = source.model_copy(update={"valid_to": LAPSES})
    await memory.sync("acme", filters=doc, sources=[dated])
    assert (await memory.fingerprint("acme")).content != plain
    await memory.sync("acme", filters=doc, sources=[source])
    assert (await memory.fingerprint("acme")).content == plain
