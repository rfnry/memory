from rfnry_memory.judge import state_for
from rfnry_memory.memory import Memory
from rfnry_memory.models import Section, SourceIn
from rfnry_memory.storage.postgres import PostgresStorage
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding


async def _memory(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.initialize()
    return memory


async def test_a_chunk_carries_its_sections_context_and_is_found_by_it_and_its_name(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    ingested = await memory.ingest(
        "acme",
        name="Gasket catalogue",
        sections=[
            Section(
                title="Line items", text="Part 4471 ships in boxes of ten.", context="ACME invoice"
            ),
            Section(title="Terms", text="Payment within thirty days."),
        ],
    )
    chunks = await memory.storage.source_chunks("acme", ingested.source.source_id)
    assert {c.title: c.context for c in chunks} == {"Line items": "ACME invoice", "Terms": None}
    by_context = await memory.search("acme", "invoice", pillars={"text"})
    assert [h.chunk.title for h in by_context.hits if h.chunk] == ["Line items"]
    by_name = await memory.search("acme", "catalogue", pillars={"text"})
    assert sorted(h.chunk.title or "" for h in by_name.hits if h.chunk) == ["Line items", "Terms"]


async def test_a_hit_shows_its_context_to_the_model_and_the_judge_only_when_set(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.ingest(
        "acme",
        name="Gasket catalogue",
        sections=[Section(title="Line items", text="Part 4471 ships.", context="ACME invoice")],
    )
    await memory.ingest("acme", name="Notes", text="Part 4471 is discontinued.")
    tools = MemoryTools(memory, corpus_id="acme")
    found = await tools.dispatch("search", {"query": "4471", "kinds": ["chunk"]})
    assert isinstance(found, dict)
    rows = {row["source_name"]: row for row in found["hits"]}
    assert rows["Gasket catalogue"]["context"] == "ACME invoice"
    assert "context" not in rows["Notes"]
    hits = await memory.search("acme", "4471", kinds=("chunk",))
    states = {h.chunk.source_name: state_for("4471", h)["passage"] for h in hits.hits if h.chunk}
    assert states["Gasket catalogue"]["context"] == "ACME invoice"
    assert "context" not in states["Notes"]


def _catalogue(context: str | None = None) -> SourceIn:
    return SourceIn(
        name="Gasket catalogue",
        sections=[
            Section(
                title="Line items",
                text="Part 4471 ships in boxes of ten.",
                locator={"page": 3},
                context=context,
            ),
            Section(text="Payment within thirty days."),
        ],
    )


def test_an_unchanged_sectioned_source_keeps_the_digest_it_had_before_context_existed(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage)
    assert (
        memory._digest_of(_catalogue())
        == "c978074a90392c1ac6d8e1b92ba13762909bdffa192ae6431148147f731e86bf"
    )


def test_setting_a_sections_context_changes_the_digest(storage: PostgresStorage) -> None:
    memory = Memory(storage)
    assert memory._digest_of(_catalogue("ACME invoice")) != memory._digest_of(_catalogue())
    assert memory._digest_of(_catalogue("ACME invoice")) != memory._digest_of(_catalogue("Other"))
