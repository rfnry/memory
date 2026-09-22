from datetime import UTC, datetime

import pytest

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import EntityIn, RelationIn, Section, Window
from rfnry_memory.storage.inmemory import InMemoryStorage

from .fakes import HashEmbedding

MANUAL = "# Pump manual\n\n## Seals\n\n" + "Replace the seal yearly. " * 30 + "\n\n## Wiring\n\nGround first."


async def _memory(**kwargs: object) -> Memory:
    memory = Memory(InMemoryStorage(), embedding=HashEmbedding(), chunk_chars=300, overlap_chars=30, **kwargs)  # type: ignore[arg-type]
    await memory.initialize()
    return memory


async def test_ingest_text_chunks_embeds_and_stores(embedding: HashEmbedding) -> None:
    memory = Memory(InMemoryStorage(), embedding=embedding, chunk_chars=300, overlap_chars=30)
    await memory.initialize()
    ingested = await memory.ingest("acme", text=MANUAL, name="manual", metadata={"line": "A"})
    assert ingested.source.name == "manual" and ingested.chunks > 3 and ingested.source.chunks == ingested.chunks
    assert len(embedding.calls) == 1 and len(embedding.calls[0]) == ingested.chunks
    chunks = await memory.storage.source_chunks("acme", ingested.source.source_id)
    assert chunks[1].title == "Pump manual > Seals" and chunks[-1].title == "Pump manual > Wiring"
    sources, _ = await memory.list_sources("acme", filters={"metadata.line": "A"})
    assert [s.source_id for s in sources] == [ingested.source.source_id]
    assert await memory.read("acme", ingested.source.source_id) == MANUAL.strip()


async def test_ingest_sections_keep_title_and_locator() -> None:
    memory = await _memory()
    ingested = await memory.ingest("acme", name="sheet 7", sections=[
        Section(title="Feed line", text="Feed line text", locator={"sheet": 7, "region": "A2"}),
        Section(title="Return line", text="Return line text", locator={"sheet": 7, "region": "C4"}),
    ])
    chunks = await memory.storage.source_chunks("acme", ingested.source.source_id)
    assert [(c.title, c.locator["region"]) for c in chunks] == [("Feed line", "A2"), ("Return line", "C4")]
    assert await memory.read("acme", ingested.source.source_id, locator={"region": "C4"}) == "Return line text"
    with pytest.raises(InputError):
        await memory.ingest("acme", name="x", text="a", sections=[Section(text="b")])
    with pytest.raises(InputError):
        await memory.ingest("acme", name="x")


async def test_remember_supersede_forget() -> None:
    memory = await _memory()
    jan, feb = datetime(2026, 1, 10, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    story = await memory.remember("acme", "The go-live is February 20.", subject="project:apollo", occurred_at=jan, metadata={"team": "core"})
    again = await memory.remember("acme", "the go-live is  february 20.", subject="project:apollo")
    assert again.story_id == story.story_id
    closed, current = await memory.supersede("acme", story.story_id, "Ana moved the go-live to March 3.", effective_at=feb)
    assert closed.valid_to == feb and current.occurred_at == feb and current.supersedes == story.story_id
    assert current.subject == "project:apollo" and current.metadata == {"team": "core"}
    with pytest.raises(InputError):
        await memory.supersede("acme", story.story_id, "again", effective_at=feb)
    with pytest.raises(InputError):
        await memory.supersede("acme", current.story_id, "too early", effective_at=jan)
    stories, _ = await memory.list_stories("acme", subject="project:apollo", as_of=datetime(2026, 3, 1, tzinfo=UTC))
    assert [s.story_id for s in stories] == [current.story_id]
    await memory.forget("acme", current.story_id)
    assert await memory.get_stories("acme", [current.story_id]) == []
    with pytest.raises(NotFoundError):
        await memory.forget("acme", "ghost")


async def test_supplied_graph_facts_are_stored_without_a_model() -> None:
    memory = await _memory(pillars={"text", "vector", "graph"})
    ingested = await memory.ingest("acme", name="sheet", text="V-101 feeds P-3",
        entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump")],
        relations=[RelationIn(from_name="V-101", to_name="P-3", type="CONNECTS_TO")])
    assert (ingested.entities, ingested.relations) == (2, 1)
    valve = await memory.storage.get_entity("acme", "v-101")
    assert valve is not None and valve.type == "valve"
    again = await memory.remember("acme", "V-101 was replaced", entities=[EntityIn(name="v-101", type="valve")])
    assert (await memory.storage.count_corpus("acme")).entities == 2
    assert await memory.storage.mentions("acme", [valve.entity_id], kinds=("chunk", "story"), filters=None, window=Window()) == [
        ("chunk", (await memory.storage.source_chunks("acme", ingested.source.source_id))[0].chunk_id), ("story", again.story_id)]
    with pytest.raises(InputError):
        await memory.remember("acme", "x", relations=[RelationIn(from_name="ghost", to_name="P-3", type="X")])


async def test_construction_rules() -> None:
    with pytest.raises(ConfigurationError):
        Memory(InMemoryStorage(), pillars={"vector"})
    with pytest.raises(ConfigurationError):
        Memory(InMemoryStorage(), pillars={"sonar"})
    memory = Memory(InMemoryStorage())
    assert memory.pillars == {"text"}
    with pytest.raises(ConfigurationError):
        await memory.ingest("acme", name="x", text="a", entities=[EntityIn(name="a", type="b")])
    with pytest.raises(InputError):
        await memory.ingest("", name="x", text="a")


async def test_reembed_after_a_model_change(embedding: HashEmbedding) -> None:
    memory = Memory(InMemoryStorage(), embedding=embedding)
    await memory.initialize()
    await memory.ingest("acme", name="a", text="alpha")
    await memory.remember("acme", "beta")
    embedding.model = "hash-8-v2"
    assert await memory.reembed("acme") == 2
    assert await memory.reembed("acme") == 0


async def test_delete_corpus_is_provable() -> None:
    memory = await _memory()
    await memory.ingest("acme", name="a", text="alpha")
    await memory.remember("acme", "beta")
    await memory.ingest("other", name="b", text="keep")
    assert not (await memory.verify_absent("acme")).clean
    deleted = await memory.delete_corpus("acme")
    assert deleted.sources == 1 and deleted.stories == 1
    assert (await memory.verify_absent("acme")).clean and not (await memory.verify_absent("other")).clean


async def test_writes_without_an_embedding_model() -> None:
    memory = Memory(InMemoryStorage())
    await memory.initialize()
    ingested = await memory.ingest("acme", name="manual", text=MANUAL)
    assert ingested.chunks > 0
    story = await memory.remember("acme", "The go-live is February 20.")
    closed, current = await memory.supersede("acme", story.story_id, "The go-live is March 3.")
    assert closed.valid_to is not None and current.supersedes == story.story_id
    with pytest.raises(ConfigurationError):
        await memory.reembed("acme")
