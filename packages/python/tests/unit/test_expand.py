from datetime import UTC, datetime
from typing import Any

import pytest

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import CheckAnswer, EntityIn, RelationIn, Section, SourceIn, Story
from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.tools import MemoryTools

from .fakes import Fixed, HashEmbedding

APOLLO = EntityIn(name="Apollo", type="project")
BOREALIS = EntityIn(name="Borealis", type="project")
CYGNUS = EntityIn(name="Cygnus", type="project")


async def _steps(memory: Memory, corpus: str = "acme", count: int = 6) -> list[str]:
    source = (
        await memory.ingest(
            corpus, name="manual", sections=[Section(text=f"Step {i}.") for i in range(count)]
        )
    ).source
    return [c.chunk_id for c in await memory.storage.source_chunks(corpus, source.source_id)]


async def test_expand_by_document_reads_the_neighbouring_chunks_in_order() -> None:
    memory = Memory(InMemoryStorage())
    chunks = await _steps(memory)

    async def around(position: int, width: int) -> list[int]:
        hits = await memory.expand("acme", ("chunk", chunks[position]), by="document", width=width)
        return [h.chunk.position for h in hits.hits if h.chunk is not None]

    assert await around(2, 2) == [0, 1, 3, 4]
    assert await around(0, 2) == [1, 2]
    assert await around(5, 1) == [4]
    narrow = await memory.expand("acme", ("chunk", chunks[2]), by="document", width=1, top_k=1)
    assert [h.provenance for h in narrow.hits] == [{"document": 1.0}, {"document": 1.0}]


async def test_expand_by_document_reads_a_storys_subject_in_time_order() -> None:
    memory = Memory(InMemoryStorage())
    told: dict[int, Story] = {}
    for month in (1, 2, 3, 4):
        remembered = await memory.remember(
            "acme",
            f"P-3 report {month}.",
            subject="asset:p-3",
            valid_from=datetime(2026, month, 1, tzinfo=UTC),
            related=0,
        )
        told[month] = remembered.story
    await memory.remember("acme", "P-9 report.", subject="asset:p-9", related=0)
    february = ("story", told[2].story_id)

    async def timeline(**options: Any) -> list[str]:
        hits = await memory.expand("acme", february, by="document", **options)
        return [h.text for h in hits.hits]

    assert await timeline() == ["P-3 report 1.", "P-3 report 3.", "P-3 report 4."]
    assert await timeline(top_k=2) == ["P-3 report 1.", "P-3 report 3."]
    assert await timeline(top_k=1) == ["P-3 report 3."]
    await memory.supersede(
        "acme",
        told[1].story_id,
        "P-3 report 1, corrected.",
        valid_from=datetime(2026, 1, 15, tzinfo=UTC),
        related=0,
    )
    assert await timeline() == ["P-3 report 1, corrected.", "P-3 report 3.", "P-3 report 4."]
    assert await timeline(history=True) == [
        "P-3 report 1.",
        "P-3 report 1, corrected.",
        "P-3 report 3.",
        "P-3 report 4.",
    ]
    bare = (await memory.remember("acme", "No subject here.", related=0)).story
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{bare.story_id}' has no subject; expand by document needs one$",
    ):
        await memory.expand("acme", ("story", bare.story_id), by="document")


async def test_expand_by_vector_leaves_out_the_start_and_calls_no_model() -> None:
    embedding = HashEmbedding()
    memory = Memory(InMemoryStorage(), embedding=embedding)
    await memory.initialize()
    first = (await memory.ingest("acme", name="a", text="The pump seal is replaced yearly.")).source
    twin = (await memory.ingest("acme", name="b", text="The pump seal is replaced yearly.")).source
    await memory.ingest("acme", name="c", text="Ground the panel first.")
    await memory.remember("acme", "Ana owns the wiring review.", related=0)
    start = ("chunk", f"{first.source_id}-0")
    embedding.calls.clear()
    one = await memory.expand("acme", start, by="vector", top_k=1)
    assert embedding.calls == []
    assert [h.id for h in one.hits] == [f"{twin.source_id}-0"]
    assert one.hits[0].provenance == {"vector": 1.0}
    every = await memory.expand("acme", start, by="vector")
    assert len(every.hits) == 3 and start[1] not in {h.id for h in every.hits}
    stories = await memory.expand("acme", start, by="vector", kinds=("story",))
    assert [h.kind for h in stories.hits] == ["story"]
    quiet = Story(
        corpus_id="acme",
        story_id="quiet",
        text="Written without a vector.",
        subject=None,
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_to=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_ids=(),
        metadata={},
    )
    await memory.storage.put_story("acme", quiet, None, None)
    assert (await memory.expand("acme", ("story", "quiet"), by="vector")).hits == []


async def test_expand_by_graph_seeds_from_the_entities_the_passage_mentions() -> None:
    memory = Memory(InMemoryStorage(), pillars={"graph"})
    feeds = RelationIn(from_name="Apollo", to_name="Borealis", type="FEEDS")
    owns = RelationIn(from_name="Borealis", to_name="Cygnus", type="OWNS")
    told: dict[str, Story] = {}
    for text, entities, relations in (
        ("Apollo is ready.", [APOLLO], []),
        ("Apollo feeds Borealis.", [APOLLO, BOREALIS], [feeds]),
        ("Borealis owns Cygnus.", [BOREALIS, CYGNUS], [owns]),
        ("Cygnus is ready.", [CYGNUS], []),
        ("Nothing named here.", [], []),
    ):
        remembered = await memory.remember(
            "acme", text, entities=entities, relations=relations, related=0
        )
        told[text] = remembered.story
    start = ("story", told["Apollo is ready."].story_id)
    walked = await memory.expand("acme", start, by="graph")
    assert [h.text for h in walked.hits] == [
        "Apollo feeds Borealis.",
        "Borealis owns Cygnus.",
        "Cygnus is ready.",
    ]
    assert walked.hits[0].provenance == {"graph": 1.0}
    fed = await memory.expand("acme", start, by="graph", relation_types=["feeds"])
    assert [h.text for h in fed.hits] == ["Apollo feeds Borealis.", "Borealis owns Cygnus."]
    unnamed = ("story", told["Nothing named here."].story_id)
    assert (await memory.expand("acme", unnamed, by="graph")).hits == []


async def test_expand_judges_hits_against_the_starting_passage() -> None:
    decided = Fixed(lambda state, questions: {n: CheckAnswer(probability=0.9) for n in questions})
    memory = Memory(InMemoryStorage(), decision=decided)
    chunks = await _steps(memory, count=3)
    hits = await memory.expand("acme", ("chunk", chunks[1]), by="document", judge=("relevant",))
    assert [h.judge.relevant for h in hits.hits if h.judge is not None] == [0.9, 0.9]
    assert len(decided.calls) == 2
    assert {s["query"] for s, _ in decided.calls if isinstance(s, dict)} == {"Step 1."}
    with pytest.raises(ConfigurationError, match=r"^memory: judge=True needs a decision model$"):
        await Memory(InMemoryStorage()).expand("acme", ("chunk", "x"), by="document", judge=True)


async def test_expand_holds_filters_and_the_window() -> None:
    memory = Memory(InMemoryStorage(), embedding=HashEmbedding())
    await memory.initialize()
    await memory.set_history("acme", True)

    def manual(word: str) -> SourceIn:
        return SourceIn(
            name="manual",
            sections=[Section(text=f"{word} step {i}.") for i in range(3)],
            metadata={"doc": "manual", "line": "A"},
        )

    old = await memory.sync("acme", filters={"metadata.doc": "manual"}, sources=[manual("Old")])
    await memory.sync("acme", filters={"metadata.doc": "manual"}, sources=[manual("New")])
    start = ("chunk", f"{old.sources[0].source_id}-0")
    assert (await memory.expand("acme", start, by="document")).hits == []
    past = await memory.expand("acme", start, by="document", history=True)
    assert [h.text for h in past.hits] == ["Old step 1.", "Old step 2."]
    lined = await memory.expand(
        "acme", start, by="document", history=True, filters={"metadata.line": "B"}
    )
    assert lined.hits == []
    await memory.ingest("acme", name="other", text="A line B note.", metadata={"line": "B"})
    near = await memory.expand("acme", start, by="vector", filters={"metadata.line": "B"})
    assert [h.text for h in near.hits] == ["A line B note."]


async def test_expand_refuses_what_it_cannot_do() -> None:
    memory = Memory(InMemoryStorage())
    [chunk] = await _steps(memory, count=1)
    start = ("chunk", chunk)
    with pytest.raises(
        InputError, match=r"^memory: by is document, vector or graph, not 'sideways'$"
    ):
        await memory.expand("acme", start, by="sideways")  # type: ignore[arg-type]
    with pytest.raises(
        InputError, match=r"^memory: passage is a \(kind, id\) pair, its kind chunk or story$"
    ):
        await memory.expand("acme", ("source", chunk), by="document")
    with pytest.raises(InputError, match=r"^memory: top_k must be at least 1$"):
        await memory.expand("acme", start, by="document", top_k=0)
    with pytest.raises(InputError, match=r"^memory: width must be at least 1$"):
        await memory.expand("acme", start, by="document", width=0)
    with pytest.raises(
        ConfigurationError, match=r"^memory: expand by vector needs the vector pillar$"
    ):
        await memory.expand("acme", start, by="vector")
    with pytest.raises(
        ConfigurationError, match=r"^memory: expand by graph needs the graph pillar$"
    ):
        await memory.expand("acme", start, by="graph")
    with pytest.raises(NotFoundError, match=r"^memory: chunk 'ghost' was not found$"):
        await memory.expand("acme", ("chunk", "ghost"), by="document")
    with pytest.raises(NotFoundError, match=rf"^memory: chunk '{chunk}' was not found$"):
        await memory.expand("other", start, by="document")


def _expand(tools: MemoryTools) -> dict[str, Any]:
    return next(d for d in tools.definitions() if d["name"] == "expand")


async def test_the_expand_tool_offers_only_the_modes_that_work() -> None:
    plain = Memory(InMemoryStorage())
    embedded = Memory(InMemoryStorage(), embedding=HashEmbedding())
    graphed = Memory(
        InMemoryStorage(), embedding=HashEmbedding(), pillars={"text", "vector", "graph"}
    )
    for memory, modes in (
        (plain, ["document"]),
        (embedded, ["document", "vector"]),
        (graphed, ["document", "vector", "graph"]),
    ):
        definition = _expand(MemoryTools(memory, corpus_id="acme"))
        properties = definition["input_schema"]["properties"]
        assert properties["by"]["enum"] == modes
        assert [
            m for m in ("document", "vector", "graph") if f"by {m}" in definition["description"]
        ] == modes
        assert ("relation_types" in properties) == ("graph" in modes)
    bound = MemoryTools(graphed, corpus_id="acme", filters={"metadata.line": "A"})
    assert _expand(bound)["input_schema"]["properties"]["by"]["enum"] == ["document", "vector"]
    with pytest.raises(InputError, match=r"^tools: by is one of document, vector$"):
        await bound.dispatch("expand", {"kind": "chunk", "id": "x", "by": "graph"})


async def test_the_expand_tool_keeps_to_the_bound_view() -> None:
    memory = Memory(InMemoryStorage(), embedding=HashEmbedding())
    await memory.initialize()

    async def lined(line: str) -> str:
        sections = [Section(text=f"Line {line} step {i}.") for i in range(2)]
        ingested = await memory.ingest(
            "acme", name=line, sections=sections, metadata={"line": line}
        )
        return ingested.source.source_id

    inside, outside = await lined("A"), await lined("B")
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
    for missing in (f"{outside}-0", "ghost"):
        with pytest.raises(NotFoundError, match=rf"^tools: chunk '{missing}' was not found$"):
            await tools.dispatch("expand", {"kind": "chunk", "id": missing, "by": "document"})
    near = await tools.dispatch("expand", {"kind": "chunk", "id": f"{inside}-0", "by": "vector"})
    assert isinstance(near, dict) and [h["text"] for h in near["hits"]] == ["Line A step 1."]
    beside = await tools.dispatch(
        "expand", {"kind": "chunk", "id": f"{inside}-0", "by": "document"}
    )
    assert isinstance(beside, dict) and [h["id"] for h in beside["hits"]] == [f"{inside}-1"]


async def test_the_expand_tool_takes_the_corpus_a_hit_names() -> None:
    memory = Memory(InMemoryStorage())
    held = {corpus: await _steps(memory, corpus, count=2) for corpus in ("hr", "ops")}
    tools = MemoryTools(memory, corpus_id=["hr", "ops"])
    schema = _expand(tools)["input_schema"]
    assert schema["properties"]["corpus_id"]["enum"] == ["hr", "ops"]
    assert "corpus_id" in schema["required"]
    start = {"kind": "chunk", "id": held["ops"][0], "by": "document"}
    found = await tools.dispatch("expand", {**start, "corpus_id": "ops"})
    assert isinstance(found, dict)
    assert [(h["corpus_id"], h["text"]) for h in found["hits"]] == [("ops", "Step 1.")]
    with pytest.raises(NotFoundError):
        await tools.dispatch("expand", {**start, "corpus_id": "hr"})
