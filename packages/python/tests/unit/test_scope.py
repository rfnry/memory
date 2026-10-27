from datetime import UTC, datetime

import pytest

from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.events import Events
from rfnry_memory.memory import Memory, _window
from rfnry_memory.models import EntityIn, RelationIn, Section, SourceIn
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.rows import name_key
from rfnry_memory.tools import MemoryTools

from .fakes import HashEmbedding

APOLLO = EntityIn(name="Apollo", type="project")
BOREALIS = EntityIn(name="Borealis", type="project")
CYGNUS = EntityIn(name="Cygnus", type="project")
DRACO = EntityIn(name="Draco", type="project")


async def _memory(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    return memory


async def _seen(
    memory: Memory,
    *,
    corpus_id: str | list[str] = "acme",
    sources: list[str] | None = None,
    around: list[str] | None = None,
    hops: int = 1,
    relation_types: list[str] | None = None,
) -> set[str]:
    hits = await memory.search(
        corpus_id,
        "status",
        pillars={"vector"},
        top_k=50,
        sources=sources,
        around=around,
        hops=hops,
        relation_types=relation_types,
    )
    return {h.text for h in hits.hits}


async def _chain(memory: Memory) -> None:
    feeds = RelationIn(from_name="Apollo", to_name="Borealis", type="FEEDS")
    owns = RelationIn(from_name="Borealis", to_name="Cygnus", type="OWNS")
    for text, entities, relations in (
        ("Apollo is ready.", [APOLLO], []),
        ("Apollo feeds Borealis.", [APOLLO, BOREALIS], [feeds]),
        ("Borealis is ready.", [BOREALIS], []),
        ("Borealis owns Cygnus.", [BOREALIS, CYGNUS], [owns]),
        ("Cygnus is ready.", [CYGNUS], []),
        ("Draco is ready.", [DRACO], []),
    ):
        await memory.remember("acme", text, entities=entities, relations=relations, related=0)


async def test_sources_keep_a_search_to_their_chunks_and_the_stories_citing_them(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    manual = (
        await memory.ingest("acme", name="manual", text="The pump seal is replaced yearly.")
    ).source
    await memory.ingest("acme", name="notes", text="The pump seal leaked in March.")
    await memory.remember(
        "acme", "The pump seal was ordered from the manual.", source_ids=(manual.source_id,)
    )
    await memory.remember("acme", "The pump seal arrived.")
    hits = await memory.search("acme", "pump seal", sources=[manual.source_id])
    assert {(h.kind, h.text) for h in hits.hits} == {
        ("chunk", "The pump seal is replaced yearly."),
        ("story", "The pump seal was ordered from the manual."),
    }
    recalled = await memory.recall("acme", "pump seal", sources=[manual.source_id])
    assert [h.text for h in recalled.hits] == ["The pump seal was ordered from the manual."]
    for nothing in (["ghost"], []):
        assert (await memory.search("acme", "pump seal", sources=nothing)).hits == []


async def test_top_k_comes_from_inside_the_scope(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    await memory.ingest(
        "acme", name="big", sections=[Section(text="gasket gasket gasket") for _ in range(8)]
    )
    small = (
        await memory.ingest(
            "acme",
            name="small",
            sections=[Section(text=f"a gasket for the ring {i}") for i in range(4)],
        )
    ).source
    hits = await memory.search("acme", "gasket", sources=[small.source_id], top_k=2)
    assert len(hits.hits) == 2
    assert all(h.chunk is not None and h.chunk.source_id == small.source_id for h in hits.hits)


async def test_keyword_search_inside_a_scope_ranks_as_a_corpus_holding_only_the_scope(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage)
    keep = [Section(text=t) for t in ("gasket pump", "pump motor", "gasket seal ring")]
    noise = [Section(text=t) for t in ("gasket gasket", "gasket seal", "pump pump pump")]
    kept = (await memory.ingest("a", name="keep", sections=keep)).source
    await memory.ingest("a", name="noise", sections=noise)
    await memory.ingest("b", name="keep", sections=keep)

    async def ranked(corpus: str, sources: list[str] | None) -> list[tuple[str, float]]:
        hits = await memory.search(corpus, "gasket pump", sources=sources)
        return [(h.text, h.provenance["text"]) for h in hits.hits]

    scoped, alone = await ranked("a", [kept.source_id]), await ranked("b", None)
    assert [t for t, _ in scoped] == [t for t, _ in alone]
    assert [s for _, s in scoped] == pytest.approx([s for _, s in alone])
    assert [s for _, s in await ranked("a", None)][:3] != pytest.approx([s for _, s in alone])


async def test_around_reaches_zero_one_and_two_hops_along_the_relations_given(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await _chain(memory)
    zero = {"Apollo is ready.", "Apollo feeds Borealis."}
    one = zero | {"Borealis is ready.", "Borealis owns Cygnus."}
    assert await _seen(memory, around=["Apollo"], hops=0) == zero
    assert await _seen(memory, around=["Apollo"], hops=1) == one
    assert await _seen(memory, around=["Apollo"], hops=2) == one | {"Cygnus is ready."}
    assert await _seen(memory, around=["Apollo"], hops=2, relation_types=["feeds"]) == one


async def test_sources_and_around_together_keep_what_both_keep(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    manual = (
        await memory.ingest("acme", name="manual", text="Apollo feeds the line.", entities=[APOLLO])
    ).source
    await memory.ingest("acme", name="notes", text="Apollo stalled at noon.", entities=[APOLLO])
    await memory.ingest("acme", name="other", text="Draco runs the line.", entities=[DRACO])
    assert await _seen(memory, around=["Apollo"], hops=0) == {
        "Apollo feeds the line.",
        "Apollo stalled at noon.",
    }
    both = await _seen(memory, around=["Apollo"], hops=0, sources=[manual.source_id])
    assert both == {"Apollo feeds the line."}


async def test_graph_search_walks_from_around_and_stays_inside_the_scope(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await _chain(memory)
    walked = await memory.search(
        "acme", "status", pillars={"graph"}, around=["Borealis"], hops=0, kinds=("story",)
    )
    assert {h.text for h in walked.hits} == {
        "Apollo feeds Borealis.",
        "Borealis is ready.",
        "Borealis owns Cygnus.",
    }
    seeded = await memory.search(
        "acme",
        "status",
        pillars={"graph"},
        around=["Borealis"],
        hops=0,
        entities=["Apollo"],
        kinds=("story",),
    )
    assert seeded.hits[0].text == "Apollo feeds Borealis."
    assert {h.text for h in seeded.hits} <= {h.text for h in walked.hits}


async def test_an_empty_scope_finds_nothing_and_never_widens(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    for corpus in ("a", "b"):
        await memory.ingest(corpus, name="sheet", text=f"The {corpus} line status.")
    await memory.ingest("a", name="named", text="Apollo status.", entities=[APOLLO])
    nobody = await memory.search(["a", "b"], "status", around=["Nobody"])
    assert nobody.hits == [] and nobody.unresolved == {"Nobody": []}
    assert set(nobody.pillars) == {"text", "vector", "graph"}
    assert all(report.error is None for report in nobody.pillars.values())
    one = await memory.search(["a", "b"], "status", around=["Apollo"], hops=0)
    assert [(h.corpus_id, h.text) for h in one.hits] == [("a", "Apollo status.")]


async def test_sources_reach_across_several_corpora(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    held = {
        corpus: (
            await memory.ingest(corpus, name="sheet", text=f"The {corpus} line status.")
        ).source
        for corpus in ("a", "b")
    }
    both = await _seen(
        memory, corpus_id=["a", "b"], sources=[held["a"].source_id, held["b"].source_id]
    )
    assert both == {"The a line status.", "The b line status."}
    one = await _seen(memory, corpus_id=["a", "b"], sources=[held["b"].source_id])
    assert one == {"The b line status."}


async def test_around_needs_the_graph_pillar_and_hops_run_from_zero_to_two(
    storage: PostgresStorage,
) -> None:
    plain = Memory(storage, embedding=HashEmbedding())
    await plain.ingest("acme", name="sheet", text="The line status.")
    with pytest.raises(ConfigurationError, match=r"^memory: around needs the graph pillar$"):
        await plain.search("acme", "status", around=["Apollo"])
    with pytest.raises(ConfigurationError, match=r"^memory: around needs the graph pillar$"):
        await plain.recall("acme", "status", around=["Apollo"])
    found = await plain.search("acme", "status", sources=["ghost"])
    assert found.hits == [] and found.pillars["text"].error is None
    graphed = await _memory(storage)
    for hops in (-1, True, 1.0):
        with pytest.raises(InputError, match=r"^memory: hops is a whole number, 0 or more$"):
            await graphed.search("acme", "status", around=["Apollo"], hops=hops)  # type: ignore[arg-type]
    assert (await graphed.search("acme", "status", around=["Apollo"], hops=3)).hits == []


async def test_the_search_tools_scope_by_sources_and_around(storage: PostgresStorage) -> None:
    plain = Memory(storage, embedding=HashEmbedding())
    graphed = await _memory(storage)
    await _chain(graphed)

    def properties(tools: MemoryTools, name: str) -> set[str]:
        definition = next(d for d in tools.definitions() if d["name"] == name)
        return set(definition["input_schema"]["properties"])

    bound = MemoryTools(graphed, corpus_id="acme", filters={"metadata.line": "A"})
    for name in ("search", "recall"):
        assert {"sources", "around", "hops"} <= properties(
            MemoryTools(graphed, corpus_id="acme"), name
        )
        assert "sources" in properties(MemoryTools(plain, corpus_id="acme"), name)
        assert not {"around", "hops"} & properties(MemoryTools(plain, corpus_id="acme"), name)
        assert "sources" in properties(bound, name)
        assert not {"around", "hops"} & properties(bound, name)
        with pytest.raises(InputError, match=r"^tools: around is not offered here$"):
            await bound.dispatch(name, {"query": "status", "around": ["Apollo"]})
    tools = MemoryTools(graphed, corpus_id="acme")
    found = await tools.dispatch("recall", {"query": "ready", "around": ["Cygnus"], "hops": 0})
    assert isinstance(found, dict)
    assert {h["text"] for h in found["hits"]} == {"Borealis owns Cygnus.", "Cygnus is ready."}
    with pytest.raises(InputError, match=r"^tools: invalid arguments for search: "):
        await tools.dispatch("search", {"query": "status", "around": ["Apollo"], "hops": 3})


async def test_a_search_given_seeds_keeps_them_beside_the_entities_it_names(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await _chain(memory)
    [apollo] = (await storage.entities_named("acme", ["Apollo"]))[name_key("Apollo")]
    found = await memory._search(
        "acme",
        "status",
        kinds=("story",),
        pillars={"graph"},
        top_k=50,
        filters=None,
        window=_window(None, None, None, False, now=datetime.now(UTC)),
        subject=None,
        judge=(),
        seeds=[apollo.entity_id],
        entities=["Draco"],
        events=Events(None),
    )
    texts = {h.text for h in found.hits}
    assert {"Apollo is ready.", "Draco is ready."} <= texts


async def test_hops_are_checked_without_around_and_an_empty_around_finds_nothing(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.ingest("acme", name="named", text="Apollo status.", entities=[APOLLO])
    for hops in (-1, True, 1.0):
        with pytest.raises(InputError, match=r"^memory: hops is a whole number, 0 or more$"):
            await memory.search("acme", "status", hops=hops)  # type: ignore[arg-type]
        with pytest.raises(InputError, match=r"^memory: hops is a whole number, 0 or more$"):
            await memory.neighbors("acme", "Apollo", hops=hops)  # type: ignore[arg-type]
    empty = await memory.search("acme", "status", around=[])
    assert empty.hits == [] and empty.unresolved == {}


async def test_filters_the_window_and_the_subject_apply_inside_a_scope(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.set_history("acme", True)
    a = (
        await memory.ingest("acme", name="a", text="Line A status.", metadata={"line": "A"})
    ).source
    b = (
        await memory.ingest("acme", name="b", text="Line B status.", metadata={"line": "B"})
    ).source
    both = [a.source_id, b.source_id]
    lined = await memory.search(
        "acme", "status", pillars={"vector"}, sources=both, filters={"metadata.line": "A"}
    )
    assert {h.text for h in lined.hits} == {"Line A status."}
    for who in ("ana", "bo"):
        await memory.remember(
            "acme",
            f"{who.title()} checked the status.",
            subject=f"user:{who}",
            source_ids=(a.source_id,),
            related=0,
        )
    ana = await memory.recall(
        "acme", "status", pillars={"vector"}, sources=both, subject="user:ana"
    )
    assert [h.text for h in ana.hits] == ["Ana checked the status."]
    doc = {"metadata.doc": "c"}
    first = await memory.sync(
        "acme",
        filters=doc,
        sources=[SourceIn(name="c", text="Old c status.", metadata={"doc": "c"})],
    )
    await memory.sync(
        "acme",
        filters=doc,
        sources=[SourceIn(name="c", text="New c status.", metadata={"doc": "c"})],
    )
    old = [first.sources[0].source_id]
    assert (await memory.search("acme", "status", pillars={"vector"}, sources=old)).hits == []
    past = await memory.search("acme", "status", pillars={"vector"}, sources=old, history=True)
    assert [h.text for h in past.hits] == ["Old c status."]


async def test_with_the_graph_standing_down_names_still_come_back_unresolved(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.remember(
        "acme", "Apollo status is green.", entities=[APOLLO], metadata={"team": "ops"}, related=0
    )
    filtered = await memory.search(
        "acme", "status", entities=["Apollo", "Nobody"], filters={"metadata.team": "ops"}
    )
    assert "graph" not in filtered.pillars and filtered.unresolved == {"Nobody": []}
    texted = await memory.search("acme", "status", entities=["Nobody"], pillars={"text"})
    assert set(texted.pillars) == {"text"} and texted.unresolved == {"Nobody": []}


async def test_relation_types_hold_past_the_first_hop_and_from_spotted_names(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await _chain(memory)

    async def walked(query: str, **kwargs: object) -> set[str]:
        hits = await memory.search(
            "acme",
            query,
            pillars={"graph"},
            kinds=("story",),
            top_k=50,
            **kwargs,  # type: ignore[arg-type]
        )
        return {h.text for h in hits.hits}

    both = await walked("status", entities=["Apollo"], relation_types=["FEEDS", "OWNS"])
    fed = await walked("status", entities=["Apollo"], relation_types=["FEEDS"])
    assert "Cygnus is ready." in both and "Cygnus is ready." not in fed
    assert "Borealis is ready." in fed
    spotted = await walked("Apollo status", relation_types=["OWNS"])
    assert "Apollo is ready." in spotted and "Borealis is ready." not in spotted


async def test_a_supersede_finds_its_related_stories_through_the_graph(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"graph"})
    held = await memory.remember("acme", "Apollo is ready.", entities=[APOLLO], related=0)
    await memory.remember("acme", "Apollo feeds the line.", entities=[APOLLO], related=0)
    await memory.remember("acme", "Draco is ready.", entities=[DRACO], related=0)
    moved = await memory.supersede(
        "acme", held.story.story_id, "Apollo is paused.", entities=[APOLLO], related=5
    )
    assert [h.text for h in moved.related.hits] == ["Apollo feeds the line."]
