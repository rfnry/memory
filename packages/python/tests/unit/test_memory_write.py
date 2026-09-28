import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from rfnry_memory.errors import ConfigurationError, InputError, ModelError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    Answer,
    Decided,
    EntityIn,
    Message,
    QuestionLike,
    RelationIn,
    Reply,
    ScoreAnswer,
    Section,
    SourceIn,
    State,
    ToolSpec,
    Window,
)
from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.storage.storage import Storage

from .fakes import Fixed, HashEmbedding, Scripted, structured_reply

MANUAL = (
    "# Pump manual\n\n## Seals\n\n"
    + "Replace the seal yearly. " * 30
    + "\n\n## Wiring\n\nGround first."
)


async def _memory(**kwargs: object) -> Memory:
    memory = Memory(
        InMemoryStorage(),
        embedding=HashEmbedding(),
        chunk_chars=300,
        overlap_chars=30,
        **kwargs,  # type: ignore[arg-type]
    )
    await memory.initialize()
    return memory


async def test_ingest_text_chunks_embeds_and_stores(embedding: HashEmbedding) -> None:
    memory = Memory(InMemoryStorage(), embedding=embedding, chunk_chars=300, overlap_chars=30)
    await memory.initialize()
    ingested = await memory.ingest("acme", text=MANUAL, name="manual", metadata={"line": "A"})
    assert ingested.source.name == "manual" and ingested.source.chunks > 3
    assert len(embedding.calls) == 1 and len(embedding.calls[0]) == ingested.source.chunks
    chunks = await memory.storage.source_chunks("acme", ingested.source.source_id)
    assert chunks[1].title == "Pump manual > Seals" and chunks[-1].title == "Pump manual > Wiring"
    sources, _ = await memory.list_sources("acme", filters={"metadata.line": "A"})
    assert [s.source_id for s in sources] == [ingested.source.source_id]
    assert await memory.read_source("acme", ingested.source.source_id) == MANUAL.strip()


async def test_ingest_sections_keep_title_and_locator() -> None:
    memory = await _memory()
    ingested = await memory.ingest(
        "acme",
        name="sheet 7",
        sections=[
            Section(title="Feed line", text="Feed line text", locator={"sheet": 7, "region": "A2"}),
            Section(
                title="Return line", text="Return line text", locator={"sheet": 7, "region": "C4"}
            ),
        ],
    )
    chunks = await memory.storage.source_chunks("acme", ingested.source.source_id)
    assert [(c.title, c.locator["region"]) for c in chunks] == [
        ("Feed line", "A2"),
        ("Return line", "C4"),
    ]
    assert (
        await memory.read_source("acme", ingested.source.source_id, locator={"region": "C4"})
        == "Return line text"
    )
    with pytest.raises(InputError):
        await memory.ingest("acme", name="x", text="a", sections=[Section(text="b")])
    with pytest.raises(InputError):
        await memory.ingest("acme", name="x")


async def test_remember_supersede_delete_story() -> None:
    memory = await _memory()
    jan, feb = datetime(2026, 1, 10, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    story = (
        await memory.remember(
            "acme",
            "The go-live is February 20.",
            subject="project:apollo",
            valid_from=jan,
            metadata={"team": "core"},
        )
    ).story
    again = (
        await memory.remember("acme", "the go-live is  february 20.", subject="project:apollo")
    ).story
    assert again.story_id == story.story_id
    current = (
        await memory.supersede(
            "acme", story.story_id, "Ana moved the go-live to March 3.", valid_from=feb
        )
    ).story
    [closed] = await memory.get_stories("acme", [story.story_id])
    assert (
        closed.valid_to == feb
        and current.valid_from == feb
        and current.supersedes == story.story_id
    )
    assert current.subject == "project:apollo" and current.metadata == {"team": "core"}
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{story.story_id}' is already superseded; supersede the story that replaced it$",
    ):
        await memory.supersede("acme", story.story_id, "again", valid_from=feb)
    with pytest.raises(InputError):
        await memory.supersede("acme", current.story_id, "too early", valid_from=jan)
    stories, _ = await memory.list_stories(
        "acme", subject="project:apollo", as_of=datetime(2026, 3, 1, tzinfo=UTC)
    )
    assert [s.story_id for s in stories] == [current.story_id]
    deleted = await memory.delete_story("acme", current.story_id)
    assert (deleted.stories, deleted.chunks) == (1, 0) and not deleted.clean
    assert await memory.get_stories("acme", [current.story_id]) == []
    with pytest.raises(NotFoundError):
        await memory.delete_story("acme", "ghost")


async def test_supplied_graph_facts_are_stored_without_a_model() -> None:
    memory = await _memory(pillars={"text", "vector", "graph"})
    ingested = await memory.ingest(
        "acme",
        name="sheet",
        text="V-101 feeds P-3",
        entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump")],
        relations=[RelationIn(from_name="V-101", to_name="P-3", type="CONNECTS_TO")],
    )
    residue = await memory.storage.count_corpus("acme")
    assert (residue.entities, residue.relations) == (2, 1)
    [valve] = await memory.storage.entities_named("acme", "v-101")
    assert valve.type == "valve"
    again = (
        await memory.remember(
            "acme", "V-101 was replaced", entities=[EntityIn(name="v-101", type="valve")]
        )
    ).story
    assert (await memory.storage.count_corpus("acme")).entities == 2
    assert await memory.storage.mentions(
        "acme",
        [valve.entity_id],
        kinds=("chunk", "story"),
        filters=None,
        window=Window(),
        subject=None,
    ) == [
        (
            "chunk",
            (await memory.storage.source_chunks("acme", ingested.source.source_id))[0].chunk_id,
        ),
        ("story", again.story_id),
    ]
    with pytest.raises(InputError):
        await memory.remember(
            "acme", "x", relations=[RelationIn(from_name="ghost", to_name="P-3", type="X")]
        )


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
    assert (await memory.verify_absent("acme")).clean and not (
        await memory.verify_absent("other")
    ).clean


async def test_writes_without_an_embedding_model() -> None:
    memory = Memory(InMemoryStorage())
    await memory.initialize()
    ingested = await memory.ingest("acme", name="manual", text=MANUAL)
    assert ingested.source.chunks > 0
    story = (await memory.remember("acme", "The go-live is February 20.")).story
    current = (await memory.supersede("acme", story.story_id, "The go-live is March 3.")).story
    [closed] = await memory.get_stories("acme", [story.story_id])
    assert closed.valid_to == current.valid_from and current.supersedes == story.story_id
    with pytest.raises(ConfigurationError):
        await memory.reembed("acme")


async def test_supersede_into_text_another_open_story_holds_is_refused() -> None:
    memory = await _memory()
    jan, feb = datetime(2026, 1, 10, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    held = (
        await memory.remember("acme", "The go-live is March 3.", subject="project:apollo")
    ).story
    story = (
        await memory.remember(
            "acme", "The go-live is February 20.", subject="project:apollo", valid_from=jan
        )
    ).story
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{held.story_id}', true since {re.escape(held.valid_from.isoformat())}, already says that; "
        r"supersede or delete it instead$",
    ):
        await memory.supersede("acme", story.story_id, " the go-live is march 3. ", valid_from=feb)
    assert (await memory.get_stories("acme", [story.story_id]))[0].valid_to is None
    current = (await memory.supersede("acme", held.story_id, "The go-live is March 3.")).story
    assert (
        current.supersedes == held.story_id
        and (await memory.get_stories("acme", [held.story_id]))[0].valid_to is not None
    )


WRITES = {"put_source", "replace_sources", "put_story", "supersede_story"}


class _Recorder:
    def __init__(self, inner: InMemoryStorage, log: list[str]) -> None:
        self._inner = inner
        self._log = log

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._inner, name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            self._log.append(f"storage.{name}")
            return await method(*args, **kwargs)

        return call


class _LoggedEmbedding(HashEmbedding):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self._log = log

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._log.append("model.embed")
        return await super().embed(texts)


class _LoggedGeneration(Scripted):
    def __init__(self, log: list[str], *replies: Reply) -> None:
        super().__init__(*replies)
        self._log = log

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self._log.append("model.generate")
        return await super().generate(system, messages, tools)


class _LoggedDecision(Fixed):
    def __init__(self, log: list[str], answers: dict[str, Answer]) -> None:
        super().__init__(answers)
        self._log = log

    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> dict[str, Answer] | Decided:
        self._log.append("model.decide")
        return await super().decide(state, questions)


async def test_every_model_call_happens_before_the_one_storage_write() -> None:
    log: list[str] = []
    extracted = {
        "entities": [{"name": "Acme Corp", "type": "company"}, {"name": "V-101", "type": "valve"}],
        "relations": [
            {
                "from_name": "Acme Corp",
                "from_type": "company",
                "to_name": "V-101",
                "to_type": "valve",
                "type": "OWNS",
            }
        ],
    }
    seed = {"entities": [{"name": "ACME Corporation", "type": "company"}], "relations": []}
    generation = _LoggedGeneration(
        log, structured_reply(seed), *(structured_reply(extracted) for _ in range(16))
    )
    decision = _LoggedDecision(
        log, {"same": ScoreAnswer(score=0.1, probabilities=[0.9, 0.05, 0.05])}
    )
    storage = InMemoryStorage()
    memory = Memory(
        cast(Storage, _Recorder(storage, log)),
        embedding=_LoggedEmbedding(log),
        generation=generation,
        decision=decision,
        relation_types=["OWNS"],
        chunk_chars=300,
        overlap_chars=30,
    )
    await memory.initialize()
    await memory.remember("acme", "ACME Corporation owns valve V-9.")

    for verb in ("ingest", "remember", "supersede", "replace"):
        log.clear()
        if verb == "ingest":
            await memory.ingest("acme", name="manual", text=MANUAL, metadata={"file": "m"})
        elif verb == "remember":
            story = (await memory.remember("acme", "Acme Corp owns V-101.")).story
        elif verb == "supersede":
            await memory.supersede("acme", story.story_id, "Acme Corp sold V-101.")
        else:
            await memory.replace(
                "acme",
                filters={"metadata.file": "m"},
                sources=[
                    SourceIn(name="manual", text=MANUAL, metadata={"file": "m"}),
                    SourceIn(name="notes", text="Acme Corp owns V-101.", metadata={"file": "m"}),
                ],
            )
        writes = [i for i, event in enumerate(log) if event.removeprefix("storage.") in WRITES]
        assert len(writes) == 1 and writes[0] == len(log) - 1, (verb, log)
        assert {"model.embed", "model.generate"} | (
            {"model.decide"} if verb == "ingest" else set()
        ) <= set(log), (verb, log)


async def test_in_memory_storage_keeps_the_dimension_it_was_first_given() -> None:
    storage = InMemoryStorage()
    await storage.initialize(dimension=8)
    await storage.initialize(dimension=8)
    await storage.initialize(dimension=None)
    with pytest.raises(
        ConfigurationError,
        match=r"^inmemory: this storage was initialized for dimension 8; this memory embeds 1024 — use another storage$",
    ):
        await storage.initialize(dimension=1024)
    bare = InMemoryStorage()
    await bare.initialize(dimension=None)
    with pytest.raises(
        ConfigurationError,
        match=r"^inmemory: this storage was initialized without vectors; this memory embeds 8 — use another storage$",
    ):
        await Memory(bare, embedding=HashEmbedding()).initialize()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weights": {"sonar": 1.0}}, r"^memory: weights name unknown pillars: \['sonar'\]$"),
        ({"weights": {"text": 0.0}}, r"^memory: weights must be positive; got 'text' = 0.0$"),
        ({"weights": {"text": -1.0}}, r"^memory: weights must be positive; got 'text' = -1.0$"),
        ({"chunk_chars": 0}, r"^memory: chunk_chars must be at least 1$"),
        (
            {"chunk_chars": 100, "overlap_chars": 100},
            r"^memory: overlap_chars must be at least 0 and less than chunk_chars \(100\)$",
        ),
        (
            {"overlap_chars": -1},
            r"^memory: overlap_chars must be at least 0 and less than chunk_chars \(2000\)$",
        ),
    ],
)
def test_construction_refuses_settings_that_cannot_work(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        Memory(InMemoryStorage(), **kwargs)


def test_source_verbs_share_one_name() -> None:
    assert {"get_source", "list_sources", "read_source", "delete_source"} <= set(
        dir(Memory)
    ) and not hasattr(Memory, "read")


async def test_a_story_is_true_from_valid_from_which_defaults_to_the_write() -> None:
    memory = await _memory()
    jan = datetime(2026, 1, 10, tzinfo=UTC)
    told = (await memory.remember("acme", "The go-live is February 20.", valid_from=jan)).story
    assert (
        told.valid_from == jan
        and told.created_at > jan
        and not any("occurred" in field for field in type(told).model_fields)
    )
    now = (await memory.remember("acme", "Ana owns the review.")).story
    assert now.valid_from == now.created_at
    later = (await memory.supersede("acme", now.story_id, "Bo owns the review.")).story
    assert (
        later.valid_from
        == later.created_at
        == (await memory.get_stories("acme", [now.story_id]))[0].valid_to
    )
    with pytest.raises(
        InputError,
        match=rf"^memory: valid_from must be after story '{told.story_id}' became true, at "
        rf"{re.escape(jan.isoformat())}; pass a later valid_from$",
    ):
        await memory.supersede("acme", told.story_id, "The go-live is March 3.", valid_from=jan)


class _Short(HashEmbedding):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [v[:7] for v in await super().embed(texts)]


class _Fewer(HashEmbedding):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return (await super().embed(texts))[:-1]


@pytest.mark.parametrize("broken", [_Short(), _Fewer()])
async def test_a_write_refuses_what_an_embedding_model_returns_outside_its_contract(
    broken: HashEmbedding,
) -> None:
    memory = await _memory()
    story = (await memory.remember("acme", "The go-live is February 20.")).story
    await memory.ingest("acme", name="v0", text="version zero", metadata={"file": "a"})
    before = await memory.verify_absent("acme")
    wrong = Memory(memory.storage, embedding=broken)
    returned = r"^memory: the embedding model returned "
    with pytest.raises(ModelError, match=returned):
        await wrong.ingest("acme", name="manual", text="one\n\ntwo")
    with pytest.raises(ModelError, match=returned):
        await wrong.remember("acme", "Ana owns the review.")
    with pytest.raises(ModelError, match=returned):
        await wrong.supersede("acme", story.story_id, "The go-live is March 3.")
    with pytest.raises(ModelError, match=returned):
        await wrong.replace(
            "acme",
            filters={"metadata.file": "a"},
            sources=[SourceIn(name="v1", text="version one", metadata={"file": "a"})],
        )
    broken.model = "hash-8-v2"
    with pytest.raises(ModelError, match=returned):
        await wrong.reembed("acme")
    assert await memory.verify_absent("acme") == before
    assert await memory.reembed("acme") == 0


async def test_a_story_needs_text_whether_remembered_or_superseding() -> None:
    memory = await _memory()
    story = (await memory.remember("acme", "The go-live is February 20.")).story
    for blank in ("", "   \n\t"):
        with pytest.raises(InputError, match=r"^memory: a story needs text$"):
            await memory.remember("acme", blank)
        with pytest.raises(InputError, match=r"^memory: a story needs text$"):
            await memory.supersede("acme", story.story_id, blank)
    [kept] = await memory.get_stories("acme", [story.story_id])
    assert kept.valid_to is None
