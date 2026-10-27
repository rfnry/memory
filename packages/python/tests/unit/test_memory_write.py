import asyncio
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from rfnry_memory import memory as memory_module
from rfnry_memory.errors import ConfigurationError, InputError, ModelError, NotFoundError
from rfnry_memory.memory import Memory
from rfnry_memory.models import (
    Answer,
    CheckAnswer,
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
    Usage,
    Window,
)
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.rows import name_key

from .fakes import Fixed, HashEmbedding, Scripted, structured_reply

MANUAL = (
    "# Pump manual\n\n## Seals\n\n"
    + "Replace the seal yearly. " * 30
    + "\n\n## Wiring\n\nGround first."
)


async def _memory(storage: PostgresStorage, **kwargs: object) -> Memory:
    memory = Memory(
        storage,
        embedding=HashEmbedding(),
        chunk_chars=300,
        overlap_chars=30,
        **kwargs,  # type: ignore[arg-type]
    )
    await memory.check()
    return memory


async def test_ingest_text_chunks_embeds_and_stores(
    storage: PostgresStorage, embedding: HashEmbedding
) -> None:
    memory = Memory(storage, embedding=embedding, chunk_chars=300, overlap_chars=30)
    await memory.check()
    ingested = await memory.ingest("acme", text=MANUAL, name="manual", metadata={"line": "A"})
    assert ingested.source.name == "manual" and ingested.source.chunks > 3
    chunks = await memory.storage.source_chunks("acme", ingested.source.source_id)
    assert len(chunks) == ingested.source.chunks and len({c.text for c in chunks}) < len(chunks)
    assert embedding.calls == [list(dict.fromkeys(c.text for c in chunks))]
    assert chunks[1].title == "Pump manual > Seals" and chunks[-1].title == "Pump manual > Wiring"
    sources, _ = await memory.list_sources("acme", filters={"metadata.line": "A"})
    assert [s.source_id for s in sources] == [ingested.source.source_id]
    assert await memory.read_source("acme", ingested.source.source_id) == MANUAL.strip()


async def test_ingest_sections_keep_title_and_locator(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
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


async def test_remember_supersede_delete_story(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
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
        await memory.remember(
            "acme",
            "the go-live is  february 20.",
            subject="project:apollo",
            metadata={"team": "core"},
        )
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
        and current.supersedes == (story.story_id,)
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


async def test_an_entity_needs_a_name_and_a_type(storage: PostgresStorage) -> None:
    memory = await _memory(storage, pillars={"text", "vector", "graph"})
    story = (await memory.remember("acme", "The line runs.")).story
    refused = r"^memory: an entity needs a name and a type$"
    for blank in (EntityIn(name="  ", type="valve"), EntityIn(name="V-1", type=" \t")):
        with pytest.raises(InputError, match=refused):
            await memory.ingest("acme", name="sheet", text="V-1 feeds P-3", entities=[blank])
        with pytest.raises(InputError, match=refused):
            await memory.remember("acme", "V-1 feeds P-3", entities=[blank])
        with pytest.raises(InputError, match=refused):
            await memory.supersede("acme", story.story_id, "V-1 feeds P-3", entities=[blank])
    assert (await memory.storage.count_corpus("acme")).entities == 0
    assert (await memory.get_stories("acme", [story.story_id]))[0].valid_to is None


async def test_supplied_graph_facts_are_stored_without_a_model(storage: PostgresStorage) -> None:
    memory = await _memory(storage, pillars={"text", "vector", "graph"})
    ingested = await memory.ingest(
        "acme",
        name="sheet",
        text="V-101 feeds P-3",
        entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump")],
        relations=[RelationIn(from_name="V-101", to_name="P-3", type="CONNECTS_TO")],
    )
    residue = await memory.storage.count_corpus("acme")
    assert (residue.entities, residue.relations) == (2, 1)
    [valve] = (await memory.storage.entities_named("acme", ["v-101"]))[name_key("v-101")]
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
            valve.entity_id,
            (
                "chunk",
                (await memory.storage.source_chunks("acme", ingested.source.source_id))[0].chunk_id,
            ),
        ),
        (valve.entity_id, ("story", again.story_id)),
    ]
    with pytest.raises(InputError):
        await memory.remember(
            "acme", "x", relations=[RelationIn(from_name="ghost", to_name="P-3", type="X")]
        )


async def test_construction_rules(storage: PostgresStorage) -> None:
    with pytest.raises(ConfigurationError):
        Memory(storage, pillars={"vector"})
    with pytest.raises(ConfigurationError):
        Memory(storage, pillars={"sonar"})
    memory = Memory(storage)
    assert memory.pillars == {"text"}
    with pytest.raises(ConfigurationError):
        await memory.ingest("acme", name="x", text="a", entities=[EntityIn(name="a", type="b")])
    with pytest.raises(InputError):
        await memory.ingest("", name="x", text="a")


async def test_reembed_after_a_model_change(
    storage: PostgresStorage, embedding: HashEmbedding
) -> None:
    memory = Memory(storage, embedding=embedding)
    await memory.check()
    await memory.ingest("acme", name="a", text="alpha")
    await memory.remember("acme", "beta")
    embedding.model = "hash-8-v2"
    assert await memory.reembed("acme") == 2
    assert await memory.reembed("acme") == 0


async def test_delete_corpus_is_provable(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    await memory.ingest("acme", name="a", text="alpha")
    await memory.remember("acme", "beta")
    await memory.ingest("other", name="b", text="keep")
    assert not (await memory.verify_absent("acme")).clean
    deleted = await memory.delete_corpus("acme")
    assert deleted.sources == 1 and deleted.stories == 1
    assert (await memory.verify_absent("acme")).clean and not (
        await memory.verify_absent("other")
    ).clean


async def test_writes_without_an_embedding_model(storage: PostgresStorage) -> None:
    memory = Memory(storage)
    await memory.check()
    ingested = await memory.ingest("acme", name="manual", text=MANUAL)
    assert ingested.source.chunks > 0
    story = (await memory.remember("acme", "The go-live is February 20.")).story
    current = (await memory.supersede("acme", story.story_id, "The go-live is March 3.")).story
    [closed] = await memory.get_stories("acme", [story.story_id])
    assert closed.valid_to == current.valid_from and current.supersedes == (story.story_id,)
    with pytest.raises(ConfigurationError):
        await memory.reembed("acme")


async def test_supersede_into_text_another_open_story_holds_is_refused(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
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
        current.supersedes == (held.story_id,)
        and (await memory.get_stories("acme", [held.story_id]))[0].valid_to is not None
    )


WRITES = {"put_source", "sync_sources", "put_story", "supersede_story"}


class _Recorder:
    def __init__(self, inner: PostgresStorage, log: list[str]) -> None:
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

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        self._log.append("model.embed")
        return await super().embed(texts, purpose=purpose)


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


async def test_every_model_call_happens_before_the_one_storage_write(
    storage: PostgresStorage,
) -> None:
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
    memory = Memory(
        cast(PostgresStorage, _Recorder(storage, log)),
        embedding=_LoggedEmbedding(log),
        generation=generation,
        decision=decision,
        relation_types=["OWNS"],
        chunk_chars=300,
        overlap_chars=30,
    )
    await memory.check()
    await memory.remember("acme", "ACME Corporation owns valve V-9.")

    for verb in ("ingest", "remember", "supersede", "sync"):
        log.clear()
        if verb == "ingest":
            await memory.ingest("acme", name="manual", text=MANUAL, metadata={"file": "m"})
        elif verb == "remember":
            story = (await memory.remember("acme", "Acme Corp owns V-101.")).story
        elif verb == "supersede":
            await memory.supersede("acme", story.story_id, "Acme Corp sold V-101.")
        else:
            await memory.sync(
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fusion": {"k": 60}}, r"^memory: fusion is RRF\(\.\.\.\) or RelativeScore\(\.\.\.\)$"),
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
    storage: PostgresStorage, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        Memory(storage, **kwargs)


def test_source_verbs_share_one_name() -> None:
    assert {"get_source", "list_sources", "read_source", "delete_source"} <= set(
        dir(Memory)
    ) and not hasattr(Memory, "read")


async def test_a_story_is_true_from_valid_from_which_defaults_to_the_write(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    jan = datetime(2026, 1, 10, tzinfo=UTC)
    told = (await memory.remember("acme", "The go-live is February 20.", valid_from=jan)).story
    assert (
        told.valid_from == jan
        and told.created_at > jan
        and not any("occurred" in field for field in type(told).model_fields)
    )
    now = (await memory.remember("acme", "Ana owns the review.")).story
    assert now.valid_from <= now.created_at
    later = (await memory.supersede("acme", now.story_id, "Bo owns the review.")).story
    [closed] = await memory.get_stories("acme", [now.story_id])
    assert closed.valid_to is not None and later.valid_from == closed.valid_to <= later.created_at
    with pytest.raises(
        InputError,
        match=rf"^memory: valid_from must not be before story '{told.story_id}' became true, at "
        rf"{re.escape(jan.isoformat())}; that moment itself corrects it from the start$",
    ):
        await memory.supersede(
            "acme", told.story_id, "The go-live is March 3.", valid_from=jan - timedelta(days=1)
        )


class _Short(HashEmbedding):
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [v[:7] for v in await super().embed(texts, purpose=purpose)]


class _Fewer(HashEmbedding):
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return (await super().embed(texts, purpose=purpose))[:-1]


@pytest.mark.parametrize("broken", [_Short(), _Fewer()])
async def test_a_write_refuses_what_an_embedding_model_returns_outside_its_contract(
    storage: PostgresStorage,
    broken: HashEmbedding,
) -> None:
    memory = await _memory(storage)
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
        await wrong.sync(
            "acme",
            filters={"metadata.file": "a"},
            sources=[SourceIn(name="v1", text="version one", metadata={"file": "a"})],
        )
    broken.model = "hash-8-v2"
    with pytest.raises(ModelError, match=returned):
        await wrong.reembed("acme")
    assert await memory.verify_absent("acme") == before
    assert await memory.reembed("acme") == 0


async def test_a_story_needs_text_whether_remembered_or_superseding(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    story = (await memory.remember("acme", "The go-live is February 20.")).story
    for blank in ("", "   \n\t"):
        with pytest.raises(InputError, match=r"^memory: a story needs text$"):
            await memory.remember("acme", blank)
        with pytest.raises(InputError, match=r"^memory: a story needs text$"):
            await memory.supersede("acme", story.story_id, blank)
    [kept] = await memory.get_stories("acme", [story.story_id])
    assert kept.valid_to is None


async def test_remembering_an_open_text_again_ignores_case_and_extra_whitespace(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    first = (
        await memory.remember("acme", "Ana prefers the Tuesday slot.", subject="user:ana")
    ).story
    again = await memory.remember("acme", "  ana PREFERS the\ttuesday   slot. ", subject="user:ana")
    assert again.story.story_id == first.story_id and again.usage.input == 0
    split = (await memory.remember("acme", "Ana prefers the Tues day slot.")).story
    assert split.story_id != first.story_id
    stories, _ = await memory.list_stories("acme")
    assert sorted(s.story_id for s in stories) == sorted([split.story_id, first.story_id])


async def test_the_same_text_is_one_story_per_subject_and_metadata(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    ana = (await memory.remember("acme", "Prefers mornings.", subject="user:ana")).story
    bo = (await memory.remember("acme", "prefers  MORNINGS.", subject="user:bo")).story
    bare = (await memory.remember("acme", "Prefers mornings.")).story
    line = (
        await memory.remember(
            "acme", "Prefers mornings.", subject="user:ana", metadata={"line": "A"}
        )
    ).story
    assert len({ana.story_id, bo.story_id, bare.story_id, line.story_id}) == 4
    repeat = await memory.remember("acme", "prefers mornings.", subject="user:bo")
    assert repeat.story.story_id == bo.story_id
    assert (await memory.remember("acme", "Prefers mornings.")).story.story_id == bare.story_id
    stories, _ = await memory.list_stories("acme")
    assert len(stories) == 4


async def test_metadata_is_compared_as_json_values(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    first = (
        await memory.remember(
            "acme", "Shift starts at six.", metadata={"site": {"zone": 3, "bay": "b"}, "line": "A"}
        )
    ).story
    for same in (
        {"line": "A", "site": {"bay": "b", "zone": 3}},
        {"line": "A", "site": {"bay": "b", "zone": 3.0}},
    ):
        repeat = await memory.remember("acme", "Shift starts at six.", metadata=same)
        assert repeat.story.story_id == first.story_id
    for other in (
        {"line": "A", "site": {"bay": "c", "zone": 3}},
        {"line": ["A"], "site": {"bay": "b", "zone": 3}},
        {"line": "A", "site": {"bay": "b", "zone": 3}, "flag": True},
        {"line": "A", "site": {"bay": "b", "zone": 3}, "flag": 1},
    ):
        written = (await memory.remember("acme", "Shift starts at six.", metadata=other)).story
        assert written.story_id != first.story_id and written.metadata == other


async def test_a_repeat_from_another_source_returns_the_first_story(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    one = (await memory.remember("acme", "Valve V-2 leaks.", source_ids=("doc-1",))).story
    two = await memory.remember("acme", "valve v-2 leaks.", source_ids=("doc-2",))
    assert two.story.story_id == one.story_id and two.story.source_ids == ("doc-1",)


async def test_a_past_story_repeats_only_with_the_same_subject_and_interval(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    march, april = datetime(2020, 3, 1, tzinfo=UTC), datetime(2020, 4, 1, tzinfo=UTC)
    past = (
        await memory.remember(
            "acme", "The audit ran.", subject="site:a", valid_from=march, valid_to=april
        )
    ).story
    again = await memory.remember(
        "acme", "the audit ran.", subject="site:a", valid_from=march, valid_to=april
    )
    assert again.story.story_id == past.story_id
    other = await memory.remember(
        "acme", "The audit ran.", subject="site:b", valid_from=march, valid_to=april
    )
    assert other.story.story_id != past.story_id


async def test_supersede_refuses_only_text_held_under_the_same_subject_and_metadata(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    await memory.remember("acme", "The go-live is March 3.", subject="project:apollo")
    zeus = (
        await memory.remember("acme", "The go-live is February 20.", subject="project:zeus")
    ).story
    current = (await memory.supersede("acme", zeus.story_id, "The go-live is March 3.")).story
    assert current.subject == "project:zeus" and current.supersedes == (zeus.story_id,)
    april = (await memory.remember("acme", "The go-live is April 1.", subject="project:zeus")).story
    with pytest.raises(InputError, match=rf"^memory: story '{current.story_id}', true since "):
        await memory.supersede("acme", april.story_id, "the go-live is march 3.")
    moved = await memory.supersede(
        "acme", april.story_id, "The go-live is March 3.", subject="project:hera"
    )
    assert moved.story.subject == "project:hera"
    tagged = (
        await memory.remember(
            "acme", "The go-live is May 5.", subject="project:zeus", metadata={"team": "core"}
        )
    ).story
    kept = await memory.supersede("acme", tagged.story_id, "The go-live is March 3.")
    assert kept.story.metadata == {"team": "core"} and kept.story.subject == "project:zeus"


async def test_a_story_can_carry_its_planned_end(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    march, april = datetime(2020, 3, 1, tzinfo=UTC), datetime(2020, 4, 1, tzinfo=UTC)
    promo = (
        await memory.remember(
            "acme", "The spring promotion runs.", subject="promo", valid_from=march, valid_to=april
        )
    ).story
    assert promo.valid_to == april
    during, _ = await memory.list_stories(
        "acme", subject="promo", as_of=datetime(2020, 3, 15, tzinfo=UTC)
    )
    after, _ = await memory.list_stories(
        "acme", subject="promo", as_of=datetime(2020, 4, 2, tzinfo=UTC)
    )
    assert [s.story_id for s in during] == [promo.story_id] and after == []
    repeat = await memory.remember(
        "acme", "the spring  promotion runs.", subject="promo", valid_from=march, valid_to=april
    )
    assert repeat.story.story_id == promo.story_id
    future = datetime.now(UTC) + timedelta(days=30)
    fair = (
        await memory.remember("acme", "The fair is on.", valid_to=future.replace(tzinfo=None))
    ).story
    assert fair.valid_to == future
    assert (await memory.remember("acme", "The fair is on.")).story.story_id == fair.story_id
    with pytest.raises(InputError, match=r"^memory: valid_to must be after valid_from$"):
        await memory.remember("acme", "Never true.", valid_from=april, valid_to=march)


async def test_a_supersede_closes_a_planned_story_before_its_end(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    start = datetime.now(UTC) - timedelta(days=1)
    end = datetime.now(UTC) + timedelta(days=30)
    promo = (
        await memory.remember("acme", "The promotion runs.", valid_from=start, valid_to=end)
    ).story
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{promo.story_id}' ends at .+, before valid_from; pass an earlier valid_from$",
    ):
        await memory.supersede(
            "acme", promo.story_id, "The promotion runs on.", valid_from=end + timedelta(days=1)
        )
    soon = datetime.now(UTC) + timedelta(days=10)
    early = (
        await memory.supersede(
            "acme",
            promo.story_id,
            "The promotion ends early.",
            valid_from=soon,
            valid_to=soon + timedelta(days=1),
        )
    ).story
    [closed] = await memory.get_stories("acme", [promo.story_id])
    assert closed.valid_to == soon and early.valid_to == soon + timedelta(days=1)
    past = (
        await memory.remember(
            "acme",
            "The fair ran.",
            valid_from=datetime(2020, 1, 1, tzinfo=UTC),
            valid_to=datetime(2020, 1, 5, tzinfo=UTC),
        )
    ).story
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{past.story_id}' ends at 2020-01-05T00:00:00\+00:00, before valid_from; pass an earlier valid_from$",
    ):
        await memory.supersede("acme", past.story_id, "The fair runs again.")
    moved = (
        await memory.supersede(
            "acme",
            past.story_id,
            "The fair moved indoors.",
            valid_from=datetime(2020, 1, 3, tzinfo=UTC),
        )
    ).story
    [closed] = await memory.get_stories("acme", [past.story_id])
    assert (closed.valid_to, closed.superseded_by) == (moved.valid_from, moved.story_id)


async def test_supersede_merges_several_stories_into_one(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    jan, feb = datetime(2026, 1, 10, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    tuesday = (
        await memory.remember(
            "acme",
            "Ana prefers Tuesday.",
            subject="user:ana",
            valid_from=jan,
            source_ids=("s1",),
            metadata={"team": "core"},
        )
    ).story
    mornings = (
        await memory.remember(
            "acme",
            "Ana prefers mornings.",
            subject="user:ana",
            valid_from=jan,
            source_ids=("s2", "s1"),
            metadata={"team": "core"},
        )
    ).story
    merged = (
        await memory.supersede(
            "acme",
            [tuesday.story_id, mornings.story_id],
            "Ana prefers Tuesday mornings.",
            valid_from=feb,
        )
    ).story
    assert merged.supersedes == (tuesday.story_id, mornings.story_id)
    assert (merged.subject, merged.metadata, merged.source_ids) == (
        "user:ana",
        {"team": "core"},
        ("s1", "s2"),
    )
    closed = await memory.get_stories("acme", [tuesday.story_id, mornings.story_id])
    assert [(s.valid_to, s.superseded_by) for s in closed] == [(feb, merged.story_id)] * 2
    bo = (
        await memory.remember(
            "acme",
            "Bo prefers Fridays.",
            subject="user:bo",
            valid_from=jan,
            metadata={"team": "ops"},
        )
    ).story
    cy = (
        await memory.remember("acme", "Cy prefers Mondays.", subject="user:cy", valid_from=jan)
    ).story
    mixed = (
        await memory.supersede(
            "acme", [bo.story_id, cy.story_id], "Bo and Cy share Mondays.", valid_from=feb
        )
    ).story
    assert (mixed.subject, mixed.metadata) == (None, {})


async def test_supersede_takes_an_end_as_remember_does(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    jan, feb = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    story = (await memory.remember("acme", "The line runs nights.", valid_from=jan)).story
    with pytest.raises(InputError, match=r"^memory: valid_to must be after valid_from$"):
        await memory.supersede(
            "acme", story.story_id, "The line runs days.", valid_from=feb, valid_to=feb
        )
    assert (await memory.get_stories("acme", [story.story_id]))[0].superseded_by is None
    days = (
        await memory.supersede(
            "acme",
            story.story_id,
            "The line runs days.",
            valid_from=feb,
            valid_to=datetime(2030, 1, 1),
        )
    ).story
    assert days.valid_to == datetime(2030, 1, 1, tzinfo=UTC)


async def test_a_retried_past_interval_returns_its_repeat_before_any_model_call(
    storage: PostgresStorage,
) -> None:
    embedding = HashEmbedding()
    empty: dict[str, Any] = {"entities": [], "relations": []}
    llm = Scripted(structured_reply(empty), structured_reply(empty), structured_reply(empty))
    memory = Memory(storage, embedding=embedding, generation=llm, relation_types=["OWNS"])
    await memory.check()
    jan1, jan5, jan9 = (datetime(2020, 1, d, tzinfo=UTC) for d in (1, 5, 9))
    first = (
        await memory.remember("acme", "The fair ran.", valid_from=jan1, valid_to=jan5, related=0)
    ).story
    spent = (len(llm.calls), len(embedding.calls))
    again = await memory.remember(
        "acme", "the fair  ran.", valid_from=jan1, valid_to=jan5, related=0
    )
    assert again.story.story_id == first.story_id
    assert (len(llm.calls), len(embedding.calls)) == spent
    assert again.usage == Usage() and again.related.hits == []
    await memory.remember("acme", "The fair runs again.", related=0)
    shown = await memory.remember("acme", "The fair ran.", valid_from=jan1, valid_to=jan5)
    assert shown.story.story_id == first.story_id and len(llm.calls) == spent[0] + 1
    assert [h.text for h in shown.related.hits] == ["The fair runs again."]
    longer = (
        await memory.remember("acme", "The fair ran.", valid_from=jan1, valid_to=jan9, related=0)
    ).story
    assert longer.story_id != first.story_id and len(llm.calls) == spent[0] + 2
    plain = Memory(storage, embedding=embedding)
    fair = EntityIn(name="Fair", type="event")
    for start, end in ((None, None), (jan1, jan5)):
        await plain.remember("acme", "The fair ran.", valid_from=start, valid_to=end, related=0)
        with pytest.raises(ConfigurationError, match=r"^memory: graph facts were supplied"):
            await plain.remember(
                "acme", "The fair ran.", valid_from=start, valid_to=end, entities=[fair]
            )


async def test_the_related_graph_leg_starts_from_the_entities_the_write_resolved(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"graph"})
    await memory.check()
    john = EntityIn(name="John Smith", type="person")
    owned = (await memory.remember("acme", "John Smith owns the shop.", entities=[john])).story
    await memory.remember("acme", "Bo likes tea.")
    signed = await memory.remember("acme", "He signed the lease.", entities=[john])
    assert [h.id for h in signed.related.hits] == [owned.story_id]
    renewed = await memory.supersede(
        "acme", signed.story.story_id, "He signed a new lease.", entities=[john]
    )
    assert [h.id for h in renewed.related.hits] == [owned.story_id]
    plain = await memory.search("acme", "He signed the lease.", kinds=("story",))
    assert plain.hits == []


async def test_supersede_several_takes_given_fields_and_refuses_a_closed_one(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    jan, feb = datetime(2026, 1, 10, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    bo = (
        await memory.remember(
            "acme", "Bo prefers Fridays.", subject="user:bo", valid_from=jan, metadata={"t": "ops"}
        )
    ).story
    cy = (
        await memory.remember(
            "acme", "Cy prefers Mondays.", subject="user:cy", valid_from=jan, metadata={"t": "core"}
        )
    ).story
    shared = (
        await memory.supersede(
            "acme",
            [bo.story_id, cy.story_id],
            "Bo and Cy share Mondays.",
            valid_from=feb,
            subject="team",
            metadata={"t": "all"},
        )
    ).story
    assert (shared.subject, shared.metadata) == ("team", {"t": "all"})
    closed = await memory.get_stories("acme", [bo.story_id, cy.story_id])
    assert [s.closed_at for s in closed] == [shared.created_at] * 2
    dee = (await memory.remember("acme", "Dee prefers Thursdays.", valid_from=jan)).story
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{bo.story_id}' is already superseded; supersede the story that replaced it$",
    ):
        await memory.supersede(
            "acme", [dee.story_id, bo.story_id], "Dee and Bo share Thursdays.", valid_from=feb
        )
    assert (await memory.get_stories("acme", [dee.story_id]))[0].superseded_by is None


async def test_supersede_refuses_a_list_it_cannot_close(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    story = (
        await memory.remember(
            "acme", "The line runs nights.", valid_from=datetime(2026, 1, 10, tzinfo=UTC)
        )
    ).story
    with pytest.raises(InputError, match=r"^memory: supersede needs at least one story_id$"):
        await memory.supersede("acme", [], "Anything.")
    with pytest.raises(InputError, match=r"^memory: story_id names a story more than once$"):
        await memory.supersede("acme", [story.story_id, story.story_id], "Anything.")
    with pytest.raises(NotFoundError, match=r"^memory: story 'ghost' was not found$"):
        await memory.supersede("acme", [story.story_id, "ghost"], "Anything.")
    assert (await memory.get_stories("acme", [story.story_id]))[0].superseded_by is None


async def test_a_write_returns_the_open_stories_it_may_repeat_or_contradict(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    jan, feb = datetime(2026, 1, 10, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    tuesday = (
        await memory.remember(
            "acme", "Ana prefers the Tuesday slot.", subject="user:ana", valid_from=jan
        )
    ).story
    monday = (
        await memory.remember(
            "acme", "Ana prefers the Monday slot.", subject="user:ana", valid_from=jan
        )
    ).story
    await memory.supersede("acme", monday.story_id, "Ana dropped the Monday slot.", valid_from=feb)
    wednesday = await memory.remember("acme", "Ana prefers the Wednesday slot.", subject="user:ana")
    related = [h.id for h in wednesday.related.hits]
    assert tuesday.story_id in related and monday.story_id not in related
    assert wednesday.story.story_id not in related
    assert all(h.judge is None for h in wednesday.related.hits)
    assert wednesday.related.usage == Usage()
    again = await memory.remember("acme", "ana prefers the wednesday  slot.", subject="user:ana")
    assert again.story.story_id == wednesday.story.story_id
    assert wednesday.story.story_id not in [h.id for h in again.related.hits]
    later = await memory.supersede(
        "acme", tuesday.story_id, "Ana moved to the Thursday slot.", valid_from=feb
    )
    assert tuesday.story_id not in [h.id for h in later.related.hits]
    quiet = await memory.remember("acme", "Bo prefers Fridays.", related=0)
    assert quiet.related.hits == [] and quiet.related.pillars == {}
    with pytest.raises(InputError, match=r"^memory: related must be at least 0$"):
        await memory.remember("acme", "Cy prefers Mondays.", related=-1)
    with pytest.raises(ConfigurationError, match=r"^memory: judge=True needs a decision model$"):
        await memory.remember("acme", "Cy prefers Mondays.", judge=True)


async def test_a_write_finds_its_related_stories_inside_the_filters(
    storage: PostgresStorage,
) -> None:
    def judged(state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Answer]:
        return {name: CheckAnswer(probability=0.5) for name in questions}

    decision = Fixed(judged)
    memory = await _memory(storage, decision=decision)
    for day in ("Monday", "Tuesday", "Thursday"):
        await memory.remember("acme", f"Ana prefers the {day} slot.", metadata={"line": "B"})
    friday = (
        await memory.remember("acme", "Ana sometimes takes Fridays.", metadata={"line": "A"})
    ).story
    view = {"metadata.line": "A"}
    written = await memory.remember(
        "acme",
        "Ana prefers the Wednesday slot.",
        metadata={"line": "A"},
        filters=view,
        related=1,
        judge=True,
    )
    assert [h.id for h in written.related.hits] == [friday.story_id]
    assert [s["passage"]["text"] for s, _ in decision.calls if isinstance(s, dict)] == [
        "Ana sometimes takes Fridays."
    ]
    later = await memory.supersede(
        "acme", friday.story_id, "Ana dropped Fridays.", filters=view, related=5
    )
    assert [h.id for h in later.related.hits] == [written.story.story_id]
    with pytest.raises(InputError, match=r"^filters: filter key 'corpus_id' is reserved"):
        await memory.remember("acme", "Bo prefers Fridays.", filters={"corpus_id": "x"}, related=0)
    with pytest.raises(InputError, match=r"^filters: filter key 'line' must address a facet"):
        await memory.supersede(
            "acme", written.story.story_id, "Ana moved.", filters={"line": "A"}, related=0
        )


async def test_every_write_path_judges_its_related_stories_and_counts_the_cost(
    storage: PostgresStorage,
) -> None:
    def judged(state: State, questions: Mapping[str, QuestionLike]) -> Decided:
        return Decided(
            answers={name: CheckAnswer(probability=0.5) for name in questions},
            usage=Usage(input=7),
        )

    memory = await _memory(storage, decision=Fixed(judged))
    jan = datetime(2026, 1, 10, tzinfo=UTC)
    tuesday = (await memory.remember("acme", "Ana prefers the Tuesday slot.", valid_from=jan)).story
    await memory.remember("acme", "Ana prefers the Monday slot.", valid_from=jan)
    written = await memory.remember(
        "acme", "Ana prefers the Wednesday slot.", related=2, judge=True
    )
    again = await memory.remember("acme", "ana prefers the wednesday  slot.", related=2, judge=True)
    moved = await memory.supersede(
        "acme",
        tuesday.story_id,
        "Ana moved to the Thursday slot.",
        valid_from=datetime(2026, 2, 1, tzinfo=UTC),
        related=2,
        judge=True,
    )
    assert again.story.story_id == written.story.story_id
    for write in (written, again, moved):
        assert len(write.related.hits) == 2
        assert all(h.judge is not None for h in write.related.hits)
        assert write.related.usage == write.usage == Usage(input=14)


async def test_the_graph_leg_finds_related_stories_that_name_the_same_entity(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, pillars={"graph"})
    await memory.check()
    apollo = EntityIn(name="Apollo", type="project")
    owned = (
        await memory.remember(
            "acme", "Ana owns Apollo.", entities=[apollo, EntityIn(name="Ana", type="person")]
        )
    ).story
    await memory.remember("acme", "Bo likes tea.")
    written = await memory.remember("acme", "Apollo ships in May.", entities=[apollo])
    assert [h.id for h in written.related.hits] == [owned.story_id]
    assert set(written.related.pillars) == {"graph"}
    viewed = await memory.remember(
        "acme", "Apollo slips to June.", entities=[apollo], filters={"metadata.line": "A"}
    )
    assert viewed.related.hits == [] and viewed.related.pillars == {}


async def test_a_write_judges_its_related_stories_when_asked(storage: PostgresStorage) -> None:
    def judged(state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Answer]:
        assert isinstance(state, dict)
        clash = 0.9 if "Tuesday" in state["passage"]["text"] else 0.1
        return {
            "relevant": CheckAnswer(probability=0.8),
            "evidence": CheckAnswer(probability=0.2),
            "contradicts": CheckAnswer(probability=clash),
            "injection": CheckAnswer(probability=0.0),
        }

    decision = Fixed(judged)
    memory = await _memory(storage, decision=decision)
    tuesday = (
        await memory.remember(
            "acme",
            "Ana prefers the Tuesday slot.",
            subject="user:ana",
            valid_from=datetime(2026, 1, 10, tzinfo=UTC),
        )
    ).story
    await memory.remember("acme", "Ana likes long walks.", subject="user:ana")
    written = await memory.remember(
        "acme", "Ana prefers the Wednesday slot.", subject="user:ana", judge=True
    )
    top = written.related.by("contradicts").hits[0]
    assert top.id == tuesday.story_id and top.judge is not None and top.judge.contradicts == 0.9
    assert {s["query"] for s, _ in decision.calls if isinstance(s, dict)} == {
        "Ana prefers the Wednesday slot."
    }


class Slow(HashEmbedding):
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        await asyncio.sleep(0.05)
        return await super().embed(texts, purpose=purpose)


async def test_a_story_is_recorded_when_its_write_lands_after_its_model_calls(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=Slow())
    await memory.check()
    told = (await memory.remember("acme", "Ana owns the review.", related=0)).story
    assert told.created_at - told.valid_from >= timedelta(milliseconds=50)
    later = (await memory.supersede("acme", told.story_id, "Bo owns the review.", related=0)).story
    [closed] = await memory.get_stories("acme", [told.story_id])
    assert later.created_at - later.valid_from >= timedelta(milliseconds=50)
    assert closed.valid_to == later.valid_from and closed.closed_at == later.created_at


async def test_a_supersede_behind_the_clock_of_the_story_it_closes_records_a_later_moment(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _memory(storage)
    jan = datetime(2026, 1, 10, tzinfo=UTC)
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 1, 20, tzinfo=UTC))
    told = (await memory.remember("acme", "Ana owns the review.", valid_from=jan)).story
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 1, 15, tzinfo=UTC))
    later = (await memory.supersede("acme", told.story_id, "Bo owns the review.")).story
    [closed] = await memory.get_stories("acme", [told.story_id])
    assert closed.closed_at is not None and closed.closed_at > told.created_at
    assert later.created_at == closed.closed_at == told.created_at + timedelta(microseconds=1)
    assert later.valid_from == closed.valid_to == datetime(2026, 1, 15, tzinfo=UTC)


async def test_an_ingest_is_recorded_when_its_write_lands_after_its_model_calls(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage, embedding=Slow())
    await memory.check()
    began = memory_module._now()
    source = (await memory.ingest("acme", name="manual", text="Replace the seal.")).source
    assert source.created_at - began >= timedelta(milliseconds=50)


async def test_a_supersede_called_before_its_story_ends_closes_it_though_its_write_lands_after(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])

    class Ticking(HashEmbedding):
        async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
            clock[0] += timedelta(seconds=2)
            return await super().embed(texts, purpose=purpose)

    memory = Memory(storage, embedding=Ticking())
    await memory.check()
    ends = clock[0] + timedelta(seconds=3)
    told = (await memory.remember("acme", "Ana owns the review.", valid_to=ends, related=0)).story
    later = (await memory.supersede("acme", told.story_id, "Bo owns the review.", related=0)).story
    [closed] = await memory.get_stories("acme", [told.story_id])
    assert later.valid_from == closed.valid_to == ends - timedelta(seconds=1)
    assert later.created_at == closed.closed_at == ends + timedelta(seconds=1)


async def test_end_story_closes_a_story_with_nothing_in_its_place(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = await _memory(storage, pillars={"text", "vector", "graph"})
    jan, feb = datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    told = (
        await memory.remember(
            "acme",
            "Ana prefers the Tuesday slot.",
            subject="user:ana",
            valid_from=jan,
            entities=[EntityIn(name="Ana", type="person"), EntityIn(name="Tuesday", type="slot")],
            relations=[RelationIn(from_name="Ana", to_name="Tuesday", type="PREFERS")],
        )
    ).story
    clock[0] = datetime(2026, 2, 10, tzinfo=UTC)
    ended = await memory.end_story("acme", told.story_id)
    assert ended.valid_to == ended.closed_at == datetime(2026, 2, 10, tzinfo=UTC)
    assert (ended.story_id, ended.text, ended.superseded_by) == (told.story_id, told.text, None)
    assert await memory.get_stories("acme", [told.story_id]) == [ended]
    assert (await memory.list_stories("acme", subject="user:ana"))[0] == []
    assert (await memory.list_stories("acme", subject="user:ana", as_of=feb))[0] == [ended]
    assert (await memory.list_stories("acme", subject="user:ana", history=True))[0] == [ended]
    known, _ = await memory.list_stories("acme", subject="user:ana", known_at=feb)
    assert [(s.story_id, s.valid_to, s.closed_at) for s in known] == [(told.story_id, None, None)]
    assert (await memory.recall("acme", "Tuesday slot", subject="user:ana")).hits == []
    every = await memory.recall("acme", "Tuesday slot", subject="user:ana", history=True)
    assert [h.id for h in every.hits] == [told.story_id]
    hood = await memory.neighbors("acme", "Ana")
    assert (hood.relations, hood.passages) == ([], [])
    then = await memory.neighbors("acme", "Ana", as_of=feb)
    assert [r.type for r in then.relations] == ["PREFERS"]
    again = (
        await memory.remember("acme", "Ana prefers the Tuesday slot.", subject="user:ana")
    ).story
    assert again.story_id != told.story_id and again.valid_from == ended.valid_to


async def test_end_story_takes_a_past_end_and_shortens_a_planned_one(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    jan, feb = datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    told = (await memory.remember("acme", "Ana prefers mornings.", valid_from=jan)).story
    ended = await memory.end_story("acme", told.story_id, valid_to=feb.replace(tzinfo=None))
    assert ended.valid_to == feb
    assert ended.closed_at is not None and ended.closed_at > told.created_at
    start = datetime.now(UTC) - timedelta(days=1)
    plan = datetime.now(UTC) + timedelta(days=30)
    promo = (
        await memory.remember(
            "acme", "The promotion runs.", subject="promo", valid_from=start, valid_to=plan
        )
    ).story
    stopped = await memory.end_story("acme", promo.story_id)
    assert stopped.closed_at is not None and stopped.valid_to is not None
    assert start < stopped.valid_to <= stopped.closed_at < plan
    known, _ = await memory.list_stories(
        "acme", subject="promo", known_at=stopped.closed_at - timedelta(microseconds=1)
    )
    assert [(s.valid_to, s.closed_at) for s in known] == [(plan, None)]
    learned, _ = await memory.list_stories(
        "acme", subject="promo", known_at=stopped.closed_at, history=True
    )
    assert [(s.valid_to, s.closed_at) for s in learned] == [(stopped.valid_to, stopped.closed_at)]
    assert (await memory.list_stories("acme", subject="promo"))[0] == []


async def test_end_story_refuses_what_it_cannot_end(storage: PostgresStorage) -> None:
    memory = await _memory(storage)
    with pytest.raises(NotFoundError, match=r"^memory: story 'missing' was not found$"):
        await memory.end_story("acme", "missing")
    jan = datetime(2026, 1, 5, tzinfo=UTC)
    told = (await memory.remember("acme", "Ana owns the review.", valid_from=jan)).story
    with pytest.raises(NotFoundError, match=rf"^memory: story '{told.story_id}' was not found$"):
        await memory.end_story("other", told.story_id)
    with pytest.raises(
        InputError,
        match=rf"^memory: valid_to must not be before story '{told.story_id}' became true, at 2026-01-05T00:00:00\+00:00; that moment itself retracts it$",
    ):
        await memory.end_story("acme", told.story_id, valid_to=jan - timedelta(days=1))
    assert await memory.get_stories("acme", [told.story_id]) == [told]
    later = (await memory.supersede("acme", told.story_id, "Bo owns the review.")).story
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{told.story_id}' is already superseded; end the story that replaced it$",
    ):
        await memory.end_story("acme", told.story_id)
    ended = await memory.end_story("acme", later.story_id)
    assert ended.valid_to is not None
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{later.story_id}' already ended, at {re.escape(ended.valid_to.isoformat())}$",
    ):
        await memory.end_story("acme", later.story_id)
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{later.story_id}' ends at {re.escape(ended.valid_to.isoformat())}, before valid_from; pass an earlier valid_from$",
    ):
        await memory.supersede("acme", later.story_id, "Cy owns the review.")
    past = (
        await memory.remember(
            "acme",
            "The fair ran.",
            valid_from=datetime(2020, 1, 1, tzinfo=UTC),
            valid_to=datetime(2020, 1, 5, tzinfo=UTC),
        )
    ).story
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{past.story_id}' already ended, at 2020-01-05T00:00:00\+00:00$",
    ):
        await memory.end_story("acme", past.story_id)
    begins = datetime.now(UTC) + timedelta(days=10)
    planned = (await memory.remember("acme", "The new line opens.", valid_from=begins)).story
    for valid_to in (None, datetime(2026, 1, 1, tzinfo=UTC)):
        with pytest.raises(
            InputError,
            match=rf"^memory: story '{planned.story_id}' has not begun; it becomes true at {re.escape(begins.isoformat())}, and a valid_to of that moment retracts it$",
        ):
            await memory.end_story("acme", planned.story_id, valid_to=valid_to)


async def test_an_end_still_to_come_plans_the_end_of_a_story_nothing_closed(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = await _memory(storage)
    jan, march = datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)
    calls = (
        await memory.remember(
            "acme", "Ana takes calls only in the morning.", subject="user:ana", valid_from=jan
        )
    ).story
    clock[0] = datetime(2026, 1, 20, tzinfo=UTC)
    planned = await memory.end_story(
        "acme", calls.story_id, valid_to=march, written_by={"user": "bo"}
    )
    assert (planned.valid_to, planned.closed_at, planned.closed_by, planned.superseded_by) == (
        march,
        clock[0],
        {"user": "bo"},
        None,
    )
    now, _ = await memory.list_stories("acme", subject="user:ana")
    assert [s.story_id for s in now] == [calls.story_id]
    assert (await memory.list_stories("acme", subject="user:ana", as_of=march))[0] == []
    before, _ = await memory.list_stories(
        "acme",
        subject="user:ana",
        known_at=clock[0] - timedelta(microseconds=1),
        as_of=datetime(2026, 4, 1, tzinfo=UTC),
    )
    assert [(s.valid_to, s.closed_at, s.closed_by) for s in before] == [(None, None, None)]
    again = await memory.remember(
        "acme", "Ana takes calls only in the morning.", subject="user:ana"
    )
    assert again.story.story_id == planned.story_id
    later = datetime(2026, 6, 1, tzinfo=UTC)
    promo = (
        await memory.remember(
            "acme", "The promotion runs.", subject="promo", valid_from=jan, valid_to=march
        )
    ).story
    longer = await memory.end_story("acme", promo.story_id, valid_to=later)
    assert (longer.valid_from, longer.valid_to) == (jan, later)
    begins = datetime(2026, 2, 1, tzinfo=UTC)
    line = (await memory.remember("acme", "The new line opens.", valid_from=begins)).story
    closes = await memory.end_story("acme", line.story_id, valid_to=march)
    assert (closes.valid_from, closes.valid_to) == (begins, march)
    for valid_to in (None, jan):
        story = (await memory.remember("acme", f"Line {valid_to} opens.", valid_from=begins)).story
        with pytest.raises(
            InputError,
            match=rf"^memory: story '{story.story_id}' has not begun; it becomes true at 2026-02-01T00:00:00\+00:00, and a valid_to of that moment retracts it$",
        ):
            await memory.end_story("acme", story.story_id, valid_to=valid_to)


async def test_a_story_whose_planned_end_passed_can_be_retracted_ended_earlier_or_corrected(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = await _memory(storage)
    jan, feb = datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    march = datetime(2026, 3, 31, tzinfo=UTC)
    told = {
        name: (
            await memory.remember(
                "acme", f"The {name} runs.", subject=name, valid_from=jan, valid_to=march
            )
        ).story
        for name in ("promo", "fair", "sale", "show")
    }
    clock[0] = datetime(2026, 4, 10, tzinfo=UTC)
    promo, fair, sale, show = told.values()
    retracted = await memory.end_story("acme", promo.story_id, valid_to=promo.valid_from)
    assert (retracted.valid_from, retracted.valid_to, retracted.closed_at) == (jan, jan, clock[0])
    ended = await memory.end_story("acme", fair.story_id, valid_to=feb)
    assert (ended.valid_to, ended.closed_at, ended.superseded_by) == (feb, clock[0], None)
    fixed = (
        await memory.supersede(
            "acme", sale.story_id, "The sale ran in February.", valid_from=sale.valid_from
        )
    ).story
    [old] = await memory.get_stories("acme", [sale.story_id])
    assert (old.valid_to, old.superseded_by, fixed.valid_from) == (jan, fixed.story_id, jan)
    moved = (
        await memory.supersede("acme", show.story_id, "The show moved online.", valid_from=feb)
    ).story
    [closed] = await memory.get_stories("acme", [show.story_id])
    assert (closed.valid_to, closed.superseded_by, closed.closed_at) == (
        feb,
        moved.story_id,
        moved.created_at,
    )
    for name in ("promo", "fair", "sale", "show"):
        believed, _ = await memory.list_stories(
            "acme", subject=name, known_at=datetime(2026, 4, 1, tzinfo=UTC), history=True
        )
        assert [(s.story_id, s.valid_to, s.closed_at) for s in believed] == [
            (told[name].story_id, march, None)
        ]
    late = (
        await memory.remember(
            "acme", "The late fair runs.", subject="late", valid_from=jan, valid_to=march
        )
    ).story
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{late.story_id}' ends at 2026-03-31T00:00:00\+00:00, before valid_from; pass an earlier valid_from$",
    ):
        await memory.supersede("acme", late.story_id, "The late fair moved.")
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{late.story_id}' already ended, at 2026-03-31T00:00:00\+00:00$",
    ):
        await memory.end_story("acme", late.story_id)
    assert await memory.get_stories("acme", [late.story_id]) == [late]


async def test_an_end_never_leaves_a_story_true_beside_one_that_says_the_same(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = await _memory(storage)
    jan, march = datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 3, 31, tzinfo=UTC)
    promo = (
        await memory.remember(
            "acme", "The promotion runs.", subject="promo", valid_from=jan, valid_to=march
        )
    ).story
    clock[0] = datetime(2026, 4, 10, tzinfo=UTC)
    again = (await memory.remember("acme", "The promotion runs.", subject="promo")).story
    assert again.story_id != promo.story_id and again.valid_from == clock[0]
    with pytest.raises(
        InputError,
        match=rf"^postgres: an end at 2026-06-01T00:00:00\+00:00 would keep story '{promo.story_id}' true beside story '{again.story_id}', which says the same from 2026-04-10T00:00:00\+00:00; end it no later than that$",
    ):
        await memory.end_story("acme", promo.story_id, valid_to=datetime(2026, 6, 1, tzinfo=UTC))
    assert await memory.get_stories("acme", [promo.story_id]) == [promo]
    longer = await memory.end_story("acme", promo.story_id, valid_to=again.valid_from)
    assert longer.valid_to == again.valid_from
    plan = (
        await memory.remember(
            "acme", "The plan holds.", subject="plan", valid_from=jan, valid_to=march
        )
    ).story
    early = (
        await memory.remember(
            "acme",
            "The plan holds.",
            subject="plan",
            valid_from=datetime(2026, 1, 1, tzinfo=UTC),
            valid_to=datetime(2026, 2, 1, tzinfo=UTC),
        )
    ).story
    held = (await memory.remember("acme", "The plan holds.", subject="plan")).story
    assert len({plan.story_id, early.story_id, held.story_id}) == 3
    await memory.end_story("acme", early.story_id, valid_to=datetime(2026, 3, 1, tzinfo=UTC))
    with pytest.raises(
        InputError,
        match=rf"^postgres: an end at 2026-05-01T00:00:00\+00:00 would keep story '{plan.story_id}' true beside story '{held.story_id}', which says the same from 2026-04-10T00:00:00\+00:00; end it no later than that$",
    ):
        await memory.end_story("acme", plan.story_id, valid_to=datetime(2026, 5, 1, tzinfo=UTC))
    clock[0] = datetime(2026, 4, 12, tzinfo=UTC)
    pending = (
        await memory.remember(
            "acme", "The fair opens.", subject="fair", valid_from=datetime(2026, 5, 1, tzinfo=UTC)
        )
    ).story
    fair, other = [
        (
            await memory.remember(
                "acme", "The fair opens.", subject="fair", valid_from=start, valid_to=march
            )
        ).story
        for start in (jan, datetime(2026, 1, 6, tzinfo=UTC))
    ]
    assert len({fair.story_id, other.story_id, pending.story_id}) == 3
    met = await memory.end_story("acme", fair.story_id, valid_to=pending.valid_from)
    assert met.valid_to == pending.valid_from
    with pytest.raises(
        InputError,
        match=rf"^postgres: an end at 2026-05-02T00:00:00\+00:00 would keep story '{other.story_id}' true beside story '{fair.story_id}', which says the same from 2026-01-05T00:00:00\+00:00; give an end that has come$",
    ):
        await memory.end_story("acme", other.story_id, valid_to=datetime(2026, 5, 2, tzinfo=UTC))


async def test_an_end_behind_the_clock_of_its_story_records_a_later_moment(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _memory(storage)
    jan = datetime(2026, 1, 10, tzinfo=UTC)
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 1, 20, tzinfo=UTC))
    told = (await memory.remember("acme", "Ana owns the review.", valid_from=jan)).story
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: datetime(2026, 1, 15, tzinfo=UTC))
    ended = await memory.end_story("acme", told.story_id)
    assert ended.valid_to == datetime(2026, 1, 15, tzinfo=UTC)
    assert ended.closed_at == told.created_at + timedelta(microseconds=1)


async def test_a_remember_told_before_a_later_end_returns_the_story_it_ended(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = await _memory(storage)
    jan = datetime(2026, 1, 5, tzinfo=UTC)
    told = (await memory.remember("acme", "Ana owns the review.", valid_from=jan)).story
    clock[0] = datetime(2026, 1, 20, tzinfo=UTC)
    ended = await memory.end_story("acme", told.story_id)
    clock[0] = datetime(2026, 1, 15, tzinfo=UTC)
    repeat = await memory.remember("acme", "Ana owns the review.")
    assert repeat.story.story_id == ended.story_id and repeat.story.valid_to == ended.valid_to
    clock[0] = datetime(2026, 1, 25, tzinfo=UTC)
    again = (await memory.remember("acme", "Ana owns the review.")).story
    assert again.story_id != told.story_id and again.valid_to is None


async def test_a_retraction_records_that_a_story_was_never_true(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = await _memory(storage, pillars={"text", "vector", "graph"})
    jan = datetime(2026, 1, 5, tzinfo=UTC)
    told = (
        await memory.remember(
            "acme",
            "Ana prefers the Tuesday slot.",
            subject="user:ana",
            valid_from=jan,
            entities=[EntityIn(name="Ana", type="person"), EntityIn(name="Tuesday", type="slot")],
            relations=[RelationIn(from_name="Ana", to_name="Tuesday", type="PREFERS")],
        )
    ).story
    clock[0] = datetime(2026, 2, 10, tzinfo=UTC)
    retracted = await memory.end_story("acme", told.story_id, valid_to=told.valid_from)
    assert retracted.valid_to == retracted.valid_from == jan
    assert (retracted.closed_at, retracted.superseded_by) == (clock[0], None)
    for moment in (jan, datetime(2026, 1, 20, tzinfo=UTC), None):
        assert (await memory.list_stories("acme", subject="user:ana", as_of=moment))[0] == []
        assert (await memory.recall("acme", "Tuesday slot", as_of=moment)).hits == []
        hood = await memory.neighbors("acme", "Ana", as_of=moment)
        assert (hood.relations, hood.passages) == ([], [])
    assert (await memory.list_stories("acme", subject="user:ana", history=True))[0] == [retracted]
    believed, _ = await memory.list_stories(
        "acme", subject="user:ana", known_at=datetime(2026, 2, 1, tzinfo=UTC)
    )
    assert [(s.story_id, s.valid_to, s.closed_at) for s in believed] == [
        (told.story_id, None, None)
    ]
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{told.story_id}' was retracted at 2026-02-10T00:00:00\+00:00$",
    ):
        await memory.end_story("acme", told.story_id, valid_to=told.valid_from)
    with pytest.raises(
        InputError,
        match=rf"^memory: story '{told.story_id}' was retracted at 2026-02-10T00:00:00\+00:00; remember a new story instead$",
    ):
        await memory.supersede("acme", told.story_id, "Ana prefers Thursday.")
    again = (
        await memory.remember("acme", "Ana prefers the Tuesday slot.", subject="user:ana")
    ).story
    assert again.story_id != told.story_id


async def test_a_story_that_has_not_begun_can_be_retracted_and_no_longer_counts(
    storage: PostgresStorage,
) -> None:
    memory = await _memory(storage)
    begins = datetime.now(UTC) + timedelta(days=10)
    planned = (await memory.remember("acme", "The new line opens.", valid_from=begins)).story
    retracted = await memory.end_story("acme", planned.story_id, valid_to=planned.valid_from)
    assert retracted.valid_to == retracted.valid_from == begins
    with pytest.raises(InputError, match=rf"^memory: story '{planned.story_id}' was retracted at "):
        await memory.supersede(
            "acme", planned.story_id, "The new line opens later.", valid_from=begins
        )
    near = await memory.remember("acme", "The new line opens soon.")
    assert planned.story_id not in [h.id for h in near.related.hits]
    again = (await memory.remember("acme", "The new line opens.", valid_from=begins)).story
    assert again.story_id != planned.story_id and again.valid_to is None


async def test_a_correction_from_the_start_replaces_a_story_that_was_never_true(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 10, tzinfo=UTC)]
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = await _memory(storage)
    jan5, jan8 = datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 8, tzinfo=UTC)
    told = (
        await memory.remember("acme", "Ana prefers Tuesday.", subject="user:ana", valid_from=jan5)
    ).story
    clock[0] = datetime(2026, 2, 10, tzinfo=UTC)
    fixed = (
        await memory.supersede(
            "acme", told.story_id, "Ana prefers Thursday.", valid_from=told.valid_from
        )
    ).story
    [old] = await memory.get_stories("acme", [told.story_id])
    assert old.valid_to == old.valid_from == fixed.valid_from == jan5
    assert (old.superseded_by, old.closed_at) == (fixed.story_id, clock[0])
    then, _ = await memory.list_stories(
        "acme", subject="user:ana", as_of=datetime(2026, 1, 20, tzinfo=UTC)
    )
    assert [s.text for s in then] == ["Ana prefers Thursday."]
    believed, _ = await memory.list_stories(
        "acme", subject="user:ana", known_at=datetime(2026, 2, 1, tzinfo=UTC)
    )
    assert [(s.text, s.valid_to) for s in believed] == [("Ana prefers Tuesday.", None)]
    every, _ = await memory.list_stories("acme", subject="user:ana", history=True)
    assert {s.story_id for s in every} == {told.story_id, fixed.story_id}
    mornings = (await memory.remember("acme", "Bo works mornings.", valid_from=jan5)).story
    mondays = (await memory.remember("acme", "Bo works Mondays.", valid_from=jan8)).story
    merged = (
        await memory.supersede(
            "acme",
            [mornings.story_id, mondays.story_id],
            "Bo works Monday mornings.",
            valid_from=jan8,
        )
    ).story
    closed = await memory.get_stories("acme", [mornings.story_id, mondays.story_id])
    assert [(s.valid_from, s.valid_to) for s in closed] == [(jan5, jan8), (jan8, jan8)]
    assert {s.superseded_by for s in closed} == {merged.story_id}
    march = datetime(2026, 3, 1, tzinfo=UTC)
    promo = (
        await memory.remember(
            "acme", "The promotion runs.", subject="promo", valid_from=jan5, valid_to=march
        )
    ).story
    clock[0] = datetime(2026, 2, 20, tzinfo=UTC)
    await memory.supersede(
        "acme", promo.story_id, "The promotion never ran.", valid_from=promo.valid_from
    )
    believed, _ = await memory.list_stories(
        "acme", subject="promo", known_at=datetime(2026, 2, 15, tzinfo=UTC)
    )
    assert [(s.text, s.valid_to) for s in believed] == [("The promotion runs.", march)]
    learned, _ = await memory.list_stories("acme", subject="promo", known_at=clock[0], history=True)
    assert {(s.text, s.valid_to == s.valid_from) for s in learned} == {
        ("The promotion runs.", True),
        ("The promotion never ran.", False),
    }
