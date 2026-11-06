import asyncio
from typing import Any

import pytest

from rfnry_memory import Chunk, Embedded, Event, Section, Staged, Story
from rfnry_memory.errors import ConfigurationError, InputError, ModelError
from rfnry_memory.memory import Memory
from rfnry_memory.postgres import PostgresStorage

from .fakes import DocumentHashEmbedding, HashEmbedding


class Wide(HashEmbedding):
    model, dimension = "wide-16", 16

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.5] * 16 for _ in texts]


class Picky(Wide):
    def __init__(self, *refused: str) -> None:
        super().__init__()
        self.refused = refused

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        if any(text in self.refused for text in texts):
            self.calls.append(list(texts))
            raise RuntimeError("vendor: input too long")
        return await super().embed(texts, purpose=purpose)


class Down(Wide):
    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        self.calls.append(list(texts))
        raise RuntimeError("vendor: unavailable")


class WideDocument(DocumentHashEmbedding):
    model = "wide-document-8"

    def __init__(self, *refused: str) -> None:
        super().__init__()
        self.refused = refused

    async def embed_in_document(self, texts: list[str]) -> list[list[float]]:
        if any(text in self.refused for text in texts):
            self.documents.append(list(texts))
            raise RuntimeError("vendor: document too long")
        return await super().embed_in_document(texts)


class Halt(Exception):
    pass


class Kept:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)


async def _old(storage: PostgresStorage) -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    return memory


def _new(storage: PostgresStorage, embedding: Any = None) -> Memory:
    return Memory(storage, embedding=embedding or Wide(), pillars={"vector"})


def _sent(embedding: HashEmbedding) -> list[str]:
    return [text for call in embedding.calls for text in call]


async def _staged_rows(storage: PostgresStorage, corpus_id: str) -> list[tuple[str, str]]:
    async with storage._pool.acquire() as conn:
        return [
            (r["passage_kind"], r["passage_id"])
            for r in await conn.fetch(
                "SELECT passage_kind, passage_id FROM memory_staged_vectors WHERE corpus_id = $1 "
                "ORDER BY passage_kind, passage_id",
                corpus_id,
            )
        ]


async def test_staging_leaves_serving_as_it_was(storage: PostgresStorage) -> None:
    old = Memory(storage, embedding=HashEmbedding())
    await old.check()
    await old.ingest("acme", name="n", text="pump seal")
    done = await Memory(storage, embedding=Wide(), pillars={"vector"}).stage_vectors("acme")
    assert done == Staged(embedded=1)
    assert (await old.search("acme", "pump seal", pillars={"vector"})).hits
    assert (await old.fingerprint("acme")).staged == {"wide-16": 1}


async def test_stage_needs_no_check_and_takes_another_dimension(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="n", text="pump seal")
    await old.remember("acme", "The pump runs.")
    fresh = PostgresStorage(pool)
    assert await _new(fresh).stage_vectors("acme") == Staged(embedded=2)
    assert fresh._dimension is None
    async with pool.acquire() as conn:
        assert await conn.fetchval(
            "SELECT array_agg(DISTINCT vector_dims(vector)) FROM memory_staged_vectors"
        ) == [16]


async def test_staging_again_embeds_only_what_is_missing(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="a", text="pump seal")
    wide = Wide()
    memory = _new(storage, wide)
    assert await memory.stage_vectors("acme") == Staged(embedded=1)
    assert await memory.stage_vectors("acme") == Staged(embedded=0)
    await old.ingest("acme", name="b", text="valve stem")
    await old.remember("acme", "The pump runs.")
    assert await memory.stage_vectors("acme") == Staged(embedded=2)
    assert wide.calls == [["pump seal"], ["valve stem", "The pump runs."]]


async def test_a_stopped_stage_keeps_what_it_wrote(storage: PostgresStorage) -> None:
    old = await _old(storage)
    for name, text in (("a", "pump seal"), ("b", "valve stem"), ("c", "panel wiring")):
        await old.ingest("acme", name=name, text=text)
    await old.remember("acme", "The pump runs.")

    async def stop(event: Event) -> None:
        raise Halt

    wide = Wide()
    memory = _new(storage, wide)
    with pytest.raises(Halt):
        await memory.stage_vectors("acme", batch=1, on_event=stop)
    assert (await old.fingerprint("acme")).staged == {"wide-16": 1}
    assert await memory.stage_vectors("acme", batch=1) == Staged(embedded=3)
    assert (await old.fingerprint("acme")).staged == {"wide-16": 4}
    assert sorted(_sent(wide)) == sorted(
        ["pump seal", "valve stem", "panel wiring", "The pump runs."]
    )


async def test_a_cancelled_stage_is_no_refusal_and_keeps_what_it_wrote(
    storage: PostgresStorage,
) -> None:
    class Cancelled(Wide):
        async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
            if self.calls:
                raise asyncio.CancelledError
            return await super().embed(texts, purpose=purpose)

    old = await _old(storage)
    await old.ingest("acme", name="a", text="pump seal")
    await old.ingest("acme", name="b", text="valve stem")
    with pytest.raises(asyncio.CancelledError):
        await _new(storage, Cancelled()).stage_vectors("acme", batch=1)
    assert (await old.fingerprint("acme")).staged == {"wide-16": 1}
    assert await _new(storage).stage_vectors("acme") == Staged(embedded=1)


async def test_filters_choose_what_goes_first_and_limit_caps_a_run(
    storage: PostgresStorage,
) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="a", text="pump seal", metadata={"line": "A"})
    await old.ingest("acme", name="b", text="valve stem", metadata={"line": "B"})
    await old.remember("acme", "Line A runs.", metadata={"line": "A"})
    await old.remember("acme", "Line B runs.", metadata={"line": "B"})
    wide = Wide()
    memory = _new(storage, wide)
    first = await memory.stage_vectors("acme", filters={"metadata.line": "A"})
    assert first == Staged(embedded=2)
    assert sorted(_sent(wide)) == ["Line A runs.", "pump seal"]
    assert await memory.stage_vectors("acme", limit=1) == Staged(embedded=1)
    assert await memory.stage_vectors("acme", limit=5, batch=1) == Staged(embedded=1)
    assert len(_sent(wide)) == 4


async def test_a_limit_counts_what_the_model_refused(storage: PostgresStorage) -> None:
    old = await _old(storage)
    for name, text in (("a", "first"), ("b", "second"), ("c", "third")):
        await old.ingest("acme", name=name, text=text)
    picky = Picky("first", "second", "third")
    done = await _new(storage, picky).stage_vectors("acme", limit=2, batch=1)
    assert (done.embedded, len(done.refused)) == (0, 2)
    assert len(picky.calls) == 2


async def test_a_refused_passage_does_not_block_the_rest(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="a", text="pump seal")
    refused = await old.ingest("acme", name="b", text="poison")
    await old.remember("acme", "The pump runs.")
    [chunk] = await storage.source_chunks("acme", refused.source.source_id)
    picky = Picky("poison")
    done = await _new(storage, picky).stage_vectors("acme")
    assert done == Staged(embedded=2, refused=[("chunk", chunk.chunk_id)])
    assert (await old.fingerprint("acme")).staged == {"wide-16": 2}
    assert ("chunk", chunk.chunk_id) not in await _staged_rows(storage, "acme")
    assert _sent(picky).count("poison") == 2
    again = await _new(storage, picky).stage_vectors("acme")
    assert again == Staged(embedded=0, refused=[("chunk", chunk.chunk_id)])


async def test_a_refused_story_is_named_as_a_story(storage: PostgresStorage) -> None:
    old = await _old(storage)
    story = (await old.remember("acme", "poison")).story
    done = await _new(storage, Picky("poison")).stage_vectors("acme")
    assert done == Staged(embedded=0, refused=[("story", story.story_id)])


async def test_a_batch_that_fails_whole_is_an_outage(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="a", text="pump seal")
    await old.ingest("acme", name="b", text="valve stem")
    down = Down()
    with pytest.raises(RuntimeError, match="^vendor: unavailable$"):
        await _new(storage, down).stage_vectors("acme")
    assert len(down.calls) == 3
    assert (await old.fingerprint("acme")).staged == {}


async def test_retired_chunks_and_closed_stories_are_staged(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.set_history("acme", True)
    first = await old.ingest("acme", name="manual", text="Replace the seal yearly.")
    await old.ingest(
        "acme", name="manual", text="Replace the seal monthly.", replaces=first.source.source_id
    )
    story = (await old.remember("acme", "The pump is new.")).story
    await old.supersede("acme", story.story_id, "The pump is worn.")
    wide = Wide()
    assert await _new(storage, wide).stage_vectors("acme") == Staged(embedded=4)
    assert sorted(_sent(wide)) == [
        "Replace the seal monthly.",
        "Replace the seal yearly.",
        "The pump is new.",
        "The pump is worn.",
    ]


async def test_a_memory_whose_model_owns_the_rows_has_nothing_to_stage(
    storage: PostgresStorage,
) -> None:
    embedding = HashEmbedding()
    memory = Memory(storage, embedding=embedding)
    await memory.check()
    await memory.ingest("acme", name="n", text="pump seal")
    await memory.remember("acme", "The pump runs.")
    embedding.calls.clear()
    assert await memory.stage_vectors("acme") == Staged(embedded=0)
    assert embedding.calls == []
    assert (await memory.fingerprint("acme")).staged == {}


async def test_a_passage_with_no_vector_at_all_is_staged(storage: PostgresStorage) -> None:
    await Memory(storage, pillars={"text"}).ingest("acme", name="n", text="pump seal")
    assert await _new(storage).stage_vectors("acme") == Staged(embedded=1)


async def test_a_document_model_stages_a_source_in_one_call(storage: PostgresStorage) -> None:
    old = await _old(storage)
    sections = [Section(text="One."), Section(text="Two."), Section(text="Three.")]
    await old.ingest("acme", name="whole", sections=sections)
    await old.remember("acme", "The pump runs.")
    aware = DocumentHashEmbedding()
    kept = Kept()
    done = await _new(storage, aware).stage_vectors("acme", batch=1, on_event=kept)
    assert done == Staged(embedded=4)
    assert aware.documents == [["One.", "Two.", "Three."]]
    assert aware.calls == [["The pump runs."]]
    assert [
        (e.name, e.done, e.total, e.computed) for e in kept.events if isinstance(e, Embedded)
    ] == [("whole", 3, 4, 3), (None, 4, 4, 1)]


async def test_a_limit_lets_a_document_model_finish_its_document_and_start_no_other(
    storage: PostgresStorage,
) -> None:
    old = await _old(storage)
    for name in ("a", "b"):
        sections = [Section(text=f"{name} one."), Section(text=f"{name} two.")]
        await old.ingest("acme", name=name, sections=sections)
    await old.remember("acme", "The pump runs.")
    aware = DocumentHashEmbedding()
    assert await _new(storage, aware).stage_vectors("acme", limit=1) == Staged(embedded=2)
    assert len(aware.documents) == 1
    assert aware.calls == []
    assert (await old.fingerprint("acme")).staged == {"hash-document-8": 2}


async def test_a_refused_source_refuses_its_chunks_and_is_tried_once(
    storage: PostgresStorage,
) -> None:
    old = await _old(storage)
    refused = await old.ingest(
        "acme",
        name="refused",
        sections=[Section(text="poison"), Section(text="Two."), Section(text="Three.")],
    )
    await old.ingest("acme", name="fine", sections=[Section(text="Fine.")])
    chunks = await storage.source_chunks("acme", refused.source.source_id)
    aware = WideDocument("poison")
    done = await _new(storage, aware).stage_vectors("acme", batch=1)
    assert done.embedded == 1
    assert sorted(done.refused) == sorted(("chunk", c.chunk_id) for c in chunks)
    assert len(aware.documents) == 2


async def test_one_large_source_refused_is_no_outage(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="large", sections=[Section(text="poison"), Section(text="Two.")])
    done = await _new(storage, WideDocument("poison")).stage_vectors("acme")
    assert (done.embedded, len(done.refused)) == (0, 2)


async def test_every_source_of_a_page_failing_is_an_outage(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="a", sections=[Section(text="poison")])
    await old.ingest("acme", name="b", sections=[Section(text="poison")])
    with pytest.raises(RuntimeError, match="^vendor: document too long$"):
        await _new(storage, WideDocument("poison")).stage_vectors("acme")


async def test_a_vector_of_the_wrong_length_is_a_model_error(storage: PostgresStorage) -> None:
    class Short(Wide):
        async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
            return [[0.5] * 3 for _ in texts]

    old = await _old(storage)
    await old.ingest("acme", name="n", text="pump seal")
    with pytest.raises(
        ModelError,
        match=r"^memory: the embedding model returned a vector of 3 dimensions; it declares 16$",
    ):
        await _new(storage, Short()).stage_vectors("acme")
    assert (await old.fingerprint("acme")).staged == {}


async def test_a_wrong_number_of_vectors_is_a_model_error_for_a_document_too(
    storage: PostgresStorage,
) -> None:
    class Short(DocumentHashEmbedding):
        async def embed_in_document(self, texts: list[str]) -> list[list[float]]:
            return (await super().embed_in_document(texts))[:-1]

    old = await _old(storage)
    await old.ingest("acme", name="n", sections=[Section(text="One."), Section(text="Two.")])
    with pytest.raises(
        ModelError, match=r"^memory: the embedding model returned 1 vectors for 2 texts$"
    ):
        await _new(storage, Short()).stage_vectors("acme")


async def test_staged_vectors_stay_in_their_corpus(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="n", text="pump seal")
    await old.ingest("globex", name="n", text="pump seal")
    wide = Wide()
    memory = _new(storage, wide)
    assert await memory.stage_vectors("acme") == Staged(embedded=1)
    assert (await old.fingerprint("globex")).staged == {}
    assert await _staged_rows(storage, "globex") == []
    assert await memory.stage_vectors("globex") == Staged(embedded=1)
    assert wide.calls == [["pump seal"], ["pump seal"]]
    await old.delete_corpus("acme")
    assert (await old.fingerprint("globex")).staged == {"wide-16": 1}


async def test_a_passage_deleted_while_it_embeds_leaves_no_staged_vector(
    storage: PostgresStorage,
) -> None:
    old = await _old(storage)
    written = await old.ingest("acme", name="n", text="pump seal")
    kept = await old.remember("acme", "The pump runs.")

    class Deleting(Wide):
        async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
            await old.delete_source("acme", written.source.source_id)
            await old.delete_story("acme", kept.story.story_id)
            return await super().embed(texts, purpose=purpose)

    await _new(storage, Deleting()).stage_vectors("acme")
    assert await _staged_rows(storage, "acme") == []


async def test_stage_reports_its_batches_up_to_what_is_missing(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="a", text="pump seal")
    await old.ingest("acme", name="b", text="valve stem")
    await old.remember("acme", "The pump runs.")
    await _new(storage).stage_vectors("acme", limit=1)
    kept = Kept()
    await _new(storage).stage_vectors("acme", batch=1, on_event=kept)
    assert [
        (e.name, e.done, e.total, e.computed) for e in kept.events if isinstance(e, Embedded)
    ] == [(None, 1, 2, 1), (None, 2, 2, 1)]


async def test_a_filtered_or_limited_stage_counts_only_what_it_will_take(
    storage: PostgresStorage,
) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="a", text="pump seal", metadata={"line": "A"})
    await old.ingest("acme", name="b", text="valve stem", metadata={"line": "B"})
    await old.ingest("acme", name="c", text="panel wiring", metadata={"line": "B"})
    kept = Kept()
    await _new(storage).stage_vectors("acme", filters={"metadata.line": "A"}, on_event=kept)
    await _new(storage).stage_vectors("acme", limit=1, on_event=kept)
    assert [(e.done, e.total) for e in kept.events if isinstance(e, Embedded)] == [(1, 1), (1, 1)]


async def test_a_page_the_model_refused_sends_no_event(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="a", text="poison")
    kept = Kept()
    await _new(storage, Picky("poison")).stage_vectors("acme", on_event=kept)
    assert kept.events == []


async def test_without_a_handler_a_stage_counts_nothing(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def uncounted(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("stage_vectors counted its total without a handler")

    old = await _old(storage)
    await old.ingest("acme", name="a", text="pump seal")
    monkeypatch.setattr(storage, "count_unstaged", uncounted)
    assert await _new(storage).stage_vectors("acme") == Staged(embedded=1)


async def test_stage_needs_the_vector_pillar(storage: PostgresStorage) -> None:
    with pytest.raises(ConfigurationError, match="^memory: stage_vectors needs the vector pillar$"):
        await Memory(storage, pillars={"text"}).stage_vectors("acme")


@pytest.mark.parametrize(
    ("given", "message"),
    [
        ({"limit": 0}, "^memory: limit must be at least 1$"),
        ({"batch": 0}, "^memory: batch must be at least 1$"),
        ({"filters": {"line": "A"}}, "must address a facet"),
    ],
)
async def test_stage_and_reembed_refuse_what_they_cannot_page(
    storage: PostgresStorage, given: dict[str, Any], message: str
) -> None:
    memory = _new(storage)
    with pytest.raises(InputError, match=message):
        await memory.stage_vectors("acme", **given)
    with pytest.raises(InputError, match=message):
        await memory.reembed("acme", **given)


async def test_reembed_takes_filters_and_a_limit(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="a", text="pump seal", metadata={"line": "A"})
    await old.ingest("acme", name="b", text="valve stem", metadata={"line": "B"})
    await old.ingest("acme", name="c", text="panel wiring", metadata={"line": "B"})
    await old.remember("acme", "Line A runs.", metadata={"line": "A"})
    renamed = HashEmbedding()
    renamed.model = "hash-8-v2"
    memory = Memory(storage, embedding=renamed)
    assert await memory.reembed("acme", filters={"metadata.line": "A"}) == 2
    assert (await memory.fingerprint("acme")).embedded == {"hash-8": 2, "hash-8-v2": 2}
    assert await memory.reembed("acme", limit=1) == 1
    assert (await memory.fingerprint("acme")).embedded == {"hash-8": 1, "hash-8-v2": 3}
    assert await memory.reembed("acme", limit=9, batch=1) == 1
    assert await memory.reembed("acme") == 0


async def test_reembed_pages_forward_rather_than_rereading_what_it_wrote(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = await _old(storage)
    await old.ingest(
        "acme", name="many", sections=[Section(text=f"Line {i} runs.") for i in range(5)]
    )
    await old.remember("acme", "The pump runs.")
    await old.remember("acme", "The valve leaks.")
    read = storage.passages_embedded_with
    pages: list[list[tuple[str, str]]] = []

    async def counted(corpus_id: str, **kwargs: Any) -> list[Chunk | Story]:
        page = await read(corpus_id, **kwargs)
        pages.append(
            [("chunk", p.chunk_id) if isinstance(p, Chunk) else ("story", p.story_id) for p in page]
        )
        return page

    async def written_elsewhere(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(storage, "passages_embedded_with", counted)
    monkeypatch.setattr(storage, "set_vectors", written_elsewhere)
    renamed = HashEmbedding()
    renamed.model = "hash-8-v2"
    done = await asyncio.wait_for(Memory(storage, embedding=renamed).reembed("acme", batch=2), 10)
    assert done == 7
    seen = [ref for page in pages for ref in page]
    assert len(seen) == len(set(seen)) == 7


async def test_reembed_lets_a_document_model_finish_its_document_and_start_no_other(
    storage: PostgresStorage,
) -> None:
    old = await _old(storage)
    for name in ("a", "b"):
        sections = [Section(text=f"{name} one."), Section(text=f"{name} two.")]
        await old.ingest("acme", name=name, sections=sections)
    await old.remember("acme", "The pump runs.")
    aware = DocumentHashEmbedding()
    memory = Memory(storage, embedding=aware)
    assert await memory.reembed("acme", limit=1) == 2
    assert len(aware.documents) == 1
    assert aware.calls == []


async def test_a_filtered_reembed_counts_only_what_it_will_take(storage: PostgresStorage) -> None:
    old = await _old(storage)
    await old.ingest("acme", name="a", text="pump seal", metadata={"line": "A"})
    await old.ingest("acme", name="b", text="valve stem", metadata={"line": "B"})
    await old.ingest("acme", name="c", text="panel wiring", metadata={"line": "B"})
    renamed = HashEmbedding()
    renamed.model = "hash-8-v2"
    memory = Memory(storage, embedding=renamed)
    kept = Kept()
    await memory.reembed("acme", filters={"metadata.line": "A"}, on_event=kept)
    await memory.reembed("acme", limit=1, on_event=kept)
    assert [(e.done, e.total) for e in kept.events if isinstance(e, Embedded)] == [(1, 1), (1, 1)]
