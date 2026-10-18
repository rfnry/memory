import re
from datetime import UTC, datetime, timedelta, timezone
from itertools import count

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import Chunking, CorpusFingerprint, Section, SourceIn
from rfnry_memory.postgres import PostgresStorage

from .fakes import HashEmbedding


def _renamed(model: str) -> HashEmbedding:
    embedding = HashEmbedding()
    embedding.model = model
    return embedding


MANUAL = "# Seals\n\nReplace the pump seal yearly.\n\n# Wiring\n\nGround the panel first."
SHEET = [
    Section(title="Feed line", text="V-101 feeds P-3.", locator={"sheet": 7}),
    Section(title="Return line", text="P-3 returns to the tank.", locator={"sheet": 7}),
]


async def _filled(memory: Memory, corpus: str = "acme") -> None:
    await memory.ingest(corpus, name="manual", text=MANUAL, metadata={"line": "A", "rev": 3})
    await memory.ingest(corpus, name="sheet 7", sections=SHEET, metadata={"line": "A"})


def _memory(storage: PostgresStorage, chunk_chars: int = 2000, overlap_chars: int = 200) -> Memory:
    return Memory(
        storage,
        embedding=HashEmbedding(),
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )


async def test_a_fingerprint_counts_the_corpus_and_names_how_it_is_indexed(
    storage: PostgresStorage,
) -> None:
    memory = _memory(storage, chunk_chars=40, overlap_chars=5)
    await _filled(memory)
    await memory.remember("acme", "Ana owns the seal plan.", subject="user:ana")
    fp = await memory.fingerprint("acme")
    assert isinstance(fp, CorpusFingerprint)
    counts = await memory.verify_absent("acme")
    assert (fp.sources, fp.chunks, fp.stories) == (2, counts.chunks, 1)
    assert counts.chunks > 2
    assert re.fullmatch(r"[0-9a-f]{16}", fp.content)
    assert re.fullmatch(r"[0-9a-f]{16}", fp.story_content)
    assert fp.embedding == "hash-8"
    assert (fp.embedded, fp.stale) == ({"hash-8": counts.chunks + 1}, 0)
    assert fp.chunking == Chunking(chunk_chars=40, overlap_chars=5, pillars=["text", "vector"])


async def test_a_fingerprint_is_the_same_on_every_call(storage: PostgresStorage) -> None:
    memory = _memory(storage)
    await _filled(memory)
    await memory.remember("acme", "Ana owns the seal plan.")
    assert await memory.fingerprint("acme") == await memory.fingerprint("acme")


async def test_identical_content_under_new_ids_leaves_the_fingerprint_unchanged(
    storage: PostgresStorage,
) -> None:
    memory = _memory(storage)
    await _filled(memory)
    before = await memory.fingerprint("acme")
    sources, _ = await memory.list_sources("acme")
    for source in sources:
        await memory.delete_source("acme", source.source_id)
    await memory.ingest("acme", name="sheet 7", sections=SHEET, metadata={"line": "A"})
    await memory.ingest("acme", name="manual", text=MANUAL, metadata={"rev": 3, "line": "A"})
    again, _ = await memory.list_sources("acme")
    assert {s.source_id for s in again}.isdisjoint({s.source_id for s in sources})
    assert await memory.fingerprint("acme") == before
    await memory.sync(
        "acme",
        filters={"metadata.line": "A"},
        sources=[
            SourceIn(name="manual", text=MANUAL, metadata={"line": "A", "rev": 3}),
            SourceIn(name="sheet 7", sections=SHEET, metadata={"line": "A"}),
        ],
    )
    assert await memory.fingerprint("acme") == before


async def test_a_change_of_text_metadata_or_name_changes_the_content(
    storage: PostgresStorage,
) -> None:
    memory = _memory(storage)
    await _filled(memory)
    before = await memory.fingerprint("acme")
    changes = [
        SourceIn(name="manual", text=MANUAL + " Twice.", metadata={"line": "A", "rev": 3}),
        SourceIn(name="manual", text=MANUAL, metadata={"line": "A", "rev": 4}),
        SourceIn(name="manual", text=MANUAL, metadata={"line": "A", "rev": "3"}),
        SourceIn(name="Manual", text=MANUAL, metadata={"line": "A", "rev": 3}),
    ]
    sheet = SourceIn(name="sheet 7", sections=SHEET, metadata={"line": "A"})
    seen = {before.content}
    for change in changes:
        await memory.sync("acme", filters={"metadata.line": "A"}, sources=[change, sheet])
        after = await memory.fingerprint("acme")
        assert after.content not in seen
        assert (after.stories, after.story_content) == (before.stories, before.story_content)
        seen.add(after.content)


async def test_stories_have_a_hash_of_their_own(storage: PostgresStorage) -> None:
    memory = _memory(storage)
    await _filled(memory)
    empty = await memory.fingerprint("acme")
    story = (
        await memory.remember(
            "acme",
            "The go-live is February 20.",
            subject="project:apollo",
            valid_from=datetime(2026, 1, 10, tzinfo=UTC),
        )
    ).story
    told = await memory.fingerprint("acme")
    assert told.content == empty.content and told.stories == 1
    assert told.story_content != empty.story_content
    await memory.supersede(
        "acme",
        story.story_id,
        "The go-live is March 3.",
        valid_from=datetime(2026, 2, 1, tzinfo=UTC),
    )
    superseded = await memory.fingerprint("acme")
    assert superseded.content == empty.content and superseded.stories == 2
    assert superseded.story_content not in {empty.story_content, told.story_content}


async def test_a_story_is_hashed_by_what_it_says_and_when_not_by_its_id(
    storage: PostgresStorage,
) -> None:
    corpora = count()

    async def told(valid_from: datetime, subject: str = "project:apollo") -> CorpusFingerprint:
        corpus, memory = f"acme-{next(corpora)}", _memory(storage)
        await memory.remember(
            corpus, "The go-live is February 20.", subject=subject, valid_from=valid_from
        )
        return await memory.fingerprint(corpus)

    utc = datetime(2026, 1, 10, 12, tzinfo=UTC)
    same = await told(utc)
    assert (await told(utc)).story_content == same.story_content
    plus_two = utc.astimezone(timezone(timedelta(hours=2)))
    assert (await told(plus_two)).story_content == same.story_content
    assert (await told(utc + timedelta(days=1))).story_content != same.story_content
    assert (await told(utc, subject="project:hermes")).story_content != same.story_content


async def test_the_order_sources_were_written_in_does_not_matter(storage: PostgresStorage) -> None:
    first, second = _memory(storage), _memory(storage)
    await _filled(first)
    await second.ingest("other", name="sheet 7", sections=SHEET, metadata={"line": "A"})
    await second.ingest("other", name="manual", text=MANUAL, metadata={"rev": 3, "line": "A"})
    assert await first.fingerprint("acme") == await second.fingerprint("other")


async def test_another_corpus_neither_moves_nor_joins_the_fingerprint(
    storage: PostgresStorage,
) -> None:
    memory = _memory(storage)
    await _filled(memory)
    before = await memory.fingerprint("acme")
    await memory.ingest("globex", name="handbook", text="Boilers are inspected every spring.")
    await memory.remember("globex", "Bo runs the spring inspection.")
    assert await memory.fingerprint("acme") == before
    other = await memory.fingerprint("globex")
    assert (other.sources, other.stories) == (1, 1)
    assert other.content != before.content
    await _filled(memory, "initech")
    assert (await memory.fingerprint("initech")).content == before.content


async def test_an_empty_corpus_and_a_memory_without_vectors_have_fingerprints_too(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage)
    fp = await memory.fingerprint("acme")
    assert (fp.sources, fp.chunks, fp.stories, fp.embedding) == (0, 0, 0, None)
    assert (fp.embedded, fp.stale) == ({}, 0)
    assert fp.chunking == Chunking(chunk_chars=2000, overlap_chars=200, pillars=["text"])
    assert fp.content == fp.story_content
    with pytest.raises(InputError, match="^memory: corpus_id is required"):
        await memory.fingerprint(" ")


async def test_vectors_of_another_model_are_stale_until_reembed(storage: PostgresStorage) -> None:
    old = Memory(storage, embedding=HashEmbedding(), chunk_chars=40, overlap_chars=5)
    await _filled(old)
    await old.remember("acme", "Ana owns the seal plan.")
    clean = await old.fingerprint("acme")
    rows = clean.chunks + clean.stories
    assert (clean.embedded, clean.stale) == ({"hash-8": rows}, 0)
    new = Memory(storage, embedding=_renamed("hash-8-v2"), chunk_chars=40, overlap_chars=5)
    moved = await new.fingerprint("acme")
    assert (moved.embedding, moved.embedded, moved.stale) == ("hash-8-v2", {"hash-8": rows}, rows)
    assert (moved.content, moved.story_content) == (clean.content, clean.story_content)
    await new.remember("acme", "Bo owns the wiring plan.")
    half = await new.fingerprint("acme")
    assert (half.embedded, half.stale) == ({"hash-8": rows, "hash-8-v2": 1}, rows)
    assert (await old.fingerprint("acme")).stale == 1
    assert await new.reembed("acme") == half.stale
    done = await new.fingerprint("acme")
    assert (done.embedded, done.stale) == ({"hash-8-v2": rows + 1}, 0)
    assert (await old.fingerprint("acme")).stale == rows + 1


async def test_passages_without_a_vector_are_stale_while_the_vector_pillar_is_on(
    storage: PostgresStorage,
) -> None:
    await storage.check(dimension=8, graph=True)
    text = Memory(storage, embedding=HashEmbedding(), pillars={"text"})
    await _filled(text)
    await text.remember("acme", "Ana owns the seal plan.")
    unembedded = await text.fingerprint("acme")
    rows = unembedded.chunks + unembedded.stories
    assert (unembedded.embedding, unembedded.embedded, unembedded.stale) == (None, {}, 0)
    vector = Memory(storage, embedding=HashEmbedding())
    missing = await vector.fingerprint("acme")
    assert (missing.embedding, missing.embedded, missing.stale) == ("hash-8", {}, rows)
    assert await vector.reembed("acme") == rows
    embedded = await vector.fingerprint("acme")
    assert (embedded.embedded, embedded.stale) == ({"hash-8": rows}, 0)
    assert (await text.fingerprint("acme")).stale == 0


async def test_without_the_vector_pillar_nothing_is_stale_whatever_is_stored(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage)
    await _filled(memory)
    await memory.remember("acme", "Ana owns the seal plan.")
    fp = await memory.fingerprint("acme")
    assert (fp.embedding, fp.embedded, fp.stale) == (None, {}, 0)
    await _filled(Memory(storage, embedding=HashEmbedding()), "other")
    rows = (await storage.count_corpus("other")).chunks
    text = await Memory(storage).fingerprint("other")
    assert (text.embedding, text.embedded, text.stale) == (None, {"hash-8": rows}, 0)


async def test_retired_counts_retired_versions_even_while_a_write_lands(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", True)
    await memory.ingest("acme", name="manual", text="Replace the seal yearly.")
    digests = storage.source_digests
    landed = iter(range(10))

    async def racing(corpus_id: str) -> list[tuple[str, dict[str, object], str, int]]:
        found = await digests(corpus_id)
        await memory.ingest("acme", name=f"notes {next(landed)}", text="Ground the panel.")
        return found

    monkeypatch.setattr(storage, "source_digests", racing)
    assert (await memory.fingerprint("acme")).retired == 0
    assert (await memory.describe("acme")).retired == 0
