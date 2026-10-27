import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.memory import Memory
from rfnry_memory.models import EntityIn, RelationIn, Story
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery import MemoryRefinery, Relevance, RetrievalReview

from .fakes import HashEmbedding

US = timedelta(microseconds=1)
STATES = ("open", "planned", "ended", "ahead", "pending", "begun", "retracted")
VERBS = ("supersede", "end", "retract", "delete")


def _on(month: int, day: int = 1) -> datetime:
    return datetime(2026, month, day, tzinfo=UTC)


async def _memory(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch, clock: list[datetime], **kwargs: Any
) -> Memory:
    monkeypatch.setattr("rfnry_memory.memory._now", lambda: clock[0])
    memory = Memory(storage, embedding=HashEmbedding(), **kwargs)
    await memory.check()
    return memory


async def _story(memory: Memory, story_id: str) -> Story:
    [story] = await memory.get_stories("acme", [story_id])
    return story


async def _told(memory: Memory, text: str, subject: str, **kwargs: Any) -> Story:
    return (await memory.remember("acme", text, subject=subject, **kwargs)).story


async def _ids(memory: Memory, subject: str, **when: Any) -> list[str]:
    stories, _ = await memory.list_stories("acme", subject=subject, **when)
    return [s.story_id for s in stories]


async def _in(memory: Memory, state: str, n: int) -> Story:
    planned = _on(1, 8) if state == "planned" else None
    story = await _told(memory, f"Fact {n} holds.", f"s{n}", valid_from=_on(1, 5), valid_to=planned)
    if state in ("ended", "ahead"):
        end = _on(1, 8) if state == "ended" else _on(3, 1)
        await memory.end_story("acme", story.story_id, valid_to=end)
    elif state in ("pending", "begun"):
        start = _on(3, 1) if state == "pending" else _on(1, 8)
        await memory.supersede("acme", story.story_id, f"Fact {n} changed.", valid_from=start)
    elif state == "retracted":
        await memory.end_story("acme", story.story_id, valid_to=story.valid_from)
    return await _story(memory, story.story_id)


async def test_each_story_state_takes_the_closings_its_rules_allow(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [_on(1, 10)]
    memory = await _memory(storage, monkeypatch, clock)
    made: dict[tuple[str, str], Story] = {}
    for state in STATES:
        for verb in VERBS:
            made[state, verb] = await _in(memory, state, len(made))
    clock[0] = _on(1, 20)
    refused: dict[tuple[str, str], str] = {}
    for (state, verb), story in made.items():
        try:
            if verb == "supersede":
                await memory.supersede("acme", story.story_id, "It moved.", valid_from=_on(1, 6))
            elif verb == "end":
                await memory.end_story("acme", story.story_id, valid_to=_on(2, 1))
            elif verb == "retract":
                await memory.end_story("acme", story.story_id, valid_to=story.valid_from)
            else:
                await memory.delete_story("acme", story.story_id)
        except InputError as error:
            refused[state, verb] = str(error)
    assert sorted(refused) == sorted(
        [(state, verb) for state in ("pending", "begun") for verb in ("supersede", "end")]
        + [("retracted", verb) for verb in ("supersede", "end", "retract")]
    )
    for state in ("pending", "begun"):
        story_id = made[state, "supersede"].story_id
        assert refused[state, "supersede"] == (
            f"memory: story {story_id!r} is already superseded; supersede the story that replaced it"
        )
        story_id = made[state, "end"].story_id
        assert refused[state, "end"] == (
            f"memory: story {story_id!r} is already superseded; end the story that replaced it"
        )
    for verb in ("supersede", "end", "retract"):
        story = made["retracted", verb]
        assert story.closed_at is not None
        moment = re.escape(story.closed_at.isoformat())
        tail = "; remember a new story instead" if verb == "supersede" else ""
        assert re.fullmatch(
            rf"memory: story '{story.story_id}' was retracted at {moment}{re.escape(tail)}",
            refused["retracted", verb],
        )
    for state in STATES:
        closed = made[state, "supersede"].closed_at is not None
        if (state, "supersede") not in refused:
            after = await _story(memory, made[state, "supersede"].story_id)
            assert after.valid_to == _on(1, 6) and after.superseded_by is not None
            assert after.revised_at == (after.closed_at if closed else None)
        if (state, "end") not in refused:
            after = await _story(memory, made[state, "end"].story_id)
            assert (after.valid_to, after.superseded_by) == (_on(2, 1), None)
            assert after.revised_at == (after.closed_at if closed else None)
        if (state, "retract") not in refused:
            before = made[state, "retract"]
            after = await _story(memory, before.story_id)
            assert (after.valid_to, after.superseded_by) == (after.valid_from, None)
            if before.superseded_by is not None:
                successor = await _story(memory, before.superseded_by)
                assert successor.valid_to is None and successor.supersedes == (before.story_id,)
        assert await memory.get_stories("acme", [made[state, "delete"].story_id]) == []


async def test_a_retracted_successor_hands_back_what_it_closed(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [_on(1, 10)]
    memory = await _memory(storage, monkeypatch, clock)
    first = await _told(memory, "The go-live is March 3.", "apollo", valid_from=_on(1, 5))
    clock[0] = _on(1, 12)
    pending = (
        await memory.supersede(
            "acme", first.story_id, "The go-live is April 7.", valid_from=_on(3, 1)
        )
    ).story
    clock[0] = _on(1, 20)
    await memory.end_story(
        "acme", pending.story_id, valid_to=pending.valid_from, written_by={"user": "ana"}
    )
    back = await _story(memory, first.story_id)
    assert (back.valid_to, back.superseded_by, back.closed_at, back.closed_by) == (
        None,
        None,
        None,
        None,
    )
    assert back.revised_at == _on(1, 20)
    assert await _ids(memory, "apollo") == [first.story_id]
    assert await _ids(memory, "apollo", as_of=_on(3, 15)) == [first.story_id]
    assert await _ids(memory, "apollo", known_at=_on(1, 15), as_of=_on(3, 15)) == [pending.story_id]
    [then], _ = await memory.list_stories(
        "acme", subject="apollo", known_at=_on(1, 15), as_of=_on(2, 1)
    )
    assert (then.valid_to, then.superseded_by, then.closed_at, then.revised_at) == (
        _on(3, 1),
        pending.story_id,
        _on(1, 12),
        None,
    )
    clock[0] = _on(1, 25)
    moved = (
        await memory.supersede(
            "acme", first.story_id, "The go-live is March 20.", valid_from=_on(3, 20)
        )
    ).story
    dated = await _story(memory, first.story_id)
    assert (dated.valid_to, dated.superseded_by, dated.revised_at) == (
        _on(3, 20),
        moved.story_id,
        _on(1, 20),
    )
    [reopened], _ = await memory.list_stories(
        "acme", subject="apollo", known_at=_on(1, 22), as_of=_on(3, 25)
    )
    assert (reopened.story_id, reopened.valid_to, reopened.revised_at) == (
        first.story_id,
        None,
        _on(1, 20),
    )
    clock[0] = _on(1, 10)
    bo = await _told(memory, "Bo owns the review.", "review", valid_from=_on(1, 5))
    begun = (
        await memory.supersede("acme", bo.story_id, "Cy owns the review.", valid_from=_on(1, 8))
    ).story
    clock[0] = _on(1, 20)
    await memory.end_story("acme", begun.story_id, valid_to=begun.valid_from)
    assert (await _story(memory, bo.story_id)).valid_to is None
    assert await _ids(memory, "review", as_of=_on(1, 9)) == [bo.story_id]
    clock[0] = _on(1, 10)
    tea = await _told(memory, "Gil prefers tea.", "gil", valid_from=_on(1, 5))
    mornings = await _told(memory, "Gil prefers mornings.", "gil", valid_from=_on(1, 5))
    merged = (
        await memory.supersede(
            "acme",
            [tea.story_id, mornings.story_id],
            "Gil prefers tea in the morning.",
            valid_from=_on(1, 8),
        )
    ).story
    clock[0] = _on(1, 20)
    await memory.end_story("acme", merged.story_id, valid_to=merged.valid_from)
    assert sorted(await _ids(memory, "gil")) == sorted([tea.story_id, mornings.story_id])


async def test_a_correction_from_the_start_takes_over_what_its_story_closed(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [_on(1, 10)]
    memory = await _memory(storage, monkeypatch, clock)
    first = await _told(memory, "The launch is May 1.", "launch", valid_from=_on(1, 5))
    clock[0] = _on(1, 12)
    second = (
        await memory.supersede(
            "acme", first.story_id, "The launch is June 1.", valid_from=_on(3, 1)
        )
    ).story
    clock[0] = _on(1, 20)
    third = (
        await memory.supersede(
            "acme",
            second.story_id,
            "The launch is July 1.",
            valid_from=second.valid_from,
            written_by={"user": "bo"},
        )
    ).story
    assert third.supersedes == (second.story_id,)
    handed = await _story(memory, first.story_id)
    assert (handed.valid_to, handed.superseded_by, handed.closed_by) == (
        _on(3, 1),
        third.story_id,
        {"user": "bo"},
    )
    assert handed.closed_at == handed.revised_at == third.created_at
    corrected = await _story(memory, second.story_id)
    assert (corrected.valid_to, corrected.superseded_by) == (_on(3, 1), third.story_id)
    assert await _ids(memory, "launch", as_of=_on(3, 5)) == [third.story_id]
    assert await _ids(memory, "launch", known_at=_on(1, 15), as_of=_on(3, 5)) == [second.story_id]
    [believed], _ = await memory.list_stories(
        "acme", subject="launch", known_at=_on(1, 15), as_of=_on(2, 1)
    )
    assert (believed.story_id, believed.superseded_by) == (first.story_id, second.story_id)
    clock[0] = _on(1, 25)
    await memory.end_story("acme", third.story_id, valid_to=third.valid_from)
    assert sorted(await _ids(memory, "launch", as_of=_on(3, 5))) == sorted(
        [first.story_id, second.story_id]
    )
    clock[0] = _on(1, 10)
    dee = await _told(memory, "Dee leads support.", "support", valid_from=_on(1, 5))
    eve = (
        await memory.supersede("acme", dee.story_id, "Eve leads support.", valid_from=_on(1, 8))
    ).story
    clock[0] = _on(1, 16)
    fay = (
        await memory.supersede("acme", eve.story_id, "Fay leads support.", valid_from=_on(1, 15))
    ).story
    clock[0] = _on(1, 20)
    await memory.end_story("acme", eve.story_id, valid_to=eve.valid_from)
    dropped = await _story(memory, eve.story_id)
    assert (dropped.valid_to, dropped.superseded_by) == (eve.valid_from, None)
    kept = await _story(memory, dee.story_id)
    assert (kept.valid_to, kept.superseded_by, kept.revised_at) == (
        _on(1, 15),
        fay.story_id,
        _on(1, 20),
    )
    assert await _ids(memory, "support", as_of=_on(1, 10)) == [dee.story_id]
    assert await _ids(memory, "support", as_of=_on(1, 16)) == [fay.story_id]
    assert await _ids(memory, "support", known_at=_on(1, 18), as_of=_on(1, 10)) == [eve.story_id]


async def test_a_reopened_story_ends_where_a_story_saying_the_same_begins(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [_on(1, 10)]
    memory = await _memory(storage, monkeypatch, clock)
    first = await _told(memory, "Ana prefers mornings.", "ana", valid_from=_on(1, 5))
    later = (
        await memory.supersede(
            "acme", first.story_id, "Ana prefers evenings.", valid_from=_on(1, 8)
        )
    ).story
    clock[0] = _on(1, 15)
    twin = await _told(memory, "Ana prefers mornings.", "ana", valid_from=_on(2, 1))
    assert twin.story_id != first.story_id
    clock[0] = _on(1, 20)
    await memory.end_story(
        "acme", later.story_id, valid_to=later.valid_from, written_by={"user": "bo"}
    )
    ended = await _story(memory, first.story_id)
    assert (ended.valid_to, ended.superseded_by, ended.closed_at, ended.closed_by) == (
        _on(2, 1),
        None,
        _on(1, 20),
        {"user": "bo"},
    )
    assert await _ids(memory, "ana", as_of=_on(1, 25)) == [first.story_id]
    assert await _ids(memory, "ana", as_of=_on(2, 5)) == [twin.story_id]
    clock[0] = _on(1, 10)
    tea = await _told(memory, "Bo prefers tea.", "bo", valid_from=_on(1, 5))
    coffee = (
        await memory.supersede("acme", tea.story_id, "Bo prefers coffee.", valid_from=_on(1, 8))
    ).story
    early = await _told(memory, "Bo prefers tea.", "bo", valid_from=_on(1, 1))
    clock[0] = _on(1, 20)
    with pytest.raises(
        InputError,
        match=rf"^postgres: retracting story '{coffee.story_id}' would reopen story '{tea.story_id}' beside story '{early.story_id}', which says the same from 2026-01-01T00:00:00\+00:00; end or delete one of them first$",
    ):
        await memory.end_story("acme", coffee.story_id, valid_to=coffee.valid_from)
    assert (await _story(memory, coffee.story_id)).valid_to is None
    assert (await _story(memory, tea.story_id)).superseded_by == coffee.story_id


async def test_known_at_reads_the_closing_memory_held_then_through_every_read(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [_on(1, 10)]
    memory = await _memory(storage, monkeypatch, clock, pillars={"text", "vector", "graph"})
    owned = await _told(
        memory,
        "Ana owns Apollo.",
        "apollo",
        valid_from=_on(1, 5),
        entities=[EntityIn(name="Ana", type="person"), EntityIn(name="Apollo", type="project")],
        relations=[RelationIn(from_name="Ana", to_name="Apollo", type="OWNS")],
    )
    other = await _told(memory, "Apollo belongs to a team.", "team", valid_from=_on(1, 5))
    for moment, end in ((_on(1, 20), _on(2, 1)), (_on(1, 30), _on(1, 25)), (_on(2, 10), _on(3, 1))):
        clock[0] = moment
        await memory.end_story("acme", owned.story_id, valid_to=end)
    clock[0] = _on(3, 5)
    probe = _on(1, 28)
    expected = {
        _on(1, 15): (None, None, None),
        _on(1, 25): (_on(2, 1), _on(1, 20), None),
        _on(2, 5): None,
        _on(2, 15): (_on(3, 1), _on(2, 10), _on(2, 10)),
    }
    for known_at, shown in expected.items():
        when: dict[str, Any] = {"known_at": known_at, "as_of": probe}
        listed, _ = await memory.list_stories("acme", subject="apollo", **when)
        recalled = await memory.recall("acme", "Ana owns Apollo", **when)
        searched = await memory.search("acme", "Ana owns Apollo", kinds=("story",), **when)
        hood = await memory.neighbors("acme", "Apollo", **when)
        around = await memory.expand("acme", ("story", other.story_id), by="vector", **when)
        found = [
            [s.story_id for s in listed],
            [h.id for h in recalled.hits if h.id == owned.story_id],
            [h.id for h in searched.hits if h.id == owned.story_id],
            [h.id for h in hood.passages],
            [h.id for h in around.hits],
        ]
        assert found == [[] if shown is None else [owned.story_id]] * 5, known_at
        assert (hood.relations != []) == (shown is not None)
        every, _ = await memory.list_stories(
            "acme", subject="apollo", known_at=known_at, history=True
        )
        [story] = every
        hit = next(h for h in recalled.hits if h.id == owned.story_id) if shown else None
        if shown is not None:
            assert (story.valid_to, story.closed_at, story.revised_at) == shown, known_at
            assert hit is not None and hit.story is not None
            assert (hit.story.valid_to, hit.story.closed_at, hit.story.revised_at) == shown
        dated, _ = await memory.list_stories(
            "acme", filters={"valid_to": {"lt": _on(2, 15)}}, known_at=known_at, history=True
        )
        ends = story.valid_to
        assert [s.story_id for s in dated] == (
            [owned.story_id] if ends is not None and ends < _on(2, 15) else []
        ), known_at
    current = await _story(memory, owned.story_id)
    assert (current.valid_to, current.closed_at, current.revised_at) == (
        _on(3, 1),
        _on(2, 10),
        _on(2, 10),
    )


async def test_revisions_under_a_frozen_clock_each_take_a_later_moment(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [_on(1, 10)]
    memory = await _memory(storage, monkeypatch, clock)
    story = await _told(memory, "Cy runs the line.", "line", valid_from=_on(1, 5))
    ended = [
        await memory.end_story("acme", story.story_id, valid_to=_on(2, day)) for day in (1, 2, 3)
    ]
    then = _on(1, 10)
    assert [e.closed_at for e in ended] == [then + US, then + 2 * US, then + 3 * US]
    assert [e.revised_at for e in ended] == [None, then + 2 * US, then + 3 * US]
    for known_at, end in ((then, None), (then + US, _on(2, 1)), (then + 2 * US, _on(2, 2))):
        [seen], _ = await memory.list_stories(
            "acme", subject="line", known_at=known_at, history=True
        )
        assert seen.valid_to == end, known_at
    later = (
        await memory.supersede("acme", story.story_id, "Di runs the line.", valid_from=_on(1, 7))
    ).story
    assert later.created_at == then + 4 * US
    retracted = await memory.end_story("acme", later.story_id, valid_to=later.valid_from)
    assert retracted.closed_at == then + 5 * US
    back = await _story(memory, story.story_id)
    assert (back.valid_to, back.closed_at, back.revised_at) == (None, None, then + 5 * US)
    for known_at, end in ((then + 3 * US, _on(2, 3)), (then + 4 * US, _on(1, 7))):
        [seen], _ = await memory.list_stories(
            "acme", subject="line", known_at=known_at, as_of=_on(1, 6)
        )
        assert (seen.story_id, seen.valid_to) == (story.story_id, end), known_at


async def test_a_replay_at_the_recorded_moment_reads_the_closing_held_then(
    storage: PostgresStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [_on(1, 10)]
    refinery = MemoryRefinery(storage)
    memory = await _memory(storage, monkeypatch, clock, refinery=refinery)
    leak = await _told(memory, "The pump seal leaks.", "pump", valid_from=_on(1, 5))
    clock[0] = _on(1, 20)
    await memory.end_story("acme", leak.story_id, valid_to=_on(3, 1))
    clock[0] = _on(2, 1)
    recorded = await memory.recall("acme", "pump seal")
    assert [h.id for h in recorded.hits] == [leak.story_id] and recorded.retrieval_id
    useful = [Relevance(corpus_id="acme", kind="story", id=leak.story_id, grade=2)]
    await refinery.review("acme", recorded.retrieval_id, RetrievalReview(relevance=useful))
    clock[0] = _on(2, 10)
    await memory.end_story("acme", leak.story_id, valid_to=_on(1, 25))
    assert (await memory.recall("acme", "pump seal", as_of=_on(2, 1))).hits == []
    candidate = Memory(storage, embedding=HashEmbedding())
    [then] = [pair async for pair in refinery.replay("acme", candidate, holdout=None)]
    assert then.error is None and then.candidate is not None
    assert [h.id for h in then.candidate.hits] == [leak.story_id]
    [now] = [pair async for pair in refinery.replay("acme", candidate, holdout=None, moment="now")]
    assert now.candidate is not None and now.candidate.hits == []
