import asyncio
from datetime import UTC, datetime

from rfnry_memory import InMemoryStorage, Memory

from providers import VoyageProvider


async def main():
    memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
    await memory.initialize()

    remembered = await memory.remember(
        "acme",
        "The go-live is February 20.",
        subject="project:apollo",
        occurred_at=datetime(2026, 1, 10, tzinfo=UTC),
    )
    story = remembered.story
    superseded = await memory.supersede(
        "acme",
        story.story_id,
        "Ana moved the go-live to March 3.",
        effective_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    current = superseded.story
    [closed] = await memory.get_stories("acme", [story.story_id])
    print(
        "closed at",
        closed.valid_to.date(),
        "| current supersedes the old one:",
        current.supersedes == story.story_id,
        "| cost:",
        remembered.usage.input + superseded.usage.input,
        "tokens in",
    )

    now = await memory.recall("acme", "when is the go-live?", subject="project:apollo")
    then = await memory.recall(
        "acme",
        "when is the go-live?",
        subject="project:apollo",
        as_of=datetime(2026, 1, 20, tzinfo=UTC),
    )
    everything = await memory.recall(
        "acme", "when is the go-live?", subject="project:apollo", history=True
    )
    print("now:", now.hits[0].text)
    print("on Jan 20:", then.hits[0].text)
    print(
        "history:",
        [(hit.text, hit.story.valid_to and hit.story.valid_to.date()) for hit in everything.hits],
    )

    stories, _ = await memory.list_stories("acme", subject="project:apollo", history=True)
    print(
        "list_stories:",
        len(stories),
        "| get_stories:",
        [s.text for s in await memory.get_stories("acme", [story.story_id])],
    )


asyncio.run(main())
