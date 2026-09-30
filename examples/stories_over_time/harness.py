from datetime import UTC, datetime

from rfnry_memory import InMemoryStorage, Memory


async def project(embedding):
    memory = Memory(InMemoryStorage(), embedding=embedding)
    await memory.initialize()
    return memory


async def move_go_live(memory):
    remembered = await memory.remember(
        "acme",
        "The go-live is February 20.",
        subject="project:apollo",
        valid_from=datetime(2026, 1, 10, tzinfo=UTC),
    )
    story = remembered.story
    superseded = await memory.supersede(
        "acme",
        story.story_id,
        "Ana moved the go-live to March 3.",
        valid_from=datetime(2026, 2, 1, tzinfo=UTC),
    )
    [closed] = await memory.get_stories("acme", [story.story_id])
    return remembered, superseded, closed


async def recall_go_live(memory):
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
    return now, then, everything


async def list_stories(memory):
    stories, _ = await memory.list_stories("acme", subject="project:apollo", history=True)
    return stories


async def get_stories(memory, story):
    return await memory.get_stories("acme", [story.story_id])
