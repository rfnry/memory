import asyncio

from harness import get_stories, list_stories, move_go_live, project, recall_go_live
from providers import VoyageProvider


async def main():
    memory = await project(VoyageProvider())

    remembered, superseded, closed = await move_go_live(memory)
    story = remembered.story
    current = superseded.story
    print(
        "closed at",
        closed.valid_to.date(),
        "| current supersedes the old one:",
        current.supersedes == story.story_id,
        "| cost:",
        remembered.usage.input + superseded.usage.input,
        "tokens in",
    )

    now, then, everything = await recall_go_live(memory)
    print("now:", now.hits[0].text)
    print("on Jan 20:", then.hits[0].text)
    print(
        "history:",
        [(hit.text, hit.story.valid_to and hit.story.valid_to.date()) for hit in everything.hits],
    )

    stories = await list_stories(memory)
    print(
        "list_stories:",
        len(stories),
        "| get_stories:",
        [s.text for s in await get_stories(memory, story)],
    )


asyncio.run(main())
