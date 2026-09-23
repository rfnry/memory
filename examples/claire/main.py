import asyncio

from rfnry_harness import Harness
from rfnry_memory import InMemoryStorage, Memory, MemoryTools

from providers import AnthropicProvider, VoyageProvider


async def main():
    memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
    await memory.initialize()
    harness = Harness("agents", generation=AnthropicProvider())
    tools = MemoryTools(memory, corpus_id="claire", filters={"metadata.user": "ana"})

    async def conversation(*turns):
        history = []
        for text in turns:
            run = await harness.run("claire", scope="ana", input=text, history=history, tools=tools)
            history = run.conversation
            print(f"> {text}\n  [{', '.join(step.tool for step in run.steps)}] {run.reply}")

    await conversation(
        "Hi Claire, I'm Ana. I'm vegetarian and I live in Lisbon.",
        "Also: my sister Rita's birthday is on May 4.",
    )
    await conversation("Some news: I moved to Porto last month.")
    await conversation(
        "Where should I take Rita to dinner for her birthday? Just name the kind of place and city."
    )

    stories, _ = await memory.list_stories("claire", history=True, filters={"metadata.user": "ana"})
    for story in stories:
        print("closed" if story.valid_to else "open  ", story.text)


asyncio.run(main())
