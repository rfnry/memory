import asyncio

from harness import claire, conversation, stories
from providers import AnthropicProvider, VoyageProvider


async def main():
    memory, engine, tools = await claire(VoyageProvider(), AnthropicProvider())

    async def talk(*turns):
        async for text, run in conversation(engine, tools, *turns):
            print(f"> {text}\n  [{', '.join(step.tool for step in run.steps)}] {run.reply}")

    await talk(
        "Hi Claire, I'm Ana. I'm vegetarian and I live in Lisbon.",
        "Also: my sister Rita's birthday is on May 4.",
    )
    await talk("Some news: I moved to Porto last month.")
    await talk(
        "Where should I take Rita to dinner for her birthday? Just name the kind of place and city."
    )

    for story in await stories(memory):
        print("closed" if story.valid_to else "open  ", story.text)


asyncio.run(main())
