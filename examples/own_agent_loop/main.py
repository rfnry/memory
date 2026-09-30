import asyncio
import json

from harness import ask, plant, stories
from providers import AnthropicProvider, VoyageProvider


def show(call):
    print(f"  {call.name}({json.dumps(call.args)})")


async def main():
    memory, tools = await plant(VoyageProvider())
    model = AnthropicProvider()
    for text in (
        "Ana replaced the P-3 seal this morning.",
        "Which kit does the P-3 seal take, and who changed it last?",
    ):
        print(f"> {text}")
        print(await ask(model, tools, text, show))
    print("stories:", [story.text for story in await stories(memory)])


asyncio.run(main())
