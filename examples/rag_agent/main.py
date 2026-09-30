import asyncio

from harness import answer, assistant, plant
from providers import AnthropicProvider, VoyageProvider

QUESTIONS = [
    "What does a mechanical seal on a centrifugal pump do?",
    "How often do we replace the seal on P-3, and who owns that plan?",
]


async def main():
    memory = await plant(VoyageProvider())

    engine, tools = assistant(memory, AnthropicProvider())
    for question in QUESTIONS:
        run = await answer(engine, tools, question)
        used = [step.tool for step in run.steps]
        print(f"\n> {question}\ntools used: {used} | {run.calls} calls,", end=" ")
        print(f"{run.usage.input} in / {run.usage.output} out")
        print(run.reply)


asyncio.run(main())
