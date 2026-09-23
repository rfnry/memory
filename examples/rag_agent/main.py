import asyncio

from rfnry_harness import Harness
from rfnry_memory import InMemoryStorage, Memory, MemoryTools

from providers import AnthropicProvider, VoyageProvider

QUESTIONS = [
    "What does a mechanical seal on a centrifugal pump do?",
    "How often do we replace the seal on P-3, and who owns that plan?",
]


async def main():
    memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
    await memory.initialize()
    await memory.ingest(
        "acme",
        name="Pump P-3 manual",
        text="Replace the mechanical seal on P-3 every six months with kit SK-300.",
    )
    await memory.remember("acme", "Ana owns the P-3 maintenance plan.", subject="pump:P-3")

    harness = Harness("agents", generation=AnthropicProvider())
    tools = MemoryTools(memory, corpus_id="acme").only("search", "recall")
    for question in QUESTIONS:
        run = await harness.run("assistant", scope="acme", input=question, tools=tools)
        used = [step.tool for step in run.steps]
        print(f"\n> {question}\ntools used: {used} | {run.calls} calls,", end=" ")
        print(f"{run.usage.input} in / {run.usage.output} out")
        print(run.reply)


asyncio.run(main())
