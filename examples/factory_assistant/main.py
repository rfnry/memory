import asyncio
from pathlib import Path

import rfnry_harness
import rfnry_memory
from rfnry_harness import Harness, Outcome, Refinery, Review
from rfnry_memory import Memory, MemoryTools

from models import claude, decision, embedding

KNOWLEDGE = {
    "north": "Can line 1 runs at 1,800 cans per minute. The night supervisor is Bruno, extension 214.",
    "south": "Furnace 2 melts at 1,550 C. If it runs hot, call the furnace lead, Carla, on extension 540.",
}


class Bucket(dict):
    async def put(self, key, data):
        self[key] = data


async def main():
    memory = Memory(rfnry_memory.InMemoryStorage(), embedding=embedding())
    await memory.initialize()
    for factory, text in KNOWLEDGE.items():
        await memory.ingest(f"factory-{factory}", name="Operations notes", text=text)

    refinery = Refinery(rfnry_harness.InMemoryStorage())
    await refinery.initialize()
    harness = Harness(Path(__file__).parent / "agents", generation=claude(rfnry_harness),
                      decision=decision(rfnry_harness), refinery=refinery)

    question = "Furnace 2 at the bottle plant is running hot. Who do I call?"
    routed = await harness.run("router", scope="acme", input=question)
    factory = routed.answers["factory"].choice
    print("routed to", factory, f"(confidence {routed.answers['factory'].confidence:.2f})")

    run = await harness.run(factory, scope="acme", input=question, tools=MemoryTools(memory, corpus_id=f"factory-{factory}"))
    print(run.reply)

    await refinery.review("acme", run.run_id, Review(stance="approval", comment="Right person, right extension."))
    await refinery.outcome("acme", run.run_id, Outcome(status="succeeded", detail="Carla answered."))
    exported = await refinery.export(Bucket(), "acme")
    print("exported:", exported.rows, *exported.keys, sep="\n  ")


asyncio.run(main())
