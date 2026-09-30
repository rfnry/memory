from pathlib import Path

import rfnry_engine
import rfnry_memory
from rfnry_engine import Engine, Outcome, Refinery, Review
from rfnry_memory import Memory, MemoryTools

KNOWLEDGE = {
    "north": (
        "Can line 1 runs at 1,800 cans per minute. The night supervisor is Bruno, extension 214."
    ),
    "south": (
        "Furnace 2 melts at 1,550 C. "
        "If it runs hot, call the furnace lead, Carla, on extension 540."
    ),
}


class Folder:
    def __init__(self, root):
        self.root = Path(root)

    async def put(self, key, data):
        (self.root / key).parent.mkdir(parents=True, exist_ok=True)
        (self.root / key).write_bytes(data)


async def factories(embedding):
    memory = Memory(rfnry_memory.InMemoryStorage(), embedding=embedding)
    await memory.initialize()
    for factory, text in KNOWLEDGE.items():
        await memory.ingest(f"factory-{factory}", name="Operations notes", text=text)
    return memory


async def assistant(generation, decision):
    refinery = Refinery(rfnry_engine.InMemoryStorage())
    await refinery.initialize()
    engine = Engine("agents", generation=generation, decision=decision, refinery=refinery)
    return engine, refinery


async def route(engine, question):
    routed = await engine.run("router", scope="acme", input=question)
    factory, incident = (
        routed.answers["factory"].choice,
        routed.answers["incident"].probability > 0.5,
    )
    return routed, factory, incident


async def handle(engine, memory, question, factory, incident):
    return await engine.run(
        factory,
        scope="acme",
        input=question,
        task="incident" if incident else None,
        tools=MemoryTools(memory, corpus_id=f"factory-{factory}"),
    )


async def refine(refinery, run):
    await refinery.review(
        "acme", run.run_id, Review(stance="approval", comment="Right person, right extension.")
    )
    await refinery.outcome(
        "acme", run.run_id, Outcome(status="succeeded", detail="Carla answered.")
    )
    return await refinery.export("acme", Folder("datasets"))
