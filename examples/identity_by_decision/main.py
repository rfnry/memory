import asyncio

from rfnry_memory import EntityIn, InMemoryStorage, Memory

from providers import TypeSafeProvider, VoyageProvider


async def pumps(memory):
    await memory.initialize()
    await memory.ingest(
        "plant",
        name="Manual",
        text="P-3 is the feed pump on line A.",
        entities=[EntityIn(name="P-3", type="pump")],
    )
    written = await memory.ingest(
        "plant",
        name="Shift log",
        text="Pump 3, the line A feed pump, was serviced today.",
        entities=[EntityIn(name="Pump 3", type="pump")],
    )
    return await memory.storage.entities_named("plant", "Pump 3"), written.usage


async def main():
    merged, usage = await pumps(
        Memory(
            InMemoryStorage(),
            embedding=VoyageProvider(),
            decision=TypeSafeProvider(),
            pillars={"text", "vector", "graph"},
        )
    )
    print(
        "with a decision model:",
        [(e.name, e.aliases) for e in merged],
        "| usage",
        usage.input,
        "in",
        usage.output,
        "out",
    )

    exact, _ = await pumps(
        Memory(InMemoryStorage(), embedding=VoyageProvider(), pillars={"text", "vector", "graph"})
    )
    print("without one:", [(e.name, e.aliases) for e in exact])


asyncio.run(main())
