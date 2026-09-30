from rfnry_memory import EntityIn, InMemoryStorage, Memory


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
    return (await memory.neighbors("plant", "Pump 3")).entities, written.usage


async def pumps_with_decision(embedding, decision):
    return await pumps(
        Memory(
            InMemoryStorage(),
            embedding=embedding,
            decision=decision,
            pillars={"text", "vector", "graph"},
        )
    )


async def pumps_without_decision(embedding):
    return await pumps(
        Memory(InMemoryStorage(), embedding=embedding, pillars={"text", "vector", "graph"})
    )
