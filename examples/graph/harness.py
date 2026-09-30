from rfnry_memory import EntityIn, InMemoryStorage, Memory, RelationIn


async def plant(embedding, generation):
    memory = Memory(
        InMemoryStorage(),
        embedding=embedding,
        generation=generation,
        relation_types=["CONNECTS_TO", "PART_OF", "FEEDS"],
    )
    await memory.initialize()

    await memory.ingest(
        "plant",
        name="P&ID sheet 7",
        text="Valve V-101 connects to pump P-3.",
        entities=[
            EntityIn(name="V-101", type="valve"),
            EntityIn(name="P-3", type="pump"),
            EntityIn(name="P-3", type="part"),
        ],
        relations=[
            RelationIn(from_name="V-101", to_name="P-3", to_type="pump", type="CONNECTS_TO")
        ],
    )
    return memory


async def extract(memory):
    return await memory.ingest(
        "plant", name="Electrical notes", text="Panel MCC-2 feeds pump P-3 through breaker 14."
    )


async def neighbors_by_name(memory):
    return await memory.neighbors("plant", "P-3")


async def neighbors_of_the_pump(memory):
    return await memory.neighbors("plant", "P-3", entity_type="pump", hops=1)
