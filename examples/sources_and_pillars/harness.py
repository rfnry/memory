from rfnry_memory import EntityIn, InMemoryStorage, Memory, RelationIn, Section

MANUAL = """# Pump P-3 manual

## Seals
Replace the mechanical seal on P-3 every twelve months with seal kit SK-300.

## Wiring
P-3 is fed from panel MCC-2, breaker 14. Ground the frame before connecting L1.
"""


async def plant(embedding):
    memory = Memory(
        InMemoryStorage(),
        embedding=embedding,
        pillars={"text", "vector", "graph"},
        weights={"graph": 2.0},
        chunk_chars=120,
        overlap_chars=0,
    )
    await memory.initialize()

    manual = await memory.ingest(
        "plant", name="Pump manual v3", text=MANUAL, metadata={"line": "A", "rev": "3"}
    )
    sheet = await memory.ingest(
        "plant",
        name="P&ID sheet 7",
        sections=[
            Section(
                title="Feed line",
                text="Valve V-101 feeds pump P-3 from tank T-1.",
                locator={"sheet": 7, "region": "A2"},
            ),
            Section(
                title="Return line",
                text="The return from P-3 passes check valve V-204.",
                locator={"sheet": 7, "region": "C4"},
            ),
        ],
        entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump")],
        relations=[RelationIn(from_name="V-101", to_name="P-3", type="CONNECTS_TO")],
    )
    return memory, manual, sheet


async def search_fused(memory):
    return await memory.search("plant", "what feeds pump P-3?", top_k=3)


async def search_meaning(memory):
    return await memory.search(
        "plant",
        "how often does the seal need changing",
        pillars={"vector"},
        kinds=("chunk",),
        top_k=1,
    )
