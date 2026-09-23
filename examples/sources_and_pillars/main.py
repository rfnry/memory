import asyncio

from rfnry_memory import EntityIn, InMemoryStorage, Memory, RelationIn, Section

from providers import VoyageProvider

MANUAL = """# Pump P-3 manual

## Seals
Replace the mechanical seal on P-3 every twelve months with seal kit SK-300.

## Wiring
P-3 is fed from panel MCC-2, breaker 14. Ground the frame before connecting L1.
"""


async def main():
    memory = Memory(
        InMemoryStorage(),
        embedding=VoyageProvider(),
        pillars={"text", "vector", "graph"},
        weights={"graph": 2.0},
        fetch_multiplier=4,
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
    print(
        "ingested",
        manual.source.chunks,
        "+",
        sheet.source.chunks,
        "chunks,",
        sheet.entities,
        "entities,",
        sheet.relations,
        "relation",
    )

    hits = await memory.search("plant", "what feeds pump P-3?", top_k=3)
    for hit in hits.hits:
        print(
            f"{hit.score:.4f}",
            hit.provenance,
            hit.chunk.title,
            hit.chunk.locator,
            hit.text.splitlines()[-1],
        )
    for name, report in hits.pillars.items():
        print(name, "returned", report.returned, "in", report.duration_ms, "ms")

    meaning = await memory.search(
        "plant",
        "how often does the seal need changing",
        pillars={"vector"},
        kinds=("chunk",),
        top_k=1,
    )
    print("vector alone:", meaning.hits[0].chunk.title, "|", meaning.hits[0].text.splitlines()[-1])


asyncio.run(main())
