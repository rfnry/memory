import asyncio

from rfnry_memory import InMemoryStorage, Memory, fuse_by_rank

from providers import VoyageProvider


async def main():
    memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
    await memory.initialize()
    await memory.ingest(
        "north-plant", name="Handbook", text="North plant: boilers are inspected every spring."
    )
    await memory.ingest(
        "south-plant", name="Handbook", text="South plant: boilers are inspected every autumn."
    )

    north = await memory.search("north-plant", "when are boilers inspected?", top_k=3)
    print("one corpus at a time:", [hit.text for hit in north.hits])

    south = await memory.search("south-plant", "when are boilers inspected?", top_k=3)
    texts = {hit.id: hit.text for hit in [*north.hits, *south.hits]}
    fused = fuse_by_rank(
        [
            ("north", 1.0, [(h.id, h.score) for h in north.hits]),
            ("south", 1.0, [(h.id, h.score) for h in south.hits]),
        ]
    )
    for key, score, provenance in fused:
        print(f"{score:.4f}", provenance, texts[key])


asyncio.run(main())
