import asyncio

from rfnry_memory import InMemoryStorage, Memory

from providers import VoyageProvider

QUESTION = "when are boilers inspected?"


async def main():
    memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
    await memory.initialize()
    await memory.ingest(
        "north-plant", name="Handbook", text="North plant: boilers are inspected every spring."
    )
    await memory.ingest(
        "south-plant", name="Handbook", text="South plant: boilers are inspected every autumn."
    )

    for plant in ("north-plant", "south-plant"):
        hits = await memory.search(plant, QUESTION, top_k=3)
        print(plant, "reads only its own:", [hit.text for hit in hits.hits])

    deleted = await memory.delete_corpus("north-plant")
    print("deleted north:", deleted.sources, "source,", deleted.chunks, "chunk")
    print("north is empty:", (await memory.verify_absent("north-plant")).clean)
    south = await memory.search("south-plant", QUESTION, top_k=3)
    print("south still answers:", south.hits[0].text)


asyncio.run(main())
