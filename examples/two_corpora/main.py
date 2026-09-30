import asyncio

from harness import delete, plants, search
from providers import VoyageProvider


async def main():
    memory = await plants(VoyageProvider())

    for plant in ("north-plant", "south-plant"):
        hits = await search(memory, plant)
        print(plant, "reads only its own:", [hit.text for hit in hits.hits])

    deleted, residue = await delete(memory, "north-plant")
    print("deleted north:", deleted.sources, "source,", deleted.chunks, "chunk")
    print("north is empty:", residue.clean)
    south = await search(memory, "south-plant")
    print("south still answers:", south.hits[0].text)


asyncio.run(main())
