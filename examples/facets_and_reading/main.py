import asyncio

from harness import delete, get, list_line_a, plant, read_sheet, search_line_a_rev_3
from providers import VoyageProvider


async def main():
    memory, v2, v3, drawing = await plant(VoyageProvider())

    hits = await search_line_a_rev_3(memory)
    print("filtered search:", {hit.chunk.source_id == v3.source.source_id for hit in hits.hits})

    sources, cursor = await list_line_a(memory)
    print("line A:", [s.name for s in sources], "next page:", cursor)
    print("get_source:", (await get(memory, v2)).metadata)
    print("read sheet 3:", await read_sheet(memory, drawing, 3))

    deleted = await delete(memory, v2)
    print("deleted:", deleted.sources, "source,", deleted.chunks, "chunks")


asyncio.run(main())
