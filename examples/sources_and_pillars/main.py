import asyncio

from harness import plant, search_fused, search_meaning
from providers import VoyageProvider


async def main():
    memory, manual, sheet = await plant(VoyageProvider())
    print("ingested", manual.source.chunks, "+", sheet.source.chunks, "chunks")

    hits = await search_fused(memory)
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

    meaning = await search_meaning(memory)
    print("vector alone:", meaning.hits[0].chunk.title, "|", meaning.hits[0].text.splitlines()[-1])


asyncio.run(main())
