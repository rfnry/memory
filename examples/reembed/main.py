import asyncio

from harness import HashEmbedding, plant, reembed, search_vector, switch
from providers import VoyageProvider


async def main():
    storage = await plant(HashEmbedding())

    after = switch(storage, VoyageProvider())
    print("re-embedded", await reembed(after), "passages with", after.embedding.model)
    print("again:", await reembed(after))
    hits = await search_vector(after)
    print("vector pillar now answers:", hits.hits[0].text)


asyncio.run(main())
