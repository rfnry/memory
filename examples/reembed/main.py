import asyncio
from hashlib import sha256

from rfnry_memory import InMemoryStorage, Memory

from models import embedding


class HashEmbedding:
    model, dimension = "hash-1024", 1024

    async def embed(self, texts):
        return [[b / 255 for b in (sha256(t.encode()).digest() * 32)] for t in texts]


async def main():
    storage = InMemoryStorage()
    before = Memory(storage, embedding=HashEmbedding())
    await before.initialize()
    await before.ingest("plant", name="Manual", text="Replace the P-3 seal every twelve months.")
    await before.remember("plant", "The P-3 seal was replaced on March 3.", subject="pump:P-3")

    after = Memory(storage, embedding=embedding())
    print("re-embedded", await after.reembed("plant"), "passages with", after.embedding.model)
    print("again:", await after.reembed("plant"))
    hits = await after.search("plant", "when did the seal get changed?", pillars={"vector"}, top_k=1)
    print("vector pillar now answers:", hits.hits[0].text)


asyncio.run(main())
