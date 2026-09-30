from hashlib import sha256

from rfnry_memory import InMemoryStorage, Memory


class HashEmbedding:
    model, dimension = "hash-1024", 1024

    async def embed(self, texts):
        return [[b / 255 for b in (sha256(t.encode()).digest() * 32)] for t in texts]


async def plant(embedding):
    storage = InMemoryStorage()
    before = Memory(storage, embedding=embedding)
    await before.initialize()
    await before.ingest("plant", name="Manual", text="Replace the P-3 seal every twelve months.")
    await before.remember("plant", "The P-3 seal was replaced on March 3.", subject="pump:P-3")
    return storage


def switch(storage, embedding):
    return Memory(storage, embedding=embedding)


async def reembed(memory):
    return await memory.reembed("plant")


async def search_vector(memory):
    return await memory.search(
        "plant", "when did the seal get changed?", pillars={"vector"}, top_k=1
    )
