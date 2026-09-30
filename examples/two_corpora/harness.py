from rfnry_memory import InMemoryStorage, Memory

QUESTION = "when are boilers inspected?"


async def plants(embedding):
    memory = Memory(InMemoryStorage(), embedding=embedding)
    await memory.initialize()
    await memory.ingest(
        "north-plant", name="Handbook", text="North plant: boilers are inspected every spring."
    )
    await memory.ingest(
        "south-plant", name="Handbook", text="South plant: boilers are inspected every autumn."
    )
    return memory


async def search(memory, plant):
    return await memory.search(plant, QUESTION, top_k=3)


async def delete(memory, plant):
    return await memory.delete_corpus(plant), await memory.verify_absent(plant)
