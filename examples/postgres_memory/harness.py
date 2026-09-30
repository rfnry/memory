from rfnry_memory import Memory, PostgresStorage


async def plant(pool, embedding):
    memory = Memory(PostgresStorage(pool), embedding=embedding)
    await memory.initialize()

    await memory.ingest(
        "example-postgres", name="Pump manual", text="Replace the P-3 seal every twelve months."
    )
    await memory.remember(
        "example-postgres", "Ana owns the P-3 maintenance plan.", subject="pump:P-3"
    )
    return memory


async def search(memory):
    return await memory.search(
        "example-postgres", "who is responsible for pump maintenance?", top_k=2
    )


async def delete(memory):
    deleted = await memory.delete_corpus("example-postgres")
    return deleted, await memory.verify_absent("example-postgres")
