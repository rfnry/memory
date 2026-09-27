import asyncio
import os

import asyncpg
from rfnry_memory import Memory, PostgresStorage

from providers import VoyageProvider

DSN = os.environ.get("MEMORY_POSTGRES_URL", "postgresql://memory:memory@127.0.0.1:55434/memory")


async def main():
    pool = await asyncpg.create_pool(DSN, server_settings={"search_path": "examples, public"})
    await pool.execute("CREATE SCHEMA IF NOT EXISTS examples")
    memory = Memory(PostgresStorage(pool), embedding=VoyageProvider())
    await memory.initialize()

    await memory.ingest(
        "example-postgres", name="Pump manual", text="Replace the P-3 seal every twelve months."
    )
    await memory.remember(
        "example-postgres", "Ana owns the P-3 maintenance plan.", subject="pump:P-3"
    )
    hits = await memory.search(
        "example-postgres", "who is responsible for pump maintenance?", top_k=2
    )
    for hit in hits.hits:
        print(hit.kind, hit.provenance, hit.text)

    deleted = await memory.delete_corpus("example-postgres")
    residue = await memory.verify_absent("example-postgres")
    print(
        "deleted:",
        deleted.sources,
        "source,",
        deleted.chunks,
        "chunk,",
        deleted.stories,
        "story | clean:",
        residue.clean,
    )
    await pool.close()


asyncio.run(main())
