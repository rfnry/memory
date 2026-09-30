import asyncio
import os

import asyncpg

from harness import delete, plant, search
from providers import VoyageProvider

DSN = os.environ.get("MEMORY_POSTGRES_URL", "postgresql://memory:memory@127.0.0.1:55434/memory")


async def main():
    pool = await asyncpg.create_pool(DSN, server_settings={"search_path": "examples, public"})
    await pool.execute("CREATE SCHEMA IF NOT EXISTS examples")
    memory = await plant(pool, VoyageProvider())

    hits = await search(memory)
    for hit in hits.hits:
        print(hit.kind, hit.provenance, hit.text)

    deleted, residue = await delete(memory)
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
