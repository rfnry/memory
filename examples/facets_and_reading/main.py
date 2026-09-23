import asyncio

from rfnry_memory import InMemoryStorage, Memory, Section, chunk_text

from providers import VoyageProvider

GUIDE = (
    "# Line A guide\n\n"
    "## Startup\nOpen V-101 before starting P-3.\n\n"
    "## Shutdown\nStop P-3, then close V-101."
)


async def main():
    memory = Memory(
        InMemoryStorage(), embedding=VoyageProvider(), chunk_chars=400, overlap_chars=40
    )
    await memory.initialize()
    print(
        "chunk_text:",
        [piece.title for piece in chunk_text(GUIDE, chunk_chars=400, overlap_chars=40)],
    )

    v2 = await memory.ingest(
        "plant", name="Line A guide v2", text=GUIDE, metadata={"line": "A", "rev": "2"}
    )
    v3 = await memory.ingest(
        "plant",
        name="Line A guide v3",
        text=GUIDE.replace("P-3", "P-3 and P-4"),
        metadata={"line": "A", "rev": "3"},
    )
    drawing = await memory.ingest(
        "plant",
        name="Line B drawing",
        metadata={"line": "B"},
        sections=[
            Section(title="Pumps", text="P-7 and P-8 run in parallel.", locator={"sheet": 2}),
            Section(title="Valves", text="V-300 isolates P-7.", locator={"sheet": 3}),
        ],
    )

    hits = await memory.search(
        "plant",
        "what to do before starting the pump",
        filters={"metadata.line": "A", "metadata.rev": ["3"]},
    )
    print("filtered search:", {hit.chunk.source_id == v3.source.source_id for hit in hits.hits})

    sources, cursor = await memory.list_sources("plant", filters={"metadata.line": "A"}, limit=10)
    print("line A:", [s.name for s in sources], "next page:", cursor)
    print("get_source:", (await memory.get_source("plant", v2.source.source_id)).metadata)
    print(
        "read sheet 3:", await memory.read("plant", drawing.source.source_id, locator={"sheet": 3})
    )

    deleted = await memory.delete_source("plant", v2.source.source_id)
    print("deleted:", deleted.sources, "source,", deleted.chunks, "chunks")


asyncio.run(main())
