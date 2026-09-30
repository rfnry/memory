from rfnry_memory import InMemoryStorage, Memory, Section

GUIDE = (
    "# Line A guide\n\n"
    "## Startup\nOpen V-101 before starting P-3.\n\n"
    "## Shutdown\nStop P-3, then close V-101."
)


async def plant(embedding):
    memory = Memory(InMemoryStorage(), embedding=embedding, chunk_chars=400, overlap_chars=40)
    await memory.initialize()

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
    return memory, v2, v3, drawing


async def search_line_a_rev_3(memory):
    return await memory.search(
        "plant",
        "what to do before starting the pump",
        filters={"metadata.line": "A", "metadata.rev": ["3"]},
    )


async def list_line_a(memory):
    return await memory.list_sources("plant", filters={"metadata.line": "A"}, limit=10)


async def get(memory, ingested):
    return await memory.get_source("plant", ingested.source.source_id)


async def read_sheet(memory, ingested, sheet):
    return await memory.read_source("plant", ingested.source.source_id, locator={"sheet": sheet})


async def delete(memory, ingested):
    return await memory.delete_source("plant", ingested.source.source_id)
