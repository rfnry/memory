import asyncio

from rfnry_memory import EntityIn, InMemoryStorage, InputError, Memory, RelationIn

from providers import Claude, Voyage


async def main():
    memory = Memory(InMemoryStorage(), embedding=Voyage(), generation=Claude(),
                    relation_types=["CONNECTS_TO", "PART_OF", "FEEDS"])
    await memory.initialize()

    await memory.ingest("plant", name="P&ID sheet 7", text="Valve V-101 connects to pump P-3.",
                        entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump"),
                                  EntityIn(name="P-3", type="part")],
                        relations=[RelationIn(from_name="V-101", to_name="P-3", to_type="pump", type="CONNECTS_TO")])

    extracted = await memory.ingest("plant", name="Electrical notes", text="Panel MCC-2 feeds pump P-3 through breaker 14.")
    print("extracted", extracted.entities, "new entities and", extracted.relations, "relations |", extracted.usage.input, "in", extracted.usage.output, "out")

    try:
        await memory.neighbors("plant", "P-3")
    except InputError as error:
        print(error)

    hood = await memory.neighbors("plant", "P-3", entity_type="pump", hops=1)
    print("entities:", sorted(e.name for e in hood.entities))
    names = {e.entity_id: e.name for e in hood.entities}
    print("relations:", sorted(f"{names.get(r.from_id, '?')} -{r.type}-> {names.get(r.to_id, '?')}" for r in hood.relations))
    print("passages:", [p.text for p in hood.passages])


asyncio.run(main())
