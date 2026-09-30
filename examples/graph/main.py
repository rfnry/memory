import asyncio

from rfnry_memory import InputError

from harness import extract, neighbors_by_name, neighbors_of_the_pump, plant
from providers import AnthropicProvider, VoyageProvider


async def main():
    memory = await plant(VoyageProvider(), AnthropicProvider())

    extracted = await extract(memory)
    print(
        "extraction cost",
        extracted.usage.input,
        "in",
        extracted.usage.output,
        "out",
    )

    try:
        await neighbors_by_name(memory)
    except InputError as error:
        print(error)

    hood = await neighbors_of_the_pump(memory)
    print("entities:", sorted(e.name for e in hood.entities))
    names = {e.entity_id: e.name for e in hood.entities}
    print(
        "relations:",
        sorted(
            f"{names.get(r.from_id, '?')} -{r.type}-> {names.get(r.to_id, '?')}"
            for r in hood.relations
        ),
    )
    print("passages:", [p.text for p in hood.passages])


asyncio.run(main())
