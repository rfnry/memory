import asyncio

from rfnry_memory import ConfigurationError

from harness import keep, plant, search_judged, search_judged_without_decision
from providers import TypeSafeProvider, VoyageProvider


async def main():
    memory = await plant(VoyageProvider(), TypeSafeProvider())

    hits = await search_judged(memory)
    print(
        "hits returned:",
        len(hits.hits),
        "| judge usage:",
        hits.usage.input,
        "in",
        hits.usage.output,
        "out",
    )
    for hit in hits.hits:
        j = hit.judge
        print(
            f"relevant={j.relevant:.2f} evidence={j.evidence:.2f} contradicts={j.contradicts:.2f} "
            f"injection={j.injection:.2f}  {hit.text[:50]}"
        )

    print("what my code keeps:", keep(hits))

    try:
        await search_judged_without_decision(memory)
    except ConfigurationError as error:
        print(error)


asyncio.run(main())
