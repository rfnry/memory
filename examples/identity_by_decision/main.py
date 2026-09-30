import asyncio

from harness import pumps_with_decision, pumps_without_decision
from providers import TypeSafeProvider, VoyageProvider


async def main():
    merged, usage = await pumps_with_decision(VoyageProvider(), TypeSafeProvider())
    print(
        "with a decision model:",
        [(e.name, e.aliases) for e in merged],
        "| usage",
        usage.input,
        "in",
        usage.output,
        "out",
    )

    exact, _ = await pumps_without_decision(VoyageProvider())
    print("without one:", [(e.name, e.aliases) for e in exact])


asyncio.run(main())
