import asyncio

from harness import assistant, factories, handle, refine, route
from providers import AnthropicProvider, TypeSafeProvider, VoyageProvider


async def main():
    memory = await factories(VoyageProvider())
    engine, refinery = await assistant(AnthropicProvider(), TypeSafeProvider())

    question = "Furnace 2 at the bottle plant is running hot. Who do I call?"
    routed, factory, incident = await route(engine, question)
    print(
        "routed to",
        factory,
        f"(confidence {routed.answers['factory'].confidence:.2f}) | incident:",
        incident,
    )

    run = await handle(engine, memory, question, factory, incident)
    print("read:", [step.args.get("path") or step.args.get("query") for step in run.steps])
    print(run.reply)

    exported = await refine(refinery, run)
    print(f"rows {exported.rows}")
    for key in exported.keys:
        print(f"  datasets/{key}")


asyncio.run(main())
