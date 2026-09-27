import asyncio
from datetime import UTC, datetime

from rfnry_memory import InMemoryStorage, Memory, MemoryTools, NotFoundError, Tool, Tools

from providers import VoyageProvider


async def now(args):
    return datetime.now(UTC).isoformat()


async def main():
    memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
    await memory.initialize()
    tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.team": "ops"})
    print("definitions:", [d["name"] for d in tools.definitions()])

    written = await tools.dispatch(
        "remember", {"text": "The ops standup moved to 9:30.", "subject": "team:ops"}
    )
    print("stamped:", written["story"]["metadata"])
    await memory.remember(
        "acme", "The sales standup is at 10:00.", subject="team:sales", metadata={"team": "sales"}
    )

    found = await tools.dispatch("recall", {"query": "when is the standup?"})
    print("the tool sees:", [hit["text"] for hit in found["hits"]])
    everyone = await memory.recall("acme", "when is the standup?")
    print("the code sees:", [hit.text for hit in everyone.hits])

    try:
        await tools.dispatch("delete_story", {"story_id": "missing"})
    except NotFoundError as error:
        print("a failed call raises:", error)

    reader = tools.only("search", "recall") + Tools([Tool("now", "The current UTC time", body=now)])
    print(
        "narrowed:",
        reader.names,
        "| without writes:",
        tools.without("remember", "supersede", "delete_story").names,
    )
    print("now:", (await reader.dispatch("now", {}))[:19])


asyncio.run(main())
