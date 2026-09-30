import asyncio

from rfnry_memory import NotFoundError

from harness import (
    ask_the_time,
    delete_missing_story,
    narrow,
    recall_standup,
    remember_ops_standup,
    remember_sales_standup,
    team_tools,
)
from providers import VoyageProvider


async def main():
    memory, tools = await team_tools(VoyageProvider())
    print("definitions:", [d["name"] for d in tools.definitions()])

    written = await remember_ops_standup(tools)
    print("stamped:", written["story"]["metadata"])
    await remember_sales_standup(memory)

    found, everyone = await recall_standup(tools, memory)
    print("the tool sees:", [hit["text"] for hit in found["hits"]])
    print("the code sees:", [hit.text for hit in everyone.hits])

    try:
        await delete_missing_story(tools)
    except NotFoundError as error:
        print("a failed call raises:", error)

    reader, readonly = narrow(tools)
    print("narrowed:", reader.names, "| without writes:", readonly.names)
    print("now:", (await ask_the_time(reader))[:19])


asyncio.run(main())
