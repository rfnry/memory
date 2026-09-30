from datetime import UTC, datetime

from rfnry_memory import InMemoryStorage, Memory, MemoryTools, Tool, Tools


async def now(args):
    return datetime.now(UTC).isoformat()


async def team_tools(embedding):
    memory = Memory(InMemoryStorage(), embedding=embedding)
    await memory.initialize()
    return memory, MemoryTools(memory, corpus_id="acme", filters={"metadata.team": "ops"})


async def remember_ops_standup(tools):
    return await tools.dispatch(
        "remember", {"text": "The ops standup moved to 9:30.", "subject": "team:ops"}
    )


async def remember_sales_standup(memory):
    await memory.remember(
        "acme", "The sales standup is at 10:00.", subject="team:sales", metadata={"team": "sales"}
    )


async def recall_standup(tools, memory):
    found = await tools.dispatch("recall", {"query": "when is the standup?"})
    everyone = await memory.recall("acme", "when is the standup?")
    return found, everyone


async def delete_missing_story(tools):
    await tools.dispatch("delete_story", {"story_id": "missing"})


def narrow(tools):
    reader = tools.only("search", "recall") + Tools([Tool("now", "The current UTC time", body=now)])
    return reader, tools.without("remember", "supersede", "delete_story")


async def ask_the_time(reader):
    return await reader.dispatch("now", {})
