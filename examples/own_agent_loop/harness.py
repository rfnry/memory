import json

from rfnry_memory import InMemoryStorage, Memory, MemoryException, MemoryTools, Message, ToolSpec

SYSTEM = (
    "You are the maintenance assistant at the Acme plant. "
    "Look up anything about the plant in memory, "
    "remember what you are told about it, and answer in one sentence."
)


async def plant(embedding):
    memory = Memory(InMemoryStorage(), embedding=embedding)
    await memory.initialize()
    await memory.ingest(
        "acme",
        name="Pump P-3 manual",
        text="Replace the mechanical seal on P-3 every six months with kit SK-300.",
    )
    return memory, MemoryTools(memory, corpus_id="acme")


async def ask(model, tools, text, on_call):
    messages = [Message(role="user", content=text)]
    specs = [ToolSpec(**definition) for definition in tools.definitions()]
    while True:
        reply = await model.generate(SYSTEM, messages, specs)
        if not reply.tool_calls:
            return reply.text
        messages.append(
            Message(
                role="assistant",
                content=reply.text,
                tool_calls=reply.tool_calls,
                reasoning=reply.reasoning,
            )
        )
        for call in reply.tool_calls:
            on_call(call)
            try:
                result = await tools.dispatch(call.name, call.args)
                content, is_error = json.dumps(result, default=str), False
            except MemoryException as error:
                content, is_error = f"{type(error).__name__}: {error}", True
            messages.append(
                Message(role="tool", tool_call_id=call.id, content=content, is_error=is_error)
            )


async def stories(memory):
    return (await memory.list_stories("acme"))[0]
