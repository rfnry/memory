import asyncio
import json

from rfnry_memory import InMemoryStorage, Memory, MemoryException, MemoryTools, Message, ToolSpec

from providers import AnthropicProvider, VoyageProvider

SYSTEM = (
    "You are the maintenance assistant at the Acme plant. "
    "Look up anything about the plant in memory, "
    "remember what you are told about it, and answer in one sentence."
)


async def ask(model, tools, text):
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
            print(f"  {call.name}({json.dumps(call.args)})")
            try:
                result = await tools.dispatch(call.name, call.args)
                content, is_error = json.dumps(result, default=str), False
            except MemoryException as error:
                content, is_error = f"{type(error).__name__}: {error}", True
            messages.append(
                Message(role="tool", tool_call_id=call.id, content=content, is_error=is_error)
            )


async def main():
    memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
    await memory.initialize()
    await memory.ingest(
        "acme",
        name="Pump P-3 manual",
        text="Replace the mechanical seal on P-3 every six months with kit SK-300.",
    )
    tools = MemoryTools(memory, corpus_id="acme")
    model = AnthropicProvider()
    for text in (
        "Ana replaced the P-3 seal this morning.",
        "Which kit does the P-3 seal take, and who changed it last?",
    ):
        print(f"> {text}")
        print(await ask(model, tools, text))
    print("stories:", [story.text for story in (await memory.list_stories("acme"))[0]])


asyncio.run(main())
