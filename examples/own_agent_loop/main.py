import asyncio
import json

from rfnry_memory import InMemoryStorage, Memory, MemoryTools, Message, ToolSpec

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
        said = [{"type": "text", "text": reply.text}] if reply.text else []
        uses = [
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}
            for call in reply.tool_calls
        ]
        messages.append(Message(role="assistant", content=said + uses, reasoning=reply.reasoning))
        for call in reply.tool_calls:
            result = await tools.dispatch(call.name, call.args)
            print(f"  {call.name}({json.dumps(call.args)})")
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    content=json.dumps(result, default=str),
                    is_error="error" in result,
                )
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
