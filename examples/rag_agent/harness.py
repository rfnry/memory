from rfnry_engine import Engine
from rfnry_memory import InMemoryStorage, Memory, MemoryTools


async def plant(embedding):
    memory = Memory(InMemoryStorage(), embedding=embedding)
    await memory.initialize()
    await memory.ingest(
        "acme",
        name="Pump P-3 manual",
        text="Replace the mechanical seal on P-3 every six months with kit SK-300.",
    )
    await memory.remember("acme", "Ana owns the P-3 maintenance plan.", subject="pump:P-3")
    return memory


def assistant(memory, generation):
    engine = Engine("agents", generation=generation)
    tools = MemoryTools(memory, corpus_id="acme").only("search", "recall")
    return engine, tools


async def answer(engine, tools, question):
    return await engine.run("assistant", scope="acme", input=question, tools=tools)
