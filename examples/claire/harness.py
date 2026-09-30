from rfnry_engine import Engine
from rfnry_memory import InMemoryStorage, Memory, MemoryTools


async def claire(embedding, generation):
    memory = Memory(InMemoryStorage(), embedding=embedding)
    await memory.initialize()
    engine = Engine("agents", generation=generation)
    tools = MemoryTools(memory, corpus_id="claire", filters={"metadata.user": "ana"})
    return memory, engine, tools


async def conversation(engine, tools, *turns):
    history = []
    for text in turns:
        run = await engine.run("claire", scope="ana", input=text, history=history, tools=tools)
        history = run.conversation
        yield text, run


async def stories(memory):
    stories, _ = await memory.list_stories("claire", history=True, filters={"metadata.user": "ana"})
    return stories
