import json
import os
from pathlib import Path

import asyncpg
from rfnry_engine import Engine
from rfnry_memory import (
    EntityIn,
    InMemoryStorage,
    Memory,
    MemoryTools,
    PostgresStorage,
    RelationIn,
    Section,
    SourceIn,
    Tools,
)

from providers import AnthropicProvider, VoyageProvider
from utils import neighborhood

PLANTS = json.loads(Path("data/plants.json").read_text())["plants"]

engine = Engine("agents", generation=AnthropicProvider())
memory: Memory


async def setup() -> None:
    global memory
    memory = Memory(
        await storage(), embedding=VoyageProvider(), pillars={"text", "vector", "graph"}
    )
    await memory.initialize()


async def storage() -> InMemoryStorage | PostgresStorage:
    if url := os.environ.get("MEMORY_POSTGRES_URL"):
        return PostgresStorage(await asyncpg.create_pool(url))
    return InMemoryStorage()


def manuals(plant: str, machine: str | None) -> Tools:
    whole_plant = MemoryTools(memory, corpus_id=plant).only("search", "read_source", "neighbors")
    if machine is None:
        return whole_plant
    one_machine = MemoryTools(memory, corpus_id=plant, filters={"metadata.machine": machine})
    return one_machine.only("search", "read_source") + whole_plant.only("neighbors")


async def ask(plant: str, question: str, task: str | None, machine: str | None) -> dict:
    facts = json.dumps(PLANTS[plant], indent=2, ensure_ascii=False)
    run = await engine.run(
        "maintenance",
        scope=plant,
        input=question,
        task=task,
        context=facts,
        tools=manuals(plant, machine),
    )
    return {"reply": run.reply, "run_id": run.run_id}


async def add_manual(
    plant: str,
    name: str,
    machine: str,
    pages: list,
    entities: list[EntityIn],
    relations: list[RelationIn],
) -> dict:
    manual = SourceIn(
        name=name,
        sections=[Section(title=p.title, text=p.text, locator={"page": p.page}) for p in pages],
        metadata={"machine": machine},
        entities=entities,
        relations=relations,
    )
    replaced = await memory.replace(plant, filters={"metadata.machine": machine}, sources=[manual])
    return {
        "source_id": replaced.sources[0].source_id,
        "pages": len(pages),
        "chunks": replaced.sources[0].chunks,
        "replaced": replaced.deleted.sources,
    }


async def neighbors(plant: str, name: str, hops: int) -> dict:
    return neighborhood(await memory.neighbors(plant, name, hops=hops))
