import json
from pathlib import Path

from rfnry_engine import Engine
from rfnry_memory import EntityIn, InMemoryStorage, Memory, MemoryTools, RelationIn, Section

from providers import AnthropicProvider

ROOT = Path(__file__).resolve().parent.parent
PLANT = json.loads((ROOT / "data" / "plant.json").read_text())
MANUAL = json.loads((ROOT / "data" / "m-13.json").read_text())
SCOPE = "plant-reno"

engine = Engine(ROOT / "agents", generation=AnthropicProvider())
memory = Memory(InMemoryStorage(), pillars={"text", "graph"})
graph = MemoryTools(memory, corpus_id=SCOPE).only("neighbors")


async def setup() -> None:
    await memory.initialize()
    await memory.ingest(
        SCOPE,
        name=MANUAL["name"],
        sections=[
            Section(title=page["title"], text=page["text"], locator={"page": page["page"]})
            for page in MANUAL["pages"]
        ],
        entities=[EntityIn(**entity) for entity in MANUAL["entities"]],
        relations=[RelationIn(**relation) for relation in MANUAL["relations"]],
    )


async def run(question: str) -> dict:
    facts = json.dumps(PLANT, indent=2, ensure_ascii=False)
    engine_run = await engine.run(
        "maintenance", scope=SCOPE, input=question, context=facts, tools=graph
    )
    return {"reply": engine_run.reply, "run_id": engine_run.run_id}


async def neighbors(name: str, hops: int) -> dict:
    hood = await memory.neighbors(SCOPE, name, hops=hops)
    names = {entity.entity_id: entity.name for entity in hood.entities}
    return {
        "entities": [{"name": e.name, "type": e.type} for e in hood.entities],
        "relations": [f"{names[r.from_id]} {r.type} {names[r.to_id]}" for r in hood.relations],
        "pages": [
            {"page": hit.chunk.locator["page"], "title": hit.chunk.title}
            for hit in hood.passages
            if hit.chunk
        ],
    }
