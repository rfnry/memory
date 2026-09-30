import json
from pathlib import Path

from rfnry_engine import Engine
from rfnry_memory import Hit, InMemoryStorage, Memory, MemoryTools, Section

from providers import decision, embedding, generation

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = {path.stem: path.read_text() for path in (ROOT / "data" / "contracts").glob("*.md")}
AUTHORITIES = json.loads((ROOT / "data" / "authorities.json").read_text())["documents"]
SCOPE = "firm-acme"

engine = Engine(ROOT / "agents", generation=generation())
memory = Memory(InMemoryStorage(), embedding=embedding(), decision=decision())
library = MemoryTools(memory, corpus_id=SCOPE).only("search")


async def setup() -> None:
    await memory.initialize()
    for document in AUTHORITIES:
        name, text = document["name"], document["text"]
        await memory.ingest(SCOPE, name=name, sections=[Section(title=name, text=text)])


async def run(contract: str, question: str) -> dict:
    engine_run = await engine.run(
        "reviewer", scope=SCOPE, input=question, context=CONTRACTS[contract], tools=library
    )
    return {"reply": engine_run.reply, "run_id": engine_run.run_id}


async def search(query: str, judge: bool) -> dict:
    hits = await memory.search(SCOPE, query, top_k=5, judge=judge)
    return {"hits": [hit(found) for found in hits.hits]}


def hit(found: Hit) -> dict:
    row = {"title": found.chunk.title if found.chunk else None, "score": round(found.score, 4)}
    if found.judge:
        row["judge"] = {name: round(p, 2) for name, p in found.judge.model_dump().items()}
    return row | {"text": found.text}
