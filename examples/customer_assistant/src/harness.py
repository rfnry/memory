import json
from datetime import date
from pathlib import Path

from rfnry_engine import Engine, Tool
from rfnry_memory import Memory, MemoryTools, PostgresStorage, Section, SourceIn, Tools

from providers import AnthropicProvider, VoyageProvider
from tools import lookup_order

ROOT = Path(__file__).resolve().parent.parent
ARTICLES = json.loads((ROOT / "data" / "articles.json").read_text())["articles"]
SCOPE = "acme-home"

engine = Engine(
    ROOT / "agents",
    generation=AnthropicProvider(),
    tools=[Tool("lookup_order", body=lookup_order)],
)
memory: Memory
help_articles: Tools


async def setup(pool) -> None:
    global memory, help_articles
    memory = Memory(PostgresStorage(pool), embedding=VoyageProvider())
    help_articles = MemoryTools(memory, corpus_id=SCOPE).only("search")
    await memory.initialize()
    await memory.delete_corpus(SCOPE)
    for slug, article in ARTICLES.items():
        await ingest(slug, article["title"], article["text"])


async def ingest(slug: str, title: str, text: str) -> dict:
    ingested = await memory.ingest(
        SCOPE, name=title, sections=[Section(title=title, text=text)], metadata={"slug": slug}
    )
    return {"slug": slug, "source_id": ingested.source.source_id}


async def sync(slug: str, title: str, text: str) -> dict:
    new = SourceIn(name=title, sections=[Section(title=title, text=text)], metadata={"slug": slug})
    synced = await memory.sync(SCOPE, filters={"metadata.slug": slug}, sources=[new])
    source = synced.sources[0]
    return {"slug": slug, "source_id": source.source_id, "replaced": synced.deleted.sources}


async def run(question: str) -> dict:
    engine_run = await engine.run(
        "support",
        scope=SCOPE,
        input=question,
        context=f"Today is {date.today().isoformat()}.",
        tools=help_articles,
    )
    return {"reply": engine_run.reply, "run_id": engine_run.run_id}
