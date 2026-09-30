import json
from datetime import date
from pathlib import Path

from rfnry_engine import Engine, Tool
from rfnry_memory import InMemoryStorage, Memory, MemoryTools, Section, SourceIn

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
memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
help_articles = MemoryTools(memory, corpus_id=SCOPE).only("search")


async def setup() -> None:
    await memory.initialize()
    for slug, article in ARTICLES.items():
        await add_article(slug, article["title"], article["text"])


async def add_article(slug: str, title: str, text: str) -> dict:
    ingested = await memory.ingest(
        SCOPE, name=title, sections=[Section(title=title, text=text)], metadata={"slug": slug}
    )
    return {"slug": slug, "source_id": ingested.source.source_id}


async def replace_article(slug: str, title: str, text: str) -> dict:
    new = SourceIn(name=title, sections=[Section(title=title, text=text)], metadata={"slug": slug})
    replaced = await memory.replace(SCOPE, filters={"metadata.slug": slug}, sources=[new])
    source = replaced.sources[0]
    return {"slug": slug, "source_id": source.source_id, "replaced": replaced.deleted.sources}


async def ask(question: str) -> dict:
    run = await engine.run(
        "support",
        scope=SCOPE,
        input=question,
        context=f"Today is {date.today().isoformat()}.",
        tools=help_articles,
    )
    return {"reply": run.reply, "run_id": run.run_id}
