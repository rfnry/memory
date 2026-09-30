import asyncio
import json
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path

from rfnry_engine import Engine, Event, Message, TextDelta, Tool
from rfnry_memory import InMemoryStorage, Memory, MemoryTools, Section, SourceIn

from providers import AnthropicProvider, VoyageProvider
from tools import lookup_order

SCOPE = "acme-home"
ARTICLES = json.loads(Path("data/articles.json").read_text())["articles"]

engine = Engine(
    "agents", generation=AnthropicProvider(), tools=[Tool("lookup_order", body=lookup_order)]
)
memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
help_articles = MemoryTools(memory, corpus_id=SCOPE).only("search", "read_source")
sessions: dict[str, list[Message]] = {}


async def setup() -> None:
    await memory.initialize()
    for slug, article in ARTICLES.items():
        await add_article(slug, article["title"], article["text"])


async def has_article(slug: str) -> bool:
    found, _ = await memory.list_sources(SCOPE, filters={"metadata.slug": slug}, limit=1)
    return bool(found)


async def add_article(slug: str, title: str, text: str) -> dict:
    ingested = await memory.ingest(
        SCOPE, name=title, sections=[Section(title=title, text=text)], metadata={"slug": slug}
    )
    return {"slug": slug, "source_id": ingested.source.source_id}


async def replace_article(slug: str, title: str, text: str) -> dict:
    new = SourceIn(name=title, sections=[Section(title=title, text=text)], metadata={"slug": slug})
    replaced = await memory.replace(SCOPE, filters={"metadata.slug": slug}, sources=[new])
    return {
        "slug": slug,
        "source_id": replaced.sources[0].source_id,
        "replaced": replaced.deleted.sources,
    }


async def ask(question: str, session: str | None) -> dict:
    run = await answer(question, session)
    return {"reply": run.reply, "run_id": run.run_id}


async def ask_stream(question: str, session: str | None) -> AsyncIterator[dict]:
    deltas: asyncio.Queue[str | None] = asyncio.Queue()

    async def on_event(event: Event) -> None:
        if isinstance(event, TextDelta):
            deltas.put_nowait(event.text)

    work = asyncio.create_task(answer(question, session, on_event))
    work.add_done_callback(lambda _: deltas.put_nowait(None))
    while (text := await deltas.get()) is not None:
        yield {"text": text}
    run = await work
    yield {"done": True, "run_id": run.run_id}


async def answer(question: str, session: str | None, on_event=None):
    history = sessions.get(session, []) if session else []
    run = await engine.run(
        "support",
        scope=SCOPE,
        input=question,
        history=history,
        context=f"Today is {date.today().isoformat()}.",
        on_event=on_event,
        tools=help_articles,
    )
    if session:
        sessions[session] = run.conversation
    return run
