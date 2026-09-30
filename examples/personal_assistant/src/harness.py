import json
from datetime import date, datetime
from pathlib import Path

from rfnry_engine import Engine
from rfnry_memory import InMemoryStorage, Memory, MemoryTools, Story

from providers import AnthropicProvider, VoyageProvider

ROOT = Path(__file__).resolve().parent.parent
FACTS = json.loads((ROOT / "data" / "me.json").read_text())["facts"]
SCOPE = "jordan"

engine = Engine(ROOT / "agents", generation=AnthropicProvider())
memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
recollection = MemoryTools(memory, corpus_id=SCOPE).only("recall")


async def setup() -> None:
    await memory.initialize()
    for known in FACTS:
        await record_fact(known["about"], known["text"], datetime.fromisoformat(known["since"]))


async def ask(question: str) -> dict:
    run = await engine.run(
        "assistant",
        scope=SCOPE,
        input=question,
        context=f"Today is {date.today().isoformat()}.",
        tools=recollection,
    )
    return {"reply": run.reply, "run_id": run.run_id}


async def record_fact(about: str, text: str, since: datetime | None) -> dict:
    current, _ = await memory.list_stories(SCOPE, filters={"metadata.about": about})
    if not current:
        remembered = await memory.remember(SCOPE, text, valid_from=since, metadata={"about": about})
        return {"fact": fact(remembered.story), "superseded": None}
    new = await memory.supersede(SCOPE, current[0].story_id, text, valid_from=since)
    [old] = await memory.get_stories(SCOPE, [current[0].story_id])
    return {"fact": fact(new.story), "superseded": fact(old)}


def fact(story: Story) -> dict:
    return {
        "about": story.metadata["about"],
        "text": story.text,
        "since": story.valid_from.date().isoformat(),
        "until": story.valid_to.date().isoformat() if story.valid_to else None,
    }
