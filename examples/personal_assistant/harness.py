import json
from datetime import date, datetime
from pathlib import Path

from rfnry_engine import Document, Engine, Image, Message, Text
from rfnry_memory import InMemoryStorage, Memory, MemoryTools, Section

from providers import AnthropicProvider, VoyageProvider
from utils import fact

SCOPE = "jordan"
ME = json.loads(Path("data/me.json").read_text())

engine = Engine("agents", generation=AnthropicProvider())
memory = Memory(InMemoryStorage(), embedding=VoyageProvider())
recollection = MemoryTools(memory, corpus_id=SCOPE).only("recall", "search", "read_source")
sessions: dict[str, list[Message]] = {}


async def setup() -> None:
    await memory.initialize()
    for known in ME["facts"]:
        await record_fact(known["about"], known["text"], datetime.fromisoformat(known["since"]))


def attachment(media_type: str, name: str, data: bytes) -> Image | Document:
    if media_type.startswith("image/"):
        return Image(media_type=media_type, data=data)
    return Document(media_type=media_type, name=name, data=data)


async def ask(question: str, session: str | None, attached: Image | Document | None) -> dict:
    history = sessions.get(session, []) if session else []
    parts = [Text(text=question), *([attached] if attached else [])]
    run = await engine.run(
        "assistant",
        scope=SCOPE,
        input=parts,
        history=history,
        context=f"Today is {date.today().isoformat()}.",
        tools=recollection,
    )
    if session:
        sessions[session] = run.conversation
    return {"reply": run.reply, "run_id": run.run_id}


async def add_note(title: str, text: str) -> dict:
    ingested = await memory.ingest(SCOPE, name=title, sections=[Section(title=title, text=text)])
    return {"source_id": ingested.source.source_id, "title": title}


async def record_fact(about: str, text: str, since: datetime | None) -> dict:
    current, _ = await memory.list_stories(
        SCOPE, subject="jordan", filters={"metadata.about": about}
    )
    if not current:
        remembered = await memory.remember(
            SCOPE, text, subject="jordan", valid_from=since, metadata={"about": about}
        )
        return {"fact": fact(remembered.story), "superseded": None}
    old = current[0].story_id
    new = await memory.supersede(SCOPE, old, text, valid_from=since)
    [closed] = await memory.get_stories(SCOPE, [old])
    return {"fact": fact(new.story), "superseded": fact(closed)}
