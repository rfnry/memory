import json
from datetime import datetime
from pathlib import Path

from rfnry_engine import Engine
from rfnry_memory import InMemoryStorage, Memory, MemoryTools, Section

from providers import decision, embedding, generation
from utils import fact

CALLBACK = "Triage is not sure who should take this: a nurse will call the patient back."
PROTOCOLS = json.loads(Path("data/protocols.json").read_text())
PATIENTS = json.loads(Path("data/patients.json").read_text())

engine = Engine("agents", generation=generation(), decision=decision())
memory = Memory(InMemoryStorage(), embedding=embedding(), decision=decision())


async def setup() -> None:
    await memory.initialize()
    for protocol in PROTOCOLS["protocols"]:
        await add_protocol(PROTOCOLS["clinic"], protocol["title"], protocol["text"])
    for patient, facts in PATIENTS["patients"].items():
        for known in facts:
            since = datetime.fromisoformat(known["since"])
            await record_fact(PATIENTS["clinic"], patient, known["about"], known["text"], since)


async def intake(clinic: str, message: str, patient: str | None) -> dict:
    state = {"message": message} | ({"patient": patient} if patient else {})
    tools = MemoryTools(memory, corpus_id=clinic).only("search", "recall")
    run = await engine.run("triage", scope=clinic, input=state, tools=tools)
    routed = run.routed_to
    return {
        "answers": {
            qid: answer.model_dump(exclude={"type"}) for qid, answer in run.answers.items()
        },
        "routed_to": routed.agent if routed else None,
        "reply": routed.reply if routed else CALLBACK,
        "run_id": run.run_id,
    }


async def add_protocol(clinic: str, title: str, text: str) -> dict:
    ingested = await memory.ingest(clinic, name=title, sections=[Section(title=title, text=text)])
    return {"source_id": ingested.source.source_id, "title": title}


async def record_fact(
    clinic: str, patient: str, about: str, text: str, since: datetime | None
) -> dict:
    subject = f"patient:{patient}"
    current, _ = await memory.list_stories(
        clinic, subject=subject, filters={"metadata.about": about}
    )
    if not current:
        remembered = await memory.remember(
            clinic, text, subject=subject, valid_from=since, metadata={"about": about}
        )
        return {"fact": fact(remembered.story), "superseded": None}
    old = current[0].story_id
    new = await memory.supersede(clinic, old, text, valid_from=since)
    [closed] = await memory.get_stories(clinic, [old])
    return {"fact": fact(new.story), "superseded": fact(closed)}


async def facts(clinic: str, patient: str, at: datetime | None) -> dict:
    stories, _ = await memory.list_stories(clinic, subject=f"patient:{patient}", as_of=at)
    return {"patient": patient, "at": at, "facts": [fact(story) for story in stories]}
