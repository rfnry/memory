import __future__

import ast
import inspect
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import pytest

import rfnry_memory
from rfnry_memory import (
    Check,
    CheckAnswer,
    EmbeddingModel,
    EntityIn,
    InMemoryStorage,
    Memory,
    QuestionLike,
    RelationIn,
    ScoreAnswer,
    Section,
    State,
)

from .fakes import Fixed, HashEmbedding, structured_reply

DOCS = Path(__file__).parents[4] / "docs" / "memory.md"

ILLUSTRATIVE = {
    "memory = Memory(PostgresStorage(pool)": "needs a Postgres pool; the integration tier runs PostgresStorage",
    "memory = Memory(PostgresStorage(await asyncpg": "needs a Postgres pool; the integration tier runs PostgresStorage",
    "run = await engine.run(": "needs rfnry-engine, which this package does not depend on",
}


def sections() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    heading = ""
    block: list[str] | None = None
    for line in DOCS.read_text(encoding="utf-8").splitlines():
        if block is not None:
            if line == "```":
                found.setdefault(heading, []).append("\n".join(block))
                block = None
            else:
                block.append(line)
        elif line.startswith("#"):
            heading = line.lstrip("# ")
        elif line == "```python":
            block = []
    return found


def runnable(block: str) -> ast.Module:
    asserted = "\n".join(
        f"assert {line.split('#')[0].strip()}" if re.search(r"\s# True$", line) else line
        for line in block.splitlines()
    )
    tree = ast.parse(asserted)
    lines = asserted.splitlines()
    tree.body = [
        node
        for node in tree.body
        if not any(lines[node.lineno - 1].startswith(prefix) for prefix in ILLUSTRATIVE)
    ]
    return tree


class NothingToExtract:
    async def generate(self, system: str, messages: list[Any], tools: list[Any]) -> Any:
        return structured_reply({"entities": [], "relations": []})


def undecided(state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
    return {
        name: CheckAnswer(probability=0.5)
        if isinstance(question, Check)
        else ScoreAnswer(score=0.0, probabilities=[1.0, 0.0, 0.0])
        for name, question in questions.items()
    }


async def given() -> dict[str, Any]:
    emb, llm, dm, storage = HashEmbedding(), NothingToExtract(), Fixed(undecided), InMemoryStorage()
    memory = Memory(
        storage,
        embedding=emb,
        generation=llm,
        decision=dm,
        pillars={"text", "vector", "graph"},
        relation_types=["CONNECTS_TO", "PART_OF", "OWNS"],
    )
    await memory.initialize()
    sheet = [
        Section(
            title="Feed line",
            text="The feed line runs from V-101 to P-3.",
            locator={"sheet": 7, "region": "A2"},
        ),
        Section(
            title="Return line",
            text="The return line leaves P-3 for the tank.",
            locator={"sheet": 7, "region": "C4"},
        ),
    ]
    ingested = await memory.ingest(
        "acme",
        name="P&ID sheet 7",
        sections=sheet,
        metadata={"line": "A", "rev": "3"},
        entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump")],
        relations=[RelationIn(from_name="V-101", to_name="P-3", type="CONNECTS_TO")],
    )

    async def now(args: dict[str, Any]) -> str:
        return datetime.now(UTC).isoformat()

    return {
        **{name: getattr(rfnry_memory, name) for name in rfnry_memory.__all__},
        "Protocol": Protocol,
        "Mapping": Mapping,
        "datetime": datetime,
        "UTC": UTC,
        "sha256": sha256,
        "HashEmbedding": HashEmbedding,
        "emb": emb,
        "llm": llm,
        "dm": dm,
        "storage": storage,
        "memory": memory,
        "sections": sheet,
        "source_id": ingested.source.source_id,
        "query": "seal replacement",
        "now": now,
        "text": "Replace the pump seal yearly.",
        "manual_text": "The seal is replaced every year.",
        "feed_text": "The feed line runs from V-101 to P-3.",
        "return_text": "The return line leaves P-3 for the tank.",
        "rungs": [Section(title="Rung 0", text="XIC(Start) OTE(Motor_1)", locator={"rung": 0})],
        "tags_text": "Motor_1 BOOL",
    }


def methods(cls: type) -> dict[str, dict[str, Any]]:
    return {
        name: dict(value.__annotations__)
        for name, value in vars(cls).items()
        if inspect.isfunction(value) and not name.startswith("_")
    }


async def then_models(namespace: dict[str, Any]) -> None:
    for name in ("EmbeddingModel", "GenerationModel", "DecisionModel"):
        documented, real = namespace[name], getattr(rfnry_memory, name)
        assert documented.__annotations__ == real.__annotations__, name
        assert methods(documented) == methods(real), name
    fake = namespace["HashEmbedding"]()
    assert isinstance(fake, EmbeddingModel)
    assert [len(v) for v in await fake.embed(["a", "b"], purpose="document")] == [
        fake.dimension
    ] * 2


async def then_tools(namespace: dict[str, Any]) -> None:
    filed = [
        s for s in (await namespace["memory"].list_sources("acme"))[0] if s.name == "seal notes"
    ]
    assert [(s.text, s.metadata) for s in filed] == [(namespace["manual_text"], {"line": "A"})]
    hits = await namespace["tools"].dispatch(
        "search", {"query": "seal replaced every year", "kinds": ["chunk"]}
    )
    assert hits["hits"][0]["source_id"] == filed[0].source_id


async def then_sources(namespace: dict[str, Any]) -> None:
    synced = namespace["synced"]
    filed, _ = await namespace["memory"].list_sources(
        "acme", filters={"metadata.file": "line-a.L5X"}
    )
    assert sorted(s.name for s in filed) == ["Routine MainRoutine", "Tags"]
    assert {s.source_id for s in filed} == {s.source_id for s in synced.sources}
    assert synced.deleted.clean


async def then_time(namespace: dict[str, Any]) -> None:
    memory = namespace["memory"]
    now = await memory.recall("acme", "go-live date", subject="project:apollo")
    assert "March 3" in now.hits[0].text
    then = await memory.recall(
        "acme",
        "go-live date",
        subject="project:apollo",
        as_of=datetime(2026, 1, 20, tzinfo=UTC),
    )
    assert "February 20" in then.hits[0].text
    told = namespace["told"]
    known = await memory.recall("acme", "go-live date", subject="project:apollo", known_at=told)
    [first] = await memory.get_stories("acme", [namespace["first"].story_id])
    assert first.closed_at is not None
    assert ("February 20" if first.closed_at > told else "March 3") in known.hits[0].text


THEN = {"Models": then_models, "Sources": then_sources, "Time": then_time, "Tools": then_tools}
FLAGS = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT | __future__.annotations.compiler_flag


@pytest.mark.parametrize("heading", list(sections()))
async def test_every_docs_example_runs(heading: str) -> None:
    namespace = await given()
    for block in sections()[heading]:
        result = eval(
            compile(runnable(block), f"memory.md: {heading}", "exec", flags=FLAGS), namespace
        )
        if inspect.iscoroutine(result):
            await result
    if heading in THEN:
        await THEN[heading](namespace)
