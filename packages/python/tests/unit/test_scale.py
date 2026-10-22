import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import CodeType
from typing import Any

import pytest

import rfnry_memory
from rfnry_memory.graph.identity import Staging
from rfnry_memory.graph.placement import place
from rfnry_memory.memory import Memory
from rfnry_memory.models import Chunk, Entity, EntityIn, RelationIn, Section, SourceIn
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.rows import name_key

from .fakes import HashEmbedding

PACKAGE = str(Path(rfnry_memory.__file__).parent)

FRAGMENTS = [
    "P-3",
    "p-3",
    "P-30",
    "p - 3",
    "V 101",
    "v  101",
    "V\t101",
    "Motor_1",
    "MOTOR_10",
    "motor_1_run",
    "tag:PLANT/Motor1",
    "tag:plant/motor1_Run",
    "routine:PLANT/Main/R1",
    "routine:plant/main/r12",
    "Ação",
    "AÇÃO-2",
    "İ",
    "a",
    "",
    " ",
    "x.y",
    "x,y",
]


def reference_place(
    chunks: list[Chunk], entities: list[EntityIn], relations: list[RelationIn]
) -> tuple[list[list[EntityIn]], list[list[RelationIn]]]:
    keys = [name_key(c.text) for c in chunks]
    named: list[list[EntityIn]] = [[] for _ in chunks]
    related: list[list[RelationIn]] = [[] for _ in chunks]
    if not chunks:
        return named, related
    for entity_in in entities:
        for index in [i for i, key in enumerate(keys) if name_key(entity_in.name) in key] or [0]:
            named[index].append(entity_in)
    for relation_in in relations:
        ends = (name_key(relation_in.from_name), name_key(relation_in.to_name))
        related[
            next((i for i, key in enumerate(keys) if all(end in key for end in ends)), 0)
        ].append(relation_in)
    return named, related


def reference_named(staging: Staging, name: str) -> list[Entity]:
    key = name_key(name)
    known = [*staging.entities.values(), *staging.aliased.values()]
    return [e for e in known if e.key == key or key in {name_key(a) for a in e.aliases}]


def _variant(rng: random.Random) -> str:
    fragment = rng.choice(FRAGMENTS)
    cased = "".join(c.upper() if rng.random() < 0.3 else c for c in fragment)
    return rng.choice(["", " ", "\n "]) + cased + rng.choice(["", " ", "\t"])


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        corpus_id="acme",
        chunk_id=f"s-{index}",
        source_id="s",
        source_name="s",
        position=index,
        title=None,
        locator={},
        text=text,
        metadata={},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _ids(placed: list[list[Any]]) -> list[list[int]]:
    return [[id(item) for item in here] for here in placed]


@pytest.mark.parametrize("seed", range(300))
def test_placement_matches_the_scan_it_replaces_on_random_inputs(seed: int) -> None:
    rng = random.Random(seed)
    chunks = [
        _chunk(
            i,
            rng.choice([" ", "\n", ", ", "", " / "]).join(
                _variant(rng) for _ in range(rng.randint(0, 6))
            ),
        )
        for i in range(rng.choice([0, 1, 2, rng.randint(3, 12)]))
    ]
    entities = [
        EntityIn(name=_variant(rng), type=rng.choice(["pump", "tag"]))
        for _ in range(rng.choice([0, rng.randint(1, 15)]))
    ]
    entities += rng.sample(entities, min(len(entities), 2))
    relations = [
        RelationIn(from_name=_variant(rng), to_name=_variant(rng), type="USES")
        for _ in range(rng.choice([0, rng.randint(1, 20)]))
    ]
    relations += [
        RelationIn(from_name=a.name, to_name=b.name, type="FEEDS")
        for a, b in zip(entities, reversed(entities), strict=True)
    ]
    named, related = place(chunks, entities, relations)
    expected_named, expected_related = reference_place(chunks, entities, relations)
    assert _ids(named) == _ids(expected_named)
    assert _ids(related) == _ids(expected_related)


@pytest.mark.parametrize("seed", range(100))
def test_staging_names_what_the_scan_it_replaces_names_on_random_inputs(seed: int) -> None:
    rng = random.Random(seed)
    staging = Staging("acme")
    stored = [
        Entity(
            corpus_id="acme",
            entity_id=f"stored-{i}",
            name=(name := _variant(rng).strip() or "x"),
            key=name_key(name),
            type=rng.choice(["pump", "tag"]),
            aliases=tuple(_variant(rng) for _ in range(rng.randint(0, 2))),
        )
        for i in range(6)
    ]
    for step in range(30):
        roll = rng.random()
        if roll < 0.4:
            name = _variant(rng).strip() or "x"
            staging.create(
                Entity(
                    corpus_id="acme",
                    entity_id=f"staged-{step}",
                    name=name,
                    key=name_key(name),
                    type=rng.choice(["pump", "tag"]),
                ),
                "passage",
            )
        elif roll < 0.7 and staging.entities:
            staging.alias(rng.choice(list(staging.entities.values())), _variant(rng))
        elif roll < 0.9:
            staging.alias(rng.choice(stored), _variant(rng))
        else:
            renamed = name_key(_variant(rng))
            moved = rng.choice(stored).model_copy(update={"key": renamed, "aliases": ()})
            staging.alias(moved, _variant(rng))
        for probe in FRAGMENTS + [_variant(rng) for _ in range(5)]:
            assert [id(e) for e in staging.named(probe)] == [
                id(e) for e in reference_named(staging, probe)
            ]


def _plc(sections: int, *, edited: bool = False) -> SourceIn:
    tags = [f"tag:PLANT/Tag{i}" for i in range(sections * 3)]
    routines = [f"routine:PLANT/Main/R{i}" for i in range(sections // 4 + 1)]
    texts: list[Section] = []
    relations: list[RelationIn] = []
    for s in range(sections):
        routine = routines[s // 4]
        used = [tags[(s * 7 + k * 13) % len(tags)] for k in range(8)]
        texts.append(
            Section(
                text=f"[PLC Routine] {routine}\n"
                + "\n".join(f"XIC({t.split('/')[1]})" for t in used)
                + f"\nUses: {', '.join(used)}"
                + ("\nNOP()" if edited and s == 0 else "")
            )
        )
        relations += [
            RelationIn(from_name=routine, to_name=t, type=kind, from_type="routine", to_type="tag")
            for t in used
            for kind in ("USES", "READS")
        ]
    return SourceIn(
        name="line-a.L5X",
        sections=texts,
        metadata={"file": "line-a.L5X"},
        entities=[EntityIn(name=t, type="tag") for t in tags]
        + [EntityIn(name=r, type="routine") for r in routines],
        relations=relations,
    )


async def _loop_iterations(storage: PostgresStorage, sections: int) -> int:
    memory = Memory(storage, embedding=HashEmbedding(), pillars={"text", "vector", "graph"})
    await memory.check()
    monitoring = sys.monitoring
    tool = next(i for i in range(6) if monitoring.get_tool(i) is None)
    iterations = 0

    def jumped(code: CodeType, *_: Any) -> Any:
        nonlocal iterations
        if code.co_filename.startswith(PACKAGE):
            iterations += 1
            return None
        return monitoring.DISABLE

    monitoring.use_tool_id(tool, "rfnry-memory scale test")
    monitoring.register_callback(tool, monitoring.events.JUMP, jumped)
    monitoring.set_events(tool, monitoring.events.JUMP)
    try:
        for edited in (False, True):
            await memory.sync(
                f"plc-{sections}",
                filters={"metadata.file": "line-a.L5X"},
                sources=[_plc(sections, edited=edited)],
            )
    finally:
        monitoring.set_events(tool, monitoring.events.NO_EVENTS)
        monitoring.register_callback(tool, monitoring.events.JUMP, None)
        monitoring.free_tool_id(tool)
        monitoring.restart_events()
    return iterations


async def test_sync_work_grows_with_the_graph_not_with_its_square(storage: PostgresStorage) -> None:
    small = await _loop_iterations(storage, 40)
    large = await _loop_iterations(storage, 160)
    assert large / small < 5
