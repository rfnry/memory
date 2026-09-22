from __future__ import annotations

import re
from uuid import uuid4

from rfnry_memory.models import (
    Decided,
    DecisionModel,
    Entity,
    EntityIn,
    Score,
    ScoreAnswer,
    Usage,
    Window,
    settle,
)
from rfnry_memory.storage.storage import Storage

SAME = Score(
    instructions=(
        "Record a describes an entity found in a passage; record b is an entity already known. "
        "Do the two records describe the same thing?"
    ),
    levels=["different things", "related but not the same thing", "the same thing under another name"],
)


def name_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def type_key(entity_type: str) -> str:
    return entity_type.strip().lower()


async def resolve(
    storage: Storage, decision: DecisionModel | None, corpus_id: str, entity: EntityIn, passage: str
) -> tuple[Entity, bool, Usage]:
    key, type_ = name_key(entity.name), type_key(entity.type)
    existing = await storage.find_entity(corpus_id, key, type_)
    if existing is not None:
        return existing, False, Usage()
    usage = Usage()
    if decision is not None:
        for candidate in await storage.similar_entities(corpus_id, entity.name, limit=5):
            refs = await storage.mentions(corpus_id, [candidate.entity_id], kinds=("chunk", "story"), filters=None, window=Window())
            evidence = await storage.get_passages(corpus_id, refs[:1])
            state = {
                "a": {"name": entity.name.strip(), "type": type_, "passage": passage[:2000]},
                "b": {
                    "name": candidate.name, "type": candidate.type, "aliases": list(candidate.aliases),
                    "passage": evidence[0].text[:2000] if evidence else "",
                },
            }
            decided = await decision.decide(state, {"same": SAME})
            if not isinstance(decided, Decided):
                decided = Decided(answers=decided)
            usage = usage + (decided.usage or Usage())
            answer = settle({"same": SAME}, decided.answers)["same"]
            if isinstance(answer, ScoreAnswer) and answer.probabilities.index(max(answer.probabilities)) == 2:
                await storage.add_alias(corpus_id, candidate.entity_id, entity.name.strip())
                merged = await storage.entities_by_id(corpus_id, [candidate.entity_id])
                return merged[0], False, usage
    created = Entity(corpus_id=corpus_id, entity_id=uuid4().hex, name=entity.name.strip(), key=key, type=type_)
    await storage.put_entity(corpus_id, created)
    return created, True, usage
