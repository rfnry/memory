from __future__ import annotations

import re
from uuid import uuid4

from rfnry_memory.models import (
    Decided,
    DecisionModel,
    Entity,
    EntityIn,
    Kind,
    Relation,
    Score,
    ScoreAnswer,
    Usage,
    Window,
    settle,
)
from rfnry_memory.storage.storage import GraphWrite, Storage

SAME = Score(
    instructions=(
        "Record a describes an entity found in a passage; record b is an entity already known. "
        "Do the two records describe the same thing?"
    ),
    levels=["different things", "related but not the same thing", "the same thing under another name"],
)

_WORD = re.compile(r"\w+")


def name_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def type_key(entity_type: str) -> str:
    return entity_type.strip().lower()


def similarity(name: str, names: list[str]) -> float:
    wanted = set(_WORD.findall(name.lower()))
    best = 0.0
    for other in names:
        tokens = set(_WORD.findall(other.lower()))
        best = max(best, len(wanted & tokens) / max(len(wanted | tokens), 1))
    return best


class Staging:
    def __init__(self, corpus_id: str) -> None:
        self.corpus_id = corpus_id
        self.entities: dict[str, Entity] = {}
        self.evidence: dict[str, str] = {}
        self.aliases: dict[str, list[str]] = {}
        self.aliased: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.mentions: list[tuple[str, Kind, str]] = []

    def created(self, key: str, type_: str) -> Entity | None:
        return next((e for e in self.entities.values() if e.key == key and e.type == type_), None)

    def named(self, name: str) -> list[Entity]:
        key = name_key(name)
        known = [*self.entities.values(), *self.aliased.values()]
        return [e for e in known if e.key == key or key in {name_key(a) for a in e.aliases}]

    def similar(self, name: str) -> list[Entity]:
        scored = [(similarity(name, [e.name, *e.aliases]), e) for e in self.entities.values()]
        return [e for overlap, e in sorted(scored, key=lambda item: (-item[0], item[1].entity_id)) if overlap > 0]

    def create(self, entity: Entity, passage: str) -> None:
        self.entities[entity.entity_id] = entity
        self.evidence[entity.entity_id] = passage

    def alias(self, entity: Entity, alias: str) -> Entity:
        staged = self.entities.get(entity.entity_id)
        if staged is not None:
            if alias not in staged.aliases:
                staged = staged.model_copy(update={"aliases": (*staged.aliases, alias)})
                self.entities[entity.entity_id] = staged
            return staged
        aliases = self.aliases.setdefault(entity.entity_id, [])
        if alias not in entity.aliases and alias not in aliases:
            aliases.append(alias)
        aliased = entity.model_copy(update={"aliases": tuple(dict.fromkeys((*entity.aliases, *aliases)))})
        self.aliased[entity.entity_id] = aliased
        return aliased

    def write(self) -> GraphWrite:
        return GraphWrite(
            entities=list(self.entities.values()),
            aliases=[(entity_id, alias) for entity_id, aliases in self.aliases.items() for alias in aliases],
            relations=self.relations,
            mentions=self.mentions,
        )


async def resolve(
    storage: Storage, decision: DecisionModel | None, staging: Staging, entity: EntityIn, passage: str
) -> tuple[Entity, bool, Usage]:
    corpus_id = staging.corpus_id
    key, type_ = name_key(entity.name), type_key(entity.type)
    existing = staging.created(key, type_) or await storage.find_entity(corpus_id, key, type_)
    if existing is not None:
        return existing, False, Usage()
    usage = Usage()
    if decision is not None:
        stored = await storage.similar_entities(corpus_id, entity.name, limit=5)
        for candidate in [*staging.similar(entity.name), *stored][:5]:
            evidence = staging.evidence.get(candidate.entity_id)
            if evidence is None:
                refs = await storage.mentions(corpus_id, [candidate.entity_id], kinds=("chunk", "story"), filters=None, window=Window(), subject=None)
                passages = await storage.get_passages(corpus_id, refs[:1])
                evidence = passages[0].text if passages else ""
            aliases = [*candidate.aliases, *staging.aliases.get(candidate.entity_id, [])]
            state = {
                "a": {"name": entity.name.strip(), "type": type_, "passage": passage[:2000]},
                "b": {"name": candidate.name, "type": candidate.type, "aliases": aliases, "passage": evidence[:2000]},
            }
            decided = await decision.decide(state, {"same": SAME})
            if not isinstance(decided, Decided):
                decided = Decided(answers=decided)
            usage = usage + (decided.usage or Usage())
            answer = settle({"same": SAME}, decided.answers)["same"]
            if isinstance(answer, ScoreAnswer) and answer.probabilities.index(max(answer.probabilities)) == 2:
                return staging.alias(candidate, entity.name.strip()), False, usage
    created = Entity(corpus_id=corpus_id, entity_id=uuid4().hex, name=entity.name.strip(), key=key, type=type_)
    staging.create(created, passage)
    return created, True, usage
