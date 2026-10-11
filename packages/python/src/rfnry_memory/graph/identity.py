from __future__ import annotations

import re
from uuid import uuid4

from rfnry_memory.models import (
    DecisionModel,
    Entity,
    EntityIn,
    Kind,
    Relation,
    Score,
    ScoreAnswer,
    Usage,
    Window,
    as_decided,
    settle,
)
from rfnry_memory.storage.storage import GraphWrite, Storage, name_key

SAME = Score(
    instructions=(
        "Record a describes an entity found in a passage; record b is an entity already known. "
        "Do the two records describe the same thing?"
    ),
    levels=[
        "different things",
        "related but not the same thing",
        "the same thing under another name",
    ],
)

_WORD = re.compile(r"\w+")


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
        self.mentions: list[tuple[str, Kind, str, str]] = []
        self.known: dict[str, Entity] = {}
        self._by_key: dict[str, set[str]] = {}
        self._order: dict[str, tuple[bool, int]] = {}
        self._stored: dict[str, list[Entity]] = {}

    def note(self, entity: Entity) -> Entity:
        if entity.entity_id not in self.entities:
            self.known.setdefault(entity.entity_id, entity)
        return entity

    def _index(self, entity: Entity, *, aliased: bool) -> None:
        self._order.setdefault(entity.entity_id, (aliased, len(self._order)))
        for key in (entity.key, *map(name_key, entity.aliases)):
            self._by_key.setdefault(key, set()).add(entity.entity_id)

    def named(self, name: str) -> list[Entity]:
        key = name_key(name)
        ids = sorted(self._by_key.get(key, ()), key=self._order.__getitem__)
        known = [self.entities[i] if i in self.entities else self.aliased[i] for i in ids]
        return [e for e in known if e.key == key or key in {name_key(a) for a in e.aliases}]

    async def stored(self, storage: Storage, name: str) -> list[Entity]:
        if name not in self._stored:
            self._stored[name] = await storage.entities_named(self.corpus_id, name)
        return self._stored[name]

    def similar(self, name: str) -> list[Entity]:
        scored = [(similarity(name, [e.name, *e.aliases]), e) for e in self.entities.values()]
        return [
            e
            for overlap, e in sorted(scored, key=lambda item: (-item[0], item[1].entity_id))
            if overlap > 0
        ]

    def create(self, entity: Entity, passage: str) -> None:
        self.entities[entity.entity_id] = entity
        self.evidence[entity.entity_id] = passage
        self._index(entity, aliased=False)

    def alias(self, entity: Entity, alias: str) -> Entity:
        staged = self.entities.get(entity.entity_id)
        if staged is not None:
            if alias not in staged.aliases:
                staged = staged.model_copy(update={"aliases": (*staged.aliases, alias)})
                self.entities[entity.entity_id] = staged
                self._index(staged, aliased=False)
            return staged
        aliases = self.aliases.setdefault(entity.entity_id, [])
        if alias not in entity.aliases and alias not in aliases:
            aliases.append(alias)
        aliased = entity.model_copy(
            update={"aliases": tuple(dict.fromkeys((*entity.aliases, *aliases)))}
        )
        self.aliased[entity.entity_id] = aliased
        self._index(aliased, aliased=True)
        return aliased

    def write(self) -> GraphWrite:
        return GraphWrite(
            entities=list(self.entities.values()),
            aliases=[
                (entity_id, alias)
                for entity_id, aliases in self.aliases.items()
                for alias in aliases
            ],
            relations=self.relations,
            mentions=self.mentions,
            known=[self.aliased.get(i, e) for i, e in self.known.items() if i not in self.entities],
        )


async def resolve(
    storage: Storage,
    decision: DecisionModel | None,
    staging: Staging,
    entity: EntityIn,
    passage: str,
) -> tuple[Entity, Usage]:
    corpus_id = staging.corpus_id
    key, type_ = name_key(entity.name), type_key(entity.type)
    known = [
        e
        for e in [
            *staging.named(entity.name),
            *await staging.stored(storage, entity.name),
        ]
        if e.type == type_
    ]
    if known:
        return staging.note(known[0]), Usage()
    usage = Usage()
    if decision is not None:
        stored = await storage.similar_entities(corpus_id, entity.name, limit=5)
        for candidate in [*staging.similar(entity.name), *stored][:5]:
            evidence = staging.evidence.get(candidate.entity_id)
            if evidence is None:
                found = await storage.mentions(
                    corpus_id,
                    [candidate.entity_id],
                    kinds=("chunk", "story"),
                    filters=None,
                    window=Window(),
                    subject=None,
                )
                passages = await storage.get_passages(corpus_id, [ref for _, ref in found][:1])
                evidence = passages[0].text if passages else ""
            aliases = [*candidate.aliases, *staging.aliases.get(candidate.entity_id, [])]
            state = {
                "a": {"name": entity.name.strip(), "type": type_, "passage": passage[:2000]},
                "b": {
                    "name": candidate.name,
                    "type": candidate.type,
                    "aliases": aliases,
                    "passage": evidence[:2000],
                },
            }
            result = as_decided(await decision.decide(state, {"same": SAME}))
            usage = usage + (result.usage or Usage())
            answer = settle({"same": SAME}, result.answers)["same"]
            if (
                isinstance(answer, ScoreAnswer)
                and answer.probabilities.index(max(answer.probabilities)) == 2
            ):
                return staging.note(staging.alias(candidate, entity.name.strip())), usage
    created = Entity(
        corpus_id=corpus_id, entity_id=uuid4().hex, name=entity.name.strip(), key=key, type=type_
    )
    staging.create(created, passage)
    return created, usage
