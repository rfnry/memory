import heapq
import re
from collections.abc import Callable, Iterable
from typing import Any
from uuid import uuid4

from rfnry_memory.errors import InputError
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
    declared_model,
    declared_settings,
    settle,
)
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.rows import GraphWrite, name_key

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


def _words(name: str) -> frozenset[str]:
    return frozenset(_WORD.findall(name.lower()))


class Staging:
    def __init__(self, corpus_id: str, ids: Callable[[], str] | None = None) -> None:
        self.corpus_id = corpus_id
        self.new_id = ids or (lambda: uuid4().hex)
        self.entities: dict[str, Entity] = {}
        self.evidence: dict[str, str] = {}
        self.aliases: dict[str, list[str]] = {}
        self.aliased: dict[str, Entity] = {}
        self._merged: dict[str, dict[str, dict[str, Any]]] = {}
        self.relations: list[Relation] = []
        self.mentions: list[tuple[str, Kind, str, str]] = []
        self.known: dict[str, Entity] = {}
        self._by_key: dict[str, set[str]] = {}
        self._order: dict[str, tuple[bool, int]] = {}
        self._stored: dict[str, list[Entity]] = {}
        self._names: dict[str, list[frozenset[str]]] = {}
        self._postings: dict[str, set[str]] = {}

    def note(self, entity: Entity) -> Entity:
        if entity.entity_id not in self.entities:
            self.known.setdefault(entity.entity_id, entity)
        return entity

    def _index(self, entity: Entity, *, aliased: bool) -> None:
        self._order.setdefault(entity.entity_id, (aliased, len(self._order)))
        for key in (entity.key, *map(name_key, entity.aliases)):
            self._by_key.setdefault(key, set()).add(entity.entity_id)

    def _spell(self, entity: Entity) -> None:
        names = [_words(name) for name in (entity.name, *entity.aliases)]
        self._names[entity.entity_id] = names
        for word in frozenset().union(*names):
            self._postings.setdefault(word, set()).add(entity.entity_id)

    def named(self, name: str) -> list[Entity]:
        key = name_key(name)
        ids = sorted(self._by_key.get(key, ()), key=self._order.__getitem__)
        known = [self.entities[i] if i in self.entities else self.aliased[i] for i in ids]
        return [e for e in known if e.key == key or key in {name_key(a) for a in e.aliases}]

    async def fetch(self, storage: PostgresStorage, names: Iterable[str]) -> None:
        missing = {name_key(name) for name in names} - self._stored.keys()
        if missing:
            self._stored.update(await storage.entities_named(self.corpus_id, sorted(missing)))

    async def stored(self, storage: PostgresStorage, name: str) -> list[Entity]:
        await self.fetch(storage, [name])
        return self._stored[name_key(name)]

    def similar(self, name: str, limit: int) -> list[Entity]:
        wanted = _words(name)
        scored: dict[str, float] = {}
        rarest = sorted(wanted, key=lambda word: (len(self._postings.get(word, ())), word))
        for seen, word in enumerate(rarest):
            best = heapq.nsmallest(limit, ((-score, i) for i, score in scored.items()))
            if len(best) == limit and -best[-1][0] > (len(wanted) - seen) / len(wanted):
                break
            for entity_id in self._postings.get(word, ()):
                if entity_id not in scored:
                    scored[entity_id] = max(
                        len(wanted & words) / len(wanted | words)
                        for words in self._names[entity_id]
                    )
        top = heapq.nsmallest(limit, ((-score, i) for i, score in scored.items()))
        return [self.entities[entity_id] for _, entity_id in top]

    def create(self, entity: Entity, passage: str) -> None:
        if entity.entity_id in self.entities:
            raise InputError(
                f"graph: entity {entity.entity_id!r} is already held; ids must be unique"
            )
        self.entities[entity.entity_id] = entity
        self.evidence[entity.entity_id] = passage
        self._index(entity, aliased=False)
        self._spell(entity)

    def alias(self, entity: Entity, alias: str, by: dict[str, Any]) -> Entity:
        staged = self.entities.get(entity.entity_id)
        if staged is not None:
            if alias not in staged.aliases:
                staged = staged.model_copy(
                    update={
                        "aliases": (*staged.aliases, alias),
                        "aliased_by": {**staged.aliased_by, alias: by},
                    }
                )
                self.entities[entity.entity_id] = staged
                self._index(staged, aliased=False)
                self._spell(staged)
            return staged
        aliases = self.aliases.setdefault(entity.entity_id, [])
        merged = self._merged.setdefault(entity.entity_id, {})
        if alias not in entity.aliases and alias not in aliases:
            aliases.append(alias)
            merged[alias] = by
        aliased = entity.model_copy(
            update={
                "aliases": tuple(dict.fromkeys((*entity.aliases, *aliases))),
                "aliased_by": {**entity.aliased_by, **merged},
            }
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
    storage: PostgresStorage,
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
    consulted: dict[str, Any] | None = None
    if decision is not None:
        stored = await storage.similar_entities(corpus_id, entity.name, limit=5)
        for candidate in [*staging.similar(entity.name, 5), *stored][:5]:
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
            consulted = {
                "model": result.model or declared_model(decision),
                "settings": result.settings
                if result.settings is not None
                else declared_settings(decision),
            }
            answer = settle({"same": SAME}, result.answers)["same"]
            if (
                isinstance(answer, ScoreAnswer)
                and answer.probabilities.index(max(answer.probabilities)) == 2
            ):
                return staging.note(staging.alias(candidate, entity.name.strip(), consulted)), usage
    created = Entity(
        corpus_id=corpus_id,
        entity_id=staging.new_id(),
        name=entity.name.strip(),
        key=key,
        type=type_,
        resolved_by=consulted,
    )
    staging.create(created, passage)
    return created, usage
