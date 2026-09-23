from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from rfnry_memory.chunking.chunker import Piece, chunk_text
from rfnry_memory.errors import (
    ConfigurationError,
    InputError,
    ModelError,
    NotFoundError,
    StorageError,
)
from rfnry_memory.filters import validate_filters
from rfnry_memory.fusion import fuse_by_rank
from rfnry_memory.graph.extraction import extract
from rfnry_memory.graph.identity import Staging, name_key, resolve, type_key
from rfnry_memory.judge import judge as judge_hits
from rfnry_memory.models import (
    Chunk,
    DecisionModel,
    Deleted,
    EmbeddingModel,
    EntityIn,
    GenerationModel,
    Hit,
    Hits,
    Ingested,
    Kind,
    Leg,
    Neighborhood,
    PillarReport,
    Ref,
    Relation,
    RelationIn,
    Residue,
    Section,
    Source,
    Story,
    Usage,
    Window,
)
from rfnry_memory.storage.storage import Storage, text_hash

PILLARS = frozenset({"text", "vector", "graph"})


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _window(as_of: datetime | None, since: datetime | None, until: datetime | None, history: bool) -> Window:
    if history and as_of is not None:
        raise InputError("memory: history=True reads every moment; it takes no as_of")
    moment = None if history else _aware(as_of) or _now()
    return Window(as_of=moment, since=_aware(since), until=_aware(until))


def _has_nul(value: Any) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, BaseModel):
        return _has_nul(value.model_dump())
    if isinstance(value, dict):
        return any(_has_nul(k) or _has_nul(v) for k, v in value.items())
    if isinstance(value, list | tuple | set | frozenset):
        return any(_has_nul(v) for v in value)
    return False


def _corpus(corpus_id: str, *values: Any) -> str:
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise InputError("memory: corpus_id is required")
    if _has_nul((corpus_id, values)):
        raise InputError("memory: text must not contain a NUL character")
    return corpus_id


def _endpoints(relation_in: RelationIn) -> list[tuple[str, str | None, str]]:
    return [(relation_in.from_name, relation_in.from_type, "from_type"), (relation_in.to_name, relation_in.to_type, "to_type")]


def _endpoint_error(relation_in: RelationIn, name: str, wanted: str | None, side: str, types: set[str]) -> InputError | None:
    if wanted is not None:
        types = types & {type_key(wanted)}
    ends = f"{relation_in.from_name!r} -> {relation_in.to_name!r}"
    if not types:
        return InputError(f"memory: relation {ends} names an unknown entity")
    if len(types) > 1:
        return InputError(f"memory: relation {ends} names {name!r}, which has several types: {sorted(types)}; pass {side}")
    return None


class Memory:
    def __init__(
        self,
        storage: Storage,
        *,
        embedding: EmbeddingModel | None = None,
        generation: GenerationModel | None = None,
        decision: DecisionModel | None = None,
        pillars: set[str] | None = None,
        relation_types: list[str] | None = None,
        weights: dict[str, float] | None = None,
        chunk_chars: int = 2000,
        overlap_chars: int = 200,
        fetch_multiplier: int = 4,
    ) -> None:
        self.storage = storage
        self.embedding = embedding
        self.generation = generation
        self.decision = decision
        default = {"text"} | ({"vector"} if embedding else set()) | ({"graph"} if generation else set())
        self.pillars = set(pillars) if pillars is not None else default
        unknown = self.pillars - PILLARS
        if unknown:
            raise ConfigurationError(f"memory: unknown pillars: {sorted(unknown)}")
        if "vector" in self.pillars and embedding is None:
            raise ConfigurationError("memory: the vector pillar needs an embedding model")
        if "graph" in self.pillars and generation is not None and not relation_types:
            raise ConfigurationError("memory: extraction needs relation_types")
        self.relation_types = list(relation_types or [])
        self.weights = {pillar: 1.0 for pillar in PILLARS} | dict(weights or {})
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.fetch_multiplier = fetch_multiplier

    async def initialize(self) -> None:
        await self.storage.initialize(dimension=self.embedding.dimension if self.embedding else None)

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if "vector" not in self.pillars or self.embedding is None or not texts:
            return None
        return await self.embedding.embed(texts)

    def _model(self) -> str | None:
        return self.embedding.model if self.embedding and "vector" in self.pillars else None

    async def ingest(
        self,
        corpus_id: str,
        *,
        name: str,
        text: str | None = None,
        sections: list[Section] | None = None,
        metadata: dict[str, Any] | None = None,
        entities: list[EntityIn] | None = None,
        relations: list[RelationIn] | None = None,
    ) -> Ingested:
        corpus_id = _corpus(corpus_id, name, text, sections, metadata, entities, relations)
        if (text is None) == (sections is None):
            raise InputError("memory: ingest takes exactly one of text= or sections=")
        started = time.monotonic()
        source_id = uuid4().hex
        pieces: list[tuple[Piece, dict[str, Any]]] = []
        if text is not None:
            pieces = [(piece, {}) for piece in chunk_text(text, chunk_chars=self.chunk_chars, overlap_chars=self.overlap_chars)]
            full_text = text.strip()
        else:
            for section in sections or []:
                for piece in chunk_text(section.text, chunk_chars=self.chunk_chars, overlap_chars=self.overlap_chars):
                    pieces.append((Piece(title=section.title or piece.title, text=piece.text), dict(section.locator)))
            full_text = "\n\n".join(s.text.strip() for s in sections or [])
        chunks = [
            Chunk(corpus_id=corpus_id, chunk_id=f"{source_id}-{index}", source_id=source_id, position=index,
                  title=piece.title, locator=locator, text=piece.text)
            for index, (piece, locator) in enumerate(pieces)
        ]
        source = Source(corpus_id=corpus_id, source_id=source_id, name=name, metadata=dict(metadata or {}),
                        text=full_text, created_at=_now(), chunks=len(chunks))
        await self._check_facts(corpus_id, entities, relations)
        vectors = await self._embed([c.text for c in chunks])
        staging = Staging(corpus_id)
        entity_count = relation_count = 0
        usage = Usage()
        for chunk in chunks:
            e, r, u = await self._stage_graph(staging, "chunk", chunk.chunk_id, chunk.text, entities, relations)
            entity_count, relation_count, usage = entity_count + e, relation_count + r, usage + u
            entities, relations = None, None
        await self.storage.put_source(corpus_id, source, chunks, vectors, self._model(), graph=staging.write())
        return Ingested(source=source, chunks=len(chunks), entities=entity_count, relations=relation_count,
                        usage=usage, duration_ms=int((time.monotonic() - started) * 1000))

    async def _check_facts(self, corpus_id: str, entities: list[EntityIn] | None, relations: list[RelationIn] | None) -> None:
        if "graph" not in self.pillars:
            if entities or relations:
                raise ConfigurationError("memory: graph facts were supplied but the graph pillar is off")
            return
        supplied: dict[str, set[str]] = {}
        for entity_in in entities or []:
            supplied.setdefault(name_key(entity_in.name), set()).add(type_key(entity_in.type))
        for relation_in in relations or []:
            for name, wanted, side in _endpoints(relation_in):
                stored = {e.type for e in await self.storage.entities_named(corpus_id, name)}
                error = _endpoint_error(relation_in, name, wanted, side, supplied.get(name_key(name), set()) | stored)
                if error is not None:
                    raise error

    async def _stage_graph(
        self,
        staging: Staging,
        kind: Kind,
        passage_id: str,
        text: str,
        entities: list[EntityIn] | None,
        relations: list[RelationIn] | None,
    ) -> tuple[int, int, Usage]:
        if "graph" not in self.pillars:
            return 0, 0, Usage()
        ids: dict[tuple[str, str], str] = {}
        created, usage = await self._stage_entities(staging, text, entities or [], ids)
        rows = await self._stage_relations(staging, kind, passage_id, relations or [], by_caller=True)
        if self.generation is not None:
            extraction, extracting = await extract(self.generation, text, self.relation_types)
            more, spent = await self._stage_entities(staging, text, [EntityIn(name=e.name, type=e.type) for e in extraction.entities], ids)
            created, usage = created + more, usage + extracting + spent
            extracted = [RelationIn(from_name=r.from_name, from_type=r.from_type, to_name=r.to_name, to_type=r.to_type, type=r.type)
                         for r in extraction.relations]
            rows += await self._stage_relations(staging, kind, passage_id, extracted, by_caller=False)
        staging.relations.extend(rows)
        staging.mentions.extend((entity_id, kind, passage_id) for entity_id in dict.fromkeys(ids.values()))
        return created, len(rows), usage

    async def _stage_entities(self, staging: Staging, text: str, entities: list[EntityIn], ids: dict[tuple[str, str], str]) -> tuple[int, Usage]:
        created = 0
        usage = Usage()
        for entity_in in entities:
            identity = (name_key(entity_in.name), type_key(entity_in.type))
            if identity in ids:
                continue
            entity, was_created, spent = await resolve(self.storage, self.decision, staging, entity_in, text)
            ids[identity] = entity.entity_id
            created += int(was_created)
            usage = usage + spent
        return created, usage

    async def _stage_relations(self, staging: Staging, kind: Kind, passage_id: str, relations: list[RelationIn], *, by_caller: bool) -> list[Relation]:
        rows: list[Relation] = []
        for relation_in in relations:
            ends: list[str] = []
            for name, wanted, side in _endpoints(relation_in):
                found = {e.entity_id: e for e in [*staging.named(name), *await self.storage.entities_named(staging.corpus_id, name)]}
                matching = [e for e in found.values() if wanted is None or e.type == type_key(wanted)]
                error = _endpoint_error(relation_in, name, wanted, side, {e.type for e in matching})
                if error is not None:
                    if by_caller:
                        raise error
                    break
                ends.append(matching[0].entity_id)
            if len(ends) == 2:
                rows.append(Relation(corpus_id=staging.corpus_id, from_id=ends[0], to_id=ends[1], type=relation_in.type.strip().upper(),
                                     evidence_kind=kind, evidence_id=passage_id))
        return rows

    async def remember(
        self,
        corpus_id: str,
        text: str,
        *,
        subject: str | None = None,
        occurred_at: datetime | None = None,
        source_ids: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        entities: list[EntityIn] | None = None,
        relations: list[RelationIn] | None = None,
    ) -> Story:
        corpus_id = _corpus(corpus_id, text, subject, source_ids, metadata, entities, relations)
        if not text.strip():
            raise InputError("memory: a story needs text")
        existing = await self.storage.find_open_story(corpus_id, text_hash(text))
        if existing is not None:
            return existing
        story = Story(corpus_id=corpus_id, story_id=uuid4().hex, text=text.strip(), subject=subject,
                      occurred_at=_aware(occurred_at), created_at=_now(), valid_to=None, supersedes=None,
                      source_ids=tuple(source_ids), metadata=dict(metadata or {}))
        await self._check_facts(corpus_id, entities, relations)
        vectors = await self._embed([story.text])
        staging = Staging(corpus_id)
        await self._stage_graph(staging, "story", story.story_id, story.text, entities, relations)
        await self.storage.put_story(corpus_id, story, vectors[0] if vectors else None, self._model(), graph=staging.write())
        return story

    async def supersede(
        self,
        corpus_id: str,
        story_id: str,
        text: str,
        *,
        effective_at: datetime | None = None,
        subject: str | None = None,
        metadata: dict[str, Any] | None = None,
        entities: list[EntityIn] | None = None,
        relations: list[RelationIn] | None = None,
    ) -> tuple[Story, Story]:
        corpus_id = _corpus(corpus_id, story_id, text, subject, metadata, entities, relations)
        stories = await self.storage.get_stories(corpus_id, [story_id])
        old = stories[0] if stories else None
        if old is None:
            raise NotFoundError(f"memory: story {story_id!r} was not found")
        if old.valid_to is not None:
            raise InputError(f"memory: story {story_id!r} is already superseded")
        effective = _aware(effective_at) or _now()
        if effective <= old.valid_from:
            raise InputError("memory: effective_at must be after the story became true")
        holder = await self.storage.find_open_story(corpus_id, text_hash(text))
        if holder is not None and holder.story_id != old.story_id:
            raise InputError(f"memory: story {holder.story_id!r} is open with that text")
        new = Story(corpus_id=corpus_id, story_id=uuid4().hex, text=text.strip(), subject=subject or old.subject,
                    occurred_at=effective, created_at=_now(), valid_to=None, supersedes=old.story_id,
                    source_ids=old.source_ids, metadata=dict(metadata if metadata is not None else old.metadata))
        await self._check_facts(corpus_id, entities, relations)
        vectors = await self._embed([new.text])
        staging = Staging(corpus_id)
        await self._stage_graph(staging, "story", new.story_id, new.text, entities, relations)
        return await self.storage.supersede_story(corpus_id, story_id, effective, new, vectors[0] if vectors else None, self._model(),
                                                  graph=staging.write())

    async def forget(self, corpus_id: str, story_id: str) -> None:
        corpus_id = _corpus(corpus_id, story_id)
        deleted = await self.storage.delete_story(corpus_id, story_id)
        if deleted.stories == 0:
            raise NotFoundError(f"memory: story {story_id!r} was not found")

    async def delete_source(self, corpus_id: str, source_id: str) -> Deleted:
        corpus_id = _corpus(corpus_id, source_id)
        deleted = await self.storage.delete_source(corpus_id, source_id)
        if deleted.sources == 0:
            raise NotFoundError(f"memory: source {source_id!r} was not found")
        return deleted

    async def get_source(self, corpus_id: str, source_id: str) -> Source | None:
        return await self.storage.get_source(_corpus(corpus_id, source_id), source_id)

    async def list_sources(
        self, corpus_id: str, *, filters: dict[str, Any] | None = None, limit: int = 100, cursor: str | None = None
    ) -> tuple[list[Source], str | None]:
        validate_filters(filters)
        return await self.storage.list_sources(_corpus(corpus_id, filters), filters=filters, limit=limit, cursor=cursor)

    async def read(self, corpus_id: str, source_id: str, *, locator: dict[str, Any] | None = None) -> str:
        corpus_id = _corpus(corpus_id, source_id, locator)
        source = await self.storage.get_source(corpus_id, source_id)
        if source is None:
            raise NotFoundError(f"memory: source {source_id!r} was not found")
        if locator is None:
            return source.text
        chunks = await self.storage.source_chunks(corpus_id, source_id)
        matching = [c.text for c in chunks if all(c.locator.get(k) == v for k, v in locator.items())]
        return "\n\n".join(matching)

    async def get_stories(self, corpus_id: str, story_ids: list[str]) -> list[Story]:
        return await self.storage.get_stories(_corpus(corpus_id, story_ids), list(story_ids))

    async def list_stories(
        self,
        corpus_id: str,
        *,
        subject: str | None = None,
        as_of: datetime | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        filters: dict[str, Any] | None = None,
        history: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[Story], str | None]:
        validate_filters(filters)
        window = _window(as_of, since, until, history)
        return await self.storage.list_stories(_corpus(corpus_id, subject, filters), subject=subject, window=window, filters=filters, limit=limit, cursor=cursor)

    async def reembed(self, corpus_id: str) -> int:
        corpus_id = _corpus(corpus_id)
        if self.embedding is None or "vector" not in self.pillars:
            raise ConfigurationError("memory: reembed needs the vector pillar")
        total = 0
        while True:
            stale = await self.storage.passages_embedded_with(corpus_id, not_model=self.embedding.model, limit=256)
            if not stale:
                return total
            refs = [("chunk", p.chunk_id) if isinstance(p, Chunk) else ("story", p.story_id) for p in stale]
            vectors = await self.embedding.embed([p.text for p in stale])
            await self.storage.set_vectors(corpus_id, refs, vectors, self.embedding.model)
            total += len(stale)

    async def delete_corpus(self, corpus_id: str) -> Deleted:
        return await self.storage.delete_corpus(_corpus(corpus_id))

    async def verify_absent(self, corpus_id: str) -> Residue:
        return await self.storage.count_corpus(_corpus(corpus_id))

    async def search(
        self,
        corpus_id: str,
        query: str,
        *,
        kinds: tuple[Kind, ...] = ("chunk", "story"),
        pillars: set[str] | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        history: bool = False,
        judge: bool = False,
    ) -> Hits:
        return await self._search(corpus_id, query, kinds=kinds, pillars=pillars, top_k=top_k, filters=filters,
                                  window=_window(None, None, None, history), subject=None, judge=judge)

    async def recall(
        self,
        corpus_id: str,
        query: str,
        *,
        top_k: int = 10,
        as_of: datetime | None = None,
        history: bool = False,
        since: datetime | None = None,
        until: datetime | None = None,
        subject: str | None = None,
        filters: dict[str, Any] | None = None,
        pillars: set[str] | None = None,
        judge: bool = False,
    ) -> Hits:
        window = _window(as_of, since, until, history)
        return await self._search(corpus_id, query, kinds=("story",), pillars=pillars, top_k=top_k, filters=filters,
                                  window=window, subject=subject, judge=judge)

    def _pillars(self, pillars: set[str] | None) -> set[str]:
        chosen = set(pillars) if pillars is not None else set(self.pillars)
        unknown = chosen - self.pillars
        if unknown:
            raise ConfigurationError(f"memory: pillars not configured: {sorted(unknown)}")
        return chosen

    async def _search(
        self,
        corpus_id: str,
        query: str,
        *,
        kinds: tuple[Kind, ...],
        pillars: set[str] | None,
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        judge: bool,
    ) -> Hits:
        corpus_id = _corpus(corpus_id, query, subject, filters)
        validate_filters(filters)
        chosen = self._pillars(pillars)
        if judge and self.decision is None:
            raise ConfigurationError("memory: judge=True needs a decision model")
        started = time.monotonic()
        if not query.strip():
            return Hits(hits=[], pillars={}, usage=Usage(), duration_ms=0)
        if filters and "graph" in chosen:
            chosen = chosen - {"graph"}
        fetch = top_k * self.fetch_multiplier
        names = sorted(chosen)
        legs = await asyncio.gather(*(self._leg(name, corpus_id, query, kinds, fetch, filters, window, subject) for name in names))
        reports = {name: report for name, (_, report) in zip(names, legs, strict=True)}
        fused = fuse_by_rank([(name, self.weights[name], ranked) for name, (ranked, _) in zip(names, legs, strict=True)])
        hits = await self._hydrate(corpus_id, fused[:top_k])
        usage = Usage()
        if judge:
            assert self.decision is not None
            hits, usage = await judge_hits(self.decision, query, hits)
        return Hits(hits=hits, pillars=reports, usage=usage, duration_ms=int((time.monotonic() - started) * 1000))

    async def _leg(
        self, name: str, corpus_id: str, query: str, kinds: tuple[Kind, ...], fetch: int, filters: dict[str, Any] | None, window: Window,
        subject: str | None,
    ) -> tuple[list[tuple[str, float]], PillarReport]:
        began = time.monotonic()
        try:
            if name == "text":
                legs = await self.storage.search_text(corpus_id, query, kinds=kinds, top_k=fetch, filters=filters, window=window, subject=subject)
            elif name == "vector":
                assert self.embedding is not None
                [vector] = await self.embedding.embed([query])
                legs = await self.storage.search_vector(corpus_id, vector, kinds=kinds, top_k=fetch, filters=filters, window=window, subject=subject)
            else:
                legs = await self._graph_leg(corpus_id, query, kinds, fetch, window, subject)
        except (StorageError, ModelError) as exc:
            return [], PillarReport(returned=0, duration_ms=int((time.monotonic() - began) * 1000), error=f"{type(exc).__name__}: {exc}")
        ranked = [(f"{kind}:{passage_id}", score) for kind, passage_id, score in legs]
        return ranked, PillarReport(returned=len(ranked), duration_ms=int((time.monotonic() - began) * 1000))

    async def _graph_leg(
        self, corpus_id: str, query: str, kinds: tuple[Kind, ...], fetch: int, window: Window, subject: str | None
    ) -> list[Leg]:
        seeds = await self.storage.entities_named_in(corpus_id, query)
        if not seeds:
            return []
        reached = {e.entity_id for e in seeds}
        frontier = list(reached)
        for _ in range(2):
            relations = await self.storage.neighbors(corpus_id, frontier, relation_types=None)
            frontier = [n for r in relations for n in (r.from_id, r.to_id) if n not in reached]
            reached.update(frontier)
            if not frontier:
                break
        refs = await self.storage.mentions(corpus_id, sorted(reached), kinds=kinds, filters=None, window=window, subject=subject)
        return [(kind, passage_id, 1.0) for kind, passage_id in refs[:fetch]]

    async def _hydrate(self, corpus_id: str, fused: list[tuple[str, float, dict[str, float]]]) -> list[Hit]:
        refs: list[Ref] = []
        for key, _, _ in fused:
            kind, _, passage_id = key.partition(":")
            refs.append((kind, passage_id))
        passages = {(("chunk", p.chunk_id) if isinstance(p, Chunk) else ("story", p.story_id)): p for p in await self.storage.get_passages(corpus_id, refs)}
        hits: list[Hit] = []
        for (_, score, provenance), ref in zip(fused, refs, strict=True):
            passage = passages.get(ref)
            if passage is None:
                continue
            if isinstance(passage, Chunk):
                hits.append(Hit(kind="chunk", id=passage.chunk_id, text=passage.text, score=score, provenance=provenance, chunk=passage))
            else:
                hits.append(Hit(kind="story", id=passage.story_id, text=passage.text, score=score, provenance=provenance, story=passage))
        return hits

    async def neighbors(
        self,
        corpus_id: str,
        entity: str,
        *,
        entity_type: str | None = None,
        hops: int = 1,
        relation_types: list[str] | None = None,
        history: bool = False,
    ) -> Neighborhood:
        corpus_id = _corpus(corpus_id, entity, entity_type, relation_types)
        if "graph" not in self.pillars:
            raise ConfigurationError("memory: neighbors needs the graph pillar")
        named = await self.storage.entities_named(corpus_id, entity)
        if entity_type is not None:
            named = [e for e in named if e.type == type_key(entity_type)]
            if not named:
                raise NotFoundError(f"memory: entity {entity!r} of type {entity_type!r} was not found")
        if not named:
            raise NotFoundError(f"memory: entity {entity!r} was not found")
        types = sorted({e.type for e in named})
        if len(types) > 1:
            raise InputError(f"memory: entity {entity!r} names entities of several types: {types}; pass entity_type")
        seed = named[0]
        reached = {seed.entity_id: seed}
        frontier = [seed.entity_id]
        relations: list[Relation] = []
        for _ in range(max(hops, 0)):
            found = await self.storage.neighbors(corpus_id, frontier, relation_types=relation_types)
            relations.extend(r for r in found if r not in relations)
            new_ids = [n for r in found for n in (r.from_id, r.to_id) if n not in reached]
            for found_entity in await self.storage.entities_by_id(corpus_id, new_ids):
                reached[found_entity.entity_id] = found_entity
            frontier = new_ids
            if not frontier:
                break
        refs = await self.storage.mentions(corpus_id, list(reached), kinds=("chunk", "story"), filters=None,
                                           window=_window(None, None, None, history), subject=None)
        passages = await self._hydrate(corpus_id, [(f"{k}:{i}", 1.0, {"graph": 1.0}) for k, i in refs])
        return Neighborhood(entities=list(reached.values()), relations=relations, passages=passages)
