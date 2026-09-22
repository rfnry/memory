from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from rfnry_memory.chunking.chunker import Piece, chunk_text
from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError
from rfnry_memory.filters import validate_filters
from rfnry_memory.graph.identity import name_key, type_key
from rfnry_memory.models import (
    Chunk,
    DecisionModel,
    Deleted,
    EmbeddingModel,
    Entity,
    EntityIn,
    GenerationModel,
    Ingested,
    Kind,
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


def _corpus(corpus_id: str) -> str:
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise InputError("corpus_id is required")
    return corpus_id


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
            raise ConfigurationError(f"unknown pillars: {sorted(unknown)}")
        if "vector" in self.pillars and embedding is None:
            raise ConfigurationError("the vector pillar needs an embedding model")
        if "graph" in self.pillars and generation is not None and not relation_types:
            raise ConfigurationError("extraction needs relation_types")
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
        corpus_id = _corpus(corpus_id)
        if (text is None) == (sections is None):
            raise InputError("ingest takes exactly one of text= or sections=")
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
        vectors = await self._embed([c.text for c in chunks])
        await self.storage.put_source(corpus_id, source, chunks, vectors, self._model())
        entity_count = relation_count = 0
        usage = Usage()
        for chunk in chunks:
            e, r, u = await self._write_graph(corpus_id, "chunk", chunk.chunk_id, chunk.text, entities, relations)
            entity_count, relation_count, usage = entity_count + e, relation_count + r, usage + u
            entities, relations = None, None
        return Ingested(source=source, chunks=len(chunks), entities=entity_count, relations=relation_count,
                        usage=usage, duration_ms=int((time.monotonic() - started) * 1000))

    async def _write_graph(
        self,
        corpus_id: str,
        kind: Kind,
        passage_id: str,
        text: str,
        entities: list[EntityIn] | None,
        relations: list[RelationIn] | None,
    ) -> tuple[int, int, Usage]:
        if not entities and not relations:
            return 0, 0, Usage()
        if "graph" not in self.pillars:
            raise ConfigurationError("graph facts were supplied but the graph pillar is off")
        ids: dict[str, str] = {}
        created = 0
        for entity_in in entities or []:
            key, type_ = name_key(entity_in.name), type_key(entity_in.type)
            existing = await self.storage.find_entity(corpus_id, key, type_)
            if existing is None:
                existing = Entity(corpus_id=corpus_id, entity_id=uuid4().hex, name=entity_in.name.strip(), key=key, type=type_)
                await self.storage.put_entity(corpus_id, existing)
                created += 1
            ids[key] = existing.entity_id
        rows: list[Relation] = []
        for relation_in in relations or []:
            from_id, to_id = ids.get(name_key(relation_in.from_name)), ids.get(name_key(relation_in.to_name))
            if from_id is None:
                found = await self.storage.get_entity(corpus_id, relation_in.from_name)
                from_id = found.entity_id if found else None
            if to_id is None:
                found = await self.storage.get_entity(corpus_id, relation_in.to_name)
                to_id = found.entity_id if found else None
            if from_id is None or to_id is None:
                raise InputError(f"relation {relation_in.from_name!r} -> {relation_in.to_name!r} names an unknown entity")
            rows.append(Relation(corpus_id=corpus_id, from_id=from_id, to_id=to_id, type=relation_in.type.strip().upper(),
                                 evidence_kind=kind, evidence_id=passage_id))
        await self.storage.put_relations(corpus_id, rows)
        await self.storage.put_mentions(corpus_id, list(ids.values()), kind, passage_id)
        return created, len(rows), Usage()

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
        corpus_id = _corpus(corpus_id)
        if not text.strip():
            raise InputError("a story needs text")
        existing = await self.storage.find_open_story(corpus_id, text_hash(text))
        if existing is not None:
            return existing
        story = Story(corpus_id=corpus_id, story_id=uuid4().hex, text=text.strip(), subject=subject,
                      occurred_at=_aware(occurred_at), created_at=_now(), valid_to=None, supersedes=None,
                      source_ids=tuple(source_ids), metadata=dict(metadata or {}))
        vectors = await self._embed([story.text])
        await self.storage.put_story(corpus_id, story, vectors[0] if vectors else None, self._model())
        await self._write_graph(corpus_id, "story", story.story_id, story.text, entities, relations)
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
    ) -> tuple[Story, Story]:
        corpus_id = _corpus(corpus_id)
        stories = await self.storage.get_stories(corpus_id, [story_id])
        old = stories[0] if stories else None
        if old is None:
            raise NotFoundError(f"story {story_id!r}")
        if old.valid_to is not None:
            raise InputError(f"story {story_id!r} is already superseded")
        effective = _aware(effective_at) or _now()
        if effective <= old.valid_from:
            raise InputError("effective_at must be after the story became true")
        new = Story(corpus_id=corpus_id, story_id=uuid4().hex, text=text.strip(), subject=subject or old.subject,
                    occurred_at=effective, created_at=_now(), valid_to=None, supersedes=old.story_id,
                    source_ids=old.source_ids, metadata=dict(metadata if metadata is not None else old.metadata))
        vectors = await self._embed([new.text])
        return await self.storage.supersede_story(corpus_id, story_id, effective, new, vectors[0] if vectors else None, self._model())

    async def forget(self, corpus_id: str, story_id: str) -> None:
        corpus_id = _corpus(corpus_id)
        deleted = await self.storage.delete_story(corpus_id, story_id)
        if deleted.stories == 0:
            raise NotFoundError(f"story {story_id!r}")

    async def delete_source(self, corpus_id: str, source_id: str) -> Deleted:
        corpus_id = _corpus(corpus_id)
        deleted = await self.storage.delete_source(corpus_id, source_id)
        if deleted.sources == 0:
            raise NotFoundError(f"source {source_id!r}")
        return deleted

    async def get_source(self, corpus_id: str, source_id: str) -> Source | None:
        return await self.storage.get_source(_corpus(corpus_id), source_id)

    async def list_sources(
        self, corpus_id: str, *, filters: dict[str, Any] | None = None, limit: int = 100, cursor: str | None = None
    ) -> tuple[list[Source], str | None]:
        validate_filters(filters)
        return await self.storage.list_sources(_corpus(corpus_id), filters=filters, limit=limit, cursor=cursor)

    async def read(self, corpus_id: str, source_id: str, *, locator: dict[str, Any] | None = None) -> str:
        corpus_id = _corpus(corpus_id)
        source = await self.storage.get_source(corpus_id, source_id)
        if source is None:
            raise NotFoundError(f"source {source_id!r}")
        if locator is None:
            return source.text
        chunks = await self.storage.source_chunks(corpus_id, source_id)
        matching = [c.text for c in chunks if all(c.locator.get(k) == v for k, v in locator.items())]
        return "\n\n".join(matching)

    async def get_stories(self, corpus_id: str, story_ids: list[str]) -> list[Story]:
        return await self.storage.get_stories(_corpus(corpus_id), list(story_ids))

    async def list_stories(
        self,
        corpus_id: str,
        *,
        subject: str | None = None,
        as_of: datetime | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[Story], str | None]:
        validate_filters(filters)
        window = Window(as_of=_aware(as_of), since=_aware(since), until=_aware(until))
        return await self.storage.list_stories(_corpus(corpus_id), subject=subject, window=window, filters=filters, limit=limit, cursor=cursor)

    async def reembed(self, corpus_id: str) -> int:
        corpus_id = _corpus(corpus_id)
        if self.embedding is None or "vector" not in self.pillars:
            raise ConfigurationError("reembed needs the vector pillar")
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
