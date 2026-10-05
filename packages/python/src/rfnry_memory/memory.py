from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal, cast
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
from rfnry_memory.filters import matches_filters, validate_filters
from rfnry_memory.fusion import fuse_by_rank
from rfnry_memory.graph.extraction import extract
from rfnry_memory.graph.identity import Staging, resolve, type_key
from rfnry_memory.judge import judge as judge_hits
from rfnry_memory.models import (
    LANGUAGES,
    Chunk,
    Chunking,
    CorpusFingerprint,
    Counts,
    DecisionModel,
    EmbeddingModel,
    EntityIn,
    GenerationModel,
    Hit,
    Hits,
    Ingested,
    Kind,
    Language,
    Leg,
    Neighborhood,
    PillarReport,
    Ref,
    Relation,
    RelationIn,
    Remembered,
    Section,
    Source,
    SourceIn,
    Story,
    Synced,
    Usage,
    Window,
)
from rfnry_memory.storage.storage import (
    GraphWrite,
    SourceWrite,
    Storage,
    chunk_digest,
    ended_on_write,
    name_key,
    text_hash,
)

PILLARS = frozenset({"text", "vector", "graph"})
KINDS = ("chunk", "story")
OVERFETCH = 4


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _ends(start: datetime, valid_to: datetime | None) -> datetime | None:
    end = _aware(valid_to)
    if end is not None and end <= start:
        raise InputError("memory: valid_to must be after valid_from")
    return end


def _closes(old: Story, now: datetime, start: datetime) -> None:
    if old.superseded_by is not None:
        raise InputError(
            f"memory: story {old.story_id!r} is already superseded; supersede the story that replaced it"
        )
    if old.valid_to is not None and old.valid_to <= now:
        raise InputError(
            f"memory: story {old.story_id!r} ended at {old.valid_to.isoformat()}; remember a new story instead"
        )
    if start <= old.valid_from:
        raise InputError(
            f"memory: valid_from must be after story {old.story_id!r} became true, at {old.valid_from.isoformat()}; pass a later valid_from"
        )
    if old.valid_to is not None and old.valid_to <= start:
        raise InputError(
            f"memory: story {old.story_id!r} ends at {old.valid_to.isoformat()}, before valid_from; pass an earlier valid_from"
        )


def _window(
    as_of: datetime | None, since: datetime | None, until: datetime | None, history: bool
) -> Window:
    if history and as_of is not None:
        raise InputError("memory: history=True reads every moment; it takes no as_of")
    moment = None if history else _aware(as_of) or _now()
    return Window(as_of=moment, since=_aware(since), until=_aware(until))


def _at_least_one(name: str, value: int) -> None:
    if value < 1:
        raise InputError(f"memory: {name} must be at least 1")


def _cursor(cursor: str | None) -> None:
    if cursor and not (cursor.isascii() and cursor.isdigit() and int(cursor) < 2**63):
        raise InputError(f"memory: cursor {cursor!r} is not one a listing returned")


def _storable(value: Any) -> None:
    if isinstance(value, str) and "\x00" in value:
        raise InputError("memory: text must not contain a NUL character")
    if isinstance(value, float) and not math.isfinite(value):
        raise InputError("memory: numbers must be finite; NaN and infinity cannot be stored")
    if isinstance(value, BaseModel):
        _storable(value.model_dump())
    elif isinstance(value, dict):
        for key, item in value.items():
            _storable(key)
            _storable(item)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            _storable(item)


def _corpus(corpus_id: str, *values: Any) -> str:
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise InputError("memory: corpus_id is required")
    _storable((corpus_id, values))
    return corpus_id


def _corpora(corpus_id: str | list[str], *values: Any) -> list[str]:
    named = list(corpus_id) if isinstance(corpus_id, list | tuple) else [corpus_id]
    if not named:
        raise InputError("memory: corpus_id is required")
    return list(dict.fromkeys(_corpus(c, *values) for c in named))


def _key(corpus_id: str, kind: str, passage_id: str) -> str:
    return json.dumps([corpus_id, kind, passage_id])


async def _embedded(
    embedding: EmbeddingModel, texts: list[str], purpose: Literal["query", "document"]
) -> list[list[float]]:
    vectors = await embedding.embed(texts, purpose=purpose)
    if len(vectors) != len(texts):
        raise ModelError(
            f"memory: the embedding model returned {len(vectors)} vectors for {len(texts)} texts"
        )
    for vector in vectors:
        if len(vector) != embedding.dimension:
            raise ModelError(
                f"memory: the embedding model returned a vector of {len(vector)} dimensions; it declares {embedding.dimension}"
            )
    return vectors


def _digest(rows: list[list[Any]]) -> str:
    lines = sorted(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) for row in rows)
    return sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def _instant(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _endpoints(relation_in: RelationIn) -> list[tuple[str, str | None, str]]:
    return [
        (relation_in.from_name, relation_in.from_type, "from_type"),
        (relation_in.to_name, relation_in.to_type, "to_type"),
    ]


def _endpoint_error(
    relation_in: RelationIn, name: str, wanted: str | None, side: str, types: set[str]
) -> InputError | None:
    if wanted is not None:
        types = types & {type_key(wanted)}
    ends = f"{relation_in.from_name!r} -> {relation_in.to_name!r}"
    if not types:
        return InputError(f"memory: relation {ends} names an unknown entity")
    if len(types) > 1:
        return InputError(
            f"memory: relation {ends} names {name!r}, which has several types: {sorted(types)}; pass {side}"
        )
    return None


def _place(
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
    ) -> None:
        self.storage = storage
        self.embedding = embedding
        self.generation = generation
        self.decision = decision
        default = (
            {"text"} | ({"vector"} if embedding else set()) | ({"graph"} if generation else set())
        )
        self.pillars = set(pillars) if pillars is not None else default
        unknown = self.pillars - PILLARS
        if unknown:
            raise ConfigurationError(f"memory: unknown pillars: {sorted(unknown)}")
        if "vector" in self.pillars and embedding is None:
            raise ConfigurationError("memory: the vector pillar needs an embedding model")
        if "graph" in self.pillars and generation is not None and not relation_types:
            raise ConfigurationError("memory: extraction needs relation_types")
        self.relation_types = list(relation_types or [])
        weights = dict(weights or {})
        if set(weights) - PILLARS:
            raise ConfigurationError(
                f"memory: weights name unknown pillars: {sorted(set(weights) - PILLARS)}"
            )
        for pillar, weight in weights.items():
            if not weight > 0:
                raise ConfigurationError(
                    f"memory: weights must be positive; got {pillar!r} = {weight!r}"
                )
        self.weights = {pillar: 1.0 for pillar in PILLARS} | weights
        if chunk_chars < 1:
            raise ConfigurationError("memory: chunk_chars must be at least 1")
        if not 0 <= overlap_chars < chunk_chars:
            raise ConfigurationError(
                f"memory: overlap_chars must be at least 0 and less than chunk_chars ({chunk_chars})"
            )
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars

    async def initialize(self) -> None:
        await self.storage.initialize(
            dimension=self.embedding.dimension if self.embedding else None
        )

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if "vector" not in self.pillars or self.embedding is None or not texts:
            return None
        return await _embedded(self.embedding, texts, "document")

    async def _chunk_vectors(
        self, corpus_id: str, chunks: list[Chunk], embedded: dict[str, list[float]]
    ) -> list[list[float]] | None:
        model = self._model()
        if model is None or self.embedding is None or not chunks:
            return None
        digests = [chunk_digest(c.text) for c in chunks]
        held = await self.storage.chunk_vectors(corpus_id, model, sorted(set(digests)))
        missing = list(
            dict.fromkeys(
                c.text
                for c, d in zip(chunks, digests, strict=True)
                if d not in held and c.text not in embedded
            )
        )
        if missing:
            embedded.update(
                zip(missing, await _embedded(self.embedding, missing, "document"), strict=True)
            )
        return [
            held[d] if d in held else embedded[c.text] for c, d in zip(chunks, digests, strict=True)
        ]

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
        source_in = SourceIn(
            name=name,
            text=text,
            sections=sections,
            metadata=metadata or {},
            entities=entities or [],
            relations=relations or [],
        )
        [write], graph, usage = await self._prepare(corpus_id, [source_in])
        await self.storage.put_source(
            corpus_id, write.source, write.chunks, write.vectors, self._model(), graph=graph
        )
        return Ingested(source=write.source, usage=usage)

    async def sync(
        self, corpus_id: str, *, filters: dict[str, Any], sources: list[SourceIn]
    ) -> Synced:
        corpus_id = _corpus(corpus_id, filters, sources)
        if not filters:
            raise InputError(
                "memory: sync needs filters naming the set it syncs — delete_corpus empties a corpus"
            )
        validate_filters(filters)
        for source_in in sources:
            if not matches_filters(source_in.metadata, filters):
                raise InputError(
                    f"memory: source {source_in.name!r} does not match the filters it syncs"
                )
        stored: dict[str, list[Source]] = {}
        for source in await self._matching(corpus_id, filters):
            if source.digest is not None:
                stored.setdefault(source.digest, []).append(source)
        kept: dict[int, Source] = {}
        for index, source_in in enumerate(sources):
            same = stored.get(self._digest_of(source_in))
            if same:
                kept[index] = same.pop(0)
        fresh = [s for index, s in enumerate(sources) if index not in kept]
        writes, graph, usage = await self._prepare(corpus_id, fresh)
        deleted = await self.storage.sync_sources(
            corpus_id,
            filters,
            writes,
            self._model(),
            graph=graph,
            keep={k.source_id: k.digest for k in kept.values() if k.digest is not None},
        )
        if deleted is None:
            kept = {}
            writes, graph, again = await self._prepare(corpus_id, sources)
            usage = usage + again
            deleted = await self.storage.sync_sources(
                corpus_id, filters, writes, self._model(), graph=graph
            )
            if deleted is None:
                raise StorageError("memory: the storage refused a sync that keeps nothing")
        written = iter(w.source for w in writes)
        return Synced(
            sources=[kept[i] if i in kept else next(written) for i in range(len(sources))],
            kept=[kept[i].source_id for i in sorted(kept)],
            deleted=deleted,
            usage=usage,
        )

    async def _matching(self, corpus_id: str, filters: dict[str, Any]) -> list[Source]:
        found: dict[str, Source] = {}
        cursor: str | None = None
        while True:
            page, cursor = await self.storage.list_sources(
                corpus_id, filters=filters, limit=500, cursor=cursor
            )
            found.update((s.source_id, s) for s in page)
            if cursor is None:
                return list(found.values())

    def _digest_of(self, source_in: SourceIn) -> str:
        extracting = "graph" in self.pillars and self.generation is not None
        settings = [
            source_in.model_dump(mode="json"),
            self.chunk_chars,
            self.overlap_chars,
            sorted(self.pillars),
            sorted(self.relation_types) if extracting else None,
        ]
        return sha256(
            json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _chunked(self, corpus_id: str, source_in: SourceIn) -> tuple[Source, list[Chunk]]:
        source_id = uuid4().hex
        sections = (
            [Section(text=source_in.text)]
            if source_in.text is not None
            else list(source_in.sections or [])
        )
        pieces = [
            (Piece(title=section.title or piece.title, text=piece.text), dict(section.locator))
            for section in sections
            for piece in chunk_text(
                section.text, chunk_chars=self.chunk_chars, overlap_chars=self.overlap_chars
            )
        ]
        chunks = [
            Chunk(
                corpus_id=corpus_id,
                chunk_id=f"{source_id}-{index}",
                source_id=source_id,
                source_name=source_in.name,
                position=index,
                title=piece.title,
                locator=locator,
                text=piece.text,
                metadata=dict(source_in.metadata),
            )
            for index, (piece, locator) in enumerate(pieces)
        ]
        source = Source(
            corpus_id=corpus_id,
            source_id=source_id,
            name=source_in.name,
            metadata=dict(source_in.metadata),
            text="\n\n".join(section.text.strip() for section in sections),
            created_at=_now(),
            chunks=len(chunks),
            digest=self._digest_of(source_in),
        )
        return source, chunks

    async def _prepare(
        self, corpus_id: str, sources: list[SourceIn]
    ) -> tuple[list[SourceWrite], GraphWrite, Usage]:
        chunked = [self._chunked(corpus_id, source_in) for source_in in sources]
        await self._check_facts(
            corpus_id,
            [e for source_in in sources for e in source_in.entities],
            [r for source_in in sources for r in source_in.relations],
        )
        embedded: dict[str, list[float]] = {}
        writes = [
            SourceWrite(source, chunks, await self._chunk_vectors(corpus_id, chunks, embedded))
            for source, chunks in chunked
        ]
        staging = Staging(corpus_id)
        usage = Usage()
        placed = [
            _place(write.chunks, source_in.entities, source_in.relations)
            for write, source_in in zip(writes, sources, strict=True)
        ]
        for write, (named, _) in zip(writes, placed, strict=True):
            for chunk, here in zip(write.chunks, named, strict=True):
                usage = usage + await self._stage_graph(
                    staging, "chunk", chunk.chunk_id, chunk.text, here, None
                )
        if "graph" in self.pillars:
            for write, (_, related) in zip(writes, placed, strict=True):
                for chunk, here_relations in zip(write.chunks, related, strict=True):
                    staging.relations.extend(
                        await self._stage_relations(
                            staging, "chunk", chunk.chunk_id, here_relations, by_caller=True
                        )
                    )
        return writes, staging.write(), usage

    async def _check_facts(
        self, corpus_id: str, entities: list[EntityIn] | None, relations: list[RelationIn] | None
    ) -> None:
        if "graph" not in self.pillars:
            if entities or relations:
                raise ConfigurationError(
                    "memory: graph facts were supplied but the graph pillar is off"
                )
            return
        supplied: dict[str, set[str]] = {}
        for entity_in in entities or []:
            supplied.setdefault(name_key(entity_in.name), set()).add(type_key(entity_in.type))
        for relation_in in relations or []:
            for name, wanted, side in _endpoints(relation_in):
                stored = {e.type for e in await self.storage.entities_named(corpus_id, name)}
                error = _endpoint_error(
                    relation_in, name, wanted, side, supplied.get(name_key(name), set()) | stored
                )
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
    ) -> Usage:
        if "graph" not in self.pillars:
            return Usage()
        ids: dict[tuple[str, str], str] = {}
        names: dict[tuple[str, str], str] = {}
        usage = await self._stage_entities(staging, text, entities or [], ids, names)
        rows = await self._stage_relations(
            staging, kind, passage_id, relations or [], by_caller=True
        )
        if self.generation is not None:
            extraction, extracting = await extract(self.generation, text, self.relation_types)
            spent = await self._stage_entities(
                staging,
                text,
                [EntityIn(name=e.name, type=e.type) for e in extraction.entities],
                ids,
                names,
            )
            usage = usage + extracting + spent
            extracted = [
                RelationIn(
                    from_name=r.from_name,
                    from_type=r.from_type,
                    to_name=r.to_name,
                    to_type=r.to_type,
                    type=r.type,
                )
                for r in extraction.relations
            ]
            rows += await self._stage_relations(
                staging, kind, passage_id, extracted, by_caller=False
            )
        staging.relations.extend(rows)
        staging.mentions.extend(
            dict.fromkeys((ids[identity], kind, passage_id, names[identity]) for identity in ids)
        )
        return usage

    async def _stage_entities(
        self,
        staging: Staging,
        text: str,
        entities: list[EntityIn],
        ids: dict[tuple[str, str], str],
        names: dict[tuple[str, str], str],
    ) -> Usage:
        usage = Usage()
        for entity_in in entities:
            identity = (name_key(entity_in.name), type_key(entity_in.type))
            if identity in ids:
                continue
            entity, spent = await resolve(self.storage, self.decision, staging, entity_in, text)
            ids[identity] = entity.entity_id
            names[identity] = entity_in.name.strip()
            usage = usage + spent
        return usage

    async def _stage_relations(
        self,
        staging: Staging,
        kind: Kind,
        passage_id: str,
        relations: list[RelationIn],
        *,
        by_caller: bool,
    ) -> list[Relation]:
        rows: list[Relation] = []
        for relation_in in relations:
            ends: list[str] = []
            said: list[str] = []
            for name, wanted, side in _endpoints(relation_in):
                found = {
                    e.entity_id: e
                    for e in [
                        *staging.named(name),
                        *await self.storage.entities_named(staging.corpus_id, name),
                    ]
                }
                matching = [
                    e for e in found.values() if wanted is None or e.type == type_key(wanted)
                ]
                error = _endpoint_error(relation_in, name, wanted, side, {e.type for e in matching})
                if error is not None:
                    if by_caller:
                        raise error
                    break
                ends.append(staging.note(matching[0]).entity_id)
                said.append(name.strip())
            if len(ends) == 2:
                staging.mentions.extend(
                    (end, kind, passage_id, name) for end, name in zip(ends, said, strict=True)
                )
                rows.append(
                    Relation(
                        corpus_id=staging.corpus_id,
                        from_id=ends[0],
                        to_id=ends[1],
                        type=relation_in.type.strip().upper(),
                        evidence_kind=kind,
                        evidence_id=passage_id,
                    )
                )
        return rows

    async def remember(
        self,
        corpus_id: str,
        text: str,
        *,
        subject: str | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        source_ids: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        entities: list[EntityIn] | None = None,
        relations: list[RelationIn] | None = None,
    ) -> Remembered:
        corpus_id = _corpus(corpus_id, text, subject, source_ids, metadata, entities, relations)
        if not text.strip():
            raise InputError("memory: a story needs text")
        now = _now()
        start = _aware(valid_from) or now
        story = Story(
            corpus_id=corpus_id,
            story_id=uuid4().hex,
            text=text.strip(),
            subject=subject,
            valid_from=start,
            valid_to=_ends(start, valid_to),
            created_at=now,
            source_ids=tuple(source_ids),
            metadata=dict(metadata or {}),
        )
        if not ended_on_write(story):
            existing = await self.storage.find_open_story(corpus_id, text_hash(text), at=now)
            if existing is not None:
                return Remembered(story=existing, usage=Usage())
        await self._check_facts(corpus_id, entities, relations)
        vectors = await self._embed([story.text])
        staging = Staging(corpus_id)
        usage = await self._stage_graph(
            staging, "story", story.story_id, story.text, entities, relations
        )
        stored = await self.storage.put_story(
            corpus_id, story, vectors[0] if vectors else None, self._model(), graph=staging.write()
        )
        return Remembered(story=stored, usage=usage)

    async def supersede(
        self,
        corpus_id: str,
        story_id: str,
        text: str,
        *,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        subject: str | None = None,
        metadata: dict[str, Any] | None = None,
        entities: list[EntityIn] | None = None,
        relations: list[RelationIn] | None = None,
    ) -> Remembered:
        corpus_id = _corpus(corpus_id, story_id, text, subject, metadata, entities, relations)
        if not text.strip():
            raise InputError("memory: a story needs text")
        stories = await self.storage.get_stories(corpus_id, [story_id])
        old = stories[0] if stories else None
        if old is None:
            raise NotFoundError(f"memory: story {story_id!r} was not found")
        now = _now()
        effective = _aware(valid_from) or now
        _closes(old, now, effective)
        end = _ends(effective, valid_to)
        holder = await self.storage.find_open_story(corpus_id, text_hash(text), at=now)
        if holder is not None and holder.story_id != old.story_id:
            raise InputError(
                f"memory: story {holder.story_id!r}, true since {holder.valid_from.isoformat()}, already says that; supersede or delete it instead"
            )
        new = Story(
            corpus_id=corpus_id,
            story_id=uuid4().hex,
            text=text.strip(),
            subject=subject or old.subject,
            valid_from=effective,
            valid_to=end,
            created_at=now,
            supersedes=(old.story_id,),
            source_ids=old.source_ids,
            metadata=dict(metadata if metadata is not None else old.metadata),
        )
        await self._check_facts(corpus_id, entities, relations)
        vectors = await self._embed([new.text])
        staging = Staging(corpus_id)
        usage = await self._stage_graph(
            staging, "story", new.story_id, new.text, entities, relations
        )
        _, current = await self.storage.supersede_story(
            corpus_id,
            story_id,
            effective,
            new,
            vectors[0] if vectors else None,
            self._model(),
            graph=staging.write(),
        )
        return Remembered(story=current, usage=usage)

    async def delete_story(self, corpus_id: str, story_id: str) -> Counts:
        corpus_id = _corpus(corpus_id, story_id)
        deleted = await self.storage.delete_story(corpus_id, story_id)
        if deleted.stories == 0:
            raise NotFoundError(f"memory: story {story_id!r} was not found")
        return deleted

    async def delete_source(self, corpus_id: str, source_id: str) -> Counts:
        corpus_id = _corpus(corpus_id, source_id)
        deleted = await self.storage.delete_source(corpus_id, source_id)
        if deleted.sources == 0:
            raise NotFoundError(f"memory: source {source_id!r} was not found")
        return deleted

    async def get_source(self, corpus_id: str, source_id: str) -> Source | None:
        return await self.storage.get_source(_corpus(corpus_id, source_id), source_id)

    async def list_sources(
        self,
        corpus_id: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[Source], str | None]:
        validate_filters(filters)
        _at_least_one("limit", limit)
        _cursor(cursor)
        return await self.storage.list_sources(
            _corpus(corpus_id, filters), filters=filters, limit=limit, cursor=cursor
        )

    async def read_source(
        self, corpus_id: str, source_id: str, *, locator: dict[str, Any] | None = None
    ) -> str:
        corpus_id = _corpus(corpus_id, source_id, locator)
        source = await self.storage.get_source(corpus_id, source_id)
        if source is None:
            raise NotFoundError(f"memory: source {source_id!r} was not found")
        if locator is None:
            return source.text
        chunks = await self.storage.source_chunks(corpus_id, source_id)
        matching = [
            c.text for c in chunks if all(c.locator.get(k) == v for k, v in locator.items())
        ]
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
        _at_least_one("limit", limit)
        _cursor(cursor)
        window = _window(as_of, since, until, history)
        return await self.storage.list_stories(
            _corpus(corpus_id, subject, filters),
            subject=subject,
            window=window,
            filters=filters,
            limit=limit,
            cursor=cursor,
        )

    async def reembed(self, corpus_id: str) -> int:
        corpus_id = _corpus(corpus_id)
        if self.embedding is None or "vector" not in self.pillars:
            raise ConfigurationError("memory: reembed needs the vector pillar")
        total = 0
        while True:
            stale = await self.storage.passages_embedded_with(
                corpus_id, not_model=self.embedding.model, limit=256
            )
            if not stale:
                return total
            refs = [
                ("chunk", p.chunk_id) if isinstance(p, Chunk) else ("story", p.story_id)
                for p in stale
            ]
            vectors = await _embedded(self.embedding, [p.text for p in stale], "document")
            await self.storage.set_vectors(corpus_id, refs, vectors, self.embedding.model)
            total += len(stale)

    async def fingerprint(self, corpus_id: str) -> CorpusFingerprint:
        corpus_id = _corpus(corpus_id)
        sources = await self.storage.source_digests(corpus_id)
        counts = await self.storage.count_corpus(corpus_id)
        vectors = await self.storage.count_vectors(corpus_id)
        model = self._model()
        stories: list[Story] = []
        cursor: str | None = None
        while True:
            page, cursor = await self.storage.list_stories(
                corpus_id, subject=None, window=Window(), filters=None, limit=1000, cursor=cursor
            )
            stories.extend(page)
            if cursor is None:
                break
        return CorpusFingerprint(
            sources=len(sources),
            chunks=counts.chunks,
            content=_digest([list(source) for source in sources]),
            stories=len(stories),
            story_content=_digest(
                [
                    [s.text, s.subject, s.metadata, _instant(s.valid_from), _instant(s.valid_to)]
                    for s in stories
                ]
            ),
            embedding=model,
            embedded=dict(sorted((m, n) for m, n in vectors.items() if m is not None)),
            stale=0 if model is None else sum(n for m, n in vectors.items() if m != model),
            chunking=Chunking(
                chunk_chars=self.chunk_chars,
                overlap_chars=self.overlap_chars,
                pillars=sorted(self.pillars),
            ),
            language=await self.storage.get_language(corpus_id),
        )

    async def set_language(self, corpus_id: str, language: Language) -> int:
        corpus_id = _corpus(corpus_id)
        if language not in LANGUAGES:
            raise InputError(
                f"memory: unknown language {language!r}; the languages are {', '.join(LANGUAGES)}"
            )
        return await self.storage.set_language(corpus_id, language)

    async def delete_corpus(self, corpus_id: str) -> Counts:
        return await self.storage.delete_corpus(_corpus(corpus_id))

    async def verify_absent(self, corpus_id: str) -> Counts:
        return await self.storage.count_corpus(_corpus(corpus_id))

    async def search(
        self,
        corpus_id: str | list[str],
        query: str,
        *,
        kinds: tuple[Kind, ...] = ("chunk", "story"),
        pillars: set[str] | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        history: bool = False,
        judge: bool = False,
    ) -> Hits:
        return await self._search(
            corpus_id,
            query,
            kinds=kinds,
            pillars=pillars,
            top_k=top_k,
            filters=filters,
            window=_window(None, None, None, history),
            subject=None,
            judge=judge,
        )

    async def recall(
        self,
        corpus_id: str | list[str],
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
        return await self._search(
            corpus_id,
            query,
            kinds=("story",),
            pillars=pillars,
            top_k=top_k,
            filters=filters,
            window=window,
            subject=subject,
            judge=judge,
        )

    def _pillars(self, pillars: set[str] | None) -> set[str]:
        chosen = set(pillars) if pillars is not None else set(self.pillars)
        unknown = chosen - self.pillars
        if unknown:
            raise ConfigurationError(f"memory: pillars not configured: {sorted(unknown)}")
        return chosen

    async def _search(
        self,
        corpus_id: str | list[str],
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
        corpus_ids = _corpora(corpus_id, query, subject, filters)
        validate_filters(filters)
        unknown = sorted(set(kinds) - set(KINDS))
        if unknown:
            raise InputError(f"memory: unknown kinds: {unknown}; choose from {list(KINDS)}")
        _at_least_one("top_k", top_k)
        chosen = self._pillars(pillars)
        if judge and self.decision is None:
            raise ConfigurationError("memory: judge=True needs a decision model")
        if not query.strip():
            return Hits(hits=[], pillars={}, usage=Usage())
        if filters and "graph" in chosen:
            chosen = chosen - {"graph"}
        fetch = top_k * OVERFETCH
        names = sorted(chosen)
        legs = await asyncio.gather(
            *(
                self._leg(name, corpus_ids, query, kinds, fetch, filters, window, subject)
                for name in names
            )
        )
        reports = {name: report for name, (_, report) in zip(names, legs, strict=True)}
        fused = fuse_by_rank(
            [
                (name, self.weights[name], ranked)
                for name, (ranked, _) in zip(names, legs, strict=True)
            ]
        )
        hits = await self._hydrate(fused[:top_k])
        usage = Usage()
        if judge:
            assert self.decision is not None
            hits, usage = await judge_hits(self.decision, query, hits)
        return Hits(hits=hits, pillars=reports, usage=usage)

    async def _leg(
        self,
        name: str,
        corpus_ids: list[str],
        query: str,
        kinds: tuple[Kind, ...],
        fetch: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> tuple[list[tuple[str, float]], PillarReport]:
        began = time.monotonic()
        vector: list[float] = []

        async def one(corpus_id: str) -> list[Leg]:
            if name == "text":
                return await self.storage.search_text(
                    corpus_id,
                    query,
                    kinds=kinds,
                    top_k=fetch,
                    filters=filters,
                    window=window,
                    subject=subject,
                )
            if name == "vector":
                return await self.storage.search_vector(
                    corpus_id,
                    vector,
                    kinds=kinds,
                    top_k=fetch,
                    filters=filters,
                    window=window,
                    subject=subject,
                )
            return await self._graph_leg(corpus_id, query, kinds, fetch, window, subject)

        try:
            if name == "vector":
                assert self.embedding is not None
                [vector] = await _embedded(self.embedding, [query], "query")
            found = await asyncio.gather(*(one(c) for c in corpus_ids), return_exceptions=True)
            failed = next((f for f in found if isinstance(f, BaseException)), None)
            if failed is not None:
                raise failed
        except (StorageError, ModelError) as exc:
            return [], PillarReport(
                returned=0,
                duration_ms=int((time.monotonic() - began) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
        merged = sorted(
            (-score, rank, index, _key(corpus_id, kind, passage_id), score)
            for index, (corpus_id, legs) in enumerate(zip(corpus_ids, found, strict=True))
            for rank, (kind, passage_id, score) in enumerate(cast(list[Leg], legs))
        )
        ranked = [(key, score) for _, _, _, key, score in merged[:fetch]]
        return ranked, PillarReport(
            returned=len(ranked), duration_ms=int((time.monotonic() - began) * 1000)
        )

    async def _graph_leg(
        self,
        corpus_id: str,
        query: str,
        kinds: tuple[Kind, ...],
        fetch: int,
        window: Window,
        subject: str | None,
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
        refs = await self.storage.mentions(
            corpus_id, sorted(reached), kinds=kinds, filters=None, window=window, subject=subject
        )
        return [(kind, passage_id, 1.0) for kind, passage_id in refs[:fetch]]

    async def _hydrate(self, fused: list[tuple[str, float, dict[str, float]]]) -> list[Hit]:
        named = [cast(tuple[str, str, str], tuple(json.loads(key))) for key, _, _ in fused]
        wanted: dict[str, list[Ref]] = {}
        for corpus_id, kind, passage_id in named:
            wanted.setdefault(corpus_id, []).append((kind, passage_id))
        found = await asyncio.gather(
            *(self.storage.get_passages(corpus_id, refs) for corpus_id, refs in wanted.items())
        )
        passages = {
            (p.corpus_id, "chunk", p.chunk_id)
            if isinstance(p, Chunk)
            else (p.corpus_id, "story", p.story_id): p
            for held in found
            for p in held
        }
        hits: list[Hit] = []
        for (_, score, provenance), ref in zip(fused, named, strict=True):
            passage = passages.get(ref)
            if passage is None:
                continue
            if isinstance(passage, Chunk):
                hits.append(
                    Hit(
                        corpus_id=passage.corpus_id,
                        kind="chunk",
                        id=passage.chunk_id,
                        text=passage.text,
                        score=score,
                        provenance=provenance,
                        chunk=passage,
                    )
                )
            else:
                hits.append(
                    Hit(
                        corpus_id=passage.corpus_id,
                        kind="story",
                        id=passage.story_id,
                        text=passage.text,
                        score=score,
                        provenance=provenance,
                        story=passage,
                    )
                )
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
        if relation_types is not None:
            relation_types = [t.strip().upper() for t in relation_types]
        named = await self.storage.entities_named(corpus_id, entity)
        if entity_type is not None:
            named = [e for e in named if e.type == type_key(entity_type)]
            if not named:
                raise NotFoundError(
                    f"memory: entity {entity!r} of type {entity_type!r} was not found"
                )
        if not named:
            raise NotFoundError(f"memory: entity {entity!r} was not found")
        types = sorted({e.type for e in named})
        if len(types) > 1:
            raise InputError(
                f"memory: entity {entity!r} names entities of several types: {types}; pass entity_type"
            )
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
        refs = await self.storage.mentions(
            corpus_id,
            list(reached),
            kinds=("chunk", "story"),
            filters=None,
            window=_window(None, None, None, history),
            subject=None,
        )
        passages = await self._hydrate(
            [(_key(corpus_id, k, i), 1.0, {"graph": 1.0}) for k, i in refs]
        )
        return Neighborhood(entities=list(reached.values()), relations=relations, passages=passages)
