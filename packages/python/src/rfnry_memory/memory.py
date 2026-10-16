from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
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
from rfnry_memory.fusion import METHODS as FUSIONS, RRF, Fusion, RelativeScore, rrf
from rfnry_memory.graph.extraction import extract
from rfnry_memory.graph.identity import Staging, resolve, type_key
from rfnry_memory.graph.placement import place
from rfnry_memory.index import HNSW, IVFFlat
from rfnry_memory.judge import JUDGEMENTS, judge as judge_hits
from rfnry_memory.models import (
    Chunk,
    Chunking,
    CorpusDescription,
    CorpusFingerprint,
    Counts,
    DecisionModel,
    DocumentEmbeddingModel,
    EmbeddingModel,
    Entity,
    EntityIn,
    GenerationModel,
    Hit,
    Hits,
    Ingested,
    JudgementName,
    Kind,
    Language,
    Leg,
    Match,
    Neighborhood,
    PillarReport,
    Ref,
    Relation,
    RelationIn,
    Remembered,
    Scope,
    Section,
    Source,
    SourceIn,
    Story,
    Synced,
    Usage,
    Window,
)
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.ranking import BM25, CoverDensity, TextRanking
from rfnry_memory.rows import (
    GraphWrite,
    SourceWrite,
    chunk_digest,
    ended_on_write,
    name_key,
    text_hash,
)

PILLARS = frozenset({"text", "vector", "graph"})
KINDS = ("chunk", "story")
MATCHES = ("any", "all", "phrase", "web")
EXPANSIONS = ("document", "vector", "graph")


def _depth(depth: object) -> int:
    if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 3:
        raise InputError("memory: depth is 0, 1, 2 or 3")
    return depth


def _fusion(fusion: object, default: Fusion) -> Fusion:
    if fusion is None:
        return default
    if not isinstance(fusion, RRF | RelativeScore):
        raise InputError("memory: fusion is RRF(...) or RelativeScore(...)")
    return fusion


def _text_ranking(text: object, default: TextRanking) -> TextRanking:
    if text is None:
        return default
    if not isinstance(text, BM25 | CoverDensity):
        raise InputError("memory: text is BM25(...) or CoverDensity(...)")
    return text


def _now() -> datetime:
    return datetime.now(UTC)


def _stamped(writes: list[SourceWrite], at: datetime) -> list[SourceWrite]:
    return [
        SourceWrite(w.source.model_copy(update={"created_at": at}), w.chunks, w.vectors)
        for w in writes
    ]


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
    as_of: datetime | None,
    since: datetime | None,
    until: datetime | None,
    history: bool,
    known_at: datetime | None = None,
) -> Window:
    if history and as_of is not None:
        raise InputError("memory: history=True reads every moment; it takes no as_of")
    known = _aware(known_at)
    moment = None if history else _aware(as_of) or known or _now()
    return Window(as_of=moment, since=_aware(since), until=_aware(until), known_at=known)


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


def _judgements(judge: bool | tuple[JudgementName, ...]) -> tuple[JudgementName, ...]:
    if isinstance(judge, bool):
        return JUDGEMENTS if judge else ()
    names = tuple(judge) if isinstance(judge, tuple | list) else ()
    if (
        not names
        or not all(isinstance(n, str) for n in names)
        or len(set(names)) != len(names)
        or not set(names) <= set(JUDGEMENTS)
    ):
        raise InputError(
            "memory: judge is True, False or a tuple of relevant, evidence, contradicts and injection, each once"
        )
    return names


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


@dataclass(frozen=True)
class _Look:
    seeds: tuple[str, ...] = ()
    spot: bool = True
    scope: Scope | None = None


def _joined(scopes: list[Scope | None]) -> Scope | None:
    held = [scope for scope in scopes if scope is not None]
    if not held:
        return None
    passages = held[0].passages
    if passages is not None:
        passages = tuple(ref for scope in held for ref in scope.passages or ())
    return Scope(sources=held[0].sources, passages=passages)


def _ranked(hops: dict[str, int], found: list[tuple[str, Ref]]) -> list[Leg]:
    reached: dict[Ref, set[str]] = {}
    for entity_id, ref in found:
        reached.setdefault(ref, set()).add(entity_id)
    total = len(reached)
    held = Counter(entity_id for entities in reached.values() for entity_id in entities)
    weight = {
        entity_id: 2.0 ** -hops[entity_id] * math.log(1 + (total - n + 0.5) / (n + 0.5))
        for entity_id, n in held.items()
    }
    legs = [
        (kind, passage_id, min(hops[e] for e in entities), sum(weight[e] for e in sorted(entities)))
        for (kind, passage_id), entities in reached.items()
    ]
    legs.sort(key=lambda leg: (leg[2], -leg[3], leg[1], leg[0]))
    return [
        (kind, passage_id, 2.0**-near * (1 + rarity / (1 + rarity)) / 2)
        for kind, passage_id, near, rarity in legs
    ]


def _checked(
    embedding: EmbeddingModel, texts: list[str], vectors: list[list[float]]
) -> list[list[float]]:
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


async def _embedded(
    embedding: EmbeddingModel, texts: list[str], purpose: Literal["query", "document"]
) -> list[list[float]]:
    return _checked(embedding, texts, await embedding.embed(texts, purpose=purpose))


async def _gather[T](awaitables: Iterable[Awaitable[T]]) -> list[T]:
    found = await asyncio.gather(*awaitables, return_exceptions=True)
    failed = next((f for f in found if isinstance(f, BaseException)), None)
    if failed is not None:
        raise failed
    return cast(list[T], found)


async def _in_document(embedding: DocumentEmbeddingModel, texts: list[str]) -> list[list[float]]:
    return _checked(embedding, texts, await embedding.embed_in_document(texts))


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


class Memory:
    def __init__(
        self,
        storage: PostgresStorage,
        *,
        embedding: EmbeddingModel | None = None,
        generation: GenerationModel | None = None,
        decision: DecisionModel | None = None,
        pillars: set[str] | None = None,
        relation_types: list[str] | None = None,
        text: TextRanking = BM25(),
        fusion: Fusion = RRF(),
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
        if not isinstance(text, BM25 | CoverDensity):
            raise ConfigurationError("memory: text is BM25(...) or CoverDensity(...)")
        self.text: TextRanking = text
        if not isinstance(fusion, RRF | RelativeScore):
            raise ConfigurationError("memory: fusion is RRF(...) or RelativeScore(...)")
        self.fusion: Fusion = fusion
        if chunk_chars < 1:
            raise ConfigurationError("memory: chunk_chars must be at least 1")
        if not 0 <= overlap_chars < chunk_chars:
            raise ConfigurationError(
                f"memory: overlap_chars must be at least 0 and less than chunk_chars ({chunk_chars})"
            )
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars

    async def check(self) -> None:
        await self.storage.check(
            dimension=self.embedding.dimension
            if self.embedding and "vector" in self.pillars
            else None,
            graph="graph" in self.pillars,
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
        if isinstance(self.embedding, DocumentEmbeddingModel):
            return await _in_document(self.embedding, [c.text for c in chunks])
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
        current = await self._matching(corpus_id, filters)
        for source in current:
            if source.digest is not None:
                stored.setdefault(source.digest, []).append(source)
        kept: dict[int, Source] = {}
        for index, source_in in enumerate(sources):
            same = stored.get(self._digest_of(source_in))
            if same:
                kept[index] = same.pop(0)
        fresh = [s for index, s in enumerate(sources) if index not in kept]
        writes, graph, usage = await self._prepare(corpus_id, fresh)
        now = max([_now(), *(s.created_at for s in current)])
        writes = _stamped(writes, now)
        done = await self.storage.sync_sources(
            corpus_id,
            filters,
            writes,
            self._model(),
            at=now,
            graph=graph,
            keep={k.source_id: k.digest for k in kept.values() if k.digest is not None},
        )
        if done is None:
            kept = {}
            writes, graph, again = await self._prepare(corpus_id, sources)
            usage = usage + again
            now = max([_now(), *(s.created_at for s in await self._matching(corpus_id, filters))])
            writes = _stamped(writes, now)
            done = await self.storage.sync_sources(
                corpus_id, filters, writes, self._model(), at=now, graph=graph
            )
            if done is None:
                raise StorageError("memory: the set moved under the sync twice; sync again")
        deleted, retired = done
        written = iter(w.source for w in writes)
        return Synced(
            sources=[kept[i] if i in kept else next(written) for i in range(len(sources))],
            kept=[kept[i].source_id for i in sorted(kept)],
            deleted=deleted,
            retired=retired,
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
        dumped = source_in.model_dump(mode="json")
        for section in dumped["sections"] or []:
            if section["context"] is None:
                del section["context"]
        settings = [
            dumped,
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
            (
                Piece(title=section.title or piece.title, text=piece.text),
                dict(section.locator),
                section.context,
            )
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
                context=context,
            )
            for index, (piece, locator, context) in enumerate(pieces)
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
        staging = Staging(corpus_id)
        await self._check_facts(
            staging,
            [e for source_in in sources for e in source_in.entities],
            [r for source_in in sources for r in source_in.relations],
        )
        embedded: dict[str, list[float]] = {}
        writes = [
            SourceWrite(source, chunks, await self._chunk_vectors(corpus_id, chunks, embedded))
            for source, chunks in chunked
        ]
        usage = Usage()
        placed = [
            place(write.chunks, source_in.entities, source_in.relations)
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
        self,
        staging: Staging,
        entities: list[EntityIn] | None,
        relations: list[RelationIn] | None,
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
                stored = {e.type for e in await staging.stored(self.storage, name)}
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
                        *await staging.stored(self.storage, name),
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
        related: int = 5,
        filters: dict[str, Any] | None = None,
        judge: bool | tuple[JudgementName, ...] = False,
    ) -> Remembered:
        corpus_id = _corpus(
            corpus_id, text, subject, source_ids, metadata, entities, relations, filters
        )
        judged = _judgements(judge)
        if not text.strip():
            raise InputError("memory: a story needs text")
        self._check_related(related, filters, judged)
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
        staging = Staging(corpus_id)
        await self._check_facts(staging, entities, relations)
        ended = ended_on_write(story)
        existing = await self.storage.find_open_story(
            corpus_id, text_hash(text), at=story.valid_from if ended else now
        )
        if existing is not None and (
            not ended
            or (existing.valid_from, existing.valid_to) == (story.valid_from, story.valid_to)
        ):
            near = await self._related(
                corpus_id, existing, None, related, filters, judged, {existing.story_id}
            )
            return Remembered(story=existing, related=near, usage=near.usage)
        vectors = await self._embed([story.text])
        usage = await self._stage_graph(
            staging, "story", story.story_id, story.text, entities, relations
        )
        vector = vectors[0] if vectors else None
        near = await self._related(
            corpus_id,
            story,
            vector,
            related,
            filters,
            judged,
            {story.story_id},
            [entity_id for entity_id, *_ in staging.mentions],
        )
        stored = await self.storage.put_story(
            corpus_id, story, vector, self._model(), graph=staging.write()
        )
        if stored.story_id != story.story_id:
            near = near.model_copy(
                update={"hits": [h for h in near.hits if h.id != stored.story_id]}
            )
        return Remembered(story=stored, related=near, usage=usage + near.usage)

    async def supersede(
        self,
        corpus_id: str,
        story_id: str | list[str],
        text: str,
        *,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        subject: str | None = None,
        metadata: dict[str, Any] | None = None,
        entities: list[EntityIn] | None = None,
        relations: list[RelationIn] | None = None,
        related: int = 5,
        filters: dict[str, Any] | None = None,
        judge: bool | tuple[JudgementName, ...] = False,
    ) -> Remembered:
        ids = [story_id] if isinstance(story_id, str) else list(story_id)
        corpus_id = _corpus(corpus_id, ids, text, subject, metadata, entities, relations, filters)
        judged = _judgements(judge)
        if not ids:
            raise InputError("memory: supersede needs at least one story_id")
        if len(set(ids)) != len(ids):
            raise InputError("memory: story_id names a story more than once")
        if not text.strip():
            raise InputError("memory: a story needs text")
        self._check_related(related, filters, judged)
        found = {s.story_id: s for s in await self.storage.get_stories(corpus_id, ids)}
        missing = next((i for i in ids if i not in found), None)
        if missing is not None:
            raise NotFoundError(f"memory: story {missing!r} was not found")
        olds = [found[i] for i in ids]
        now = _now()
        effective = _aware(valid_from) or now
        for old in olds:
            _closes(old, now, effective)
        end = _ends(effective, valid_to)
        holder = await self.storage.find_open_story(corpus_id, text_hash(text), at=now)
        if holder is not None and holder.story_id not in ids:
            raise InputError(
                f"memory: story {holder.story_id!r}, true since {holder.valid_from.isoformat()}, already says that; supersede or delete it instead"
            )
        subjects = {old.subject for old in olds}
        held = [old.metadata for old in olds]
        new = Story(
            corpus_id=corpus_id,
            story_id=uuid4().hex,
            text=text.strip(),
            subject=subject or (olds[0].subject if len(subjects) == 1 else None),
            valid_from=effective,
            valid_to=end,
            created_at=now,
            supersedes=tuple(ids),
            source_ids=tuple(dict.fromkeys(s for old in olds for s in old.source_ids)),
            metadata=dict(
                metadata
                if metadata is not None
                else (held[0] if all(m == held[0] for m in held) else {})
            ),
        )
        staging = Staging(corpus_id)
        await self._check_facts(staging, entities, relations)
        vectors = await self._embed([new.text])
        usage = await self._stage_graph(
            staging, "story", new.story_id, new.text, entities, relations
        )
        vector = vectors[0] if vectors else None
        near = await self._related(
            corpus_id,
            new,
            vector,
            related,
            filters,
            judged,
            {new.story_id, *ids},
            [entity_id for entity_id, *_ in staging.mentions],
        )
        _, current = await self.storage.supersede_story(
            corpus_id, ids, new, vector, self._model(), graph=staging.write()
        )
        return Remembered(story=current, related=near, usage=usage + near.usage)

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

    async def forget_retired(self, corpus_id: str, *, before: datetime) -> Counts:
        corpus_id = _corpus(corpus_id)
        return await self.storage.forget_retired(
            corpus_id, before=before.replace(tzinfo=before.tzinfo or UTC)
        )

    async def get_source(self, corpus_id: str, source_id: str) -> Source | None:
        return await self.storage.get_source(_corpus(corpus_id, source_id), source_id)

    async def list_sources(
        self,
        corpus_id: str,
        *,
        filters: dict[str, Any] | None = None,
        known_at: datetime | None = None,
        history: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[Source], str | None]:
        validate_filters(filters)
        _at_least_one("limit", limit)
        _cursor(cursor)
        return await self.storage.list_sources(
            _corpus(corpus_id, filters),
            filters=filters,
            limit=limit,
            cursor=cursor,
            known_at=_aware(known_at),
            history=history,
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
        known_at: datetime | None = None,
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
        window = _window(as_of, since, until, history, known_at)
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
            embedding = self.embedding
            if isinstance(embedding, DocumentEmbeddingModel):
                for source_id in dict.fromkeys(p.source_id for p in stale if isinstance(p, Chunk)):
                    whole = await self.storage.source_chunks(corpus_id, source_id)
                    if not whole:
                        continue
                    vectors = await _in_document(embedding, [c.text for c in whole])
                    await self.storage.set_vectors(
                        corpus_id, [("chunk", c.chunk_id) for c in whole], vectors, embedding.model
                    )
                    total += len(whole)
                stale = [p for p in stale if not isinstance(p, Chunk)]
                if not stale:
                    continue
            refs = [
                ("chunk", p.chunk_id) if isinstance(p, Chunk) else ("story", p.story_id)
                for p in stale
            ]
            vectors = await _embedded(embedding, [p.text for p in stale], "document")
            await self.storage.set_vectors(corpus_id, refs, vectors, embedding.model)
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
            chunks=sum(chunks for *_, chunks in sources),
            content=_digest([[name, metadata, sha] for name, metadata, sha, _ in sources]),
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
            history=await self.storage.get_history(corpus_id),
            retired=counts.sources - len(sources),
            vector_index=await self.storage.get_vector_index(corpus_id),
            text=self.text,
            fusion=self.fusion,
        )

    async def describe(self, corpus_id: str) -> CorpusDescription:
        corpus_id = _corpus(corpus_id)
        sources = await self.storage.source_digests(corpus_id)
        counts = await self.storage.count_corpus(corpus_id)
        window = _window(None, None, None, False)
        stories = 0
        cursor: str | None = None
        while True:
            page, cursor = await self.storage.list_stories(
                corpus_id, subject=None, window=window, filters=None, limit=1000, cursor=cursor
            )
            stories += len(page)
            if cursor is None:
                break
        entity_types, relation_types = await self.storage.graph_types(corpus_id)
        return CorpusDescription(
            sources=len(sources),
            chunks=sum(chunks for *_, chunks in sources),
            stories=stories,
            retired=counts.sources - len(sources),
            entities=counts.entities,
            relations=counts.relations,
            entity_types=entity_types,
            relation_types=relation_types,
            language=await self.storage.get_language(corpus_id),
            history=await self.storage.get_history(corpus_id),
            pillars=sorted(self.pillars),
            judge=self.decision is not None,
            expand=[mode for mode in EXPANSIONS if mode == "document" or mode in self.pillars],
        )

    async def set_language(self, corpus_id: str, language: Language) -> int:
        corpus_id = _corpus(corpus_id, language)
        changed = (
            await self.storage.set_language(corpus_id, language)
            if isinstance(language, str)
            else None
        )
        if changed is None:
            raise InputError(
                f"memory: {language!r} is not a text search configuration in this database"
            )
        return changed

    async def set_history(self, corpus_id: str, keep: bool) -> None:
        corpus_id = _corpus(corpus_id)
        if not isinstance(keep, bool):
            raise InputError("memory: keep is True or False")
        await self.storage.set_history(corpus_id, keep)

    async def set_vector_index(
        self, corpus_id: str, index: HNSW | IVFFlat | None, *, rebuild: bool = False
    ) -> int:
        corpus_id = _corpus(corpus_id)
        if index is not None and not isinstance(index, HNSW | IVFFlat):
            raise InputError("memory: index is HNSW(...), IVFFlat(...) or None")
        if not isinstance(rebuild, bool):
            raise InputError("memory: rebuild is True or False")
        if rebuild and index is None:
            raise InputError("memory: rebuild needs an index method")
        if self.embedding is None:
            raise ConfigurationError("memory: a vector index needs an embedding model")
        return await self.storage.set_vector_index(corpus_id, index, rebuild=rebuild)

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
        as_of: datetime | None = None,
        known_at: datetime | None = None,
        history: bool = False,
        judge: bool | tuple[JudgementName, ...] = False,
        entities: list[str] | None = None,
        relation_types: list[str] | None = None,
        sources: list[str] | None = None,
        around: list[str] | None = None,
        hops: int = 1,
        text: TextRanking | None = None,
        fusion: Fusion | None = None,
        match: Match = "any",
        depth: int = 2,
    ) -> Hits:
        return await self._search(
            corpus_id,
            query,
            kinds=kinds,
            pillars=pillars,
            top_k=top_k,
            filters=filters,
            window=_window(as_of, None, None, history, known_at),
            subject=None,
            judge=_judgements(judge),
            entities=entities,
            relation_types=relation_types,
            sources=sources,
            around=around,
            hops=hops,
            text=text,
            fusion=fusion,
            match=match,
            depth=depth,
        )

    async def recall(
        self,
        corpus_id: str | list[str],
        query: str,
        *,
        top_k: int = 10,
        as_of: datetime | None = None,
        known_at: datetime | None = None,
        history: bool = False,
        since: datetime | None = None,
        until: datetime | None = None,
        subject: str | None = None,
        filters: dict[str, Any] | None = None,
        pillars: set[str] | None = None,
        judge: bool | tuple[JudgementName, ...] = False,
        entities: list[str] | None = None,
        relation_types: list[str] | None = None,
        sources: list[str] | None = None,
        around: list[str] | None = None,
        hops: int = 1,
        text: TextRanking | None = None,
        fusion: Fusion | None = None,
        match: Match = "any",
        depth: int = 2,
    ) -> Hits:
        window = _window(as_of, since, until, history, known_at)
        return await self._search(
            corpus_id,
            query,
            kinds=("story",),
            pillars=pillars,
            top_k=top_k,
            filters=filters,
            window=window,
            subject=subject,
            judge=_judgements(judge),
            entities=entities,
            relation_types=relation_types,
            sources=sources,
            around=around,
            hops=hops,
            text=text,
            fusion=fusion,
            match=match,
            depth=depth,
        )

    def _pillars(self, pillars: set[str] | None) -> set[str]:
        chosen = set(pillars) if pillars is not None else set(self.pillars)
        unknown = chosen - self.pillars
        if unknown:
            raise ConfigurationError(f"memory: pillars not configured: {sorted(unknown)}")
        return chosen

    async def _related(
        self,
        corpus_id: str,
        story: Story,
        vector: list[float] | None,
        count: int,
        filters: dict[str, Any] | None,
        judge: tuple[JudgementName, ...],
        exclude: set[str],
        seeds: list[str] | None = None,
    ) -> Hits:
        if count == 0:
            return Hits(hits=[], pillars={}, usage=Usage())
        found = await self._search(
            corpus_id,
            story.text,
            kinds=("story",),
            pillars=None,
            top_k=count + len(exclude),
            filters=filters,
            window=Window(as_of=_now(), open=True),
            subject=None,
            judge=(),
            vector=vector,
            seeds=seeds,
        )
        kept = [hit for hit in found.hits if hit.id not in exclude][:count]
        usage = Usage()
        if judge and kept:
            assert self.decision is not None
            kept, usage = await judge_hits(self.decision, story.text, kept, judge)
        return Hits(hits=kept, pillars=found.pillars, usage=usage)

    def _check_related(
        self, related: int, filters: dict[str, Any] | None, judge: tuple[JudgementName, ...]
    ) -> None:
        if related < 0:
            raise InputError("memory: related must be at least 0")
        validate_filters(filters)
        if judge and self.decision is None:
            raise ConfigurationError("memory: judge=True needs a decision model")

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
        judge: tuple[JudgementName, ...],
        vector: list[float] | None = None,
        seeds: list[str] | None = None,
        entities: list[str] | None = None,
        relation_types: list[str] | None = None,
        sources: list[str] | None = None,
        around: list[str] | None = None,
        hops: int = 1,
        text: TextRanking | None = None,
        fusion: Fusion | None = None,
        match: Match = "any",
        depth: int = 2,
    ) -> Hits:
        corpus_ids = _corpora(
            corpus_id, query, subject, filters, entities, relation_types, sources, around
        )
        validate_filters(filters)
        unknown = sorted(set(kinds) - set(KINDS))
        if unknown:
            raise InputError(f"memory: unknown kinds: {unknown}; choose from {list(KINDS)}")
        _at_least_one("top_k", top_k)
        chosen = self._pillars(pillars)
        if judge and self.decision is None:
            raise ConfigurationError("memory: judge=True needs a decision model")
        if (entities is not None or relation_types is not None) and "graph" not in self.pillars:
            raise ConfigurationError("memory: entities and relation_types need the graph pillar")
        if around is not None and "graph" not in self.pillars:
            raise ConfigurationError("memory: around needs the graph pillar")
        if not 0 <= hops <= 2:
            raise InputError("memory: hops is 0, 1 or 2")
        ranking = _text_ranking(text, self.text)
        fusing = _fusion(fusion, self.fusion)
        if match not in MATCHES:
            raise InputError("memory: match is any, all, phrase or web")
        _depth(depth)
        if not query.strip():
            return Hits(hits=[], pillars={}, usage=Usage())
        if filters and "graph" in chosen:
            chosen = chosen - {"graph"}
        types = None if relation_types is None else [t.strip().upper() for t in relation_types]
        looks = dict.fromkeys(corpus_ids, _Look(seeds=tuple(seeds or ())))
        unresolved: dict[str, list[str]] = {}
        if entities is not None or sources is not None or around is not None:
            looks, unresolved = await self._looks(
                corpus_ids, kinds, filters, window, subject, entities, types, sources, around, hops
            )
        fetch = top_k * fusing.window
        names = sorted(chosen)
        legs = await asyncio.gather(
            *(
                self._leg(
                    name,
                    corpus_ids,
                    query,
                    kinds,
                    fetch,
                    filters,
                    window,
                    subject,
                    vector,
                    looks,
                    types,
                    ranking,
                    match,
                    depth,
                )
                for name in names
            )
        )
        reports = {name: report for name, (_, report) in zip(names, legs, strict=True)}
        fused = FUSIONS[fusing.method].fuse(
            fusing, [(name, ranked) for name, (ranked, _) in zip(names, legs, strict=True)]
        )
        hits = await self._hydrate(fused[:top_k], window.known_at)
        usage = Usage()
        if judge:
            assert self.decision is not None
            hits, usage = await judge_hits(self.decision, query, hits, judge)
        return Hits(hits=hits, pillars=reports, usage=usage, unresolved=unresolved)

    async def _looks(
        self,
        corpus_ids: list[str],
        kinds: tuple[Kind, ...],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        entities: list[str] | None,
        relation_types: list[str] | None,
        sources: list[str] | None,
        around: list[str] | None,
        hops: int,
    ) -> tuple[dict[str, _Look], dict[str, list[str]]]:
        seeds: dict[str, list[str]] = {c: [] for c in corpus_ids}
        unresolved: dict[str, list[str]] = {}
        if entities is not None:
            seeds, unresolved = await self._resolve(corpus_ids, entities)
        passages: dict[str, tuple[Ref, ...] | None] = dict.fromkeys(corpus_ids)
        if around is not None:
            centres, missing = await self._resolve(corpus_ids, around)
            unresolved = unresolved | missing
            if entities is None:
                seeds = centres
            for corpus_id in corpus_ids:
                reached = await self._reach(
                    corpus_id, centres[corpus_id], hops, relation_types, window
                )
                found = (
                    await self.storage.mentions(
                        corpus_id,
                        sorted(reached),
                        kinds=kinds,
                        filters=filters,
                        window=window,
                        subject=subject,
                    )
                    if reached
                    else []
                )
                passages[corpus_id] = tuple(dict.fromkeys(ref for _, ref in found))
        held = None if sources is None else tuple(sources)
        scoped = sources is not None or around is not None
        looks = {
            c: _Look(
                seeds=tuple(seeds[c]),
                spot=entities is None and around is None,
                scope=Scope(sources=held, passages=passages[c]) if scoped else None,
            )
            for c in corpus_ids
        }
        return looks, unresolved

    async def _resolve(
        self, corpus_ids: list[str], names: list[str]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        wanted = list(dict.fromkeys(names))
        pairs = [(c, name) for c in corpus_ids for name in wanted]
        found = await _gather(self.storage.entities_named(c, name) for c, name in pairs)
        ids: dict[str, list[str]] = {c: [] for c in corpus_ids}
        resolved: set[str] = set()
        for (corpus_id, name), entities in zip(pairs, found, strict=True):
            ids[corpus_id].extend(e.entity_id for e in entities)
            if entities:
                resolved.add(name)
        lookups = [(name, c) for name in wanted if name not in resolved for c in corpus_ids]
        similar = await _gather(
            self.storage.similar_entities(c, name, limit=5) for name, c in lookups
        )
        suggested: dict[str, list[str]] = {}
        for (name, _), entities in zip(lookups, similar, strict=True):
            suggested.setdefault(name, []).extend(e.name for e in entities)
        unresolved = {name: list(dict.fromkeys(held))[:5] for name, held in suggested.items()}
        return {c: list(dict.fromkeys(held)) for c, held in ids.items()}, unresolved

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
        given: list[float] | None,
        looks: dict[str, _Look],
        relation_types: list[str] | None,
        ranking: TextRanking,
        match: Match,
        depth: int,
    ) -> tuple[list[tuple[str, float]], PillarReport]:
        began = time.monotonic()
        try:
            if name == "text":
                ranked = await self._text_leg(
                    corpus_ids, query, kinds, fetch, filters, window, subject, looks, ranking, match
                )
            else:
                ranked = await self._corpus_legs(
                    name,
                    corpus_ids,
                    query,
                    kinds,
                    fetch,
                    filters,
                    window,
                    subject,
                    given,
                    looks,
                    relation_types,
                    depth,
                )
        except (StorageError, ModelError) as exc:
            return [], PillarReport(
                returned=0,
                duration_ms=int((time.monotonic() - began) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
        return ranked, PillarReport(
            returned=len(ranked), duration_ms=int((time.monotonic() - began) * 1000)
        )

    async def _text_leg(
        self,
        corpus_ids: list[str],
        query: str,
        kinds: tuple[Kind, ...],
        fetch: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        looks: dict[str, _Look],
        ranking: TextRanking,
        match: Match,
    ) -> list[tuple[str, float]]:
        groups = [corpus_ids]
        if len(corpus_ids) > 1:
            languages = await _gather(self.storage.get_language(c) for c in corpus_ids)
            grouped: dict[str, list[str]] = {}
            for corpus_id, language in zip(corpus_ids, languages, strict=True):
                grouped.setdefault(language, []).append(corpus_id)
            groups = list(grouped.values())
        found = await _gather(
            self.storage.search_text(
                group,
                query,
                kinds=kinds,
                top_k=fetch,
                filters=filters,
                window=window,
                subject=subject,
                ranking=ranking,
                match=match,
                scope=_joined([looks[c].scope for c in group]),
            )
            for group in groups
        )
        if len(found) == 1:
            return [
                (_key(corpus_id, kind, passage_id), score)
                for corpus_id, (kind, passage_id, score) in found[0][:fetch]
            ]
        merged: list[tuple[int, int, str, float]] = []
        for index, found_in in enumerate(found):
            best = 0.0
            if found_in:
                _, (_, _, best) = found_in[0]
            merged.extend(
                (rank, index, _key(corpus_id, kind, passage_id), score / best if best else 0.0)
                for rank, (corpus_id, (kind, passage_id, score)) in enumerate(found_in)
            )
        merged.sort()
        return [(key, score) for _, _, key, score in merged[:fetch]]

    async def _corpus_legs(
        self,
        name: str,
        corpus_ids: list[str],
        query: str,
        kinds: tuple[Kind, ...],
        fetch: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        given: list[float] | None,
        looks: dict[str, _Look],
        relation_types: list[str] | None,
        depth: int,
    ) -> list[tuple[str, float]]:
        vector: list[float] = []
        if name == "vector":
            if given is not None:
                vector = given
            else:
                assert self.embedding is not None
                [vector] = await _embedded(self.embedding, [query], "query")

        async def one(corpus_id: str) -> list[Leg]:
            if name == "vector":
                return await self.storage.search_vector(
                    corpus_id,
                    vector,
                    kinds=kinds,
                    top_k=fetch,
                    filters=filters,
                    window=window,
                    subject=subject,
                    scope=looks[corpus_id].scope,
                )
            return await self._graph_leg(
                corpus_id,
                query,
                kinds,
                fetch,
                window,
                subject,
                looks[corpus_id],
                relation_types,
                depth,
            )

        found = await _gather(one(c) for c in corpus_ids)
        merged = sorted(
            (-score, rank, index, _key(corpus_id, kind, passage_id), score)
            for index, (corpus_id, legs) in enumerate(zip(corpus_ids, found, strict=True))
            for rank, (kind, passage_id, score) in enumerate(legs)
        )
        return [(key, score) for _, _, _, key, score in merged[:fetch]]

    async def _graph_leg(
        self,
        corpus_id: str,
        query: str,
        kinds: tuple[Kind, ...],
        fetch: int,
        window: Window,
        subject: str | None,
        look: _Look,
        relation_types: list[str] | None,
        depth: int,
    ) -> list[Leg]:
        seeds = list(look.seeds)
        if look.spot:
            named = await self.storage.entities_named_in(corpus_id, query)
            seeds = [e.entity_id for e in named] + seeds
        return await self._walk(
            corpus_id,
            seeds,
            relation_types=relation_types,
            kinds=kinds,
            fetch=fetch,
            filters=None,
            window=window,
            subject=subject,
            scope=look.scope,
            depth=depth,
        )

    async def _walk(
        self,
        corpus_id: str,
        seeds: list[str],
        *,
        relation_types: list[str] | None,
        kinds: tuple[Kind, ...],
        fetch: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        scope: Scope | None,
        depth: int,
    ) -> list[Leg]:
        hops = await self._reach(corpus_id, seeds, depth, relation_types, window)
        if not hops:
            return []
        found = await self.storage.mentions(
            corpus_id,
            sorted(hops),
            kinds=kinds,
            filters=filters,
            window=window,
            subject=subject,
            scope=scope,
        )
        return _ranked(hops, found)[:fetch]

    async def _reach(
        self,
        corpus_id: str,
        seeds: list[str],
        depth: int,
        relation_types: list[str] | None,
        window: Window,
    ) -> dict[str, int]:
        reached = dict.fromkeys(seeds, 0)
        frontier = list(reached)
        for hop in range(1, depth + 1):
            if not frontier:
                break
            relations = await self.storage.neighbors(
                corpus_id, frontier, relation_types=relation_types, window=window
            )
            frontier = list(
                dict.fromkeys(
                    n for r in relations for n in (r.from_id, r.to_id) if n not in reached
                )
            )
            reached.update(dict.fromkeys(frontier, hop))
        return reached

    async def _hydrate(
        self, fused: list[tuple[str, float, dict[str, float]]], known_at: datetime | None = None
    ) -> list[Hit]:
        named = [cast(tuple[str, str, str], tuple(json.loads(key))) for key, _, _ in fused]
        wanted: dict[str, list[Ref]] = {}
        for corpus_id, kind, passage_id in named:
            wanted.setdefault(corpus_id, []).append((kind, passage_id))
        found = await asyncio.gather(
            *(
                self.storage.get_passages(corpus_id, refs, known_at=known_at)
                for corpus_id, refs in wanted.items()
            )
        )
        passages = {
            (p.corpus_id, "chunk", p.chunk_id)
            if isinstance(p, Chunk)
            else (p.corpus_id, "story", p.story_id): p
            for held in found
            for p in held
        }
        mentioned = await self._mentioned(wanted)
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
                        entities=mentioned.get(ref, []),
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
                        entities=mentioned.get(ref, []),
                    )
                )
        return hits

    async def _mentioned(
        self, wanted: dict[str, list[Ref]]
    ) -> dict[tuple[str, str, str], list[Entity]]:
        if "graph" not in self.pillars or not wanted:
            return {}
        corpora = list(wanted)
        found = await _gather(self.storage.passage_entities(c, wanted[c]) for c in corpora)
        mentioned: dict[tuple[str, str, str], list[Entity]] = {}
        for corpus_id, pairs in zip(corpora, found, strict=True):
            for (kind, passage_id), entity in pairs:
                mentioned.setdefault((corpus_id, kind, passage_id), []).append(entity)
        return {
            ref: sorted(entities, key=lambda e: (e.key, e.type, e.entity_id))
            for ref, entities in mentioned.items()
        }

    async def expand(
        self,
        corpus_id: str,
        passage: Ref,
        *,
        by: Literal["document", "vector", "graph"],
        top_k: int = 10,
        width: int = 2,
        kinds: tuple[Kind, ...] = ("chunk", "story"),
        relation_types: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        as_of: datetime | None = None,
        known_at: datetime | None = None,
        history: bool = False,
        judge: bool | tuple[JudgementName, ...] = False,
        depth: int = 2,
    ) -> Hits:
        corpus_id = _corpus(corpus_id, passage, relation_types, filters)
        if by not in EXPANSIONS:
            raise InputError(f"memory: by is document, vector or graph, not {by!r}")
        _depth(depth)
        if not (
            isinstance(passage, tuple | list)
            and len(passage) == 2
            and passage[0] in KINDS
            and isinstance(passage[1], str)
        ):
            raise InputError("memory: passage is a (kind, id) pair, its kind chunk or story")
        _at_least_one("top_k", top_k)
        _at_least_one("width", width)
        validate_filters(filters)
        unknown = sorted(set(kinds) - set(KINDS))
        if unknown:
            raise InputError(f"memory: unknown kinds: {unknown}; choose from {list(KINDS)}")
        judged = _judgements(judge)
        if judged and self.decision is None:
            raise ConfigurationError("memory: judge=True needs a decision model")
        if by != "document" and by not in self.pillars:
            raise ConfigurationError(f"memory: expand by {by} needs the {by} pillar")
        window = _window(as_of, None, None, history, known_at)
        ref: Ref = (passage[0], passage[1])
        found = await self.storage.get_passages(corpus_id, [ref], known_at=window.known_at)
        if not found:
            raise NotFoundError(f"memory: {ref[0]} {ref[1]!r} was not found")
        start = found[0]
        if by == "document":
            ranked = await self._document(corpus_id, start, top_k, width, filters, window)
        elif by == "vector":
            ranked = await self._nearby(corpus_id, ref, kinds, top_k, filters, window)
        else:
            types = None if relation_types is None else [t.strip().upper() for t in relation_types]
            ranked = await self._connected(
                corpus_id, ref, kinds, top_k, types, filters, window, depth
            )
        hits = await self._hydrate(rrf.fuse(RRF(), [(by, ranked)]), window.known_at)
        usage = Usage()
        if judged and hits:
            assert self.decision is not None
            hits, usage = await judge_hits(self.decision, start.text, hits, judged)
        return Hits(hits=hits, pillars={}, usage=usage)

    async def _document(
        self,
        corpus_id: str,
        start: Chunk | Story,
        top_k: int,
        width: int,
        filters: dict[str, Any] | None,
        window: Window,
    ) -> list[tuple[str, float]]:
        if isinstance(start, Story):
            if start.subject is None:
                raise InputError(
                    f"memory: story {start.story_id!r} has no subject; expand by document needs one"
                )
            stories: list[Story] = []
            cursor: str | None = None
            while True:
                page, cursor = await self.storage.list_stories(
                    corpus_id,
                    subject=start.subject,
                    window=window,
                    filters=filters,
                    limit=1000,
                    cursor=cursor,
                )
                stories.extend(page)
                if cursor is None:
                    break
            moment = start.valid_from
            others = [s for s in stories if s.story_id != start.story_id]
            others.sort(key=lambda s: (abs(s.valid_from - moment), s.valid_from, s.story_id))
            kept = sorted(others[:top_k], key=lambda s: (s.valid_from, s.story_id))
            return [(_key(corpus_id, "story", s.story_id), 1.0) for s in kept]
        around = await self.storage.chunks_around(
            corpus_id, start.source_id, start.position, width=width, window=window, filters=filters
        )
        return [
            (_key(corpus_id, "chunk", c.chunk_id), 1.0)
            for c in around
            if c.chunk_id != start.chunk_id
        ]

    async def _nearby(
        self,
        corpus_id: str,
        ref: Ref,
        kinds: tuple[Kind, ...],
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
    ) -> list[tuple[str, float]]:
        vector = await self.storage.passage_vector(corpus_id, ref)
        if vector is None:
            return []
        legs = await self.storage.search_vector(
            corpus_id,
            vector,
            kinds=kinds,
            top_k=top_k + 1,
            filters=filters,
            window=window,
            subject=None,
        )
        return [(_key(corpus_id, k, i), score) for k, i, score in legs if (k, i) != ref][:top_k]

    async def _connected(
        self,
        corpus_id: str,
        ref: Ref,
        kinds: tuple[Kind, ...],
        top_k: int,
        relation_types: list[str] | None,
        filters: dict[str, Any] | None,
        window: Window,
        depth: int,
    ) -> list[tuple[str, float]]:
        mentioned = await self.storage.passage_entities(corpus_id, [ref])
        legs = await self._walk(
            corpus_id,
            [entity.entity_id for _, entity in mentioned],
            relation_types=relation_types,
            kinds=kinds,
            fetch=top_k + 1,
            filters=filters,
            window=window,
            subject=None,
            scope=None,
            depth=depth,
        )
        return [(_key(corpus_id, k, i), score) for k, i, score in legs if (k, i) != ref][:top_k]

    async def neighbors(
        self,
        corpus_id: str,
        entity: str,
        *,
        entity_type: str | None = None,
        hops: int = 1,
        relation_types: list[str] | None = None,
        as_of: datetime | None = None,
        known_at: datetime | None = None,
        history: bool = False,
    ) -> Neighborhood:
        corpus_id = _corpus(corpus_id, entity, entity_type, relation_types)
        if "graph" not in self.pillars:
            raise ConfigurationError("memory: neighbors needs the graph pillar")
        window = _window(as_of, None, None, history, known_at)
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
            found = await self.storage.neighbors(
                corpus_id, frontier, relation_types=relation_types, window=window
            )
            relations.extend(r for r in found if r not in relations)
            new_ids = [n for r in found for n in (r.from_id, r.to_id) if n not in reached]
            for found_entity in await self.storage.entities_by_id(corpus_id, new_ids):
                reached[found_entity.entity_id] = found_entity
            frontier = new_ids
            if not frontier:
                break
        mentioned = await self.storage.mentions(
            corpus_id,
            list(reached),
            kinds=("chunk", "story"),
            filters=None,
            window=window,
            subject=None,
        )
        refs = list(dict.fromkeys(ref for _, ref in mentioned))
        passages = await self._hydrate(
            [(_key(corpus_id, k, i), 1.0, {"graph": 1.0}) for k, i in refs], window.known_at
        )
        return Neighborhood(entities=list(reached.values()), relations=relations, passages=passages)
