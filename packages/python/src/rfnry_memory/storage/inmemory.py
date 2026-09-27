from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError, StorageError
from rfnry_memory.filters import matches_filters
from rfnry_memory.graph.identity import name_key, similarity
from rfnry_memory.models import (
    Chunk,
    Counts,
    Entity,
    Kind,
    Leg,
    Ref,
    Relation,
    Source,
    Story,
    Window,
)
from rfnry_memory.storage.storage import GraphWrite, SourceWrite, text_hash, total

_WORD = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bm25(
    query: list[str], docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75
) -> list[float]:
    if not docs:
        return []
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    scores: list[float] = []
    for doc in docs:
        tf = Counter(doc)
        score = 0.0
        for term in query:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(score)
    return scores


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _in_window(story: Story, window: Window) -> bool:
    start = story.valid_from
    if window.as_of is not None and (
        start > window.as_of or (story.valid_to is not None and story.valid_to <= window.as_of)
    ):
        return False
    if window.since is not None and start < window.since:
        return False
    return not (window.until is not None and start > window.until)


@dataclass
class _Passage:
    kind: Kind
    id: str
    text: str
    metadata: dict[str, Any]
    vector: list[float] | None
    model: str | None
    chunk: Chunk | None = None
    story: Story | None = None


def _of_subject(passage: _Passage, subject: str | None) -> bool:
    return subject is None or (passage.story is not None and passage.story.subject == subject)


@dataclass
class _Corpus:
    sources: dict[str, Source] = field(default_factory=dict)
    passages: dict[Ref, _Passage] = field(default_factory=dict)
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    mentions: set[tuple[str, Kind, str]] = field(default_factory=set)


class InMemoryStorage:
    def __init__(self) -> None:
        self._corpora: dict[str, _Corpus] = {}
        self.dimension: int | None = None
        self._initialized = False

    def _corpus(self, corpus_id: str) -> _Corpus:
        return self._corpora.setdefault(corpus_id, _Corpus())

    async def initialize(self, *, dimension: int | None) -> None:
        if not self._initialized:
            self.dimension, self._initialized = dimension, True
            return
        if dimension is None or dimension == self.dimension:
            return
        held = f"for dimension {self.dimension}" if self.dimension else "without vectors"
        raise ConfigurationError(
            f"inmemory: this storage was initialized {held}; this memory embeds {dimension} — use another storage"
        )

    async def put_source(
        self,
        corpus_id: str,
        source: Source,
        chunks: list[Chunk],
        vectors: list[list[float]] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> None:
        if source.corpus_id != corpus_id or any(c.corpus_id != corpus_id for c in chunks):
            raise StorageError("inmemory: row corpus does not match the call corpus")
        corpus = self._corpus(corpus_id)
        graph = graph or GraphWrite()
        ids = _graph_ids(corpus, corpus_id, graph, {("chunk", c.chunk_id) for c in chunks})
        _delete_source(corpus, source.source_id)
        _put_source(corpus, SourceWrite(source, chunks, vectors), model)
        _apply_graph(corpus, graph, ids)

    async def get_source(self, corpus_id: str, source_id: str) -> Source | None:
        return self._corpus(corpus_id).sources.get(source_id)

    async def list_sources(
        self, corpus_id: str, *, filters: dict[str, Any] | None, limit: int, cursor: str | None
    ) -> tuple[list[Source], str | None]:
        rows = [
            s
            for s in self._corpus(corpus_id).sources.values()
            if matches_filters(s.metadata, filters)
        ]
        rows.sort(key=lambda s: (s.created_at, s.source_id), reverse=True)
        return _page(rows, limit, cursor)

    async def source_chunks(self, corpus_id: str, source_id: str) -> list[Chunk]:
        chunks = [
            p.chunk
            for p in self._corpus(corpus_id).passages.values()
            if p.chunk is not None and p.chunk.source_id == source_id
        ]
        return sorted(chunks, key=lambda c: c.position)

    async def delete_source(self, corpus_id: str, source_id: str) -> Counts:
        return _delete_source(self._corpus(corpus_id), source_id)

    async def replace_sources(
        self,
        corpus_id: str,
        filters: dict[str, Any],
        writes: list[SourceWrite],
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> Counts:
        if any(
            source.corpus_id != corpus_id or any(c.corpus_id != corpus_id for c in chunks)
            for source, chunks, _ in writes
        ):
            raise StorageError("inmemory: row corpus does not match the call corpus")
        corpus = self._corpus(corpus_id)
        graph = graph or GraphWrite()
        ids = _graph_ids(
            corpus, corpus_id, graph, {("chunk", c.chunk_id) for w in writes for c in w.chunks}
        )
        matching = [
            s.source_id for s in corpus.sources.values() if matches_filters(s.metadata, filters)
        ]
        deleted = total([_delete_source(corpus, source_id) for source_id in matching])
        for write in writes:
            _put_source(corpus, write, model)
        _apply_graph(corpus, graph, ids)
        return deleted

    async def put_story(
        self,
        corpus_id: str,
        story: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> Story:
        if story.corpus_id != corpus_id:
            raise StorageError("inmemory: row corpus does not match the call corpus")
        corpus = self._corpus(corpus_id)
        holder = await self.find_open_story(corpus_id, text_hash(story.text))
        if holder is not None and holder.story_id != story.story_id:
            return holder
        graph = graph or GraphWrite()
        ids = _graph_ids(corpus, corpus_id, graph, {("story", story.story_id)})
        corpus.passages[("story", story.story_id)] = _Passage(
            "story", story.story_id, story.text, dict(story.metadata), vector, model, story=story
        )
        _apply_graph(corpus, graph, ids)
        return story

    async def find_open_story(self, corpus_id: str, hash: str) -> Story | None:
        for passage in self._corpus(corpus_id).passages.values():
            story = passage.story
            if story is not None and story.valid_to is None and text_hash(story.text) == hash:
                return story
        return None

    async def supersede_story(
        self,
        corpus_id: str,
        old_id: str,
        valid_to: datetime,
        new: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> tuple[Story, Story]:
        if new.corpus_id != corpus_id:
            raise StorageError("inmemory: row corpus does not match the call corpus")
        corpus = self._corpus(corpus_id)
        passage = corpus.passages.get(("story", old_id))
        if passage is None or passage.story is None:
            raise NotFoundError(f"inmemory: story {old_id!r} was not found")
        if passage.story.valid_to is not None:
            raise InputError(
                f"inmemory: story {old_id!r} is already superseded; supersede the story that replaced it"
            )
        holder = await self.find_open_story(corpus_id, text_hash(new.text))
        if holder is not None and holder.story_id != old_id:
            raise InputError(
                f"inmemory: story {holder.story_id!r}, true since {holder.valid_from.isoformat()}, already says that; supersede or forget it instead"
            )
        graph = graph or GraphWrite()
        ids = _graph_ids(corpus, corpus_id, graph, {("story", new.story_id)})
        closed = passage.story.model_copy(update={"valid_to": valid_to})
        passage.story = closed
        corpus.passages[("story", new.story_id)] = _Passage(
            "story", new.story_id, new.text, dict(new.metadata), vector, model, story=new
        )
        _apply_graph(corpus, graph, ids)
        return closed, new

    async def get_stories(self, corpus_id: str, story_ids: list[str]) -> list[Story]:
        corpus = self._corpus(corpus_id)
        found = [corpus.passages.get(("story", sid)) for sid in story_ids]
        return [p.story for p in found if p is not None and p.story is not None]

    async def list_stories(
        self,
        corpus_id: str,
        *,
        subject: str | None,
        window: Window,
        filters: dict[str, Any] | None,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[Story], str | None]:
        rows = [
            p.story
            for p in self._corpus(corpus_id).passages.values()
            if p.story is not None
            and (subject is None or p.story.subject == subject)
            and _in_window(p.story, window)
            and matches_filters(p.story.metadata, filters)
        ]
        rows.sort(key=lambda s: (s.valid_from, s.story_id), reverse=True)
        return _page(rows, limit, cursor)

    async def delete_story(self, corpus_id: str, story_id: str) -> Counts:
        corpus = self._corpus(corpus_id)
        removed = corpus.passages.pop(("story", story_id), None)
        return _drop_evidence(
            corpus, Counts(stories=int(removed is not None)), {("story", story_id)}
        )

    def _candidates(
        self,
        corpus_id: str,
        kinds: tuple[Kind, ...],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> list[_Passage]:
        return [
            p
            for p in self._corpus(corpus_id).passages.values()
            if p.kind in kinds
            and matches_filters(p.metadata, filters)
            and (p.story is None or _in_window(p.story, window))
            and _of_subject(p, subject)
        ]

    async def search_text(
        self,
        corpus_id: str,
        query: str,
        *,
        kinds: tuple[Kind, ...],
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> list[Leg]:
        candidates = self._candidates(corpus_id, kinds, filters, window, subject)
        scores = _bm25(_tokens(query), [_tokens(p.text) for p in candidates])
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda item: item[1], reverse=True
        )
        legs: list[Leg] = [(p.kind, p.id, score) for p, score in ranked if score > 0]
        return legs[:top_k]

    async def search_vector(
        self,
        corpus_id: str,
        vector: list[float],
        *,
        kinds: tuple[Kind, ...],
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> list[Leg]:
        candidates = [
            p
            for p in self._candidates(corpus_id, kinds, filters, window, subject)
            if p.vector is not None
        ]
        ranked = sorted(
            ((p, _cosine(vector, p.vector or [])) for p in candidates),
            key=lambda item: item[1],
            reverse=True,
        )
        legs: list[Leg] = [(p.kind, p.id, score) for p, score in ranked]
        return legs[:top_k]

    async def get_passages(self, corpus_id: str, refs: list[Ref]) -> list[Chunk | Story]:
        corpus = self._corpus(corpus_id)
        found: list[Chunk | Story] = []
        for ref in refs:
            passage = corpus.passages.get(ref)
            if passage is None:
                continue
            found.append(passage.chunk if passage.chunk is not None else passage.story)  # type: ignore[arg-type]
        return found

    async def similar_entities(self, corpus_id: str, name: str, *, limit: int) -> list[Entity]:
        scored = []
        for entity in self._corpus(corpus_id).entities.values():
            overlap = similarity(name, [entity.name, *entity.aliases])
            if overlap > 0:
                scored.append((overlap, entity))
        scored.sort(key=lambda item: (-item[0], item[1].entity_id))
        return [entity for _, entity in scored[:limit]]

    async def entities_named(self, corpus_id: str, name_or_key: str) -> list[Entity]:
        key = name_key(name_or_key)
        found = [
            e
            for e in self._corpus(corpus_id).entities.values()
            if e.key == key or key in {name_key(a) for a in e.aliases}
        ]
        return sorted(found, key=lambda e: e.entity_id)

    async def entities_by_id(self, corpus_id: str, entity_ids: list[str]) -> list[Entity]:
        entities = self._corpus(corpus_id).entities
        return [entities[i] for i in entity_ids if i in entities]

    async def entities_named_in(self, corpus_id: str, text: str) -> list[Entity]:
        haystack = name_key(text)
        found = [
            e
            for e in self._corpus(corpus_id).entities.values()
            if e.key in haystack or any(name_key(a) in haystack for a in e.aliases)
        ]
        return sorted(found, key=lambda e: e.entity_id)

    async def neighbors(
        self, corpus_id: str, entity_ids: list[str], *, relation_types: list[str] | None
    ) -> list[Relation]:
        wanted = set(entity_ids)
        return [
            r
            for r in self._corpus(corpus_id).relations
            if (r.from_id in wanted or r.to_id in wanted)
            and (relation_types is None or r.type in relation_types)
        ]

    async def mentions(
        self,
        corpus_id: str,
        entity_ids: list[str],
        *,
        kinds: tuple[Kind, ...],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> list[Ref]:
        corpus = self._corpus(corpus_id)
        wanted = set(entity_ids)
        refs: list[Ref] = []
        for entity_id, kind, passage_id in sorted(corpus.mentions):
            if entity_id not in wanted or kind not in kinds or (kind, passage_id) in refs:
                continue
            passage = corpus.passages.get((kind, passage_id))
            if (
                passage is None
                or not matches_filters(passage.metadata, filters)
                or not _of_subject(passage, subject)
            ):
                continue
            if passage.story is not None and not _in_window(passage.story, window):
                continue
            refs.append((kind, passage_id))
        return refs

    async def passages_embedded_with(
        self, corpus_id: str, *, not_model: str, limit: int
    ) -> list[Chunk | Story]:
        stale = [p for p in self._corpus(corpus_id).passages.values() if p.model != not_model]
        return [p.chunk if p.chunk is not None else p.story for p in stale[:limit]]  # type: ignore[misc]

    async def set_vectors(
        self, corpus_id: str, refs: list[Ref], vectors: list[list[float]], model: str
    ) -> None:
        corpus = self._corpus(corpus_id)
        for ref, vector in zip(refs, vectors, strict=True):
            passage = corpus.passages.get(ref)
            if passage is not None:
                passage.vector = vector
                passage.model = model

    async def delete_corpus(self, corpus_id: str) -> Counts:
        residue = await self.count_corpus(corpus_id)
        self._corpora.pop(corpus_id, None)
        return residue

    async def count_corpus(self, corpus_id: str) -> Counts:
        corpus = self._corpora.get(corpus_id)
        if corpus is None:
            return Counts()
        return Counts(
            sources=len(corpus.sources),
            chunks=sum(1 for p in corpus.passages.values() if p.kind == "chunk"),
            stories=sum(1 for p in corpus.passages.values() if p.kind == "story"),
            entities=len(corpus.entities),
            relations=len(corpus.relations),
            mentions=len(corpus.mentions),
        )


def _page[T](rows: list[T], limit: int, cursor: str | None) -> tuple[list[T], str | None]:
    offset = int(cursor) if cursor else 0
    page = rows[offset : offset + limit]
    next_cursor = str(offset + limit) if offset + limit < len(rows) else None
    return page, next_cursor


def _put_source(corpus: _Corpus, write: SourceWrite, model: str | None) -> None:
    source, chunks, vectors = write
    corpus.sources[source.source_id] = source
    for index, chunk in enumerate(chunks):
        corpus.passages[("chunk", chunk.chunk_id)] = _Passage(
            "chunk",
            chunk.chunk_id,
            chunk.text,
            dict(source.metadata),
            vectors[index] if vectors is not None else None,
            model,
            chunk=chunk,
        )


def _delete_source(corpus: _Corpus, source_id: str) -> Counts:
    refs = {
        ref
        for ref, passage in corpus.passages.items()
        if passage.chunk is not None and passage.chunk.source_id == source_id
    }
    for ref in refs:
        del corpus.passages[ref]
    removed = corpus.sources.pop(source_id, None)
    return _drop_evidence(corpus, Counts(sources=int(removed is not None), chunks=len(refs)), refs)


def _drop_evidence(corpus: _Corpus, deleted: Counts, refs: set[Ref]) -> Counts:
    before_relations = len(corpus.relations)
    corpus.relations = [r for r in corpus.relations if (r.evidence_kind, r.evidence_id) not in refs]
    before_mentions = len(corpus.mentions)
    corpus.mentions = {m for m in corpus.mentions if (m[1], m[2]) not in refs}
    return deleted.model_copy(
        update={
            "relations": deleted.relations + before_relations - len(corpus.relations),
            "mentions": deleted.mentions + before_mentions - len(corpus.mentions),
        }
    )


def _graph_ids(
    corpus: _Corpus, corpus_id: str, graph: GraphWrite, passages: set[Ref]
) -> dict[str, str]:
    graph.check(corpus_id, passages)
    by_key = {(e.key, e.type): e.entity_id for e in corpus.entities.values()}
    ids: dict[str, str] = {}
    for entity in graph.entities:
        ids[entity.entity_id] = by_key.setdefault((entity.key, entity.type), entity.entity_id)
    missing = graph.named(ids) - set(corpus.entities) - set(ids.values())
    if missing:
        raise StorageError(f"inmemory: graph facts name unknown entities: {sorted(missing)}")
    return ids


def _apply_graph(corpus: _Corpus, graph: GraphWrite, ids: dict[str, str]) -> None:
    for entity in graph.entities:
        target = ids[entity.entity_id]
        if target not in corpus.entities:
            corpus.entities[target] = entity
        else:
            _alias(corpus, target, entity.aliases)
    for entity_id, alias in graph.aliases:
        _alias(corpus, ids.get(entity_id, entity_id), (alias,))
    existing = {
        (r.from_id, r.to_id, r.type, r.evidence_kind, r.evidence_id) for r in corpus.relations
    }
    for relation in graph.relations:
        row = relation.model_copy(
            update={
                "from_id": ids.get(relation.from_id, relation.from_id),
                "to_id": ids.get(relation.to_id, relation.to_id),
            }
        )
        signature = (row.from_id, row.to_id, row.type, row.evidence_kind, row.evidence_id)
        if signature not in existing:
            corpus.relations.append(row)
            existing.add(signature)
    for entity_id, kind, passage_id in graph.mentions:
        corpus.mentions.add((ids.get(entity_id, entity_id), kind, passage_id))


def _alias(corpus: _Corpus, entity_id: str, aliases: tuple[str, ...]) -> None:
    entity = corpus.entities[entity_id]
    added = tuple(a for a in dict.fromkeys(aliases) if a not in entity.aliases)
    if added:
        corpus.entities[entity_id] = entity.model_copy(
            update={"aliases": (*entity.aliases, *added)}
        )
