from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Any

import snowballstemmer

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError, StorageError
from rfnry_memory.filters import matches_filters
from rfnry_memory.graph.identity import similarity
from rfnry_memory.models import (
    Chunk,
    Counts,
    Entity,
    Kind,
    Language,
    Leg,
    Ref,
    Relation,
    Source,
    Story,
    Window,
)
from rfnry_memory.storage.stopwords import ENGLISH, PORTUGUESE, SPANISH
from rfnry_memory.storage.storage import (
    GraphWrite,
    SourceWrite,
    chunk_digest,
    ended_on_write,
    is_open,
    known,
    known_chunk,
    name_key,
    reconciled,
    source_seen,
    successors,
    text_hash,
    total,
)

_TOKEN = re.compile(r"[-+]?[0-9]+\.[0-9]+|[^\W_]+|[-+][0-9]+")
_STOP_WORDS = {"english": ENGLISH, "spanish": SPANISH, "portuguese": PORTUGUESE}


def _lower(token: str) -> str:
    return "".join(c.lower()[0] for c in token)


_FOLDS = str.maketrans(
    {"ß": "ss", "æ": "ae", "ø": "o", "ł": "l", "œ": "oe", "đ": "d", "þ": "th", "ð": "d", "ı": "i"}
    | {"ĸ": "q", "ŋ": "n", "ŧ": "t", "ħ": "h"}
)


def _unaccent(token: str) -> str:
    folded = unicodedata.normalize("NFKD", token.translate(_FOLDS))
    return "".join(c for c in folded if not unicodedata.combining(c))


def lexemes(text: str, language: Language = "english") -> list[str]:
    stop = _STOP_WORDS.get(language, frozenset())
    stem = snowballstemmer.stemmer(language).stemWord if language in _STOP_WORDS else (lambda w: w)
    fold = language != "english"
    plain = "".join(" " if c.isalnum() and not (c.isalpha() or c.isascii()) else c for c in text)
    found: list[str] = []
    for token in map(_lower, _TOKEN.findall(plain)):
        if fold:
            token = _unaccent(token)
        if not token.isalpha():
            found.append(token)
        elif token not in stop:
            found.append(stem(token))
    return found


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
    if window.as_of is not None:
        if story.valid_to is not None and story.valid_to <= window.as_of:
            return False
        if window.open and story.superseded_by is not None:
            return False
        if not window.open and start > window.as_of:
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
    written_to: datetime | None = None


def _known(passage: _Passage, window: Window) -> Story | None:
    story = passage.story
    if story is None or window.known_at is None:
        return story
    if story.created_at > window.known_at:
        return None
    return known(story, passage.written_to, window.known_at)


def _seen(passage: _Passage, window: Window) -> Story | None:
    view = _known(passage, window)
    return view if view is not None and _in_window(view, window) else None


def _chunk_view(corpus: _Corpus, chunk: Chunk, window: Window) -> Chunk | None:
    source = corpus.sources.get(chunk.source_id)
    if source is None:
        return None
    if source_seen(source, known_at=window.known_at, history=window.as_of is None) is None:
        return None
    return known_chunk(chunk, window.known_at)


def _visible(corpus: _Corpus, passage: _Passage, window: Window) -> bool:
    if passage.chunk is not None:
        return _chunk_view(corpus, passage.chunk, window) is not None
    return _seen(passage, window) is not None


def _evidenced(corpus: _Corpus, relation: Relation, window: Window) -> bool:
    passage = corpus.passages.get((relation.evidence_kind, relation.evidence_id))
    if passage is None:
        return False
    return _visible(corpus, passage, window)


def _searched(passage: _Passage) -> str:
    chunk = passage.chunk
    if chunk is None:
        return passage.text
    return " ".join(
        part for part in (chunk.source_name, chunk.title, chunk.context, passage.text) if part
    )


def _of_subject(passage: _Passage, subject: str | None) -> bool:
    return subject is None or (passage.story is not None and passage.story.subject == subject)


@dataclass
class _Corpus:
    sources: dict[str, Source] = field(default_factory=dict)
    passages: dict[Ref, _Passage] = field(default_factory=dict)
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    mentions: set[tuple[str, Kind, str, str]] = field(default_factory=set)
    language: Language | None = None
    history: bool | None = None


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
        touched: set[str] = set()
        _delete_source(corpus, source.source_id, touched)
        _put_source(corpus, SourceWrite(source, chunks, vectors), model)
        _reconcile(corpus, touched | _apply_graph(corpus, graph, ids))

    async def get_source(self, corpus_id: str, source_id: str) -> Source | None:
        return self._corpus(corpus_id).sources.get(source_id)

    async def list_sources(
        self,
        corpus_id: str,
        *,
        filters: dict[str, Any] | None,
        limit: int,
        cursor: str | None,
        known_at: datetime | None = None,
        history: bool = False,
    ) -> tuple[list[Source], str | None]:
        rows = [
            view
            for s in self._corpus(corpus_id).sources.values()
            if (view := source_seen(s, known_at=known_at, history=history)) is not None
            and matches_filters(view.metadata, filters)
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

    async def source_digests(self, corpus_id: str) -> list[tuple[str, dict[str, Any], str, int]]:
        return [
            (s.name, s.metadata, sha256(s.text.encode("utf-8")).hexdigest(), s.chunks)
            for s in self._corpus(corpus_id).sources.values()
            if s.retired_at is None
        ]

    async def delete_source(self, corpus_id: str, source_id: str) -> Counts:
        corpus = self._corpus(corpus_id)
        touched: set[str] = set()
        deleted = total(
            [_delete_source(corpus, version, touched) for version in _versions(corpus, source_id)]
        )
        return deleted.model_copy(update={"entities": _reconcile(corpus, touched)})

    async def forget_retired(self, corpus_id: str, *, before: datetime) -> Counts:
        corpus = self._corpus(corpus_id)
        gone = sorted(
            s.source_id
            for s in corpus.sources.values()
            if s.retired_at is not None and s.retired_at < before
        )
        touched: set[str] = set()
        deleted = total([_delete_source(corpus, source_id, touched) for source_id in gone])
        return deleted.model_copy(update={"entities": _reconcile(corpus, touched)})

    async def sync_sources(
        self,
        corpus_id: str,
        filters: dict[str, Any],
        writes: list[SourceWrite],
        model: str | None,
        *,
        at: datetime,
        graph: GraphWrite | None = None,
        keep: dict[str, str] | None = None,
    ) -> tuple[Counts, list[str]] | None:
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
        matching = {
            s.source_id: s
            for s in corpus.sources.values()
            if s.retired_at is None and matches_filters(s.metadata, filters)
        }
        keep = keep or {}
        if any(
            source_id not in matching or matching[source_id].digest != digest
            for source_id, digest in keep.items()
        ) or (corpus.history and any(s.created_at > at for s in matching.values())):
            return None
        dropped = sorted(source_id for source_id in matching if source_id not in keep)
        touched: set[str] = set()
        if corpus.history:
            later = successors(writes)
            for source_id in dropped:
                _retire(corpus, matching[source_id], at, later.get(matching[source_id].name))
            deleted, retired = Counts(), dropped
        else:
            deleted = total([_delete_source(corpus, source_id, touched) for source_id in dropped])
            retired = []
        for write in writes:
            _put_source(corpus, write, model)
        pruned = _reconcile(corpus, touched | _apply_graph(corpus, graph, ids))
        return deleted.model_copy(update={"entities": pruned}), retired

    async def chunk_vectors(
        self, corpus_id: str, model: str, digests: list[str]
    ) -> dict[str, list[float]]:
        wanted = set(digests)
        found: dict[str, list[float]] = {}
        for passage in self._corpus(corpus_id).passages.values():
            if passage.chunk is None or passage.vector is None or passage.model != model:
                continue
            digest = chunk_digest(passage.text)
            if digest in wanted:
                found.setdefault(digest, list(passage.vector))
        return found

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
        holder = _holder(corpus, story)
        if holder is not None and holder.story_id != story.story_id:
            return holder
        graph = graph or GraphWrite()
        ids = _graph_ids(corpus, corpus_id, graph, {("story", story.story_id)})
        corpus.passages[("story", story.story_id)] = _Passage(
            "story",
            story.story_id,
            story.text,
            dict(story.metadata),
            vector,
            model,
            story=story,
            written_to=story.valid_to,
        )
        _reconcile(corpus, _apply_graph(corpus, graph, ids))
        return story

    async def find_open_story(self, corpus_id: str, hash: str, *, at: datetime) -> Story | None:
        return _open_holder(self._corpus(corpus_id), hash, at)

    async def supersede_story(
        self,
        corpus_id: str,
        old_ids: list[str],
        new: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> tuple[list[Story], Story]:
        if new.corpus_id != corpus_id:
            raise StorageError("inmemory: row corpus does not match the call corpus")
        if not old_ids or len(set(old_ids)) != len(old_ids):
            raise InputError("inmemory: supersede_story needs distinct story ids")
        corpus = self._corpus(corpus_id)
        olds: list[tuple[_Passage, Story]] = []
        for old_id in old_ids:
            passage = corpus.passages.get(("story", old_id))
            if passage is None or passage.story is None:
                raise NotFoundError(f"inmemory: story {old_id!r} was not found")
            _closable(passage.story, new)
            olds.append((passage, passage.story))
        holder = _open_holder(corpus, text_hash(new.text), new.created_at)
        if holder is not None and holder.story_id not in old_ids:
            raise InputError(
                f"inmemory: story {holder.story_id!r}, true since {holder.valid_from.isoformat()}, already says that; supersede or delete it instead"
            )
        graph = graph or GraphWrite()
        ids = _graph_ids(corpus, corpus_id, graph, {("story", new.story_id)})
        closed: list[Story] = []
        for passage, story in olds:
            passage.story = story.model_copy(
                update={
                    "valid_to": new.valid_from,
                    "superseded_by": new.story_id,
                    "closed_at": new.created_at,
                }
            )
            closed.append(passage.story)
        corpus.passages[("story", new.story_id)] = _Passage(
            "story",
            new.story_id,
            new.text,
            dict(new.metadata),
            vector,
            model,
            story=new,
            written_to=new.valid_to,
        )
        _reconcile(corpus, _apply_graph(corpus, graph, ids))
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
            view
            for p in self._corpus(corpus_id).passages.values()
            if (view := _seen(p, window)) is not None
            and (subject is None or view.subject == subject)
            and matches_filters(view.metadata, filters)
        ]
        rows.sort(key=lambda s: (s.valid_from, s.story_id), reverse=True)
        return _page(rows, limit, cursor)

    async def delete_story(self, corpus_id: str, story_id: str) -> Counts:
        corpus = self._corpus(corpus_id)
        removed = corpus.passages.pop(("story", story_id), None)
        touched: set[str] = set()
        deleted = _drop_evidence(
            corpus, Counts(stories=int(removed is not None)), {("story", story_id)}, touched
        )
        return deleted.model_copy(update={"entities": _reconcile(corpus, touched)})

    def _candidates(
        self,
        corpus_id: str,
        kinds: tuple[Kind, ...],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> list[_Passage]:
        corpus = self._corpus(corpus_id)
        return [
            p
            for p in corpus.passages.values()
            if p.kind in kinds
            and matches_filters(p.metadata, filters)
            and _visible(corpus, p, window)
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
        language = self._corpus(corpus_id).language or "english"
        scores = _bm25(
            lexemes(query, language), [lexemes(_searched(p), language) for p in candidates]
        )
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda item: (-item[1], item[0].id)
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
            key=lambda item: (-item[1], item[0].id),
        )
        legs: list[Leg] = [(p.kind, p.id, score) for p, score in ranked]
        return legs[:top_k]

    async def get_passages(
        self, corpus_id: str, refs: list[Ref], *, known_at: datetime | None = None
    ) -> list[Chunk | Story]:
        corpus = self._corpus(corpus_id)
        found: list[Chunk | Story] = []
        for ref in refs:
            passage = corpus.passages.get(ref)
            if passage is None:
                continue
            if passage.chunk is not None:
                found.append(known_chunk(passage.chunk, known_at))
            elif passage.story is not None:
                found.append(known(passage.story, passage.written_to, known_at))
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
        self,
        corpus_id: str,
        entity_ids: list[str],
        *,
        relation_types: list[str] | None,
        window: Window,
    ) -> list[Relation]:
        corpus = self._corpus(corpus_id)
        wanted = set(entity_ids)
        found = [
            r
            for r in corpus.relations
            if (r.from_id in wanted or r.to_id in wanted)
            and (relation_types is None or r.type in relation_types)
            and _evidenced(corpus, r, window)
        ]
        return sorted(
            found, key=lambda r: (r.from_id, r.to_id, r.type, r.evidence_kind, r.evidence_id)
        )

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
        mentioned = {(k, p) for e, k, p, _ in corpus.mentions if e in wanted and k in kinds}
        refs: list[Ref] = []
        for kind, passage_id in sorted(mentioned):
            passage = corpus.passages.get((kind, passage_id))
            if (
                passage is None
                or not matches_filters(passage.metadata, filters)
                or not _of_subject(passage, subject)
            ):
                continue
            if not _visible(corpus, passage, window):
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
            corpora=int(corpus.language is not None or corpus.history is not None),
        )

    async def get_language(self, corpus_id: str) -> Language:
        return self._corpus(corpus_id).language or "english"

    async def set_language(self, corpus_id: str, language: Language) -> int:
        corpus = self._corpus(corpus_id)
        before, corpus.language = corpus.language or "english", language
        return 0 if before == language else len(corpus.passages)

    async def get_history(self, corpus_id: str) -> bool:
        return bool(self._corpus(corpus_id).history)

    async def set_history(self, corpus_id: str, keep: bool) -> None:
        self._corpus(corpus_id).history = keep

    async def count_vectors(self, corpus_id: str) -> dict[str | None, int]:
        passages = self._corpus(corpus_id).passages.values()
        return dict(Counter(p.model if p.vector is not None else None for p in passages))


def _page[T](rows: list[T], limit: int, cursor: str | None) -> tuple[list[T], str | None]:
    offset = int(cursor) if cursor else 0
    page = rows[offset : offset + limit]
    next_cursor = str(offset + limit) if offset + limit < len(rows) else None
    return page, next_cursor


def _closable(story: Story, new: Story) -> None:
    if story.superseded_by is not None:
        raise InputError(
            f"inmemory: story {story.story_id!r} is already superseded; supersede the story that replaced it"
        )
    if story.valid_to is not None and story.valid_to <= new.created_at:
        raise InputError(
            f"inmemory: story {story.story_id!r} ended at {story.valid_to.isoformat()}; remember a new story instead"
        )


def _open_holder(corpus: _Corpus, hash: str, at: datetime) -> Story | None:
    for passage in corpus.passages.values():
        story = passage.story
        if story is not None and is_open(story, at) and text_hash(story.text) == hash:
            return story
    return None


def _holder(corpus: _Corpus, story: Story) -> Story | None:
    hash = text_hash(story.text)
    if not ended_on_write(story):
        return _open_holder(corpus, hash, story.created_at)
    for passage in corpus.passages.values():
        held = passage.story
        if (
            held is not None
            and text_hash(held.text) == hash
            and (held.valid_from, held.valid_to) == (story.valid_from, story.valid_to)
        ):
            return held
    return None


def _put_source(corpus: _Corpus, write: SourceWrite, model: str | None) -> None:
    source, chunks, vectors = write
    corpus.sources[source.source_id] = source
    for index, written in enumerate(chunks):
        chunk = written.model_copy(
            update={"source_name": source.name, "metadata": dict(source.metadata)}
        )
        corpus.passages[("chunk", chunk.chunk_id)] = _Passage(
            "chunk",
            chunk.chunk_id,
            chunk.text,
            dict(source.metadata),
            vectors[index] if vectors is not None else None,
            model,
            chunk=chunk,
        )


def _retire(corpus: _Corpus, source: Source, at: datetime, replaced_by: str | None) -> None:
    corpus.sources[source.source_id] = source.model_copy(
        update={"retired_at": at, "replaced_by": replaced_by}
    )
    for passage in corpus.passages.values():
        if passage.chunk is not None and passage.chunk.source_id == source.source_id:
            passage.chunk = passage.chunk.model_copy(update={"retired_at": at})


def _versions(corpus: _Corpus, source_id: str) -> list[str]:
    if source_id not in corpus.sources:
        return []
    chain = [source_id]
    for later in chain:
        chain.extend(
            s.source_id
            for s in corpus.sources.values()
            if s.replaced_by == later and s.source_id not in chain
        )
    return chain


def _delete_source(corpus: _Corpus, source_id: str, touched: set[str]) -> Counts:
    refs = {
        ref
        for ref, passage in corpus.passages.items()
        if passage.chunk is not None and passage.chunk.source_id == source_id
    }
    for ref in refs:
        del corpus.passages[ref]
    removed = corpus.sources.pop(source_id, None)
    return _drop_evidence(
        corpus, Counts(sources=int(removed is not None), chunks=len(refs)), refs, touched
    )


def _drop_evidence(corpus: _Corpus, deleted: Counts, refs: set[Ref], touched: set[str]) -> Counts:
    before_relations = len(corpus.relations)
    touched.update(
        e
        for r in corpus.relations
        if (r.evidence_kind, r.evidence_id) in refs
        for e in (r.from_id, r.to_id)
    )
    corpus.relations = [r for r in corpus.relations if (r.evidence_kind, r.evidence_id) not in refs]
    before_mentions = len(corpus.mentions)
    touched.update(m[0] for m in corpus.mentions if (m[1], m[2]) in refs)
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
    named = graph.named({})
    for entity in graph.known:
        if entity.entity_id in named and entity.entity_id not in corpus.entities:
            ids[entity.entity_id] = by_key.setdefault((entity.key, entity.type), entity.entity_id)
    missing = graph.named(ids) - set(corpus.entities) - set(ids.values())
    if missing:
        raise StorageError(f"inmemory: graph facts name unknown entities: {sorted(missing)}")
    return ids


def _apply_graph(corpus: _Corpus, graph: GraphWrite, ids: dict[str, str]) -> set[str]:
    for entity in graph.entities:
        target = ids[entity.entity_id]
        if target not in corpus.entities:
            corpus.entities[target] = entity
        else:
            _alias(corpus, target, entity.aliases)
    recreated: set[str] = set()
    for entity in graph.known:
        if entity.entity_id not in ids:
            continue
        target = ids[entity.entity_id]
        if target not in corpus.entities:
            corpus.entities[target] = entity.model_copy(update={"entity_id": target})
        recreated.add(target)
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
    for entity_id, kind, passage_id, name in graph.mentions:
        target = ids.get(entity_id, entity_id)
        corpus.mentions.add((target, kind, passage_id, name))
        entity = corpus.entities[target]
        if name_key(name) not in {entity.key, *map(name_key, entity.aliases)}:
            _alias(corpus, target, (name,))
    return recreated


def _reconcile(corpus: _Corpus, touched: set[str]) -> int:
    pruned = 0
    for entity_id in sorted(touched & set(corpus.entities)):
        entity = corpus.entities[entity_id]
        names = [m[3] for m in corpus.mentions if m[0] == entity_id]
        related = any(entity_id in (r.from_id, r.to_id) for r in corpus.relations)
        taken = {
            e.key
            for e in corpus.entities.values()
            if e.type == entity.type and e.entity_id != entity_id
        }
        kept = reconciled(entity, names, related, taken)
        if kept is None:
            del corpus.entities[entity_id]
            pruned += 1
        else:
            corpus.entities[entity_id] = kept
    return pruned


def _alias(corpus: _Corpus, entity_id: str, aliases: tuple[str, ...]) -> None:
    entity = corpus.entities[entity_id]
    added = tuple(a for a in dict.fromkeys(aliases) if a not in entity.aliases)
    if added:
        corpus.entities[entity_id] = entity.model_copy(
            update={"aliases": (*entity.aliases, *added)}
        )
