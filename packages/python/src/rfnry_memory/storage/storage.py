from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any, NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from rfnry_memory.errors import StorageError
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


def text_hash(text: str) -> str:
    return sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def chunk_digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def total(counts: list[Counts]) -> Counts:
    return Counts(**{name: sum(getattr(c, name) for c in counts) for name in Counts.model_fields})


class SourceWrite(NamedTuple):
    source: Source
    chunks: list[Chunk]
    vectors: list[list[float]] | None


class GraphWrite(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    aliases: list[tuple[str, str]] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    mentions: list[tuple[str, Kind, str]] = Field(default_factory=list)

    def check(self, corpus_id: str, passages: set[Ref]) -> None:
        if any(e.corpus_id != corpus_id for e in self.entities) or any(
            r.corpus_id != corpus_id for r in self.relations
        ):
            raise StorageError("storage: row corpus does not match the call corpus")
        evidence = {(r.evidence_kind, r.evidence_id) for r in self.relations} | {
            (kind, pid) for _, kind, pid in self.mentions
        }
        if not evidence <= passages:
            raise StorageError(
                "storage: graph facts must be evidenced by a passage of the same write"
            )

    def named(self, ids: dict[str, str]) -> set[str]:
        named = {e for e, _ in self.aliases} | {e for e, _, _ in self.mentions}
        named |= {r.from_id for r in self.relations} | {r.to_id for r in self.relations}
        return {ids.get(e, e) for e in named}


@runtime_checkable
class Storage(Protocol):
    async def initialize(self, *, dimension: int | None) -> None: ...

    async def put_source(
        self,
        corpus_id: str,
        source: Source,
        chunks: list[Chunk],
        vectors: list[list[float]] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> None: ...

    async def get_source(self, corpus_id: str, source_id: str) -> Source | None: ...

    async def list_sources(
        self, corpus_id: str, *, filters: dict[str, Any] | None, limit: int, cursor: str | None
    ) -> tuple[list[Source], str | None]: ...

    async def source_chunks(self, corpus_id: str, source_id: str) -> list[Chunk]: ...

    async def source_digests(self, corpus_id: str) -> list[tuple[str, dict[str, Any], str]]: ...

    async def delete_source(self, corpus_id: str, source_id: str) -> Counts: ...

    async def sync_sources(
        self,
        corpus_id: str,
        filters: dict[str, Any],
        writes: list[SourceWrite],
        model: str | None,
        *,
        graph: GraphWrite | None = None,
        keep: dict[str, str] | None = None,
    ) -> Counts | None: ...

    async def chunk_vectors(
        self, corpus_id: str, model: str, digests: list[str]
    ) -> dict[str, list[float]]: ...

    async def put_story(
        self,
        corpus_id: str,
        story: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> Story: ...

    async def find_open_story(self, corpus_id: str, hash: str) -> Story | None: ...

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
    ) -> tuple[Story, Story]: ...

    async def get_stories(self, corpus_id: str, story_ids: list[str]) -> list[Story]: ...

    async def list_stories(
        self,
        corpus_id: str,
        *,
        subject: str | None,
        window: Window,
        filters: dict[str, Any] | None,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[Story], str | None]: ...

    async def delete_story(self, corpus_id: str, story_id: str) -> Counts: ...

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
    ) -> list[Leg]: ...

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
    ) -> list[Leg]: ...

    async def get_passages(self, corpus_id: str, refs: list[Ref]) -> list[Chunk | Story]: ...

    async def similar_entities(self, corpus_id: str, name: str, *, limit: int) -> list[Entity]: ...

    async def entities_named(self, corpus_id: str, name_or_key: str) -> list[Entity]: ...

    async def entities_by_id(self, corpus_id: str, entity_ids: list[str]) -> list[Entity]: ...

    async def entities_named_in(self, corpus_id: str, text: str) -> list[Entity]: ...

    async def neighbors(
        self, corpus_id: str, entity_ids: list[str], *, relation_types: list[str] | None
    ) -> list[Relation]: ...

    async def mentions(
        self,
        corpus_id: str,
        entity_ids: list[str],
        *,
        kinds: tuple[Kind, ...],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> list[Ref]: ...

    async def passages_embedded_with(
        self, corpus_id: str, *, not_model: str, limit: int
    ) -> list[Chunk | Story]: ...

    async def set_vectors(
        self, corpus_id: str, refs: list[Ref], vectors: list[list[float]], model: str
    ) -> None: ...

    async def delete_corpus(self, corpus_id: str) -> Counts: ...

    async def count_corpus(self, corpus_id: str) -> Counts: ...

    async def get_language(self, corpus_id: str) -> Language: ...

    async def set_language(self, corpus_id: str, language: Language) -> int: ...

    async def count_vectors(self, corpus_id: str) -> dict[str | None, int]: ...
