from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from rfnry_memory.models import (
    Chunk,
    Counts,
    Entity,
    Kind,
    Language,
    Leg,
    Ref,
    Relation,
    Scope,
    Source,
    Story,
    VectorIndex,
    Window,
)
from rfnry_memory.rows import GraphWrite, SourceWrite


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
        self,
        corpus_id: str,
        *,
        filters: dict[str, Any] | None,
        limit: int,
        cursor: str | None,
        known_at: datetime | None = None,
        history: bool = False,
    ) -> tuple[list[Source], str | None]: ...

    async def source_chunks(self, corpus_id: str, source_id: str) -> list[Chunk]: ...

    async def source_digests(
        self, corpus_id: str
    ) -> list[tuple[str, dict[str, Any], str, int]]: ...

    async def delete_source(self, corpus_id: str, source_id: str) -> Counts: ...

    async def forget_retired(self, corpus_id: str, *, before: datetime) -> Counts: ...

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
    ) -> tuple[Counts, list[str]] | None: ...

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

    async def find_open_story(self, corpus_id: str, hash: str, *, at: datetime) -> Story | None: ...

    async def supersede_story(
        self,
        corpus_id: str,
        old_ids: list[str],
        new: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> tuple[list[Story], Story]: ...

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
        corpus_ids: list[str],
        query: str,
        *,
        kinds: tuple[Kind, ...],
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        scope: Scope | None = None,
    ) -> list[tuple[str, Leg]]: ...

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
        scope: Scope | None = None,
    ) -> list[Leg]: ...

    async def get_passages(
        self, corpus_id: str, refs: list[Ref], *, known_at: datetime | None = None
    ) -> list[Chunk | Story]: ...

    async def similar_entities(self, corpus_id: str, name: str, *, limit: int) -> list[Entity]: ...

    async def entities_named(self, corpus_id: str, name_or_key: str) -> list[Entity]: ...

    async def entities_by_id(self, corpus_id: str, entity_ids: list[str]) -> list[Entity]: ...

    async def entities_named_in(self, corpus_id: str, text: str) -> list[Entity]: ...

    async def neighbors(
        self,
        corpus_id: str,
        entity_ids: list[str],
        *,
        relation_types: list[str] | None,
        window: Window,
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
        scope: Scope | None = None,
    ) -> list[tuple[str, Ref]]: ...

    async def passage_vector(self, corpus_id: str, passage: Ref) -> list[float] | None: ...

    async def chunks_around(
        self,
        corpus_id: str,
        source_id: str,
        position: int,
        *,
        width: int,
        window: Window,
        filters: dict[str, Any] | None,
    ) -> list[Chunk]: ...

    async def passage_entities(
        self, corpus_id: str, passages: list[Ref]
    ) -> list[tuple[Ref, Entity]]: ...

    async def graph_types(self, corpus_id: str) -> tuple[dict[str, int], dict[str, int]]: ...

    async def passages_embedded_with(
        self, corpus_id: str, *, not_model: str, limit: int
    ) -> list[Chunk | Story]: ...

    async def set_vectors(
        self, corpus_id: str, refs: list[Ref], vectors: list[list[float]], model: str
    ) -> None: ...

    async def delete_corpus(self, corpus_id: str) -> Counts: ...

    async def count_corpus(self, corpus_id: str) -> Counts: ...

    async def get_language(self, corpus_id: str) -> Language: ...

    async def set_language(self, corpus_id: str, language: Language) -> int | None: ...

    async def get_history(self, corpus_id: str) -> bool: ...

    async def set_history(self, corpus_id: str, keep: bool) -> None: ...

    async def count_vectors(self, corpus_id: str) -> dict[str | None, int]: ...

    async def get_vector_index(self, corpus_id: str) -> VectorIndex | None: ...

    async def set_vector_index(
        self,
        corpus_id: str,
        on: bool,
        *,
        m: int,
        ef_construction: int,
        precision: Literal["full", "half"] | None,
    ) -> int: ...
