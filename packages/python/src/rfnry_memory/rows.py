import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from rfnry_memory.errors import StorageError
from rfnry_memory.models import (
    Chunk,
    Counts,
    Entity,
    Kind,
    Ref,
    Relation,
    Source,
    Story,
)


def text_hash(text: str) -> str:
    return sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def name_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def alias_keys(aliases: Iterable[str]) -> list[str]:
    return [key for key in dict.fromkeys(map(name_key, aliases)) if key]


def _edge(text: str, at: int) -> bool:
    return at < 0 or at >= len(text) or not (text[at].isalnum() or text[at] in "_-")


def whole(key: str, text: str) -> bool:
    start = text.find(key) if key else -1
    while start >= 0:
        if _edge(text, start - 1) and _edge(text, start + len(key)):
            return True
        start = text.find(key, start + 1)
    return False


def reconciled(entity: Entity, names: list[str], related: bool, taken: set[str]) -> Entity | None:
    if not names:
        return entity if related else None
    held: dict[str, str] = {}
    for name in sorted(names):
        held.setdefault(name_key(name), name)
    name, key = entity.name, entity.key
    free = sorted(set(held) - taken)
    if key not in held and free:
        name, key = held[free[0]], free[0]
    aliases = tuple(a for a in entity.aliases if name_key(a) in held and name_key(a) != key)
    aliased_by = {a: by for a, by in entity.aliased_by.items() if a in aliases}
    return entity.model_copy(
        update={"name": name, "key": key, "aliases": aliases, "aliased_by": aliased_by}
    )


def chunk_digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def total(counts: list[Counts]) -> Counts:
    return Counts(**{name: sum(getattr(c, name) for c in counts) for name in Counts.model_fields})


def successors(writes: list[SourceWrite]) -> dict[str, str]:
    names = Counter(write.source.name for write in writes)
    return {w.source.name: w.source.source_id for w in writes if names[w.source.name] == 1}


class Closing(NamedTuple):
    closed_at: datetime
    valid_to: datetime
    successor: str | None
    closed_by: dict[str, Any]
    revised: datetime | None


def after(moment: datetime, held: Iterable[datetime | None]) -> datetime:
    return max([moment, *(m + timedelta(microseconds=1) for m in held if m is not None)])


def _held(
    closed: datetime | None, closing: Closing | None, known_at: datetime
) -> tuple[bool, Closing | None]:
    if closing is not None:
        return closing.closed_at <= known_at, closing
    return closed is not None and closed <= known_at, None


def known(
    story: Story,
    written_to: datetime | None,
    known_at: datetime | None,
    closing: Closing | None = None,
) -> Story:
    if known_at is None:
        return story
    held, archived = _held(story.closed_at, closing, known_at)
    if archived is None:
        if held or story.closed_at is None:
            return story
        return story.model_copy(
            update={
                "valid_to": written_to,
                "superseded_by": None,
                "closed_at": None,
                "closed_by": None,
            }
        )
    return story.model_copy(
        update={
            "valid_to": archived.valid_to if held else written_to,
            "superseded_by": archived.successor if held else None,
            "closed_at": archived.closed_at if held else None,
            "closed_by": archived.closed_by if held else None,
            "revised_at": archived.revised,
        }
    )


def known_source(
    source: Source,
    written_to: datetime | None,
    known_at: datetime | None,
    closing: Closing | None = None,
) -> Source:
    if known_at is None:
        return source
    held, archived = _held(source.retired_at, closing, known_at)
    if archived is None:
        if held or source.retired_at is None:
            return source
        return source.model_copy(
            update={
                "retired_at": None,
                "replaced_by": None,
                "valid_to": written_to,
                "closed_by": None,
            }
        )
    return source.model_copy(
        update={
            "retired_at": archived.closed_at if held else None,
            "replaced_by": archived.successor if held else None,
            "valid_to": archived.valid_to if held else written_to,
            "closed_by": archived.closed_by if held else None,
            "revised_at": archived.revised,
        }
    )


def known_chunk(
    chunk: Chunk,
    written_to: datetime | None,
    known_at: datetime | None,
    closing: Closing | None = None,
) -> Chunk:
    if known_at is None:
        return chunk
    held, archived = _held(chunk.retired_at, closing, known_at)
    if archived is None:
        if held or chunk.retired_at is None:
            return chunk
        return chunk.model_copy(update={"retired_at": None, "valid_to": written_to})
    return chunk.model_copy(
        update={
            "retired_at": archived.closed_at if held else None,
            "valid_to": archived.valid_to if held else written_to,
            "revised_at": archived.revised,
        }
    )


def ended_on_write(story: Story) -> bool:
    return story.valid_to is not None and story.valid_to <= story.created_at


class RetrievalRow(BaseModel):
    retrieval_id: str
    corpus_ids: list[str]
    verb: str
    at: datetime
    recorded: datetime
    duration_ms: int
    body: dict[str, Any]


class RetrievalLabelRow(BaseModel):
    retrieval_id: str
    label_id: str
    source: str
    at: datetime
    body: dict[str, Any]


class SourceWrite(NamedTuple):
    source: Source
    chunks: list[Chunk]
    vectors: list[list[float]] | None


class GraphWrite(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    aliases: list[tuple[str, str]] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    mentions: list[tuple[str, Kind, str, str]] = Field(default_factory=list)
    known: list[Entity] = Field(default_factory=list)

    def check(self, corpus_id: str, passages: set[Ref]) -> None:
        if any(e.corpus_id != corpus_id for e in [*self.entities, *self.known]) or any(
            r.corpus_id != corpus_id for r in self.relations
        ):
            raise StorageError("rows: row corpus does not match the call corpus")
        evidence = {(r.evidence_kind, r.evidence_id) for r in self.relations} | {
            (m[1], m[2]) for m in self.mentions
        }
        if not evidence <= passages:
            raise StorageError("rows: graph facts must be evidenced by a passage of the same write")

    def named(self, ids: dict[str, str]) -> set[str]:
        named = {e for e, _ in self.aliases} | {m[0] for m in self.mentions}
        named |= {r.from_id for r in self.relations} | {r.to_id for r in self.relations}
        return {ids.get(e, e) for e in named}
