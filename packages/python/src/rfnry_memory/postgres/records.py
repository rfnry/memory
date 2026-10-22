import json
from datetime import datetime
from typing import Any

from rfnry_memory.errors import InputError
from rfnry_memory.models import (
    Chunk,
    Entity,
    Relation,
    Source,
    Story,
)
from rfnry_memory.rows import (
    known,
    known_chunk,
    known_source,
)


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _vector(vector: list[float] | None) -> str | None:
    return None if vector is None else "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _source(r: Any, known_at: datetime | None = None) -> Source:
    return known_source(
        Source(
            corpus_id=r["corpus_id"],
            source_id=r["source_id"],
            name=r["name"],
            metadata=_json(r["metadata"]),
            text=r["text"],
            created_at=r["created_at"],
            confirmed_at=r["confirmed_at"],
            chunks=r["chunks"],
            digest=r["digest"],
            retired_at=r["retired_at"],
            replaced_by=r["replaced_by"],
            valid_from=r["valid_from"],
            valid_to=r["valid_to"],
            written_by=_json(r["written_by"]),
            closed_by=_json(r["closed_by"]),
        ),
        r["written_to"],
        known_at,
    )


def _chunk(r: Any, known_at: datetime | None = None) -> Chunk:
    return known_chunk(
        Chunk(
            corpus_id=r["corpus_id"],
            chunk_id=r["chunk_id"],
            source_id=r["source_id"],
            source_name=r["source_name"],
            position=r["position"],
            title=r["title"],
            locator=_json(r["locator"]),
            text=r["text"],
            metadata=_json(r["metadata"]),
            created_at=r["created_at"],
            confirmed_at=r["confirmed_at"],
            context=r["context"],
            retired_at=r["retired_at"],
            valid_from=r["valid_from"],
            valid_to=r["valid_to"],
            written_by=_json(r["written_by"]),
        ),
        r["written_to"],
        known_at,
    )


def _story(r: Any, known_at: datetime | None = None) -> Story:
    return known(
        Story(
            corpus_id=r["corpus_id"],
            story_id=r["story_id"],
            text=r["text"],
            subject=r["subject"],
            valid_from=r["valid_from"],
            valid_to=r["valid_to"],
            created_at=r["created_at"],
            supersedes=tuple(_json(r["supersedes"])),
            superseded_by=r["superseded_by"],
            closed_at=r["closed_at"],
            source_ids=tuple(_json(r["source_ids"])),
            metadata=_json(r["metadata"]),
            written_by=_json(r["written_by"]),
            closed_by=_json(r["closed_by"]),
        ),
        r["written_to"],
        known_at,
    )


def _entity(r: Any) -> Entity:
    return Entity(
        corpus_id=r["corpus_id"],
        entity_id=r["entity_id"],
        name=r["name"],
        key=r["key"],
        type=r["type"],
        aliases=tuple(_json(r["aliases"])),
    )


def _relation(r: Any) -> Relation:
    return Relation(
        corpus_id=r["corpus_id"],
        from_id=r["from_id"],
        to_id=r["to_id"],
        type=r["type"],
        evidence_kind=r["evidence_kind"],
        evidence_id=r["evidence_id"],
    )


async def _taken(conn: Any, corpus_id: str, kind: str, ids: list[str]) -> None:
    seen: set[str] = set()
    for made in ids:
        if made in seen:
            raise InputError(f"postgres: {kind} {made!r} is given twice; ids must be unique")
        seen.add(made)
    table, column = (
        ("memory_stories", "story_id") if kind == "story" else ("memory_sources", "source_id")
    )
    held = await conn.fetchval(
        f"SELECT {column} FROM {table} WHERE corpus_id = $1 AND {column} = ANY($2::text[]) LIMIT 1",
        corpus_id,
        ids,
    )
    if held is not None:
        raise InputError(f"postgres: {kind} {held!r} is already held; ids must be unique")


def _count(status: str) -> int:
    return int(status.rsplit(" ", 1)[-1])
