import json
from collections.abc import Iterable
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
    Closing,
    known,
    known_chunk,
    known_source,
)

Key = tuple[str, str, str]
Closings = dict[Key, Closing]

_ARCHIVED = {
    "story": ("story", "story_id"),
    "source": ("source", "source_id"),
    "chunk": ("source", "source_id"),
}


_CLOSINGS_SQL = (
    "SELECT DISTINCT ON (h.corpus_id, h.kind, h.id) h.corpus_id, h.kind, h.id, h.closed_at, h.valid_to, "
    "h.successor, h.closed_by, (SELECT max(p.revised_at) FROM memory_closings p WHERE p.corpus_id = h.corpus_id "
    "AND p.kind = h.kind AND p.id = h.id AND p.revised_at <= $4) AS revised FROM memory_closings h "
    "JOIN unnest($1::text[], $2::text[], $3::text[]) AS w(corpus_id, kind, id) "
    "ON h.corpus_id = w.corpus_id AND h.kind = w.kind AND h.id = w.id WHERE h.revised_at > $4 "
    "ORDER BY h.corpus_id, h.kind, h.id, h.revised_at"
)


def _revised(r: Any, row: str, known_at: datetime | None) -> Key | None:
    if known_at is None or r["revised_at"] is None or r["revised_at"] <= known_at:
        return None
    kind, column = _ARCHIVED[row]
    return (r["corpus_id"], kind, r[column])


async def _closings(
    conn: Any, rows: Iterable[tuple[str, Iterable[Any]]], known_at: datetime | None
) -> Closings:
    keys = sorted(
        {k for row, records in rows for r in records if (k := _revised(r, row, known_at))}
    )
    if not keys:
        return {}
    records = await conn.fetch(
        _CLOSINGS_SQL,
        [k[0] for k in keys],
        [k[1] for k in keys],
        [k[2] for k in keys],
        known_at,
    )
    return {
        (r["corpus_id"], r["kind"], r["id"]): Closing(
            r["closed_at"], r["valid_to"], r["successor"], _json(r["closed_by"]), r["revised"]
        )
        for r in records
    }


def _closing(
    r: Any, row: str, known_at: datetime | None, closings: Closings | None
) -> Closing | None:
    key = _revised(r, row, known_at)
    return None if key is None or closings is None else closings.get(key)


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _vector(vector: list[float] | None) -> str | None:
    return None if vector is None else "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _source(r: Any, known_at: datetime | None = None, closings: Closings | None = None) -> Source:
    return known_source(
        Source(
            corpus_id=r["corpus_id"],
            source_id=r["source_id"],
            name=r["name"],
            metadata=_json(r["metadata"]),
            text=r["text"],
            created_at=r["created_at"],
            confirmed_at=r["confirmed_at"],
            confirmed_by=_json(r["confirmed_by"]),
            chunks=r["chunks"],
            digest=r["digest"],
            retired_at=r["retired_at"],
            revised_at=r["revised_at"],
            replaced_by=r["replaced_by"],
            valid_from=r["valid_from"],
            valid_to=r["valid_to"],
            written_by=_json(r["written_by"]),
            closed_by=_json(r["closed_by"]),
        ),
        r["written_to"],
        known_at,
        _closing(r, "source", known_at, closings),
    )


def _chunk(r: Any, known_at: datetime | None = None, closings: Closings | None = None) -> Chunk:
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
            revised_at=r["revised_at"],
            valid_from=r["valid_from"],
            valid_to=r["valid_to"],
            written_by=_json(r["written_by"]),
            closed_by=_json(r["closed_by"]),
            extracted_by=_json(r["extracted_by"]),
        ),
        r["written_to"],
        known_at,
        _closing(r, "chunk", known_at, closings),
    )


def _story(r: Any, known_at: datetime | None = None, closings: Closings | None = None) -> Story:
    return known(
        Story(
            corpus_id=r["corpus_id"],
            story_id=r["story_id"],
            text=r["text"],
            subject=r["subject"],
            valid_from=r["valid_from"],
            valid_to=r["valid_to"],
            created_at=r["created_at"],
            confirmed_at=r["confirmed_at"] or r["created_at"],
            supersedes=tuple(_json(r["supersedes"])),
            superseded_by=r["superseded_by"],
            closed_at=r["closed_at"],
            revised_at=r["revised_at"],
            source_ids=tuple(_json(r["source_ids"])),
            metadata=_json(r["metadata"]),
            written_by=_json(r["written_by"]),
            confirmed_by=_json(r["confirmed_by"]),
            closed_by=_json(r["closed_by"]),
            extracted_by=_json(r["extracted_by"]),
        ),
        r["written_to"],
        known_at,
        _closing(r, "story", known_at, closings),
    )


def _entity(r: Any) -> Entity:
    return Entity(
        corpus_id=r["corpus_id"],
        entity_id=r["entity_id"],
        name=r["name"],
        key=r["key"],
        type=r["type"],
        aliases=tuple(_json(r["aliases"])),
        aliased_by=_json(r["aliased_by"]),
        resolved_by=_json(r["resolved_by"]),
    )


def _relation(r: Any) -> Relation:
    return Relation(
        corpus_id=r["corpus_id"],
        from_id=r["from_id"],
        to_id=r["to_id"],
        type=r["type"],
        evidence_kind=r["evidence_kind"],
        evidence_id=r["evidence_id"],
        extracted=r["extracted"],
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
