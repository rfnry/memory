import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from rfnry_memory.errors import InputError, NotFoundError, StorageError
from rfnry_memory.models import (
    Counts,
    Story,
    Window,
)
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.graph import _drop_evidence, _reconcile, _write_graph
from rfnry_memory.postgres.keys import keyed, refuse_unfit
from rfnry_memory.postgres.language import _CONFIG, _pin
from rfnry_memory.postgres.locks import _closings_lock, _language_lock
from rfnry_memory.postgres.records import (
    _closings,
    _count,
    _json,
    _prior,
    _story,
    _taken,
    _vector,
)
from rfnry_memory.postgres.sql import _facet_sql, _subject_sql, _window_sql
from rfnry_memory.postgres.vectors import _copy_vectors, _delete_vectors, _vectors_table
from rfnry_memory.rows import (
    GraphWrite,
    after,
    ended_on_write,
    text_hash,
)


def _retracted(story: Story) -> bool:
    return (
        story.closed_at is not None
        and story.superseded_by is None
        and story.valid_to == story.valid_from
    )


def _closable(story: Story, new: Story) -> None:
    if story.superseded_by is not None:
        raise InputError(
            f"postgres: story {story.story_id!r} is already superseded; supersede the story that replaced it"
        )
    if story.closed_at is not None and _retracted(story):
        raise InputError(
            f"postgres: story {story.story_id!r} was retracted at {story.closed_at.isoformat()}; remember a new story instead"
        )
    if story.valid_to is not None and story.valid_to <= new.valid_from:
        raise InputError(
            f"postgres: story {story.story_id!r} ends at {story.valid_to.isoformat()}, before valid_from; pass an earlier valid_from"
        )
    if new.valid_from < story.valid_from:
        raise InputError(
            f"postgres: valid_from must not be before story {story.story_id!r} became true, at {story.valid_from.isoformat()}"
        )


def _endable(story: Story, end: datetime) -> None:
    if story.superseded_by is not None and end != story.valid_from:
        raise InputError(
            f"postgres: story {story.story_id!r} is already superseded; end the story that replaced it"
        )
    if story.closed_at is not None and _retracted(story):
        raise InputError(
            f"postgres: story {story.story_id!r} was retracted at {story.closed_at.isoformat()}"
        )
    if end < story.valid_from:
        raise InputError(
            f"postgres: valid_to must not be before story {story.story_id!r} became true, at {story.valid_from.isoformat()}"
        )


_SAME = "subject IS NOT DISTINCT FROM $4 AND metadata::jsonb = $5::jsonb"


_OPEN_SQL = (
    "SELECT * FROM memory_stories WHERE corpus_id = $1 AND text_hash = $2 AND superseded_by IS NULL "
    "AND (valid_to IS NULL OR (valid_to > $3 AND valid_to > valid_from)) "
    f'AND {_SAME} AND story_id <> ALL($6::text[]) ORDER BY story_id COLLATE "C" LIMIT 1'
)


_TWINS_SQL = (
    "SELECT story_id, valid_from FROM memory_stories WHERE corpus_id = $1 AND text_hash = $2 "
    "AND superseded_by IS NULL AND (valid_to IS NULL OR (valid_to > $3 AND valid_to > valid_from)) "
    f"AND {_SAME} AND story_id <> $6 AND ($7::timestamptz IS NULL OR valid_from < $7) "
    'AND (valid_to IS NULL OR valid_to > $8) ORDER BY valid_from, story_id COLLATE "C" LIMIT 1'
)


_CONFIRM_SQL = (
    "UPDATE memory_stories SET confirmed_at = $3, confirmed_by = $4::jsonb WHERE corpus_id = $1 "
    "AND story_id = $2 AND coalesce(confirmed_at, created_at) < $3 RETURNING *"
)


_ARCHIVE_SQL = (
    "INSERT INTO memory_closings (corpus_id, kind, id, closed_at, valid_to, successor, closed_by, revised_at) "
    "SELECT corpus_id, 'story', story_id, closed_at, valid_to, superseded_by, coalesce(closed_by, '{}'), $3::timestamptz "
    "FROM memory_stories WHERE corpus_id = $1 AND story_id = ANY($2::text[]) AND closed_at IS NOT NULL"
)


_CLOSE_SQL = (
    "UPDATE memory_stories s SET valid_to = coalesce(c.ends, s.written_to), superseded_by = c.successor, "
    "closed_at = CASE WHEN c.ends IS NULL THEN NULL ELSE $3::timestamptz END, "
    "closed_by = CASE WHEN c.ends IS NULL THEN NULL ELSE $4::jsonb END, "
    "revised_at = CASE WHEN s.closed_at IS NULL THEN s.revised_at ELSE $3::timestamptz END "
    "FROM unnest($2::text[], $5::timestamptz[], $6::text[]) AS c(story_id, ends, successor) "
    "WHERE s.corpus_id = $1 AND s.story_id = c.story_id RETURNING s.*"
)


Change = tuple[Any, datetime | None, str | None]


def _open(story: Story, end: datetime | None, at: datetime) -> bool:
    return story.superseded_by is None and (end is None or (end > at and end > story.valid_from))


def _in_force(end: datetime | None, start: datetime, at: datetime) -> bool:
    return end is None or (end > at and end > start)


async def _twin(conn: Any, story: Story, end: datetime | None, at: datetime) -> Any:
    return await conn.fetchrow(
        _TWINS_SQL,
        story.corpus_id,
        text_hash(story.text),
        at,
        story.subject,
        json.dumps(story.metadata),
        story.story_id,
        end,
        story.valid_from,
    )


async def _beside(conn: Any, story: Story, end: datetime, at: datetime) -> None:
    if not _open(story, end, at):
        return
    twin = await _twin(conn, story, end, at)
    if twin is None:
        return
    since = twin["valid_from"]
    fix = "end it no later than that" if since > story.valid_from else "give an end that has come"
    raise InputError(
        f"postgres: an end at {end.isoformat()} would keep story {story.story_id!r} true "
        f"beside story {twin['story_id']!r}, which says the same from {since.isoformat()}; {fix}"
    )


async def _untwinned(
    conn: Any, retracted: str, record: Any, end: datetime | None, at: datetime
) -> datetime | None:
    story = _story(record)
    reach = record["written_to"] if end is None else end
    if not _in_force(reach, story.valid_from, at):
        return end
    twin = await _twin(conn, story, reach, at)
    if twin is None:
        return end
    since: datetime = twin["valid_from"]
    if since <= story.valid_from:
        raise InputError(
            f"postgres: retracting story {retracted!r} would reopen story {story.story_id!r} beside story "
            f"{twin['story_id']!r}, which says the same from {since.isoformat()}; end or delete one of them first"
        )
    return since


async def _returned(
    conn: Any,
    corpus_id: str,
    retracted: str,
    record: Any,
    replacement: Story | None,
    at: datetime,
    by: dict[str, Any],
) -> tuple[Change, dict[str, Any]]:
    closed_at = record["closed_at"]
    while True:
        prior = await _prior(conn, corpus_id, "story", record["story_id"], closed_at)
        if prior is not None and prior["successor"] is not None:
            successor = await conn.fetchrow(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = $2",
                corpus_id,
                prior["successor"],
            )
            if successor is None or successor["valid_to"] != successor["valid_from"]:
                return (record, prior["valid_to"], prior["successor"]), _json(prior["closed_by"])
            replacement = (
                await _replacement(conn, corpus_id, successor["superseded_by"]) or replacement
            )
            closed_at = prior["closed_at"]
            continue
        cap = record["written_to"] if prior is None else prior["valid_to"]
        if replacement is not None and (cap is None or replacement.valid_from < cap):
            return (record, replacement.valid_from, replacement.story_id), by
        end = None if prior is None else prior["valid_to"]
        held = await _untwinned(conn, retracted, record, end, at)
        if prior is None or held != end:
            return (record, held, None), by
        return (record, held, None), _json(prior["closed_by"])


async def _fits(conn: Any, corpus_id: str, changes: list[Change], at: datetime) -> None:
    back = [
        r["story_id"]
        for r, end, _ in changes
        if r["closed_at"] is not None
        and not _in_force(r["valid_to"], r["valid_from"], at)
        and (end is None or _in_force(end, r["valid_from"], at))
    ]
    await refuse_unfit(conn, "memory_stories", "story_id", "story", corpus_id, back)


async def _close(
    conn: Any, corpus_id: str, changes: list[Change], closed_by: dict[str, Any], at: datetime
) -> list[Story]:
    ids = [r["story_id"] for r, _, _ in changes]
    if not ids:
        return []
    await _fits(conn, corpus_id, changes, at)
    await conn.execute("SELECT memory_count($1, 'story', $2::text[], -1)", corpus_id, ids)
    await conn.execute(_ARCHIVE_SQL, corpus_id, ids, at)
    updated = await conn.fetch(
        _CLOSE_SQL,
        corpus_id,
        ids,
        at,
        json.dumps(closed_by),
        [end for _, end, _ in changes],
        [successor for _, _, successor in changes],
    )
    await conn.execute("SELECT memory_count($1, 'story', $2::text[], 1)", corpus_id, ids)
    by_id = {r["story_id"]: _story(r) for r in updated}
    return [by_id[i] for i in ids]


async def _locked(conn: Any, corpus_id: str, ids: list[str]) -> dict[str, Any]:
    return {
        r["story_id"]: r
        for r in await conn.fetch(
            "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = ANY($2::text[]) "
            'ORDER BY story_id COLLATE "C" FOR UPDATE',
            corpus_id,
            ids,
        )
    }


async def _closed_by(conn: Any, corpus_id: str, ids: list[str]) -> list[Any]:
    if not ids:
        return []
    return list(
        await conn.fetch(
            "SELECT story_id, text_hash, superseded_by FROM memory_stories "
            'WHERE corpus_id = $1 AND superseded_by = ANY($2::text[]) ORDER BY story_id COLLATE "C"',
            corpus_id,
            ids,
        )
    )


async def _replacement(conn: Any, corpus_id: str, story_id: str | None) -> Story | None:
    seen: set[str] = set()
    while story_id is not None and story_id not in seen:
        seen.add(story_id)
        record = await conn.fetchrow(
            "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = $2",
            corpus_id,
            story_id,
        )
        if record is None:
            return None
        story = _story(record)
        if story.valid_to != story.valid_from:
            return story
        story_id = story.superseded_by
    return None


async def _text_lock(conn: Any, corpus_id: str, hashes: list[str]) -> None:
    keys = [f"memory.story:{corpus_id}:{hash}" for hash in sorted(set(hashes))]
    if keys:
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || ':' || k, 0)) "
            "FROM unnest($1::text[]) WITH ORDINALITY AS t(k, o) ORDER BY o",
            keys,
        )


class Stories(Base):
    def _insert_story_sql(self) -> str:
        columns = (
            "corpus_id, story_id, text, text_hash, subject, valid_from, valid_to, created_at, "
            "supersedes, superseded_by, closed_at, written_to, source_ids, metadata, config"
        )
        values = (
            "$1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11, $12, $13::jsonb, $14::json, "
            + _CONFIG
        )
        if self._dimension:
            return (
                f"INSERT INTO memory_stories ({columns}, embedding, embedding_model, written_by, "
                f"closed_by, extracted_by, confirmed_at, confirmed_by, embedding_settings) VALUES "
                f"({values}, $15::vector, $16, $17::jsonb, $18::jsonb, $19::jsonb, $20, $21::jsonb, "
                "$22::jsonb)"
            )
        return (
            f"INSERT INTO memory_stories ({columns}, embedding_model, written_by, closed_by, "
            f"extracted_by, confirmed_at, confirmed_by) VALUES ({values}, $15, $16::jsonb, $17::jsonb, "
            "$18::jsonb, $19, $20::jsonb)"
        )

    def _story_params(
        self,
        story: Story,
        vector: list[float] | None,
        model: str | None,
        settings: dict[str, Any] | None,
    ) -> list[Any]:
        return [
            story.corpus_id,
            story.story_id,
            story.text,
            text_hash(story.text),
            story.subject,
            story.valid_from,
            story.valid_to,
            story.created_at,
            json.dumps(list(story.supersedes)),
            story.superseded_by,
            story.closed_at,
            story.valid_to,
            json.dumps(list(story.source_ids)),
            json.dumps(story.metadata),
            *([_vector(vector)] if self._dimension else []),
            model,
            json.dumps(story.written_by),
            None if story.closed_by is None else json.dumps(story.closed_by),
            None if story.extracted_by is None else json.dumps(story.extracted_by),
            story.confirmed_at,
            None if story.confirmed_by is None else json.dumps(story.confirmed_by),
            *([None if settings is None else json.dumps(settings)] if self._dimension else []),
        ]

    async def _held(self, conn: Any, corpus_id: str, story: Story) -> Story | None:
        hash, same = text_hash(story.text), (story.subject, json.dumps(story.metadata))
        if ended_on_write(story):
            record = await conn.fetchrow(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND text_hash = $2 "
                f'AND {_SAME} AND valid_from = $3 AND valid_to = $6 ORDER BY story_id COLLATE "C" LIMIT 1',
                corpus_id,
                hash,
                story.valid_from,
                *same,
                story.valid_to,
            )
        else:
            record = await conn.fetchrow(_OPEN_SQL, corpus_id, hash, story.created_at, *same, [])
        return _story(record) if record else None

    async def put_story(
        self,
        corpus_id: str,
        story: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Story:
        if story.corpus_id != corpus_id:
            raise StorageError("postgres: row corpus does not match the call corpus")
        self._check_vectors(vector)
        if story.closed_at is not None and story.closed_by is None:
            story = story.model_copy(update={"closed_by": {}})
        graph = graph or GraphWrite()
        graph.check(corpus_id, {("story", story.story_id)})
        async with (
            self._guard("put_story"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=True)
            await _pin(conn, corpus_id)
            table = await _vectors_table(conn, corpus_id)
            await _text_lock(conn, corpus_id, [text_hash(story.text)])
            holder = await self._held(conn, corpus_id, story)
            if holder is not None and holder.story_id != story.story_id:
                confirmed = await conn.fetchrow(
                    _CONFIRM_SQL,
                    corpus_id,
                    holder.story_id,
                    story.created_at,
                    json.dumps(story.written_by),
                )
                return holder if confirmed is None else _story(confirmed)
            await _taken(conn, corpus_id, "story", [story.story_id])
            await conn.execute(
                self._insert_story_sql(), *self._story_params(story, vector, model, settings)
            )
            await conn.execute(
                "SELECT memory_count($1, 'story', $2::text[], 1)", corpus_id, [story.story_id]
            )
            await _copy_vectors(conn, table, corpus_id, "story", [story.story_id])
            recreated: set[str] = set()
            await _write_graph(conn, corpus_id, graph, recreated)
            await _reconcile(conn, corpus_id, recreated)
        return story

    async def confirm_story(
        self, corpus_id: str, story_id: str, *, at: datetime, confirmed_by: dict[str, Any]
    ) -> Story | None:
        async with self._guard("confirm_story"), self._pool.acquire() as conn:
            record = await conn.fetchrow(
                _CONFIRM_SQL, corpus_id, story_id, at, json.dumps(confirmed_by)
            ) or await conn.fetchrow(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = $2",
                corpus_id,
                story_id,
            )
        return _story(record) if record else None

    async def find_open_story(
        self,
        corpus_id: str,
        hash: str,
        *,
        subject: str | None,
        metadata: dict[str, Any],
        at: datetime,
        excluding: Sequence[str] = (),
    ) -> Story | None:
        async with self._guard("find_open_story"), self._pool.acquire() as conn:
            record = await conn.fetchrow(
                _OPEN_SQL, corpus_id, hash, at, subject, json.dumps(metadata), list(excluding)
            )
        return _story(record) if record else None

    async def supersede_story(
        self,
        corpus_id: str,
        old_ids: list[str],
        new: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
        settings: dict[str, Any] | None = None,
    ) -> tuple[list[Story], Story]:
        if new.corpus_id != corpus_id:
            raise StorageError("postgres: row corpus does not match the call corpus")
        if not old_ids or len(set(old_ids)) != len(old_ids):
            raise InputError("postgres: supersede_story needs distinct story ids")
        self._check_vectors(vector)
        graph = graph or GraphWrite()
        graph.check(corpus_id, {("story", new.story_id)})
        async with (
            self._guard("supersede_story"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=True)
            await _pin(conn, corpus_id)
            table = await _vectors_table(conn, corpus_id)
            seen = await conn.fetch(
                "SELECT story_id, text_hash, valid_from FROM memory_stories "
                "WHERE corpus_id = $1 AND story_id = ANY($2::text[])",
                corpus_id,
                list(old_ids),
            )
            corrected = [r["story_id"] for r in seen if r["valid_from"] == new.valid_from]
            if corrected:
                await _closings_lock(conn, corpus_id)
            handed = [
                r
                for r in await _closed_by(conn, corpus_id, corrected)
                if r["story_id"] not in old_ids
            ]
            await _text_lock(
                conn,
                corpus_id,
                [text_hash(new.text), *(r["text_hash"] for r in [*seen, *handed])],
            )
            rows = await _locked(conn, corpus_id, [*old_ids, *(r["story_id"] for r in handed)])
            for old_id in old_ids:
                if old_id not in rows:
                    raise NotFoundError(f"postgres: story {old_id!r} was not found")
                _closable(_story(rows[old_id]), new)
            holder = await conn.fetchrow(
                _OPEN_SQL,
                corpus_id,
                text_hash(new.text),
                new.created_at,
                new.subject,
                json.dumps(new.metadata),
                list(old_ids),
            )
            if holder is not None:
                raise InputError(
                    f"postgres: story {holder['story_id']!r}, true since {holder['valid_from'].isoformat()}, already says that; supersede or delete it instead"
                )
            await _taken(conn, corpus_id, "story", [new.story_id])
            at = after(
                new.created_at,
                (r[c] for r in rows.values() for c in ("created_at", "closed_at", "revised_at")),
            )
            new = new.model_copy(update={"created_at": at, "confirmed_at": at})
            closing: list[Change] = [(rows[i], new.valid_from, new.story_id) for i in old_ids]
            closed = await _close(conn, corpus_id, closing, new.written_by, at)
            taken: list[Change] = [
                (rows[r["story_id"]], new.valid_from, new.story_id)
                for r in handed
                if r["story_id"] in rows and rows[r["story_id"]]["superseded_by"] in corrected
            ]
            await _close(conn, corpus_id, taken, new.written_by, at)
            await conn.execute(
                self._insert_story_sql(), *self._story_params(new, vector, model, settings)
            )
            await conn.execute(
                "SELECT memory_count($1, 'story', $2::text[], 1)", corpus_id, [new.story_id]
            )
            await _copy_vectors(conn, table, corpus_id, "story", [new.story_id])
            recreated: set[str] = set()
            await _write_graph(conn, corpus_id, graph, recreated)
            await _reconcile(conn, corpus_id, recreated)
        return closed, new

    async def end_story(
        self,
        corpus_id: str,
        story_id: str,
        *,
        valid_to: datetime,
        closed_at: datetime,
        closed_by: dict[str, Any] | None = None,
    ) -> Story:
        by = closed_by or {}
        async with (
            self._guard("end_story"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=True)
            seen = await conn.fetchrow(
                "SELECT text_hash, valid_from FROM memory_stories WHERE corpus_id = $1 AND story_id = $2",
                corpus_id,
                story_id,
            )
            if seen is None:
                raise NotFoundError(f"postgres: story {story_id!r} was not found")
            retracting = valid_to == seen["valid_from"]
            if retracting:
                await _closings_lock(conn, corpus_id)
            handed = await _closed_by(conn, corpus_id, [story_id] if retracting else [])
            await _text_lock(
                conn, corpus_id, [seen["text_hash"], *(r["text_hash"] for r in handed)]
            )
            rows = await _locked(conn, corpus_id, [story_id, *(r["story_id"] for r in handed)])
            if story_id not in rows:
                raise NotFoundError(f"postgres: story {story_id!r} was not found")
            story = _story(rows[story_id])
            _endable(story, valid_to)
            at = after(
                closed_at,
                (r[c] for r in rows.values() for c in ("created_at", "closed_at", "revised_at")),
            )
            await _beside(conn, story, valid_to, at)
            [ended] = await _close(conn, corpus_id, [(rows[story_id], valid_to, None)], by, at)
            replacement = await _replacement(conn, corpus_id, story.superseded_by)
            closed = [
                rows[r["story_id"]]
                for r in handed
                if r["story_id"] in rows and rows[r["story_id"]]["superseded_by"] == story_id
            ]
            for record in sorted(
                closed, key=lambda r: (r["created_at"], r["story_id"]), reverse=True
            ):
                change, closer = await _returned(
                    conn, corpus_id, story_id, record, replacement, at, by
                )
                await _close(conn, corpus_id, [change], closer, at)
        return ended

    async def get_stories(self, corpus_id: str, story_ids: list[str]) -> list[Story]:
        async with self._guard("get_stories"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = ANY($2::text[])",
                corpus_id,
                list(story_ids),
            )
        by_id = {r["story_id"]: _story(r) for r in records}
        return [by_id[i] for i in story_ids if i in by_id]

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
        params: list[Any] = [corpus_id]
        async with self._guard("list_stories"), self._pool.acquire() as conn:
            filters = await keyed(conn, filters, window.as_of if window.now else None)
            sql = "SELECT * FROM memory_stories WHERE corpus_id = $1"
            sql += (
                _subject_sql(subject, params)
                + _window_sql(window, params)
                + _facet_sql(
                    filters, params, cast="::jsonb", kind="story", known_at=window.known_at
                )
            )
            offset = int(cursor) if cursor else 0
            params.extend([limit + 1, offset])
            sql += f' ORDER BY valid_from DESC, story_id COLLATE "C" DESC LIMIT ${len(params) - 1} OFFSET ${len(params)}'
            records = await conn.fetch(sql, *params)
            closings = await _closings(conn, [("story", records)], window.known_at)
        rows = [_story(r, window.known_at, closings) for r in records]
        return rows[:limit], (str(offset + limit) if len(rows) > limit else None)

    async def delete_story(self, corpus_id: str, story_id: str) -> Counts:
        async with (
            self._guard("delete_story"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=True)
            await conn.execute(
                "SELECT 1 FROM memory_stories WHERE corpus_id = $1 AND story_id = $2 FOR UPDATE",
                corpus_id,
                story_id,
            )
            await conn.execute(
                "SELECT memory_count($1, 'story', $2::text[], -1)", corpus_id, [story_id]
            )
            table = await _vectors_table(conn, corpus_id)
            touched: set[str] = set()
            mentions, relations = await _drop_evidence(
                conn, corpus_id, "story", [story_id], touched
            )
            vectors = await _delete_vectors(conn, table, "story", [story_id])
            stories = await conn.execute(
                "DELETE FROM memory_stories WHERE corpus_id = $1 AND story_id = $2",
                corpus_id,
                story_id,
            )
            closings = await conn.execute(
                "DELETE FROM memory_closings WHERE corpus_id = $1 AND kind = 'story' AND id = $2",
                corpus_id,
                story_id,
            )
            pruned = await _reconcile(conn, corpus_id, touched)
        return Counts(
            stories=_count(stories),
            mentions=mentions,
            relations=relations,
            vectors=vectors,
            entities=pruned,
            closings=_count(closings),
        )
