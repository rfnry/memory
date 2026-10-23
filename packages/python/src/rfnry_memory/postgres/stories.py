import json
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
from rfnry_memory.postgres.keys import keyed
from rfnry_memory.postgres.language import _CONFIG, _pin
from rfnry_memory.postgres.locks import _language_lock
from rfnry_memory.postgres.records import _count, _story, _taken, _vector
from rfnry_memory.postgres.sql import _facet_sql, _subject_sql, _window_sql
from rfnry_memory.postgres.vectors import _copy_vectors, _delete_vectors, _vectors_table
from rfnry_memory.rows import (
    GraphWrite,
    ended_on_write,
    text_hash,
)


def _closable(story: Story, new: Story) -> None:
    if story.superseded_by is not None:
        raise InputError(
            f"postgres: story {story.story_id!r} is already superseded; supersede the story that replaced it"
        )
    if story.closed_at is not None and story.valid_to == story.valid_from:
        raise InputError(
            f"postgres: story {story.story_id!r} was retracted at {story.closed_at.isoformat()}; remember a new story instead"
        )
    if story.valid_to is not None and (
        story.closed_at is not None or story.valid_to <= new.valid_from
    ):
        raise InputError(
            f"postgres: story {story.story_id!r} ended at {story.valid_to.isoformat()}; remember a new story instead"
        )
    if new.valid_from < story.valid_from:
        raise InputError(
            f"postgres: valid_from must not be before story {story.story_id!r} became true, at {story.valid_from.isoformat()}"
        )


def _endable(story: Story, end: datetime) -> None:
    if story.superseded_by is not None:
        raise InputError(
            f"postgres: story {story.story_id!r} is already superseded; end the story that replaced it"
        )
    if story.closed_at is not None and story.valid_to == story.valid_from:
        raise InputError(
            f"postgres: story {story.story_id!r} was retracted at {story.closed_at.isoformat()}"
        )
    if story.valid_to is not None and (story.closed_at is not None or story.valid_to <= end):
        raise InputError(
            f"postgres: story {story.story_id!r} already ended, at {story.valid_to.isoformat()}"
        )
    if end < story.valid_from:
        raise InputError(
            f"postgres: valid_to must not be before story {story.story_id!r} became true, at {story.valid_from.isoformat()}"
        )


_SAME = "subject IS NOT DISTINCT FROM $4 AND metadata::jsonb = $5::jsonb"


_OPEN_SQL = (
    "SELECT * FROM memory_stories WHERE corpus_id = $1 AND text_hash = $2 AND superseded_by IS NULL "
    "AND (valid_to IS NULL OR (valid_to > $3 AND valid_to > valid_from)) "
    f'AND {_SAME} ORDER BY story_id COLLATE "C" LIMIT 1'
)


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
                f"closed_by) VALUES ({values}, $15::vector, $16, $17::jsonb, $18::jsonb)"
            )
        return (
            f"INSERT INTO memory_stories ({columns}, embedding_model, written_by, closed_by) "
            f"VALUES ({values}, $15, $16::jsonb, $17::jsonb)"
        )

    def _story_params(
        self, story: Story, vector: list[float] | None, model: str | None
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
            record = await conn.fetchrow(_OPEN_SQL, corpus_id, hash, story.created_at, *same)
        return _story(record) if record else None

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
                return holder
            await _taken(conn, corpus_id, "story", [story.story_id])
            await conn.execute(self._insert_story_sql(), *self._story_params(story, vector, model))
            await conn.execute(
                "SELECT memory_count($1, 'story', $2::text[], 1)", corpus_id, [story.story_id]
            )
            await _copy_vectors(conn, table, corpus_id, "story", [story.story_id])
            recreated: set[str] = set()
            await _write_graph(conn, corpus_id, graph, recreated)
            await _reconcile(conn, corpus_id, recreated)
        return story

    async def find_open_story(
        self,
        corpus_id: str,
        hash: str,
        *,
        subject: str | None,
        metadata: dict[str, Any],
        at: datetime,
    ) -> Story | None:
        async with self._guard("find_open_story"), self._pool.acquire() as conn:
            record = await conn.fetchrow(
                _OPEN_SQL, corpus_id, hash, at, subject, json.dumps(metadata)
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
            hashes = [
                r["text_hash"]
                for r in await conn.fetch(
                    "SELECT text_hash FROM memory_stories WHERE corpus_id = $1 AND story_id = ANY($2::text[])",
                    corpus_id,
                    list(old_ids),
                )
            ]
            await _text_lock(conn, corpus_id, [text_hash(new.text), *hashes])
            rows = {
                r["story_id"]: _story(r)
                for r in await conn.fetch(
                    "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = ANY($2::text[]) "
                    'ORDER BY story_id COLLATE "C" FOR UPDATE',
                    corpus_id,
                    list(old_ids),
                )
            }
            for old_id in old_ids:
                if old_id not in rows:
                    raise NotFoundError(f"postgres: story {old_id!r} was not found")
                _closable(rows[old_id], new)
            holder = await conn.fetchrow(
                _OPEN_SQL,
                corpus_id,
                text_hash(new.text),
                new.created_at,
                new.subject,
                json.dumps(new.metadata),
            )
            if holder is not None and holder["story_id"] not in old_ids:
                raise InputError(
                    f"postgres: story {holder['story_id']!r}, true since {holder['valid_from'].isoformat()}, already says that; supersede or delete it instead"
                )
            await _taken(conn, corpus_id, "story", [new.story_id])
            await conn.execute(
                "SELECT memory_count($1, 'story', $2::text[], -1)", corpus_id, list(old_ids)
            )
            closed = await conn.fetch(
                "UPDATE memory_stories SET valid_to = $3, superseded_by = $4, closed_at = $5, "
                "closed_by = $6::jsonb WHERE corpus_id = $1 AND story_id = ANY($2::text[]) RETURNING *",
                corpus_id,
                list(old_ids),
                new.valid_from,
                new.story_id,
                new.created_at,
                json.dumps(new.written_by),
            )
            await conn.execute(self._insert_story_sql(), *self._story_params(new, vector, model))
            await conn.execute(
                "SELECT memory_count($1, 'story', $2::text[], 1)", corpus_id, [new.story_id]
            )
            await _copy_vectors(conn, table, corpus_id, "story", [new.story_id])
            recreated: set[str] = set()
            await _write_graph(conn, corpus_id, graph, recreated)
            await _reconcile(conn, corpus_id, recreated)
        by_id = {r["story_id"]: _story(r) for r in closed}
        return [by_id[i] for i in old_ids], new

    async def end_story(
        self,
        corpus_id: str,
        story_id: str,
        *,
        valid_to: datetime,
        closed_at: datetime,
        closed_by: dict[str, Any] | None = None,
    ) -> Story:
        async with (
            self._guard("end_story"),
            self._pool.acquire() as conn,
            conn.transaction(isolation="read_committed"),
        ):
            await _language_lock(conn, corpus_id, shared=True)
            record = await conn.fetchrow(
                "SELECT * FROM memory_stories WHERE corpus_id = $1 AND story_id = $2 FOR UPDATE",
                corpus_id,
                story_id,
            )
            if record is None:
                raise NotFoundError(f"postgres: story {story_id!r} was not found")
            _endable(_story(record), valid_to)
            await conn.execute(
                "SELECT memory_count($1, 'story', $2::text[], -1)", corpus_id, [story_id]
            )
            ended = await conn.fetchrow(
                "UPDATE memory_stories SET valid_to = $3, closed_at = $4, closed_by = $5::jsonb "
                "WHERE corpus_id = $1 AND story_id = $2 RETURNING *",
                corpus_id,
                story_id,
                valid_to,
                closed_at,
                json.dumps(closed_by or {}),
            )
        return _story(ended)

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
            filters = await keyed(conn, filters, window.now)
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
        rows = [_story(r, window.known_at) for r in records]
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
            pruned = await _reconcile(conn, corpus_id, touched)
        return Counts(
            stories=_count(stories),
            mentions=mentions,
            relations=relations,
            vectors=vectors,
            entities=pruned,
        )
