from __future__ import annotations

import json
from typing import Any

from rfnry_memory.errors import StorageError
from rfnry_memory.models import (
    Entity,
    Kind,
    Ref,
    Relation,
    Scope,
    Window,
)
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.language import _CONFIG
from rfnry_memory.postgres.migrations import _ALIAS_KEY
from rfnry_memory.postgres.records import _entity, _json, _relation
from rfnry_memory.postgres.sql import (
    _chunks_sql,
    _facet_sql,
    _subject_kinds,
    _subject_sql,
    _window_sql,
    _within_sql,
)
from rfnry_memory.rows import (
    GraphWrite,
    name_key,
    reconciled,
)


async def _write_graph(conn: Any, corpus_id: str, graph: GraphWrite, touched: set[str]) -> None:
    ids: dict[str, str] = {}
    for entity in sorted(graph.entities, key=lambda e: (e.key, e.type)):
        ids[entity.entity_id] = await conn.fetchval(
            "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases, config) "
            f"VALUES ($1, $2, $3, $4, $5, $6::jsonb, {_CONFIG}) "
            "ON CONFLICT (corpus_id, key, type) DO UPDATE SET aliases = memory_entities.aliases || coalesce((SELECT jsonb_agg(a) "
            "FROM jsonb_array_elements_text(EXCLUDED.aliases) AS a WHERE NOT memory_entities.aliases ? a), '[]'::jsonb) "
            "RETURNING entity_id",
            corpus_id,
            entity.entity_id,
            entity.key,
            entity.name,
            entity.type,
            json.dumps(list(dict.fromkeys(entity.aliases))),
        )
    named = sorted(graph.named(ids))
    found = {
        r["entity_id"]
        for r in await conn.fetch(
            "SELECT entity_id FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[]) "
            'ORDER BY entity_id COLLATE "C" FOR KEY SHARE',
            corpus_id,
            named,
        )
    }
    known = {e.entity_id: e for e in graph.known}
    for entity_id in sorted((set(named) - found) & set(known)):
        entity = known[entity_id]
        ids[entity_id] = await conn.fetchval(
            "INSERT INTO memory_entities (corpus_id, entity_id, key, name, type, aliases, config) "
            f"VALUES ($1, $2, $3, $4, $5, $6::jsonb, {_CONFIG}) ON CONFLICT DO NOTHING RETURNING entity_id",
            corpus_id,
            entity.entity_id,
            entity.key,
            entity.name,
            entity.type,
            json.dumps(list(dict.fromkeys(entity.aliases))),
        ) or await conn.fetchval(
            "SELECT entity_id FROM memory_entities WHERE corpus_id = $1 AND (entity_id = $2 OR (key = $3 AND type = $4)) "
            "ORDER BY entity_id = $2 DESC LIMIT 1 FOR KEY SHARE",
            corpus_id,
            entity.entity_id,
            entity.key,
            entity.type,
        )
        touched.add(ids[entity_id])
    missing = set(named) - found - set(known)
    if missing:
        raise StorageError(f"postgres: graph facts name unknown entities: {sorted(missing)}")
    for entity_id, alias in sorted((ids.get(e, e), a) for e, a in graph.aliases):
        await conn.execute(
            "UPDATE memory_entities SET aliases = CASE WHEN aliases ? $3 THEN aliases ELSE aliases || to_jsonb($3::text) END "
            "WHERE corpus_id = $1 AND entity_id = $2",
            corpus_id,
            entity_id,
            alias,
        )
    if graph.relations:
        await conn.executemany(
            "INSERT INTO memory_relations (corpus_id, from_id, to_id, type, evidence_kind, evidence_id) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING",
            [
                (
                    corpus_id,
                    ids.get(r.from_id, r.from_id),
                    ids.get(r.to_id, r.to_id),
                    r.type,
                    r.evidence_kind,
                    r.evidence_id,
                )
                for r in graph.relations
            ],
        )
    if graph.mentions:
        await conn.executemany(
            "INSERT INTO memory_mentions (corpus_id, entity_id, passage_kind, passage_id, name) VALUES ($1, $2, $3, $4, $5) ON CONFLICT DO NOTHING",
            [
                (corpus_id, ids.get(e, e), kind, passage_id, name)
                for e, kind, passage_id, name in graph.mentions
            ],
        )
        await _carry(conn, corpus_id, graph.mentions, ids)


async def _carry(
    conn: Any,
    corpus_id: str,
    mentions: list[tuple[str, Kind, str, str]],
    ids: dict[str, str],
) -> None:
    said: dict[str, list[str]] = {}
    for entity_id, _, _, name in mentions:
        said.setdefault(ids.get(entity_id, entity_id), []).append(name)
    for r in await conn.fetch(
        "SELECT entity_id, key, aliases FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[]) "
        'ORDER BY entity_id COLLATE "C"',
        corpus_id,
        sorted(said),
    ):
        carried = {r["key"], *map(name_key, _json(r["aliases"]))}
        new: list[str] = []
        for name in sorted(said[r["entity_id"]]):
            if name_key(name) not in carried:
                carried.add(name_key(name))
                new.append(name)
        if new:
            await conn.execute(
                "UPDATE memory_entities SET aliases = aliases || $3::jsonb WHERE corpus_id = $1 AND entity_id = $2",
                corpus_id,
                r["entity_id"],
                json.dumps(new),
            )


async def _drop_evidence(
    conn: Any, corpus_id: str, kind: Kind, passage_ids: list[str], touched: set[str]
) -> tuple[int, int]:
    if not passage_ids:
        return 0, 0
    mentioned = await conn.fetch(
        "DELETE FROM memory_mentions WHERE corpus_id = $1 AND passage_kind = $2 AND passage_id = ANY($3::text[]) "
        "RETURNING entity_id",
        corpus_id,
        kind,
        passage_ids,
    )
    related = await conn.fetch(
        "DELETE FROM memory_relations WHERE corpus_id = $1 AND evidence_kind = $2 AND evidence_id = ANY($3::text[]) "
        "RETURNING from_id, to_id",
        corpus_id,
        kind,
        passage_ids,
    )
    touched.update(r["entity_id"] for r in mentioned)
    touched.update(e for r in related for e in (r["from_id"], r["to_id"]))
    return len(mentioned), len(related)


async def _reconcile(conn: Any, corpus_id: str, touched: set[str]) -> int:
    if not touched:
        return 0
    ids = sorted(touched)
    entities = [
        _entity(r)
        for r in await conn.fetch(
            "SELECT * FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[]) "
            'ORDER BY entity_id COLLATE "C" FOR UPDATE',
            corpus_id,
            ids,
        )
    ]
    names: dict[str, list[str]] = {}
    for r in await conn.fetch(
        "SELECT entity_id, name FROM memory_mentions WHERE corpus_id = $1 AND entity_id = ANY($2::text[])",
        corpus_id,
        ids,
    ):
        names.setdefault(r["entity_id"], []).append(r["name"])
    related = {
        r["entity_id"]
        for r in await conn.fetch(
            "SELECT from_id AS entity_id FROM memory_relations WHERE corpus_id = $1 AND from_id = ANY($2::text[]) "
            "UNION SELECT to_id FROM memory_relations WHERE corpus_id = $1 AND to_id = ANY($2::text[])",
            corpus_id,
            ids,
        )
    }
    holders: dict[tuple[str, str], set[str]] = {}
    for r in await conn.fetch(
        "SELECT entity_id, key, type FROM memory_entities WHERE corpus_id = $1 AND key = ANY($2::text[])",
        corpus_id,
        sorted({name_key(n) for held in names.values() for n in held}),
    ):
        holders.setdefault((r["key"], r["type"]), set()).add(r["entity_id"])
    pruned: list[str] = []
    for entity in entities:
        held = names.get(entity.entity_id, [])
        taken = {
            key
            for key in map(name_key, held)
            if holders.get((key, entity.type), set()) - {entity.entity_id}
        }
        kept = reconciled(entity, held, entity.entity_id in related, taken)
        if kept is None:
            pruned.append(entity.entity_id)
        elif kept != entity:
            blocked = reconciled(
                entity, held, entity.entity_id in related, {name_key(n) for n in held}
            )
            assert blocked is not None
            await _settle(conn, corpus_id, entity, kept, blocked)
            holders.setdefault((kept.key, kept.type), set()).add(kept.entity_id)
            holders.get((entity.key, entity.type), set()).discard(entity.entity_id)
    if pruned:
        await conn.execute(
            "DELETE FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[])",
            corpus_id,
            pruned,
        )
    return len(pruned)


async def _settle(conn: Any, corpus_id: str, entity: Entity, kept: Entity, blocked: Entity) -> None:
    update = (
        "UPDATE memory_entities SET name = $3, key = $4, aliases = $5::jsonb "
        "WHERE corpus_id = $1 AND entity_id = $2"
    )
    if kept.key != entity.key:
        try:
            async with conn.transaction():
                await conn.execute(
                    update,
                    corpus_id,
                    entity.entity_id,
                    kept.name,
                    kept.key,
                    json.dumps(list(kept.aliases)),
                )
            return
        except Exception as exc:
            if getattr(exc, "sqlstate", None) != "23505":
                raise
            kept = blocked
    await conn.execute(
        update,
        corpus_id,
        entity.entity_id,
        kept.name,
        kept.key,
        json.dumps(list(kept.aliases)),
    )


class Graph(Base):
    async def similar_entities(self, corpus_id: str, name: str, *, limit: int) -> list[Entity]:
        async with self._guard("similar_entities"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT *, greatest(similarity(name, $2), coalesce((SELECT max(similarity(a, $2)) "
                "FROM jsonb_array_elements_text(aliases) AS a), 0)) AS sim FROM memory_entities "
                'WHERE corpus_id = $1 ORDER BY sim DESC, entity_id COLLATE "C" LIMIT $3',
                corpus_id,
                name,
                limit,
            )
        return [_entity(r) for r in records if float(r["sim"]) > 0.3]

    async def entities_named(self, corpus_id: str, name_or_key: str) -> list[Entity]:
        key = name_key(name_or_key)
        async with self._guard("entities_named"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND (key = $2 OR "
                "(aliases <> '[]'::jsonb AND memory_alias_keys(aliases) @> ARRAY[$2])) "
                'ORDER BY entity_id COLLATE "C"',
                corpus_id,
                key,
            )
        return [_entity(r) for r in records]

    async def entities_by_id(self, corpus_id: str, entity_ids: list[str]) -> list[Entity]:
        async with self._guard("entities_by_id"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND entity_id = ANY($2::text[])",
                corpus_id,
                list(entity_ids),
            )
        by_id = {r["entity_id"]: _entity(r) for r in records}
        return [by_id[i] for i in entity_ids if i in by_id]

    async def entities_named_in(self, corpus_id: str, text: str) -> list[Entity]:
        haystack = name_key(text)
        async with self._guard("entities_named_in"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM memory_entities WHERE corpus_id = $1 AND (position(key IN $2) > 0 OR EXISTS "
                f'(SELECT 1 FROM jsonb_array_elements_text(aliases) AS a WHERE position({_ALIAS_KEY} IN $2) > 0)) ORDER BY entity_id COLLATE "C"',
                corpus_id,
                haystack,
            )
        return [_entity(r) for r in records]

    async def neighbors(
        self,
        corpus_id: str,
        entity_ids: list[str],
        *,
        relation_types: list[str] | None,
        window: Window,
    ) -> list[Relation]:
        params: list[Any] = [corpus_id, list(entity_ids)]
        sql = (
            "SELECT r.* FROM memory_relations r WHERE r.corpus_id = $1 "
            "AND (r.from_id = ANY($2::text[]) OR r.to_id = ANY($2::text[]))"
        )
        if relation_types is not None:
            params.append(list(relation_types))
            sql += f" AND r.type = ANY(${len(params)}::text[])"
        sql += (
            " AND ((r.evidence_kind = 'chunk' AND EXISTS (SELECT 1 FROM memory_chunks c "
            "WHERE c.corpus_id = r.corpus_id AND c.chunk_id = r.evidence_id"
            + _chunks_sql(window, params, "c.")
            + ")) OR (r.evidence_kind = 'story' AND EXISTS (SELECT 1 FROM memory_stories s "
            "WHERE s.corpus_id = r.corpus_id AND s.story_id = r.evidence_id"
            + _window_sql(window, params, alias="s.")
            + ")))"
        )
        async with self._guard("neighbors"), self._pool.acquire() as conn:
            records = await conn.fetch(
                sql + ' ORDER BY r.from_id COLLATE "C", r.to_id COLLATE "C", r.type COLLATE "C", '
                'r.evidence_kind COLLATE "C", r.evidence_id COLLATE "C"',
                *params,
            )
        return [_relation(r) for r in records]

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
    ) -> list[tuple[str, Ref]]:
        found: list[tuple[str, Ref]] = []
        kinds = _subject_kinds(kinds, subject)
        async with self._guard("mentions"), self._pool.acquire() as conn:
            if "chunk" in kinds:
                params: list[Any] = [corpus_id, list(entity_ids)]
                facets = (
                    _facet_sql(filters, params, "c.metadata")
                    + _chunks_sql(window, params, "c.")
                    + _within_sql("chunk", scope, params, "c.")
                )
                records = await conn.fetch(
                    'SELECT DISTINCT m.entity_id COLLATE "C" AS entity_id, c.chunk_id COLLATE "C" AS chunk_id '
                    "FROM memory_mentions m JOIN memory_chunks c ON c.corpus_id = m.corpus_id AND c.chunk_id = m.passage_id "
                    f"WHERE m.corpus_id = $1 AND m.passage_kind = 'chunk' AND m.entity_id = ANY($2::text[]){facets} "
                    "ORDER BY chunk_id, entity_id",
                    *params,
                )
                found.extend((r["entity_id"], ("chunk", r["chunk_id"])) for r in records)
            if "story" in kinds:
                params = [corpus_id, list(entity_ids)]
                facets = (
                    _facet_sql(filters, params, "s.metadata::jsonb")
                    + _window_sql(window, params, alias="s.")
                    + _subject_sql(subject, params, alias="s.")
                    + _within_sql("story", scope, params, "s.")
                )
                records = await conn.fetch(
                    'SELECT DISTINCT m.entity_id COLLATE "C" AS entity_id, s.story_id COLLATE "C" AS story_id '
                    "FROM memory_mentions m JOIN memory_stories s ON s.corpus_id = m.corpus_id AND s.story_id = m.passage_id "
                    f"WHERE m.corpus_id = $1 AND m.passage_kind = 'story' AND m.entity_id = ANY($2::text[]){facets} "
                    "ORDER BY story_id, entity_id",
                    *params,
                )
                found.extend((r["entity_id"], ("story", r["story_id"])) for r in records)
        return found

    async def passage_entities(
        self, corpus_id: str, passages: list[Ref]
    ) -> list[tuple[Ref, Entity]]:
        async with self._guard("passage_entities"), self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT w.passage_kind, w.passage_id, e.* FROM (SELECT DISTINCT m.passage_kind, m.passage_id, m.entity_id "
                "FROM memory_mentions m WHERE m.corpus_id = $1 AND (m.passage_kind, m.passage_id) IN "
                "(SELECT * FROM unnest($2::text[], $3::text[]))) AS w "
                "JOIN memory_entities e ON e.corpus_id = $1 AND e.entity_id = w.entity_id "
                'ORDER BY w.passage_kind COLLATE "C", w.passage_id COLLATE "C", w.entity_id COLLATE "C"',
                corpus_id,
                [kind for kind, _ in passages],
                [passage_id for _, passage_id in passages],
            )
        return [((r["passage_kind"], r["passage_id"]), _entity(r)) for r in records]

    async def graph_types(self, corpus_id: str) -> tuple[dict[str, int], dict[str, int]]:
        async with self._guard("graph_types"), self._pool.acquire() as conn:
            entities = await conn.fetch(
                "SELECT type, count(*) AS n FROM memory_entities WHERE corpus_id = $1 "
                'GROUP BY type ORDER BY type COLLATE "C"',
                corpus_id,
            )
            relations = await conn.fetch(
                "SELECT type, count(*) AS n FROM memory_relations WHERE corpus_id = $1 "
                'GROUP BY type ORDER BY type COLLATE "C"',
                corpus_id,
            )
        return (
            {r["type"]: int(r["n"]) for r in entities},
            {r["type"]: int(r["n"]) for r in relations},
        )
