import json
import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError
from rfnry_memory.judge import judge as judge_hits
from rfnry_memory.models import Chunk
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.refinery.export import ExportBucket, export_key, export_row
from rfnry_memory.refinery.holdout import HOLDOUT_PERCENT, held_out, sampled
from rfnry_memory.refinery.metrics import (
    Key,
    ndcg_at,
    percentile,
    recall_at,
    reciprocal_rank,
    unique,
)
from rfnry_memory.refinery.records import (
    RecordedHit,
    Relevance,
    Retrieval,
    RetrievalExport,
    RetrievalLabel,
    RetrievalLabeled,
    RetrievalMetrics,
    RetrievalOutcome,
    RetrievalReplay,
    RetrievalReview,
    RetrievalScores,
    label_of,
    retrieval_of,
)
from rfnry_memory.rows import RetrievalLabelRow, RetrievalRow

if TYPE_CHECKING:
    from rfnry_memory.memory import Memory

Source = Literal["human", "world", "judge"]
Moment = Literal["recorded", "now"]
REPORTED: tuple[Source, ...] = ("human", "world")


def named(corpus_id: str | list[str]) -> list[str]:
    corpora = list(corpus_id) if isinstance(corpus_id, list | tuple) else [corpus_id]
    if not corpora or not all(isinstance(c, str) and c.strip() for c in corpora):
        raise InputError("refinery: corpus_id is required")
    return sorted(dict.fromkeys(corpora))


class MemoryRefinery:
    def __init__(self, storage: PostgresStorage, *, sample: float = 1.0) -> None:
        if not isinstance(sample, int | float) or not math.isfinite(sample) or not 0 <= sample <= 1:
            raise ConfigurationError("refinery: sample is a rate from 0 to 1")
        self.storage = storage
        self.sample = float(sample)

    def is_holdout(self, retrieval_id: str, *, holdout_percent: int = HOLDOUT_PERCENT) -> bool:
        if not 0 <= holdout_percent <= 100:
            raise InputError(f"refinery: holdout_percent must be 0-100, got {holdout_percent}")
        return held_out(retrieval_id, holdout_percent)

    def _records(self, retrieval_id: str) -> bool:
        return sampled(retrieval_id, self.sample)

    async def retrieval(self, corpus_id: str | list[str], retrieval_id: str) -> Retrieval:
        row = await self.storage.get_retrieval(named(corpus_id), retrieval_id)
        if row is None:
            raise NotFoundError(f"refinery: retrieval {retrieval_id!r} was not found")
        return retrieval_of(row)

    async def retrievals(
        self,
        corpus_id: str | list[str],
        *,
        since: datetime | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[Retrieval], str | None]:
        if limit < 1:
            raise InputError("refinery: limit must be at least 1")
        if cursor is not None and not (cursor.isdigit() and cursor.isascii()):
            raise InputError(f"refinery: cursor {cursor!r} is not one a listing returned")
        rows, following = await self.storage.list_retrievals(
            named(corpus_id), since=since, limit=limit, cursor=cursor
        )
        return [retrieval_of(row) for row in rows], following

    async def forget_recorded(self, corpus_id: str | list[str], *, before: datetime) -> int:
        moment = before if before.tzinfo is not None else before.replace(tzinfo=UTC)
        return await self.storage.forget_retrievals(named(corpus_id), before=moment)

    async def _readable(self, corpora: list[str], retrieval_id: str) -> RetrievalRow:
        row = await self.storage.get_retrieval(corpora, retrieval_id)
        if row is None:
            raise NotFoundError(f"refinery: retrieval {retrieval_id!r} was not found")
        return row

    async def _labeled(
        self,
        corpora: list[str],
        retrieval_id: str,
        source: str,
        body: dict[str, Any],
        label_id: str | None,
    ) -> RetrievalLabel:
        if label_id is not None and (not isinstance(label_id, str) or not label_id.strip()):
            raise InputError("refinery: label_id is a non-empty string")
        row = RetrievalLabelRow(
            retrieval_id=retrieval_id,
            label_id=label_id or uuid4().hex,
            source=source,
            at=datetime.now(UTC),
            body=body,
        )
        stored = await self.storage.put_retrieval_label(corpora, row)
        if stored is None:
            raise NotFoundError(f"refinery: retrieval {retrieval_id!r} was not found")
        return label_of(stored)

    async def review(
        self,
        corpus_id: str | list[str],
        retrieval_id: str,
        review: RetrievalReview,
        *,
        label_id: str | None = None,
    ) -> RetrievalLabel:
        corpora = named(corpus_id)
        _within(await self._readable(corpora, retrieval_id), review.relevance)
        return await self._labeled(
            corpora, retrieval_id, "human", {"review": review.model_dump(mode="json")}, label_id
        )

    async def outcome(
        self,
        corpus_id: str | list[str],
        retrieval_id: str,
        outcome: RetrievalOutcome,
        *,
        label_id: str | None = None,
    ) -> RetrievalLabel:
        corpora = named(corpus_id)
        _within(await self._readable(corpora, retrieval_id), outcome.cited)
        return await self._labeled(
            corpora, retrieval_id, "world", {"outcome": outcome.model_dump(mode="json")}, label_id
        )

    async def labels(self, corpus_id: str | list[str], retrieval_id: str) -> list[RetrievalLabel]:
        corpora = named(corpus_id)
        await self._readable(corpora, retrieval_id)
        return [
            label_of(row) for row in await self.storage.retrieval_labels(corpora, [retrieval_id])
        ]

    async def label(
        self, corpus_id: str | list[str], memory: Memory, *, limit: int = 500
    ) -> RetrievalLabeled:
        if memory.decision is None:
            raise ConfigurationError("refinery: label needs a memory with a decision model")
        if limit < 1:
            raise InputError("refinery: limit must be at least 1")
        corpora = named(corpus_id)
        done = await self.storage.judged_retrievals(corpora)
        rows, _ = await self.storage.list_retrievals(corpora, since=None, limit=None, cursor=None)
        pending = [
            row
            for row in reversed(rows)
            if row.retrieval_id not in done and row.verb != "resembling"
        ][:limit]
        labeled = RetrievalLabeled()
        for row in pending:
            try:
                judged = await _judged(row, memory)
            except Exception as exc:
                labeled.errors[row.retrieval_id] = f"{type(exc).__name__}: {exc}"
                continue
            body = {"judged": [r.model_dump(mode="json") for r in judged]}
            await self._labeled(corpora, row.retrieval_id, "judge", body, None)
            labeled.judged += 1
        return labeled

    async def _selected(
        self,
        corpora: list[str],
        *,
        holdout: bool | None,
        labeled: bool,
        sources: tuple[Source, ...],
        since: datetime | None,
        limit: int | None,
    ) -> list[tuple[RetrievalRow, list[RetrievalLabel]]]:
        if not sources or not set(sources) <= {"human", "world", "judge"}:
            raise InputError("refinery: sources are among human, world and judge")
        if limit is not None and limit < 1:
            raise InputError("refinery: limit must be at least 1")
        rows, _ = await self.storage.list_retrievals(corpora, since=since, limit=None, cursor=None)
        rows.reverse()
        labels: dict[str, list[RetrievalLabel]] = {}
        for stored in await self.storage.retrieval_labels(corpora, [r.retrieval_id for r in rows]):
            labels.setdefault(stored.retrieval_id, []).append(label_of(stored))
        picked: list[tuple[RetrievalRow, list[RetrievalLabel]]] = []
        for row in rows:
            if holdout is not None and held_out(row.retrieval_id, HOLDOUT_PERCENT) != holdout:
                continue
            own = labels.get(row.retrieval_id, [])
            if labeled and not any(label.source in sources for label in own):
                continue
            picked.append((row, own))
            if limit is not None and len(picked) == limit:
                break
        return picked

    async def replay(
        self,
        corpus_id: str | list[str],
        candidate: Memory,
        *,
        holdout: bool | None = True,
        labeled: bool = True,
        moment: Literal["recorded", "now"] = "recorded",
        sources: tuple[Literal["human", "world", "judge"], ...] = ("human", "world"),
        since: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[RetrievalReplay]:
        if moment not in ("recorded", "now"):
            raise InputError("refinery: moment is recorded or now")
        picked = await self._selected(
            named(corpus_id),
            holdout=holdout,
            labeled=labeled,
            sources=sources,
            since=since,
            limit=limit,
        )
        for row, own in picked:
            recorded = retrieval_of(row)
            try:
                produced = await _rerun(candidate, recorded, moment)
            except Exception as exc:
                yield RetrievalReplay(
                    recorded=recorded, labels=own, error=f"{type(exc).__name__}: {exc}"
                )
                continue
            yield RetrievalReplay(recorded=recorded, labels=own, candidate=produced)

    async def measure(
        self,
        corpus_id: str | list[str],
        candidate: Memory | None = None,
        *,
        k: int = 10,
        holdout: bool | None = True,
        labeled: bool = True,
        moment: Literal["recorded", "now"] = "recorded",
        sources: tuple[Literal["human", "world", "judge"], ...] = ("human", "world"),
        since: datetime | None = None,
        limit: int | None = None,
    ) -> RetrievalMetrics:
        if k < 1:
            raise InputError("refinery: k must be at least 1")
        corpora = named(corpus_id)
        pairs: list[tuple[Retrieval, list[RetrievalLabel], Retrieval | None]] = []
        if candidate is None:
            for row, own in await self._selected(
                corpora, holdout=holdout, labeled=labeled, sources=sources, since=since, limit=limit
            ):
                recorded = retrieval_of(row)
                pairs.append((recorded, own, recorded))
        else:
            async for replayed in self.replay(
                corpora,
                candidate,
                holdout=holdout,
                labeled=labeled,
                moment=moment,
                sources=sources,
                since=since,
                limit=limit,
            ):
                pairs.append((replayed.recorded, replayed.labels, replayed.candidate))
        same = candidate is None or candidate.storage is self.storage
        documents: dict[str, list[tuple[float, float, float]]] = {}
        passages: dict[str, list[tuple[float, float, float]]] = {}
        durations: list[float] = []
        scored = 0
        for _, own, served in pairs:
            if served is None:
                continue
            durations.append(served.duration_ms)
            graded, filed = _grades(own, sources)
            if not any(grade >= 1 for grade in filed.values()):
                continue
            scored += 1
            names = await self._names(filed) if candidate is not None else {}
            lists = {"fused": served.hits, **served.candidates}
            for name, hits in lists.items():
                ranked = unique([_document(hit, filed, names) for hit in hits])
                documents.setdefault(name, []).append(_scores(ranked, filed, k))
                if same:
                    keyed = unique([(hit.corpus_id, hit.kind, hit.id) for hit in hits])
                    passages.setdefault(name, []).append(_scores(keyed, graded, k))
        return RetrievalMetrics(
            k=k,
            retrievals=len(durations),
            labeled=scored,
            errors=sum(served is None for _, _, served in pairs),
            documents={name: _mean(values) for name, values in sorted(documents.items())},
            passages={name: _mean(values) for name, values in sorted(passages.items())}
            if same
            else None,
            latency_ms={"p50": percentile(durations, 0.5), "p95": percentile(durations, 0.95)},
        )

    async def _names(self, filed: dict[Key, int]) -> dict[tuple[str, str], Key]:
        names: dict[tuple[str, str], Key] = {}
        for key in filed:
            corpus_id, kind, document_id = key
            if kind != "source":
                continue
            source = await self.storage.get_source(corpus_id, document_id)
            if source is not None:
                names[(corpus_id, source.name)] = key
        return names

    async def export(
        self,
        corpus_id: str | list[str],
        bucket: ExportBucket,
        *,
        since: datetime | None = None,
        holdout_percent: int = HOLDOUT_PERCENT,
    ) -> RetrievalExport:
        if not 0 <= holdout_percent <= 100:
            raise InputError(f"refinery: holdout_percent must be 0-100, got {holdout_percent}")
        corpora = named(corpus_id)
        rows, _ = await self.storage.list_retrievals(corpora, since=None, limit=None, cursor=None)
        stored = await self.storage.retrieval_labels(corpora, [r.retrieval_id for r in rows])
        labels: dict[str, list[RetrievalLabel]] = {}
        for label_row in stored:
            labels.setdefault(label_row.retrieval_id, []).append(label_of(label_row))
        if since is not None:
            relabeled = {row.retrieval_id for row in stored if row.at > since}
            rows = [row for row in rows if row.recorded > since or row.retrieval_id in relabeled]
        if not rows:
            return RetrievalExport(keys=[], rows=0, missing=0, watermark=since)
        rows.sort(key=lambda row: (row.recorded, row.retrieval_id))
        stamps = [row.recorded for row in rows]
        lines: list[dict[str, Any]] = []
        missing = 0
        for row in rows:
            recording = retrieval_of(row)
            own = labels.get(row.retrieval_id, [])
            stamps.extend(label.at for label in own)
            graded = _winning(own)
            hits = {(hit.corpus_id, hit.kind, hit.id) for hit in recording.hits}
            wanted = [k for k, (g, _) in graded.items() if g >= 1 or k in hits]
            texts = await self._texts(wanted)
            missing += len(wanted) - len(texts)
            holdout = held_out(row.retrieval_id, holdout_percent)
            lines.append(export_row(recording, graded, texts, own, holdout=holdout))
        key = export_key(corpora, datetime.now(UTC))
        body = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"
        await bucket.put(key, body.encode("utf-8"))
        return RetrievalExport(keys=[key], rows=len(lines), missing=missing, watermark=max(stamps))

    async def _texts(self, keys: list[Key]) -> dict[Key, str]:
        texts: dict[Key, str] = {}
        corpora = sorted({corpus_id for corpus_id, _, _ in keys})
        for corpus_id in corpora:
            refs = [(kind, i) for c, kind, i in keys if c == corpus_id and kind != "source"]
            for passage in await self.storage.get_passages(corpus_id, refs):
                if isinstance(passage, Chunk):
                    texts[(corpus_id, "chunk", passage.chunk_id)] = passage.text
                else:
                    texts[(corpus_id, "story", passage.story_id)] = passage.text
            for c, kind, i in keys:
                if c == corpus_id and kind == "source":
                    source = await self.storage.get_source(corpus_id, i)
                    if source is not None:
                        texts[(c, kind, i)] = source.text
        return texts


def _within(row: RetrievalRow, relevance: list[Relevance]) -> None:
    for graded in relevance:
        if graded.corpus_id not in row.corpus_ids:
            raise InputError(
                f"refinery: retrieval {row.retrieval_id!r} read {', '.join(row.corpus_ids)}; "
                f"a grade naming {graded.corpus_id!r} is not one of them"
            )


def _graded(hit: RecordedHit, probability: float | None) -> Relevance:
    return Relevance(
        corpus_id=hit.corpus_id,
        kind=hit.kind,
        id=hit.id,
        grade=1 if probability is not None and probability >= 0.5 else 0,
        probability=probability,
    )


async def _judged(row: RetrievalRow, memory: Memory) -> list[Relevance]:
    hits = [RecordedHit.model_validate(hit) for hit in row.body["hits"]]
    if hits and all(hit.judge is not None and hit.judge.relevant is not None for hit in hits):
        return [_graded(hit, hit.judge.relevant if hit.judge else None) for hit in hits]
    query = row.body["query"]
    if query is None:
        kind, passage_id = row.body["arguments"]["passage"]
        found = await memory.storage.get_passages(row.corpus_ids[0], [(kind, passage_id)])
        if not found:
            return []
        query = found[0].text
    fused: list[tuple[str, float, dict[str, float]]] = [
        (json.dumps([hit.corpus_id, hit.kind, hit.id]), hit.score, {}) for hit in hits
    ]
    present = await memory._hydrate(fused)
    assert memory.decision is not None
    judged, _ = await judge_hits(memory.decision, query, present, ("relevant",))
    by_id = {(hit.corpus_id, hit.kind, hit.id): hit for hit in hits}
    return [
        _graded(by_id[(hit.corpus_id, hit.kind, hit.id)], hit.judge.relevant if hit.judge else None)
        for hit in judged
    ]


def _owner(corpus_id: str, kind: str, passage_id: str) -> tuple[str, str, str]:
    if kind == "chunk":
        return (corpus_id, "source", passage_id.rsplit("-", 1)[0])
    return (corpus_id, kind, passage_id)


def _document(hit: RecordedHit, filed: dict[Key, int], names: dict[tuple[str, str], Key]) -> Key:
    key = (
        (hit.corpus_id, "source", hit.source_id)
        if hit.kind == "chunk" and hit.source_id is not None
        else _owner(hit.corpus_id, hit.kind, hit.id)
    )
    if key in filed or hit.source_name is None:
        return key
    return names.get((hit.corpus_id, hit.source_name), key)


def _grades(
    labels: list[RetrievalLabel], sources: tuple[Source, ...]
) -> tuple[dict[Key, int], dict[Key, int]]:
    merged = _winning([label for label in labels if label.source in sources])
    passages = {key: grade for key, (grade, _) in merged.items() if key[1] != "source"}
    documents: dict[Key, int] = {}
    for key, (grade, _) in merged.items():
        owner = _owner(*key)
        documents[owner] = max(documents.get(owner, 0), grade)
    return passages, documents


def _scores(ranked: list[Key], grades: dict[Key, int], k: int) -> tuple[float, float, float]:
    return recall_at(ranked, grades, k), reciprocal_rank(ranked, grades), ndcg_at(ranked, grades, k)


def _mean(values: list[tuple[float, float, float]]) -> RetrievalScores:
    count = len(values)
    return RetrievalScores(
        recall=sum(v[0] for v in values) / count,
        mrr=sum(v[1] for v in values) / count,
        ndcg=sum(v[2] for v in values) / count,
        measured=count,
    )


def _when(value: Any) -> datetime | None:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


async def _rerun(candidate: Memory, recorded: Retrieval, moment: Moment) -> Retrieval:
    from rfnry_memory.memory import _replaying

    args = recorded.arguments
    known = _when(args.get("known_at"))
    if moment == "recorded" and known is None:
        known = recorded.at
    window: dict[str, Any] = {
        "as_of": _when(args.get("as_of")),
        "known_at": known,
        "history": args.get("history", False),
    }
    caught: list[RetrievalRow] = []
    token = _replaying.set(caught)
    try:
        if recorded.verb in ("search", "recall"):
            pillars = None
            if args.get("pillars") is not None:
                pillars = set(args["pillars"]) & candidate.pillars
                if not pillars:
                    raise InputError(
                        "refinery: none of the recording's pillars is configured on the candidate"
                    )
            shared = {
                "pillars": pillars,
                "top_k": args["top_k"],
                "filters": args["filters"],
                "entities": args["entities"],
                "relation_types": args["relation_types"],
                "sources": args["sources"],
                "around": args["around"],
                "hops": args["hops"],
                "match": args["match"],
                "depth": args["depth"],
                **window,
            }
            if recorded.verb == "search":
                await candidate.search(
                    args["corpus_id"], args["query"], kinds=tuple(args["kinds"]), **shared
                )
            else:
                await candidate.recall(
                    args["corpus_id"],
                    args["query"],
                    since=_when(args["since"]),
                    until=_when(args["until"]),
                    subject=args["subject"],
                    **shared,
                )
        elif recorded.verb == "expand":
            kind, passage_id = args["passage"]
            await candidate.expand(
                args["corpus_id"],
                (kind, passage_id),
                by=args["by"],
                top_k=args["top_k"],
                width=args["width"],
                kinds=tuple(args["kinds"]),
                relation_types=args["relation_types"],
                filters=args["filters"],
                depth=args["depth"],
                **window,
            )
        else:
            await candidate.resembling(
                args["corpus_id"],
                text=args["text"],
                source_id=args["source_id"],
                source_corpus=args["source_corpus"],
                filters=args["filters"],
                limit=args["limit"],
                **window,
            )
    finally:
        _replaying.reset(token)
    return retrieval_of(caught[0])


def _winning(labels: list[RetrievalLabel]) -> dict[Key, tuple[int, str]]:
    by_source: dict[str, dict[Key, tuple[int, str]]] = {"judge": {}, "world": {}, "human": {}}
    for label in sorted(labels, key=lambda label: label.at):
        graded = (
            label.review.relevance
            if label.review is not None
            else label.outcome.cited
            if label.outcome is not None
            else label.judged or []
        )
        for relevance in graded:
            key = (relevance.corpus_id, relevance.kind, relevance.id)
            by_source[label.source][key] = (relevance.grade, label.source)
    return {**by_source["judge"], **by_source["world"], **by_source["human"]}
