import argparse
import asyncio
import json
import os
import shlex
import sys
import time
from importlib import import_module
from inspect import isawaitable
from typing import Any, TypeGuard

import asyncpg

from rfnry_memory.cli import _confirmed, _scrubbed
from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.events import Embedded, Event
from rfnry_memory.filters import validate_filters
from rfnry_memory.memory import Memory
from rfnry_memory.models import EmbeddingModel, Ref
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.postgres.schema import schema_problems
from rfnry_memory.postgres.switch import CorpusVectors, apply_swap, plan_swap, vector_status

_EVERY = 5.0


_NAMED = 5


_SHOWN = 20


_SAID = 500


_SWAPPING = "one transaction: every vector moves or none does, and a stop changes nothing"


def _said(exc: BaseException, url: str | None) -> str:
    said = " ".join(_scrubbed(str(exc), url).split())[:_SAID]
    return ": ".join(filter(None, [type(exc).__name__, said]))


def _is_model(held: object) -> TypeGuard[EmbeddingModel]:
    return isinstance(held, EmbeddingModel) and not isinstance(held, type)


async def load_embedding(path: str, url: str | None = None) -> EmbeddingModel:
    module, colon, attribute = path.partition(":")
    if not (module and colon and attribute):
        raise ConfigurationError(f"cli: --embedding is module:attribute; it was given {path!r}")
    sys.path.insert(0, os.getcwd())
    try:
        loaded = import_module(module)
    except (Exception, SystemExit) as exc:
        raise ConfigurationError(f"cli: cannot import {module}: {_said(exc, url)}") from None
    if not hasattr(loaded, attribute):
        raise ConfigurationError(f"cli: {module} has no attribute {attribute}")
    made = getattr(loaded, attribute)
    if callable(made) and not _is_model(made):
        try:
            made = made()
            if isawaitable(made):
                made = await made
        except (Exception, SystemExit) as exc:
            raise ConfigurationError(f"cli: {path} raised {_said(exc, url)}") from None
    if not _is_model(made):
        raise ConfigurationError(
            f"cli: {path} is not an EmbeddingModel: it has model, dimension and embed"
        )
    return made


def _parsed(filters: str | None) -> dict[str, Any] | None:
    if filters is None:
        return None
    try:
        parsed = json.loads(filters)
    except ValueError as exc:
        raise InputError(f"cli: --filters is not JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise InputError("cli: --filters is one JSON object")
    validate_filters(parsed)
    return parsed


def _check_writable(path: str | None) -> None:
    if path is None:
        return
    try:
        with open(path, "a", encoding="utf-8"):
            pass
    except OSError as exc:
        raise InputError(f"cli: --refused cannot be written: {exc}") from None


async def _corpora(args: argparse.Namespace, conn: Any) -> list[str]:
    held = [
        r["corpus_id"]
        for r in await conn.fetch("SELECT corpus_id FROM memory_corpora ORDER BY corpus_id")
    ]
    if args.corpus is None:
        return held
    unknown = sorted(set(args.corpus) - set(held))
    if unknown:
        listed = ", ".join(repr(corpus_id) for corpus_id in unknown)
        raise InputError(f"cli: schema {args.schema} holds no corpus {listed}")
    return list(dict.fromkeys(args.corpus))


def _command(
    args: argparse.Namespace, action: str, *given: str, corpora: list[str] | None = None
) -> str:
    named = [flag for corpus_id in corpora or [] for flag in ("--corpus", corpus_id)]
    schema = [] if args.schema == "memory" else ["--schema", args.schema]
    return shlex.join(["rfnry-memory", "vectors", action, *given, *named, *schema])


def _where(args: argparse.Namespace, corpora: list[str]) -> tuple[str, list[str]]:
    if len(corpora) <= _NAMED:
        return f"in {', '.join(corpora)}", corpora
    return f"in {len(corpora):,} corpora", args.corpus or []


def _held(models: dict[str, int], width: int) -> str:
    return ", ".join(f"{model} {count:>{width},}" for model, count in models.items()) or "none"


def _index(row: CorpusVectors) -> str:
    return f"index to build: {row.to_build}" if row.to_build else row.index or "none"


def _table(rows: list[CorpusVectors]) -> list[str]:
    width = max(
        len(f"{count:,}")
        for row in rows
        for count in (row.passages, *row.on_row.values(), *row.staged.values())
    )
    heads = ("corpus", "passages", "on the row", "staged", "index")
    cells = [
        heads,
        *(
            (
                row.corpus_id,
                f"{row.passages:>{max(width, len(heads[1]))},}",
                _held(row.on_row, width),
                _held(row.staged, width),
                _index(row),
            )
            for row in rows
        ),
    ]
    widths = [max(len(cell) for cell in column) for column in zip(*cells, strict=True)]
    return [
        "  ".join(cell.ljust(size) for cell, size in zip(line, widths, strict=True)).rstrip()
        for line in cells
    ]


def _next_for(
    args: argparse.Namespace, model: str, rows: list[CorpusVectors], dimension: int
) -> list[str]:
    steps: list[str] = []
    sizes = {row.staged_dimensions[model] for row in rows if model in row.staged_dimensions}
    other = sorted(sizes - {dimension})
    if other:
        steps.append(
            f"{model}: staged at {other[0]} dimensions, the schema holds {dimension} — "
            "its swap covers the whole schema"
        )
    lacking = [row.corpus_id for row in rows if row.missing[model]]
    if lacking:
        where, scope = _where(args, lacking)
        run = _command(args, "stage", "--embedding", "MODULE:NAME", corpora=scope)
        count = sum(row.missing[model] for row in rows)
        steps.append(f"{model}: {count:,} missing, {where} — run {run}")
    elif any(row.on_row.get(model, 0) < row.passages for row in rows):
        forget = _command(args, "forget", "--model", model, corpora=args.corpus)
        drop = f"or {forget} to drop its staged vectors"
        if other and args.corpus:
            check = _command(args, "status", "--model", model)
            steps.append(
                f"{model}: nothing missing in the corpora named, the only ones checked — "
                f"run {check} to check the whole schema its swap needs, {drop}"
            )
        else:
            swap = _command(args, "swap", "--model", model, corpora=args.corpus)
            steps.append(f"{model}: nothing missing — run {swap} to move to it, {drop}")
    return steps


def _next_steps(args: argparse.Namespace, rows: list[CorpusVectors], dimension: int) -> list[str]:
    models = [args.model] if args.model else sorted({m for row in rows for m in row.missing})
    steps = [step for model in models for step in _next_for(args, model, rows, dimension)]
    building = [row.corpus_id for row in rows if row.to_build]
    if building:
        where, scope = _where(args, building)
        steps.append(f"index to build, {where} — run {_command(args, 'index', corpora=scope)}")
    if steps:
        return steps
    if models:
        return [f"{', '.join(models)}: on every row, nothing to run"]
    return ["no vectors here, nothing to run"]


async def _status(args: argparse.Namespace, conn: Any, dimension: int) -> int:
    rows = await vector_status(conn, await _corpora(args, conn), model=args.model)
    models = sorted({model for row in rows for model in row.missing})
    missing = {model: sum(row.missing[model] for row in rows) for model in models}
    if args.json:
        facts = {
            "schema": args.schema,
            "dimension": dimension,
            "missing": missing,
            "corpora": [row.model_dump() for row in rows],
        }
        print(json.dumps(facts, indent=2, ensure_ascii=False))
    else:
        print(f"schema {args.schema}: vectors of {dimension} dimensions")
        if rows:
            print()
            print("\n".join(_table(rows)))
            print()
            print("\n".join(_next_steps(args, rows, dimension)))
        else:
            print("no corpora")
    return 1 if missing.get(args.model) else 0


def _left(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return "under a minute left"
    if seconds <= 7200:
        return f"about {minutes} min left"
    return f"about {seconds / 3600:.1f} h left"


class _Progress:
    def __init__(self) -> None:
        self.written = 0
        self._worked_ms = 0
        self._shown = -_EVERY

    async def show(self, event: Event) -> None:
        if not isinstance(event, Embedded):
            return
        self.written += event.computed
        self._worked_ms += event.duration_ms
        moment = time.monotonic()
        if moment - self._shown < _EVERY and event.done < event.total:
            return
        self._shown = moment
        rate = self.written * 1000 / max(self._worked_ms, 1)
        estimate = (
            f", {_left((event.total - event.done) / rate)}" if event.done < event.total else ""
        )
        print(
            f"{event.corpus_id}: {event.done:,} of {event.total:,}, {rate:,.0f}/s{estimate}",
            file=sys.stderr,
        )


def _counted(count: int, noun: str) -> str:
    return f"{count:,} {noun}{'' if count == 1 else 's'}"


def _list_refused(model: str, refused: list[tuple[str, Ref]], every: str | None) -> None:
    if refused:
        passages = _counted(len(refused), "passage")
        print(
            f"cli: {model} refused {passages}: missing until a later run embeds them",
            file=sys.stderr,
        )
        for corpus_id, (kind, passage_id) in refused[:_SHOWN]:
            print(f"  {corpus_id} {kind} {passage_id}", file=sys.stderr)
    if every is not None:
        with open(every, "w", encoding="utf-8") as kept:
            for corpus_id, (kind, passage_id) in refused:
                kept.write(json.dumps({"corpus_id": corpus_id, "kind": kind, "id": passage_id}))
                kept.write("\n")
        if refused:
            print(f"  all of them are in {every}", file=sys.stderr)
    elif len(refused) > _SHOWN:
        print(
            f"  and {len(refused) - _SHOWN:,} more — add --refused FILE to write every one down",
            file=sys.stderr,
        )


def _kept(written: int) -> str:
    return f"{_counted(written, 'vector')} written and kept; running it again continues"


async def _stage(args: argparse.Namespace, conn: Any) -> int:
    filters = _parsed(args.filters)
    corpora = await _corpora(args, conn)
    _check_writable(args.refused)
    embedding = await load_embedding(args.embedding, args.url)
    model = embedding.model
    search_path = await conn.fetchval("SHOW search_path")
    progress = _Progress()
    refused: list[tuple[str, Ref]] = []
    stopped = 0
    try:
        async with asyncpg.create_pool(
            args.url,
            min_size=1,
            max_size=2,
            server_settings={"search_path": search_path, "statement_timeout": "0"},
        ) as pool:
            memory = Memory(PostgresStorage(pool), embedding=embedding, pillars={"vector"})
            for corpus_id in corpora:
                print(f"{corpus_id}: reading what {model} lacks", file=sys.stderr)
                staged = await memory.stage_vectors(
                    corpus_id,
                    filters=filters,
                    limit=None if args.limit is None else args.limit - progress.written,
                    batch=args.batch,
                    on_event=progress.show,
                )
                refused.extend((corpus_id, ref) for ref in staged.refused)
                print(
                    f"{corpus_id}: {_counted(staged.embedded, 'vector')} of {model} staged"
                    if staged.embedded or staged.refused
                    else f"{corpus_id}: nothing to stage for {model}"
                )
                if args.limit is not None and progress.written >= args.limit:
                    print(f"limit of {args.limit:,} reached; running it again continues")
                    break
    except InputError, ConfigurationError:
        raise
    except KeyboardInterrupt, asyncio.CancelledError:
        print(f"cli: stopped — {_kept(progress.written)}", file=sys.stderr)
        stopped = 130
    except (Exception, SystemExit) as exc:
        print(
            f"cli: stopped by {_said(exc, args.url)} — {_kept(progress.written)}", file=sys.stderr
        )
        stopped = 1
    _list_refused(model, refused, args.refused)
    if stopped:
        return stopped
    if refused:
        return 1
    print(f"next: {_command(args, 'status', '--model', model, corpora=args.corpus)}")
    return 0


async def _swapped(args: argparse.Namespace, conn: Any, setting: str) -> int:
    corpora = await _corpora(args, conn)
    shown = await plan_swap(
        conn,
        model=args.model,
        corpus_ids=None if args.corpus is None else corpora,
        allow_missing=args.allow_missing,
        no_index=args.no_index,
    )
    for refusal in shown.refusals:
        print(f"cli: {refusal}", file=sys.stderr)
    if shown.refusals:
        return 1
    if args.sql:
        print(f"{setting};")
        print("BEGIN;")
        for statement in shown.statements:
            print(f"{statement};")
        print("COMMIT;")
        return 0
    for step in shown.steps:
        print(f"  {step}")
    if not _confirmed(args, _SWAPPING):
        print("nothing was run")
        return 0
    print(f"schema {args.schema}: waiting for running writes, then moving the vectors")
    await apply_swap(conn, shown)
    print(f"schema {args.schema}: {args.model} is on the rows")
    print(f"next: {_command(args, 'status', '--model', args.model, corpora=args.corpus)}")
    return 0


async def _swap(args: argparse.Namespace, conn: Any, setting: str) -> int:
    try:
        return await _swapped(args, conn, setting)
    except KeyboardInterrupt, asyncio.CancelledError:
        print("cli: stopped — the swap rolled back, nothing moved", file=sys.stderr)
        return 130


async def run(args: argparse.Namespace, conn: Any, setting: str) -> int:
    await conn.execute("SET statement_timeout = 0")
    problems, dimension = await schema_problems(conn, dimension=None, graph=False)
    for problem in problems:
        print(f"cli: {problem}", file=sys.stderr)
    if problems:
        return 1
    if dimension is None:
        print(
            f"cli: schema {args.schema} holds no vectors — "
            f"run rfnry-memory set --dimension N --schema {args.schema}",
            file=sys.stderr,
        )
        return 1
    if args.action == "status":
        return await _status(args, conn, dimension)
    if args.action == "swap":
        return await _swap(args, conn, setting)
    return await _stage(args, conn)
