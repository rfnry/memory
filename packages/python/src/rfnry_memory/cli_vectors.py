import argparse
import asyncio
import json
import os
import shlex
import sys
import time
from collections import Counter
from importlib import import_module
from inspect import isawaitable
from typing import Any, Literal, TypeGuard

import asyncpg

from rfnry_memory.cli import _confirmed, _scrubbed
from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError
from rfnry_memory.events import Embedded, Event
from rfnry_memory.filters import validate_filters
from rfnry_memory.memory import Memory
from rfnry_memory.models import EmbeddingModel, Ref
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.postgres.schema import schema_problems
from rfnry_memory.postgres.switch import (
    _FORGET_BATCH,
    CorpusVectors,
    IndexPlan,
    SwapPlan,
    UnknownCommitError,
    _agreeing,
    _index_text,
    apply_swap,
    forget_staged,
    hold_staging,
    labeled,
    named_scope,
    plan_index,
    plan_swap,
    rename_model,
    staged_of,
    vector_status,
)
from rfnry_memory.postgres.vectors import _HALF_DIMENSIONS

_EVERY = 5.0


_SHOWN = 20


_SAID = 500


_SWAPPING = "one transaction: every vector moves or none does, and a stop changes nothing"


_RENAMING = "one transaction: every label moves or none does, and a stop changes nothing"


_BUILDING = "an index takes real time at scale; searches run exactly until it is built"


_HELD_UNBUILT = 3


_COMMIT_UNKNOWN = 4


_ASIDE = "aside"


_FORGET_KEPT = "what was deleted stays deleted; running it again finishes"


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
    target = os.path.realpath(path)
    parent = os.path.dirname(target)
    writable = (
        os.access(target, os.W_OK) and not os.path.isdir(target)
        if os.path.exists(target)
        else os.path.isdir(parent) and os.access(parent, os.W_OK | os.X_OK)
    )
    if not writable:
        raise InputError(f"cli: --refused cannot be written: {path!r}")


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
    named, scope = named_scope(corpora, args.corpus)
    return f"in {named}", scope


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
    forget = _command(args, "forget", "--model", model, corpora=args.corpus)
    lacking = [row.corpus_id for row in rows if row.missing[model]]
    if lacking:
        where, scope = _where(args, lacking)
        run = _command(args, "stage", "--embedding", "MODULE:NAME", corpora=scope)
        count = sum(row.missing[model] for row in rows)
        steps.append(f"{model}: {count:,} missing, {where} — run {run}")
    elif any(row.on_row.get(model, 0) < row.passages for row in rows):
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
    elif staged := sum(row.staged.get(model, 0) for row in rows):
        steps.append(
            f"{model}: on every row, {staged:,} still staged — run {forget} "
            "to drop its staged vectors"
        )
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


def _pool(args: argparse.Namespace, search_path: str) -> Any:
    return asyncpg.create_pool(
        args.url,
        min_size=1,
        max_size=2,
        server_settings={"search_path": search_path, "statement_timeout": "0"},
    )


async def _stage(args: argparse.Namespace, conn: Any) -> int:
    filters = _parsed(args.filters)
    corpora = await _corpora(args, conn)
    _check_writable(args.refused)
    embedding = await load_embedding(args.embedding, args.url)
    model = embedding.model
    await hold_staging(conn, model)
    search_path = await conn.fetchval("SHOW search_path")
    progress = _Progress()
    refused: list[tuple[str, Ref]] = []
    stopped = 0
    try:
        async with _pool(args, search_path) as pool:
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
    _list_refused(model, refused, args.refused if refused or not stopped else None)
    if stopped:
        return stopped
    if refused:
        return 1
    print(f"next: {_command(args, 'status', '--model', model, corpora=args.corpus)}")
    return 0


def _unbuilt(args: argparse.Namespace, corpora: list[str], held: str) -> str:
    where, scope = _where(args, corpora)
    run = _command(args, "index", corpora=scope)
    return (
        f"{held}the index {where} is still to build, its searches run exactly, "
        f"and {run} finishes it"
    )


def _run_as(args: argparse.Namespace, owner: str, corpora: list[str]) -> tuple[str, str]:
    where, scope = _where(args, corpora)
    return where, f"{_command(args, 'index', '--url', 'URL', corpora=scope)} as {owner}"


def _refuse_others(args: argparse.Namespace, others: dict[str, list[str]]) -> None:
    for owner, corpora in sorted(others.items()):
        where, run = _run_as(args, owner, corpora)
        print(
            f"cli: only {owner} builds the index {where}, so its table keeps its owner — run {run}",
            file=sys.stderr,
        )


def _owed(args: argparse.Namespace, plan: IndexPlan, left_alone: bool) -> list[str]:
    binary = plan.unfit if left_alone else []
    fitting = [c for c in plan.builds if c not in binary] if left_alone else []
    return [
        *([_command(args, "index", corpora=_where(args, fitting)[1])] if fitting else []),
        *(
            [_command(args, "index", "--precision", "binary", corpora=_where(args, binary)[1])]
            if binary
            else []
        ),
        *(_run_as(args, owner, corpora)[1] for owner, corpora in sorted(plan.others.items())),
    ]


async def _build_recorded(
    pool: Any,
    storage: PostgresStorage,
    corpus_id: str,
    precision: Literal["full", "half", "binary"] | None,
) -> None:
    settings = (await plan_index(pool, [corpus_id], precision)).builds.get(corpus_id)
    if settings is None:
        print(f"{corpus_id}: nothing recorded to build any more", flush=True)
        return
    print(f"{corpus_id}: building {_index_text(settings)}", flush=True)
    try:
        count = await storage.set_vector_index(
            corpus_id, settings, rebuild=False, only_recorded=True
        )
    except NotFoundError:
        print(f"{corpus_id}: nothing recorded to build any more", flush=True)
        return
    print(f"{corpus_id}: built, {_counted(count, 'vector')}", flush=True)


async def _build(
    args: argparse.Namespace,
    conn: Any,
    corpora: list[str],
    *,
    precision: Literal["full", "half", "binary"] | None = None,
    held: str = "",
    left_alone: bool = False,
) -> int:
    failed = _HELD_UNBUILT if held else 1
    stopped = _HELD_UNBUILT if held else 130
    left = corpora
    owed: list[str] = []
    code = 0
    try:
        plan = await plan_index(conn, corpora, precision)
        if held:
            owed = _owed(args, plan, left_alone)
        else:
            _refuse_others(args, plan.others)
            code = 1 if plan.others else 0
        if plan.unfit and not left_alone:
            where, scope = _where(args, plan.unfit)
            run = _command(args, "index", "--precision", "binary", corpora=scope)
            print(
                f"cli: {held}the index {where} reaches {_HALF_DIMENSIONS} dimensions and this "
                f"schema holds {plan.dimension} — run {run}",
                file=sys.stderr,
            )
            code = failed
        elif plan.builds and not left_alone:
            left = list(plan.builds)
            print(f"schema {args.schema}: {_BUILDING}", flush=True)
            async with _pool(args, await conn.fetchval("SHOW search_path")) as pool:
                storage = PostgresStorage(pool)
                await storage.check(dimension=None, graph=False)
                for corpus_id in plan.builds:
                    await _build_recorded(pool, storage, corpus_id, precision)
                    left.remove(corpus_id)
        elif not (plan.others or held):
            print(f"schema {args.schema}: no index to build")
    except KeyboardInterrupt, asyncio.CancelledError:
        print(f"cli: stopped — {_unbuilt(args, left, held)}", file=sys.stderr)
        code = stopped
    except Exception as exc:
        print(
            f"cli: stopped by {_said(exc, args.url)} — {_unbuilt(args, left, held)}",
            file=sys.stderr,
        )
        code = failed
    if held and not (owed or code):
        owed = [_command(args, "status", "--model", args.model, corpora=args.corpus)]
    for step in owed:
        print(f"next: {step}")
    return code


async def _swapped(args: argparse.Namespace, conn: Any, setting: str) -> SwapPlan | int:
    corpora = await _corpora(args, conn)
    shown = await plan_swap(
        conn,
        model=args.model,
        corpus_ids=None if args.corpus is None else corpora,
        allow_missing=args.allow_missing,
        no_index=args.no_index,
    )
    for refusal in shown.refusals:
        print(_scrubbed(f"cli: {refusal}", args.url), file=sys.stderr)
    if shown.refusals:
        return 1
    if args.sql:
        print(f"{setting};")
        print("BEGIN ISOLATION LEVEL READ COMMITTED;")
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
    return shown


async def _swap(args: argparse.Namespace, conn: Any, setting: str) -> int:
    try:
        moved = await _swapped(args, conn, setting)
    except KeyboardInterrupt, asyncio.CancelledError:
        print("cli: stopped — the swap rolled back, nothing moved", file=sys.stderr)
        return 130
    except UnknownCommitError as exc:
        print(exc, file=sys.stderr)
        return _COMMIT_UNKNOWN
    if not isinstance(moved, SwapPlan):
        return moved
    held = f"the swap held and {args.model} is on the rows; "
    return await _build(args, conn, moved.corpora, held=held, left_alone=args.no_index)


async def _forget(args: argparse.Namespace, conn: Any) -> int:
    corpora = await _corpora(args, conn)
    model = args.model
    staged = await staged_of(conn, model, corpora)
    if not staged:
        print(f"schema {args.schema}: nothing staged for {model}")
        return 0
    where, _ = _where(args, list(staged))
    count = sum(staged.values())
    print(
        f"  {_counted(count, 'staged vector')} of {model} {_agreeing(count, 'goes', 'go')}, {where}"
    )
    if not _confirmed(
        args, f"what goes is gone: moving to {model} after this means staging it again"
    ):
        print("nothing was run")
        return 0
    print(f"schema {args.schema}: deleting in batches of {_FORGET_BATCH:,}, each committed")
    try:
        deleted = await forget_staged(conn, model=model, corpus_ids=corpora)
    except ConfigurationError:
        raise
    except KeyboardInterrupt, asyncio.CancelledError:
        print(f"cli: stopped — {_FORGET_KEPT}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"cli: stopped by {_said(exc, args.url)} — {_FORGET_KEPT}", file=sys.stderr)
        return 1
    print(f"schema {args.schema}: {_counted(deleted, 'staged vector')} of {model} deleted")
    print(f"a rollback to {model} now means staging it again")
    return 0


async def _untaken_aside(conn: Any, corpora: list[str]) -> str:
    label, tried = _ASIDE, 1
    while await labeled(conn, label, corpora):
        tried += 1
        label = f"{_ASIDE}-{tried}"
    return label


def _name_taken(
    args: argparse.Namespace, labels: Counter[str], staged: dict[str, int], set_aside: str
) -> str:
    model, to = args.model, args.to
    where, _ = _where(args, sorted(labels))
    waiting = sum(staged.values())
    on_rows = labels.total() - waiting
    if not on_rows:
        forget = _command(args, "forget", "--model", to, corpora=args.corpus)
        return (
            f"cli: {to} already labels {_counted(waiting, 'staged vector')}, {where} — "
            f"give another name, or drop them with {forget}"
        )
    aside = _command(args, "rename", "--model", to, "--to", set_aside, corpora=args.corpus)
    again = _command(args, "rename", "--model", model, "--to", to, corpora=args.corpus)
    also = f" and {waiting:,} staged" if waiting else ""
    return (
        f"cli: {to} already labels {_counted(on_rows, 'vector')} on rows{also}, {where}, and "
        "vectors forget clears only what is staged — give another name, or hold writes and run "
        f"{aside}, then {again}, then reembed under the new adapter, which writes only the rows "
        "set aside"
    )


async def _rename(args: argparse.Namespace, conn: Any) -> int:
    corpora = await _corpora(args, conn)
    model, to = args.model, args.to
    held = await labeled(conn, model, corpora)
    if not held:
        check = _command(args, "status", corpora=args.corpus)
        refusal = f"cli: no vector here is labeled {model} — check the name with {check}"
        print(_scrubbed(refusal, args.url), file=sys.stderr)
        return 1
    taken = await labeled(conn, to, corpora)
    if taken:
        staged = await staged_of(conn, to, corpora)
        set_aside = await _untaken_aside(conn, corpora)
        print(_scrubbed(_name_taken(args, taken, staged, set_aside), args.url), file=sys.stderr)
        return 1
    where, _ = _where(args, sorted(held))
    take = _agreeing(held.total(), "takes", "take")
    print(
        f"  {_counted(held.total(), 'vector')} labeled {model} {take} the label {to}, {where}; "
        "no vector changes"
    )
    print(
        "  writes wait until the rename ends; reads go on; "
        f"after it a Memory of {model} finds nothing by vector there"
    )
    if not _confirmed(args, _RENAMING):
        print("nothing was run")
        return 0
    try:
        renamed = await rename_model(conn, model=model, to=to, corpus_ids=corpora)
    except KeyboardInterrupt, asyncio.CancelledError:
        print("cli: stopped — the rename rolled back, nothing changed", file=sys.stderr)
        return 130
    except UnknownCommitError as exc:
        print(exc, file=sys.stderr)
        return _COMMIT_UNKNOWN
    print(
        f"schema {args.schema}: {_counted(renamed, 'vector')} of {model} "
        f"{_agreeing(renamed, 'is', 'are')} labeled {to}"
    )
    print(f"next: {_command(args, 'status', '--model', to, corpora=args.corpus)}")
    return 0


async def run(args: argparse.Namespace, conn: Any, setting: str) -> int:
    await conn.execute("SET statement_timeout = 0")
    problems, dimension = await schema_problems(conn, dimension=None, graph=False)
    for problem in problems:
        print(f"cli: {problem}", file=sys.stderr)
    if problems:
        return 1
    if dimension is None:
        raise InputError(
            f"cli: schema {args.schema} holds no vectors — "
            f"run rfnry-memory set --dimension N --schema {args.schema}"
        )
    if args.action == "status":
        return await _status(args, conn, dimension)
    if args.action == "swap":
        return await _swap(args, conn, setting)
    if args.action == "index":
        return await _build(args, conn, await _corpora(args, conn), precision=args.precision)
    if args.action == "forget":
        return await _forget(args, conn)
    if args.action == "rename":
        return await _rename(args, conn)
    return await _stage(args, conn)
