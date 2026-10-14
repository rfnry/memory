from __future__ import annotations

import argparse
import asyncio
import sys
from functools import partial
from importlib.util import find_spec
from typing import Any
from urllib.parse import urlsplit

from rfnry_memory.errors import MemoryException

_WARNING = "take a snapshot of the database and stop app instances running the old version first"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfnry-memory", description="prepare and upgrade the database rfnry-memory runs on"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, summary: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.add_argument("--url", help="postgres connection string; without it, the PG* variables")
        sub.add_argument("--schema", default="memory", help="the postgres schema (default: memory)")
        return sub

    create = command("create", "create the schema with this version's tables")
    vectors = create.add_mutually_exclusive_group(required=True)
    vectors.add_argument("--dimension", type=int, help="the embedding dimension")
    vectors.add_argument("--no-vector", action="store_true", help="create no vector columns")
    create.add_argument("--no-graph", action="store_true", help="do not install pg_trgm")
    create.add_argument("--language", help="the default text search configuration of new corpora")
    create.add_argument("--sql", action="store_true", help="print the statements and run nothing")
    command("check", "say whether the schema fits this version")
    upgrade = command("upgrade", "carry the schema to this version")
    upgrade.add_argument("--sql", action="store_true", help="print the statements and run nothing")
    upgrade.add_argument("--yes", action="store_true", help="run without asking")
    change = command("set", "change a setting of the schema")
    setting = change.add_mutually_exclusive_group(required=True)
    setting.add_argument("--dimension", type=int, help="the embedding dimension")
    setting.add_argument("--graph", action="store_true", help="install pg_trgm")
    setting.add_argument("--language", help="the default text search configuration of new corpora")
    change.add_argument("--sql", action="store_true", help="print the statements and run nothing")
    change.add_argument("--yes", action="store_true", help="run without asking")
    return parser


def _scrubbed(text: str, url: str | None) -> str:
    if not url:
        return text
    password = urlsplit(url).password
    text = text.replace(url, "<url>")
    return text.replace(f":{password}@", ":***@") if password else text


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


async def _connect(args: argparse.Namespace) -> tuple[Any, str]:
    import asyncpg

    conn = await asyncpg.connect(args.url)
    held = await conn.fetch(
        "SELECT DISTINCT quote_ident(n.nspname) AS name FROM pg_extension e "
        "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname IN ('vector', 'pg_trgm') "
        "AND n.nspname NOT IN ($1, 'public') ORDER BY 1",
        args.schema,
    )
    path = [_quoted(args.schema), "public", *(r["name"] for r in held)]
    setting = f"SET search_path TO {', '.join(path)}"
    await conn.execute(setting)
    return conn, setting


def _confirmed(args: argparse.Namespace) -> bool:
    if args.yes:
        return True
    print(_WARNING)
    return input("run these steps? [y/N] ").strip().lower() in ("y", "yes")


async def _run(args: argparse.Namespace) -> int:
    from rfnry_memory.postgres import migrations
    from rfnry_memory.postgres.schema import _problems

    conn, setting = await _connect(args)
    try:
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)", args.schema
        )
        if not exists and args.command != "create":
            print(
                f"cli: there is no schema {args.schema} — run rfnry-memory create --schema {args.schema}",
                file=sys.stderr,
            )
            return 1
        if args.command == "check":
            problems, held = await _problems(conn, dimension=None, graph=False)
            if problems:
                for problem in problems:
                    print(f"cli: {problem}", file=sys.stderr)
                return 1
            vectors = "no vectors" if held is None else f"vectors of {held} dimensions"
            print(f"schema {args.schema} fits rfnry-memory: version {migrations.LATEST}, {vectors}")
            return 0
        before: list[str] = []
        planner: migrations.Planner
        if args.command == "create":
            before = [f"CREATE SCHEMA IF NOT EXISTS {_quoted(args.schema)}"]
            planner = partial(
                migrations.plan_create,
                dimension=None if args.no_vector else args.dimension,
                graph=not args.no_graph,
                language=args.language,
            )
        elif args.command == "upgrade":
            planner = migrations.plan_upgrade
        else:
            planner = partial(
                migrations.plan_set,
                dimension=args.dimension,
                graph=args.graph,
                language=args.language,
            )
        shown = await migrations.apply(conn, planner, run=False, before=before)
        if args.sql:
            print(f"{setting};")
            for statement in shown.statements:
                print(f"{statement};")
            return 0
        if not shown.steps:
            print(f"schema {args.schema}: nothing to do")
            return 0
        for step in shown.steps:
            print(f"  {step}")
        if args.command != "create" and not _confirmed(args):
            print("nothing was run")
            return 0
        await migrations.apply(conn, planner, expected=shown, before=before)
        print(f"schema {args.schema} is at version {migrations.LATEST}")
        return 0
    finally:
        await conn.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.schema in ("public", "information_schema") or args.schema.startswith("pg_"):
        print(
            f"cli: memory is not created in {args.schema!r} — name a schema of its own",
            file=sys.stderr,
        )
        return 1
    if args.command in ("create", "set") and args.dimension is not None and args.dimension < 1:
        print("cli: a dimension is a positive int", file=sys.stderr)
        return 1
    if find_spec("asyncpg") is None:
        print("cli: install rfnry-memory[postgres] to use this command", file=sys.stderr)
        return 1
    try:
        return asyncio.run(_run(args))
    except MemoryException as exc:
        print(_scrubbed(str(exc), args.url), file=sys.stderr)
        return 1
    except Exception as exc:
        print(_scrubbed(f"cli: {type(exc).__name__}: {exc}", args.url), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
