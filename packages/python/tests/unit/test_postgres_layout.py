import ast
import re
from pathlib import Path

import rfnry_memory.postgres
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.postgres.base import Base
from rfnry_memory.postgres.records import _MOMENTS

PACKAGE = Path(rfnry_memory.postgres.__file__).parent
CONCERNS = {"sources", "stories", "search", "corpus"}


def _imported(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            node.module == "rfnry_memory.postgres" or (node.module is None and node.level)
        ):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.removeprefix("rfnry_memory.postgres."))
        elif isinstance(node, ast.Import):
            found |= {alias.name.removeprefix("rfnry_memory.postgres.") for alias in node.names}
    return found


def test_only_the_package_imports_sources_stories_search_and_corpus() -> None:
    wrong = [
        f"{path.name} imports {name}"
        for path in sorted(PACKAGE.glob("*.py"))
        if path.name != "__init__.py"
        for name in sorted(_imported(ast.parse(path.read_text())) & CONCERNS)
    ]
    assert not wrong, "\n".join(wrong)


def test_no_two_concerns_define_the_same_method() -> None:
    owners: dict[str, list[str]] = {}
    for cls in PostgresStorage.__mro__:
        if issubclass(cls, Base) and cls is not PostgresStorage:
            for name, value in vars(cls).items():
                if callable(value):
                    owners.setdefault(name, []).append(cls.__name__)
    assert {name: found for name, found in owners.items() if len(found) > 1} == {}


def test_a_relative_import_of_a_concern_is_seen() -> None:
    assert _imported(ast.parse("from . import sources")) == {"sources"}
    assert _imported(ast.parse("from .search import Search")) == {"search"}


SORTING = re.compile(r"\b(ORDER BY|DISTINCT ON)\b")
ENDING = re.compile(r"\b(LIMIT|OFFSET|FOR|FETCH)\b|;")
DIRECTION = re.compile(r"\s+(ASC|DESC|NULLS FIRST|NULLS LAST)$")


def _text(node: ast.expr) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    assert isinstance(node, ast.JoinedStr)
    return "".join(
        part.value
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
        else "{" + ast.unparse(part) + "}"
        for part in node.values
    )


def _strings(unit: ast.AST) -> list[ast.expr]:
    inner = {
        id(part)
        for node in ast.walk(unit)
        if isinstance(node, ast.JoinedStr)
        for part in node.values
    }
    found = [
        node
        for node in ast.walk(unit)
        if id(node) not in inner
        and (
            isinstance(node, ast.JoinedStr)
            or (isinstance(node, ast.Constant) and isinstance(node.value, str))
        )
    ]
    return sorted(found, key=lambda node: (node.lineno, node.col_offset))


def _keys(clause: str) -> list[str]:
    keys, depth, start = [], 0, 0
    for at, char in enumerate(clause + ","):
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                keys.append(clause[start:at])
                return keys
            depth -= 1
        elif char == "," and depth == 0:
            keys.append(clause[start:at])
            start = at + 1
    return keys


def _bare_sort_keys(tree: ast.Module) -> list[tuple[int, str]]:
    units = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        or (isinstance(node, ast.Assign) and node in tree.body)
    ]
    found = []
    for unit in units:
        strings = _strings(unit)
        if not any("_columns(" in _text(s) for s in strings):
            continue
        for string in strings:
            text = _text(string)
            for match in SORTING.finditer(text):
                before = text[: match.start()]
                opened = before.rfind("(")
                if opened > before.rfind(")") and "SELECT" not in before[opened:]:
                    continue
                clause = text[match.end() :].lstrip()
                if match.group(1) == "DISTINCT ON":
                    clause = clause.removeprefix("(")
                ended = ENDING.search(clause)
                for key in _keys(clause[: ended.start()] if ended else clause):
                    bare = DIRECTION.sub("", key.strip())
                    if bare in _MOMENTS:
                        found.append((string.lineno, bare))
    return found


def test_a_query_reading_moments_through_columns_sorts_by_the_table_column() -> None:
    wrong = [
        f"{path.name}:{line} sorts by the bare {name!r}, which names the clamped output column"
        for path in sorted(PACKAGE.glob("*.py"))
        for line, name in _bare_sort_keys(ast.parse(path.read_text()))
    ]
    assert not wrong, "\n".join(wrong)


def test_the_sort_key_check_tells_a_bare_moment_from_a_qualified_one_or_an_expression() -> None:
    def sorted_by(sql: str) -> list[str]:
        tree = ast.parse(f"def f():\n    return f\"SELECT {{_columns('story')}} {sql}\"")
        return [name for _, name in _bare_sort_keys(tree)]

    assert sorted_by("FROM s ORDER BY valid_from DESC, story_id LIMIT 3") == ["valid_from"]
    assert sorted_by("FROM s ORDER BY id, created_at NULLS LAST") == ["created_at"]
    assert sorted_by("FROM s ORDER BY recorded") == ["recorded"]
    assert sorted_by("FROM (SELECT DISTINCT ON (closed_at) closed_at FROM c) AS s") == ["closed_at"]
    assert sorted_by("FROM s ORDER BY memory_stories.valid_from DESC LIMIT 3") == []
    assert sorted_by("FROM s ORDER BY valid_from <= $1 DESC, s.valid_from") == []
    assert sorted_by("FROM s WHERE x = ANY(SELECT jsonb_agg(a ORDER BY created_at))") == []
    assert sorted_by("FROM s ORDER BY score DESC, id") == []
