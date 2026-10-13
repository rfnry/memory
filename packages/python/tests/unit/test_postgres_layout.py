import ast
from pathlib import Path

import rfnry_memory.postgres
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.postgres.base import Base

PACKAGE = Path(rfnry_memory.postgres.__file__).parent
CONCERNS = {"sources", "stories", "search", "corpus"}


def _imported(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "rfnry_memory.postgres":
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
