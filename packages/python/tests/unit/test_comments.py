import ast
import io
import tokenize
from pathlib import Path

import rfnry_memory

SRC = Path(rfnry_memory.__file__).parent

_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def test_the_source_has_no_comments_and_no_docstrings() -> None:
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text()
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                found.append(f"{path.relative_to(SRC)}:{token.start[0]} {token.string}")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, _DOCUMENTED) and ast.get_docstring(node) is not None:
                found.append(f"{path.relative_to(SRC)}:{getattr(node, 'lineno', 1)} docstring")
    assert found == []
