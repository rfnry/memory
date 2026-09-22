import ast
from pathlib import Path

import rfnry_memory

SRC = Path(rfnry_memory.__file__).parent
LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
ALLOWED = {
    ("tools.py", "_error(type(exc).__name__, str(exc))"): "relays a MemoryException whose own message already carries its module prefix",
}


def _sources() -> list[tuple[Path, ast.Module]]:
    return [(path, ast.parse(path.read_text(), filename=str(path))) for path in sorted(SRC.rglob("*.py"))]


def _literal(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        text = ""
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                text += part.value
            else:
                return text + "{"
        return text
    return None


def _prefixed(message: str, stem: str) -> bool:
    prefix = f"{stem}: "
    return message.startswith(prefix) and len(message) > len(prefix) and not message[len(prefix)].isupper()


def _messages(tree: ast.Module) -> list[tuple[ast.expr, ast.expr | None]]:
    found: list[tuple[ast.expr, ast.expr | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            found.append((node.exc, node.exc.args[0] if node.exc.args else None))
        elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Name) and node.exc.id[:1].isupper():
            found.append((node.exc, None))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_error":
            found.append((node, node.args[1] if len(node.args) > 1 else None))
    return found


def test_every_error_message_starts_with_its_module_name() -> None:
    wrong = []
    for path, tree in _sources():
        for node, first in _messages(tree):
            text = None if first is None else _literal(first)
            if text is None:
                if (path.name, ast.unparse(node)) not in ALLOWED:
                    wrong.append(f"{path.relative_to(SRC)}:{node.lineno} has no literal message: {ast.unparse(node)}")
            elif not _prefixed(text, path.stem):
                wrong.append(f"{path.relative_to(SRC)}:{node.lineno} {text!r}")
    assert not wrong, "\n".join(wrong)


def test_every_log_message_is_lowercase() -> None:
    wrong = []
    for path, tree in _sources():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in LOG_METHODS):
                continue
            owner = ast.unparse(node.func.value).lower()
            if "log" not in owner:
                continue
            args = node.args[1:] if node.func.attr == "log" else node.args
            text = _literal(args[0]) if args else None
            if text is None or text[:1].isupper():
                wrong.append(f"{path.relative_to(SRC)}:{node.lineno} {ast.unparse(node)}")
    assert not wrong, "\n".join(wrong)


def test_the_allowlist_names_only_what_exists() -> None:
    present = {(path.name, ast.unparse(node)) for path, tree in _sources() for node, _ in _messages(tree)}
    assert set(ALLOWED) <= present
