import __future__

import ast
import dataclasses
import importlib
import inspect
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import rfnry_memory

API = Path(__file__).parents[4] / "docs" / "api.md"

Def = ast.FunctionDef | ast.AsyncFunctionDef


def entries() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for section in re.split(r"^### ", API.read_text(encoding="utf-8"), flags=re.M)[1:]:
        heading, _, body = section.partition("\n")
        block = re.search(r"^```python\n(.*?)^```$", body, flags=re.M | re.S)
        found.append((heading.strip("`"), block[1] if block else ""))
    return found


def is_protocol(target: object) -> bool:
    return inspect.isclass(target) and bool(getattr(target, "_is_protocol", False))


def verbs(cls: type) -> list[str]:
    return [n for n, v in vars(cls).items() if not n.startswith("_") and inspect.isfunction(v)]


def members(protocol: type) -> list[str]:
    return [
        *inspect.get_annotations(protocol),
        *(
            n
            for n, v in vars(protocol).items()
            if not n.startswith("_") and (inspect.isfunction(v) or isinstance(v, property))
        ),
    ]


def exported() -> dict[str, Any]:
    return {name: getattr(rfnry_memory, name) for name in rfnry_memory.__all__}


def core_classes() -> dict[str, type]:
    return {
        name: target
        for name, target in exported().items()
        if inspect.isclass(target)
        and not issubclass(target, BaseModel | BaseException)
        and not is_protocol(target)
    }


def implemented(cls: type) -> dict[str, type]:
    return {
        member: protocol
        for protocol in exported().values()
        if is_protocol(protocol) and set(members(protocol)) <= set(vars(cls))
        for member in members(protocol)
    }


def public(name: str) -> Any:
    owner, _, verb = name.partition(".")
    target = exported().get(owner)
    if not verb or target is None:
        return target
    return getattr(target, verb) if inspect.isclass(target) and verb in verbs(target) else None


def documented(node: Def) -> Any:
    node.decorator_list = []
    namespace: dict[str, Any] = {}
    module = ast.Module(body=[node], type_ignores=[])
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(API), "exec", flags=flags, dont_inherit=True), namespace)
    return namespace[node.name]


def same_signature(node: Def, code: Any) -> bool:
    if isinstance(code, property):
        code = code.fget
    if not callable(code) or isinstance(node, ast.AsyncFunctionDef) != (
        inspect.iscoroutinefunction(code)
    ):
        return False
    return inspect.signature(documented(node)) == inspect.signature(code)


def assigned(name: str) -> str:
    imports = ast.parse(inspect.getsource(rfnry_memory)).body
    module = next(
        node.module
        for node in imports
        if isinstance(node, ast.ImportFrom) and any(alias.name == name for alias in node.names)
    )
    tree = ast.parse(inspect.getsource(importlib.import_module(str(module))))
    return next(
        ast.unparse(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == [name]
    )


def declares(node: ast.stmt) -> str | None:
    if isinstance(node, Def | ast.ClassDef):
        return node.name
    return ast.unparse(node.targets[0]) if isinstance(node, ast.Assign) else None


def mismatches(name: str, block: str) -> list[str]:
    target = public(name)
    body = ast.parse(block).body
    if [declares(node) for node in body] != [name.rpartition(".")[2]]:
        return [name]
    node = body[0]
    if isinstance(node, Def):
        return [] if same_signature(node, target) else [name]
    if isinstance(node, ast.Assign):
        return [] if ast.unparse(node.value) == assigned(name) else [name]
    return [
        f"{name}.{member.name}"
        for member in getattr(node, "body", [])
        if isinstance(member, Def)
        and not same_signature(member, inspect.getattr_static(target, member.name, None))
    ]


def declared(cls: type) -> list[str] | None:
    if issubclass(cls, BaseModel):
        return list(cls.model_fields)
    if dataclasses.is_dataclass(cls):
        return [field.name for field in dataclasses.fields(cls)]
    if issubclass(cls, tuple):
        return list(getattr(cls, "_fields", ()))
    if is_protocol(cls):
        return members(cls)
    return None


def listed(block: str, protocol: bool) -> list[str]:
    node = next(node for node in ast.parse(block).body if isinstance(node, ast.ClassDef))
    fields = [ast.unparse(n.target) for n in node.body if isinstance(n, ast.AnnAssign)]
    methods = [n.name for n in node.body if isinstance(n, Def)]
    return fields + methods if protocol else fields


def test_every_public_name_has_one_entry_and_every_entry_names_something_public() -> None:
    names = [name for name, _ in entries()]
    assert [name for name in rfnry_memory.__all__ if names.count(name) != 1] == []
    assert [name for name in names if public(name) is None or names.count(name) != 1] == []
    assert [name for name, block in entries() if not block] == []


def test_every_verb_of_a_core_class_has_its_own_entry() -> None:
    names = {name for name, _ in entries()}
    missing = [
        f"{name}.{verb}"
        for name, cls in core_classes().items()
        for verb in verbs(cls)
        if verb not in implemented(cls) and f"{name}.{verb}" not in names
    ]
    assert missing == []


def test_a_protocol_documents_the_verbs_of_what_implements_it() -> None:
    differing = [
        f"{name}.{verb}"
        for name, cls in core_classes().items()
        for verb, protocol in implemented(cls).items()
        if verb in verbs(cls)
        and inspect.signature(getattr(cls, verb)) != inspect.signature(getattr(protocol, verb))
    ]
    assert differing == []


def test_every_signature_in_the_reference_matches_the_code() -> None:
    assert [found for name, block in entries() for found in mismatches(name, block)] == []


def test_every_model_record_and_protocol_lists_exactly_its_fields() -> None:
    differing = [
        name
        for name, block in entries()
        if inspect.isclass(target := public(name))
        and (fields := declared(target)) is not None
        and listed(block, is_protocol(target)) != fields
    ]
    assert differing == []
