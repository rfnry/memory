import shlex
from pathlib import Path

import pytest

import rfnry_memory
from rfnry_memory.cli import _parser

DOCS = Path(__file__).parents[4] / "docs" / "memory.md"


def shown_commands() -> list[str]:
    shown: set[str] = set()
    fenced = False
    for line in DOCS.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = line in ("```sh", "```bash")
        elif fenced and line.startswith("rfnry-memory "):
            shown.add(line)
    return sorted(shown)


def test_every_public_name_is_documented() -> None:
    text = DOCS.read_text(encoding="utf-8")
    assert [name for name in rfnry_memory.__all__ if name not in text] == []


@pytest.mark.parametrize("line", shown_commands())
def test_every_command_the_docs_show_is_one_the_cli_takes(line: str) -> None:
    _parser().parse_args(shlex.split(line, comments=True)[1:])
