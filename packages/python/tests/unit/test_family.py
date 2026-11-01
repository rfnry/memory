import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[4]
SRC = "packages/python/src"

BLOCKS = {
    "parts and messages": (
        "rfnry_memory/models.py",
        "rfnry_engine/models.py",
        "class Text(BaseModel):",
        "    is_error: bool = False\n",
    ),
    "tools": (
        "rfnry_memory/tools.py",
        "rfnry_engine/tools/tools.py",
        "Result = ",
        "if n not in names})\n",
    ),
    "state": ("rfnry_memory/models.py", "rfnry_engine/models.py", "State = ", "list[Any]\n"),
    "generation and decision contract": (
        "rfnry_memory/models.py",
        "rfnry_engine/models.py",
        "class Usage(BaseModel):",
        "    ) -> ReplyLike: ...\n",
    ),
    "decision model": (
        "rfnry_memory/models.py",
        "rfnry_engine/models.py",
        "class DecisionModel(Protocol):",
        "| DecidedLike: ...\n",
    ),
}


def sibling(root: Path) -> Path:
    try:
        found = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        found = ""
    return (Path(found).parent if found else root).parent / "engine"


def cut(path: Path, start: str, end: str) -> str:
    text = path.read_text(encoding="utf-8")
    first = text.index(start)
    return text[first : text.index(end, first) + len(end)]


@pytest.mark.parametrize("name", BLOCKS)
def test_the_shared_block_is_byte_identical_with_the_sibling_library(name: str) -> None:
    own, theirs, start, end = BLOCKS[name]
    engine = sibling(ROOT)
    if not engine.is_dir():
        pytest.fail(
            f"tests: no rfnry-engine checkout at {engine}; clone it beside memory's checkout"
        )
    assert cut(ROOT / SRC / own, start, end) == cut(engine / SRC / theirs, start, end)


def test_a_worktree_finds_the_sibling_beside_its_main_checkout(tmp_path: Path) -> None:
    tmp_path = tmp_path.resolve()
    main = tmp_path / "memory"
    worktree = main / ".worktrees" / "quality"

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(main), *args], check=True, capture_output=True)

    main.mkdir()
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "t")
    git("worktree", "add", "-q", "--detach", str(worktree))
    assert sibling(worktree) == tmp_path / "engine"
    assert sibling(main) == tmp_path / "engine"
    assert sibling(tmp_path / "elsewhere") == tmp_path / "engine"
