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


def siblings(root: Path) -> list[Path]:
    places = [root.parent / "engine"]
    try:
        common = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return places
    main = (root / common).resolve().parent.parent / "engine"
    return places if main in places else [*places, main]


def cut(path: Path, start: str, end: str) -> str:
    text = path.read_text(encoding="utf-8")
    first = text.index(start)
    return text[first : text.index(end, first) + len(end)]


@pytest.mark.parametrize("name", BLOCKS)
def test_the_shared_block_is_byte_identical_with_the_sibling_library(name: str) -> None:
    own, theirs, start, end = BLOCKS[name]
    looked = siblings(ROOT)
    sibling = next((place for place in looked if place.is_dir()), None)
    if sibling is None:
        pytest.skip(f"no sibling checkout at {' nor at '.join(map(str, looked))}")
    assert cut(ROOT / SRC / own, start, end) == cut(sibling / SRC / theirs, start, end)


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
    assert siblings(worktree) == [main / ".worktrees" / "engine", tmp_path / "engine"]
    assert siblings(main) == [tmp_path / "engine"]
    assert siblings(tmp_path / "elsewhere") == [tmp_path / "engine"]
