from pathlib import Path

import pytest

ROOT = Path(__file__).parents[4]
SIBLING = ROOT.parent / "engine"
SRC = "packages/python/src"

BLOCKS = {
    "parts and messages": ("rfnry_memory/models.py", "rfnry_engine/models.py", "class Text(BaseModel):", "    is_error: bool = False\n"),
    "tools": ("rfnry_memory/tools.py", "rfnry_engine/tools/tools.py", "Result = ", "if n not in names})\n"),
}


def cut(path: Path, start: str, end: str) -> str:
    text = path.read_text(encoding="utf-8")
    first = text.index(start)
    return text[first : text.index(end, first) + len(end)]


@pytest.mark.parametrize("name", BLOCKS)
def test_the_shared_block_is_byte_identical_with_the_sibling_library(name: str) -> None:
    own, theirs, start, end = BLOCKS[name]
    sibling = SIBLING / SRC / theirs
    if not sibling.exists():
        pytest.skip(f"no sibling checkout at {SIBLING}")
    assert cut(ROOT / SRC / own, start, end) == cut(sibling, start, end)
