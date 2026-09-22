from pathlib import Path

import rfnry_memory

DOCS = Path(__file__).parents[4] / "docs" / "memory.md"


def test_every_public_name_is_documented() -> None:
    text = DOCS.read_text(encoding="utf-8")
    assert [name for name in rfnry_memory.__all__ if name not in text] == []
