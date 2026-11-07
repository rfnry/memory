import time

import pytest

from rfnry_memory.chunking.chunker import chunk_text
from rfnry_memory.chunking.structure import HeadingSpan, build_heading_spans

DOC = (
    """# Pump manual

## Installation

Mount the pump on a level surface. """
    + "Tighten every bolt to the listed torque. " * 40
    + """

## Wiring

```
L1 -> terminal 1
L2 -> terminal 2
```

Connect the ground first.
"""
)


def test_chunks_carry_their_section_and_respect_size() -> None:
    pieces = chunk_text(DOC, chunk_chars=300, overlap_chars=40)
    assert pieces and all(p.text for p in pieces)
    titles = {p.title for p in pieces}
    assert "Pump manual > Installation" in titles and "Pump manual > Wiring" in titles
    fenced = [p for p in pieces if p.text.startswith("```")]
    assert len(fenced) == 1 and "terminal 2" in fenced[0].text
    prose = [p for p in pieces if not p.text.startswith("```")]
    assert all(len(p.text) <= 300 for p in prose)


def test_overlap_repeats_text_between_neighbours() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    pieces = chunk_text(text, chunk_chars=200, overlap_chars=60)
    assert len(pieces) > 2
    first, second = pieces[0].text, pieces[1].text
    assert first.split()[-1] in second.split()


def test_empty_and_small_texts() -> None:
    assert chunk_text("   ", chunk_chars=100, overlap_chars=10) == []
    [piece] = chunk_text("just a line", chunk_chars=100, overlap_chars=10)
    assert piece.title is None and piece.text == "just a line"


def test_an_edit_stops_at_the_next_heading() -> None:
    first = "\n\n".join(["a" * 24, "b" * 24, "c" * 24])
    second = "\n\n".join(["d" * 24, "e" * 24, "f" * 24])
    before = chunk_text(f"## A\n\n{first}\n\n## B\n\n{second}", chunk_chars=60, overlap_chars=0)
    edited = first.replace("a" * 24, "a" * 24 + " plus an added sentence here")
    after = chunk_text(f"## A\n\n{edited}\n\n## B\n\n{second}", chunk_chars=60, overlap_chars=0)
    assert [p.text for p in before if p.title == "B"] == [p.text for p in after if p.title == "B"]


def test_a_heading_starts_a_chunk_even_when_the_last_one_has_room() -> None:
    pieces = chunk_text("## A\n\nshort\n\n## B\n\nshort", chunk_chars=2000, overlap_chars=200)
    assert [p.title for p in pieces] == ["A", "B"]


def test_overlap_does_not_cross_a_heading() -> None:
    pieces = chunk_text("## A\n\n" + "x " * 60 + "\n\n## B\n\nyy", chunk_chars=60, overlap_chars=20)
    assert not any("x" in p.text for p in pieces if p.title == "B")


def test_a_heading_inside_a_code_fence_does_not_cut() -> None:
    pieces = chunk_text("```\n## not a heading\n```\n\nafter", chunk_chars=2000, overlap_chars=0)
    assert "## not a heading" in pieces[0].text


@pytest.mark.parametrize("marker", ["##", "## ", "##   ", "## \t", "## \r"])
def test_a_marker_with_no_title_reads_the_next_line_as_body(marker: str) -> None:
    text = f"# Manual\n\n{marker}\nbody line\n"
    [piece] = chunk_text(text, chunk_chars=2000, overlap_chars=0)
    assert piece.title == "Manual"
    assert build_heading_spans(text.strip()) == [
        HeadingSpan(start=0, end=len(text.strip()), path=("Manual",))
    ]


def test_a_heading_with_no_body_joins_the_next_range() -> None:
    pieces = chunk_text(
        "# Pump Manual v3\n\n## Introduction\n\nThis manual covers the pump.\n\n"
        "## Safety\n\nWear gloves.",
        chunk_chars=2000,
        overlap_chars=200,
    )
    assert [p.text for p in pieces] == [
        "# Pump Manual v3\n\n## Introduction\n\nThis manual covers the pump.",
        "## Safety\n\nWear gloves.",
    ]
    assert [p.title for p in pieces] == ["Pump Manual v3", "Pump Manual v3 > Safety"]
    [joined] = chunk_text("# T\n## S\ntext", chunk_chars=2000, overlap_chars=200)
    assert joined.text == "# T\n## S\ntext"


def test_a_chain_of_headings_joins_the_body_after_it() -> None:
    [piece] = chunk_text("# A\n## B\n### C\ntext", chunk_chars=2000, overlap_chars=200)
    assert piece.text == "# A\n## B\n### C\ntext"
    assert piece.title == "A"


def test_a_heading_with_no_body_before_a_fence_stays_alone() -> None:
    pieces = chunk_text(
        "## Wiring\n\n```\nL1 -> terminal 1\n```\n\nConnect the ground first.",
        chunk_chars=2000,
        overlap_chars=200,
    )
    assert [p.text for p in pieces] == [
        "## Wiring",
        "```\nL1 -> terminal 1\n```",
        "Connect the ground first.",
    ]


def test_a_document_of_only_headings_still_has_chunks() -> None:
    pieces = chunk_text("# A\n## B\n### C", chunk_chars=2000, overlap_chars=200)
    assert [p.text for p in pieces] == ["# A\n## B\n### C"]


def test_an_edit_to_a_later_section_leaves_the_title_chunk_as_it_was() -> None:
    start = "# Manual\n\n## Intro\n\nCovers the pump.\n\n## Safety\n\n"
    before = chunk_text(start + "Wear gloves.", chunk_chars=2000, overlap_chars=200)
    after = chunk_text(start + "Wear gloves and goggles.", chunk_chars=2000, overlap_chars=200)
    assert before[0] == after[0]
    assert before[0].text == "# Manual\n\n## Intro\n\nCovers the pump."
    assert before[1] != after[1]


def test_a_long_chain_of_headings_chunks_in_linear_time() -> None:
    text = "\n".join(f"# H{i}" for i in range(8000)) + "\ntext"
    started = time.monotonic()
    [piece] = chunk_text(text, chunk_chars=10_000_000, overlap_chars=0)
    assert time.monotonic() - started < 3
    assert piece.text.endswith("text")


def test_text_after_a_fence_on_its_closing_line_is_not_a_heading() -> None:
    pieces = chunk_text(
        "intro\n\n```\ncode\n```# tail\n## Next\n\nbody", chunk_chars=2000, overlap_chars=0
    )
    assert [p.text for p in pieces] == [
        "intro",
        "```\ncode\n```",
        "# tail",
        "## Next\n\nbody",
    ]
