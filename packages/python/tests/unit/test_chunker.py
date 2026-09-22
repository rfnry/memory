from rfnry_memory.chunking.chunker import chunk_text

DOC = """# Pump manual

## Installation

Mount the pump on a level surface. """ + "Tighten every bolt to the listed torque. " * 40 + """

## Wiring

```
L1 -> terminal 1
L2 -> terminal 2
```

Connect the ground first.
"""


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
