from __future__ import annotations

from pydantic import BaseModel

from rfnry_memory.chunking.splitter import RecursiveTextSplitter
from rfnry_memory.chunking.structure import (
    build_heading_spans,
    find_atomic_regions,
    section_path_at,
)


class Piece(BaseModel):
    title: str | None
    text: str


def chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[Piece]:
    text = text.strip()
    if not text:
        return []
    regions = find_atomic_regions(text)
    spans = build_heading_spans(text, regions)
    splitter = RecursiveTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap_chars)
    pieces: list[Piece] = []

    def free(start: int, end: int) -> None:
        for piece, _, offset in splitter.split_text_with_offsets(text[start:end]):
            pieces.append(Piece(title=section_path_at(start + offset, spans), text=piece))

    cursor = 0
    for region in regions:
        free(cursor, region.start)
        pieces.append(Piece(title=section_path_at(region.start, spans), text=region.content.strip()))
        cursor = region.end
    free(cursor, len(text))
    return [piece for piece in pieces if piece.text]
