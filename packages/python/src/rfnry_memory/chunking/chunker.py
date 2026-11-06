from pydantic import BaseModel

from rfnry_memory.chunking.splitter import RecursiveTextSplitter
from rfnry_memory.chunking.structure import (
    build_heading_spans,
    find_atomic_regions,
    holds_body,
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
        cuts = [start]
        seen = start
        body_since_cut = False
        for span in spans:
            if not start < span.start < end:
                continue
            body_since_cut = body_since_cut or holds_body(text, seen, span.start)
            seen = span.start
            if body_since_cut:
                cuts.append(span.start)
                body_since_cut = False
        for range_start, range_end in zip(cuts, [*cuts[1:], end], strict=True):
            for piece, offset in splitter.split_text_with_offsets(text[range_start:range_end]):
                title = section_path_at(range_start + offset, spans)
                pieces.append(Piece(title=title, text=piece))

    cursor = 0
    for region in regions:
        free(cursor, region.start)
        pieces.append(
            Piece(title=section_path_at(region.start, spans), text=region.content.strip())
        )
        cursor = region.end
    free(cursor, len(text))
    return [piece for piece in pieces if piece.text]
