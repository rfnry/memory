import re
from collections.abc import Callable
from typing import Literal


def _split_with_separator(
    text: str,
    separator: str,
    keep_separator: bool | Literal["start", "end"],
) -> list[tuple[int, int]]:
    if not separator:
        return [(i, i + 1) for i in range(len(text))]

    spans: list[tuple[int, int]] = []
    prev = 0
    for m in re.finditer(re.escape(separator), text):
        if keep_separator == "end":
            spans.append((prev, m.end()))
            prev = m.end()
        elif keep_separator:
            spans.append((prev, m.start()))
            prev = m.start()
        else:
            spans.append((prev, m.start()))
            prev = m.end()
    spans.append((prev, len(text)))
    return [(s, e) for s, e in spans if e > s]


class RecursiveTextSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = False,
        length_function: Callable[[str], int] = len,
        strip_whitespace: bool = True,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self._keep_separator = keep_separator
        self._length_function = length_function
        self._strip_whitespace = strip_whitespace

    def split_text(self, text: str) -> list[str]:
        return [chunk for chunk, _, _ in self.split_text_with_offsets(text)]

    def split_text_with_flags(self, text: str) -> list[tuple[str, bool]]:
        return [(chunk, hard) for chunk, hard, _ in self.split_text_with_offsets(text)]

    def split_text_with_offsets(self, text: str) -> list[tuple[str, bool, int]]:
        return self._split_flagged(text, self._separators, 0)

    def _strip_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        if not self._strip_whitespace:
            return start, end
        chunk = text[start:end]
        lead = len(chunk) - len(chunk.lstrip())
        trail = len(chunk) - len(chunk.rstrip())
        return start + lead, end - trail

    def _emit(
        self, text: str, start: int, end: int, hard: bool, base: int
    ) -> list[tuple[str, bool, int]]:
        s, e = self._strip_span(text, start, end)
        if e <= s:
            return []
        return [(text[s:e], hard, base + s)]

    def _merge_spans(
        self, spans: list[tuple[int, int]], separator: str, text: str
    ) -> list[tuple[int, int]]:
        joiner = "" if self._keep_separator else separator
        joiner_len = self._length_function(joiner)

        chunks: list[tuple[int, int]] = []
        current: list[tuple[int, int]] = []
        total = 0

        for span in spans:
            piece_len = self._length_function(text[span[0] : span[1]])
            candidate = total + piece_len + (joiner_len if current else 0)

            if candidate > self._chunk_size and current:
                chunks.append((current[0][0], current[-1][1]))

                while total > self._chunk_overlap or (
                    total + piece_len + (joiner_len if current else 0) > self._chunk_size
                    and total > 0
                ):
                    removed_len = self._length_function(text[current[0][0] : current[0][1]]) + (
                        joiner_len if len(current) > 1 else 0
                    )
                    total -= removed_len
                    current = current[1:]

            current.append(span)
            total += piece_len + (joiner_len if len(current) > 1 else 0)

        if current:
            chunks.append((current[0][0], current[-1][1]))

        return chunks

    def _split_flagged(
        self, text: str, separators: list[str], base: int
    ) -> list[tuple[str, bool, int]]:
        if not text:
            return []

        if self._length_function(text) <= self._chunk_size:
            return self._emit(text, 0, len(text), False, base)

        separator = separators[-1]
        remaining_separators: list[str] = []
        for i, sep in enumerate(separators):
            if not sep:
                separator = sep
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1 :]
                break

        is_char_level = separator == ""

        spans = _split_with_separator(text, separator, self._keep_separator)

        flagged: list[tuple[str, bool, int]] = []
        good_spans: list[tuple[int, int]] = []

        for span in spans:
            piece = text[span[0] : span[1]]
            if self._length_function(piece) < self._chunk_size:
                good_spans.append(span)
            else:
                if good_spans:
                    hard = is_char_level
                    for s, e in self._merge_spans(good_spans, separator, text):
                        flagged.extend(self._emit(text, s, e, hard, base))
                    good_spans = []

                if remaining_separators:
                    flagged.extend(self._split_flagged(piece, remaining_separators, base + span[0]))
                else:
                    if self._length_function(piece) > self._chunk_size:
                        step = max(1, self._chunk_size - self._chunk_overlap)
                        start = 0
                        while start < len(piece):
                            flagged.extend(
                                self._emit(
                                    piece,
                                    start,
                                    min(start + self._chunk_size, len(piece)),
                                    True,
                                    base + span[0],
                                )
                            )
                            start += step
                    else:
                        flagged.extend(self._emit(text, span[0], span[1], is_char_level, base))

        if good_spans:
            hard = is_char_level
            for s, e in self._merge_spans(good_spans, separator, text):
                flagged.extend(self._emit(text, s, e, hard, base))

        return flagged
