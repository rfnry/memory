import re

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split(text: str, separator: str) -> list[tuple[int, int]]:
    if not separator:
        return [(i, i + 1) for i in range(len(text))]
    spans: list[tuple[int, int]] = []
    prev = 0
    for m in re.finditer(re.escape(separator), text):
        spans.append((prev, m.start()))
        prev = m.end()
    spans.append((prev, len(text)))
    return [(s, e) for s, e in spans if e > s]


class RecursiveTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split_text_with_offsets(self, text: str) -> list[tuple[str, int]]:
        return self._split(text, SEPARATORS, 0)

    def _emit(self, text: str, start: int, end: int, base: int) -> list[tuple[str, int]]:
        chunk = text[start:end]
        s = start + len(chunk) - len(chunk.lstrip())
        e = end - (len(chunk) - len(chunk.rstrip()))
        if e <= s:
            return []
        return [(text[s:e], base + s)]

    def _merge(self, spans: list[tuple[int, int]], separator: str) -> list[tuple[int, int]]:
        joiner = len(separator)
        chunks: list[tuple[int, int]] = []
        current: list[tuple[int, int]] = []
        total = 0
        for span in spans:
            size = span[1] - span[0]
            if total + size + (joiner if current else 0) > self._chunk_size and current:
                chunks.append((current[0][0], current[-1][1]))
                while total > self._chunk_overlap or (total + size + (joiner if current else 0) > self._chunk_size and total > 0):
                    total -= current[0][1] - current[0][0] + (joiner if len(current) > 1 else 0)
                    current = current[1:]
            current.append(span)
            total += size + (joiner if len(current) > 1 else 0)
        if current:
            chunks.append((current[0][0], current[-1][1]))
        return chunks

    def _split(self, text: str, separators: list[str], base: int) -> list[tuple[str, int]]:
        if not text:
            return []
        if len(text) <= self._chunk_size:
            return self._emit(text, 0, len(text), base)
        separator = separators[-1]
        remaining: list[str] = []
        for i, sep in enumerate(separators):
            if not sep:
                separator = sep
                break
            if sep in text:
                separator = sep
                remaining = separators[i + 1 :]
                break
        found: list[tuple[str, int]] = []
        good: list[tuple[int, int]] = []
        for span in _split(text, separator):
            piece = text[span[0] : span[1]]
            if len(piece) < self._chunk_size:
                good.append(span)
                continue
            for s, e in self._merge(good, separator):
                found.extend(self._emit(text, s, e, base))
            good = []
            if remaining:
                found.extend(self._split(piece, remaining, base + span[0]))
            elif len(piece) > self._chunk_size:
                step = max(1, self._chunk_size - self._chunk_overlap)
                for start in range(0, len(piece), step):
                    found.extend(self._emit(piece, start, min(start + self._chunk_size, len(piece)), base + span[0]))
            else:
                found.extend(self._emit(text, span[0], span[1], base))
        for s, e in self._merge(good, separator):
            found.extend(self._emit(text, s, e, base))
        return found
