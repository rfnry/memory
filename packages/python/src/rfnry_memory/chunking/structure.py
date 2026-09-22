from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE_OPEN_RE = re.compile(r"```[a-zA-Z0-9_+-]*")
_TABLE_RE = re.compile(
    r"(?:^\|[^\n]+\|\n)"
    r"(?:^\|[\s\-:|]+\|\n)"
    r"(?:^\|[^\n]+\|\n?)+",
    re.MULTILINE,
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)


@dataclass
class AtomicRegion:
    start: int
    end: int
    kind: str
    content: str


@dataclass
class HeadingSpan:
    start: int
    end: int
    path: tuple[str, ...]


def _iter_code_fences(text: str) -> list[AtomicRegion]:
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line)

    regions: list[AtomicRegion] = []
    i = 0
    n = len(lines)
    while i < n:
        if _FENCE_OPEN_RE.fullmatch(lines[i].rstrip("\r\n")):
            j = i + 1
            while j < n and not lines[j].startswith("```"):
                j += 1
            if j < n:
                start = starts[i]
                end = starts[j] + 3
                regions.append(AtomicRegion(start, end, "code_fence", text[start:end]))
                i = j + 1
                continue
        i += 1
    return regions


def find_atomic_regions(text: str) -> list[AtomicRegion]:
    regions: list[AtomicRegion] = []
    regions.extend(_iter_code_fences(text))
    for m in _TABLE_RE.finditer(text):
        regions.append(AtomicRegion(m.start(), m.end(), "table", m.group(0)))
    regions.sort(key=lambda r: (r.start, -r.end))
    merged: list[AtomicRegion] = []
    for region in regions:
        if merged and region.end <= merged[-1].end:
            continue
        if merged and region.start < merged[-1].end:
            prev = merged[-1]
            merged[-1] = AtomicRegion(
                prev.start, region.end, prev.kind, text[prev.start : region.end]
            )
            continue
        merged.append(region)
    return merged


def build_heading_spans(
    text: str,
    exclude_regions: list[AtomicRegion] | None = None,
) -> list[HeadingSpan]:
    headings: list[tuple[int, int, str]] = []
    exclude = exclude_regions or []
    for m in _HEADING_RE.finditer(text):
        offset = m.start()
        if any(r.start <= offset < r.end for r in exclude):
            continue
        title = m.group(2).strip()
        title = title.rstrip("#").rstrip() or title
        headings.append((offset, len(m.group(1)), title))
    if not headings:
        return []
    spans: list[HeadingSpan] = []
    stack: list[tuple[int, str]] = []
    for i, (offset, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        spans.append(
            HeadingSpan(
                start=offset,
                end=end,
                path=tuple(t for _, t in stack),
            )
        )
    return spans


def section_path_at(offset: int, spans: list[HeadingSpan]) -> str | None:
    for s in spans:
        if s.start <= offset < s.end:
            return " > ".join(s.path)
    return None
