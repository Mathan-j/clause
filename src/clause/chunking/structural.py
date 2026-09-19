import re
from itertools import pairwise

from clause.chunking.base import make_chunk
from clause.models import Chunk, Document

# RBI numbering: "1.", "4.2", "4(1)(v)", "(a)", and annex headings.
BOUNDARY = re.compile(
    r"(?<=[.\s])(?=(?:\d+(?:\.\d+)*\.\s)|(?:\d+\(\d+\)(?:\([a-z]+\))?\s)|(?:\([a-z]\)\s)|(?:Annex\b))"
)
SENTENCE_END = re.compile(r"(?<=[.;:])\s")


class StructuralChunker:
    """Split on RBI's own numbering, then repair sizes.

    Boundaries come from the document's structure rather than a character
    count, so a chunk tends to be a complete regulatory provision -- which is
    what a citation should point at.
    """

    name = "structural"

    def __init__(self, min_chunk_chars: int, max_chunk_chars: int) -> None:
        if min_chunk_chars >= max_chunk_chars:
            raise ValueError("min_chunk_chars must be smaller than max_chunk_chars")
        self._min = min_chunk_chars
        self._max = max_chunk_chars

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        spans = self._split_on_structure(text)
        spans = self._merge_small(spans)
        spans = self._split_large(text, spans)

        return [
            make_chunk(document, self.name, ordinal, start, end)
            for ordinal, (start, end) in enumerate(spans)
        ]

    def _split_on_structure(self, text: str) -> list[tuple[int, int]]:
        cuts = [0, *(m.start() for m in BOUNDARY.finditer(text)), len(text)]
        cuts = sorted(set(cuts))
        return [(a, b) for a, b in pairwise(cuts) if b > a]

    def _merge_small(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if merged and (merged[-1][1] - merged[-1][0]) < self._min:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        # a trailing runt merges backwards rather than standing alone
        if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < self._min:
            last = merged.pop()
            merged[-1] = (merged[-1][0], last[1])
        return merged

    def _split_large(self, text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for start, end in spans:
            cursor = start
            while end - cursor > self._max:
                window_end = cursor + self._max
                breaks = [m.end() for m in SENTENCE_END.finditer(text, cursor, window_end)]
                cut = breaks[-1] if breaks else window_end
                out.append((cursor, cut))
                cursor = cut
            out.append((cursor, end))
        return out
