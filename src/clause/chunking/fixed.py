from clause.chunking.base import make_chunk
from clause.models import Chunk, Document


class FixedWindowChunker:
    """Overlapping fixed-size windows, snapped to word boundaries.

    This is the baseline. It exists so the structural chunker's Phase 2 numbers
    mean something: a structure-aware strategy with nothing to beat is an
    unfalsifiable claim.
    """

    name = "fixed_window"

    def __init__(self, window_chars: int, overlap_chars: int) -> None:
        if window_chars <= 0:
            raise ValueError("window_chars must be positive")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")
        if overlap_chars >= window_chars:
            raise ValueError("overlap_chars must be smaller than window_chars")
        self._window = window_chars
        self._overlap = overlap_chars

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        ordinal = 0
        stride = self._window - self._overlap

        while start < len(text):
            end = min(start + self._window, len(text))
            if end < len(text):
                space = text.rfind(" ", start + stride, end)
                if space > start:
                    end = space
            chunks.append(make_chunk(document, self.name, ordinal, start, end))
            ordinal += 1
            if end >= len(text):
                break
            start = max(end - self._overlap, start + 1)

        return chunks
