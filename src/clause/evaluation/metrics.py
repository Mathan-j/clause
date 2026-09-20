"""Retrieval metrics over character spans.

Deliberately free of any dependency on the database, Qdrant or the embedding
model: this is the one part of Phase 2 that CI can execute, which is what makes
the regression gate meaningful rather than decorative.
"""

from collections.abc import Sequence
from dataclasses import dataclass

#: Every evaluation run retrieves exactly this many results, and every metric is
#: computed from that one ranked list, so the figures in a row stay mutually
#: consistent. Recorded in the report fingerprint so a change to it is visible.
RETRIEVAL_DEPTH = 10


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character range `[char_start, char_end)` within one document.

    Invariant: `char_start >= 0` and `char_start < char_end` (non-empty, forward, non-negative).
    """

    doc_id: str
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.char_start < 0:
            raise ValueError(f"char_start must not be negative: {self.char_start}")
        if self.char_start >= self.char_end:
            raise ValueError(
                f"span must be non-empty and forward: got "
                f"[{self.char_start}:{self.char_end}] in {self.doc_id!r}"
            )


@dataclass(frozen=True, slots=True)
class Retrieved:
    doc_id: str
    char_start: int
    char_end: int
    score: float

    def __post_init__(self) -> None:
        if self.char_start < 0:
            raise ValueError(f"char_start must not be negative: {self.char_start}")
        if self.char_start >= self.char_end:
            raise ValueError(
                f"span must be non-empty and forward: got "
                f"[{self.char_start}:{self.char_end}] in {self.doc_id!r}"
            )


def overlaps(a: Span, b: Span) -> bool:
    """True when two spans share at least one character of the same document.

    Overlap rather than containment: a chunk boundary that splits the answer in
    half still retrieved the answer, and scoring it as a miss would penalise the
    chunker for the golden set's labelling choices.
    """
    if a.doc_id != b.doc_id:
        return False
    return a.char_start < b.char_end and b.char_start < a.char_end


def first_hit_rank(results: Sequence[Retrieved], answers: Sequence[Span]) -> int | None:
    """1-based rank of the first result overlapping any acceptable span.

    `None` when nothing in `results` hits. Ranks are 1-based because that is what
    MRR's reciprocal expects; a 0-based rank would make the top result infinite.
    """
    for rank, result in enumerate(results, start=1):
        span = Span(doc_id=result.doc_id, char_start=result.char_start, char_end=result.char_end)
        if any(overlaps(span, answer) for answer in answers):
            return rank
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of questions whose first hit falls within the top k."""
    if not ranks:
        return 0.0
    hit = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hit / len(ranks)


def mrr(ranks: Sequence[int | None], cutoff: int = RETRIEVAL_DEPTH) -> float:
    """Mean reciprocal rank, scoring anything past `cutoff` as zero."""
    if not ranks:
        return 0.0
    total = sum(1.0 / rank for rank in ranks if rank is not None and rank <= cutoff)
    return total / len(ranks)
