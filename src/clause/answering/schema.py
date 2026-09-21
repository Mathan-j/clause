"""Structured answer types.

A `Sentence` references citations by **1-based index into the hits presented to
the model**, not by chunk id. `Hit` carries no chunk id, and an index is what a
GBNF grammar can constrain to a valid range at generation time -- so a citation
pointing at nothing cannot be generated. Resolution to an actual span happens in
`resolver.py`, which is where an unresolvable citation is still possible and is
still a hard failure: the document can be absent, or the span out of range,
between retrieval and resolution.

`Citation` deliberately does not carry `effective_date`: it is NULL for all 61
documents in this corpus, so carrying it here would add an always-empty field.
`published_date` and `doc_type` are the fields actually populated.
"""

from dataclasses import dataclass
from datetime import date

MAX_CITATIONS_PER_SENTENCE = 3


class AnsweringError(Exception):
    """Base for every failure in the answering path."""


class UnresolvableCitationError(AnsweringError):
    """A citation did not resolve to a span in the corpus.

    CLAUDE.md: an answer containing a citation that does not resolve is a hard
    failure, never a warning. Nothing catches this and returns a degraded answer.
    """


class UncitedClaimError(AnsweringError):
    """A sentence marked factual carried no citation."""


class ModelNotAvailableError(AnsweringError):
    """The GGUF model file is not present, and this process will not download it."""


@dataclass(frozen=True, slots=True)
class Citation:
    doc_id: str
    char_start: int
    char_end: int
    source_url: str
    published_date: date
    doc_type: str
    text: str

    def __post_init__(self) -> None:
        if self.char_start < 0:
            raise ValueError(f"char_start must not be negative: {self.char_start}")
        if self.char_start >= self.char_end:
            raise ValueError(
                f"span must be non-empty and forward: got "
                f"[{self.char_start}:{self.char_end}] in {self.doc_id!r}"
            )
        expected = self.char_end - self.char_start
        if len(self.text) != expected:
            raise ValueError(
                f"citation text length {len(self.text)} does not match its span "
                f"[{self.char_start}:{self.char_end}] (expected {expected}) in "
                f"{self.doc_id!r}: the text must be the slice, not a paraphrase of it"
            )


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    citation_indices: tuple[int, ...]
    factual: bool

    def __post_init__(self) -> None:
        # A factual sentence with no citation is deliberately constructible here.
        # `validate.enforce` is the single enforcement point for the citation
        # contract; rejecting it in both places would leave enforce's branch
        # unreachable and its test would have to fake an object to reach it.
        for i in self.citation_indices:
            if i < 1:
                raise ValueError(f"citation indices are 1-based, got {i}")


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    """What the model produced, before any citation was resolved."""

    question: str
    sentences: tuple[Sentence, ...]


@dataclass(frozen=True, slots=True)
class Answer:
    """A draft whose citations have all resolved against the corpus."""

    question: str
    sentences: tuple[Sentence, ...]
    citations: tuple[Citation, ...]


class RefusalReason:
    LOW_SCORE = "low_score"
    NO_CANDIDATES = "no_candidates"


@dataclass(frozen=True, slots=True)
class Refusal:
    question: str
    reason: str
    top_score: float | None
