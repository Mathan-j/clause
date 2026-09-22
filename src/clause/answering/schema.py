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

`AnswerDraft` carries the `hits` it was drafted against, not just the question
and sentences. Without that binding, an answerer could be shown one hit list and
`resolve_citations` could be called with a different one -- a re-retrieval, a
truncated top-k, a different strategy -- and every cited index would silently
resolve against the wrong document. Fixed by elimination: there is exactly one
hit list a draft's indices can mean, because it travels with the draft.
"""

from dataclasses import dataclass
from datetime import date

from clause.retrieve import Hit

MAX_CITATIONS_PER_SENTENCE = 3


class AnsweringError(ValueError):
    """Base for every failure in the answering path.

    Inherits from `ValueError`, not `Exception`, so every existing
    `except ValueError` and `pytest.raises(ValueError)` -- the dataclass
    `__post_init__` convention this module already follows -- keeps working,
    while a caller that instead writes `except AnsweringError:` to record a
    contract breach (Task 10) now catches every one of them. Nothing has to
    be traded off between the two.
    """


class UnresolvableCitationError(AnsweringError):
    """A citation did not resolve to a span in the corpus.

    CLAUDE.md: an answer containing a citation that does not resolve is a hard
    failure, never a warning. Nothing catches this and returns a degraded answer.
    """


class UncitedClaimError(AnsweringError):
    """A sentence marked factual carried no citation."""


class TooManyCitationsError(AnsweringError):
    """A sentence cited more than `MAX_CITATIONS_PER_SENTENCE` indices.

    Was a bare `ValueError` -- outside the hierarchy a caller catching
    `AnsweringError` to record a contract breach relies on. Every failure in the
    answering path belongs in that hierarchy, per `AnsweringError`'s own docstring.
    """


class GenerationError(AnsweringError):
    """The model produced nothing usable -- truncated, or not parseable.

    Distinct from a citation failure: no draft existed for the contract to
    enforce, so this is a failure to generate rather than a breached
    contract, and the evaluation counts it separately.
    """


class ModelNotAvailableError(AnsweringError):
    """The GGUF model file is not present, and this process will not download it."""


class InvalidAnswerError(AnsweringError):
    """An `Answer` was constructed that could not have come from a sound `enforce`.

    Raised by `Answer.__post_init__` -- see its docstring for what it checks.
    """


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
        # Plain ValueError, deliberately not AnsweringError: these are value-object
        # invariants on construction inputs. In the real pipeline the resolver
        # builds a Citation from corpus data it already trusts, so a failure here
        # means the input data is corrupt, not that an answer's contract was
        # breached -- Answer's own checks below are the contract checks.
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
        # Plain ValueError, deliberately not AnsweringError -- see Citation's
        # __post_init__ comment: this is a value-object invariant on a
        # construction input, not a contract check.
        #
        # A factual sentence with no citation is deliberately constructible here.
        # `validate.enforce` is the single enforcement point for the citation
        # contract; rejecting it in both places would leave enforce's branch
        # unreachable and its test would have to fake an object to reach it.
        for i in self.citation_indices:
            if i < 1:
                raise ValueError(f"citation indices are 1-based, got {i}")


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    """What the model produced, before any citation was resolved.

    `hits` is the exact hit list the sentences' `citation_indices` were drafted
    against -- see the module docstring. `resolve_citations` reads it from here
    rather than taking a second, independently-suppliable `hits` argument.
    """

    question: str
    sentences: tuple[Sentence, ...]
    hits: tuple[Hit, ...]


@dataclass(frozen=True, slots=True)
class Answer:
    """A draft whose citations have all resolved against the corpus.

    Carries its own invariants rather than trusting the function that built it:
    every sentence's `citation_indices` must be a valid 1-based position into
    `citations`, and there must be at least one sentence. Both are checkable by
    constructing a bad `Answer` directly -- `enforce` is *a* way to build a
    conforming one, not the only thing standing between this type and a broken one.
    """

    question: str
    sentences: tuple[Sentence, ...]
    citations: tuple[Citation, ...]

    def __post_init__(self) -> None:
        # InvalidAnswerError, not a plain ValueError: unlike Citation's and
        # Sentence's checks above, these are the citation contract itself, not a
        # value-object invariant on a construction input. Task 10's
        # `except AnsweringError:` must catch this the same way it catches a cap
        # violation or an uncited claim -- InvalidAnswerError is one, since
        # AnsweringError now inherits from ValueError, existing
        # pytest.raises(ValueError) call sites for this check keep working too.
        if not self.sentences:
            raise InvalidAnswerError("an answer must carry at least one sentence")
        for sentence in self.sentences:
            for i in sentence.citation_indices:
                if not 1 <= i <= len(self.citations):
                    raise InvalidAnswerError(
                        f"citation index {i} is out of range for "
                        f"{len(self.citations)} citation(s): {sentence.text!r}"
                    )


class RefusalReason:
    LOW_SCORE = "low_score"
    NO_CANDIDATES = "no_candidates"


@dataclass(frozen=True, slots=True)
class Refusal:
    question: str
    reason: str
    top_score: float | None
