"""Enforce the citation contract over a resolved draft.

This is the gate CLAUDE.md describes: "an answer containing a citation that does
not resolve is a hard failure, never a warning". Every failure here raises. There
is deliberately no lenient mode, no `strict=False`, and no path that returns a
partial answer -- an answer that has been allowed through with a broken citation
is worse than no answer, because it looks the same as a sound one.
"""

from collections.abc import Sequence

from clause.answering.schema import (
    MAX_CITATIONS_PER_SENTENCE,
    Answer,
    AnswerDraft,
    Citation,
    UncitedClaimError,
    UnresolvableCitationError,
)


def enforce(draft: AnswerDraft, citations: Sequence[Citation]) -> Answer:
    """Return an `Answer`, or raise. Never returns a degraded result."""
    used: dict[int, None] = {}
    for sentence in draft.sentences:
        if sentence.factual and not sentence.citation_indices:
            raise UncitedClaimError(
                f"factual sentence carries no citation: {sentence.text!r}"
            )
        if len(sentence.citation_indices) > MAX_CITATIONS_PER_SENTENCE:
            raise ValueError(
                f"a sentence may carry at most {MAX_CITATIONS_PER_SENTENCE} citations, "
                f"got {len(sentence.citation_indices)}: {sentence.text!r}"
            )
        for index in sentence.citation_indices:
            if index > len(citations):
                raise UnresolvableCitationError(
                    f"citation index {index} has no resolved citation "
                    f"({len(citations)} resolved)"
                )
            used.setdefault(index, None)

    return Answer(
        question=draft.question,
        sentences=draft.sentences,
        citations=tuple(citations[i - 1] for i in sorted(used)),
    )
