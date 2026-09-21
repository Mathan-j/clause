"""Enforce the citation contract over a resolved draft.

This is the gate CLAUDE.md describes: "an answer containing a citation that does
not resolve is a hard failure, never a warning". Every failure here raises. There
is deliberately no lenient mode, no `strict=False`, and no path that returns a
partial answer -- an answer that has been allowed through with a broken citation
is worse than no answer, because it looks the same as a sound one.

`resolve_citations` keys its mapping by the *original* hit index a sentence cites,
which is not generally contiguous from 1 (a draft can cite only hit 2, or hits 3
and 7). An `Answer`'s `citations` tuple has no such gaps -- it holds exactly the
citations actually used, in ascending original-index order. So this function
**renumbers**: each sentence's `citation_indices` is rewritten from original hit
indices to 1-based positions in that tuple, and it is that rewritten `Sentence`
that goes into the returned `Answer`. This is what makes `answer.citations[i - 1]`
a correct, load-bearing invariant for any index carried by any sentence in an
`Answer` -- including downstream consumers, such as a support-signal computation
over cited citations, that will do exactly that arithmetic.
"""

from collections.abc import Mapping

from clause.answering.schema import (
    MAX_CITATIONS_PER_SENTENCE,
    Answer,
    AnswerDraft,
    Citation,
    Sentence,
    UncitedClaimError,
    UnresolvableCitationError,
)


def enforce(draft: AnswerDraft, resolved: Mapping[int, Citation]) -> Answer:
    """Return an `Answer`, or raise. Never returns a degraded result.

    `resolved` is keyed by the original hit index a sentence cites (exactly what
    `resolve_citations` returns). The returned `Answer` carries citations only for
    indices actually referenced by some sentence, and every sentence's
    `citation_indices` is rewritten to 1-based positions into that tuple.
    """
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
            if index not in resolved:
                raise UnresolvableCitationError(
                    f"citation index {index} has no resolved citation "
                    f"({len(resolved)} resolved)"
                )
            used.setdefault(index, None)

    order = sorted(used)
    position = {original: position for position, original in enumerate(order, start=1)}
    citations = tuple(resolved[original] for original in order)

    sentences = tuple(
        Sentence(
            text=sentence.text,
            citation_indices=tuple(position[i] for i in sentence.citation_indices),
            factual=sentence.factual,
        )
        for sentence in draft.sentences
    )

    return Answer(question=draft.question, sentences=sentences, citations=citations)
