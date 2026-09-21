from datetime import date

import pytest

from clause.answering.schema import (
    MAX_CITATIONS_PER_SENTENCE,
    AnswerDraft,
    Citation,
    Sentence,
    UncitedClaimError,
    UnresolvableCitationError,
)
from clause.answering.validate import enforce


def _c(start: int = 0, end: int = 4) -> Citation:
    return Citation(
        doc_id="d",
        char_start=start,
        char_end=end,
        source_url="u",
        published_date=date(2025, 1, 1),
        doc_type="notification",
        text="x" * (end - start),
    )


def test_a_clean_draft_becomes_an_answer() -> None:
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="a", citation_indices=(1,), factual=True),)
    )
    answer = enforce(draft, [_c()])
    assert answer.question == "q"
    assert len(answer.citations) == 1


def test_a_factual_sentence_without_a_citation_raises() -> None:
    """This is the single enforcement point, so the object constructs normally."""
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="Banks must verify.", citation_indices=(), factual=True),),
    )
    with pytest.raises(UncitedClaimError, match="Banks must verify"):
        enforce(draft, [])


def test_an_index_with_no_matching_citation_raises() -> None:
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="a", citation_indices=(2,), factual=True),)
    )
    with pytest.raises(UnresolvableCitationError, match="index 2"):
        enforce(draft, [_c()])


def test_more_citations_than_the_cap_raises() -> None:
    n = MAX_CITATIONS_PER_SENTENCE + 1
    draft = AnswerDraft(
        question="q",
        sentences=(
            Sentence(text="a", citation_indices=tuple(range(1, n + 1)), factual=True),
        ),
    )
    with pytest.raises(ValueError, match="at most"):
        enforce(draft, [_c(i, i + 4) for i in range(0, n * 4, 4)])


def test_an_answer_with_no_factual_sentences_needs_no_citations() -> None:
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="I could not find this.", citation_indices=(), factual=False),),
    )
    answer = enforce(draft, [])
    assert answer.citations == ()


def test_unused_citations_are_dropped_not_reported() -> None:
    """A citation nothing references is not evidence; carrying it would inflate the count."""
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="a", citation_indices=(1,), factual=True),)
    )
    answer = enforce(draft, [_c(0, 4), _c(4, 8)])
    assert len(answer.citations) == 1
