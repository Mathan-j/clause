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
    answer = enforce(draft, {1: _c()})
    assert answer.question == "q"
    assert len(answer.citations) == 1


def test_a_factual_sentence_without_a_citation_raises() -> None:
    """This is the single enforcement point, so the object constructs normally."""
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="Banks must verify.", citation_indices=(), factual=True),),
    )
    with pytest.raises(UncitedClaimError, match="Banks must verify"):
        enforce(draft, {})


def test_an_index_with_no_matching_citation_raises() -> None:
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="a", citation_indices=(2,), factual=True),)
    )
    with pytest.raises(UnresolvableCitationError, match="index 2"):
        enforce(draft, {1: _c()})


def test_more_citations_than_the_cap_raises() -> None:
    n = MAX_CITATIONS_PER_SENTENCE + 1
    draft = AnswerDraft(
        question="q",
        sentences=(
            Sentence(text="a", citation_indices=tuple(range(1, n + 1)), factual=True),
        ),
    )
    resolved = {i: _c(i * 4, i * 4 + 4) for i in range(1, n + 1)}
    with pytest.raises(ValueError, match="at most"):
        enforce(draft, resolved)


def test_an_answer_with_no_factual_sentences_needs_no_citations() -> None:
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="I could not find this.", citation_indices=(), factual=False),),
    )
    answer = enforce(draft, {})
    assert answer.citations == ()


def test_unused_citations_are_dropped_not_reported() -> None:
    """A citation nothing references is not evidence; carrying it would inflate the count."""
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="a", citation_indices=(1,), factual=True),)
    )
    answer = enforce(draft, {1: _c(0, 4), 2: _c(4, 8)})
    assert len(answer.citations) == 1


def test_citing_only_a_non_first_hit_produces_an_answer() -> None:
    """Pins the seam bug: a draft citing only hit 2 must not be treated as citing a
    citation that does not exist just because the resolved mapping's *size* is 1.
    """
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="a", citation_indices=(2,), factual=True),)
    )
    answer = enforce(draft, {2: _c()})
    assert len(answer.citations) == 1
    assert answer.sentences[0].citation_indices == (1,)


def test_citation_indices_are_renumbered_to_positions_in_answer_citations() -> None:
    """Every index any sentence in the answer carries must resolve positionally:
    `answer.citations[i - 1]` is correct for any `i` in any sentence's indices.
    """
    hit1 = _c(0, 4)
    hit3 = _c(8, 12)
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="a", citation_indices=(3, 1), factual=True),),
    )
    answer = enforce(draft, {1: hit1, 3: hit3})
    new_indices = answer.sentences[0].citation_indices
    assert len(new_indices) == 2
    for i in new_indices:
        assert answer.citations[i - 1] in (hit1, hit3)
    # relative order preserved: original (3, 1) -> renumbered positions of hit3, hit1
    resolved_citations = [answer.citations[i - 1] for i in new_indices]
    assert resolved_citations == [hit3, hit1]


def test_an_index_absent_from_the_resolved_mapping_raises() -> None:
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="a", citation_indices=(5,), factual=True),)
    )
    with pytest.raises(UnresolvableCitationError, match="index 5"):
        enforce(draft, {1: _c()})
