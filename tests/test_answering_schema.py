from datetime import date

import pytest

from clause.answering.schema import (
    Answer,
    AnswerDraft,
    Citation,
    Refusal,
    RefusalReason,
    Sentence,
)


def _citation() -> Citation:
    return Citation(
        doc_id="rbi-12866",
        char_start=100,
        char_end=200,
        source_url="https://example.invalid/12866",
        published_date=date(2025, 6, 12),
        doc_type="master_direction",
        text="x" * 100,
    )


def test_citation_span_must_be_forward_and_non_empty() -> None:
    with pytest.raises(ValueError, match="forward"):
        Citation(
            doc_id="d",
            char_start=5,
            char_end=5,
            source_url="u",
            published_date=date(2025, 1, 1),
            doc_type="notification",
            text="",
        )


def test_citation_text_length_must_match_its_span() -> None:
    """A citation whose text is not its span is not a citation, it is a claim."""
    with pytest.raises(ValueError, match="length"):
        Citation(
            doc_id="d",
            char_start=0,
            char_end=10,
            source_url="u",
            published_date=date(2025, 1, 1),
            doc_type="notification",
            text="too short",
        )


def test_a_factual_sentence_with_no_citation_is_constructible_here() -> None:
    """Deliberately NOT rejected at construction.

    `validate.enforce` is the single place the citation contract is enforced. If
    `Sentence` also rejected this, `enforce`'s branch could never fire in practice
    and its test would have to build the object through `object.__setattr__` to
    reach it -- a check that cannot fail for the reason it exists.
    """
    s = Sentence(text="Banks must verify identity.", citation_indices=(), factual=True)
    assert s.citation_indices == ()


def test_a_non_factual_sentence_may_carry_no_citation() -> None:
    s = Sentence(text="Here is what the circulars say.", citation_indices=(), factual=False)
    assert s.citation_indices == ()


def test_citation_indices_must_be_one_based() -> None:
    with pytest.raises(ValueError, match="1-based"):
        Sentence(text="x", citation_indices=(0,), factual=True)


def test_draft_and_answer_carry_the_question() -> None:
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="a", citation_indices=(1,), factual=True),),
    )
    assert draft.question == "q"
    answer = Answer(question="q", sentences=draft.sentences, citations=(_citation(),))
    assert answer.citations[0].doc_id == "rbi-12866"


def test_refusal_records_why_and_the_score_that_caused_it() -> None:
    r = Refusal(question="q", reason=RefusalReason.LOW_SCORE, top_score=0.21)
    assert r.reason == "low_score"
    assert r.top_score == 0.21


def test_a_no_candidates_refusal_has_no_score() -> None:
    r = Refusal(question="q", reason=RefusalReason.NO_CANDIDATES, top_score=None)
    assert r.top_score is None
