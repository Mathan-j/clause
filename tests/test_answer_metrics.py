from datetime import date

from clause.answering.schema import Citation, Sentence
from clause.evaluation.answer_metrics import (
    content_terms,
    summarise_support,
    support_signal,
)


def _c(text: str) -> Citation:
    return Citation(
        doc_id="d",
        char_start=0,
        char_end=len(text),
        source_url="u",
        published_date=date(2025, 1, 1),
        doc_type="notification",
        text=text,
    )


def test_content_terms_keeps_numbers_and_capitalised_terms() -> None:
    terms = content_terms("Banks must report within 7 days to the Reserve Bank.")
    assert "7" in terms
    assert "reserve" in terms
    assert "banks" in terms


def test_content_terms_drops_stopwords_and_punctuation() -> None:
    terms = content_terms("the and to of a")
    assert terms == frozenset()


def test_a_sentence_fully_covered_by_its_citation_scores_one() -> None:
    s = Sentence(text="Report within 7 days.", citation_indices=(1,), factual=True)
    assert support_signal(s, [_c("Entities shall report within 7 days of detection.")]) == 1.0


def test_a_sentence_sharing_nothing_with_its_citation_scores_zero() -> None:
    s = Sentence(text="Capital adequacy is 11 percent.", citation_indices=(1,), factual=True)
    assert support_signal(s, [_c("Customer identification requires a valid document.")]) == 0.0


def test_a_non_factual_sentence_is_not_scored() -> None:
    s = Sentence(text="Here is a summary.", citation_indices=(), factual=False)
    assert support_signal(s, []) == 1.0


def test_the_summary_reports_a_distribution_not_a_single_score() -> None:
    """Named so it cannot be quoted as a faithfulness score, because it is not one."""
    d = summarise_support([0.0, 0.5, 1.0, 1.0])
    assert d.n == 4
    assert d.zero == 1
    assert d.at_or_above_half == 3
    assert 0.6 < d.mean < 0.65


def test_an_empty_summary_is_not_a_perfect_one() -> None:
    d = summarise_support([])
    assert d.n == 0
    assert d.mean == 0.0
