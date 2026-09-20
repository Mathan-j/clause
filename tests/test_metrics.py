import pytest

from clause.evaluation.metrics import (
    RETRIEVAL_DEPTH,
    Retrieved,
    Span,
    first_hit_rank,
    mrr,
    overlaps,
    recall_at_k,
)


def _span(start: int, end: int, doc: str = "d1") -> Span:
    return Span(doc_id=doc, char_start=start, char_end=end)


def _hit(start: int, end: int, doc: str = "d1", score: float = 1.0) -> Retrieved:
    return Retrieved(doc_id=doc, char_start=start, char_end=end, score=score)


def test_overlapping_spans_overlap() -> None:
    assert overlaps(_span(100, 200), _span(150, 250))


def test_containment_counts_as_overlap() -> None:
    assert overlaps(_span(100, 200), _span(120, 180))
    assert overlaps(_span(120, 180), _span(100, 200))


def test_touching_spans_do_not_overlap() -> None:
    """[100,200) and [200,300) share no character. Half-open intervals."""
    assert not overlaps(_span(100, 200), _span(200, 300))


def test_a_single_shared_character_is_an_overlap() -> None:
    assert overlaps(_span(100, 200), _span(199, 300))


def test_spans_in_different_documents_never_overlap() -> None:
    assert not overlaps(_span(100, 200, "d1"), _span(100, 200, "d2"))


def test_first_hit_rank_is_one_based() -> None:
    results = [_hit(0, 50), _hit(100, 200), _hit(300, 400)]
    assert first_hit_rank(results, [_span(150, 160)]) == 2


def test_first_hit_rank_takes_the_earliest_hit() -> None:
    results = [_hit(100, 200), _hit(300, 400)]
    assert first_hit_rank(results, [_span(350, 360), _span(150, 160)]) == 1


def test_first_hit_rank_is_none_when_nothing_hits() -> None:
    assert first_hit_rank([_hit(0, 50)], [_span(900, 950)]) is None


def test_first_hit_rank_with_no_results_is_none() -> None:
    assert first_hit_rank([], [_span(0, 10)]) is None


def test_duplicate_acceptable_spans_do_not_change_the_rank() -> None:
    results = [_hit(0, 50), _hit(100, 200)]
    once = first_hit_rank(results, [_span(150, 160)])
    twice = first_hit_rank(results, [_span(150, 160), _span(150, 160)])
    assert once == twice == 2


def test_recall_at_k_counts_questions_hit_within_k() -> None:
    ranks = [1, 3, None, 7]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.5
    assert recall_at_k(ranks, 10) == 0.75


def test_recall_of_an_empty_golden_set_is_zero_not_a_crash() -> None:
    assert recall_at_k([], 5) == 0.0


def test_mrr_averages_reciprocal_ranks() -> None:
    assert mrr([1, 2, None]) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_mrr_scores_a_rank_beyond_the_cutoff_as_zero() -> None:
    assert mrr([11], cutoff=10) == 0.0
    assert mrr([10], cutoff=10) == pytest.approx(0.1)


def test_retrieval_depth_is_ten() -> None:
    """Pinned by the spec: every metric is computed from one list of 10."""
    assert RETRIEVAL_DEPTH == 10


def test_a_reversed_span_is_rejected() -> None:
    with pytest.raises(ValueError, match="forward"):
        Span(doc_id="d1", char_start=200, char_end=100)


def test_a_zero_width_span_is_rejected() -> None:
    """A zero-width span contains no characters, so it can share none."""
    with pytest.raises(ValueError, match="non-empty"):
        Span(doc_id="d1", char_start=100, char_end=100)


def test_a_negative_offset_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        Span(doc_id="d1", char_start=-1, char_end=10)


def test_retrieved_enforces_the_same_invariant() -> None:
    with pytest.raises(ValueError, match="forward"):
        Retrieved(doc_id="d1", char_start=50, char_end=50, score=1.0)
