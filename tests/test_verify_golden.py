import collections

import pytest
from scripts.verify_golden import apply_decision, select_sample

from clause.evaluation.golden import Answer, GoldenQuestion


def _q(qid: str, bucket: str) -> GoldenQuestion:
    return GoldenQuestion(
        qid=qid,
        question="?",
        bucket=bucket,
        answers=(Answer("rbi-1", 0, 10, "a" * 64),),
        provenance="drafted",
        notes="",
    )


QUESTIONS = [_q(f"q{i:03d}", b) for i, b in enumerate(["definitional"] * 8 + ["procedural"] * 8)]


def test_sample_is_balanced_across_buckets() -> None:
    sample = select_sample(QUESTIONS, per_bucket=3, seed=1)
    counts = collections.Counter(q.bucket for q in sample)
    assert counts == {"definitional": 3, "procedural": 3}


def test_sample_is_deterministic_for_a_seed() -> None:
    # Same seed produces identical results
    assert [q.qid for q in select_sample(QUESTIONS, 3, seed=7)] == [
        q.qid for q in select_sample(QUESTIONS, 3, seed=7)
    ]
    # Different seeds produce different results
    assert [q.qid for q in select_sample(QUESTIONS, 3, seed=7)] != [
        q.qid for q in select_sample(QUESTIONS, 3, seed=8)
    ]


def test_sample_takes_everything_when_a_bucket_is_smaller_than_asked() -> None:
    sample = select_sample(QUESTIONS[:2], per_bucket=5, seed=1)
    assert len(sample) == 2


def test_confirming_marks_it_human_verified() -> None:
    assert apply_decision(QUESTIONS[0], "confirm").provenance == "human_verified"


def test_correcting_replaces_the_span_and_marks_it_corrected() -> None:
    fixed = apply_decision(QUESTIONS[0], "correct", span=(50, 90))
    assert fixed.provenance == "human_corrected"
    assert fixed.answers[0].char_start == 50
    assert fixed.answers[0].char_end == 90
    assert fixed.answers[0].content_sha256 == QUESTIONS[0].answers[0].content_sha256


def test_an_unknown_decision_is_rejected() -> None:
    with pytest.raises(ValueError, match="decision"):
        apply_decision(QUESTIONS[0], "maybe")


def test_negative_span_start_is_rejected() -> None:
    """Span validation rejects negative char_start."""
    with pytest.raises(ValueError, match="span start must not be negative"):
        apply_decision(QUESTIONS[0], "correct", span=(-10, 50))


def test_reversed_span_is_rejected() -> None:
    """Span validation rejects reversed spans (char_start > char_end)."""
    with pytest.raises(ValueError, match="span must be non-empty and forward"):
        apply_decision(QUESTIONS[0], "correct", span=(90, 50))


def test_zero_width_span_is_rejected() -> None:
    """Span validation rejects zero-width spans (char_start == char_end)."""
    with pytest.raises(ValueError, match="span must be non-empty and forward"):
        apply_decision(QUESTIONS[0], "correct", span=(50, 50))
