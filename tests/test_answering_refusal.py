from datetime import date

from clause.answering.refusal import DEFAULT_THRESHOLD, SWEEP_THRESHOLDS, decide, sweep
from clause.answering.schema import RefusalReason
from clause.retrieve import Hit


def _hit(score: float) -> Hit:
    return Hit(
        doc_id="d",
        strategy="structural",
        ordinal=0,
        char_start=0,
        char_end=4,
        score=score,
        text="abcd",
        source_url="u",
        published_date=date(2025, 1, 1),
        regulated_entity=(),
    )


def test_no_hits_refuses_with_its_own_reason() -> None:
    """Distinct from a low score: nothing was retrieved at all."""
    r = decide("q", [], DEFAULT_THRESHOLD)
    assert r is not None
    assert r.reason == RefusalReason.NO_CANDIDATES
    assert r.top_score is None


def test_a_score_below_the_threshold_refuses() -> None:
    r = decide("q", [_hit(0.10)], 0.35)
    assert r is not None
    assert r.reason == RefusalReason.LOW_SCORE
    assert r.top_score == 0.10


def test_a_score_exactly_at_the_threshold_proceeds() -> None:
    """The boundary is inclusive; a question is not refused for tying the bar."""
    assert decide("q", [_hit(0.35)], 0.35) is None


def test_a_score_above_the_threshold_proceeds() -> None:
    assert decide("q", [_hit(0.90)], 0.35) is None


def test_only_the_top_hit_decides() -> None:
    assert decide("q", [_hit(0.90), _hit(0.01)], 0.35) is None
    r = decide("q", [_hit(0.01), _hit(0.90)], 0.35)
    assert r is not None and r.top_score == 0.01


def test_sweep_is_monotone_in_the_threshold() -> None:
    """Raising the bar can never refuse fewer questions. A violation is a bug."""
    counts = sweep([0.1, 0.4, 0.8, None])
    values = [counts[t] for t in SWEEP_THRESHOLDS]
    assert values == sorted(values)


def test_sweep_counts_a_missing_score_as_refused_at_every_threshold() -> None:
    counts = sweep([None])
    assert all(counts[t] == 1 for t in SWEEP_THRESHOLDS)


def test_sweep_covers_the_documented_thresholds() -> None:
    assert SWEEP_THRESHOLDS[0] == 0.0
    assert len(set(SWEEP_THRESHOLDS)) == len(SWEEP_THRESHOLDS)
    assert sorted(SWEEP_THRESHOLDS) == list(SWEEP_THRESHOLDS)
