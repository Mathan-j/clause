"""Acceptance tests for `make eval` end to end.

These assert *properties* of the report -- recall monotonic in k, metrics in
range, every bucket present for every strategy -- rather than fixed values, so
they stay meaningful when the numbers move.

Deliberately does not read `reports/eval.md` / `reports/eval.json`: those are
committed artifacts that Task 11's CI gate protects, and a test fixture that
overwrote them on every `pytest` run would silently replace a good report with
a bad one (or leave the tree dirty) regardless of whether the test passed. The
`evaluated` fixture (tests/conftest.py) runs the real evaluation against
`tmp_path` instead, and these tests assert on what that run produced.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from clause.evaluation.golden import BUCKETS

pytestmark = [pytest.mark.db, pytest.mark.qdrant, pytest.mark.model]

Evaluated = tuple[Path, Path]


def _report(evaluated: Evaluated) -> dict[str, Any]:
    _, json_path = evaluated
    result: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    return result


def test_eval_writes_both_artifacts(evaluated: Evaluated) -> None:
    md_path, json_path = evaluated
    assert md_path.exists()
    assert json_path.exists()


def test_both_strategies_are_measured(evaluated: Evaluated) -> None:
    report = _report(evaluated)
    assert set(report["strategies"]) == {"structural", "fixed_window"}


def test_every_bucket_appears_for_every_strategy(evaluated: Evaluated) -> None:
    report = _report(evaluated)
    for data in report["strategies"].values():
        assert set(data["per_bucket"]) == set(BUCKETS)


def test_recall_is_monotonic_in_k(evaluated: Evaluated) -> None:
    """recall@1 <= recall@5 <= recall@10 always. A violation means a bug."""
    report = _report(evaluated)
    for data in report["strategies"].values():
        for r in [data["overall"], *data["per_bucket"].values()]:
            assert r["recall_at_1"] <= r["recall_at_5"] <= r["recall_at_10"]


def test_metrics_are_within_range(evaluated: Evaluated) -> None:
    report = _report(evaluated)
    for data in report["strategies"].values():
        for r in [data["overall"], *data["per_bucket"].values()]:
            for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10"):
                assert 0.0 <= r[key] <= 1.0


def test_the_report_records_how_ground_truth_was_made(evaluated: Evaluated) -> None:
    report = _report(evaluated)
    assert sum(report["golden_provenance"].values()) >= 60
