"""`clause.evaluation.gate.check` is pure -- no file I/O, no database, no CLI --
so it is exercised here purely against dicts. CLI-level behaviour (loading
`reports/eval.json` and `reports/baseline.json`, rebuilding the tree
fingerprint, the empty-database fallback and the notes it must print, exit
codes) lives in `tests/test_gate_cli.py`.
"""

import copy

from clause.evaluation.gate import check

FP = {
    "manifest_sha256": "m" * 64,
    "chunk_counts": {"structural": 336, "fixed_window": 318},
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "retrieval_depth": 10,
    "golden_path": "data/golden/kyc-v1.jsonl",
    "golden_sha256": "g" * 64,
    "git_commit": "abc1234",
    "retrieval_code_sha256": "c" * 64,
}

REPORT = {
    "fingerprint": FP,
    "golden_provenance": {"drafted": 60},
    "strategies": {
        "structural": {"overall": {"recall_at_5": 0.80}, "per_bucket": {}},
        "fixed_window": {"overall": {"recall_at_5": 0.70}, "per_bucket": {}},
    },
}

BASELINE = {"structural": {"recall_at_5": 0.75}, "fixed_window": {"recall_at_5": 0.65}}


def test_a_report_above_baseline_passes() -> None:
    assert check(REPORT, BASELINE, FP) == []


def test_a_recall_drop_fails() -> None:
    dropped = copy.deepcopy(REPORT)
    dropped["strategies"]["structural"]["overall"]["recall_at_5"] = 0.60
    problems = check(dropped, BASELINE, FP)
    assert any("structural" in p and "recall@5" in p for p in problems)


def test_equal_to_baseline_passes() -> None:
    equal = copy.deepcopy(REPORT)
    equal["strategies"]["structural"]["overall"]["recall_at_5"] = 0.75
    assert check(equal, BASELINE, FP) == []


def test_a_stale_fingerprint_fails() -> None:
    """Not re-running the eval must fail as surely as regressing."""
    moved = {**FP, "golden_sha256": "z" * 64}
    problems = check(REPORT, BASELINE, moved)
    assert any("golden_sha256" in p for p in problems)


def test_a_changed_model_fails_the_fingerprint() -> None:
    moved = {**FP, "embedding_model": "something-else"}
    assert any("embedding_model" in p for p in check(REPORT, BASELINE, moved))


def test_a_retrieval_code_change_fails_the_fingerprint() -> None:
    """This is the field that catches editing retrieve.py/embed.py/index.py/
    chunking/*.py/metrics.py, degrading recall, and committing without a
    fresh `make eval` -- none of the other fields is a function of that
    code, so before this field existed, that exact regression passed every
    check silently.
    """
    moved = {**FP, "retrieval_code_sha256": "d" * 64}
    assert any("retrieval_code_sha256" in p for p in check(REPORT, BASELINE, moved))


def test_no_baseline_skips_the_regression_check() -> None:
    """A first run has nothing to regress against, so `check()` reports no
    problem -- it is the CLI's job (tests/test_gate_cli.py), not this
    function's, to tell a human that no regression check ran.
    """
    assert check(REPORT, None, FP) == []


def test_a_strategy_missing_from_the_report_fails() -> None:
    partial = copy.deepcopy(REPORT)
    del partial["strategies"]["fixed_window"]
    assert any("fixed_window" in p for p in check(partial, BASELINE, FP))
