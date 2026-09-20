import json
from pathlib import Path

import pytest

from clause.evaluation.golden import Answer, GoldenQuestion
from clause.evaluation.report import (
    CHANCE_BASELINE_CHUNK_COUNTS,
    CHANCE_BASELINE_GOLDEN_SHA256,
    CHANCE_BASELINE_WORST,
    BucketResult,
    Fingerprint,
    StrategyResult,
    build_report,
    render_markdown,
    write_report,
)

FP = Fingerprint(
    manifest_sha256="m" * 64,
    chunk_counts={"structural": 336, "fixed_window": 318},
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_depth=10,
    golden_path="data/golden/kyc-v1.jsonl",
    golden_sha256="g" * 64,
    git_commit="abc1234",
)

#: A fingerprint that matches the chance-baseline constants exactly (real
#: committed golden set + real chunk counts), for tests that need "not stale".
FP_CONSISTENT = Fingerprint(
    manifest_sha256="m" * 64,
    chunk_counts=dict(CHANCE_BASELINE_CHUNK_COUNTS),
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_depth=10,
    golden_path="data/golden/kyc-v1.jsonl",
    golden_sha256=CHANCE_BASELINE_GOLDEN_SHA256,
    git_commit="abc1234",
)

RESULT = StrategyResult(
    strategy="structural",
    per_bucket={"definitional": BucketResult(15, 0.6, 0.8, 0.9, 0.7)},
    overall=BucketResult(60, 0.5, 0.75, 0.85, 0.62),
)

RESULT_FIXED_WINDOW = StrategyResult(
    strategy="fixed_window",
    per_bucket={"definitional": BucketResult(15, 0.5, 0.7, 0.8, 0.6)},
    overall=BucketResult(60, 0.4, 0.7, 0.8, 0.55),
)


def _q(
    qid: str,
    bucket: str = "numeric_threshold",
    notes: str = "",
    n_answers: int = 1,
    provenance: str = "drafted",
) -> GoldenQuestion:
    answers = tuple(
        Answer(
            doc_id="doc",
            char_start=i * 10,
            char_end=i * 10 + 5,
            content_sha256="s" * 64,
        )
        for i in range(n_answers)
    )
    return GoldenQuestion(
        qid=qid,
        question="Some question?",
        bucket=bucket,
        answers=answers,
        provenance=provenance,
        notes=notes,
    )


def test_report_carries_the_fingerprint() -> None:
    report = build_report([RESULT], FP, {"drafted": 45, "human_verified": 15}, golden=[])
    assert report["fingerprint"]["golden_sha256"] == "g" * 64
    assert report["fingerprint"]["retrieval_depth"] == 10


def test_report_records_the_provenance_split() -> None:
    report = build_report([RESULT], FP, {"drafted": 45, "human_verified": 15}, golden=[])
    assert report["golden_provenance"] == {"drafted": 45, "human_verified": 15}


def test_markdown_states_the_precision_identity() -> None:
    """precision@1 and recall@1 are the same number; the report must say so.

    ``PRECISION_NOTE`` renders the phrase with backticks around ``recall@1``,
    so the bare-word assertion from the brief can never match -- it is
    ``identical to `recall@1```, not ``identical to recall@1``.
    """
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}, golden=[]))
    assert "precision@1" in md
    assert "identical to `recall@1`" in md


def test_markdown_never_prints_precision_as_a_separate_column() -> None:
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}, golden=[]))
    header = next(line for line in md.splitlines() if "recall@1" in line and "|" in line)
    assert "precision@1" not in header


def test_markdown_names_every_bucket_and_strategy() -> None:
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}, golden=[]))
    assert "structural" in md
    assert "definitional" in md


def test_write_report_emits_both_artifacts(tmp_path: Path) -> None:
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[])
    md, js = tmp_path / "eval.md", tmp_path / "eval.json"
    write_report(report, md, js)
    assert "structural" in md.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8"))["fingerprint"]["git_commit"] == "abc1234"


# --- guard: `golden` must be a required argument, not defaultable ---


def test_build_report_requires_the_golden_set() -> None:
    """A three-argument call must fail loudly, not silently omit the four additions.

    `golden` used to default to `()`, which meant a caller that forgot to pass
    it got a report with the sub-kind, heading-span and answers-per-question
    sections silently empty, and every existing test still passed. Requiring
    the argument turns that omission into a `TypeError` at the call site.
    """
    with pytest.raises(TypeError):
        build_report([RESULT], FP, {"drafted": 60})  # type: ignore[call-arg]


# --- guard: the fingerprint must survive into the committed JSON artifact ---


def test_fingerprint_round_trips_through_json_byte_for_byte() -> None:
    """Every fingerprint field, not just one, must reach the JSON artifact.

    CI's gate (Task 11) reads `reports/eval.json` as a plain dict and compares
    every fingerprint field against the working tree. If any field were
    dropped on the way into the report dict, the gate would silently stop
    checking it.
    """
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[])
    reloaded = json.loads(json.dumps(report))
    assert reloaded["fingerprint"] == FP.to_dict()


# --- addition (i): kind= sub-kinds within a bucket ---


def test_numeric_threshold_kind_subkinds_are_rendered() -> None:
    golden = [
        _q("num-a", notes="kind=regulatory_threshold"),
        _q("num-b", notes="kind=entry_count"),
        _q("num-c", notes="kind=identifier"),
        _q("num-d", notes="kind=identifier"),
    ]
    report = build_report([RESULT], FP, {"drafted": 60}, golden=golden)
    assert report["golden_composition"]["bucket_kind_breakdown"]["numeric_threshold"] == {
        "regulatory_threshold": 1,
        "entry_count": 1,
        "identifier": 2,
    }
    md = render_markdown(report)
    assert "`kind=` sub-kinds" in md
    assert "regulatory_threshold" in md
    assert "entry_count" in md
    assert "| identifier | 2 |" in md


def test_a_single_averaged_bucket_row_still_appears_alongside_subkinds() -> None:
    """Sub-kinds supplement the bucket row; they do not replace it."""
    golden = [_q("num-a", notes="kind=regulatory_threshold")]
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}, golden=golden))
    assert "definitional" in md  # the ordinary per-bucket row is untouched


# --- addition (ii): span=heading questions surfaced separately ---


def test_heading_span_questions_are_listed_by_qid() -> None:
    golden = [
        _q("num-006", notes="kind=entry_count span=heading"),
        _q("num-007", notes="kind=entry_count span=heading"),
        _q("num-001", notes="kind=regulatory_threshold"),
    ]
    report = build_report([RESULT], FP, {"drafted": 60}, golden=golden)
    assert report["golden_composition"]["heading_span_questions"] == ["num-006", "num-007"]
    md = render_markdown(report)
    assert "`span=heading`" in md
    assert "num-006" in md
    assert "num-007" in md
    assert "num-001" not in md.split("### `span=heading` questions")[1].split("###")[0]


def test_no_heading_questions_says_so_explicitly() -> None:
    golden = [_q("num-001", notes="kind=regulatory_threshold")]
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}, golden=golden))
    assert "No question in this golden set carries a `span=heading` tag." in md


# --- addition (iii): answers-per-question distribution ---


def test_answers_per_question_distribution_is_published() -> None:
    golden = [
        _q("q1", n_answers=1),
        _q("q2", n_answers=1),
        _q("q3", n_answers=3),
    ]
    report = build_report([RESULT], FP, {"drafted": 60}, golden=golden)
    assert report["golden_composition"]["answers_per_question_histogram"] == {"1": 2, "3": 1}
    md = render_markdown(report)
    assert "Answers per question" in md
    assert "| 1 | 2 |" in md
    assert "| 3 | 1 |" in md


# --- addition (iv): chance baseline beside the headline numbers ---


def test_chance_baseline_appears_next_to_each_strategys_recall() -> None:
    report = build_report([RESULT, RESULT_FIXED_WINDOW], FP, {"drafted": 60}, golden=[])
    assert report["strategies"]["structural"]["chance_baseline_recall_at_5"] == 0.055
    assert report["strategies"]["fixed_window"]["chance_baseline_recall_at_5"] == 0.058
    assert report["chance_baseline"]["worst_single_question"] == CHANCE_BASELINE_WORST

    md = render_markdown(report)
    structural_section = md.split("### structural")[1].split("### fixed_window")[0]
    assert "0.055" in structural_section
    assert str(CHANCE_BASELINE_WORST) in structural_section
    fixed_window_section = md.split("### fixed_window")[1]
    assert "0.058" in fixed_window_section


def test_chance_baseline_survives_into_json() -> None:
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[])
    reloaded = json.loads(json.dumps(report))
    assert reloaded["strategies"]["structural"]["chance_baseline_recall_at_5"] == 0.055
    assert reloaded["chance_baseline"]["recall_at_5"]["structural"] == 0.055


# --- chance-baseline drift detection ---


def test_stale_chance_baseline_warns_when_golden_set_has_moved() -> None:
    """FP's golden_sha256 is a dummy value, not the real committed file's hash."""
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[])
    assert report["chance_baseline"]["stale"] is True
    reasons = report["chance_baseline"]["staleness_reasons"]
    assert any("golden set has changed" in r for r in reasons)
    md = render_markdown(report)
    assert "STALE CHANCE BASELINE" in md


def test_stale_chance_baseline_warns_when_chunk_counts_have_moved() -> None:
    moved = Fingerprint(
        manifest_sha256=FP_CONSISTENT.manifest_sha256,
        chunk_counts={"structural": 999, "fixed_window": 318},
        embedding_model=FP_CONSISTENT.embedding_model,
        retrieval_depth=FP_CONSISTENT.retrieval_depth,
        golden_path=FP_CONSISTENT.golden_path,
        golden_sha256=FP_CONSISTENT.golden_sha256,
        git_commit=FP_CONSISTENT.git_commit,
    )
    report = build_report([RESULT], moved, {"drafted": 60}, golden=[])
    assert report["chance_baseline"]["stale"] is True
    assert any(
        "indexed corpus has changed" in r for r in report["chance_baseline"]["staleness_reasons"]
    )
    md = render_markdown(report)
    assert "STALE CHANCE BASELINE" in md


def test_consistent_fingerprint_shows_no_staleness_warning() -> None:
    report = build_report([RESULT], FP_CONSISTENT, {"drafted": 60}, golden=[])
    assert report["chance_baseline"]["stale"] is False
    assert report["chance_baseline"]["staleness_reasons"] == []
    md = render_markdown(report)
    assert "STALE" not in md


# --- provenance: drafted-only ground truth must not read as verified ---


def test_all_drafted_provenance_triggers_an_explicit_warning() -> None:
    md = render_markdown(build_report([RESULT], FP, {"drafted": 65}, golden=[]))
    assert "No question in this golden set has completed human verification" in md


def test_some_verified_provenance_suppresses_the_warning() -> None:
    md = render_markdown(
        build_report([RESULT], FP, {"drafted": 45, "human_verified": 20}, golden=[])
    )
    assert "No question in this golden set has completed human verification" not in md


# --- source of golden-set-derived numbers must be unambiguous ---


def test_golden_composition_section_labels_its_own_source() -> None:
    md = render_markdown(
        build_report([RESULT], FP, {"drafted": 60}, golden=[_q("num-a")])
    )
    assert "not from any strategy's retrieval results" in md
