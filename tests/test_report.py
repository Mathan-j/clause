import json
from pathlib import Path

import pytest

from clause.evaluation.golden import Answer, GoldenQuestion
from clause.evaluation.report import (
    CHANCE_BASELINE_CHUNK_COUNTS,
    CHANCE_BASELINE_GOLDEN_SHA256,
    CHANCE_BASELINE_RECALL_AT_5,
    CHANCE_BASELINE_WORST,
    BucketResult,
    ChanceBaseline,
    Fingerprint,
    StrategyResult,
    build_report,
    render_markdown,
    write_report,
)

#: A measured chance baseline that agrees with the pinned reference exactly --
#: the "no mismatch" case most tests want. Tests for the mismatch tripwire
#: build their own, deliberately different, `ChanceBaseline`.
CB = {
    strategy: ChanceBaseline(
        recall_at_5=CHANCE_BASELINE_RECALL_AT_5[strategy],
        worst_question_recall_at_5=CHANCE_BASELINE_WORST[strategy],
    )
    for strategy in CHANCE_BASELINE_RECALL_AT_5
}

FP = Fingerprint(
    manifest_sha256="m" * 64,
    chunk_counts={"structural": 336, "fixed_window": 318},
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_depth=10,
    golden_path="data/golden/kyc-v1.jsonl",
    golden_sha256="g" * 64,
    git_commit="abc1234",
    retrieval_code_sha256="c" * 64,
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
    retrieval_code_sha256="c" * 64,
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
    report = build_report(
        [RESULT], FP, {"drafted": 45, "human_verified": 15}, golden=[], chance_baseline=CB
    )
    assert report["fingerprint"]["golden_sha256"] == "g" * 64
    assert report["fingerprint"]["retrieval_depth"] == 10


def test_report_records_the_provenance_split() -> None:
    report = build_report(
        [RESULT], FP, {"drafted": 45, "human_verified": 15}, golden=[], chance_baseline=CB
    )
    assert report["golden_provenance"] == {"drafted": 45, "human_verified": 15}


def test_markdown_states_the_precision_identity() -> None:
    """precision@1 and recall@1 are the same number; the report must say so.

    ``PRECISION_NOTE`` renders the phrase with backticks around ``recall@1``,
    so the bare-word assertion from the brief can never match -- it is
    ``identical to `recall@1```, not ``identical to recall@1``.
    """
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}, golden=[], chance_baseline=CB))
    assert "precision@1" in md
    assert "identical to `recall@1`" in md


def test_markdown_never_prints_precision_as_a_separate_column() -> None:
    """Pinned to the per-bucket results table specifically (the one whose
    header starts with the `bucket` column), not "the first line that looks
    like a table". `ed60154` (Task 10's fix round) inserted a
    strategy-comparison table above the per-strategy sections that also
    contains `recall@1` and `|` in its header -- `next(...)` silently
    latched onto that instead, so this test stopped checking the table it
    was written about the moment that table existed, and a `precision@1`
    column added to the per-bucket table (the thing this test forbids) would
    have passed it. See `clause.evaluation.report._render_strategy`'s header
    line, which is the one line this must inspect.
    """
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}, golden=[], chance_baseline=CB))
    bucket_table_headers = [
        line for line in md.splitlines() if line.startswith("| bucket ") and "recall@1" in line
    ]
    assert bucket_table_headers, "no per-bucket results table header found to check"
    assert all("precision@1" not in header for header in bucket_table_headers)


def test_markdown_names_every_bucket_and_strategy() -> None:
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}, golden=[], chance_baseline=CB))
    assert "structural" in md
    assert "definitional" in md


def test_write_report_emits_both_artifacts(tmp_path: Path) -> None:
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[], chance_baseline=CB)
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


def test_build_report_requires_the_chance_baseline() -> None:
    """A caller that forgot to measure the chance baseline must fail loudly too.

    `chance_baseline` used to be read from a module constant at render time;
    fix round 3 made it caller-supplied data instead, specifically so a wrong
    original computation can be caught by comparing against the pin -- which
    only works if the caller is forced to actually measure and pass it.
    """
    with pytest.raises(TypeError):
        build_report([RESULT], FP, {"drafted": 60}, golden=[])  # type: ignore[call-arg]


# --- guard: the fingerprint must survive into the committed JSON artifact ---


def test_fingerprint_round_trips_through_json_byte_for_byte() -> None:
    """Every fingerprint field, not just one, must reach the JSON artifact.

    CI's gate (Task 11) reads `reports/eval.json` as a plain dict and compares
    every fingerprint field against the working tree. If any field were
    dropped on the way into the report dict, the gate would silently stop
    checking it.
    """
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[], chance_baseline=CB)
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
    report = build_report([RESULT], FP, {"drafted": 60}, golden=golden, chance_baseline=CB)
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
    report = build_report([RESULT], FP, {"drafted": 60}, golden=golden, chance_baseline=CB)
    md = render_markdown(report)
    assert "definitional" in md  # the ordinary per-bucket row is untouched


# --- addition (ii): span=heading questions surfaced separately ---


def test_heading_span_questions_are_listed_by_qid() -> None:
    golden = [
        _q("num-006", notes="kind=entry_count span=heading"),
        _q("num-007", notes="kind=entry_count span=heading"),
        _q("num-001", notes="kind=regulatory_threshold"),
    ]
    report = build_report([RESULT], FP, {"drafted": 60}, golden=golden, chance_baseline=CB)
    assert report["golden_composition"]["heading_span_questions"] == ["num-006", "num-007"]
    md = render_markdown(report)
    assert "`span=heading`" in md
    assert "num-006" in md
    assert "num-007" in md
    assert "num-001" not in md.split("### `span=heading` questions")[1].split("###")[0]


def test_no_heading_questions_says_so_explicitly() -> None:
    golden = [_q("num-001", notes="kind=regulatory_threshold")]
    report = build_report([RESULT], FP, {"drafted": 60}, golden=golden, chance_baseline=CB)
    md = render_markdown(report)
    assert "No question in this golden set carries a `span=heading` tag." in md


# --- addition (iii): answers-per-question distribution ---


def test_answers_per_question_distribution_is_published() -> None:
    golden = [
        _q("q1", n_answers=1),
        _q("q2", n_answers=1),
        _q("q3", n_answers=3),
    ]
    report = build_report([RESULT], FP, {"drafted": 60}, golden=golden, chance_baseline=CB)
    assert report["golden_composition"]["answers_per_question_histogram"] == {"1": 2, "3": 1}
    md = render_markdown(report)
    assert "Answers per question" in md
    assert "| 1 | 2 |" in md
    assert "| 3 | 1 |" in md


# --- addition (iv): chance baseline beside the headline numbers ---


def test_chance_baseline_appears_next_to_each_strategys_recall() -> None:
    report = build_report(
        [RESULT, RESULT_FIXED_WINDOW], FP, {"drafted": 60}, golden=[], chance_baseline=CB
    )
    assert report["strategies"]["structural"]["chance_baseline_recall_at_5"] == 0.058
    assert report["strategies"]["fixed_window"]["chance_baseline_recall_at_5"] == 0.084
    assert report["chance_baseline"]["pinned"]["worst_single_question"] == CHANCE_BASELINE_WORST

    md = render_markdown(report)
    # Strategies render sorted: "fixed_window" before "structural" ("f" < "s").
    fixed_window_section = md.split("### fixed_window")[1].split("### structural")[0]
    structural_section = md.split("### structural")[1].split("## Provenance fingerprint")[0]
    assert "0.058" in structural_section
    assert "0.375" in structural_section  # structural's own worst question
    assert "0.084" in fixed_window_section
    assert "0.536" in fixed_window_section  # fixed_window's own, higher, worst question


def test_each_strategy_gets_its_own_worst_question_not_a_shared_one() -> None:
    """fixed_window's worst (0.536) must not be attributed to structural or vice versa.

    Strategies render in sorted order, so `fixed_window` comes *before*
    `structural` ("f" < "s") -- both sections are sliced explicitly by their
    start and end markers rather than assumed order.
    """
    report = build_report(
        [RESULT, RESULT_FIXED_WINDOW], FP, {"drafted": 60}, golden=[], chance_baseline=CB
    )
    md = render_markdown(report)
    fixed_window_section = md.split("### fixed_window")[1].split("### structural")[0]
    structural_section = md.split("### structural")[1].split("## Provenance fingerprint")[0]
    assert "0.536" not in structural_section
    assert "0.375" not in fixed_window_section


def test_chance_baseline_survives_into_json() -> None:
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[], chance_baseline=CB)
    reloaded = json.loads(json.dumps(report))
    assert reloaded["strategies"]["structural"]["chance_baseline_recall_at_5"] == 0.058
    assert reloaded["chance_baseline"]["pinned"]["recall_at_5"]["structural"] == 0.058
    assert reloaded["chance_baseline"]["pinned"]["worst_single_question"]["fixed_window"] == 0.536
    assert reloaded["chance_baseline"]["measured"]["structural"]["recall_at_5"] == 0.058


# --- fix round 2, finding 1: the floor must not license a per-bucket comparison ---


def test_chance_floor_is_scoped_to_the_overall_row_only() -> None:
    """The floor is an aggregate over the whole golden set, not per bucket.

    Printing it directly above a per-bucket table invites a reader to compare
    a bucket's recall@5 against the aggregate floor, which the aggregate does
    not license -- buckets can have materially different true floors that
    this renderer has no way to compute.
    """
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[], chance_baseline=CB)
    md = render_markdown(report)
    section = md.split("### structural")[1].split("## Provenance fingerprint")[0]
    assert "applies only to the **overall** row" in section
    assert "per-bucket chance floors were not computed" in section


def test_removing_the_overall_row_scoping_language_is_caught() -> None:
    """Guard: without the scoping language, nothing distinguishes this floor
    sentence from one that (wrongly) licenses a bucket-by-bucket comparison."""
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[], chance_baseline=CB)
    md = render_markdown(report)
    assert "computed once over the **whole golden set** (not per bucket)" in md


# --- fix round 2, finding 2: chunk counts must not print as a raw dict ---


def test_fingerprint_chunk_counts_render_as_readable_rows_not_a_dict() -> None:
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}, golden=[], chance_baseline=CB))
    fingerprint_section = md.split("## Provenance fingerprint")[1]
    assert "{'" not in fingerprint_section
    assert "chunk count (structural) | 336" in fingerprint_section
    assert "chunk count (fixed_window) | 318" in fingerprint_section


# --- fix round 2, finding 3: recall rounds to 2 decimals, not 3 ---


def test_recall_rounds_to_two_decimals_not_three() -> None:
    """5/17 = 0.294117...; a 3rd decimal implies resolution a 17-question
    bucket does not have, contradicting BUCKET_SIZE_NOTE two sections above.

    Only the finding's target (recall) is checked against 2 decimals; MRR is
    intentionally left at 3 -- the finding was specific to the recall columns.
    """
    odd_bucket = BucketResult(17, 5 / 17, 5 / 17, 5 / 17, 5 / 17)
    result = StrategyResult(
        strategy="structural", per_bucket={"numeric_threshold": odd_bucket}, overall=odd_bucket
    )
    md = render_markdown(build_report([result], FP, {"drafted": 60}, golden=[], chance_baseline=CB))
    row = next(line for line in md.splitlines() if line.startswith("| numeric_threshold |"))
    _, _, recall_1, recall_5, recall_10, mrr = (c.strip() for c in row.strip("|").split("|"))
    assert (recall_1, recall_5, recall_10) == ("0.29", "0.29", "0.29")
    assert mrr == "0.294"


# --- fix round 2, finding 4: one dash/ellipsis convention throughout ---


def test_no_unicode_ellipsis_appears_only_ascii_truncation() -> None:
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[_q("num-a")], chance_baseline=CB)
    md = render_markdown(report)
    assert "…" not in md  # the "…" character
    assert "..." in md  # truncated hashes use ASCII instead


# --- fix round 3: chance baseline is measured data, checked against a pin ---


def test_measured_chance_baseline_matching_the_pin_shows_no_mismatch() -> None:
    report = build_report([RESULT], FP_CONSISTENT, {"drafted": 60}, golden=[], chance_baseline=CB)
    assert report["chance_baseline"]["mismatch"] is False
    assert report["chance_baseline"]["mismatch_reasons"] == []
    assert report["strategies"]["structural"]["chance_baseline_mismatch_reasons"] == []
    md = render_markdown(report)
    assert "MISMATCH" not in md


def test_measured_chance_baseline_disagreeing_with_the_pin_fires_the_tripwire() -> None:
    """A wrong pin cannot be caught by the golden_sha256/chunk_counts drift
    check alone -- those inputs can match exactly while the pinned *value*
    was simply wrong from the moment it was recorded (which is what actually
    happened: fix round 2 corrected a `fixed_window` pin of 0.058 that should
    have been 0.084). This is the check that catches that class of error."""
    wrong_measurement = {
        "structural": ChanceBaseline(recall_at_5=0.058, worst_question_recall_at_5=0.375),
        "fixed_window": ChanceBaseline(recall_at_5=0.070, worst_question_recall_at_5=0.536),
    }
    report = build_report(
        [RESULT, RESULT_FIXED_WINDOW],
        FP_CONSISTENT,
        {"drafted": 60},
        golden=[],
        chance_baseline=wrong_measurement,
    )
    assert report["chance_baseline"]["mismatch"] is True
    reasons = report["chance_baseline"]["mismatch_reasons"]
    assert any("fixed_window" in r and "0.070" in r and "0.084" in r for r in reasons)
    assert report["strategies"]["structural"]["chance_baseline_mismatch_reasons"] == []
    fixed_window_reasons = report["strategies"]["fixed_window"]["chance_baseline_mismatch_reasons"]
    assert any("0.070" in r and "0.084" in r for r in fixed_window_reasons)

    md = render_markdown(report)
    assert "CHANCE BASELINE MISMATCH" in md
    fixed_window_section = md.split("### fixed_window")[1].split("### structural")[0]
    structural_section = md.split("### structural")[1].split("## Provenance fingerprint")[0]
    assert "CHANCE BASELINE MISMATCH" in fixed_window_section
    assert "CHANCE BASELINE MISMATCH" not in structural_section


def test_removing_the_mismatch_check_is_caught() -> None:
    """Guard: disagreement between measured and pinned must actually be
    computed, not just plumbed through -- this proves the comparison itself
    exists rather than the tripwire text being printed unconditionally."""
    wrong_measurement = {
        "structural": ChanceBaseline(recall_at_5=0.058, worst_question_recall_at_5=0.375),
        "fixed_window": ChanceBaseline(recall_at_5=0.084, worst_question_recall_at_5=0.536),
    }
    report = build_report(
        [RESULT, RESULT_FIXED_WINDOW],
        FP_CONSISTENT,
        {"drafted": 60},
        golden=[],
        chance_baseline=wrong_measurement,
    )
    # A measurement that exactly matches the pin must NOT trip the mismatch
    # warning -- if it did unconditionally, this would be the test to catch it.
    assert report["chance_baseline"]["mismatch"] is False
    assert "MISMATCH" not in render_markdown(report)


def test_worst_question_mismatch_alone_also_trips_the_tripwire() -> None:
    """The recall@5 mean and the worst-question figure are checked independently."""
    wrong_measurement = {
        "structural": ChanceBaseline(recall_at_5=0.058, worst_question_recall_at_5=0.400),
    }
    report = build_report(
        [RESULT], FP_CONSISTENT, {"drafted": 60}, golden=[], chance_baseline=wrong_measurement
    )
    assert report["chance_baseline"]["mismatch"] is True
    reasons = report["strategies"]["structural"]["chance_baseline_mismatch_reasons"]
    assert any("worst-question" in r and "0.400" in r and "0.375" in r for r in reasons)


# --- chance-baseline drift detection ---


def test_stale_chance_baseline_warns_when_golden_set_has_moved() -> None:
    """FP's golden_sha256 is a dummy value, not the real committed file's hash."""
    report = build_report([RESULT], FP, {"drafted": 60}, golden=[], chance_baseline=CB)
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
        retrieval_code_sha256=FP_CONSISTENT.retrieval_code_sha256,
    )
    report = build_report([RESULT], moved, {"drafted": 60}, golden=[], chance_baseline=CB)
    assert report["chance_baseline"]["stale"] is True
    assert any(
        "indexed corpus has changed" in r for r in report["chance_baseline"]["staleness_reasons"]
    )
    md = render_markdown(report)
    assert "STALE CHANCE BASELINE" in md


def test_consistent_fingerprint_shows_no_staleness_warning() -> None:
    report = build_report([RESULT], FP_CONSISTENT, {"drafted": 60}, golden=[], chance_baseline=CB)
    assert report["chance_baseline"]["stale"] is False
    assert report["chance_baseline"]["staleness_reasons"] == []
    md = render_markdown(report)
    assert "STALE" not in md


# --- provenance: drafted-only ground truth must not read as verified ---


def test_all_drafted_provenance_triggers_an_explicit_warning() -> None:
    md = render_markdown(build_report([RESULT], FP, {"drafted": 65}, golden=[], chance_baseline=CB))
    assert "No question in this golden set has completed human verification" in md


def test_some_verified_provenance_suppresses_the_warning() -> None:
    report = build_report(
        [RESULT], FP, {"drafted": 45, "human_verified": 20}, golden=[], chance_baseline=CB
    )
    md = render_markdown(report)
    assert "No question in this golden set has completed human verification" not in md


# --- source of golden-set-derived numbers must be unambiguous ---


def test_golden_composition_section_labels_its_own_source() -> None:
    md = render_markdown(
        build_report([RESULT], FP, {"drafted": 60}, golden=[_q("num-a")], chance_baseline=CB)
    )
    assert "not from any strategy's retrieval results" in md
