import json
from pathlib import Path

from clause.evaluation.answer_metrics import SupportDistribution
from clause.evaluation.answer_report import (
    AnswerFingerprint,
    ThresholdRow,
    build_answer_report,
    render_answer_markdown,
    write_answer_report,
)

FP = AnswerFingerprint(
    model_file="qwen2.5-3b-instruct-q4_k_m.gguf",
    model_sha256="m" * 64,
    n_ctx=4096,
    threshold=0.35,
    golden_sha256="g" * 64,
    adversarial_sha256="a" * 64,
    corpus_documents=61,
    git_commit="abc1234",
)

ROWS = (
    ThresholdRow(
        0.0, answered=65, refused=0, false_refusals=0,
        adversarial_refused=0, adversarial_answered=30,
    ),
    ThresholdRow(
        0.5, answered=20, refused=45, false_refusals=8,
        adversarial_refused=22, adversarial_answered=8,
    ),
)

SUPPORT = SupportDistribution(n=65, mean=0.61, at_or_above_half=44, zero=6)


def test_the_report_carries_its_fingerprint() -> None:
    r = build_answer_report(ROWS, FP, SUPPORT, resolution_rate=1.0)
    assert r["fingerprint"]["model_sha256"] == "m" * 64


def test_the_report_states_the_support_signal_is_not_faithfulness() -> None:
    md = render_answer_markdown(build_answer_report(ROWS, FP, SUPPORT, resolution_rate=1.0))
    assert "not entailment" in md
    assert "faithfulness score" not in md.replace("not a faithfulness score", "")


def test_the_report_renders_the_whole_sweep() -> None:
    md = render_answer_markdown(build_answer_report(ROWS, FP, SUPPORT, resolution_rate=1.0))
    assert "0.00" in md and "0.50" in md


def test_a_resolution_rate_below_one_is_called_out_loudly() -> None:
    """Below 1.0 means the contract REJECTED drafts -- it working, not failing.

    A returned `Answer` with an unresolvable citation cannot exist, because
    `enforce` raises before one is constructed. So this figure measures the
    generator, and the report must not call it a breach of the contract.
    """
    md = render_answer_markdown(build_answer_report(ROWS, FP, SUPPORT, resolution_rate=0.98))
    assert "rejected" in md.lower()
    assert "contract working, not failing" in md
    assert "CITATION CONTRACT BREACH" not in md


def test_a_resolution_rate_of_one_says_so_without_alarm() -> None:
    md = render_answer_markdown(build_answer_report(ROWS, FP, SUPPORT, resolution_rate=1.0))
    assert "CITATION CONTRACT BREACH" not in md
    assert "rejected" not in md.lower()


def test_write_emits_both_artifacts(tmp_path: Path) -> None:
    r = build_answer_report(ROWS, FP, SUPPORT, resolution_rate=1.0)
    md, js = tmp_path / "answers.md", tmp_path / "answers.json"
    write_answer_report(r, md, js)
    assert "sweep" in md.read_text(encoding="utf-8").lower()
    assert json.loads(js.read_text(encoding="utf-8"))["fingerprint"]["git_commit"] == "abc1234"
