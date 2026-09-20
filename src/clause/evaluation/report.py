"""Render the evaluation into one artifact for humans and one for CI.

Pure rendering: this module touches no database, no Qdrant, and no embedding
model. Its only job is to turn dataclasses that some other layer already
computed (plus the golden set itself, which is a committed file) into
`reports/eval.md` and `reports/eval.json`.

Four things are rendered here that are not obvious from the metrics alone,
because they were ruled on during Phase 2 rather than specified up front (see
the SDD ledger, Task 3's rulings): the `numeric_threshold` bucket's `kind=`
sub-kinds, the `span=heading` questions, the answers-per-question
distribution, and the chance baseline. None of the four requires retrieval
data -- they are properties of the golden set (and, for the chance baseline,
of the corpus it was labelled against), not of any strategy's results. That
is deliberate: a reader must be able to tell a good recall number from an
easy one, and the golden set is where "easy" comes from.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clause.evaluation.golden import GoldenQuestion, parse_tags


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """What produced these numbers.

    CI compares this against the working tree, which is what makes gating a
    committed artifact defensible: without it, "CI fails on a deliberate
    regression" would be satisfiable by simply not re-running the eval.
    """

    manifest_sha256: str
    chunk_counts: dict[str, int]
    embedding_model: str
    retrieval_depth: int
    golden_path: str
    golden_sha256: str
    git_commit: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BucketResult:
    n: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float


@dataclass(frozen=True, slots=True)
class StrategyResult:
    strategy: str
    per_bucket: dict[str, BucketResult]
    overall: BucketResult


PRECISION_NOTE = (
    "`precision@1` is arithmetically identical to `recall@1` under "
    "set-of-acceptable-spans semantics: both are \"did the single top result "
    "hit\". It is reported once rather than printed twice under two names. A "
    "genuine `precision@k` for k > 1 is not computable from this golden set, "
    "which labels answers rather than labelling every chunk."
)

BUCKET_SIZE_NOTE = (
    "Each bucket holds roughly 16 questions, so a single question changing "
    "rank moves that bucket's recall by about 6 percentage points. The `n` "
    "column in every table below is what lets a reader judge how much weight "
    "a per-bucket number can bear -- these figures do not imply precision "
    "beyond what `n` supports."
)

#: Computed once, over the golden set's real per-question answer-span counts
#: against each strategy's actual chunk corpus, during Phase 2 (SDD ledger,
#: Task 3 ruling on "questions with many spans make recall@5 nearly free").
#: This module does not recompute it -- doing so needs the indexed corpus,
#: which a pure renderer does not have -- it only publishes it next to the
#: numbers it exists to put a floor under.
CHANCE_BASELINE_RECALL_AT_5: dict[str, float] = {
    "structural": 0.055,
    "fixed_window": 0.058,
}
CHANCE_BASELINE_WORST = 0.375

CHANCE_BASELINE_NOTE = (
    "Chance floor: the probability that a *random* top-5 already contains an "
    "acceptable span, computed over this golden set's real per-question span "
    "counts against each strategy's actual chunk corpus (SDD ledger, Task 3 "
    "ruling; not recomputed by this renderer). Recall@5 must be read against "
    "this floor, not as a bare number -- recall@5 of 0.800 against a chance "
    "floor of 0.055 is strong evidence, recall@5 of 0.800 against a floor of "
    f"0.375 would not be. The single worst question in the golden set reaches "
    f"a chance floor of {CHANCE_BASELINE_WORST:.3f} on its own, because it "
    "happens to carry an unusually large number of acceptable spans -- a "
    "property of the golden set's labelling, not of either strategy."
)

NO_VERIFICATION_WARNING = (
    "**No question in this golden set has completed human verification.** "
    "Every question is currently `provenance: drafted`: a model wrote the "
    "question and answer spans and a human sampled the result, but no line-by"
    "-line human verification pass has run. Every score below is measured "
    "against drafted ground truth, not verified ground truth -- read it as a "
    "first honest baseline, not as validated fact."
)


def _kind_breakdown(golden: Sequence[GoldenQuestion]) -> dict[str, dict[str, int]]:
    """Count questions by their `kind=` tag, grouped by bucket.

    A bucket like `numeric_threshold` can bundle several different retrieval
    tasks under one name (looking up a percentage vs. counting list entries
    vs. reading off an identifier). Averaging them into one row hides that a
    model can be good at one and bad at another; this is the composition that
    average was computed over.
    """
    breakdown: dict[str, dict[str, int]] = {}
    for q in golden:
        kind = parse_tags(q.notes).get("kind")
        if kind is None:
            continue
        by_kind = breakdown.setdefault(q.bucket, {})
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return breakdown


def _heading_span_qids(golden: Sequence[GoldenQuestion]) -> list[str]:
    """Questions whose ground-truth span is a document heading, not body text."""
    return sorted(q.qid for q in golden if parse_tags(q.notes).get("span") == "heading")


def _answer_count_histogram(golden: Sequence[GoldenQuestion]) -> dict[int, int]:
    """How many acceptable spans each question carries, bucketed by count.

    Ground truth here is a *set* of acceptable spans, so questions are not
    equally hard: a question with 30 acceptable spans is far easier to land
    in a random top-5 than one with a single span. This distribution is the
    input the chance baseline above was computed from.
    """
    hist: dict[int, int] = {}
    for q in golden:
        n = len(q.answer_spans())
        hist[n] = hist.get(n, 0) + 1
    return hist


def build_report(
    results: Sequence[StrategyResult],
    fingerprint: Fingerprint,
    provenance: dict[str, int],
    golden: Sequence[GoldenQuestion] = (),
) -> dict[str, Any]:
    """Assemble the report dict that both artifacts render from.

    `golden` is optional so a caller that only has aggregated `StrategyResult`
    objects (as this module's own acceptance tests do) still gets a valid
    report; when it is supplied, the golden-set-derived sections below are
    populated from it rather than from any strategy's results, and the
    renderer labels them as such.
    """
    kind_breakdown = _kind_breakdown(golden)
    return {
        "fingerprint": fingerprint.to_dict(),
        "golden_provenance": provenance,
        "chance_baseline": {
            "recall_at_5": dict(sorted(CHANCE_BASELINE_RECALL_AT_5.items())),
            "worst_single_question": CHANCE_BASELINE_WORST,
        },
        "golden_composition": {
            "n_questions": len(golden),
            "bucket_kind_breakdown": {
                bucket: dict(sorted(kinds.items()))
                for bucket, kinds in sorted(kind_breakdown.items())
            },
            "heading_span_questions": _heading_span_qids(golden),
            "answers_per_question_histogram": {
                str(k): v for k, v in sorted(_answer_count_histogram(golden).items())
            },
        },
        "strategies": {
            r.strategy: {
                "overall": asdict(r.overall),
                "per_bucket": {b: asdict(v) for b, v in sorted(r.per_bucket.items())},
                "chance_baseline_recall_at_5": CHANCE_BASELINE_RECALL_AT_5.get(r.strategy),
            }
            for r in results
        },
    }


def _row(label: str, r: dict[str, Any]) -> str:
    return (
        f"| {label} | {r['n']} | {r['recall_at_1']:.3f} | {r['recall_at_5']:.3f} "
        f"| {r['recall_at_10']:.3f} | {r['mrr_at_10']:.3f} |"
    )


def render_markdown(report: dict[str, Any]) -> str:
    fp = report["fingerprint"]
    provenance: dict[str, int] = report["golden_provenance"]
    composition = report.get("golden_composition", {})

    verified = sum(n for state, n in provenance.items() if state != "drafted")

    lines = [
        "# Retrieval evaluation",
        "",
        "Generated by `make eval`. Every figure here is produced by that command "
        "from the committed golden set; nothing in this file is estimated.",
        "",
        "## How to read these numbers",
        "",
        f"- Retrieval depth is fixed at {fp['retrieval_depth']}; "
        f"MRR is **MRR@{fp['retrieval_depth']}**.",
        "- A hit means a retrieved chunk's character span overlapped an acceptable "
        "answer span in the same document.",
        f"- {PRECISION_NOTE}",
        f"- {BUCKET_SIZE_NOTE}",
        f"- {CHANCE_BASELINE_NOTE}",
        "",
        "## Ground-truth provenance",
        "",
        "The golden set was drafted by a model and sampled by a human. This is the split:",
        "",
        "| provenance | questions |",
        "|---|---:|",
    ]
    for state, n in sorted(provenance.items()):
        lines.append(f"| {state} | {n} |")
    lines.append("")
    if verified == 0:
        lines.append(NO_VERIFICATION_WARNING)
        lines.append("")

    lines += [
        "## Golden set composition",
        "",
        "_These figures come from the golden set itself, not from any "
        "strategy's retrieval results -- they describe the measurement, not "
        "what was measured._",
        "",
        f"The golden set holds **{composition.get('n_questions', 0)}** questions.",
        "",
    ]

    breakdown: dict[str, dict[str, int]] = composition.get("bucket_kind_breakdown", {})
    lines += ["### `kind=` sub-kinds", ""]
    if breakdown:
        lines.append(
            "A bucket-level average can hide that a model is good at one kind "
            "of question within it and bad at another. Sub-kinds are tagged in "
            "the golden set's `notes` field and read via "
            "`clause.evaluation.golden.parse_tags`:"
        )
        lines.append("")
        for bucket, kinds in breakdown.items():
            lines.append(f"**`{bucket}`** (n={sum(kinds.values())}):")
            lines.append("")
            lines.append("| kind | n |")
            lines.append("|---|---:|")
            for kind, n in sorted(kinds.items()):
                lines.append(f"| {kind} | {n} |")
            lines.append("")
    else:
        lines += ["No question in this golden set carries a `kind=` tag.", ""]

    heading_qids: list[str] = composition.get("heading_span_questions", [])
    lines += ["### `span=heading` questions", ""]
    if heading_qids:
        lines.append(
            f"**{len(heading_qids)}** question(s) have ground-truth spans that "
            "are document headings rather than body text: "
            + ", ".join(f"`{q}`" for q in heading_qids)
            + ". Whether these score as hits swings between trivial and "
            "unanswerable depending on how a chunker treats heading blocks -- "
            "a property of chunking, not of retrieval quality -- and must not "
            "be read as the latter."
        )
    else:
        lines.append("No question in this golden set carries a `span=heading` tag.")
    lines.append("")

    histogram: dict[str, int] = composition.get("answers_per_question_histogram", {})
    lines += [
        "### Answers per question",
        "",
        "Ground truth here is a *set* of acceptable spans, not one right "
        "answer, so questions are not equally hard: a question with many "
        "acceptable spans is easier to land in a random top-5 than one with a "
        "single span. This distribution is what the chance baseline above is "
        "computed from.",
        "",
        "| acceptable spans | questions |",
        "|---:|---:|",
    ]
    for k in sorted(histogram, key=int):
        lines.append(f"| {k} | {histogram[k]} |")
    lines.append("")

    lines += ["## Results", ""]
    for strategy, data in sorted(report["strategies"].items()):
        lines += [f"### {strategy}", ""]
        cb = data.get("chance_baseline_recall_at_5")
        if cb is not None:
            lines.append(
                f"Chance floor for recall@5 on this strategy's chunk corpus: "
                f"**{cb:.3f}** (worst single question in the golden set: "
                f"{CHANCE_BASELINE_WORST:.3f}). Read recall@5 below against "
                "this floor, not as a bare number."
            )
            lines.append("")
        lines += [
            "| bucket | n | recall@1 | recall@5 | recall@10 | MRR@10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for bucket, r in data["per_bucket"].items():
            lines.append(_row(bucket, r))
        lines.append(_row("**overall**", data["overall"]))
        lines.append("")
    lines += [
        "## Provenance fingerprint",
        "",
        "| field | value |",
        "|---|---|",
        f"| embedding model | `{fp['embedding_model']}` |",
        f"| retrieval depth | {fp['retrieval_depth']} |",
        f"| manifest sha256 | `{fp['manifest_sha256'][:16]}…` |",
        f"| golden set | `{fp['golden_path']}` (`{fp['golden_sha256'][:16]}…`) |",
        f"| chunk counts | {fp['chunk_counts']} |",
        f"| git commit | `{fp['git_commit']}` |",
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict[str, Any], md_path: Path, json_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
