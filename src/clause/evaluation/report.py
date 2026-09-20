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
from collections.abc import Mapping, Sequence
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


@dataclass(frozen=True, slots=True)
class ChanceBaseline:
    """The measured probability a uniform random top-5 already contains an
    acceptable chunk, for one strategy, computed fresh against the live
    corpus this run.

    Method (for whoever computes this -- it will be a different implementer,
    since this module has no corpus access): an "acceptable" chunk for a
    question is one whose span overlaps any acceptable answer span **in the
    same document**; a chunk overlapping two acceptable spans still counts
    once, not twice. With `N` chunks in the strategy's collection and `m`
    acceptable chunks for a question, `P = 1 - C(N-m, 5) / C(N, 5)`.
    `recall_at_5` is the mean of that probability over every question in the
    golden set; `worst_question_recall_at_5` is the maximum.

    This is *measured* data, distinct from `CHANCE_BASELINE_RECALL_AT_5` /
    `CHANCE_BASELINE_WORST` below, which are a pinned human-recorded
    reference. The two are compared at render time and any disagreement is
    reported rather than silently preferring one -- a pinned constant can
    itself be wrong (as happened once already; see fix round 2/3 in the SDD
    ledger), and only a live measurement compared against the pin can catch
    that a drift check on the pin's own inputs cannot.
    """

    recall_at_5: float
    worst_question_recall_at_5: float


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

#: PINNED REFERENCE values, human-recorded, not computed by this module (see
#: `ChanceBaseline` above for the method and for how the *measured* figure
#: `build_report` receives each run relates to these). They exist so a reader
#: -- and `_chance_baseline_mismatch` below -- has something fixed to compare
#: a fresh measurement against; moving them is a deliberate human act, the
#: same philosophy as Task 11's `reports/baseline.json`, precisely because a
#: harness that quietly re-pins its own reference every run cannot detect a
#: regression in it.
#:
#: Corrected during Task 9 fix round 2: the value first recorded for
#: `fixed_window` was 0.058, which understated its real 0.084 by about 45% and
#: had the two strategies backwards -- `fixed_window` has fewer chunks (318 vs
#: 336) but a higher share of them overlap an answer span, so its floor is the
#: *higher* of the two, not the lower. Fix round 3 stopped reading these as a
#: live value at render time -- they could not have caught their own original
#: error, since the drift check below only catches the golden set or corpus
#: *changing*, not the original computation being wrong -- and made them a
#: pinned reference that an independently measured value is checked against
#: instead.
CHANCE_BASELINE_RECALL_AT_5: dict[str, float] = {
    "structural": 0.058,
    "fixed_window": 0.084,
}

#: The single worst (highest-floor) question's pinned chance figure, per
#: strategy. Both strategies' worst is `proc-002` (30 acceptable chunks under
#: `structural`, 45 under `fixed_window`), but the resulting floor differs
#: by strategy and must be presented as such, not as one shared number --
#: that was the other half of the fix round 2 correction.
CHANCE_BASELINE_WORST: dict[str, float] = {
    "structural": 0.375,
    "fixed_window": 0.536,
}

#: The golden-set file and corpus the pinned constants above were computed
#: against (SDD ledger, Task 3 ruling). `golden_sha256` is
#: `sha256(data/golden/kyc-v1.jsonl)` as committed; `chunk_counts` is the
#: `clause_structural` / `clause_fixed_window` collection sizes at that time.
#: If the fingerprint of the report being rendered disagrees with either, the
#: pin's own stated inputs no longer describe this run, and
#: `_chance_baseline_staleness` below turns that into a warning -- distinct
#: from, and in addition to, `_chance_baseline_mismatch`'s check of whether
#: the pinned *value* agrees with what was actually measured this run.
CHANCE_BASELINE_GOLDEN_SHA256 = "c138b258815f3fa2b4b7c2f94855454207494c3c63520d9e7558ab2851f2e03f"
CHANCE_BASELINE_CHUNK_COUNTS: dict[str, int] = {
    "structural": 336,
    "fixed_window": 318,
}

CHANCE_BASELINE_NOTE = (
    "Chance floor: the probability that a *random* top-5 already contains an "
    "acceptable chunk, over this golden set's real per-question "
    "acceptable-chunk counts against each strategy's actual chunk corpus. "
    "Recall@5 must be read against this floor, not as a bare number: a "
    "strong recall@5 against a low chance floor is real evidence of "
    "retrieval quality, the same recall@5 against a high floor may not be. "
    "The floor differs by strategy -- a strategy with fewer chunks that "
    "overlap answer spans less often can still have a *higher* floor than "
    "one with more chunks, if a larger share of its chunks overlap an "
    "answer. Three distinct figures appear below for each strategy: the "
    "floor **measured this run** against the live corpus, the floor "
    "**pinned** as a human-recorded reference value (SDD ledger, Task 3 "
    "method), and a warning if either the two disagree or the pin's own "
    "recorded inputs no longer match this run's golden set or corpus."
)

FILTER_COVERAGE_NOTE = (
    "**Metadata filters are not exercised by this evaluation.** `evaluate()` "
    "calls `search()` with no `published_after`/`published_before`/`entities` "
    "filter for every question in this golden set, because none of these "
    "questions are filtered queries. The date-range and regulated-entity "
    "filters built for retrieval are real and tested in isolation "
    "(`tests/test_retrieve_filter.py`), but their effect on recall -- whether "
    "a filtered query's top-k performs the same as, better than, or worse "
    "than an unfiltered one -- is not measured anywhere in this report. Read "
    "every number below as unfiltered-query recall only."
)

NO_VERIFICATION_WARNING = (
    "**No question in this golden set has completed human verification.** "
    "Every question is currently `provenance: drafted`: a model wrote the "
    "question and answer spans and a human sampled the result, but no line-by"
    "-line human verification pass has run. Every score below is measured "
    "against drafted ground truth, not verified ground truth -- read it as a "
    "first honest baseline, not as validated fact."
)


def _chance_baseline_staleness(fingerprint: Fingerprint) -> list[str]:
    """Reasons the chance-baseline constants no longer match this run, if any.

    Empty means the constants still describe what is being measured. Both
    checks compare against the fingerprint rather than anything recomputed --
    this module has no corpus access to recompute the floor itself.
    """
    reasons: list[str] = []
    if fingerprint.golden_sha256 != CHANCE_BASELINE_GOLDEN_SHA256:
        reasons.append(
            "the golden set has changed since the chance baseline was computed "
            f"(this run's golden_sha256 is `{fingerprint.golden_sha256[:16]}...`, "
            f"the chance baseline was computed against "
            f"`{CHANCE_BASELINE_GOLDEN_SHA256[:16]}...`)"
        )
    if fingerprint.chunk_counts != CHANCE_BASELINE_CHUNK_COUNTS:
        reasons.append(
            "the indexed corpus has changed since the chance baseline was computed "
            f"(this run's chunk counts are {fingerprint.chunk_counts}, the chance "
            f"baseline was computed against {CHANCE_BASELINE_CHUNK_COUNTS})"
        )
    return reasons


def _chance_baseline_mismatch(strategy: str, measured: ChanceBaseline) -> list[str]:
    """Reasons this run's measured chance baseline disagrees with the pin, if any.

    Distinct from `_chance_baseline_staleness`: that function asks whether the
    pin's own stated inputs (golden set, corpus) still match this run. This
    function asks whether the pinned *value* agrees with what was actually
    measured -- which can fail even when the inputs match, if the pinned
    value was wrong from the moment it was recorded. Rounded to 3 decimals
    before comparing, since the pins are only recorded to that precision.
    """
    reasons: list[str] = []
    pinned_recall = CHANCE_BASELINE_RECALL_AT_5.get(strategy)
    if pinned_recall is not None and round(measured.recall_at_5, 3) != round(pinned_recall, 3):
        reasons.append(
            f"measured recall@5 floor is {measured.recall_at_5:.3f} but the "
            f"pinned reference is {pinned_recall:.3f}"
        )
    pinned_worst = CHANCE_BASELINE_WORST.get(strategy)
    if pinned_worst is not None and round(
        measured.worst_question_recall_at_5, 3
    ) != round(pinned_worst, 3):
        reasons.append(
            f"measured worst-question floor is "
            f"{measured.worst_question_recall_at_5:.3f} but the pinned "
            f"reference is {pinned_worst:.3f}"
        )
    return reasons


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
    golden: Sequence[GoldenQuestion],
    chance_baseline: Mapping[str, ChanceBaseline],
) -> dict[str, Any]:
    """Assemble the report dict that both artifacts render from.

    `golden` and `chance_baseline` are both required, not defaulted: a caller
    that forgot to load the golden set, or forgot to measure the chance
    baseline against the live corpus, must fail loudly at the call site
    rather than silently ship a report with those sections empty or with a
    stale pinned value standing in unchallenged for a live measurement.
    """
    kind_breakdown = _kind_breakdown(golden)
    staleness = _chance_baseline_staleness(fingerprint)
    mismatches: dict[str, list[str]] = {
        strategy: reasons
        for strategy, measured in chance_baseline.items()
        if (reasons := _chance_baseline_mismatch(strategy, measured))
    }
    return {
        "fingerprint": fingerprint.to_dict(),
        "golden_provenance": provenance,
        "chance_baseline": {
            "measured": {
                strategy: asdict(cb) for strategy, cb in sorted(chance_baseline.items())
            },
            "pinned": {
                "recall_at_5": dict(sorted(CHANCE_BASELINE_RECALL_AT_5.items())),
                "worst_single_question": dict(sorted(CHANCE_BASELINE_WORST.items())),
            },
            "computed_against": {
                "golden_sha256": CHANCE_BASELINE_GOLDEN_SHA256,
                "chunk_counts": dict(sorted(CHANCE_BASELINE_CHUNK_COUNTS.items())),
            },
            "stale": bool(staleness),
            "staleness_reasons": staleness,
            "mismatch": bool(mismatches),
            "mismatch_reasons": [
                f"{strategy}: {reason}"
                for strategy, reasons in sorted(mismatches.items())
                for reason in reasons
            ],
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
                "chance_baseline_recall_at_5": (
                    chance_baseline[r.strategy].recall_at_5
                    if r.strategy in chance_baseline
                    else None
                ),
                "chance_baseline_worst_question": (
                    chance_baseline[r.strategy].worst_question_recall_at_5
                    if r.strategy in chance_baseline
                    else None
                ),
                "chance_baseline_pinned_recall_at_5": CHANCE_BASELINE_RECALL_AT_5.get(
                    r.strategy
                ),
                "chance_baseline_pinned_worst_question": CHANCE_BASELINE_WORST.get(
                    r.strategy
                ),
                "chance_baseline_mismatch_reasons": mismatches.get(r.strategy, []),
            }
            for r in results
        },
    }


def _row(label: str, r: dict[str, Any]) -> str:
    """Recall is rounded to 2 decimals, not 3.

    A bucket of ~16-17 questions can only land on multiples of 1/n -- a
    third decimal implies a resolution the corpus doesn't have, which is
    exactly what `BUCKET_SIZE_NOTE` warns against two sections above. Two
    decimals still distinguishes every attainable value at these bucket
    sizes.
    """
    return (
        f"| {label} | {r['n']} | {r['recall_at_1']:.2f} | {r['recall_at_5']:.2f} "
        f"| {r['recall_at_10']:.2f} | {r['mrr_at_10']:.3f} |"
    )


def _staleness_warning(chance_baseline: dict[str, Any]) -> str | None:
    reasons: list[str] = chance_baseline.get("staleness_reasons", [])
    if not reasons:
        return None
    return (
        "**STALE CHANCE BASELINE PIN:** the pinned chance-baseline figures "
        "below no longer describe this run -- " + "; ".join(reasons) + ". "
        "Recompute the pin (SDD ledger, Task 3 method) before trusting it."
    )


def _mismatch_warning(chance_baseline: dict[str, Any]) -> str | None:
    reasons: list[str] = chance_baseline.get("mismatch_reasons", [])
    if not reasons:
        return None
    return (
        "**CHANCE BASELINE MISMATCH:** this run's measured chance baseline "
        "disagrees with the pinned reference value -- " + "; ".join(reasons) + ". "
        "The pin may have been wrong when it was recorded, independent of "
        "whether the golden set or corpus have since changed; recompute and "
        "update it (SDD ledger, Task 3 method) rather than trusting either "
        "number blindly."
    )


def _render_header(
    fp: dict[str, Any], staleness_warning: str | None, mismatch_warning: str | None
) -> list[str]:
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
    ]
    if mismatch_warning is not None:
        lines.append(f"- {mismatch_warning}")
    if staleness_warning is not None:
        lines.append(f"- {staleness_warning}")
    lines.append("")
    return lines


def _render_limitations() -> list[str]:
    return [
        "## Limitations",
        "",
        FILTER_COVERAGE_NOTE,
        "",
    ]


def _render_provenance(provenance: dict[str, int]) -> list[str]:
    verified = sum(n for state, n in provenance.items() if state != "drafted")
    lines = [
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
        lines += [NO_VERIFICATION_WARNING, ""]
    return lines


def _render_kind_breakdown(breakdown: dict[str, dict[str, int]]) -> list[str]:
    lines = ["### `kind=` sub-kinds", ""]
    if not breakdown:
        return [*lines, "No question in this golden set carries a `kind=` tag.", ""]
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
    return lines


def _render_heading_questions(heading_qids: list[str]) -> list[str]:
    lines = ["### `span=heading` questions", ""]
    if not heading_qids:
        lines.append("No question in this golden set carries a `span=heading` tag.")
        lines.append("")
        return lines
    lines.append(
        f"**{len(heading_qids)}** question(s) have ground-truth spans that "
        "are document headings rather than body text: "
        + ", ".join(f"`{q}`" for q in heading_qids)
        + ". Whether these score as hits swings between trivial and "
        "unanswerable depending on how a chunker treats heading blocks -- "
        "a property of chunking, not of retrieval quality -- and must not "
        "be read as the latter."
    )
    lines.append("")
    return lines


def _render_answer_histogram(histogram: dict[str, int]) -> list[str]:
    lines = [
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
    return lines


def _render_composition(composition: dict[str, Any]) -> list[str]:
    lines = [
        "## Golden set composition",
        "",
        "_These figures come from the golden set itself, not from any "
        "strategy's retrieval results -- they describe the measurement, not "
        "what was measured._",
        "",
        f"The golden set holds **{composition.get('n_questions', 0)}** questions.",
        "",
    ]
    lines += _render_kind_breakdown(composition.get("bucket_kind_breakdown", {}))
    lines += _render_heading_questions(composition.get("heading_span_questions", []))
    lines += _render_answer_histogram(composition.get("answers_per_question_histogram", {}))
    return lines


def _render_strategy(
    strategy: str,
    data: dict[str, Any],
    staleness_warning: str | None,
) -> list[str]:
    lines = [f"### {strategy}", ""]
    measured = data.get("chance_baseline_recall_at_5")
    measured_worst = data.get("chance_baseline_worst_question")
    pinned = data.get("chance_baseline_pinned_recall_at_5")
    pinned_worst = data.get("chance_baseline_pinned_worst_question")
    mismatch_reasons: list[str] = data.get("chance_baseline_mismatch_reasons", [])
    if measured is not None:
        measured_worst_clause = f"{measured_worst:.3f}" if measured_worst is not None else "n/a"
        pinned_clause = f"{pinned:.3f}" if pinned is not None else "not pinned"
        pinned_worst_clause = f"{pinned_worst:.3f}" if pinned_worst is not None else "not pinned"
        floor_line = (
            "Chance floor for recall@5, computed once over the **whole "
            "golden set** (not per bucket), on this strategy's chunk "
            f"corpus. Measured this run: **{measured:.3f}** (this run's "
            f"worst single question: {measured_worst_clause}). Pinned "
            f"reference: {pinned_clause} (worst: {pinned_worst_clause}). "
            "This floor applies only to the **overall** row below -- "
            "per-bucket chance floors were not computed, and a bucket's "
            "recall@5 must not be read against this aggregate figure."
        )
        if mismatch_reasons:
            floor_line += (
                " **CHANCE BASELINE MISMATCH:** " + "; ".join(mismatch_reasons) + ". "
                "Recompute and update the pin (SDD ledger, Task 3 method) "
                "before trusting either number."
            )
        if staleness_warning is not None:
            floor_line += " " + staleness_warning
        lines.append(floor_line)
        lines.append("")
    lines += [
        "| bucket | n | recall@1 | recall@5 | recall@10 | MRR@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bucket, r in data["per_bucket"].items():
        lines.append(_row(bucket, r))
    lines.append(_row("**overall**", data["overall"]))
    lines.append("")
    return lines


def _render_comparison(report: dict[str, Any]) -> list[str]:
    """Both strategies' raw numbers and their own live-measured chance floor,
    side by side, before either strategy gets its own section.

    Facts only -- no derived ratio, no asserted ordering. A bare recall
    ranking and a chance-floor-normalised one can disagree (one strategy can
    lead on raw recall@5 while trailing the other relative to its own floor),
    and reading four numbers split across two later sections does not
    surface that on its own. Putting all of them in one table, before the
    per-strategy detail, does.
    """
    lines = [
        "### Strategy comparison",
        "",
        "A bare recall ranking and a chance-floor-normalised one can "
        "disagree -- one strategy can lead on raw recall while the other "
        "leads once each is read against its own chance floor. This table "
        "puts both strategies' overall numbers and their own live-measured "
        "chance floor in one place, before either gets its own section "
        "below, so that comparison is visible rather than requiring the "
        "reader to hold four numbers across two sections.",
        "",
        "| strategy | n | recall@1 | recall@5 | recall@10 | MRR@10 "
        "| chance floor (recall@5) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, data in sorted(report["strategies"].items()):
        overall = data["overall"]
        floor = data.get("chance_baseline_recall_at_5")
        floor_clause = f"{floor:.3f}" if floor is not None else "n/a"
        lines.append(
            f"| {strategy} | {overall['n']} | {overall['recall_at_1']:.2f} "
            f"| {overall['recall_at_5']:.2f} | {overall['recall_at_10']:.2f} "
            f"| {overall['mrr_at_10']:.3f} | {floor_clause} |"
        )
    lines.append("")
    return lines


CROSS_REFERENCE_NOTE = (
    "**Interpretation, not measurement:** `cross_reference` questions are "
    "two-hop -- the acceptable answer text (an amendment, a cross-cited "
    "provision) typically lives in a document other than the one whose "
    "wording the question echoes, so the answer can share little lexical or "
    "semantic overlap with the query itself. A single dense-retrieval query "
    "with no decomposition or reranking is poorly suited to that gap, and "
    "this is largely independent of which chunker produced the corpus -- "
    "both strategies show the same pattern on this bucket. This is a "
    "reading of *why* the number below is low, not a separate measurement, "
    "and it should not be over-read: at n=16, the honest statement is that "
    "both strategies are weaker on this small, hard slice, not a precise "
    "gap."
)


def _render_results(report: dict[str, Any], staleness_warning: str | None) -> list[str]:
    lines = ["## Results", ""]
    lines += _render_comparison(report)
    lines += [CROSS_REFERENCE_NOTE, ""]
    for strategy, data in sorted(report["strategies"].items()):
        lines += _render_strategy(strategy, data, staleness_warning)
    return lines


def _render_fingerprint(fp: dict[str, Any]) -> list[str]:
    lines = [
        "## Provenance fingerprint",
        "",
        "| field | value |",
        "|---|---|",
        f"| embedding model | `{fp['embedding_model']}` |",
        f"| retrieval depth | {fp['retrieval_depth']} |",
        f"| manifest sha256 | `{fp['manifest_sha256'][:16]}...` |",
        f"| golden set | `{fp['golden_path']}` (`{fp['golden_sha256'][:16]}...`) |",
    ]
    for strategy, count in sorted(fp["chunk_counts"].items()):
        lines.append(f"| chunk count ({strategy}) | {count} |")
    lines.append(f"| git commit | `{fp['git_commit']}` |")
    lines.append("")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    fp = report["fingerprint"]
    provenance: dict[str, int] = report["golden_provenance"]
    composition = report.get("golden_composition", {})
    chance_baseline = report.get("chance_baseline", {})
    staleness_warning = _staleness_warning(chance_baseline)
    mismatch_warning = _mismatch_warning(chance_baseline)

    lines = _render_header(fp, staleness_warning, mismatch_warning)
    lines += _render_limitations()
    lines += _render_provenance(provenance)
    lines += _render_composition(composition)
    lines += _render_results(report, staleness_warning)
    lines += _render_fingerprint(fp)
    return "\n".join(lines)


def write_report(report: dict[str, Any], md_path: Path, json_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
