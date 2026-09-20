"""Present drafted golden-set labels to a human and record their judgement.

The model drafted every label in this set. This script is how a human samples
them, so the eval report can state what fraction of ground truth a person
actually checked instead of leaving a reader to assume.

Usage:
    uv run python -m scripts.verify_golden --per-bucket 4
"""

import argparse
import dataclasses
import random
import sys
from collections import defaultdict
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session

from clause.db.schema import DocumentRow
from clause.evaluation.golden import (
    GoldenQuestion,
    load_golden,
    provenance_split,
    write_golden,
)

CONTEXT_CHARS = 300
DECISIONS = ("confirm", "correct", "reject")


def select_sample(
    questions: list[GoldenQuestion], per_bucket: int, seed: int
) -> list[GoldenQuestion]:
    """A deterministic, bucket-balanced sample.

    Balanced because a sample drawn at random over the whole set would
    under-represent whichever bucket is smallest, and the buckets exist precisely
    to expose where retrieval is weak.
    """
    by_bucket: dict[str, list[GoldenQuestion]] = defaultdict(list)
    for q in questions:
        by_bucket[q.bucket].append(q)
    rng = random.Random(seed)
    sample: list[GoldenQuestion] = []
    for bucket in sorted(by_bucket):
        pool = sorted(by_bucket[bucket], key=lambda q: q.qid)
        sample.extend(rng.sample(pool, min(per_bucket, len(pool))))
    return sample


def apply_decision(
    question: GoldenQuestion, decision: str, span: tuple[int, int] | None = None
) -> GoldenQuestion:
    """Apply a human's decision to a golden question.

    Args:
        question: The question to update.
        decision: One of "confirm", "correct", "reject".
        span: Required for "correct": (char_start, char_end).

    Raises:
        ValueError: If decision is unknown or span is invalid.
    """
    if decision == "confirm":
        return dataclasses.replace(question, provenance="human_verified")
    if decision == "correct":
        if span is None:
            raise ValueError("correcting a question needs a replacement span")
        start, end = span
        # Validate span before applying
        if start < 0:
            raise ValueError(f"span start must not be negative: {start}")
        if start >= end:
            raise ValueError(
                f"span must be non-empty and forward: [{start}:{end}]"
            )
        first = question.answers[0]
        corrected = dataclasses.replace(first, char_start=start, char_end=end)
        return dataclasses.replace(
            question, answers=(corrected, *question.answers[1:]), provenance="human_corrected"
        )
    raise ValueError(f"unknown decision {decision!r}; expected one of {DECISIONS}")


def _render(question: GoldenQuestion, texts: dict[str, str]) -> str:
    lines = [f"\n{'=' * 78}", f"{question.qid}  [{question.bucket}]", "", question.question, ""]
    for a in question.answers:
        text = texts.get(a.doc_id, "")
        before = text[max(0, a.char_start - CONTEXT_CHARS) : a.char_start]
        answer = text[a.char_start : a.char_end]
        after = text[a.char_end : a.char_end + CONTEXT_CHARS]
        lines += [
            f"--- {a.doc_id} [{a.char_start}:{a.char_end}] ---",
            f"...{before}",
            f">>> {answer} <<<",
            f"{after}...",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify_golden", description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("data/golden/kyc-v1.jsonl"))
    parser.add_argument("--per-bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--database-url",
        default="postgresql+psycopg://clause:clause@localhost:5434/clause",
    )
    args = parser.parse_args(argv)

    questions = load_golden(args.golden)
    by_qid = {q.qid: q for q in questions}

    engine = sa.create_engine(args.database_url)
    with Session(engine) as session:
        texts = {d.doc_id: d.text for d in session.scalars(sa.select(DocumentRow)).all()}

    sample = select_sample(questions, args.per_bucket, args.seed)
    print(f"Reviewing {len(sample)} of {len(questions)} questions.", file=sys.stderr)
    print("For each: [c]onfirm, [e]dit the span, [s]kip, [q]uit and save.", file=sys.stderr)

    for question in sample:
        print(_render(question, texts))
        choice = input("[c/e/s/q] > ").strip().lower()
        if choice == "q":
            break
        if choice == "c":
            by_qid[question.qid] = apply_decision(question, "confirm")
        elif choice == "e":
            raw = input("replacement span as start:end > ").strip()
            start, _, end = raw.partition(":")
            by_qid[question.qid] = apply_decision(
                question, "correct", span=(int(start), int(end))
            )

    write_golden(args.golden, [by_qid[q.qid] for q in questions])
    print(f"\nprovenance now: {provenance_split(list(by_qid.values()))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
