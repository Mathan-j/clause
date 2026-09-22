"""Out-of-corpus questions, for measuring refusal.

Weighted toward near-misses that retrieve well. A question the retriever scores
near zero refuses at every threshold, so a set built from those would report a
high refusal rate while testing nothing about where the threshold should sit.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

KINDS: tuple[str, ...] = (
    "adjacent_domain",
    "uncovered_kyc",
    "time_shifted",
    "wrong_jurisdiction",
)


@dataclass(frozen=True, slots=True)
class AdversarialQuestion:
    qid: str
    question: str
    kind: str
    why_unanswerable: str


def _rows(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if line.strip():
                yield lineno, json.loads(line)


def load_adversarial(path: Path) -> list[AdversarialQuestion]:
    questions: list[AdversarialQuestion] = []
    for lineno, row in _rows(path):
        missing = {"qid", "question", "kind", "why_unanswerable"} - set(row)
        if missing:
            raise ValueError(f"line {lineno}: missing {sorted(missing)}")
        kind = str(row["kind"])
        if kind not in KINDS:
            raise ValueError(f"line {lineno}: unknown kind {kind!r}, expected one of {KINDS}")
        questions.append(
            AdversarialQuestion(
                qid=str(row["qid"]),
                question=str(row["question"]),
                kind=kind,
                why_unanswerable=str(row["why_unanswerable"]),
            )
        )
    return questions
