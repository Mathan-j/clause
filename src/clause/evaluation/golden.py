"""The golden set: questions, acceptable answer spans, and their validation.

Ground truth is a set of character spans per question, not a single chunk id.
Much of this corpus is near-identical sanctions-list updates; insisting on one
right answer would punish a retriever that returned an equally correct sibling.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from clause.evaluation.metrics import Span

BUCKETS: tuple[str, ...] = (
    "definitional",
    "numeric_threshold",
    "procedural",
    "cross_reference",
)

PROVENANCE: tuple[str, ...] = ("drafted", "human_verified", "human_corrected")


class GoldenSetError(Exception):
    """The golden set is unusable. Never downgraded to a warning."""


@dataclass(frozen=True, slots=True)
class Answer:
    doc_id: str
    char_start: int
    char_end: int
    content_sha256: str
    """The document text this span was labelled against. See `validate_spans`."""


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    qid: str
    question: str
    bucket: str
    answers: tuple[Answer, ...]
    provenance: str
    notes: str

    def answer_spans(self) -> tuple[Span, ...]:
        return tuple(
            Span(doc_id=a.doc_id, char_start=a.char_start, char_end=a.char_end)
            for a in self.answers
        )


def write_golden(path: Path, questions: Iterable[GoldenQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for q in questions:
            row = {
                "qid": q.qid,
                "question": q.question,
                "bucket": q.bucket,
                "answers": [
                    {
                        "doc_id": a.doc_id,
                        "char_start": a.char_start,
                        "char_end": a.char_end,
                        "content_sha256": a.content_sha256,
                    }
                    for a in q.answers
                ],
                "provenance": q.provenance,
                "notes": q.notes,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_golden(path: Path) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenSetError(f"line {lineno}: not valid JSON: {exc}") from exc

        qid = row.get("qid", f"<line {lineno}>")
        if row.get("bucket") not in BUCKETS:
            raise GoldenSetError(f"{qid}: bucket {row.get('bucket')!r} not one of {BUCKETS}")
        if row.get("provenance") not in PROVENANCE:
            raise GoldenSetError(
                f"{qid}: provenance {row.get('provenance')!r} not one of {PROVENANCE}"
            )
        raw_answers = row.get("answers") or []
        if not raw_answers:
            raise GoldenSetError(f"{qid}: needs at least one answer span")
        if qid in seen:
            raise GoldenSetError(f"duplicate qid {qid!r} at line {lineno}")
        seen.add(qid)

        # Validate span ordering before constructing Answer objects.
        # This ensures that a malformed span raises GoldenSetError (documented)
        # rather than ValueError from Span.__post_init__ (undocumented).
        answers: list[Answer] = []
        for a in raw_answers:
            char_start = int(a["char_start"])
            char_end = int(a["char_end"])
            if char_start < 0:
                raise GoldenSetError(f"{qid}: char_start must not be negative: {char_start}")
            if char_start >= char_end:
                raise GoldenSetError(
                    f"{qid}: span must be non-empty and forward: "
                    f"[{char_start}:{char_end}]"
                )
            answers.append(
                Answer(
                    doc_id=a["doc_id"],
                    char_start=char_start,
                    char_end=char_end,
                    content_sha256=a["content_sha256"],
                )
            )

        questions.append(
            GoldenQuestion(
                qid=qid,
                question=row["question"],
                bucket=row["bucket"],
                answers=tuple(answers),
                provenance=row["provenance"],
                notes=row.get("notes", ""),
            )
        )
    return questions


def validate_spans(
    questions: Sequence[GoldenQuestion], documents: Mapping[str, tuple[str, str]]
) -> None:
    """Raise unless every span still points where it was labelled.

    `documents` maps `doc_id` to `(text, content_sha256)`.

    This is deliberately fatal rather than tolerant. Phase 1 supplied the lesson:
    when extraction changed, every manifest hash went stale and the whole test
    suite still passed, because a tolerant path absorbed the mismatch -- a green
    result concealing a dead check. Scoring retrieval against offsets that no
    longer point at the labelled text would produce a plausible table measuring
    nothing.
    """
    problems: list[str] = []
    for q in questions:
        for a in q.answers:
            stored = documents.get(a.doc_id)
            if stored is None:
                problems.append(f"{q.qid}: unknown document {a.doc_id!r}")
                continue
            text, sha = stored
            if a.content_sha256 != sha:
                problems.append(
                    f"{q.qid}: span in {a.doc_id} was labelled against document "
                    f"{a.content_sha256[:12]}… but the stored document is {sha[:12]}…"
                )
            elif not 0 <= a.char_start < a.char_end <= len(text):
                problems.append(
                    f"{q.qid}: span [{a.char_start}:{a.char_end}] outside "
                    f"{a.doc_id} of {len(text)} chars"
                )
    if problems:
        raise GoldenSetError(
            "golden set does not match the stored corpus; re-label or re-ingest:\n  "
            + "\n  ".join(problems)
        )


def provenance_split(questions: Sequence[GoldenQuestion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.provenance] = counts.get(q.provenance, 0) + 1
    return counts
