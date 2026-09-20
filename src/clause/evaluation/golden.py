"""The golden set: questions, acceptable answer spans, and their validation.

Ground truth is a set of character spans per question, not a single chunk id.
Much of this corpus is near-identical sanctions-list updates; insisting on one
right answer would punish a retriever that returned an equally correct sibling.
"""

import contextlib
import json
import os
import re
import tempfile
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

_TAG = re.compile(r"[a-z_]+=[a-z_0-9]+")


def parse_tags(notes: str) -> dict[str, str]:
    """Read the `key=value` tags a question carries in its free-text `notes`.

    Some caveats about a question are not properties of its spans and so cannot
    be derived from the corpus -- which retrieval task a `numeric_threshold`
    question really poses, or that its span sits in a document heading a chunker
    may isolate or drop. Carried as prose in a report they get dropped at the
    first hand-off; carried here they are data the renderer groups on, and a
    test can assert they stay complete.

    Tags are whitespace-delimited tokens anywhere in `notes`, so prose and tags
    coexist without a positional convention that would rot. Unknown keys are
    returned as-is: the vocabulary is the caller's business, not the parser's.
    A repeated key keeps its last occurrence.
    """
    tags: dict[str, str] = {}
    for token in notes.split():
        if _TAG.fullmatch(token):
            key, _, value = token.partition("=")
            tags[key] = value
    return tags


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
    """Write questions to a golden set file atomically.

    Writes to a temporary file in the destination directory, then uses os.replace
    to swap it into place atomically. This ensures the original file is never
    truncated or left in a partially-written state if the process crashes or is
    interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file in the same directory so os.replace is atomic
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as tmp_fh:
            tmp_path = tmp_fh.name
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
                tmp_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        # File is now closed; atomically replace the destination with the complete temp file
        os.replace(tmp_path, path)
    except Exception:
        # Clean up the temporary file if something went wrong
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
        raise


def _row_to_question(row: dict[str, object], lineno: int, qid: str) -> GoldenQuestion:
    """Convert a parsed JSONL row to a GoldenQuestion, validating all fields."""
    # Validate bucket and provenance
    if row.get("bucket") not in BUCKETS:
        raise GoldenSetError(f"{qid}: bucket {row.get('bucket')!r} not one of {BUCKETS}")
    if row.get("provenance") not in PROVENANCE:
        raise GoldenSetError(
            f"{qid}: provenance {row.get('provenance')!r} not one of {PROVENANCE}"
        )

    # Validate answers is a list
    raw_answers = row.get("answers")
    if not isinstance(raw_answers, list):
        raise GoldenSetError(f"line {lineno}: answers must be a list")
    if not raw_answers:
        raise GoldenSetError(f"{qid}: needs at least one answer span")

    # Process and validate each answer
    answers: list[Answer] = []
    for a in raw_answers:
        if not isinstance(a, dict):
            raise GoldenSetError(f"line {lineno}: answer entry must be an object")

        # Check required answer keys
        for required_key in ("doc_id", "char_start", "char_end", "content_sha256"):
            if required_key not in a:
                raise GoldenSetError(
                    f"line {lineno}: missing required answer key '{required_key}'"
                )

        # Validate char_start and char_end are integers (not floats or other types)
        char_start_raw = a["char_start"]
        char_end_raw = a["char_end"]

        if not isinstance(char_start_raw, int) or isinstance(char_start_raw, bool):
            raise GoldenSetError(
                f"line {lineno}: char_start must be an integer, "
                f"got {type(char_start_raw).__name__}"
            )
        if not isinstance(char_end_raw, int) or isinstance(char_end_raw, bool):
            raise GoldenSetError(
                f"line {lineno}: char_end must be an integer, "
                f"got {type(char_end_raw).__name__}"
            )

        char_start = char_start_raw
        char_end = char_end_raw

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

    return GoldenQuestion(
        qid=qid,
        question=str(row["question"]),
        bucket=str(row["bucket"]),
        answers=tuple(answers),
        provenance=str(row["provenance"]),
        notes=str(row.get("notes", "")),
    )


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

        try:
            # Check that the parsed row is a dict
            if not isinstance(row, dict):
                raise GoldenSetError(f"line {lineno}: expected an object, got {type(row).__name__}")

            # Validate qid early so we can use it in error messages
            qid_raw = row.get("qid")
            if not isinstance(qid_raw, str):
                raise GoldenSetError(
                    f"line {lineno}: qid must be a string, got {type(qid_raw).__name__}"
                )
            if not qid_raw:
                raise GoldenSetError(f"line {lineno}: qid must be non-empty")
            qid = qid_raw

            # Check required keys are present
            for required_key in ("question", "bucket", "provenance", "answers"):
                if required_key not in row:
                    raise GoldenSetError(f"line {lineno}: missing required key '{required_key}'")

            # Check for duplicate qids
            if qid in seen:
                raise GoldenSetError(f"duplicate qid {qid!r} at line {lineno}")
            seen.add(qid)

            # Convert row to GoldenQuestion
            q = _row_to_question(row, lineno, qid)
            questions.append(q)
        except GoldenSetError:
            # Re-raise GoldenSetError as-is without wrapping
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            # This should not happen after the above checks, but catch it just in case
            raise GoldenSetError(f"line {lineno}: unexpected error: {exc}") from exc

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
    """Count questions by provenance state.

    Returns a dict with provenance states as keys and counts as values.
    Only states that actually occur in the input are included; absent states
    are omitted. Use `.get(state, 0)` to handle sparse data.
    """
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.provenance] = counts.get(q.provenance, 0) + 1
    return counts
