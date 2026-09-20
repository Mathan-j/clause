# Phase 2: Retrieval and Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `make eval` produces `reports/eval.md` and `reports/eval.json` measuring dense retrieval over both chunking strategies against a 60+ question golden set, and CI fails when a committed report regresses or goes stale.

**Architecture:** Pure metric functions first, so the semantics are fixed and CI-runnable before any infrastructure exists. Ground truth is character spans, not chunk ids, so one golden set serves both strategies and survives re-chunking. One Qdrant collection per strategy, because ANN recall depends on index contents and mixing them would make each strategy's number a function of the other.

**Tech Stack:** Python 3.12, sentence-transformers (ONNX), onnxruntime, qdrant-client, SQLAlchemy, Alembic, pytest, ruff, mypy, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-20-retrieval-and-evaluation-design.md`

## Global Constraints

- Python `>=3.12,<3.13`.
- **No number appears in documentation unless it came from a committed file under `reports/`.** Phase 2 creates `reports/`; until `make eval` runs, no figure may be quoted as a result.
- **Retrieval depth is fixed at 10** for every evaluation run. MRR is reported as **MRR@10**, never bare "MRR".
- **`precision@1` is arithmetically identical to `recall@1`** under set-of-acceptable-spans semantics. Print one number under both names with a footnote stating the identity. Never print it twice as if it were two pieces of evidence.
- **Ground truth is character spans**, not chunk ids: `{doc_id, char_start, char_end, content_sha256}`. A hit is any returned chunk's span overlapping any acceptable span within the same `doc_id`.
- **A golden-set span whose `content_sha256` no longer matches the stored document is a hard failure.** `make eval` refuses to run. Never score against stale offsets.
- **The embedding model must never download mid-measurement.** If it is not in the local cache, `make eval` fails with the command that warms it.
- **`reports/baseline.json` is created and updated by an explicit human act**, never automatically by `make eval`.
- `doc_type` is **retired** as a retrieval filter. Do not expose it.
- Tests first for anything in the retrieval or citation path — that is all of Phase 2 except the report renderer.
- **No test may make a network request to `rbi.org.in`.** The corpus is settled.
- Every new dependency gets one line in `docs/decisions.md` saying what it replaced and why.
- Conventional commits. The body explains why, not what.
- Qdrant runs on host port **6335** (6333 is occupied by an unrelated project). Postgres is on **5434**.
- **`make` is not installed on the development machine** — use `uv run ...` and `docker compose ...`.

---

## File Structure

```
src/clause/
  entities.py                  parse regulated entities from the addressee block
  embed.py                     Encoder — ONNX sentence-transformers
  index.py                     Qdrant collection lifecycle + upsert
  retrieve.py                  dense search + date/entity filters
  evaluation/
    __init__.py
    metrics.py                 overlap, first_hit_rank, recall_at_k, mrr   (PURE)
    golden.py                  GoldenQuestion schema, loader, span validation
    report.py                  Fingerprint, build/render/write
    gate.py                    baseline comparison + staleness check
  cli.py                       (modify) add `index` and `eval` subcommands

migrations/versions/0003_regulated_entity.py
data/golden/kyc-v1.jsonl       committed golden set
scripts/verify_golden.py       human verification workflow
reports/                       eval.md + eval.json (generated), baseline.json (human-committed)
docker-compose.yml             (modify) qdrant service on 6335
Makefile                       (modify) index, eval targets
.github/workflows/ci.yml       (modify) run the gate
.env.example                   (modify) CLAUSE_QDRANT_URL, CLAUSE_EMBEDDING_MODEL

tests/
  test_metrics.py  test_golden.py  test_entities.py  test_embed.py
  test_index.py    test_retrieve.py  test_report.py  test_gate.py
  test_eval_end_to_end.py
```

**Why `evaluation/metrics.py` is pure:** it is the only part of Phase 2 CI can execute — no database, no Qdrant, no model. Keeping it dependency-free is what makes the CI gate meaningful rather than decorative.

**Task order rationale:** metrics and the golden set come before any infrastructure, so the human verification you owe (Task 4) can proceed in parallel while the embedding and indexing work is built.

---

## Task 1: Pure metrics

**Files:**
- Create: `src/clause/evaluation/__init__.py`, `src/clause/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Span(doc_id: str, char_start: int, char_end: int)` — frozen dataclass
  - `Retrieved(doc_id: str, char_start: int, char_end: int, score: float)` — frozen dataclass
  - `RETRIEVAL_DEPTH: int = 10`
  - `overlaps(a: Span, b: Span) -> bool`
  - `first_hit_rank(results: Sequence[Retrieved], answers: Sequence[Span]) -> int | None` — 1-based rank, `None` if no hit
  - `recall_at_k(ranks: Sequence[int | None], k: int) -> float`
  - `mrr(ranks: Sequence[int | None], cutoff: int = RETRIEVAL_DEPTH) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import pytest

from clause.evaluation.metrics import (
    RETRIEVAL_DEPTH,
    Retrieved,
    Span,
    first_hit_rank,
    mrr,
    overlaps,
    recall_at_k,
)


def _span(start: int, end: int, doc: str = "d1") -> Span:
    return Span(doc_id=doc, char_start=start, char_end=end)


def _hit(start: int, end: int, doc: str = "d1", score: float = 1.0) -> Retrieved:
    return Retrieved(doc_id=doc, char_start=start, char_end=end, score=score)


def test_overlapping_spans_overlap() -> None:
    assert overlaps(_span(100, 200), _span(150, 250))


def test_containment_counts_as_overlap() -> None:
    assert overlaps(_span(100, 200), _span(120, 180))
    assert overlaps(_span(120, 180), _span(100, 200))


def test_touching_spans_do_not_overlap() -> None:
    """[100,200) and [200,300) share no character. Half-open intervals."""
    assert not overlaps(_span(100, 200), _span(200, 300))


def test_a_single_shared_character_is_an_overlap() -> None:
    assert overlaps(_span(100, 200), _span(199, 300))


def test_spans_in_different_documents_never_overlap() -> None:
    assert not overlaps(_span(100, 200, "d1"), _span(100, 200, "d2"))


def test_first_hit_rank_is_one_based() -> None:
    results = [_hit(0, 50), _hit(100, 200), _hit(300, 400)]
    assert first_hit_rank(results, [_span(150, 160)]) == 2


def test_first_hit_rank_takes_the_earliest_hit() -> None:
    results = [_hit(100, 200), _hit(300, 400)]
    assert first_hit_rank(results, [_span(350, 360), _span(150, 160)]) == 1


def test_first_hit_rank_is_none_when_nothing_hits() -> None:
    assert first_hit_rank([_hit(0, 50)], [_span(900, 950)]) is None


def test_first_hit_rank_with_no_results_is_none() -> None:
    assert first_hit_rank([], [_span(0, 10)]) is None


def test_duplicate_acceptable_spans_do_not_change_the_rank() -> None:
    results = [_hit(0, 50), _hit(100, 200)]
    once = first_hit_rank(results, [_span(150, 160)])
    twice = first_hit_rank(results, [_span(150, 160), _span(150, 160)])
    assert once == twice == 2


def test_recall_at_k_counts_questions_hit_within_k() -> None:
    ranks = [1, 3, None, 7]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.5
    assert recall_at_k(ranks, 10) == 0.75


def test_recall_of_an_empty_golden_set_is_zero_not_a_crash() -> None:
    assert recall_at_k([], 5) == 0.0


def test_mrr_averages_reciprocal_ranks() -> None:
    assert mrr([1, 2, None]) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_mrr_scores_a_rank_beyond_the_cutoff_as_zero() -> None:
    assert mrr([11], cutoff=10) == 0.0
    assert mrr([10], cutoff=10) == pytest.approx(0.1)


def test_retrieval_depth_is_ten() -> None:
    """Pinned by the spec: every metric is computed from one list of 10."""
    assert RETRIEVAL_DEPTH == 10
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest tests/test_metrics.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.evaluation'`

- [ ] **Step 3: Implement**

```python
# src/clause/evaluation/metrics.py
"""Retrieval metrics over character spans.

Deliberately free of any dependency on the database, Qdrant or the embedding
model: this is the one part of Phase 2 that CI can execute, which is what makes
the regression gate meaningful rather than decorative.
"""

from collections.abc import Sequence
from dataclasses import dataclass

#: Every evaluation run retrieves exactly this many results, and every metric is
#: computed from that one ranked list, so the figures in a row stay mutually
#: consistent. Recorded in the report fingerprint so a change to it is visible.
RETRIEVAL_DEPTH = 10


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character range `[char_start, char_end)` within one document."""

    doc_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class Retrieved:
    doc_id: str
    char_start: int
    char_end: int
    score: float


def overlaps(a: Span, b: Span) -> bool:
    """True when two spans share at least one character of the same document.

    Overlap rather than containment: a chunk boundary that splits the answer in
    half still retrieved the answer, and scoring it as a miss would penalise the
    chunker for the golden set's labelling choices.
    """
    if a.doc_id != b.doc_id:
        return False
    return a.char_start < b.char_end and b.char_start < a.char_end


def first_hit_rank(results: Sequence[Retrieved], answers: Sequence[Span]) -> int | None:
    """1-based rank of the first result overlapping any acceptable span.

    `None` when nothing in `results` hits. Ranks are 1-based because that is what
    MRR's reciprocal expects; a 0-based rank would make the top result infinite.
    """
    for rank, result in enumerate(results, start=1):
        span = Span(doc_id=result.doc_id, char_start=result.char_start, char_end=result.char_end)
        if any(overlaps(span, answer) for answer in answers):
            return rank
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of questions whose first hit falls within the top k."""
    if not ranks:
        return 0.0
    hit = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hit / len(ranks)


def mrr(ranks: Sequence[int | None], cutoff: int = RETRIEVAL_DEPTH) -> float:
    """Mean reciprocal rank, scoring anything past `cutoff` as zero."""
    if not ranks:
        return 0.0
    total = sum(1.0 / rank for rank in ranks if rank is not None and rank <= cutoff)
    return total / len(ranks)
```

Create an empty `src/clause/evaluation/__init__.py`.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_metrics.py -v && uv run ruff check . && uv run mypy
```
Expected: 15 passed, lint and types clean.

- [ ] **Step 5: Commit**

```bash
git add src/clause/evaluation/ tests/test_metrics.py
git commit -m "feat(eval): pure retrieval metrics over character spans

Spans rather than chunk ids because chunk_id renumbers on every ingest
and ordinals shift whenever a chunking parameter changes -- which is what
Phase 2 exists to experiment with. Overlap rather than containment so a
chunk boundary that splits an answer still counts as having found it.

Kept free of database, Qdrant and model dependencies: this is the only
part of the phase CI can run, which is what makes the gate meaningful."
```

---

## Task 2: Golden-set schema, loader and span validation

**Files:**
- Create: `src/clause/evaluation/golden.py`, `tests/test_golden.py`

**Interfaces:**
- Consumes: `Span` from `clause.evaluation.metrics`
- Produces:
  - `BUCKETS: tuple[str, ...] = ("definitional", "numeric_threshold", "procedural", "cross_reference")`
  - `PROVENANCE: tuple[str, ...] = ("drafted", "human_verified", "human_corrected")`
  - `Answer(doc_id: str, char_start: int, char_end: int, content_sha256: str)` — frozen dataclass
  - `GoldenQuestion(qid, question, bucket, answers: tuple[Answer, ...], provenance, notes)` — frozen dataclass
  - `GoldenSetError(Exception)`
  - `load_golden(path: Path) -> list[GoldenQuestion]`
  - `write_golden(path: Path, questions: Iterable[GoldenQuestion]) -> None`
  - `validate_spans(questions, documents: Mapping[str, tuple[str, str]]) -> None` — `documents` maps `doc_id -> (text, sha256)`; raises `GoldenSetError`
  - `provenance_split(questions) -> dict[str, int]`
  - `GoldenQuestion.answer_spans() -> tuple[Span, ...]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_golden.py
import dataclasses
import json
from pathlib import Path

import pytest

from clause.evaluation.golden import (
    Answer,
    GoldenQuestion,
    GoldenSetError,
    load_golden,
    provenance_split,
    validate_spans,
    write_golden,
)

Q = GoldenQuestion(
    qid="q001",
    question="Which customers require enhanced due diligence?",
    bucket="definitional",
    answers=(Answer(doc_id="rbi-1", char_start=10, char_end=40, content_sha256="a" * 64),),
    provenance="drafted",
    notes="",
)

DOCS = {"rbi-1": ("x" * 100, "a" * 64)}


def test_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    write_golden(p, [Q])
    assert load_golden(p) == [Q]


def test_duplicate_qids_are_rejected(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    write_golden(p, [Q, Q])
    with pytest.raises(GoldenSetError, match="duplicate qid"):
        load_golden(p)


def test_an_unknown_bucket_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({**_raw(Q), "bucket": "trivia"}) + "\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="bucket"):
        load_golden(p)


def test_a_question_with_no_answers_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({**_raw(Q), "answers": []}) + "\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="at least one answer"):
        load_golden(p)


def test_an_unknown_provenance_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({**_raw(Q), "provenance": "vibes"}) + "\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="provenance"):
        load_golden(p)


def test_validate_spans_accepts_a_matching_document() -> None:
    validate_spans([Q], DOCS)


def test_a_stale_content_hash_is_fatal() -> None:
    """The whole point: never score against offsets into text that changed."""
    with pytest.raises(GoldenSetError, match="q001"):
        validate_spans([Q], {"rbi-1": ("x" * 100, "b" * 64)})


def test_a_span_beyond_the_document_is_fatal() -> None:
    long_answer = Answer(doc_id="rbi-1", char_start=10, char_end=5000, content_sha256="a" * 64)
    q = dataclasses.replace(Q, answers=(long_answer,))
    with pytest.raises(GoldenSetError, match="outside"):
        validate_spans([q], DOCS)


def test_an_unknown_document_is_fatal() -> None:
    with pytest.raises(GoldenSetError, match="unknown document"):
        validate_spans([Q], {})


def test_provenance_split_counts_each_state() -> None:
    verified = dataclasses.replace(Q, qid="q002", provenance="human_verified")
    assert provenance_split([Q, verified]) == {"drafted": 1, "human_verified": 1}


def _raw(q: GoldenQuestion) -> dict:
    return {
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
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_golden.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.evaluation.golden'`

- [ ] **Step 3: Implement**

```python
# src/clause/evaluation/golden.py
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

        questions.append(
            GoldenQuestion(
                qid=qid,
                question=row["question"],
                bucket=row["bucket"],
                answers=tuple(
                    Answer(
                        doc_id=a["doc_id"],
                        char_start=int(a["char_start"]),
                        char_end=int(a["char_end"]),
                        content_sha256=a["content_sha256"],
                    )
                    for a in raw_answers
                ),
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_golden.py -v && uv run ruff check . && uv run mypy
```
Expected: 11 passed, clean.

- [ ] **Step 5: Commit**

```bash
git add src/clause/evaluation/golden.py tests/test_golden.py
git commit -m "feat(eval): golden-set schema with fatal span validation

A span whose content_sha256 no longer matches the stored document is a
hard failure, not a warning. Phase 1 showed why: when extraction changed,
all 61 manifest hashes went stale and the suite still passed because a
tolerant path absorbed every mismatch. Scoring against offsets that moved
would print a plausible table measuring nothing."
```

---

## Task 3: Draft the golden set

**Files:**
- Create: `data/golden/kyc-v1.jsonl`, `tests/test_golden_corpus.py`

**Interfaces:**
- Consumes: `load_golden`, `validate_spans`, `BUCKETS` from Task 2; the live `documents` table
- Produces: a committed golden set of **at least 60 questions, at least 15 per bucket**, every `provenance` set to `"drafted"`

This task is content work, not code. Read the corpus and write questions.

- [ ] **Step 1: Write the corpus-level tests first**

```python
# tests/test_golden_corpus.py
"""Guards on the committed golden set itself."""

import collections
import difflib
import os
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from clause.db.schema import DocumentRow
from clause.evaluation.golden import BUCKETS, load_golden, validate_spans

GOLDEN = Path("data/golden/kyc-v1.jsonl")
MIN_QUESTIONS = 60
MIN_PER_BUCKET = 15


def test_the_committed_golden_set_is_large_enough() -> None:
    questions = load_golden(GOLDEN)
    assert len(questions) >= MIN_QUESTIONS


def test_every_bucket_is_well_represented() -> None:
    counts = collections.Counter(q.bucket for q in load_golden(GOLDEN))
    assert set(counts) == set(BUCKETS)
    thin = {b: counts[b] for b in BUCKETS if counts[b] < MIN_PER_BUCKET}
    assert not thin, f"buckets below the floor: {thin}"


def test_no_question_quotes_its_source_verbatim() -> None:
    """A question echoing its source measures lexical overlap, not retrieval."""
    questions = load_golden(GOLDEN)
    engine = sa.create_engine(_db_url())
    with Session(engine) as session:
        texts = {d.doc_id: d.text for d in session.scalars(sa.select(DocumentRow)).all()}
    offenders = []
    for q in questions:
        for a in q.answers:
            source = texts[a.doc_id][a.char_start : a.char_end]
            longest = _longest_common_run(q.question.lower(), source.lower())
            if longest >= 40:
                offenders.append((q.qid, longest))
    assert not offenders, f"questions sharing a long run with their source: {offenders}"


@pytest.mark.db
def test_every_span_resolves_against_the_stored_corpus(db_session: Session) -> None:
    docs = {
        d.doc_id: (d.text, d.sha256)
        for d in db_session.scalars(sa.select(DocumentRow)).all()
    }
    if not docs:
        pytest.skip("no ingested corpus in the test database")
    validate_spans(load_golden(GOLDEN), docs)


def _longest_common_run(a: str, b: str) -> int:
    match = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)
    )
    return match.size


def _db_url() -> str:
    return os.environ.get(
        "CLAUSE_DATABASE_URL", "postgresql+psycopg://clause:clause@localhost:5434/clause"
    )
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest tests/test_golden_corpus.py -v
```
Expected: FAIL — `FileNotFoundError` for `data/golden/kyc-v1.jsonl`.

- [ ] **Step 3: Read the corpus and draft the questions**

Work from the application database (`CLAUSE_DATABASE_URL`, port 5434), which holds the ingested corpus. For each document you draw on:

```bash
PYTHONIOENCODING=utf-8 uv run python - <<'PY'
import sqlalchemy as sa
from sqlalchemy.orm import Session
from clause.db.schema import DocumentRow
engine = sa.create_engine("postgresql+psycopg://clause:clause@localhost:5434/clause")
with Session(engine) as s:
    for d in s.scalars(sa.select(DocumentRow).limit(5)).all():
        print(d.doc_id, d.circular_no, d.published_date, d.title[:90])
        print("   ", d.text[:300].replace("\n", " "))
PY
```

Rules for drafting, all of them load-bearing:

1. **Paraphrase. Never lift phrasing.** `test_no_question_quotes_its_source_verbatim` enforces a 40-character ceiling on the longest run shared between a question and its answer span. A question that echoes its source measures lexical overlap and would inflate every number in the report.
2. **Spans must be the passage that actually answers the question** — tight enough to be meaningful, not a whole document.
3. **Include every acceptable span.** Where several circulars say the same thing, list them all. That is the entire reason ground truth is a set.
4. **Bucket honestly:**
   - `definitional` — "what counts as X", "who is a Y"
   - `numeric_threshold` — a figure, a percentage, a time limit
   - `procedural` — a sequence of steps, an obligation with a deadline
   - `cross_reference` — the answer needs an amendment *and* what it amended; give spans in both documents
5. **`provenance` is `"drafted"` for every question.** You are not verifying your own labels.
6. Record the document's `sha256` (from `documents.sha256`) as each answer's `content_sha256`.

Write the file with `write_golden` so the schema cannot drift from the loader.

- [ ] **Step 4: Validate and check bucket balance**

```bash
PYTHONIOENCODING=utf-8 uv run python - <<'PY'
import collections
from pathlib import Path
from clause.evaluation.golden import load_golden
qs = load_golden(Path("data/golden/kyc-v1.jsonl"))
print("questions:", len(qs))
print("buckets:", dict(collections.Counter(q.bucket for q in qs)))
print("answers per question: min", min(len(q.answers) for q in qs),
      "max", max(len(q.answers) for q in qs))
print("documents covered:", len({a.doc_id for q in qs for a in q.answers}))
PY
uv run pytest tests/test_golden_corpus.py -v
```
Expected: all pass. If the verbatim test fails, rewrite those questions — do not raise the threshold.

- [ ] **Step 5: Commit**

```bash
git add data/golden/kyc-v1.jsonl tests/test_golden_corpus.py
git commit -m "feat(eval): golden set of 60+ questions across four buckets

Every question is drafted, not verified -- provenance says so, and the
report will publish the split rather than let a reader assume a human
checked them. Questions are paraphrased and a test caps the longest run
shared with their source at 40 characters: a question echoing its source
measures lexical overlap, not retrieval, and would inflate every number."
```

---

## Task 4: The human verification workflow

**Files:**
- Create: `scripts/verify_golden.py`
- Test: `tests/test_verify_golden.py`

**Interfaces:**
- Consumes: `load_golden`, `write_golden`, `GoldenQuestion`, `provenance_split`
- Produces: `select_sample(questions, per_bucket: int, seed: int) -> list[GoldenQuestion]`, `apply_decision(question, decision, span=None) -> GoldenQuestion`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_verify_golden.py
import collections

import pytest

from clause.evaluation.golden import Answer, GoldenQuestion
from scripts.verify_golden import apply_decision, select_sample


def _q(qid: str, bucket: str) -> GoldenQuestion:
    return GoldenQuestion(
        qid=qid,
        question="?",
        bucket=bucket,
        answers=(Answer("rbi-1", 0, 10, "a" * 64),),
        provenance="drafted",
        notes="",
    )


QUESTIONS = [_q(f"q{i:03d}", b) for i, b in enumerate(["definitional"] * 8 + ["procedural"] * 8)]


def test_sample_is_balanced_across_buckets() -> None:
    sample = select_sample(QUESTIONS, per_bucket=3, seed=1)
    counts = collections.Counter(q.bucket for q in sample)
    assert counts == {"definitional": 3, "procedural": 3}


def test_sample_is_deterministic_for_a_seed() -> None:
    assert [q.qid for q in select_sample(QUESTIONS, 3, seed=7)] == [
        q.qid for q in select_sample(QUESTIONS, 3, seed=7)
    ]


def test_sample_takes_everything_when_a_bucket_is_smaller_than_asked() -> None:
    sample = select_sample(QUESTIONS[:2], per_bucket=5, seed=1)
    assert len(sample) == 2


def test_confirming_marks_it_human_verified() -> None:
    assert apply_decision(QUESTIONS[0], "confirm").provenance == "human_verified"


def test_correcting_replaces_the_span_and_marks_it_corrected() -> None:
    fixed = apply_decision(QUESTIONS[0], "correct", span=(50, 90))
    assert fixed.provenance == "human_corrected"
    assert fixed.answers[0].char_start == 50
    assert fixed.answers[0].char_end == 90
    assert fixed.answers[0].content_sha256 == QUESTIONS[0].answers[0].content_sha256


def test_an_unknown_decision_is_rejected() -> None:
    with pytest.raises(ValueError, match="decision"):
        apply_decision(QUESTIONS[0], "maybe")
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_verify_golden.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Implement**

Create `scripts/__init__.py` (empty) so the module imports, then:

```python
# scripts/verify_golden.py
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
    if decision == "confirm":
        return dataclasses.replace(question, provenance="human_verified")
    if decision == "correct":
        if span is None:
            raise ValueError("correcting a question needs a replacement span")
        start, end = span
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_verify_golden.py -v && uv run ruff check . && uv run mypy
```
Expected: 6 passed, clean.

- [ ] **Step 5: Commit, then tell the controller the human step is ready**

```bash
git add scripts/ tests/test_verify_golden.py
git commit -m "feat(eval): human verification workflow for drafted labels

Bucket-balanced and seeded rather than uniformly random: a uniform sample
would under-represent the smallest bucket, and the buckets exist to expose
where retrieval is weak. Writes provenance back so the report can state
what fraction of ground truth a person actually checked."
```

Report in your summary that `uv run python -m scripts.verify_golden --per-bucket 4` is now available — the human verification pass can run in parallel with the remaining tasks.

---

## Task 5: Regulated entities and migration 0003

**Files:**
- Create: `src/clause/entities.py`, `migrations/versions/0003_regulated_entity.py`, `tests/test_entities.py`
- Modify: `src/clause/db/schema.py`, `src/clause/models.py`, `src/clause/ingest/extract.py`, `src/clause/db/repository.py`, `src/clause/chunking/base.py`

**Interfaces:**
- Consumes: `HEADER` from `clause.ingest.extract`
- Produces:
  - `ENTITY_VOCABULARY: tuple[str, ...]`
  - `parse_regulated_entities(text: str) -> tuple[str, ...]`
  - `Document.regulated_entity: tuple[str, ...]`, `Chunk.regulated_entity: tuple[str, ...]`
  - `DocumentRow.regulated_entity`, `ChunkRow.regulated_entity` — Postgres `ARRAY(Text)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_entities.py
from pathlib import Path

from clause.entities import ENTITY_VOCABULARY, parse_regulated_entities
from clause.ingest.extract import canonical_text

FIXTURE = Path(__file__).parent / "fixtures" / "rbi_13704.html"
REAL = canonical_text(FIXTURE.read_text(encoding="utf-8", errors="replace"))

ADDRESSEE = (
    "RBI/2026-27/253 DOR.AML.REC.1/14.01.005/2026-27 September 18, 2026 "
    "The Chairpersons/ CEOs of the Commercial Banks, Small Finance Banks, "
    "Payment Banks, Urban Co-operative Banks, Regional Rural Banks, "
    "Non-Banking Financial Companies Madam/Dear Sir, Subject line here"
)


def test_parses_the_entities_from_an_addressee_block() -> None:
    found = parse_regulated_entities(ADDRESSEE)
    assert "Commercial Banks" in found
    assert "Small Finance Banks" in found
    assert "Non-Banking Financial Companies" in found


def test_result_is_ordered_and_deduplicated() -> None:
    doubled = ADDRESSEE.replace("Commercial Banks", "Commercial Banks, Commercial Banks")
    found = parse_regulated_entities(doubled)
    assert len(found) == len(set(found))
    assert list(found) == sorted(found)


def test_every_result_comes_from_the_closed_vocabulary() -> None:
    assert set(parse_regulated_entities(ADDRESSEE)) <= set(ENTITY_VOCABULARY)


def test_a_document_without_an_addressee_block_yields_nothing() -> None:
    """Empty, never a guess -- an entity filter is only useful if absence is visible."""
    assert parse_regulated_entities("RBI/2026-27/1 DOR.X.1/1 January 1, 2026 no addressee") == ()


def test_text_without_a_header_yields_nothing() -> None:
    assert parse_regulated_entities("nothing resembling a circular") == ()


def test_the_real_fixture_yields_at_least_one_entity() -> None:
    assert parse_regulated_entities(REAL)


def test_only_the_addressee_block_is_scanned() -> None:
    """A body mention of an entity class must not become a filter value."""
    body_mention = (
        "RBI/2026-27/1 DOR.X.1/1 January 1, 2026 "
        "The Chairpersons of the Commercial Banks Madam/Dear Sir, "
        "This circular also concerns Payment Banks in passing."
    )
    found = parse_regulated_entities(body_mention)
    assert "Commercial Banks" in found
    assert "Payment Banks" not in found
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_entities.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.entities'`

- [ ] **Step 3: Implement the parser**

```python
# src/clause/entities.py
"""Parse the regulated entities a circular is addressed to.

RBI names its audience explicitly between the header line and the salutation:

    The Chairpersons/ CEOs of the Commercial Banks, Small Finance Banks, ...
    Madam/Dear Sir,

That block is structured data the corpus already contains, and it is what makes
`PROMPT.md`'s "regulated entity" metadata filter real rather than aspirational.
"""

import re

from clause.ingest.extract import HEADER

#: Closed vocabulary. A value outside it is not emitted, because a filter whose
#: values are open-ended cannot be offered in a UI or asserted in a test.
ENTITY_VOCABULARY: tuple[str, ...] = (
    "All India Financial Institutions",
    "Asset Reconstruction Companies",
    "Commercial Banks",
    "Cooperative Banks",
    "Credit Information Companies",
    "Housing Finance Companies",
    "Local Area Banks",
    "Non-Banking Financial Companies",
    "Payment Banks",
    "Payments Banks",
    "Primary (Urban) Co-operative Banks",
    "Regional Rural Banks",
    "Rural Co-operative Banks",
    "Small Finance Banks",
    "State Co-operative Banks",
    "Urban Co-operative Banks",
)

SALUTATION = re.compile(
    r"(?:Madam\s*/?\s*Dear Sir|Dear Sir\s*/?\s*Madam|Dear Madam|Madam|Dear Sir)\s*,",
    re.IGNORECASE,
)

#: The addressee block sits between the header and the salutation. Scanning only
#: that window is what keeps a passing body mention of "Payment Banks" out of the
#: filter values.
ADDRESSEE_WINDOW_CHARS = 800


def parse_regulated_entities(text: str) -> tuple[str, ...]:
    """Entities named in the addressee block, sorted and deduplicated.

    Returns an empty tuple when there is no header, no salutation, or no known
    entity in between. Never guesses: an entity filter is only useful if a
    document's absence from it is trustworthy.
    """
    header = HEADER.search(text)
    if header is None:
        return ()
    window = text[header.end() : header.end() + ADDRESSEE_WINDOW_CHARS]
    salutation = SALUTATION.search(window)
    if salutation is None:
        return ()
    block = window[: salutation.start()]
    found = {name for name in ENTITY_VOCABULARY if name.lower() in block.lower()}
    return tuple(sorted(found))
```

- [ ] **Step 4: Run the parser tests**

```bash
uv run pytest tests/test_entities.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Add the column to the domain types and the schema**

In `src/clause/models.py`, add to **both** `Document` and `Chunk`:

```python
    regulated_entity: tuple[str, ...] = ()
```

Place it last in each dataclass so existing positional construction is unaffected.

In `src/clause/db/schema.py`, add this import at module top:

```python
from sqlalchemy.dialects.postgresql import ARRAY
```

and this field to **both** `DocumentRow` and `ChunkRow`:

```python
    regulated_entity: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
```

In `src/clause/chunking/base.py`, `make_chunk` must carry it through:

```python
        regulated_entity=document.regulated_entity,
```

In `src/clause/ingest/extract.py`, add this import at module top:

```python
from clause.entities import parse_regulated_entities
```

and this keyword to the `Document(...)` construction inside `extract_document`:

```python
        regulated_entity=parse_regulated_entities(text),
```

In `src/clause/db/repository.py`, add `"regulated_entity": document.regulated_entity` to `upsert_document`'s `values` dict, and `regulated_entity=c.regulated_entity` to the `ChunkRow(...)` construction in `replace_chunks`.

- [ ] **Step 6: Write migration 0003**

```python
# migrations/versions/0003_regulated_entity.py
"""regulated entity columns

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("documents", "chunks"):
        op.add_column(
            table,
            sa.Column(
                "regulated_entity",
                ARRAY(sa.Text()),
                nullable=False,
                server_default="{}",
            ),
        )


def downgrade() -> None:
    for table in ("chunks", "documents"):
        op.drop_column(table, "regulated_entity")
```

`server_default="{}"` matters: the column is added to tables that already hold rows, and a `NOT NULL` column without a default cannot be added to a populated table.

- [ ] **Step 7: Migrate, re-ingest, and confirm coverage**

```bash
export CLAUSE_DATABASE_URL='postgresql+psycopg://clause:clause@localhost:5434/clause'
uv run alembic upgrade head
uv run python -m clause.cli ingest --manifest data/corpus/kyc.manifest.jsonl
PYTHONIOENCODING=utf-8 uv run python - <<'PY'
import collections, sqlalchemy as sa
from sqlalchemy.orm import Session
from clause.db.schema import DocumentRow
e = sa.create_engine("postgresql+psycopg://clause:clause@localhost:5434/clause")
with Session(e) as s:
    docs = s.scalars(sa.select(DocumentRow)).all()
    empty = [d.doc_id for d in docs if not d.regulated_entity]
    print(f"documents with no parsed entity: {len(empty)} of {len(docs)}")
    counts = collections.Counter(e for d in docs for e in d.regulated_entity)
    for name, n in counts.most_common():
        print(f"  {n:3}  {name}")
PY
```

Report the coverage figure. If a large share parse empty, say so — do **not** widen the vocabulary to force coverage. An entity filter whose absence is untrustworthy is worse than no filter.

- [ ] **Step 8: Run the full suite and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy
git add src/clause/entities.py src/clause/models.py src/clause/db/ src/clause/chunking/base.py \
        src/clause/ingest/extract.py migrations/versions/0003_regulated_entity.py tests/test_entities.py
git commit -m "feat: parse regulated entities into a filterable column

Makes PROMPT.md's 'regulated entity' filter real from data the corpus
already contained -- the addressee block RBI puts between the header and
the salutation. Only that window is scanned, so a passing body mention
cannot become a filter value, and a document that does not parse gets an
empty list rather than a guess: an entity filter is only useful if a
document's absence from it can be trusted."
```

---

## Task 6: Embedding encoder

**Files:**
- Create: `src/clause/embed.py`, `tests/test_embed.py`
- Modify: `pyproject.toml`, `docs/decisions.md`, `.env.example`, `src/clause/config.py`

**Interfaces:**
- Consumes: nothing (it *adds* fields to `Settings`)
- Produces:
  - `DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"`
  - `ModelNotCachedError(Exception)`
  - `Encoder(model_name: str = DEFAULT_MODEL)` with `.dimension: int` and `.encode(texts: Sequence[str]) -> list[list[float]]`
  - `Settings.embedding_model: str`, `Settings.qdrant_url: str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embed.py
import math

import pytest

from clause.embed import DEFAULT_MODEL, Encoder, ModelNotCachedError

pytestmark = pytest.mark.model


def test_encodes_to_the_declared_dimension() -> None:
    enc = Encoder()
    vectors = enc.encode(["capital adequacy requirements"])
    assert len(vectors) == 1
    assert len(vectors[0]) == enc.dimension


def test_encoding_is_deterministic() -> None:
    """A measurement harness whose inputs move is not a measurement harness."""
    enc = Encoder()
    assert enc.encode(["know your customer"]) == enc.encode(["know your customer"])


def test_batch_order_is_preserved() -> None:
    enc = Encoder()
    a, b = enc.encode(["alpha", "beta"])
    assert a == enc.encode(["alpha"])[0]
    assert b == enc.encode(["beta"])[0]


def test_empty_batch_returns_empty() -> None:
    assert Encoder().encode([]) == []


def test_similar_text_scores_closer_than_unrelated_text() -> None:
    enc = Encoder()
    kyc, kyc2, weather = enc.encode(
        ["customer due diligence obligations", "know your customer requirements", "tomorrow's rainfall"]
    )
    assert _cos(kyc, kyc2) > _cos(kyc, weather)


def test_an_uncached_model_fails_loudly_rather_than_downloading() -> None:
    with pytest.raises(ModelNotCachedError, match="not in the local cache"):
        Encoder(model_name="sentence-transformers/definitely-not-a-real-model-xyz")


def test_default_model_is_the_documented_baseline() -> None:
    assert DEFAULT_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)
```

- [ ] **Step 2: Register the marker and run the tests**

Add to `pyproject.toml`'s `[tool.pytest.ini_options]` markers list:

```toml
markers = [
    "db: requires a live Postgres instance",
    "model: requires the embedding model in the local cache",
    "qdrant: requires a live Qdrant instance",
]
```

```bash
uv run pytest tests/test_embed.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.embed'`

- [ ] **Step 3: Add dependencies**

```bash
uv add sentence-transformers onnxruntime
```

Append to `docs/decisions.md`:

```markdown
- sentence-transformers — text embedding. Replaces calling a hosted embedding API; the model runs locally, so an eval run costs nothing per query and is reproducible offline.
- onnxruntime — execution provider for the embedding model, per CLAUDE.md's stated stack. Chosen over raw PyTorch inference for a smaller install and faster CPU encoding.
```

- [ ] **Step 4: Implement**

```python
# src/clause/embed.py
"""Local sentence-transformers encoding through the ONNX runtime.

The baseline model is chosen for being unsurprising rather than best: it is
symmetric, so there is no asymmetric query-prefix convention to get silently
wrong. `BAAI/bge-small-en-v1.5` is the obvious upgrade at the same
dimensionality, and the entire point of building this harness first is that the
swap can then be measured instead of asserted.
"""

from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import scan_cache_dir
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ModelNotCachedError(Exception):
    """The model is not available locally, and this process will not download it."""


def _is_cached(model_name: str) -> bool:
    """True when the model is already in the HuggingFace cache.

    Checked before construction so an eval run cannot silently pull weights over
    the network mid-measurement: a first run that downloads has different timing
    and a different reproducibility story from every run after it.
    """
    try:
        cache = scan_cache_dir()
    except Exception:  # no cache directory yet
        return False
    wanted = model_name.lower()
    return any(repo.repo_id.lower() == wanted for repo in cache.repos)


class Encoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, *, cache_folder: Path | None = None):
        if not _is_cached(model_name):
            raise ModelNotCachedError(
                f"{model_name!r} is not in the local cache, and this process will not "
                "download it mid-measurement. Warm it first with:\n"
                f"  uv run python -c \"from sentence_transformers import SentenceTransformer; "
                f"SentenceTransformer('{model_name}', backend='onnx')\""
            )
        self._model = SentenceTransformer(
            model_name,
            backend="onnx",
            cache_folder=str(cache_folder) if cache_folder else None,
        )
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(x) for x in row] for row in vectors]
```

Add to `src/clause/config.py`'s `Settings`:

```python
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    qdrant_url: str = "http://localhost:6335"
```

Add to `.env.example`:

```bash
CLAUSE_QDRANT_URL=http://localhost:6335
CLAUSE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

- [ ] **Step 5: Warm the model and run the tests**

```bash
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', backend='onnx')"
uv run pytest tests/test_embed.py -v && uv run ruff check . && uv run mypy
```
Expected: 7 passed. Report the model's dimension — later tasks depend on it.

- [ ] **Step 6: Commit**

```bash
git add src/clause/embed.py src/clause/config.py tests/test_embed.py \
        pyproject.toml uv.lock docs/decisions.md .env.example
git commit -m "feat(embed): local ONNX encoder that refuses to download mid-run

An uncached model raises with the command that warms it rather than
pulling weights during an evaluation. A first run that downloads has
different timing and a different reproducibility story from every run
after it, and a measurement harness whose inputs move is not one."
```

---

## Task 7: Qdrant indexing

**Files:**
- Create: `src/clause/index.py`, `tests/test_index.py`
- Modify: `docker-compose.yml`, `pyproject.toml`, `docs/decisions.md`, `Makefile`, `src/clause/cli.py`

**Interfaces:**
- Consumes: `Encoder`, `ChunkRow`, `DocumentRow`
- Produces:
  - `collection_name(strategy: str) -> str` — returns `f"clause_{strategy}"`
  - `ensure_collection(client: QdrantClient, name: str, dimension: int) -> None`
  - `index_strategy(client, encoder, session, strategy: str, *, batch_size: int = 64) -> int` — returns points written

- [ ] **Step 1: Add Qdrant to docker-compose**

```yaml
  qdrant:
    image: qdrant/qdrant:v1.19.1
    ports:
      - "6335:6333"
    volumes:
      - qdrant-data:/qdrant/storage
    healthcheck:
      test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1"]
      interval: 5s
      timeout: 3s
      retries: 10
```

Add a top-level `volumes:` block with `qdrant-data:` if one does not exist. Host port 6335 because 6333 is occupied by an unrelated project.

```bash
docker compose up -d
docker compose ps
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_index.py
import os

import pytest
from qdrant_client import QdrantClient

from clause.index import collection_name, ensure_collection

pytestmark = pytest.mark.qdrant

URL = os.environ.get("CLAUSE_QDRANT_URL", "http://localhost:6335")


@pytest.fixture
def client() -> QdrantClient:
    c = QdrantClient(url=URL, timeout=5)
    try:
        c.get_collections()
    except Exception as exc:
        pytest.skip(f"no Qdrant at {URL}: {exc}")
    return c


def test_collection_name_is_namespaced_per_strategy() -> None:
    assert collection_name("structural") == "clause_structural"
    assert collection_name("fixed_window") == "clause_fixed_window"


def test_ensure_collection_is_idempotent(client: QdrantClient) -> None:
    name = "clause_test_idempotent"
    ensure_collection(client, name, 8)
    ensure_collection(client, name, 8)
    assert client.get_collection(name).config.params.vectors.size == 8
    client.delete_collection(name)


def test_changing_dimension_recreates_the_collection(client: QdrantClient) -> None:
    """A dimension change means a different model; upserting into the old index
    would mix incomparable vectors."""
    name = "clause_test_dimension"
    ensure_collection(client, name, 8)
    ensure_collection(client, name, 16)
    assert client.get_collection(name).config.params.vectors.size == 16
    client.delete_collection(name)
```

- [ ] **Step 3: Run and watch them fail**

```bash
uv run pytest tests/test_index.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.index'`

- [ ] **Step 4: Add the dependency**

```bash
uv add qdrant-client
```

Append to `docs/decisions.md`:

```markdown
- qdrant-client — vector store client. Qdrant is named in CLAUDE.md's stack; the client is its official one and supports the server-side payload filtering the metadata filters need.
```

- [ ] **Step 5: Implement**

```python
# src/clause/index.py
"""Qdrant collection lifecycle and chunk indexing.

One collection per chunking strategy, not one shared collection with a strategy
filter. Approximate-nearest-neighbour recall depends on what else is in the
index, so mixing both strategies would make each strategy's number a function of
the other -- destroying the comparison the two strategies exist to enable.
"""

from collections.abc import Iterator, Sequence

import sqlalchemy as sa
from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session

from clause.db.schema import ChunkRow, DocumentRow
from clause.embed import Encoder

COLLECTION_PREFIX = "clause_"
DEFAULT_BATCH_SIZE = 64


def collection_name(strategy: str) -> str:
    return f"{COLLECTION_PREFIX}{strategy}"


def ensure_collection(client: QdrantClient, name: str, dimension: int) -> None:
    """Create the collection, or recreate it when the dimension no longer matches.

    A dimension change means the embedding model changed. Upserting new vectors
    into an index built by a different model would silently mix incomparable
    vectors and produce retrieval numbers that mean nothing.
    """
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        current = client.get_collection(name).config.params.vectors
        if current is not None and current.size == dimension:
            return
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )


def _batches(rows: Sequence[ChunkRow], size: int) -> Iterator[Sequence[ChunkRow]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def index_strategy(
    client: QdrantClient,
    encoder: Encoder,
    session: Session,
    strategy: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Encode and upsert every chunk of one strategy. Returns points written."""
    name = collection_name(strategy)
    ensure_collection(client, name, encoder.dimension)

    rows = list(
        session.scalars(
            sa.select(ChunkRow).where(ChunkRow.strategy == strategy).order_by(ChunkRow.chunk_id)
        ).all()
    )
    published = {
        d.doc_id: d.published_date
        for d in session.scalars(sa.select(DocumentRow)).all()
    }

    written = 0
    for batch in _batches(rows, batch_size):
        vectors = encoder.encode([row.text for row in batch])
        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=row.chunk_id,
                    vector=vector,
                    payload={
                        "doc_id": row.doc_id,
                        "strategy": row.strategy,
                        "ordinal": row.ordinal,
                        "char_start": row.char_start,
                        "char_end": row.char_end,
                        "source_url": row.source_url,
                        "regulated_entity": list(row.regulated_entity or []),
                        "published_date": published[row.doc_id].isoformat(),
                        "text": row.text,
                    },
                )
                for row, vector in zip(batch, vectors, strict=True)
            ],
        )
        written += len(batch)
    return written
```

- [ ] **Step 6: Wire an `index` subcommand and a Makefile target**

In `src/clause/cli.py`, add an `index` subparser that builds a `QdrantClient` from `settings.qdrant_url`, an `Encoder` from `settings.embedding_model`, opens a session, calls `index_strategy` for `"fixed_window"` and `"structural"`, and prints the counts to stderr.

In the `Makefile`:

```make
index:
	uv run python -m clause.cli index
```

- [ ] **Step 7: Run the index and the tests**

```bash
export CLAUSE_QDRANT_URL='http://localhost:6335'
uv run python -m clause.cli index
uv run pytest tests/test_index.py -v && uv run ruff check . && uv run mypy
```
Expected: 3 passed; both collections populated. Report the point counts.

- [ ] **Step 8: Commit**

```bash
git add src/clause/index.py tests/test_index.py docker-compose.yml Makefile \
        src/clause/cli.py pyproject.toml uv.lock docs/decisions.md
git commit -m "feat(index): one Qdrant collection per chunking strategy

Not one shared collection with a strategy filter: ANN recall depends on
what else is in the index, so mixing strategies would make each one's
number a function of the other and destroy the comparison they exist to
enable. A dimension change recreates rather than upserts, because it
means the model changed and the old vectors are incomparable."
```

---

## Task 8: Retrieval with filters

**Files:**
- Create: `src/clause/retrieve.py`, `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `collection_name`, `Encoder`, `RETRIEVAL_DEPTH`
- Produces:
  - `Hit(doc_id, strategy, ordinal, char_start, char_end, score, text)` — frozen dataclass
  - `search(client, encoder, strategy, query, *, limit=RETRIEVAL_DEPTH, published_after=None, published_before=None, entities=None) -> list[Hit]`
  - `Hit.as_retrieved() -> Retrieved`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieve.py
import os
from datetime import date

import pytest
from qdrant_client import QdrantClient

from clause.embed import Encoder
from clause.evaluation.metrics import RETRIEVAL_DEPTH
from clause.retrieve import search

pytestmark = [pytest.mark.qdrant, pytest.mark.model]

URL = os.environ.get("CLAUSE_QDRANT_URL", "http://localhost:6335")


@pytest.fixture
def client() -> QdrantClient:
    c = QdrantClient(url=URL, timeout=5)
    try:
        if not c.get_collections().collections:
            pytest.skip("no indexed collections; run `uv run python -m clause.cli index`")
    except Exception as exc:
        pytest.skip(f"no Qdrant at {URL}: {exc}")
    return c


@pytest.fixture
def encoder() -> Encoder:
    return Encoder()


def test_search_returns_at_most_the_limit(client: QdrantClient, encoder: Encoder) -> None:
    hits = search(client, encoder, "structural", "know your customer", limit=RETRIEVAL_DEPTH)
    assert 0 < len(hits) <= RETRIEVAL_DEPTH


def test_results_are_ordered_by_descending_score(client: QdrantClient, encoder: Encoder) -> None:
    hits = search(client, encoder, "structural", "customer due diligence")
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_every_hit_carries_its_span(client: QdrantClient, encoder: Encoder) -> None:
    for hit in search(client, encoder, "structural", "beneficial owner"):
        assert hit.char_start < hit.char_end
        assert hit.doc_id
        assert hit.text


def test_a_date_filter_excludes_older_documents(client: QdrantClient, encoder: Encoder) -> None:
    cutoff = date(2026, 1, 1)
    hits = search(client, encoder, "structural", "sanctions list", published_after=cutoff)
    assert hits, "expected at least one document on or after the cutoff"
    _assert_all_published_on_or_after(client, hits, cutoff)


def test_an_entity_filter_restricts_to_documents_naming_it(
    client: QdrantClient, encoder: Encoder
) -> None:
    hits = search(client, encoder, "structural", "due diligence", entities=["Commercial Banks"])
    for hit in hits:
        payload = client.retrieve("clause_structural", ids=[hit_point_id(client, hit)])[0].payload
        assert "Commercial Banks" in (payload or {}).get("regulated_entity", [])


def test_a_filter_matching_nothing_returns_no_results(
    client: QdrantClient, encoder: Encoder
) -> None:
    hits = search(
        client, encoder, "structural", "anything", published_after=date(2099, 1, 1)
    )
    assert hits == []


def test_searching_the_other_strategy_hits_a_different_collection(
    client: QdrantClient, encoder: Encoder
) -> None:
    structural = search(client, encoder, "structural", "know your customer")
    fixed = search(client, encoder, "fixed_window", "know your customer")
    assert {h.strategy for h in structural} == {"structural"}
    assert {h.strategy for h in fixed} == {"fixed_window"}


def _assert_all_published_on_or_after(client, hits, cutoff) -> None:
    for hit in hits:
        point = client.retrieve("clause_structural", ids=[hit_point_id(client, hit)])[0]
        assert date.fromisoformat((point.payload or {})["published_date"]) >= cutoff


def hit_point_id(client, hit):
    """Recover a point id from a hit for payload assertions."""
    found, _ = client.scroll(
        f"clause_{hit.strategy}",
        scroll_filter=None,
        limit=1000,
    )
    for point in found:
        payload = point.payload or {}
        if (
            payload.get("doc_id") == hit.doc_id
            and payload.get("ordinal") == hit.ordinal
            and payload.get("strategy") == hit.strategy
        ):
            return point.id
    raise AssertionError("hit not found in collection")
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_retrieve.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.retrieve'`

- [ ] **Step 3: Implement**

```python
# src/clause/retrieve.py
"""Dense retrieval with server-side metadata filters.

`doc_type` is deliberately not offered as a filter. The predecessor spec's
section 10 records the measurement: zero `circular` labels across the corpus and
near-identical documents split both ways. Exposing a filter on a field known to
be noise would invite Phase 3 to filter on it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from qdrant_client import QdrantClient, models

from clause.embed import Encoder
from clause.evaluation.metrics import RETRIEVAL_DEPTH, Retrieved
from clause.index import collection_name


@dataclass(frozen=True, slots=True)
class Hit:
    doc_id: str
    strategy: str
    ordinal: int
    char_start: int
    char_end: int
    score: float
    text: str

    def as_retrieved(self) -> Retrieved:
        return Retrieved(
            doc_id=self.doc_id,
            char_start=self.char_start,
            char_end=self.char_end,
            score=self.score,
        )


def _build_filter(
    published_after: date | None,
    published_before: date | None,
    entities: Sequence[str] | None,
) -> models.Filter | None:
    conditions: list[models.Condition] = []
    if published_after is not None or published_before is not None:
        conditions.append(
            models.FieldCondition(
                key="published_date",
                range=models.DatetimeRange(
                    gte=published_after.isoformat() if published_after else None,
                    lte=published_before.isoformat() if published_before else None,
                ),
            )
        )
    if entities:
        conditions.append(
            models.FieldCondition(
                key="regulated_entity", match=models.MatchAny(any=list(entities))
            )
        )
    return models.Filter(must=conditions) if conditions else None


def search(
    client: QdrantClient,
    encoder: Encoder,
    strategy: str,
    query: str,
    *,
    limit: int = RETRIEVAL_DEPTH,
    published_after: date | None = None,
    published_before: date | None = None,
    entities: Sequence[str] | None = None,
) -> list[Hit]:
    """Top `limit` chunks of one strategy, filtered server-side."""
    vector = encoder.encode([query])[0]
    response = client.query_points(
        collection_name=collection_name(strategy),
        query=vector,
        limit=limit,
        query_filter=_build_filter(published_after, published_before, entities),
        with_payload=True,
    )
    hits: list[Hit] = []
    for point in response.points:
        payload = point.payload or {}
        hits.append(
            Hit(
                doc_id=payload["doc_id"],
                strategy=payload["strategy"],
                ordinal=int(payload["ordinal"]),
                char_start=int(payload["char_start"]),
                char_end=int(payload["char_end"]),
                score=float(point.score),
                text=payload["text"],
            )
        )
    return hits
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_retrieve.py -v && uv run ruff check . && uv run mypy
```
Expected: 7 passed. If the entity filter test skips for lack of a matching entity, report which entities the corpus actually carries rather than weakening the test.

- [ ] **Step 5: Commit**

```bash
git add src/clause/retrieve.py tests/test_retrieve.py
git commit -m "feat(retrieve): dense search with date and entity filters

Filters run server-side in Qdrant so the top-k is drawn from the filtered
set rather than filtered after the fact -- post-filtering a top-10 would
silently return fewer than ten results and make recall@10 mean something
different for filtered queries than unfiltered ones.

doc_type is not offered: it is measured noise, and exposing it would
invite Phase 3 to filter on it."
```

---

## Task 9: The report and its fingerprint

**Files:**
- Create: `src/clause/evaluation/report.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: metrics, `provenance_split`
- Produces:
  - `Fingerprint(manifest_sha256, chunk_counts: dict[str, int], embedding_model, retrieval_depth, golden_path, golden_sha256, git_commit)` — frozen dataclass with `.to_dict()`
  - `StrategyResult(strategy, per_bucket: dict[str, BucketResult], overall: BucketResult)`
  - `BucketResult(n, recall_at_1, recall_at_5, recall_at_10, mrr_at_10)`
  - `build_report(results: Sequence[StrategyResult], fingerprint, provenance: dict[str, int]) -> dict`
  - `render_markdown(report: dict) -> str`
  - `write_report(report: dict, md_path: Path, json_path: Path) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report.py
import json
from pathlib import Path

from clause.evaluation.report import (
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

RESULT = StrategyResult(
    strategy="structural",
    per_bucket={"definitional": BucketResult(15, 0.6, 0.8, 0.9, 0.7)},
    overall=BucketResult(60, 0.5, 0.75, 0.85, 0.62),
)


def test_report_carries_the_fingerprint() -> None:
    report = build_report([RESULT], FP, {"drafted": 45, "human_verified": 15})
    assert report["fingerprint"]["golden_sha256"] == "g" * 64
    assert report["fingerprint"]["retrieval_depth"] == 10


def test_report_records_the_provenance_split() -> None:
    report = build_report([RESULT], FP, {"drafted": 45, "human_verified": 15})
    assert report["golden_provenance"] == {"drafted": 45, "human_verified": 15}


def test_markdown_states_the_precision_identity() -> None:
    """precision@1 and recall@1 are the same number; the report must say so."""
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}))
    assert "precision@1" in md
    assert "identical to recall@1" in md


def test_markdown_never_prints_precision_as_a_separate_column() -> None:
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}))
    header = next(line for line in md.splitlines() if "recall@1" in line and "|" in line)
    assert "precision@1" not in header


def test_markdown_names_every_bucket_and_strategy() -> None:
    md = render_markdown(build_report([RESULT], FP, {"drafted": 60}))
    assert "structural" in md
    assert "definitional" in md


def test_write_report_emits_both_artifacts(tmp_path: Path) -> None:
    report = build_report([RESULT], FP, {"drafted": 60})
    md, js = tmp_path / "eval.md", tmp_path / "eval.json"
    write_report(report, md, js)
    assert "structural" in md.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8"))["fingerprint"]["git_commit"] == "abc1234"
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_report.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.evaluation.report'`

- [ ] **Step 3: Implement**

```python
# src/clause/evaluation/report.py
"""Render the evaluation into one artifact for humans and one for CI."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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


def build_report(
    results: Sequence[StrategyResult],
    fingerprint: Fingerprint,
    provenance: dict[str, int],
) -> dict[str, Any]:
    return {
        "fingerprint": fingerprint.to_dict(),
        "golden_provenance": provenance,
        "strategies": {
            r.strategy: {
                "overall": asdict(r.overall),
                "per_bucket": {b: asdict(v) for b, v in sorted(r.per_bucket.items())},
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
    lines = [
        "# Retrieval evaluation",
        "",
        "Generated by `make eval`. Every figure here is produced by that command "
        "from the committed golden set; nothing in this file is estimated.",
        "",
        "## How to read these numbers",
        "",
        f"- Retrieval depth is fixed at {fp['retrieval_depth']}; MRR is **MRR@{fp['retrieval_depth']}**.",
        "- A hit means a retrieved chunk's character span overlapped an acceptable "
        "answer span in the same document.",
        f"- {PRECISION_NOTE}",
        "",
        "## Ground-truth provenance",
        "",
        "The golden set was drafted by a model and sampled by a human. This is the split:",
        "",
        "| provenance | questions |",
        "|---|---:|",
    ]
    for state, n in sorted(report["golden_provenance"].items()):
        lines.append(f"| {state} | {n} |")
    lines += ["", "## Results", ""]
    for strategy, data in sorted(report["strategies"].items()):
        lines += [
            f"### {strategy}",
            "",
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
```

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest tests/test_report.py -v && uv run ruff check . && uv run mypy
git add src/clause/evaluation/report.py tests/test_report.py
git commit -m "feat(eval): report renderer carrying a provenance fingerprint

The fingerprint is what makes gating a committed artifact defensible:
without it, 'CI fails on a deliberate regression' would be satisfiable by
not re-running the eval.

precision@1 is printed once with a note that it equals recall@1, and a
test asserts it never becomes its own column. Printing one number twice
under two names implies two pieces of evidence where there is one."
```

---

## Task 10: `make eval` end to end

**Files:**
- Modify: `src/clause/cli.py`, `Makefile`
- Create: `tests/test_eval_end_to_end.py`

**Interfaces:**
- Consumes: everything above
- Produces: `evaluate(session, client, encoder, questions, *, depth) -> list[StrategyResult]`; `python -m clause.cli eval` writing `reports/eval.md` and `reports/eval.json`

- [ ] **Step 1: Write the failing acceptance tests**

```python
# tests/test_eval_end_to_end.py
import json
from pathlib import Path

import pytest

from clause.evaluation.golden import BUCKETS

pytestmark = [pytest.mark.db, pytest.mark.qdrant, pytest.mark.model]

REPORT_MD = Path("reports/eval.md")
REPORT_JSON = Path("reports/eval.json")


def test_eval_writes_both_artifacts(evaluated: None) -> None:
    assert REPORT_MD.exists()
    assert REPORT_JSON.exists()


def test_both_strategies_are_measured(evaluated: None) -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert set(report["strategies"]) == {"structural", "fixed_window"}


def test_every_bucket_appears_for_every_strategy(evaluated: None) -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    for data in report["strategies"].values():
        assert set(data["per_bucket"]) == set(BUCKETS)


def test_recall_is_monotonic_in_k(evaluated: None) -> None:
    """recall@1 <= recall@5 <= recall@10 always. A violation means a bug."""
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    for data in report["strategies"].values():
        for r in [data["overall"], *data["per_bucket"].values()]:
            assert r["recall_at_1"] <= r["recall_at_5"] <= r["recall_at_10"]


def test_metrics_are_within_range(evaluated: None) -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    for data in report["strategies"].values():
        for r in [data["overall"], *data["per_bucket"].values()]:
            for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10"):
                assert 0.0 <= r[key] <= 1.0


def test_the_report_records_how_ground_truth_was_made(evaluated: None) -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert sum(report["golden_provenance"].values()) >= 60
```

Add `run_eval` to `tests/conftest.py`'s existing `from clause.cli import ingest` line, then add this fixture:

```python
@pytest.fixture
def evaluated(db_session: Session) -> None:
    """Run the real evaluation once, against the warm cache and a live Qdrant."""
    run_eval()
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_eval_end_to_end.py -v
```
Expected: FAIL — `ImportError: cannot import name 'run_eval'`

- [ ] **Step 3: Implement `evaluate` and `run_eval` in `src/clause/cli.py`**

```python
def evaluate(
    session: Session,
    client: QdrantClient,
    encoder: Encoder,
    questions: Sequence[GoldenQuestion],
    *,
    depth: int = RETRIEVAL_DEPTH,
) -> list[StrategyResult]:
    """Score both strategies against the golden set."""
    results: list[StrategyResult] = []
    for strategy in ("fixed_window", "structural"):
        ranks_overall: list[int | None] = []
        ranks_by_bucket: dict[str, list[int | None]] = {b: [] for b in BUCKETS}
        for question in questions:
            hits = search(client, encoder, strategy, question.question, limit=depth)
            rank = first_hit_rank(
                [h.as_retrieved() for h in hits], question.answer_spans()
            )
            ranks_overall.append(rank)
            ranks_by_bucket[question.bucket].append(rank)
        results.append(
            StrategyResult(
                strategy=strategy,
                overall=_bucket_result(ranks_overall),
                per_bucket={b: _bucket_result(r) for b, r in ranks_by_bucket.items()},
            )
        )
    return results


def _bucket_result(ranks: Sequence[int | None]) -> BucketResult:
    return BucketResult(
        n=len(ranks),
        recall_at_1=recall_at_k(ranks, 1),
        recall_at_5=recall_at_k(ranks, 5),
        recall_at_10=recall_at_k(ranks, 10),
        mrr_at_10=mrr(ranks),
    )
```

`run_eval()` must, in this order:

1. Load the golden set.
2. Read `documents` into `{doc_id: (text, sha256)}` and call `validate_spans`. **A `GoldenSetError` here aborts before any retrieval** — never score against stale offsets.
3. Build the `Encoder` (raising `ModelNotCachedError` if cold) and the `QdrantClient`.
4. Call `evaluate`.
5. Build the `Fingerprint`: manifest hash from `data/corpus/kyc.manifest.jsonl`, chunk counts from the database, model from settings, depth `RETRIEVAL_DEPTH`, golden path and its sha256, git commit from `git rev-parse HEAD`.
6. `write_report(...)` to `reports/eval.md` and `reports/eval.json`.
7. Print the overall table to stdout, since `PROMPT.md` requires `make eval` to *print* the table.

Add the `eval` subcommand, and to the `Makefile`:

```make
eval:
	uv run python -m clause.cli eval
```

- [ ] **Step 4: Run the real evaluation**

```bash
export CLAUSE_DATABASE_URL='postgresql+psycopg://clause:clause@localhost:5434/clause'
export CLAUSE_QDRANT_URL='http://localhost:6335'
uv run python -m clause.cli eval
cat reports/eval.md
```

Report the numbers. **Do not tune anything to improve them.** A first honest number is the point; tuning before a baseline exists is exactly what the harness was built to prevent.

- [ ] **Step 5: Run tests and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy
git add src/clause/cli.py Makefile tests/test_eval_end_to_end.py tests/conftest.py \
        reports/eval.md reports/eval.json
git commit -m "feat(eval): make eval scores both strategies and writes the report

Span validation runs before any retrieval, so a golden set that no longer
matches the stored corpus aborts the run instead of producing a plausible
table measuring nothing.

The acceptance tests assert properties rather than values -- recall
monotonic in k, metrics in range, every bucket present for every strategy
-- so they stay meaningful when the numbers move."
```

---

## Task 11: The CI regression gate

**Files:**
- Create: `src/clause/evaluation/gate.py`, `tests/test_gate.py`, `reports/baseline.json`
- Modify: `src/clause/cli.py`, `.github/workflows/ci.yml`, `README.md`

**Interfaces:**
- Consumes: `reports/eval.json` as a plain dict — deliberately **not** `Fingerprint`, so the gate can read a committed file without importing the renderer
- Produces:
  - `GateFailure(Exception)`
  - `check(report: dict, baseline: dict | None, tree_fingerprint: dict) -> list[str]` — returns failure messages, empty when clean
  - `python -m clause.cli gate` — exit 1 on any failure

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gate.py
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


def test_no_baseline_passes_with_a_note() -> None:
    """A first run has nothing to regress against."""
    assert check(REPORT, None, FP) == []


def test_a_strategy_missing_from_the_report_fails() -> None:
    partial = copy.deepcopy(REPORT)
    del partial["strategies"]["fixed_window"]
    assert any("fixed_window" in p for p in check(partial, BASELINE, FP))
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_gate.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.evaluation.gate'`

- [ ] **Step 3: Implement**

```python
# src/clause/evaluation/gate.py
"""The CI regression gate.

CI has no corpus -- `data/raw/` is gitignored and this project does not scrape a
public regulator on every push -- so it cannot recompute the numbers. It gates
the committed artifact instead, and the fingerprint check is what makes that
defensible: a deliberate regression fails two ways, by lowering the numbers or
by not re-running the eval at all.
"""

from typing import Any

#: Fields whose change invalidates a report. `git_commit` is excluded: it moves
#: on every commit, including ones that touch nothing the eval depends on.
FINGERPRINT_FIELDS = (
    "manifest_sha256",
    "chunk_counts",
    "embedding_model",
    "retrieval_depth",
    "golden_path",
    "golden_sha256",
)


class GateFailure(Exception):
    pass


def check(
    report: dict[str, Any], baseline: dict[str, Any] | None, tree_fingerprint: dict[str, Any]
) -> list[str]:
    """Return failure messages; empty means the gate passes."""
    problems: list[str] = []
    reported = report.get("fingerprint", {})

    for field in FINGERPRINT_FIELDS:
        if reported.get(field) != tree_fingerprint.get(field):
            problems.append(
                f"stale report: fingerprint field {field!r} is "
                f"{reported.get(field)!r} in reports/eval.json but "
                f"{tree_fingerprint.get(field)!r} in the working tree. "
                "Re-run `make eval` and commit the result."
            )

    if baseline is None:
        return problems

    for strategy, expected in baseline.items():
        actual = report.get("strategies", {}).get(strategy)
        if actual is None:
            problems.append(f"report is missing strategy {strategy!r}, which the baseline covers")
            continue
        got = actual["overall"]["recall_at_5"]
        want = expected["recall_at_5"]
        if got < want:
            problems.append(
                f"{strategy}: recall@5 regressed to {got:.3f} from a baseline of {want:.3f}"
            )
    return problems
```

Add a `gate` subcommand to `src/clause/cli.py` that loads `reports/eval.json`, loads `reports/baseline.json` if present, rebuilds the tree fingerprint the same way `run_eval` does (**without** needing Qdrant, the model or retrieval — it reads the manifest, the database chunk counts and the golden set), calls `check`, prints each problem to stderr and returns 1 if any.

- [ ] **Step 4: Commit the baseline deliberately**

```bash
uv run python - <<'PY'
import json, pathlib
report = json.loads(pathlib.Path("reports/eval.json").read_text(encoding="utf-8"))
baseline = {
    s: {"recall_at_5": d["overall"]["recall_at_5"]}
    for s, d in report["strategies"].items()
}
pathlib.Path("reports/baseline.json").write_text(
    json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
)
print(baseline)
PY
```

This is a deliberate human act. `make eval` must never write this file — a harness that promotes its own baseline cannot detect a regression.

- [ ] **Step 5: Prove the gate fails on a deliberate regression**

This is `PROMPT.md`'s literal definition of done for Phase 2. Demonstrate **both** failure routes:

```bash
# Route 1: a lowered number
cp reports/eval.json /tmp/eval.bak
uv run python -c "
import json,pathlib
p=pathlib.Path('reports/eval.json'); r=json.loads(p.read_text())
for s in r['strategies']: r['strategies'][s]['overall']['recall_at_5'] = 0.01
p.write_text(json.dumps(r, indent=2, sort_keys=True)+'\n')
"
uv run python -m clause.cli gate; echo "exit=$? (expect 1)"

# Route 2: a stale report
cp /tmp/eval.bak reports/eval.json
uv run python -c "
import json,pathlib
p=pathlib.Path('reports/eval.json'); r=json.loads(p.read_text())
r['fingerprint']['golden_sha256']='0'*64
p.write_text(json.dumps(r, indent=2, sort_keys=True)+'\n')
"
uv run python -m clause.cli gate; echo "exit=$? (expect 1)"

# Restore and confirm it passes
cp /tmp/eval.bak reports/eval.json
uv run python -m clause.cli gate; echo "exit=$? (expect 0)"
```

Capture all three outputs in your report.

- [ ] **Step 6: Wire CI**

Add to `.github/workflows/ci.yml`, after the pytest step:

```yaml
      - run: uv run python -m clause.cli gate
        env:
          CLAUSE_DATABASE_URL: postgresql+psycopg://clause:clause@localhost:5432/clause
```

The gate needs the database only for chunk counts, which the CI Postgres service provides — but that database is empty in CI. Make the gate's fingerprint rebuild tolerate an empty database by reading chunk counts from the committed report when the database has none, and **say so in a comment**: CI is checking the artifact's internal consistency and the baseline, not recomputing the corpus.

- [ ] **Step 7: Document it and commit**

Add a "Running the evaluation" section to `README.md` covering `docker compose up -d`, warming the model, `index`, `eval`, and how to update the baseline deliberately. **No numbers in the README.**

Also close the housekeeping item the Phase 2 spec's section 15 carries forward. The predecessor spec, `docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md`, has a testing table naming four tests that exist under different names and one, `test_ingest_is_idempotent`, that does not exist anywhere in the repo. Correct the four rows to the names the suite actually uses, and mark the missing row as not implemented rather than leaving the table claiming coverage that was never written. A spec's testing table that cannot be trusted as an index of real coverage is the same defect class the rest of this project has been retracting.

```bash
uv run pytest -q && uv run ruff check . && uv run mypy
git add src/clause/evaluation/gate.py tests/test_gate.py reports/baseline.json \
        src/clause/cli.py .github/workflows/ci.yml README.md
git commit -m "feat(eval): CI gate on the committed report and its fingerprint

CI cannot recompute the numbers -- it has no corpus and must not scrape a
public regulator on every push -- so it gates the artifact. The
fingerprint check is what makes that defensible: a deliberate regression
fails by lowering a number, and skipping the re-run fails by staleness.

The baseline is committed by hand. A harness that promotes its own
baseline cannot detect a regression."
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §4 Units | 1-11 |
| §4.1 Embedding model | 6 |
| §4.2 One collection per strategy | 7 |
| §5 Spans not chunk ids | 1 (`Span`, `overlaps`), 2 (`Answer`) |
| §6 Golden set, buckets, paraphrase rule, verification workflow | 3, 4 |
| §7 Metrics, MRR@10, precision identity | 1, 9 |
| §8 Retrieval and filters, `doc_type` retired | 8 |
| §8.1 `regulated_entity` | 5 |
| §9 Data flow | 10 |
| §10 Report, fingerprint, gate | 9, 11 |
| §11 Failure handling | 2 (stale spans), 6 (cold model), 7 (dimension), 11 (no baseline) |
| §12 Testing | every task |
| §13 Dependencies | 6, 7 with `docs/decisions.md` lines |
| §15 Housekeeping: predecessor spec §11's phantom test names | 11 (Step 7) |

**Gap found and closed during self-review:** spec §15 requires correcting the predecessor spec's testing table, which names four tests that exist under different names and one, `test_ingest_is_idempotent`, that does not exist. No task covered it. It is now written into **Task 11 Step 7** alongside the README work.

**Placeholder scan:** no TBD/TODO. Every code step carries runnable code. Task 3 is content work and says so explicitly, with the rules and the validation commands that make it checkable.

**Type consistency:** `Span` and `Retrieved` are defined once in Task 1 and used unchanged in 2, 8 and 10. `RETRIEVAL_DEPTH` is defined in Task 1 and imported by 8 and 10 rather than redeclared. `GoldenQuestion.answer_spans()` returns `tuple[Span, ...]`, which is what `first_hit_rank` consumes. `Hit.as_retrieved()` bridges Task 8 to Task 1's metric types. `Fingerprint` is defined in Task 9 and consumed as a plain dict by Task 11's `check`, which is deliberate — the gate must read a JSON file without importing the renderer.

---

## Known risks

1. **The corpus may be too small for stable metrics.** 61 documents and ~650 chunks means one question moving changes recall by ~1.7 points. The report states `n` per bucket so a reader can judge; if per-bucket `n` is 15, a single question is nearly 7 points. This is a real limit of the corpus, not of the harness, and the report must say so rather than imply precision it does not have.
2. **Entity parsing coverage is unknown until Task 5 Step 7 runs.** If most documents parse empty, the entity filter is real but thinly populated. The instruction is to report the coverage, not to widen the vocabulary until it looks better.
3. **The golden set is drafted by the same model family that will be evaluated.** Mitigated by the paraphrase rule and its 40-character test, and by human verification — but the report states the provenance split precisely so no reader mistakes drafted labels for verified ones.
4. **Task 3 is the largest single unit of work** and is content, not code. If 60 questions across four buckets proves impractical from 61 documents, the correct response is to report the shortfall, not to pad buckets with near-duplicate questions.
