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
MAX_SHARED_RUN = 40
INGEST_HINT = (
    "no ingested corpus in the application database; run "
    "`uv run python -m clause.cli ingest --manifest data/corpus/kyc.manifest.jsonl` first"
)


def test_the_committed_golden_set_is_large_enough() -> None:
    questions = load_golden(GOLDEN)
    assert len(questions) >= MIN_QUESTIONS


def test_every_bucket_is_well_represented() -> None:
    counts = collections.Counter(q.bucket for q in load_golden(GOLDEN))
    assert set(counts) == set(BUCKETS)
    thin = {b: counts[b] for b in BUCKETS if counts[b] < MIN_PER_BUCKET}
    assert not thin, f"buckets below the floor: {thin}"


def test_every_question_is_drafted_provenance() -> None:
    """Task 4 samples and re-labels; until then nothing claims a human checked it."""
    not_drafted = [q.qid for q in load_golden(GOLDEN) if q.provenance != "drafted"]
    assert not not_drafted, f"questions claiming verification they have not had: {not_drafted}"


def test_no_question_quotes_its_source_verbatim() -> None:
    """A question echoing its source measures lexical overlap, not retrieval."""
    questions = load_golden(GOLDEN)
    texts = _corpus_texts()
    if not texts:
        pytest.skip(INGEST_HINT)
    offenders = []
    for q in questions:
        for a in q.answers:
            source = texts[a.doc_id][a.char_start : a.char_end]
            longest = _longest_common_run(q.question.lower(), source.lower())
            if longest >= MAX_SHARED_RUN:
                offenders.append((q.qid, longest))
    assert not offenders, f"questions sharing a long run with their source: {offenders}"


def test_every_span_resolves_against_the_stored_corpus() -> None:
    """Read the application database, not the (deliberately empty) test one.

    `db_session` points at `clause_test`, which the suite builds and tears down on
    every run and which therefore never holds the ingested corpus. Taking that
    fixture here would skip on every run, silently, forever -- a green result
    concealing the one check that pins every labelled offset to real text.
    """
    engine = sa.create_engine(_db_url())
    with Session(engine) as session:
        docs = {
            d.doc_id: (d.text, d.sha256)
            for d in session.scalars(sa.select(DocumentRow)).all()
        }
    engine.dispose()
    if not docs:
        pytest.skip(INGEST_HINT)
    validate_spans(load_golden(GOLDEN), docs)


def _corpus_texts() -> dict[str, str]:
    engine = sa.create_engine(_db_url())
    with Session(engine) as session:
        texts = {d.doc_id: d.text for d in session.scalars(sa.select(DocumentRow)).all()}
    engine.dispose()
    return texts


def _longest_common_run(a: str, b: str) -> int:
    match = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)
    )
    return match.size


def _db_url() -> str:
    return os.environ.get(
        "CLAUSE_DATABASE_URL", "postgresql+psycopg://clause:clause@localhost:5434/clause"
    )
