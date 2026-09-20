"""Guards on the committed golden set itself."""

import collections
import difflib
import os
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from clause.db.schema import DocumentRow
from clause.evaluation.golden import BUCKETS, PROVENANCE, load_golden, parse_tags, validate_spans

GOLDEN = Path("data/golden/kyc-v1.jsonl")
MIN_QUESTIONS = 60
MIN_PER_BUCKET = 15
MAX_SHARED_RUN = 40

# The `numeric_threshold` bucket mixes three retrieval tasks, and the corpus holds
# only five true thresholds -- too few to split into a bucket of its own without
# dropping below MIN_PER_BUCKET. The sub-kind is therefore carried per question so
# the eval report can break the bucket out instead of averaging over three tasks.
NUMERIC_KINDS = frozenset({"regulatory_threshold", "entry_count", "identifier"})
NOTE_TAGS = frozenset({"standing_obligation_not_sequence"})
TAG_KEYS = frozenset({"kind", "span", "note"})

# Spans lying entirely inside a document's heading block. A chunker that isolates
# or drops heading blocks makes these trivially easy or impossible for reasons
# that have nothing to do with retrieval quality, so they must stay visible per
# question rather than folded into a bucket average.
HEADING_SPAN_QIDS = frozenset({"num-006", "num-007", "num-008", "num-009", "num-010"})
SALUTATIONS = ("The Chairpersons", "Madam", "Dear Sir")
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


def test_every_question_declares_a_recognised_provenance() -> None:
    """Every question is `drafted` today, but that must be free to change: the
    user's own next step is a human verification pass (Task 4), and asserting
    "all drafted" here would make completing it a build-breaking act. The
    actual split belongs in the report (`golden_provenance`,
    `NO_VERIFICATION_WARNING`), which reads the real state rather than
    asserting one; this only guards that the *value itself* is one of the
    declared states -- `load_golden` already refuses anything else at parse
    time, so this is the second, cheaper line of defence documenting that
    intent for this file specifically.
    """
    unrecognised = [
        (q.qid, q.provenance) for q in load_golden(GOLDEN) if q.provenance not in PROVENANCE
    ]
    assert not unrecognised, f"questions with an unrecognised provenance: {unrecognised}"


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
    docs = _corpus_documents()
    if not docs:
        pytest.skip(INGEST_HINT)
    validate_spans(load_golden(GOLDEN), docs)


def test_the_tag_vocabulary_is_closed_and_every_numeric_question_declares_its_kind() -> None:
    """A convention nothing enforces drifts, and these tags are what the report groups on."""
    questions = load_golden(GOLDEN)
    missing, wrong_kind, stray, unknown = [], [], [], []
    for q in questions:
        tags = parse_tags(q.notes)
        unknown.extend(f"{q.qid}:{k}={v}" for k, v in tags.items() if k not in TAG_KEYS)
        if "note" in tags and tags["note"] not in NOTE_TAGS:
            unknown.append(f"{q.qid}:note={tags['note']}")
        if "span" in tags and tags["span"] != "heading":
            unknown.append(f"{q.qid}:span={tags['span']}")
        if q.bucket == "numeric_threshold":
            if "kind" not in tags:
                missing.append(q.qid)
            elif tags["kind"] not in NUMERIC_KINDS:
                wrong_kind.append((q.qid, tags["kind"]))
        elif "kind" in tags:
            stray.append(q.qid)
    assert not missing, f"numeric_threshold questions with no kind= tag: {missing}"
    assert not wrong_kind, f"kind= values outside {sorted(NUMERIC_KINDS)}: {wrong_kind}"
    assert not stray, f"kind= tag on a question outside numeric_threshold: {stray}"
    assert not unknown, f"tags outside the declared vocabulary: {unknown}"


def test_the_heading_span_tag_matches_the_questions_that_carry_one() -> None:
    """Tagged and untagged must both be wrong-proof: the set is checked, not the count."""
    tagged = {q.qid for q in load_golden(GOLDEN) if parse_tags(q.notes).get("span") == "heading"}
    assert tagged == HEADING_SPAN_QIDS, f"span=heading drifted: {tagged ^ HEADING_SPAN_QIDS}"


def test_heading_tagged_spans_really_sit_inside_a_document_heading() -> None:
    """Cross-check the tag against the corpus, so neither side can rot unnoticed."""
    texts = _corpus_texts()
    if not texts:
        pytest.skip(INGEST_HINT)
    actual = set()
    for q in load_golden(GOLDEN):
        limits = [_heading_end(texts[a.doc_id]) for a in q.answers]
        if all(limit and a.char_end <= limit for a, limit in zip(q.answers, limits, strict=True)):
            actual.add(q.qid)
    assert actual == HEADING_SPAN_QIDS, (
        f"spans inside a heading block disagree with the span=heading tag: "
        f"{actual ^ HEADING_SPAN_QIDS}"
    )


def _heading_end(text: str) -> int:
    """Offset where the masthead stops and the addressee line begins, or 0 if unclear."""
    offsets = [i for i in (text.find(s) for s in SALUTATIONS) if i > 0]
    return min(offsets) if offsets else 0


def _corpus_documents() -> dict[str, tuple[str, str]]:
    """`{doc_id: (text, sha256)}` from the application database, or `{}` if it
    cannot be reached at all -- a hardcoded fallback host/port (`_db_url()`
    below) has nothing listening on it in CI, where only
    `CLAUSE_TEST_DATABASE_URL` is set. That must skip via `INGEST_HINT`, the
    same as a database that connects fine but holds no corpus, rather than
    hard-failing with a raw `OperationalError` in the one environment that
    matters most.
    """
    try:
        engine = sa.create_engine(_db_url())
        with Session(engine) as session:
            docs = {
                d.doc_id: (d.text, d.sha256)
                for d in session.scalars(sa.select(DocumentRow)).all()
            }
        engine.dispose()
    except sa.exc.SQLAlchemyError:
        return {}
    return docs


def _corpus_texts() -> dict[str, str]:
    return {doc_id: text for doc_id, (text, _sha256) in _corpus_documents().items()}


def _longest_common_run(a: str, b: str) -> int:
    match = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)
    )
    return match.size


def _db_url() -> str:
    return os.environ.get(
        "CLAUSE_DATABASE_URL", "postgresql+psycopg://clause:clause@localhost:5434/clause"
    )
