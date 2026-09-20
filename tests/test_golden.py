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


def test_loader_rejects_negative_char_start(tmp_path: Path) -> None:
    """A span with negative char_start should raise GoldenSetError at load time."""
    p = tmp_path / "g.jsonl"
    bad_answer = {
        "doc_id": "rbi-1",
        "char_start": -1,
        "char_end": 40,
        "content_sha256": "a" * 64,
    }
    p.write_text(
        json.dumps({**_raw(Q), "answers": [bad_answer]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenSetError, match="char_start must not be negative"):
        load_golden(p)


def test_loader_rejects_reversed_span(tmp_path: Path) -> None:
    """A span where char_start >= char_end should raise GoldenSetError at load time."""
    p = tmp_path / "g.jsonl"
    bad_answer = {
        "doc_id": "rbi-1",
        "char_start": 40,
        "char_end": 10,
        "content_sha256": "a" * 64,
    }
    p.write_text(
        json.dumps({**_raw(Q), "answers": [bad_answer]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenSetError, match="span must be non-empty and forward"):
        load_golden(p)


def test_loader_rejects_zero_width_span(tmp_path: Path) -> None:
    """A span where char_start == char_end should raise GoldenSetError at load time."""
    p = tmp_path / "g.jsonl"
    bad_answer = {
        "doc_id": "rbi-1",
        "char_start": 40,
        "char_end": 40,
        "content_sha256": "a" * 64,
    }
    p.write_text(
        json.dumps({**_raw(Q), "answers": [bad_answer]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenSetError, match="span must be non-empty and forward"):
        load_golden(p)


def test_loader_rejects_malformed_line_not_object(tmp_path: Path) -> None:
    """A line that is valid JSON but not an object should raise GoldenSetError."""
    p = tmp_path / "g.jsonl"
    p.write_text("42\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="line 1"):
        load_golden(p)


def test_loader_rejects_missing_question_key(tmp_path: Path) -> None:
    """Missing 'question' key should raise GoldenSetError naming the line."""
    p = tmp_path / "g.jsonl"
    bad_q = {**_raw(Q)}
    del bad_q["question"]
    p.write_text(json.dumps(bad_q) + "\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match=r"line 1.*question"):
        load_golden(p)


def test_loader_rejects_missing_answer_key(tmp_path: Path) -> None:
    """Missing 'doc_id' in answer should raise GoldenSetError naming the line."""
    p = tmp_path / "g.jsonl"
    bad_answer = {
        "char_start": 10,
        "char_end": 40,
        "content_sha256": "a" * 64,
    }
    p.write_text(
        json.dumps({**_raw(Q), "answers": [bad_answer]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenSetError, match=r"line 1.*doc_id"):
        load_golden(p)


def test_loader_rejects_non_integer_char_start(tmp_path: Path) -> None:
    """Non-integer char_start should raise GoldenSetError naming the line."""
    p = tmp_path / "g.jsonl"
    bad_answer = {
        "doc_id": "rbi-1",
        "char_start": "abc",
        "char_end": 40,
        "content_sha256": "a" * 64,
    }
    p.write_text(
        json.dumps({**_raw(Q), "answers": [bad_answer]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenSetError, match=r"line 1.*char_start"):
        load_golden(p)


def test_loader_rejects_float_char_start(tmp_path: Path) -> None:
    """Float char_start should be rejected, not silently truncated."""
    p = tmp_path / "g.jsonl"
    bad_answer = {
        "doc_id": "rbi-1",
        "char_start": 10.9,
        "char_end": 40,
        "content_sha256": "a" * 64,
    }
    p.write_text(
        json.dumps({**_raw(Q), "answers": [bad_answer]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenSetError, match=r"line 1.*char_start.*integer"):
        load_golden(p)


def test_loader_rejects_answer_entry_not_object(tmp_path: Path) -> None:
    """An answer entry that is not an object should raise GoldenSetError."""
    p = tmp_path / "g.jsonl"
    p.write_text(
        json.dumps({**_raw(Q), "answers": ["not a dict"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenSetError, match=r"line 1.*answer"):
        load_golden(p)


def test_loader_rejects_non_string_qid(tmp_path: Path) -> None:
    """Non-string qid should be rejected."""
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({**_raw(Q), "qid": 12345}) + "\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match=r"line 1.*qid.*string"):
        load_golden(p)


def test_loader_rejects_empty_string_qid(tmp_path: Path) -> None:
    """Empty string qid should be rejected."""
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({**_raw(Q), "qid": ""}) + "\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match=r"line 1.*qid.*non-empty"):
        load_golden(p)


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
