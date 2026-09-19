import dataclasses
from datetime import UTC, date, datetime
from itertools import pairwise

import pytest

from clause.chunking.base import assert_slices, make_chunk
from clause.chunking.fixed import FixedWindowChunker
from clause.models import Document

ADVERSARIAL = [
    "short",
    "a " * 5000,
    "para with nbsp. " * 300,
    "कुछ हिन्दी पाठ. " * 200,
    "1. First. 2. Second. 3.1 Nested. (a) Lettered. " * 100,
]


def _doc(text: str) -> Document:
    return Document(
        doc_id="d1", rbi_id=1, url="https://example.test/d1",
        circular_no="RBI/2026-27/1", dept_ref="DOR.X.1", title="t",
        doc_type="circular", published_date=date(2026, 9, 18), effective_date=None,
        sha256="a" * 64, fetched_at=datetime(2026, 9, 19, tzinfo=UTC), text=text,
    )


@pytest.mark.parametrize("text", ADVERSARIAL)
def test_fixed_window_chunks_are_pure_slices(text: str) -> None:
    doc = _doc(text)
    chunks = FixedWindowChunker(window_chars=200, overlap_chars=50).chunk(doc)
    assert_slices(doc, chunks)


@pytest.mark.parametrize("text", ADVERSARIAL)
def test_fixed_window_covers_every_character(text: str) -> None:
    doc = _doc(text)
    chunks = FixedWindowChunker(window_chars=200, overlap_chars=50).chunk(doc)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(doc.text)
    for prev, nxt in pairwise(chunks):
        assert nxt.char_start <= prev.char_end, "gap between chunks would drop characters"


def test_chunks_carry_their_own_provenance() -> None:
    doc = _doc("x" * 1000)
    chunk = FixedWindowChunker(window_chars=200, overlap_chars=50).chunk(doc)[0]
    assert chunk.source_url == doc.url
    assert chunk.doc_type == doc.doc_type
    assert chunk.effective_date == doc.effective_date


def test_ordinals_are_dense_and_ascending() -> None:
    doc = _doc("y" * 1000)
    chunks = FixedWindowChunker(window_chars=200, overlap_chars=50).chunk(doc)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_assert_slices_catches_a_transformed_chunk() -> None:
    doc = _doc("hello world")
    good = make_chunk(doc, "fixed_window", 0, 0, 5)
    # dataclasses.replace, not __dict__: Chunk uses slots=True and has no __dict__.
    tampered = dataclasses.replace(good, text="HELLO")
    with pytest.raises(AssertionError):
        assert_slices(doc, [tampered])


def test_empty_document_yields_no_chunks() -> None:
    doc = _doc("")
    assert FixedWindowChunker(window_chars=200, overlap_chars=50).chunk(doc) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_chars": 0, "overlap_chars": 0},
        {"window_chars": -5, "overlap_chars": -1},
        {"window_chars": 10, "overlap_chars": -1},
        {"window_chars": 100, "overlap_chars": 100},
    ],
)
def test_rejects_non_positive_or_inverted_bounds(kwargs: dict[str, int]) -> None:
    # Constructor-only, never .chunk(): a non-positive window_chars would
    # surface only as an opaque ValueError from make_chunk once chunking ran,
    # rather than a clear rejection at construction time.
    with pytest.raises(ValueError):
        FixedWindowChunker(**kwargs)
