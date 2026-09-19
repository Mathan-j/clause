from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path

import pytest

from clause.chunking.base import assert_slices
from clause.chunking.structural import StructuralChunker
from clause.ingest.extract import canonical_text
from clause.models import Document

FIXTURES = Path(__file__).parent / "fixtures"
_RAW_HTML = (FIXTURES / "rbi_13704.html").read_text(encoding="utf-8", errors="replace")
REAL_TEXT = canonical_text(_RAW_HTML)


def _doc(text: str) -> Document:
    return Document(
        doc_id="d1", rbi_id=1, url="https://example.test/d1",
        circular_no="RBI/2026-27/1", dept_ref="DOR.X.1", title="t",
        doc_type="circular", published_date=date(2026, 9, 18), effective_date=None,
        sha256="a" * 64, fetched_at=datetime(2026, 9, 19, tzinfo=UTC), text=text,
    )


CHUNKER = StructuralChunker(min_chunk_chars=50, max_chunk_chars=300)


def test_chunks_are_pure_slices_of_the_real_document() -> None:
    doc = _doc(REAL_TEXT)
    assert_slices(doc, CHUNKER.chunk(doc))


def test_splits_on_rbi_numbering() -> None:
    text = (
        "Preamble text that is long enough to stand alone as its own chunk here. "
        "1. First numbered paragraph with sufficient length to survive merging rules. "
        "2. Second numbered paragraph also long enough to be its own separate chunk. "
    )
    chunks = CHUNKER.chunk(_doc(text))
    assert len(chunks) >= 2
    assert any(c.text.lstrip().startswith("1.") for c in chunks)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "tiny",
        "x" * 5000,
        "1. " + ("word " * 2000),
        "(a) alpha. (b) beta. (c) gamma. " * 80,
        "1. " + "y" * 3000,  # numbered boundary then an unbreakable run
    ],
)
def test_invariant_holds_on_adversarial_input(text: str) -> None:
    doc = _doc(text)
    assert_slices(doc, CHUNKER.chunk(doc))


def test_no_chunk_exceeds_the_maximum_by_more_than_one_sentence() -> None:
    doc = _doc("Sentence here. " * 400)
    for c in CHUNKER.chunk(doc):
        assert len(c.text) <= 300 * 2


def test_covers_every_character() -> None:
    doc = _doc(REAL_TEXT)
    chunks = CHUNKER.chunk(doc)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(doc.text)
    for prev, nxt in pairwise(chunks):
        assert nxt.char_start == prev.char_end


def test_strategy_name_is_recorded() -> None:
    doc = _doc(REAL_TEXT)
    assert {c.strategy for c in CHUNKER.chunk(doc)} == {"structural"}
