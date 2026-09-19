import dataclasses
import unicodedata
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from clause.ingest.extract import ExtractionError, canonical_text, extract_document, parse_header
from clause.models import ManifestEntry

FIXTURES = Path(__file__).parent / "fixtures"
RAW = (FIXTURES / "rbi_13704.html").read_bytes()

ENTRY = ManifestEntry(
    doc_id="rbi-13704",
    rbi_id=13704,
    url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13704&Mode=0",
    circular_no="RBI/2026-27/262",
    dept_ref="DOR.AML.REC.223/14.01.005/2026-27",
    title="t",
    published_date=date(2026, 9, 18),
    content_sha256="a" * 64,
)


def test_canonical_text_is_nfc_normalised() -> None:
    out = canonical_text("<p>café</p>")
    assert out == unicodedata.normalize("NFC", out)
    assert out == "café"


def test_canonical_text_collapses_whitespace_and_nbsp() -> None:
    assert canonical_text("<p>a  b\t\tc</p>") == "a b c"


def test_canonical_text_is_idempotent() -> None:
    once = canonical_text(RAW.decode("utf-8", errors="replace"))
    assert canonical_text(f"<p>{once}</p>") == once


def test_parse_header_reads_the_real_document() -> None:
    circular_no, dept_ref, published = parse_header(canonical_text(RAW.decode("utf-8", "replace")))
    assert circular_no == "RBI/2026-27/262"
    assert dept_ref.startswith("DOR.AML.REC.223")
    assert published == date(2026, 9, 18)


def test_parse_header_raises_when_absent() -> None:
    with pytest.raises(ExtractionError):
        parse_header("no header here at all")


def test_parse_header_raises_extraction_error_on_invalid_date() -> None:
    with pytest.raises(ExtractionError):
        parse_header(
            "RBI/2026-27/262 DOR.AML.REC.1/14.01/2026-27 February 30, 2026"
        )


def test_parse_header_raises_extraction_error_on_unknown_month() -> None:
    with pytest.raises(ExtractionError):
        parse_header(
            "RBI/2026-27/262 DOR.AML.REC.1/14.01/2026-27 Septembre 18, 2026"
        )


def test_classify_uses_the_document_body_not_leading_chrome() -> None:
    """Leading chrome says 'Master Circular'; the document itself says 'Master Direction'.

    Classifying from the head of the page picks up the chrome and gets this wrong.
    """
    chrome = "<p>" + ("Master Circular navigation menu " * 20) + "</p>"
    body = (
        "<p>RBI/2026-27/262 DOR.AML.REC.1/14.01.005/2026-27 September 18, 2026 "
        "Master Direction on Know Your Customer " + ("regulatory text " * 40) + "</p>"
    )
    entry = dataclasses.replace(ENTRY, title="")  # empty title so only the body decides
    doc = extract_document(
        entry, (chrome + body).encode("utf-8"), fetched_at=datetime(2026, 9, 19, tzinfo=UTC)
    )
    assert doc.doc_type == "master_direction"


def test_extract_document_populates_the_document() -> None:
    doc = extract_document(ENTRY, RAW, fetched_at=datetime(2026, 9, 19, tzinfo=UTC))
    assert doc.doc_id == "rbi-13704"
    assert doc.circular_no == "RBI/2026-27/262"
    assert doc.doc_type in {"master_direction", "circular", "notification"}
    assert len(doc.text) > 500
    assert doc.text == canonical_text(RAW.decode("utf-8", errors="replace"))
    assert doc.effective_date is None


def test_extract_document_rejects_short_text() -> None:
    with pytest.raises(ExtractionError, match="too short"):
        extract_document(
            ENTRY, b"<html><body>RBI/2026-27/262 tiny</body></html>",
            fetched_at=datetime(2026, 9, 19, tzinfo=UTC),
        )
