"""Pins `_classify`'s real behaviour over the cached corpus.

`test_extract.py::test_extract_document_populates_the_document` only asserts
`doc.doc_type in {"master_direction", "circular", "notification"}` — a tautology
that cannot fail no matter what the classifier does. This file instead asserts
specific, real `doc_id -> doc_type` mappings observed against the warm
`data/raw/` cache, so a change in classification — to `_classify`'s keywords, to
what text it is given, or to extraction upstream of it — is visible as a failing
assertion here rather than silently passing.

See docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md section 10
for the documented limitation this pins: `_classify` produces zero `circular`
labels on the real corpus, and two near-identical "Implementation of Section 51A
of UAPA" documents (rbi-12922, rbi-13310) are classified differently from each
other. The cause is keyword presence in the text `_classify` hashes, which is
`title + window` — rbi-12922 contains "master direction" in both its title and
its window, rbi-13310 in neither. It is *not* the window boundary: widening
`CLASSIFY_WINDOW_CHARS` from 600 to 2,000 changes zero labels across all 61
documents, so a change to that constant alone would not fail these pins.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from clause.ingest.extract import extract_document
from clause.sources.manifest import load_manifest

MANIFEST = Path("data/corpus/kyc.manifest.jsonl")
RAW_CACHE_DIR = Path("data/raw")

# A representative sample, not the full 61: two documents with the exact same
# subject line ("Implementation of Section 51A of UAPA, 1967: ...") that this
# classifier nonetheless splits across both non-"circular" labels, plus one
# genuine master_direction amendment and one genuine notification.
EXPECTED_DOC_TYPES = {
    "rbi-12922": "master_direction",
    "rbi-13310": "notification",
    "rbi-12893": "master_direction",
    "rbi-13704": "notification",
}


def _require_warm_cache() -> None:
    if not MANIFEST.exists():
        pytest.skip("manifest not present")
    missing = [
        doc_id for doc_id in EXPECTED_DOC_TYPES if not (RAW_CACHE_DIR / f"{doc_id}.html").exists()
    ]
    if missing:
        pytest.skip(
            f"warm data/raw/ cache required for {missing}; populate it with "
            "`uv run python -m clause.cli ingest --manifest data/corpus/kyc.manifest.jsonl`"
        )


def test_classifier_pins_real_doc_type_labels_over_the_cached_corpus() -> None:
    _require_warm_cache()
    entries = {e.doc_id: e for e in load_manifest(MANIFEST)}
    for doc_id, expected_type in EXPECTED_DOC_TYPES.items():
        entry = entries[doc_id]
        raw = (RAW_CACHE_DIR / f"{doc_id}.html").read_bytes()
        doc = extract_document(entry, raw, fetched_at=datetime.now(UTC))
        assert doc.doc_type == expected_type, (
            f"{doc_id}: expected doc_type {expected_type!r}, got {doc.doc_type!r} "
            "-- classification has changed; update this pin deliberately if the "
            "change is intended, do not just widen the assertion"
        )


def test_classifier_produces_zero_circular_labels_over_the_full_corpus() -> None:
    """Documents the classifier's known limitation rather than hiding it.

    If this starts failing because `circular` labels appear, that is good news and
    this test (and the spec section documenting the limitation) should be updated
    to reflect it.
    """
    if not MANIFEST.exists():
        pytest.skip("manifest not present")
    entries = load_manifest(MANIFEST)
    missing = [e.doc_id for e in entries if not (RAW_CACHE_DIR / f"{e.doc_id}.html").exists()]
    if missing:
        pytest.skip(f"warm data/raw/ cache required for the full corpus; missing {len(missing)}")

    doc_types = set()
    for entry in entries:
        raw = (RAW_CACHE_DIR / f"{entry.doc_id}.html").read_bytes()
        doc = extract_document(entry, raw, fetched_at=datetime.now(UTC))
        doc_types.add(doc.doc_type)

    assert "circular" not in doc_types, (
        "the classifier now produces a 'circular' label on the real corpus; this "
        "test and the spec's documented limitation (section 5.3/10) are stale and "
        "should be updated, not just made to pass"
    )
