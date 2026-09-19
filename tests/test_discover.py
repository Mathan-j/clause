import hashlib
from pathlib import Path

from clause.sources.discover import build_entry, is_kyc_document
from clause.sources.manifest import load_manifest

FIXTURES = Path(__file__).parent / "fixtures"
RAW = (FIXTURES / "rbi_13704.html").read_bytes()


def test_recognises_a_kyc_document() -> None:
    assert is_kyc_document("Know Your Customer Amendment Directions, 2026", "") is True


def test_rejects_an_unrelated_document() -> None:
    assert is_kyc_document("Priority Sector Lending targets", "housing loans") is False


def test_build_entry_derives_id_and_hash() -> None:
    entry = build_entry(13704, "https://example.test/d", RAW, "Know Your Customer")
    assert entry.doc_id == "rbi-13704"
    assert entry.sha256 == hashlib.sha256(RAW).hexdigest()
    assert entry.circular_no == "RBI/2026-27/262"


def test_committed_manifest_is_loadable_and_large_enough() -> None:
    entries = load_manifest(Path("data/corpus/kyc.manifest.jsonl"))
    assert len(entries) >= 50, "PROMPT.md Phase 1 requires at least 50 real documents"
    assert len({e.doc_id for e in entries}) == len(entries)
    assert all(len(e.sha256) == 64 for e in entries)
