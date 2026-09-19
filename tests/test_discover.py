import hashlib
from pathlib import Path

from clause.ingest.extract import canonical_text
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
    expected_text = canonical_text(RAW.decode("utf-8", errors="replace"))
    assert entry.content_sha256 == hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
    assert entry.circular_no == "RBI/2026-27/262"


def test_committed_manifest_is_loadable_and_large_enough() -> None:
    entries = load_manifest(Path("data/corpus/kyc.manifest.jsonl"))
    assert len(entries) >= 50, "PROMPT.md Phase 1 requires at least 50 real documents"
    assert len({e.doc_id for e in entries}) == len(entries)
    assert all(len(e.content_sha256) == 64 for e in entries)


ADDRESSEE_MARKERS = (
    "Madam",
    "Dear Sir",
    "The Chairperson",
    "All Scheduled",
    "Chief Executive",
)


def test_committed_manifest_titles_are_not_addressee_blocks() -> None:
    """A title that is just the salutation/addressee list is not a title.

    `_title_of` used to slice straight after the RBI header line, which for many
    circulars lands on "The Chairpersons/ CEOs of ... Madam/Dear Sir," rather than
    the subject line. This pins the fix against the committed manifest so a future
    regression in `_title_of` (or a re-run that reintroduces it) is caught here
    rather than discovered by a human skimming titles again.
    """
    entries = load_manifest(Path("data/corpus/kyc.manifest.jsonl"))
    for entry in entries:
        assert not any(marker in entry.title for marker in ADDRESSEE_MARKERS), (
            f"{entry.doc_id} title looks like an addressee block, not a subject "
            f"line: {entry.title!r}"
        )
