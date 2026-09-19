from datetime import date
from pathlib import Path

import pytest

from clause.models import ManifestEntry
from clause.sources.manifest import load_manifest, write_manifest

ENTRY = ManifestEntry(
    doc_id="rbi-13704",
    rbi_id=13704,
    url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13704&Mode=0",
    circular_no="RBI/2026-27/262",
    dept_ref="DOR.AML.REC.223/14.01.005/2026-27",
    title="Reserve Bank of India (Rural Co-operative Banks - Know Your Customer) "
    "Amendment Directions, 2026",
    published_date=date(2026, 9, 18),
    content_sha256="a" * 64,
)


def test_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    write_manifest(p, [ENTRY])
    assert load_manifest(p) == [ENTRY]


def test_duplicate_doc_ids_are_rejected(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    write_manifest(p, [ENTRY, ENTRY])
    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_manifest(p)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    write_manifest(p, [ENTRY])
    p.write_text(p.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(load_manifest(p)) == 1
