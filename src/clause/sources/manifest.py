import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from clause.models import ManifestEntry


def write_manifest(path: Path, entries: Iterable[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for e in entries:
            row = {
                "doc_id": e.doc_id,
                "rbi_id": e.rbi_id,
                "url": e.url,
                "circular_no": e.circular_no,
                "dept_ref": e.dept_ref,
                "title": e.title,
                "published_date": e.published_date.isoformat(),
                "sha256": e.sha256,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        entry = ManifestEntry(
            doc_id=row["doc_id"],
            rbi_id=int(row["rbi_id"]),
            url=row["url"],
            circular_no=row["circular_no"],
            dept_ref=row["dept_ref"],
            title=row["title"],
            published_date=date.fromisoformat(row["published_date"]),
            sha256=row["sha256"],
        )
        if entry.doc_id in seen:
            raise ValueError(f"duplicate doc_id {entry.doc_id!r} at line {lineno}")
        seen.add(entry.doc_id)
        entries.append(entry)
    return entries
