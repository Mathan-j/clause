import json
import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from clause.models import ManifestEntry

# `Fetcher._cache_path` (clause.ingest.fetch) builds `raw_cache_dir / f"{doc_id}.html"`
# from this field and *writes* to it. An unvalidated doc_id containing a path
# separator (or `..`) could traverse out of the cache directory, or an absolute-path
# doc_id could escape it entirely. Exploiting this needs commit access to the
# manifest, but load_manifest is the trust boundary where the corpus is frozen, and
# it validated nothing before this check existed.
DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


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
                "content_sha256": e.content_sha256,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        doc_id = row["doc_id"]
        if not DOC_ID_PATTERN.match(doc_id):
            raise ValueError(
                f"invalid doc_id {doc_id!r} at line {lineno}: must match "
                f"{DOC_ID_PATTERN.pattern!r} (used verbatim to build a cache file "
                "path; '..' or a path separator could write outside the cache dir)"
            )
        entry = ManifestEntry(
            doc_id=doc_id,
            rbi_id=int(row["rbi_id"]),
            url=row["url"],
            circular_no=row["circular_no"],
            dept_ref=row["dept_ref"],
            title=row["title"],
            published_date=date.fromisoformat(row["published_date"]),
            content_sha256=row["content_sha256"],
        )
        if entry.doc_id in seen:
            raise ValueError(f"duplicate doc_id {entry.doc_id!r} at line {lineno}")
        seen.add(entry.doc_id)
        entries.append(entry)
    return entries
