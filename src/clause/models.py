from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    doc_id: str
    rbi_id: int
    url: str
    circular_no: str
    dept_ref: str
    title: str
    published_date: date
    content_sha256: str
    """sha256 of `canonical_text(raw_html)` at discovery time — NOT of the raw response
    bytes. www.rbi.org.in sits behind a WAF that injects a freshly randomised token into
    every raw response, and even the extracted canonical text has been observed to vary
    between fetches in trailing page furniture (a PDF-size widget), so this field cannot
    guarantee whole-document byte stability either. It is a discovery-time snapshot used
    to detect content drift on a best-effort basis; identity fields (`circular_no`,
    `dept_ref`, `published_date`) are the check that does not depend on page chrome. See
    `docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md` section 9.
    """


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    rbi_id: int
    url: str
    circular_no: str
    dept_ref: str
    title: str
    doc_type: str
    published_date: date
    effective_date: date | None
    sha256: str
    fetched_at: datetime
    text: str
    """Canonical, immutable. Every char offset in the system indexes into this string."""
    regulated_entity: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Chunk:
    doc_id: str
    strategy: str
    ordinal: int
    char_start: int
    char_end: int
    text: str
    source_url: str
    effective_date: date | None
    doc_type: str
    regulated_entity: tuple[str, ...] = ()
