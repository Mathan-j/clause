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
    sha256: str


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
