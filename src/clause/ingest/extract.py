import hashlib
import re
import unicodedata
from datetime import date, datetime

from clause.entities import parse_regulated_entities
from clause.htmltext import document_text
from clause.models import Document, ManifestEntry
from clause.rbi_format import HEADER

MIN_TEXT_CHARS = 500

_WS = re.compile(r"[ \t\xa0]+")

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

CLASSIFY_WINDOW_CHARS = 600


class ExtractionError(Exception):
    """The response parsed, but is not a usable document."""


def canonical_text(html: str) -> str:
    """Produce the one immutable string every character offset indexes into.

    Called exactly once per document. Nothing downstream re-normalises: both
    chunkers slice this string and never transform it, which is what makes the
    citation round-trip hold by construction.

    Built from `document_text`, so it is the notification alone -- the surrounding
    site template is excluded before any offset is assigned.
    """
    text = document_text(html)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = _WS.sub(" ", text)
    return text.strip()


def parse_header(text: str) -> tuple[str, str, date]:
    match = HEADER.search(text)
    if match is None:
        raise ExtractionError("no RBI header line (circular no / dept ref / date) found")

    raw = match.group("published")
    try:
        month_name, rest = raw.split(" ", 1)
        day_str, year_str = rest.split(", ", 1)
        month = _MONTHS[month_name.lower()]
        published = date(int(year_str), month, int(day_str))
    except (KeyError, ValueError) as exc:
        raise ExtractionError(f"unparseable header date {raw!r}: {exc}") from exc

    return match.group("circular_no"), match.group("dept_ref"), published


def _classify(title: str, body: str) -> str:
    haystack = f"{title} {body}".lower()
    if "master direction" in haystack:
        return "master_direction"
    if "circular" in haystack:
        return "circular"
    return "notification"


def extract_document(entry: ManifestEntry, raw: bytes, *, fetched_at: datetime) -> Document:
    text = canonical_text(raw.decode("utf-8", errors="replace"))
    if len(text) < MIN_TEXT_CHARS:
        raise ExtractionError(f"extracted text too short: {len(text)} chars")

    circular_no, dept_ref, published = parse_header(text)

    # Anchor on the header regex's own match position, not the first occurrence
    # of circular_no in the text: a breadcrumb or nav link repeating the
    # circular number ahead of the real header would otherwise re-anchor the
    # window on chrome again. parse_header has already succeeded against this
    # same text, so this search is expected to match; the `if` is belt-and-braces.
    header_match = HEADER.search(text)
    header_at = header_match.start() if header_match else 0
    window = text[header_at : header_at + CLASSIFY_WINDOW_CHARS]

    return Document(
        doc_id=entry.doc_id,
        rbi_id=entry.rbi_id,
        url=entry.url,
        circular_no=circular_no,
        dept_ref=dept_ref,
        title=entry.title,
        doc_type=_classify(entry.title, window),
        published_date=published,
        effective_date=None,
        # Recomputed here, at extraction time, from the exact text this row stores
        # (document.text == this `text`) — NOT copied from entry.content_sha256,
        # which is a discovery-time hash that need not match either the raw fetched
        # bytes or documents.text (see ManifestEntry.content_sha256's own docstring,
        # and docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md
        # section 9). This way Document.sha256 is a checkable integrity value for
        # what is actually in documents.text: sha256(document.text.encode()).
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        fetched_at=fetched_at,
        text=text,
        regulated_entity=parse_regulated_entities(text),
    )
