"""One-off corpus discovery. NOT invoked by `make ingest`.

Run by hand, review the output, commit the manifest. That review step is what
freezes the corpus, which is what makes Phase 2's eval numbers attributable to
code changes rather than to RBI publishing something overnight.

`is_kyc_document` only inspects `text[:4000]` (see its haystack below). A KYC/AML
reference that first appears later in a long document is a false negative this
function will not catch — a corpus-narrowing risk, not a corpus-widening one. This
is a plausible contributor to how few hits the first discovery pass returned, and is
noted here rather than fixed, since widening the window changes what documents are
considered "on-topic" and that decision belongs to a reviewed change, not a quiet one.

`ManifestEntry.content_sha256` is the sha256 of `canonical_text(raw_html)` at
discovery time, not of the raw response bytes. www.rbi.org.in sits behind a WAF that
injects a freshly randomised token into every raw response (so a raw-byte hash never
reproduces), and even `canonical_text` has been observed to vary between fetches in
trailing page furniture (a PDF-size widget), so `content_sha256` is a best-effort
snapshot, not a guarantee of whole-document stability. See
`docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md` section 9.

Usage:
    uv run python -m clause.sources.discover --start-id 13704 --count 400 \
        --out data/corpus/kyc.manifest.jsonl

    # Re-fetch every entry already in a committed manifest to refresh its title and
    # content_sha256 (does not change corpus selection; aborts if header identity —
    # circular_no/dept_ref/published_date — has drifted):
    uv run python -m clause.sources.discover --resync data/corpus/kyc.manifest.jsonl \
        --out data/corpus/kyc.manifest.jsonl
"""

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import httpx

from clause.ingest.extract import HEADER, ExtractionError, canonical_text, parse_header
from clause.ingest.validate import ValidationError, validate_response
from clause.models import ManifestEntry
from clause.sources.manifest import load_manifest, write_manifest

URL_TEMPLATE = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={rbi_id}&Mode=0"

MIN_DOCUMENT_CHARS = 500
TITLE_SEARCH_CHARS = 600
TITLE_WINDOW_CHARS = 200
DEFAULT_COUNT = 400
DEFAULT_TARGET = 60
DEFAULT_INTERVAL_S = 2.0
DEFAULT_USER_AGENT = "clause-research/0.1 (+mailto:you@example.com)"
MIN_MANIFEST_ENTRIES = 50

KYC_TERMS: tuple[str, ...] = (
    "know your customer",
    "kyc",
    "anti-money laundering",
    "money laundering",
    "prevention of money laundering",
    "pmla",
    "customer due diligence",
    "beneficial owner",
    "combating the financing of terrorism",
)


def is_kyc_document(title: str, text: str) -> bool:
    haystack = f"{title} {text[:4000]}".lower()
    return any(term in haystack for term in KYC_TERMS)


def _content_sha256(text: str) -> str:
    """sha256 of the canonical extracted text — what `ManifestEntry.content_sha256` stores.

    Deliberately not a hash of raw response bytes: see the module docstring for why a
    raw-byte hash is unusable against this source.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_entry(rbi_id: int, url: str, raw: bytes, title: str) -> ManifestEntry:
    text = canonical_text(raw.decode("utf-8", errors="replace"))
    circular_no, dept_ref, published = parse_header(text)
    return ManifestEntry(
        doc_id=f"rbi-{rbi_id}",
        rbi_id=rbi_id,
        url=url,
        circular_no=circular_no,
        dept_ref=dept_ref,
        title=title.strip() or circular_no,
        published_date=published,
        content_sha256=_content_sha256(text),
    )


# Many RBI circulars address a salutation to regulated entities between the header
# line and the actual subject line, e.g.:
#   "...September 18, 2026  The Chairpersons/ CEOs of ... All India Financial
#   Institutions  Madam/Dear Sir,  Implementation of Section 51A of UAPA, 1967: ..."
# Without skipping past the salutation, `_title_of` returns the addressee list
# instead of the subject line. Spellings observed in the corpus: "Madam/Dear Sir,",
# "Madam/ Dear Sir,", "Dear Sir/Madam,", "Dear Sir / Madam,", "Dear Sir/ Madam,",
# "Dear Madam,", "Madam,", "Dear Sir," — this pattern is not guaranteed exhaustive
# against spellings RBI has not yet used.
SALUTATION = re.compile(
    r"(?:Madam\s*/?\s*Dear Sir|Dear Sir\s*/?\s*Madam|Dear Madam|Madam|Dear Sir)\s*,",
    re.IGNORECASE,
)


def _title_of(text: str) -> str:
    """Return a fixed ~200-character window of text following the RBI header line
    (and salutation, if any) — an approximation of the subject line, not a clean
    extraction of it. The window is a raw character count, not a sentence or
    paragraph boundary, so long subjects are truncated mid-word or mid-sentence
    (e.g. "...Amendment Directions, 2026 R" or "...ISIL (Da'esh) &"); of the 61
    titles currently committed to `data/corpus/kyc.manifest.jsonl`, none is a clean,
    exact subject line. Titles are useful for a human skimming the manifest, not as
    a field to match on exactly.

    The brief's original approach re-derived the date string with
    ``strftime("%B %-d, %Y")`` and searched for it in the text. That is
    locale-dependent in the same way the `strptime` bug in `parse_header`
    was (a non-English `LC_TIME` produces a month name the English text
    never contains, silently returning "" for every title), and it also
    needed a platform-specific `%-d`/`%#d` branch. `HEADER` has already
    located the date during header parsing, so reuse its match position
    instead of re-deriving and re-finding the date string.

    Many circulars additionally interpose an addressee block and a salutation
    ("Madam/Dear Sir,") between the header and the subject line; slicing straight
    after `HEADER`'s match captures that addressee block instead of the subject.
    When a salutation is found within the search window, the returned title starts
    after it.
    """
    match = HEADER.search(text)
    if match is None:
        return ""
    window = text[match.end() : match.end() + TITLE_SEARCH_CHARS]
    salutation = SALUTATION.search(window)
    if salutation is not None:
        window = window[salutation.end() :]
    return window[:TITLE_WINDOW_CHARS].strip()


class ManifestDriftError(Exception):
    """A re-fetched document's header no longer matches the frozen manifest.

    The frozen corpus is the entire point of the manifest: Phase 2's eval numbers are
    only attributable to a code change if the corpus underneath them has not moved.
    This is raised rather than silently accepted so a drifted document is investigated,
    not quietly folded into the manifest.
    """


def _run_resync(manifest_path: Path, out_path: Path, *, interval: float, user_agent: str) -> int:
    """Re-fetch every entry in an existing manifest and refresh its derived fields.

    Performs no discovery and changes no corpus-selection decision: it only refetches
    each entry's own URL and rewrites `title` (current `_title_of`) and
    `content_sha256` (current `_content_sha256`, i.e. sha256 of `canonical_text`) from
    that fetch. Every other field is carried over byte-identical from the source
    manifest.

    `content_sha256` is *recomputed*, never carried over: the field this replaces
    (`sha256`, retired — see the field rename in `clause.models.ManifestEntry`) hashed
    raw response bytes, which were never a hash of canonical_text to begin with and are
    not reproducible against this source anyway (see the module docstring), so there is
    nothing meaningful in the old value to preserve.

    Drift check: compares the header fields `parse_header` recovers from the fresh
    fetch (`circular_no`, `dept_ref`, `published_date`) against the manifest's
    recorded values, and raises `ManifestDriftError` if any differ. This catches RBI
    replacing a circular under a new number/date or altering its own header. It
    cannot catch a silent in-place body edit that keeps the same header — the same
    limitation `content_sha256` has once page-chrome noise is tolerated (see
    `Fetcher._verify` in `clause.ingest.fetch`, which accepts a `content_sha256`
    mismatch precisely when header identity still matches).
    """
    entries = load_manifest(manifest_path)
    resynced: list[ManifestEntry] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for entry in entries:
            time.sleep(interval)
            response = client.get(entry.url, headers={"User-Agent": user_agent})
            body = response.content
            validate_response(
                status_code=response.status_code,
                content_type=response.headers.get("content-type", ""),
                body=body.decode("utf-8", errors="replace"),
                min_chars=MIN_DOCUMENT_CHARS,
            )
            text = canonical_text(body.decode("utf-8", errors="replace"))
            circular_no, dept_ref, published = parse_header(text)
            if (circular_no, dept_ref, published) != (
                entry.circular_no,
                entry.dept_ref,
                entry.published_date,
            ):
                raise ManifestDriftError(
                    f"{entry.doc_id}: manifest header (circular_no={entry.circular_no!r}, "
                    f"dept_ref={entry.dept_ref!r}, published_date={entry.published_date}) "
                    f"but the document at {entry.url} now parses as "
                    f"(circular_no={circular_no!r}, dept_ref={dept_ref!r}, "
                    f"published_date={published}). RBI appears to have revised this "
                    "circular's header since the manifest was frozen. Not overwriting "
                    "the manifest — investigate this document by hand before re-running."
                )
            new_title = _title_of(text)
            new_content_sha256 = _content_sha256(text)
            resynced.append(
                ManifestEntry(
                    doc_id=entry.doc_id,
                    rbi_id=entry.rbi_id,
                    url=entry.url,
                    circular_no=entry.circular_no,
                    dept_ref=entry.dept_ref,
                    title=new_title.strip() or entry.circular_no,
                    published_date=entry.published_date,
                    content_sha256=new_content_sha256,
                )
            )
            print(
                f"  resynced {entry.doc_id}: title={new_title[:50]!r} "
                f"content_sha256={new_content_sha256[:12]}...",
                file=sys.stderr,
            )

    write_manifest(out_path, resynced)
    print(f"wrote {len(resynced)} resynced entries to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-id", type=int)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--resync",
        type=Path,
        help=(
            "Re-fetch every entry already in this manifest (no new discovery), "
            "verify its header identity is unchanged, and rewrite its title and "
            "content_sha256 from the fresh fetch. Writes the result to --out."
        ),
    )
    args = parser.parse_args()

    if args.resync is not None:
        return _run_resync(
            args.resync, args.out, interval=args.interval, user_agent=args.user_agent
        )

    if args.start_id is None:
        parser.error("--start-id is required unless --resync is given")

    entries: list[ManifestEntry] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for offset in range(args.count):
            if len(entries) >= args.target:
                break
            rbi_id = args.start_id - offset
            url = URL_TEMPLATE.format(rbi_id=rbi_id)
            time.sleep(args.interval)
            try:
                response = client.get(url, headers={"User-Agent": args.user_agent})
                body = response.content
                validate_response(
                    status_code=response.status_code,
                    content_type=response.headers.get("content-type", ""),
                    body=body.decode("utf-8", errors="replace"),
                    min_chars=MIN_DOCUMENT_CHARS,
                )
                text = canonical_text(body.decode("utf-8", errors="replace"))
                title = _title_of(text)
                if not is_kyc_document(title, text):
                    continue
                entries.append(build_entry(rbi_id, url, body, title))
                print(f"  kept {rbi_id}: {title[:70]}", file=sys.stderr)
            except (ValidationError, ExtractionError, httpx.HTTPError) as exc:
                print(f"  skip {rbi_id}: {exc}", file=sys.stderr)
                continue

    write_manifest(args.out, entries)
    print(f"wrote {len(entries)} entries to {args.out}", file=sys.stderr)
    return 0 if len(entries) >= MIN_MANIFEST_ENTRIES else 1


if __name__ == "__main__":
    raise SystemExit(main())
