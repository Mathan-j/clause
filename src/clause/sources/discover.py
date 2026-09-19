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

Usage:
    uv run python -m clause.sources.discover --start-id 13704 --count 400 \
        --out data/corpus/kyc.manifest.jsonl

    # Re-fetch every entry already in a committed manifest to correct their titles
    # (does not change corpus selection; aborts if any sha256 has drifted):
    uv run python -m clause.sources.discover --retitle data/corpus/kyc.manifest.jsonl \
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
        sha256=hashlib.sha256(raw).hexdigest(),
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
    """Return the subject line following the RBI header line (and salutation, if any).

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


def _run_retitle(manifest_path: Path, out_path: Path, *, interval: float, user_agent: str) -> int:
    """Re-fetch every entry in an existing manifest and correct its title in place.

    Performs no discovery and changes no corpus-selection decision: it only refetches
    each entry's own URL and rewrites `title` using the current `_title_of` logic.
    Every other field, *including the originally recorded sha256*, is carried over
    byte-identical from the source manifest — this function never adopts a freshly
    fetched hash into the manifest.

    Drift check: this does **not** compare a live sha256 of the raw response against
    the manifest's recorded one. www.rbi.org.in sits behind an F5 BIG-IP WAF that
    injects a `f5_cspm` client-side-persistence script containing a freshly randomised
    token into (at least) every `NotificationUser.aspx` response. Fetching the same
    URL twice, two seconds apart, was confirmed to produce three different raw-byte
    sha256 values for the same document while `canonical_text` extracted from each was
    byte-identical — i.e. a raw-byte hash comparison against a live refetch would
    report "drift" on every single entry, always, independent of whether the actual
    circular content changed. Comparing raw bytes is therefore not a usable freeze
    check for this source.

    Instead, this compares the header fields `parse_header` recovers from the fresh
    fetch (`circular_no`, `dept_ref`, `published_date`) against the manifest's
    recorded values, and raises `ManifestDriftError` if any differ. This catches RBI
    replacing a circular under a new number/date or altering its own header, but it
    cannot detect a silent in-place body edit that keeps the same header — no
    canonical-text hash was captured at discovery time to compare against, only the
    (now known to be unusable for this purpose) raw-byte sha256. A future task wanting
    a stronger guarantee would need to start persisting a canonical-text hash in the
    manifest going forward.
    """
    entries = load_manifest(manifest_path)
    retitled: list[ManifestEntry] = []
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
            retitled.append(
                ManifestEntry(
                    doc_id=entry.doc_id,
                    rbi_id=entry.rbi_id,
                    url=entry.url,
                    circular_no=entry.circular_no,
                    dept_ref=entry.dept_ref,
                    title=new_title.strip() or entry.circular_no,
                    published_date=entry.published_date,
                    sha256=entry.sha256,
                )
            )
            print(f"  retitled {entry.doc_id}: {new_title[:70]}", file=sys.stderr)

    write_manifest(out_path, retitled)
    print(f"wrote {len(retitled)} retitled entries to {out_path}", file=sys.stderr)
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
        "--retitle",
        type=Path,
        help=(
            "Re-fetch every entry already in this manifest (no new discovery), "
            "verify its sha256 is unchanged, and rewrite its title with the current "
            "_title_of logic. Writes the result to --out."
        ),
    )
    args = parser.parse_args()

    if args.retitle is not None:
        return _run_retitle(
            args.retitle, args.out, interval=args.interval, user_agent=args.user_agent
        )

    if args.start_id is None:
        parser.error("--start-id is required unless --retitle is given")

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
