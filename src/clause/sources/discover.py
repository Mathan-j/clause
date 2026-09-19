"""One-off corpus discovery. NOT invoked by `make ingest`.

Run by hand, review the output, commit the manifest. That review step is what
freezes the corpus, which is what makes Phase 2's eval numbers attributable to
code changes rather than to RBI publishing something overnight.

Usage:
    uv run python -m clause.sources.discover --start-id 13704 --count 400 \
        --out data/corpus/kyc.manifest.jsonl
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import httpx

from clause.ingest.extract import HEADER, ExtractionError, canonical_text, parse_header
from clause.ingest.validate import ValidationError, validate_response
from clause.models import ManifestEntry
from clause.sources.manifest import write_manifest

URL_TEMPLATE = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={rbi_id}&Mode=0"

MIN_DOCUMENT_CHARS = 500
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


def _title_of(text: str) -> str:
    """Return the text immediately following the RBI header line.

    The brief's original approach re-derived the date string with
    ``strftime("%B %-d, %Y")`` and searched for it in the text. That is
    locale-dependent in the same way the `strptime` bug in `parse_header`
    was (a non-English `LC_TIME` produces a month name the English text
    never contains, silently returning "" for every title), and it also
    needed a platform-specific `%-d`/`%#d` branch. `HEADER` has already
    located the date during header parsing, so reuse its match position
    instead of re-deriving and re-finding the date string.
    """
    match = HEADER.search(text)
    if match is None:
        return ""
    return text[match.end() : match.end() + TITLE_WINDOW_CHARS].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-id", type=int, required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    args = parser.parse_args()

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
