import hashlib
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from clause.config import Settings
from clause.ingest.extract import ExtractionError, canonical_text, parse_header
from clause.ingest.validate import validate_response
from clause.models import ManifestEntry

HTTP_OK = 200


class FetchError(Exception):
    """A document could not be retrieved."""


class HashMismatchError(FetchError):
    """Content differs from the manifest in a way page-chrome noise cannot explain.

    See docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md section 9:
    neither a raw-byte hash nor a whole-page canonical-text hash is reproducible
    against www.rbi.org.in on every fetch (the site's WAF injects a per-response
    token, and page furniture such as a PDF-size widget has been observed to vary
    between fetches of an unchanged document). A `content_sha256` mismatch on a live
    fetch is therefore tolerated when the document's header identity (circular_no,
    dept_ref, published_date) still matches the manifest; this error is raised only
    when header identity has also changed or no longer parses — the case it actually
    exists to catch: RBI revising, renumbering or replacing a circular.
    """


class RateLimiter:
    """Single-flight minimum interval between requests.

    robots.txt returns 418 to every client tried, so compliance with it cannot
    be claimed. A conservative fixed floor stands in its place.
    """

    def __init__(
        self,
        min_interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            elapsed = self._clock() - self._last
            remaining = self._min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


class Fetcher:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client,
        limiter: RateLimiter,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._s = settings
        self._client = client
        self._limiter = limiter
        self._sleep = sleep

    def _cache_path(self, entry: ManifestEntry) -> Path:
        return self._s.raw_cache_dir / f"{entry.doc_id}.html"

    def fetch(self, entry: ManifestEntry) -> bytes:
        """Return raw bytes for this document, using the cache when warm.

        Raises ValidationError if the response is not a real circular, and
        HashMismatchError if it differs from the manifest in a way that cannot be
        explained by RBI's page-chrome noise (see `HashMismatchError`).
        """
        path = self._cache_path(entry)
        if path.exists():
            cached = path.read_bytes()
            self._verify_cached(entry, cached)
            return cached

        response = self._get_with_retries(entry)
        body = response.content

        validate_response(
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
            body=body.decode("utf-8", errors="replace"),
            min_chars=self._s.min_document_chars,
        )
        self._verify_live(entry, body)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return body

    def _verify_live(self, entry: ManifestEntry, body: bytes) -> None:
        """Two-tier check, tolerant of WAF/page-chrome noise, for a live network fetch.

        Fast path: if `content_sha256` (sha256 of `canonical_text`) matches the
        manifest exactly, accept immediately.

        Otherwise, fall back to header identity: reparse `circular_no`, `dept_ref`
        and `published_date` from the fetched text. If they still match the
        manifest, the content difference is page-chrome noise (a WAF token, a
        volatile widget) rather than a real revision — log a warning naming the
        document and accept. Only when header identity has also changed, or the
        header no longer parses, is this a hard failure: that combination is what
        actually indicates RBI revised, renumbered or replaced the circular.
        """
        text = canonical_text(body.decode("utf-8", errors="replace"))
        actual_content = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual_content == entry.content_sha256:
            return

        try:
            circular_no, dept_ref, published = parse_header(text)
        except ExtractionError as exc:
            raise HashMismatchError(
                f"{entry.doc_id}: manifest content_sha256 {entry.content_sha256} but "
                f"fetched content hashes to {actual_content}, and its header no longer "
                f"parses ({exc}). Not page-chrome noise — re-run discovery and review "
                "this document by hand rather than editing the manifest."
            ) from exc

        if (circular_no, dept_ref, published) == (
            entry.circular_no,
            entry.dept_ref,
            entry.published_date,
        ):
            print(
                f"WARNING: {entry.doc_id}: content_sha256 mismatch (manifest "
                f"{entry.content_sha256}, fetched {actual_content}) but header identity "
                "(circular_no/dept_ref/published_date) is unchanged — treating as RBI "
                "page-chrome noise (a WAF token or a volatile widget), not a revision. "
                "Accepting.",
                file=sys.stderr,
            )
            return

        raise HashMismatchError(
            f"{entry.doc_id}: manifest content_sha256 {entry.content_sha256} but fetched "
            f"content hashes to {actual_content}, AND header identity changed (manifest "
            f"circular_no={entry.circular_no!r} dept_ref={entry.dept_ref!r} "
            f"published_date={entry.published_date} vs fetched "
            f"circular_no={circular_no!r} dept_ref={dept_ref!r} published_date={published}"
            "). RBI appears to have revised this circular in place; re-run discovery and "
            "review the diff rather than editing the manifest."
        )

    def _verify_cached(self, entry: ManifestEntry, body: bytes) -> None:
        """Strict check against a file already sitting in the on-disk cache.

        No header-identity fallback here, unlike `_verify_live`: these bytes are
        ours, written by this process after `_verify_live` already accepted them —
        not a fresh network response, so they carry none of the WAF/page-chrome
        volatility a live fetch does. A mismatch here means the cached file was
        corrupted or modified on disk after caching, which is exactly the failure
        mode byte-level hashing still has real value catching. The asymmetry with
        `_verify_live` is deliberate: tolerance belongs at the network boundary
        where the noise originates, not at the disk-read boundary where it does not.
        """
        text = canonical_text(body.decode("utf-8", errors="replace"))
        actual_content = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual_content != entry.content_sha256:
            raise HashMismatchError(
                f"{entry.doc_id}: cached file content_sha256 mismatch (manifest "
                f"{entry.content_sha256}, cached file hashes to {actual_content}). The "
                "cached copy on disk appears corrupted or modified after caching; delete "
                "it and re-fetch rather than trusting it."
            )

    def _get_with_retries(self, entry: ManifestEntry) -> httpx.Response:
        """Return the response itself, not just its bytes.

        The validation gate must see the real status code and content-type
        header; passing it hardcoded values would mean a PDF or an error page
        sailed through the check it exists to perform.
        """
        last: Exception | None = None
        for attempt in range(self._s.max_retries + 1):
            self._limiter.wait()
            try:
                response = self._client.get(
                    entry.url,
                    headers={"User-Agent": self._s.user_agent},
                    timeout=30.0,
                    follow_redirects=True,
                )
            except httpx.HTTPError as exc:
                last = exc
            else:
                if response.status_code == HTTP_OK:
                    return response
                last = FetchError(f"{entry.doc_id}: HTTP {response.status_code}")

            if attempt < self._s.max_retries:
                backoff = (2.0**attempt) + random.uniform(0, 0.5)
                self._sleep(backoff)

        raise FetchError(
            f"{entry.doc_id}: giving up after {self._s.max_retries + 1} attempts: {last}"
        )
