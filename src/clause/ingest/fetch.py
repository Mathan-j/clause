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
    between fetches of an unchanged document). A `content_sha256` mismatch is
    therefore tolerated when the document's header identity (circular_no, dept_ref,
    published_date) still matches the manifest; this error is raised only when
    header identity has also changed or no longer parses — the case it actually
    exists to catch: RBI revising, renumbering or replacing a circular. The same
    check applies whether the bytes came from a live fetch or the on-disk cache
    (`Fetcher._verify`) — a cache entry is not held to a stricter standard than the
    fetch that produced it, since it may itself have been accepted through this
    same tolerant branch.
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

        Raises ValidationError if the content (live or cached) is not a real
        circular, and HashMismatchError if it differs from the manifest in a way
        that cannot be explained by RBI's page-chrome noise (see
        `HashMismatchError`). Both checks run on the cache-hit path too, not only
        on a live fetch: a cached file is not assumed trustworthy just because it
        is ours, since it may have been corrupted on disk, or may itself have been
        accepted through `_verify`'s tolerant branch when it was first written.
        """
        path = self._cache_path(entry)
        if path.exists():
            cached = path.read_bytes()
            validate_response(
                status_code=HTTP_OK,
                content_type="text/html",
                body=cached.decode("utf-8", errors="replace"),
                min_chars=self._s.min_document_chars,
            )
            self._verify(entry, cached, source="cache")
            return cached

        response = self._get_with_retries(entry)
        body = response.content

        validate_response(
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
            body=body.decode("utf-8", errors="replace"),
            min_chars=self._s.min_document_chars,
        )
        self._verify(entry, body, source="fetch")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return body

    def _verify(self, entry: ManifestEntry, body: bytes, *, source: str) -> None:
        """Two-tier check, tolerant of WAF/page-chrome noise, applied identically
        whether `body` came from a live fetch or the on-disk cache.

        Fast path: if `content_sha256` (sha256 of `canonical_text`) matches the
        manifest exactly, accept immediately.

        Otherwise, fall back to header identity: reparse `circular_no`, `dept_ref`
        and `published_date` from the text. If they still match the manifest, the
        content difference is page-chrome noise (a WAF token, a volatile widget)
        rather than a real revision — log a warning naming the document and
        accept. Only when header identity has also changed, or the header no
        longer parses, is this a hard failure: that combination is what actually
        indicates RBI revised, renumbered or replaced the circular.

        `source` is used only in the warning/error text ("fetch" or "cache"); the
        logic is identical on both paths. It is deliberately identical: an earlier
        version of this method held the cache path to a stricter, non-tolerant
        standard on the reasoning that cached bytes are "ours" and therefore free
        of network noise. That reasoning missed that the bytes sitting in the
        cache may themselves have been accepted through this same tolerant branch
        when they were first fetched — in which case they were never expected to
        match `content_sha256` exactly, and a strict re-check would reject them
        forever, on every subsequent read, for a document that was never actually
        wrong. See docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md
        section 9.
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
                f"{source} content hashes to {actual_content}, and its header no "
                f"longer parses ({exc}). Not page-chrome noise — re-run discovery "
                "and review this document by hand rather than editing the manifest."
            ) from exc

        if (circular_no, dept_ref, published) == (
            entry.circular_no,
            entry.dept_ref,
            entry.published_date,
        ):
            print(
                f"WARNING: {entry.doc_id}: content_sha256 mismatch (manifest "
                f"{entry.content_sha256}, {source} {actual_content}) but header "
                "identity (circular_no/dept_ref/published_date) is unchanged — "
                "treating as RBI page-chrome noise (a WAF token or a volatile "
                "widget), not a revision. Accepting.",
                file=sys.stderr,
            )
            return

        raise HashMismatchError(
            f"{entry.doc_id}: manifest content_sha256 {entry.content_sha256} but "
            f"{source} content hashes to {actual_content}, AND header identity "
            f"changed (manifest circular_no={entry.circular_no!r} "
            f"dept_ref={entry.dept_ref!r} published_date={entry.published_date} vs "
            f"{source} circular_no={circular_no!r} dept_ref={dept_ref!r} "
            f"published_date={published}). RBI appears to have revised this "
            "circular in place; re-run discovery and review the diff rather than "
            "editing the manifest."
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
