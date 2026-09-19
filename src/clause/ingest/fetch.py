import hashlib
import random
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from clause.config import Settings
from clause.ingest.validate import validate_response
from clause.models import ManifestEntry

HTTP_OK = 200


class FetchError(Exception):
    """A document could not be retrieved."""


class HashMismatchError(FetchError):
    """Content differs from the manifest. RBI revises circulars in place."""


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
        HashMismatchError if it differs from the manifest.
        """
        path = self._cache_path(entry)
        if path.exists():
            cached = path.read_bytes()
            self._verify(entry, cached)
            return cached

        response = self._get_with_retries(entry)
        body = response.content

        validate_response(
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
            body=body.decode("utf-8", errors="replace"),
            min_chars=self._s.min_document_chars,
        )
        self._verify(entry, body)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return body

    def _verify(self, entry: ManifestEntry, body: bytes) -> None:
        actual = hashlib.sha256(body).hexdigest()
        if actual != entry.sha256:
            raise HashMismatchError(
                f"{entry.doc_id}: manifest sha256 {entry.sha256} but content is {actual}. "
                "RBI may have revised this circular; re-run discovery and review the diff "
                "rather than editing the manifest."
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
