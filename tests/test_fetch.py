import hashlib
from datetime import date
from pathlib import Path

import httpx
import pytest

from clause.config import Settings
from clause.ingest.fetch import Fetcher, HashMismatchError, RateLimiter
from clause.ingest.validate import ValidationError
from clause.models import ManifestEntry

FIXTURES = Path(__file__).parent / "fixtures"
CIRCULAR = (FIXTURES / "rbi_13704.html").read_bytes()
BLOCK = (FIXTURES / "rbi_block_page.html").read_bytes()


def _entry(sha: str) -> ManifestEntry:
    return ManifestEntry(
        doc_id="rbi-13704",
        rbi_id=13704,
        url="https://example.test/doc",
        circular_no="RBI/2026-27/262",
        dept_ref="DOR.AML.REC.223/14.01.005/2026-27",
        title="t",
        published_date=date(2026, 9, 18),
        sha256=sha,
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url="postgresql+psycopg://x/y",
        user_agent="test",
        raw_cache_dir=tmp_path / "raw",
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetches_and_caches(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(200, content=CIRCULAR, headers={"content-type": "text/html"})

    s = _settings(tmp_path)
    f = Fetcher(s, _client(handler), RateLimiter(0.0))
    entry = _entry(hashlib.sha256(CIRCULAR).hexdigest())

    assert f.fetch(entry) == CIRCULAR
    assert (s.raw_cache_dir / "rbi-13704.html").exists()
    assert len(calls) == 1

    # second call is served from cache: no further network I/O
    assert f.fetch(entry) == CIRCULAR
    assert len(calls) == 1


def test_hash_mismatch_is_fatal(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=CIRCULAR, headers={"content-type": "text/html"})

    f = Fetcher(_settings(tmp_path), _client(handler), RateLimiter(0.0))
    with pytest.raises(HashMismatchError):
        f.fetch(_entry("b" * 64))


def test_block_page_is_never_cached(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=BLOCK, headers={"content-type": "text/html"})

    s = _settings(tmp_path)
    f = Fetcher(s, _client(handler), RateLimiter(0.0))
    with pytest.raises(ValidationError):
        f.fetch(_entry(hashlib.sha256(BLOCK).hexdigest()))
    assert not (s.raw_cache_dir / "rbi-13704.html").exists()


def test_non_html_response_is_rejected_and_not_cached(tmp_path: Path) -> None:
    """The gate must see the real content-type header, not a hardcoded one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=CIRCULAR, headers={"content-type": "application/pdf"})

    s = _settings(tmp_path)
    f = Fetcher(s, _client(handler), RateLimiter(0.0))
    with pytest.raises(ValidationError, match="content type"):
        f.fetch(_entry(hashlib.sha256(CIRCULAR).hexdigest()))
    assert not (s.raw_cache_dir / "rbi-13704.html").exists()


def test_retries_then_succeeds(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=CIRCULAR, headers={"content-type": "text/html"})

    s = _settings(tmp_path)
    f = Fetcher(s, _client(handler), RateLimiter(0.0), sleep=lambda _: None)
    assert f.fetch(_entry(hashlib.sha256(CIRCULAR).hexdigest())) == CIRCULAR
    assert attempts["n"] == 3


def test_rate_limiter_waits_between_calls() -> None:
    now = {"t": 0.0}
    slept: list[float] = []
    def _sleep(d: float) -> None:
        slept.append(d)
        now["t"] += d

    limiter = RateLimiter(2.0, clock=lambda: now["t"], sleep=_sleep)
    limiter.wait()   # first call does not wait
    limiter.wait()   # second must wait the full interval
    assert slept == [2.0]
