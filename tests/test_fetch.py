import hashlib
from datetime import date
from pathlib import Path

import httpx
import pytest

from clause.config import Settings
from clause.ingest.extract import canonical_text
from clause.ingest.fetch import Fetcher, HashMismatchError, RateLimiter
from clause.ingest.validate import ValidationError
from clause.models import ManifestEntry

FIXTURES = Path(__file__).parent / "fixtures"
CIRCULAR = (FIXTURES / "rbi_13704.html").read_bytes()
BLOCK = (FIXTURES / "rbi_block_page.html").read_bytes()

# The real header baked into the CIRCULAR fixture (see test_extract.py).
REAL_CIRCULAR_NO = "RBI/2026-27/262"
REAL_DEPT_REF = "DOR.AML.REC.223/14.01.005/2026-27"
REAL_PUBLISHED = date(2026, 9, 18)


def _content_sha256(raw: bytes) -> str:
    """The same hash `ManifestEntry.content_sha256` stores: sha256 of canonical_text."""
    text = canonical_text(raw.decode("utf-8", errors="replace"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entry(
    content_sha256: str,
    *,
    circular_no: str = REAL_CIRCULAR_NO,
    dept_ref: str = REAL_DEPT_REF,
    published_date: date = REAL_PUBLISHED,
) -> ManifestEntry:
    return ManifestEntry(
        doc_id="rbi-13704",
        rbi_id=13704,
        url="https://example.test/doc",
        circular_no=circular_no,
        dept_ref=dept_ref,
        title="t",
        published_date=published_date,
        content_sha256=content_sha256,
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
    entry = _entry(_content_sha256(CIRCULAR))

    assert f.fetch(entry) == CIRCULAR
    assert (s.raw_cache_dir / "rbi-13704.html").exists()
    assert len(calls) == 1

    # second call is served from cache: no further network I/O
    assert f.fetch(entry) == CIRCULAR
    assert len(calls) == 1


def test_chrome_only_mismatch_is_accepted_when_header_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A content_sha256 mismatch alone is not fatal when header identity still matches.

    This is the case discovered against the real site: RBI's WAF and volatile page
    furniture (a PDF-size widget) can shift `content_sha256` between fetches of an
    otherwise-unchanged document. As long as circular_no/dept_ref/published_date still
    parse to the manifest's recorded values, the document is accepted and a warning is
    logged rather than the fetch failing.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=CIRCULAR, headers={"content-type": "text/html"})

    s = _settings(tmp_path)
    f = Fetcher(s, _client(handler), RateLimiter(0.0))
    # Deliberately wrong content_sha256, but header fields (the defaults) match the
    # real content of CIRCULAR exactly.
    entry = _entry("b" * 64)

    assert f.fetch(entry) == CIRCULAR
    assert (s.raw_cache_dir / "rbi-13704.html").exists()

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert entry.doc_id in err


def test_header_identity_mismatch_is_fatal(tmp_path: Path) -> None:
    """A content_sha256 mismatch AND a header identity mismatch is a real revision."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=CIRCULAR, headers={"content-type": "text/html"})

    f = Fetcher(_settings(tmp_path), _client(handler), RateLimiter(0.0))
    entry = _entry("b" * 64, circular_no="RBI/2020-21/999")
    with pytest.raises(HashMismatchError):
        f.fetch(entry)


def _cached(tmp_path: Path, content: bytes) -> tuple[Settings, Path]:
    """Pre-populate the cache directly, bypassing fetch(), and return (settings, path).

    Used to test the cache-hit path in isolation, including cases the network
    would never actually put on disk (a corrupted file, a file that only ever
    got there via the tolerant branch) without needing a live fetch first.
    """
    s = _settings(tmp_path)
    s.raw_cache_dir.mkdir(parents=True, exist_ok=True)
    path = s.raw_cache_dir / "rbi-13704.html"
    path.write_bytes(content)
    return s, path


def _no_network_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError(
        "the cache-hit path must not make a network request: reaching this "
        "handler means path.exists() was not honoured"
    )


def test_cached_chrome_only_mismatch_is_accepted_when_header_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The cache-hit path applies the same tolerant check as a live fetch.

    A bug fixed in this round: an earlier version held the cache path to a
    *strict* standard, reasoning that cached bytes are "ours" and therefore
    free of WAF/page-chrome noise. That reasoning missed that the bytes on disk
    may themselves have been accepted through the tolerant branch when first
    fetched, in which case a strict re-check would reject a perfectly good
    document on every subsequent read, forever. This is the regression test
    for that trap.
    """
    s, _ = _cached(tmp_path, CIRCULAR)
    f = Fetcher(s, _client(_no_network_handler), RateLimiter(0.0))
    # Deliberately wrong content_sha256, but header fields (the defaults) match
    # the real content of CIRCULAR exactly.
    entry = _entry("b" * 64)

    assert f.fetch(entry) == CIRCULAR

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert entry.doc_id in err


def test_cached_header_identity_mismatch_is_fatal(tmp_path: Path) -> None:
    """A cached file whose header identity no longer matches is a real revision.

    Not a corruption-specific scenario: the manifest itself could be stale, or
    the file could have been swapped - either way, header identity diverging
    from the manifest is never page-chrome noise, cached or not.
    """
    s, _ = _cached(tmp_path, CIRCULAR)
    f = Fetcher(s, _client(_no_network_handler), RateLimiter(0.0))
    entry = _entry("b" * 64, circular_no="RBI/2020-21/999")

    with pytest.raises(HashMismatchError):
        f.fetch(entry)


def test_cached_file_failing_content_gate_is_rejected(tmp_path: Path) -> None:
    """A cached file that fails validate_response is rejected before any hash check.

    Covers gross corruption the hash comparison alone would not name usefully:
    a block page that somehow ended up on disk, well below min_document_chars,
    or missing a circular reference entirely. `content_sha256` is irrelevant
    here - the entry's value does not matter because the content gate fires
    first, exactly as it does on a live fetch.
    """
    s, _ = _cached(tmp_path, BLOCK)
    f = Fetcher(s, _client(_no_network_handler), RateLimiter(0.0))
    entry = _entry(_content_sha256(BLOCK))

    with pytest.raises(ValidationError):
        f.fetch(entry)


def test_block_page_is_never_cached(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=BLOCK, headers={"content-type": "text/html"})

    s = _settings(tmp_path)
    f = Fetcher(s, _client(handler), RateLimiter(0.0))
    with pytest.raises(ValidationError):
        f.fetch(_entry(_content_sha256(BLOCK)))
    assert not (s.raw_cache_dir / "rbi-13704.html").exists()


def test_non_html_response_is_rejected_and_not_cached(tmp_path: Path) -> None:
    """The gate must see the real content-type header, not a hardcoded one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=CIRCULAR, headers={"content-type": "application/pdf"})

    s = _settings(tmp_path)
    f = Fetcher(s, _client(handler), RateLimiter(0.0))
    with pytest.raises(ValidationError, match="content type"):
        f.fetch(_entry(_content_sha256(CIRCULAR)))
    assert not (s.raw_cache_dir / "rbi-13704.html").exists()


def test_retries_then_succeeds(tmp_path: Path) -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=CIRCULAR, headers={"content-type": "text/html"})

    s = _settings(tmp_path)
    f = Fetcher(s, _client(handler), RateLimiter(0.0), sleep=sleeps.append)
    assert f.fetch(_entry(_content_sha256(CIRCULAR))) == CIRCULAR
    assert attempts["n"] == 3
    assert len(sleeps) == 2, "one backoff sleep between each failed attempt and the next"
    assert all(d > 0 for d in sleeps), "backoff must actually wait"
    assert sleeps[1] > sleeps[0], "backoff must grow between attempts"


def test_rate_limiter_waits_between_calls() -> None:
    now = {"t": 0.0}
    slept: list[float] = []

    def _sleep(d: float) -> None:
        slept.append(d)
        now["t"] += d

    limiter = RateLimiter(2.0, clock=lambda: now["t"], sleep=_sleep)
    limiter.wait()  # first call does not wait
    limiter.wait()  # second must wait the full interval
    assert slept == [2.0]
