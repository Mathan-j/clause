# Phase 0 + Phase 1: Scaffold, Ingestion and Storage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A repository that lints, type-checks and goes green in CI, and a documented command that ingests 50+ real RBI KYC/AML documents into Postgres, chunked by two strategies, where every chunk's stored character span provably slices the stored source text.

**Architecture:** A committed manifest freezes the corpus. A rate-limited fetcher populates a hash-verified on-disk cache, refusing anything that fails a content-validation gate. Extraction produces one immutable canonical text per document; both chunkers are pure slicers over that string, so the citation round-trip holds by construction. Everything lands in Postgres via SQLAlchemy + Alembic.

**Tech Stack:** Python 3.12, uv, httpx, selectolax, SQLAlchemy 2.0, Alembic, psycopg 3, pydantic-settings, pytest, ruff, mypy, Docker Compose, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- Python `>=3.12,<3.13`. `.python-version` is already committed as `3.12`.
- **No number in documentation unless it came from a committed file under `reports/`.** The parameters in this plan are inputs, never results.
- **Every chunk stores** `(doc_id, char_start, char_end, source_url, effective_date, doc_type)`. The last three are denormalised onto the `chunks` table deliberately — see spec §5.4.
- **An unresolvable citation is a hard failure, never a warning.**
- Tests first for anything in the retrieval or citation path. That is every module here except `cli.py`.
- **No test may make a network request to `rbi.org.in`.** Fixtures are captured once, by hand, in Task 4.
- Every new dependency gets one line in `docs/decisions.md` saying what it replaced and why.
- Conventional commits. The body explains why, not what.
- Ask before spending money, pushing to a remote, or deploying.
- The Stop hook (`.claude/hooks/gate.sh`) runs ruff and pytest and refuses to let a turn end on a red build. It resolves tools from `.venv/`, which `uv sync` creates in Task 1.

**Initial parameters** (spec §5.3 — inputs, not measurements):

| Parameter | Value |
|---|---|
| `min_chunk_chars` | 400 |
| `max_chunk_chars` | 2000 |
| `window_chars` | 1200 |
| `overlap_chars` | 200 |
| `min_document_chars` | 500 |
| `min_request_interval_s` | 2.0 |
| `max_retries` | 3 |

---

## File Structure

```
pyproject.toml                  deps, ruff, mypy, pytest config
Makefile                        install lint test ingest up down
docker-compose.yml              postgres
alembic.ini                     migration config
.env.example
.github/workflows/ci.yml
docs/decisions.md               one line per dependency

src/clause/
  __init__.py                   __version__
  config.py                     Settings (pydantic-settings)
  models.py                     ManifestEntry, Document, Chunk — domain types
  htmltext.py                   visible_text(html) — shared by validate and extract
  sources/
    manifest.py                 load_manifest, write_manifest
    discover.py                 one-off discovery script (NOT part of make ingest)
  ingest/
    validate.py                 validate_response — the content gate
    fetch.py                    RateLimiter, Fetcher
    extract.py                  canonical_text, parse_header, extract_document
  chunking/
    base.py                     Chunker protocol, make_chunk — the invariant's enforcement point
    fixed.py                    FixedWindowChunker
    structural.py               StructuralChunker
  db/
    schema.py                   SQLAlchemy models
    session.py                  engine + session factory
    repository.py               upsert_document, replace_chunks
  cli.py                        `python -m clause.cli ingest`

migrations/versions/            alembic revisions
data/corpus/kyc.manifest.jsonl  committed, frozen corpus
data/raw/                       gitignored cache

tests/
  conftest.py
  fixtures/rbi_13704.html       real circular
  fixtures/rbi_block_page.html  captured bot-check page
  test_*.py
```

**Why these boundaries:** `htmltext.py` is separate because both the validation gate and the extractor need visible text, and validation must not depend on extraction (validation runs first, on untrusted input). `base.py` holds `make_chunk`, the single constructor every chunker must use — that is what makes the slice invariant structural rather than a convention each chunker is trusted to follow.

---

## Task 1: Phase 0 scaffold and green CI

**Files:**
- Create: `pyproject.toml`, `Makefile`, `src/clause/__init__.py`, `tests/test_smoke.py`, `.github/workflows/ci.yml`, `docs/decisions.md`, `.env.example`

**Interfaces:**
- Consumes: nothing
- Produces: `clause.__version__: str`; a working `uv sync`, `make lint`, `make test`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "clause"
version = "0.1.0"
description = "Regulatory answer engine with span-level citations over RBI circulars"
requires-python = ">=3.12,<3.13"
dependencies = []

[dependency-groups]
dev = ["ruff>=0.6", "mypy>=1.11", "pytest>=8.3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/clause"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "PL", "RUF"]

[tool.mypy]
python_version = "3.12"
strict = true
files = ["src/clause"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers -q"
markers = ["db: requires a live Postgres instance"]
```

- [ ] **Step 2: Write `src/clause/__init__.py` and the smoke test**

```python
# src/clause/__init__.py
__version__ = "0.1.0"
```

```python
# tests/test_smoke.py
from clause import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 3: Write the `Makefile`**

Recipe lines must be indented with a literal TAB, not spaces.

```make
.PHONY: install lint test ingest eval serve up down

install:
	uv sync

lint:
	uv run ruff check .
	uv run mypy

test:
	uv run pytest

ingest:
	uv run python -m clause.cli ingest --manifest data/corpus/kyc.manifest.jsonl

up:
	docker compose up -d

down:
	docker compose down
```

- [ ] **Step 4: Write `.env.example` and `docs/decisions.md`**

```bash
# .env.example
CLAUSE_DATABASE_URL=postgresql+psycopg://clause:clause@localhost:5432/clause
CLAUSE_USER_AGENT=clause-research/0.1 (+mailto:you@example.com)
```

```markdown
# Decisions

One line per dependency: what it is, what it replaced, why.

- ruff — lint + format. Replaces flake8 + isort + black; one tool, one config, far faster.
- mypy — static types. Chosen over pyright to keep the toolchain inside the Python package set.
- pytest — test runner. Replaces unittest; fixtures and parametrisation are worth the dependency.
```

- [ ] **Step 5: Run lint and tests locally**

```bash
uv sync
make lint
make test
```
Expected: ruff clean, mypy clean, 1 test passed.

- [ ] **Step 6: Write the CI workflow**

```yaml
# .github/workflows/ci.yml
name: ci

on:
  push:
  pull_request:

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run mypy
      - run: uv run pytest
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml Makefile uv.lock src/clause/__init__.py tests/test_smoke.py \
        .github/workflows/ci.yml docs/decisions.md .env.example
git commit -m "feat: scaffold python package with lint, types, tests and CI

Phase 0 adds no runtime dependencies. uv sync creates .venv, which is
where the Stop-hook gate already resolves ruff and pytest from, so the
gate becomes active with this commit."
```

**Done when:** CI is green on push.

---

## Task 2: Configuration

**Files:**
- Create: `src/clause/config.py`, `tests/test_config.py`
- Modify: `pyproject.toml` (add `pydantic-settings`), `docs/decisions.md`

**Interfaces:**
- Consumes: nothing
- Produces: `clause.config.Settings` with fields `database_url: str`, `user_agent: str`, `raw_cache_dir: Path`, `min_request_interval_s: float`, `max_retries: int`, `min_document_chars: int`, `min_chunk_chars: int`, `max_chunk_chars: int`, `window_chars: int`, `overlap_chars: int`; and `get_settings() -> Settings`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

from clause.config import Settings


def test_defaults_match_the_spec() -> None:
    s = Settings(database_url="postgresql+psycopg://x/y", user_agent="t")
    assert s.min_chunk_chars == 400
    assert s.max_chunk_chars == 2000
    assert s.window_chars == 1200
    assert s.overlap_chars == 200
    assert s.min_document_chars == 500
    assert s.min_request_interval_s == 2.0
    assert s.max_retries == 3
    assert s.raw_cache_dir == Path("data/raw")


def test_environment_overrides_defaults(monkeypatch) -> None:
    monkeypatch.setenv("CLAUSE_WINDOW_CHARS", "999")
    monkeypatch.setenv("CLAUSE_DATABASE_URL", "postgresql+psycopg://x/y")
    monkeypatch.setenv("CLAUSE_USER_AGENT", "t")
    assert Settings().window_chars == 999


def test_overlap_must_be_smaller_than_window() -> None:
    import pytest

    with pytest.raises(ValueError):
        Settings(
            database_url="postgresql+psycopg://x/y",
            user_agent="t",
            window_chars=100,
            overlap_chars=100,
        )
```

- [ ] **Step 2: Run it and watch it fail**

```bash
uv run pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.config'`

- [ ] **Step 3: Add the dependency**

```bash
uv add pydantic-settings
```
Append to `docs/decisions.md`:
```markdown
- pydantic-settings — environment-driven config with validation. Replaces hand-rolled os.environ reads; we get type coercion and failure at startup rather than at first use.
```

- [ ] **Step 4: Implement**

```python
# src/clause/config.py
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every value here is an input, never a measurement."""

    model_config = SettingsConfigDict(env_prefix="CLAUSE_", env_file=".env", extra="ignore")

    database_url: str
    user_agent: str

    raw_cache_dir: Path = Path("data/raw")

    # Fetcher
    min_request_interval_s: float = Field(default=2.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

    # Validation gate
    min_document_chars: int = Field(default=500, gt=0)

    # Chunking
    min_chunk_chars: int = Field(default=400, gt=0)
    max_chunk_chars: int = Field(default=2000, gt=0)
    window_chars: int = Field(default=1200, gt=0)
    overlap_chars: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def _check_sizes(self) -> "Settings":
        if self.overlap_chars >= self.window_chars:
            raise ValueError("overlap_chars must be smaller than window_chars")
        if self.min_chunk_chars >= self.max_chunk_chars:
            raise ValueError("min_chunk_chars must be smaller than max_chunk_chars")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_config.py -v && make lint
```
Expected: 3 passed, lint clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/clause/config.py tests/test_config.py docs/decisions.md
git commit -m "feat: environment-driven configuration

Every chunking and fetching parameter is configurable because each one
is a guess that Phase 2 will correct. Overlap-versus-window is validated
at construction so a nonsensical pair fails at startup, not silently at
the first document."
```

---

## Task 3: Domain types and manifest I/O

**Files:**
- Create: `src/clause/models.py`, `src/clause/sources/__init__.py`, `src/clause/sources/manifest.py`, `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ManifestEntry(doc_id: str, rbi_id: int, url: str, circular_no: str, dept_ref: str, title: str, published_date: date, sha256: str)` — frozen dataclass
  - `Document(doc_id, rbi_id, url, circular_no, dept_ref, title, doc_type: str, published_date: date, effective_date: date | None, sha256: str, fetched_at: datetime, text: str)` — frozen dataclass
  - `Chunk(doc_id: str, strategy: str, ordinal: int, char_start: int, char_end: int, text: str, source_url: str, effective_date: date | None, doc_type: str)` — frozen dataclass
  - `load_manifest(path: Path) -> list[ManifestEntry]`
  - `write_manifest(path: Path, entries: Iterable[ManifestEntry]) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
from datetime import date
from pathlib import Path

import pytest

from clause.models import ManifestEntry
from clause.sources.manifest import load_manifest, write_manifest

ENTRY = ManifestEntry(
    doc_id="rbi-13704",
    rbi_id=13704,
    url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13704&Mode=0",
    circular_no="RBI/2026-27/262",
    dept_ref="DOR.AML.REC.223/14.01.005/2026-27",
    title="Reserve Bank of India (Rural Co-operative Banks - Know Your Customer) "
    "Amendment Directions, 2026",
    published_date=date(2026, 9, 18),
    sha256="a" * 64,
)


def test_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    write_manifest(p, [ENTRY])
    assert load_manifest(p) == [ENTRY]


def test_duplicate_doc_ids_are_rejected(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    write_manifest(p, [ENTRY, ENTRY])
    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_manifest(p)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    write_manifest(p, [ENTRY])
    p.write_text(p.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(load_manifest(p)) == 1
```

- [ ] **Step 2: Run it and watch it fail**

```bash
uv run pytest tests/test_manifest.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.models'`

- [ ] **Step 3: Implement the domain types**

```python
# src/clause/models.py
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
```

- [ ] **Step 4: Implement manifest I/O**

```python
# src/clause/sources/manifest.py
import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from clause.models import ManifestEntry


def write_manifest(path: Path, entries: Iterable[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for e in entries:
            row = {
                "doc_id": e.doc_id,
                "rbi_id": e.rbi_id,
                "url": e.url,
                "circular_no": e.circular_no,
                "dept_ref": e.dept_ref,
                "title": e.title,
                "published_date": e.published_date.isoformat(),
                "sha256": e.sha256,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        entry = ManifestEntry(
            doc_id=row["doc_id"],
            rbi_id=int(row["rbi_id"]),
            url=row["url"],
            circular_no=row["circular_no"],
            dept_ref=row["dept_ref"],
            title=row["title"],
            published_date=date.fromisoformat(row["published_date"]),
            sha256=row["sha256"],
        )
        if entry.doc_id in seen:
            raise ValueError(f"duplicate doc_id {entry.doc_id!r} at line {lineno}")
        seen.add(entry.doc_id)
        entries.append(entry)
    return entries
```

Create an empty `src/clause/sources/__init__.py`.

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_manifest.py -v && make lint
```
Expected: 3 passed, lint clean.

- [ ] **Step 6: Commit**

```bash
git add src/clause/models.py src/clause/sources/ tests/test_manifest.py
git commit -m "feat: domain types and manifest I/O

doc_id uniqueness is enforced on load rather than trusted, because the
manifest is the mechanism that freezes the corpus and a duplicate would
silently drop a document from the ingest."
```

---

## Task 4: HTML text extraction and the content-validation gate

This is the control that matters most (spec §7.1). A blocked request returns HTTP 200 with an HTML body, so the status code cannot be trusted.

**Files:**
- Create: `src/clause/htmltext.py`, `src/clause/ingest/__init__.py`, `src/clause/ingest/validate.py`, `tests/test_validate.py`, `tests/fixtures/rbi_13704.html`, `tests/fixtures/rbi_block_page.html`
- Modify: `pyproject.toml`, `docs/decisions.md`

**Interfaces:**
- Consumes: `clause.config.Settings`
- Produces:
  - `visible_text(html: str) -> str` — script/style stripped, entities decoded, whitespace runs collapsed
  - `ValidationError(Exception)`
  - `validate_response(*, status_code: int, content_type: str, body: str, min_chars: int) -> None` — raises `ValidationError`, returns `None` on success

- [ ] **Step 1: Capture the fixtures (one-off, requires network)**

This is the only step in the entire plan that touches `rbi.org.in`. Run it once by hand; no test ever repeats it.

```bash
mkdir -p tests/fixtures
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"

# A real circular
curl -s -A "$UA" \
  "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13704&Mode=0" \
  -o tests/fixtures/rbi_13704.html

# The bot-check page. robots.txt reliably returns it.
curl -s -A "$UA" "https://www.rbi.org.in/robots.txt" \
  -o tests/fixtures/rbi_block_page.html

grep -c "RBI/" tests/fixtures/rbi_13704.html        # expect >= 1
grep -c "Unauthorised Access" tests/fixtures/rbi_block_page.html  # expect 1
```

If the block page is not returned (RBI may change behaviour), fail loudly rather than inventing a fixture: stop and report it, because `test_block_page_is_rejected` is the test that protects the corpus from being silently worthless.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_validate.py
from pathlib import Path

import pytest

from clause.htmltext import visible_text
from clause.ingest.validate import ValidationError, validate_response

FIXTURES = Path(__file__).parent / "fixtures"
CIRCULAR = (FIXTURES / "rbi_13704.html").read_text(encoding="utf-8", errors="replace")
BLOCK = (FIXTURES / "rbi_block_page.html").read_text(encoding="utf-8", errors="replace")


def _ok(body: str) -> None:
    validate_response(status_code=200, content_type="text/html", body=body, min_chars=500)


def test_real_circular_passes() -> None:
    _ok(CIRCULAR)


def test_block_page_is_rejected_despite_http_200() -> None:
    with pytest.raises(ValidationError, match="bot-check"):
        _ok(BLOCK)


def test_non_200_is_rejected() -> None:
    with pytest.raises(ValidationError, match="status"):
        validate_response(status_code=503, content_type="text/html", body=CIRCULAR, min_chars=500)


def test_non_html_is_rejected() -> None:
    with pytest.raises(ValidationError, match="content type"):
        validate_response(
            status_code=200, content_type="application/pdf", body=CIRCULAR, min_chars=500
        )


def test_html_without_a_circular_reference_is_rejected() -> None:
    with pytest.raises(ValidationError, match="circular reference"):
        _ok("<html><body>" + "padding text " * 200 + "</body></html>")


def test_short_document_is_rejected() -> None:
    with pytest.raises(ValidationError, match="too short"):
        validate_response(
            status_code=200,
            content_type="text/html",
            body="<html><body>RBI/2026-27/262 short</body></html>",
            min_chars=500,
        )


def test_visible_text_strips_scripts_and_styles() -> None:
    html = "<html><head><style>p{color:red}</style></head><body><script>x=1</script><p>Hello</p></body></html>"
    out = visible_text(html)
    assert "Hello" in out
    assert "color" not in out
    assert "x=1" not in out


def test_visible_text_decodes_entities_and_collapses_whitespace() -> None:
    assert visible_text("<p>a &amp;   b\n\n\nc</p>") == "a & b c"
```

- [ ] **Step 3: Run and watch them fail**

```bash
uv run pytest tests/test_validate.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.htmltext'`

- [ ] **Step 4: Add the dependency**

```bash
uv add selectolax
```
Append to `docs/decisions.md`:
```markdown
- selectolax — HTML parsing. Replaces BeautifulSoup + lxml; a C parser with a direct text() path, and we only need text extraction, not a tree API.
```

- [ ] **Step 5: Implement `visible_text`**

```python
# src/clause/htmltext.py
import re

from selectolax.parser import HTMLParser

_WS = re.compile(r"\s+")


def visible_text(html: str) -> str:
    """Visible text with scripts, styles and navigation removed.

    Entities are decoded and whitespace runs collapsed to single spaces. Used by
    both the validation gate and the extractor so they agree on what 'the text'
    means.
    """
    tree = HTMLParser(html)
    for tag in ("script", "style", "noscript"):
        for node in tree.css(tag):
            node.decompose()
    body = tree.body or tree.root
    if body is None:
        return ""
    return _WS.sub(" ", body.text(separator=" ")).strip()
```

- [ ] **Step 6: Implement the gate**

```python
# src/clause/ingest/validate.py
import re

from clause.htmltext import visible_text

BLOCK_MARKERS: tuple[str, ...] = ("Unauthorised Access", "Support ID:")
CIRCULAR_REFERENCE = re.compile(r"RBI/\d{4}-\d{2}/\d+")


class ValidationError(Exception):
    """A response that must never become a cached document."""


def validate_response(*, status_code: int, content_type: str, body: str, min_chars: int) -> None:
    """Raise unless this response is plausibly a real RBI circular.

    The offset round-trip proves fidelity, not correctness: it would pass just
    as happily on fifty identical bot-check pages. Because a blocked request
    returns HTTP 200 with an HTML body, the status code cannot stand in for
    this check.
    """
    if status_code != 200:
        raise ValidationError(f"bad status: {status_code}")

    if "html" not in content_type.lower():
        raise ValidationError(f"unexpected content type: {content_type!r}")

    for marker in BLOCK_MARKERS:
        if marker in body:
            raise ValidationError(f"bot-check page detected (marker {marker!r})")

    if not CIRCULAR_REFERENCE.search(body):
        raise ValidationError("no RBI circular reference found in body")

    text = visible_text(body)
    if len(text) < min_chars:
        raise ValidationError(f"document too short: {len(text)} < {min_chars} chars")
```

Create an empty `src/clause/ingest/__init__.py`.

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/test_validate.py -v && make lint
```
Expected: 8 passed, lint clean.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/clause/htmltext.py src/clause/ingest/ \
        tests/test_validate.py tests/fixtures/
git commit -m "feat: content validation gate rejects bot-check pages

RBI returns HTTP 200 with an HTML body when it blocks a request, so the
status code cannot be trusted. Without this gate the cache would fill
with identical block pages, every offset assertion would still pass, and
the corpus would be silently worthless -- the round-trip invariant proves
fidelity, not correctness."
```

---

## Task 5: Rate-limited fetcher with hash-verified cache

**Files:**
- Create: `src/clause/ingest/fetch.py`, `tests/test_fetch.py`
- Modify: `pyproject.toml`, `docs/decisions.md`

**Interfaces:**
- Consumes: `ManifestEntry`, `validate_response`, `Settings`
- Produces:
  - `RateLimiter(min_interval_s: float, *, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep)` with `.wait() -> None`
  - `FetchError(Exception)`, `HashMismatchError(FetchError)`
  - `Fetcher(settings: Settings, client: httpx.Client, limiter: RateLimiter)` with `.fetch(entry: ManifestEntry) -> bytes` — returns raw bytes, writes `settings.raw_cache_dir / f"{entry.doc_id}.html"`, performs zero network I/O on a cache hit

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fetch.py
from datetime import date
from pathlib import Path

import httpx
import pytest

from clause.config import Settings
from clause.ingest.fetch import Fetcher, HashMismatchError, RateLimiter
from clause.models import ManifestEntry
from clause.ingest.validate import ValidationError

FIXTURES = Path(__file__).parent / "fixtures"
CIRCULAR = (FIXTURES / "rbi_13704.html").read_bytes()
BLOCK = (FIXTURES / "rbi_block_page.html").read_bytes()

import hashlib


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
    limiter = RateLimiter(
        2.0, clock=lambda: now["t"], sleep=lambda d: (slept.append(d), now.__setitem__("t", now["t"] + d))
    )
    limiter.wait()   # first call does not wait
    limiter.wait()   # second must wait the full interval
    assert slept == [2.0]
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_fetch.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.ingest.fetch'`

- [ ] **Step 3: Add the dependency**

```bash
uv add httpx
```
Append to `docs/decisions.md`:
```markdown
- httpx — HTTP client. Replaces requests; native timeouts, and MockTransport lets every fetcher test run without a socket.
```

- [ ] **Step 4: Implement**

```python
# src/clause/ingest/fetch.py
import hashlib
import random
import time
from collections.abc import Callable

import httpx

from clause.config import Settings
from clause.ingest.validate import validate_response
from clause.models import ManifestEntry


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

    def _cache_path(self, entry: ManifestEntry):
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
                if response.status_code == 200:
                    return response
                last = FetchError(f"{entry.doc_id}: HTTP {response.status_code}")

            if attempt < self._s.max_retries:
                backoff = (2.0**attempt) + random.uniform(0, 0.5)
                self._sleep(backoff)

        raise FetchError(f"{entry.doc_id}: giving up after {self._s.max_retries + 1} attempts: {last}")
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_fetch.py -v && make lint
```
Expected: 5 passed, lint clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/clause/ingest/fetch.py tests/test_fetch.py docs/decisions.md
git commit -m "feat: rate-limited fetcher with hash-verified cache

A hash mismatch is fatal and never auto-healed: RBI revises circulars in
place, and silently accepting new bytes under an old manifest entry would
let the corpus drift while appearing frozen.

robots.txt is unreachable (418), so the rate limit is a conservative
fixed floor rather than a claim of compliance."
```

---

## Task 6: Extraction to canonical text

**Files:**
- Create: `src/clause/ingest/extract.py`, `tests/test_extract.py`

**Interfaces:**
- Consumes: `visible_text`, `ManifestEntry`, `Document`
- Produces:
  - `canonical_text(html: str) -> str`
  - `parse_header(text: str) -> tuple[str, str, date]` — returns `(circular_no, dept_ref, published_date)`, raises `ExtractionError`
  - `ExtractionError(Exception)`
  - `extract_document(entry: ManifestEntry, raw: bytes, *, fetched_at: datetime) -> Document`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_extract.py
import unicodedata
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from clause.ingest.extract import ExtractionError, canonical_text, extract_document, parse_header
from clause.models import ManifestEntry

FIXTURES = Path(__file__).parent / "fixtures"
RAW = (FIXTURES / "rbi_13704.html").read_bytes()

ENTRY = ManifestEntry(
    doc_id="rbi-13704",
    rbi_id=13704,
    url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13704&Mode=0",
    circular_no="RBI/2026-27/262",
    dept_ref="DOR.AML.REC.223/14.01.005/2026-27",
    title="t",
    published_date=date(2026, 9, 18),
    sha256="a" * 64,
)


def test_canonical_text_is_nfc_normalised() -> None:
    out = canonical_text("<p>café</p>")
    assert out == unicodedata.normalize("NFC", out)
    assert out == "café"


def test_canonical_text_collapses_whitespace_and_nbsp() -> None:
    assert canonical_text("<p>a  b\t\tc</p>") == "a b c"


def test_canonical_text_is_idempotent() -> None:
    once = canonical_text(RAW.decode("utf-8", errors="replace"))
    assert canonical_text(f"<p>{once}</p>") == once


def test_parse_header_reads_the_real_document() -> None:
    circular_no, dept_ref, published = parse_header(canonical_text(RAW.decode("utf-8", "replace")))
    assert circular_no == "RBI/2026-27/262"
    assert dept_ref.startswith("DOR.AML.REC.223")
    assert published == date(2026, 9, 18)


def test_parse_header_raises_when_absent() -> None:
    with pytest.raises(ExtractionError):
        parse_header("no header here at all")


def test_extract_document_populates_the_document() -> None:
    doc = extract_document(ENTRY, RAW, fetched_at=datetime(2026, 9, 19, tzinfo=UTC))
    assert doc.doc_id == "rbi-13704"
    assert doc.circular_no == "RBI/2026-27/262"
    assert doc.doc_type in {"master_direction", "circular", "notification"}
    assert len(doc.text) > 500
    assert doc.text == canonical_text(RAW.decode("utf-8", errors="replace"))


def test_extract_document_rejects_short_text() -> None:
    with pytest.raises(ExtractionError, match="too short"):
        extract_document(
            ENTRY, b"<html><body>RBI/2026-27/262 tiny</body></html>",
            fetched_at=datetime(2026, 9, 19, tzinfo=UTC),
        )
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_extract.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.ingest.extract'`

- [ ] **Step 3: Implement**

```python
# src/clause/ingest/extract.py
import re
import unicodedata
from datetime import date, datetime

from clause.htmltext import visible_text
from clause.models import Document, ManifestEntry

MIN_TEXT_CHARS = 500

HEADER = re.compile(
    r"(?P<circular_no>RBI/\d{4}-\d{2}/\d+)\s+"
    r"(?P<dept_ref>[A-Z]{2,}(?:\.[A-Z0-9]+)+[A-Z0-9./-]*)\s+"
    r"(?P<published>[A-Z][a-z]+ \d{1,2}, \d{4})"
)

_WS = re.compile(r"[ \t ]+")


class ExtractionError(Exception):
    """The response parsed, but is not a usable document."""


def canonical_text(html: str) -> str:
    """Produce the one immutable string every character offset indexes into.

    Called exactly once per document. Nothing downstream re-normalises: both
    chunkers slice this string and never transform it, which is what makes the
    citation round-trip hold by construction.
    """
    text = visible_text(html)
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    text = _WS.sub(" ", text)
    return text.strip()


def parse_header(text: str) -> tuple[str, str, date]:
    match = HEADER.search(text)
    if match is None:
        raise ExtractionError("no RBI header line (circular no / dept ref / date) found")
    published = datetime.strptime(match.group("published"), "%B %d, %Y").date()
    return match.group("circular_no"), match.group("dept_ref"), published


def _classify(title: str, text: str) -> str:
    haystack = f"{title} {text[:400]}".lower()
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

    return Document(
        doc_id=entry.doc_id,
        rbi_id=entry.rbi_id,
        url=entry.url,
        circular_no=circular_no,
        dept_ref=dept_ref,
        title=entry.title,
        doc_type=_classify(entry.title, text),
        published_date=published,
        effective_date=None,
        sha256=entry.sha256,
        fetched_at=fetched_at,
        text=text,
    )
```

`effective_date` is `None` at Phase 1. RBI states effective dates in prose, inconsistently; parsing them is Phase 3 work and inventing a value now would put an unmeasured claim in the citation tuple. The column exists and is nullable.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_extract.py -v && make lint
```
Expected: 7 passed, lint clean.

- [ ] **Step 5: Commit**

```bash
git add src/clause/ingest/extract.py tests/test_extract.py
git commit -m "feat: extraction produces one immutable canonical text

Normalisation happens exactly once, here. Every character offset in the
system indexes into this string and nothing downstream re-normalises --
a second normalisation pass anywhere would silently shift every stored
span off its source.

effective_date stays null: RBI states it in prose, and guessing it would
put an unmeasured value inside the citation tuple."
```

---

## Task 7: Chunker protocol and the fixed-window baseline

**Files:**
- Create: `src/clause/chunking/__init__.py`, `src/clause/chunking/base.py`, `src/clause/chunking/fixed.py`, `tests/test_chunking.py`

**Interfaces:**
- Consumes: `Document`, `Chunk`, `Settings`
- Produces:
  - `Chunker` Protocol: attribute `name: str`, method `chunk(self, document: Document) -> list[Chunk]`
  - `make_chunk(document: Document, strategy: str, ordinal: int, start: int, end: int) -> Chunk` — the only sanctioned way to build a `Chunk`
  - `assert_slices(document: Document, chunks: list[Chunk]) -> None`
  - `FixedWindowChunker(window_chars: int, overlap_chars: int)` with `name = "fixed_window"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunking.py
from datetime import UTC, date, datetime

import pytest

from clause.chunking.base import assert_slices, make_chunk
from clause.chunking.fixed import FixedWindowChunker
from clause.models import Document

ADVERSARIAL = [
    "short",
    "a " * 5000,
    "para with nbsp. " * 300,
    "कुछ हिन्दी पाठ. " * 200,
    "1. First. 2. Second. 3.1 Nested. (a) Lettered. " * 100,
]


def _doc(text: str) -> Document:
    return Document(
        doc_id="d1", rbi_id=1, url="https://example.test/d1",
        circular_no="RBI/2026-27/1", dept_ref="DOR.X.1", title="t",
        doc_type="circular", published_date=date(2026, 9, 18), effective_date=None,
        sha256="a" * 64, fetched_at=datetime(2026, 9, 19, tzinfo=UTC), text=text,
    )


@pytest.mark.parametrize("text", ADVERSARIAL)
def test_fixed_window_chunks_are_pure_slices(text: str) -> None:
    doc = _doc(text)
    chunks = FixedWindowChunker(window_chars=200, overlap_chars=50).chunk(doc)
    assert_slices(doc, chunks)


@pytest.mark.parametrize("text", ADVERSARIAL)
def test_fixed_window_covers_every_character(text: str) -> None:
    doc = _doc(text)
    chunks = FixedWindowChunker(window_chars=200, overlap_chars=50).chunk(doc)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(doc.text)
    for prev, nxt in zip(chunks, chunks[1:], strict=True):
        assert nxt.char_start <= prev.char_end, "gap between chunks would drop characters"


def test_chunks_carry_their_own_provenance() -> None:
    doc = _doc("x" * 1000)
    chunk = FixedWindowChunker(window_chars=200, overlap_chars=50).chunk(doc)[0]
    assert chunk.source_url == doc.url
    assert chunk.doc_type == doc.doc_type
    assert chunk.effective_date == doc.effective_date


def test_ordinals_are_dense_and_ascending() -> None:
    doc = _doc("y" * 1000)
    chunks = FixedWindowChunker(window_chars=200, overlap_chars=50).chunk(doc)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_assert_slices_catches_a_transformed_chunk() -> None:
    import dataclasses

    doc = _doc("hello world")
    good = make_chunk(doc, "fixed_window", 0, 0, 5)
    # dataclasses.replace, not __dict__: Chunk uses slots=True and has no __dict__.
    tampered = dataclasses.replace(good, text="HELLO")
    with pytest.raises(AssertionError):
        assert_slices(doc, [tampered])


def test_empty_document_yields_no_chunks() -> None:
    doc = _doc("")
    assert FixedWindowChunker(window_chars=200, overlap_chars=50).chunk(doc) == []
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_chunking.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.chunking'`

- [ ] **Step 3: Implement the protocol and the invariant's enforcement point**

```python
# src/clause/chunking/base.py
from typing import Protocol, runtime_checkable

from clause.models import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, document: Document) -> list[Chunk]: ...


def make_chunk(document: Document, strategy: str, ordinal: int, start: int, end: int) -> Chunk:
    """The only sanctioned way to build a Chunk.

    text is sliced from document.text here rather than passed in, so a chunker
    physically cannot emit a chunk whose text differs from its span. That is
    what makes the citation round-trip a structural property rather than a
    convention each chunker is trusted to follow.
    """
    if not 0 <= start < end <= len(document.text):
        raise ValueError(f"span ({start}, {end}) outside document of {len(document.text)} chars")
    return Chunk(
        doc_id=document.doc_id,
        strategy=strategy,
        ordinal=ordinal,
        char_start=start,
        char_end=end,
        text=document.text[start:end],
        source_url=document.url,
        effective_date=document.effective_date,
        doc_type=document.doc_type,
    )


def assert_slices(document: Document, chunks: list[Chunk]) -> None:
    """Assert the invariant. Used by tests and by the ingest CLI before writing."""
    for c in chunks:
        actual = document.text[c.char_start : c.char_end]
        assert actual == c.text, (
            f"{c.doc_id}#{c.ordinal} ({c.strategy}): span [{c.char_start}:{c.char_end}] "
            f"yields {actual[:60]!r} but chunk stores {c.text[:60]!r}"
        )
```

- [ ] **Step 4: Implement the baseline chunker**

```python
# src/clause/chunking/fixed.py
from clause.chunking.base import make_chunk
from clause.models import Chunk, Document


class FixedWindowChunker:
    """Overlapping fixed-size windows, snapped to word boundaries.

    This is the baseline. It exists so the structural chunker's Phase 2 numbers
    mean something: a structure-aware strategy with nothing to beat is an
    unfalsifiable claim.
    """

    name = "fixed_window"

    def __init__(self, window_chars: int, overlap_chars: int) -> None:
        if overlap_chars >= window_chars:
            raise ValueError("overlap_chars must be smaller than window_chars")
        self._window = window_chars
        self._overlap = overlap_chars

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        ordinal = 0
        stride = self._window - self._overlap

        while start < len(text):
            end = min(start + self._window, len(text))
            if end < len(text):
                space = text.rfind(" ", start + stride, end)
                if space > start:
                    end = space
            chunks.append(make_chunk(document, self.name, ordinal, start, end))
            ordinal += 1
            if end >= len(text):
                break
            start = max(end - self._overlap, start + 1)

        return chunks
```

Create an empty `src/clause/chunking/__init__.py`.

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_chunking.py -v && make lint
```
Expected: 16 passed (5 parametrised × 2 + 6), lint clean.

- [ ] **Step 6: Commit**

```bash
git add src/clause/chunking/ tests/test_chunking.py
git commit -m "feat: chunker protocol and fixed-window baseline

make_chunk slices text from the document rather than accepting it, so a
chunker cannot emit a chunk whose stored text differs from its span. The
round-trip test is then an alarm on a property the design already
guarantees, rather than the only thing standing between us and broken
citations."
```

---

## Task 8: Structural chunker

**Files:**
- Create: `src/clause/chunking/structural.py`, `tests/test_chunking_structural.py`

**Interfaces:**
- Consumes: `make_chunk`, `assert_slices`, `Document`, `Chunk`
- Produces: `StructuralChunker(min_chunk_chars: int, max_chunk_chars: int)` with `name = "structural"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunking_structural.py
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from clause.chunking.base import assert_slices
from clause.chunking.structural import StructuralChunker
from clause.ingest.extract import canonical_text
from clause.models import Document

FIXTURES = Path(__file__).parent / "fixtures"
REAL_TEXT = canonical_text((FIXTURES / "rbi_13704.html").read_text(encoding="utf-8", errors="replace"))


def _doc(text: str) -> Document:
    return Document(
        doc_id="d1", rbi_id=1, url="https://example.test/d1",
        circular_no="RBI/2026-27/1", dept_ref="DOR.X.1", title="t",
        doc_type="circular", published_date=date(2026, 9, 18), effective_date=None,
        sha256="a" * 64, fetched_at=datetime(2026, 9, 19, tzinfo=UTC), text=text,
    )


CHUNKER = StructuralChunker(min_chunk_chars=50, max_chunk_chars=300)


def test_chunks_are_pure_slices_of_the_real_document() -> None:
    doc = _doc(REAL_TEXT)
    assert_slices(doc, CHUNKER.chunk(doc))


def test_splits_on_rbi_numbering() -> None:
    text = (
        "Preamble text that is long enough to stand alone as its own chunk here. "
        "1. First numbered paragraph with sufficient length to survive merging rules. "
        "2. Second numbered paragraph also long enough to be its own separate chunk. "
    )
    chunks = CHUNKER.chunk(_doc(text))
    assert len(chunks) >= 2
    assert any(c.text.lstrip().startswith("1.") for c in chunks)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "tiny",
        "x" * 5000,
        "1. " + ("word " * 2000),
        "(a) alpha. (b) beta. (c) gamma. " * 80,
    ],
)
def test_invariant_holds_on_adversarial_input(text: str) -> None:
    doc = _doc(text)
    assert_slices(doc, CHUNKER.chunk(doc))


def test_no_chunk_exceeds_the_maximum_by_more_than_one_sentence() -> None:
    doc = _doc("Sentence here. " * 400)
    for c in CHUNKER.chunk(doc):
        assert len(c.text) <= 300 * 2


def test_covers_every_character() -> None:
    doc = _doc(REAL_TEXT)
    chunks = CHUNKER.chunk(doc)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(doc.text)
    for prev, nxt in zip(chunks, chunks[1:], strict=True):
        assert nxt.char_start == prev.char_end


def test_strategy_name_is_recorded() -> None:
    doc = _doc(REAL_TEXT)
    assert {c.strategy for c in CHUNKER.chunk(doc)} == {"structural"}
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_chunking_structural.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.chunking.structural'`

- [ ] **Step 3: Implement**

```python
# src/clause/chunking/structural.py
import re

from clause.chunking.base import make_chunk
from clause.models import Chunk, Document

# RBI numbering: "1.", "4.2", "4(1)(v)", "(a)", and annex headings.
BOUNDARY = re.compile(
    r"(?<=[.\s])(?=(?:\d+(?:\.\d+)*\.\s)|(?:\d+\(\d+\)(?:\([a-z]+\))?\s)|(?:\([a-z]\)\s)|(?:Annex\b))"
)
SENTENCE_END = re.compile(r"(?<=[.;:])\s")


class StructuralChunker:
    """Split on RBI's own numbering, then repair sizes.

    Boundaries come from the document's structure rather than a character
    count, so a chunk tends to be a complete regulatory provision -- which is
    what a citation should point at.
    """

    name = "structural"

    def __init__(self, min_chunk_chars: int, max_chunk_chars: int) -> None:
        if min_chunk_chars >= max_chunk_chars:
            raise ValueError("min_chunk_chars must be smaller than max_chunk_chars")
        self._min = min_chunk_chars
        self._max = max_chunk_chars

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        spans = self._split_on_structure(text)
        spans = self._merge_small(spans)
        spans = self._split_large(text, spans)

        return [
            make_chunk(document, self.name, ordinal, start, end)
            for ordinal, (start, end) in enumerate(spans)
        ]

    def _split_on_structure(self, text: str) -> list[tuple[int, int]]:
        cuts = [0, *(m.start() for m in BOUNDARY.finditer(text)), len(text)]
        cuts = sorted(set(cuts))
        return [(a, b) for a, b in zip(cuts, cuts[1:], strict=True) if b > a]

    def _merge_small(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if merged and (merged[-1][1] - merged[-1][0]) < self._min:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        # a trailing runt merges backwards rather than standing alone
        if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < self._min:
            last = merged.pop()
            merged[-1] = (merged[-1][0], last[1])
        return merged

    def _split_large(self, text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for start, end in spans:
            cursor = start
            while end - cursor > self._max:
                window_end = cursor + self._max
                breaks = [m.end() for m in SENTENCE_END.finditer(text, cursor, window_end)]
                cut = breaks[-1] if breaks else window_end
                out.append((cursor, cut))
                cursor = cut
            out.append((cursor, end))
        return out
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_chunking_structural.py -v && make lint
```
Expected: 10 passed, lint clean.

- [ ] **Step 5: Commit**

```bash
git add src/clause/chunking/structural.py tests/test_chunking_structural.py
git commit -m "feat: structure-aware chunker splitting on RBI numbering

Boundaries follow the document's own provision structure, so a chunk
tends to be a complete rule rather than an arbitrary window. Whether that
actually retrieves better is a Phase 2 question; this only makes the
comparison possible."
```

---

## Task 9: Postgres schema, migrations and repository

**Files:**
- Create: `docker-compose.yml`, `alembic.ini`, `migrations/env.py`, `migrations/script.py.mako`, `migrations/versions/0001_initial.py`, `src/clause/db/__init__.py`, `src/clause/db/schema.py`, `src/clause/db/session.py`, `src/clause/db/repository.py`, `tests/conftest.py`, `tests/test_repository.py`
- Modify: `pyproject.toml`, `docs/decisions.md`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `Document`, `Chunk`, `Settings`
- Produces:
  - `DocumentRow`, `ChunkRow` — SQLAlchemy declarative models
  - `make_engine(url: str) -> Engine`, `session_factory(engine) -> sessionmaker[Session]`
  - `upsert_document(session: Session, document: Document) -> None`
  - `replace_chunks(session: Session, doc_id: str, strategy: str, chunks: list[Chunk]) -> None`

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: clause
      POSTGRES_PASSWORD: clause
      POSTGRES_DB: clause
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U clause"]
      interval: 5s
      timeout: 3s
      retries: 10
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/conftest.py
import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from clause.db.schema import Base

DB_URL = os.environ.get(
    "CLAUSE_TEST_DATABASE_URL", "postgresql+psycopg://clause:clause@localhost:5432/clause"
)


@pytest.fixture
def db_session() -> Iterator[Session]:
    try:
        create_engine(DB_URL).connect().close()
    except Exception as exc:  # noqa: BLE001 - any connection failure means no database
        pytest.skip(f"no Postgres at {DB_URL}: {exc}")
    engine = create_engine(DB_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)
    engine.dispose()
```

```python
# tests/test_repository.py
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from clause.chunking.fixed import FixedWindowChunker
from clause.db.repository import replace_chunks, upsert_document
from clause.db.schema import ChunkRow, DocumentRow
from clause.models import Document

pytestmark = pytest.mark.db

DOC = Document(
    doc_id="rbi-1", rbi_id=1, url="https://example.test/1",
    circular_no="RBI/2026-27/1", dept_ref="DOR.X.1", title="t",
    doc_type="circular", published_date=date(2026, 9, 18), effective_date=None,
    sha256="a" * 64, fetched_at=datetime(2026, 9, 19, tzinfo=UTC), text="word " * 400,
)


def test_upsert_is_idempotent(db_session) -> None:
    upsert_document(db_session, DOC)
    upsert_document(db_session, DOC)
    db_session.commit()
    assert len(db_session.scalars(select(DocumentRow)).all()) == 1


def test_replace_chunks_swaps_only_that_strategy(db_session) -> None:
    upsert_document(db_session, DOC)
    fixed = FixedWindowChunker(200, 50).chunk(DOC)
    replace_chunks(db_session, DOC.doc_id, "fixed_window", fixed)
    replace_chunks(db_session, DOC.doc_id, "structural", fixed[:2])
    db_session.commit()

    rows = db_session.scalars(select(ChunkRow)).all()
    assert {r.strategy for r in rows} == {"fixed_window", "structural"}

    replace_chunks(db_session, DOC.doc_id, "fixed_window", fixed[:1])
    db_session.commit()
    rows = db_session.scalars(select(ChunkRow)).all()
    assert len([r for r in rows if r.strategy == "fixed_window"]) == 1
    assert len([r for r in rows if r.strategy == "structural"]) == 2


def test_chunk_rows_store_their_own_provenance(db_session) -> None:
    upsert_document(db_session, DOC)
    replace_chunks(db_session, DOC.doc_id, "fixed_window", FixedWindowChunker(200, 50).chunk(DOC))
    db_session.commit()
    row = db_session.scalars(select(ChunkRow)).first()
    assert row is not None
    assert row.source_url == DOC.url
    assert row.doc_type == DOC.doc_type
```

- [ ] **Step 3: Start Postgres and watch the tests fail**

```bash
make up
uv run pytest tests/test_repository.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.db'`

- [ ] **Step 4: Add dependencies**

```bash
uv add "sqlalchemy>=2.0" alembic "psycopg[binary]"
```
Append to `docs/decisions.md`:
```markdown
- SQLAlchemy 2.0 — schema and queries. Replaces hand-written SQL; typed declarative models keep the row shape and the domain types in one place.
- Alembic — migrations. Chosen over ad-hoc DDL scripts so the schema has a history and CI can build it from empty.
- psycopg (binary) — Postgres driver. Replaces psycopg2; version 3 is maintained and ships wheels.
```

- [ ] **Step 5: Implement the schema**

```python
# src/clause/db/schema.py
from datetime import date, datetime

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    rbi_id: Mapped[int] = mapped_column(Integer)
    url: Mapped[str] = mapped_column(Text)
    circular_no: Mapped[str] = mapped_column(String(64))
    dept_ref: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(Text)
    doc_type: Mapped[str] = mapped_column(String(32))
    published_date: Mapped[date]
    effective_date: Mapped[date | None]
    sha256: Mapped[str] = mapped_column(String(64))
    fetched_at: Mapped[datetime]
    text: Mapped[str] = mapped_column(Text)


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id", ondelete="CASCADE"))
    strategy: Mapped[str] = mapped_column(String(32))
    ordinal: Mapped[int] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    # Denormalised from documents so a retrieved chunk carries its own
    # provenance. CLAUDE.md requires every chunk to store these; a citation
    # that needs a second query to say where it came from can be separated
    # from its source.
    source_url: Mapped[str] = mapped_column(Text)
    effective_date: Mapped[date | None]
    doc_type: Mapped[str] = mapped_column(String(32))

    created_at: Mapped[datetime]


Index("ix_chunks_doc_strategy", ChunkRow.doc_id, ChunkRow.strategy)
```

- [ ] **Step 6: Implement session and repository**

```python
# src/clause/db/session.py
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
```

```python
# src/clause/db/repository.py
from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from clause.db.schema import ChunkRow, DocumentRow
from clause.models import Chunk, Document


def upsert_document(session: Session, document: Document) -> None:
    values = {
        "doc_id": document.doc_id,
        "rbi_id": document.rbi_id,
        "url": document.url,
        "circular_no": document.circular_no,
        "dept_ref": document.dept_ref,
        "title": document.title,
        "doc_type": document.doc_type,
        "published_date": document.published_date,
        "effective_date": document.effective_date,
        "sha256": document.sha256,
        "fetched_at": document.fetched_at,
        "text": document.text,
    }
    stmt = insert(DocumentRow).values(**values)
    stmt = stmt.on_conflict_do_update(index_elements=[DocumentRow.doc_id], set_=values)
    session.execute(stmt)


def replace_chunks(session: Session, doc_id: str, strategy: str, chunks: list[Chunk]) -> None:
    """Replace this document's chunks for one strategy, leaving others intact.

    Delete-then-insert in the caller's transaction, so a failure mid-ingest
    leaves no partial rows.
    """
    session.execute(
        delete(ChunkRow).where(ChunkRow.doc_id == doc_id, ChunkRow.strategy == strategy)
    )
    now = datetime.now(UTC)
    session.add_all(
        [
            ChunkRow(
                doc_id=c.doc_id,
                strategy=c.strategy,
                ordinal=c.ordinal,
                char_start=c.char_start,
                char_end=c.char_end,
                text=c.text,
                source_url=c.source_url,
                effective_date=c.effective_date,
                doc_type=c.doc_type,
                created_at=now,
            )
            for c in chunks
        ]
    )
```

Create an empty `src/clause/db/__init__.py`.

- [ ] **Step 7: Generate and verify the migration**

```bash
uv run alembic init -t generic migrations
```
Then in `alembic.ini` set `sqlalchemy.url =` (leave empty) and in `migrations/env.py` replace the `target_metadata = None` line with:

```python
import os

from clause.db.schema import Base

target_metadata = Base.metadata
config.set_main_option(
    "sqlalchemy.url",
    os.environ.get("CLAUSE_DATABASE_URL", "postgresql+psycopg://clause:clause@localhost:5432/clause"),
)
```

```bash
uv run alembic revision --autogenerate -m "initial documents and chunks"
uv run alembic upgrade head
uv run alembic downgrade base && uv run alembic upgrade head
```
Expected: the migration applies, reverses and re-applies cleanly.

- [ ] **Step 8: Add Postgres to CI**

In `.github/workflows/ci.yml`, add to the `check` job:

```yaml
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: clause
          POSTGRES_PASSWORD: clause
          POSTGRES_DB: clause
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U clause"
          --health-interval 5s
          --health-timeout 3s
          --health-retries 10
```

and set the environment for the pytest step:

```yaml
      - run: uv run pytest
        env:
          CLAUSE_TEST_DATABASE_URL: postgresql+psycopg://clause:clause@localhost:5432/clause
```

The `db` marker exists for local convenience; CI always has a database, so database tests always run rather than being skipped into irrelevance.

- [ ] **Step 9: Run tests**

```bash
uv run pytest tests/test_repository.py -v && make lint
```
Expected: 3 passed, lint clean.

- [ ] **Step 10: Commit**

```bash
git add docker-compose.yml alembic.ini migrations/ src/clause/db/ \
        tests/conftest.py tests/test_repository.py pyproject.toml uv.lock \
        docs/decisions.md .github/workflows/ci.yml
git commit -m "feat: postgres schema, migrations and repository

replace_chunks is scoped to one strategy so re-chunking with one
algorithm cannot silently delete the other's rows -- Phase 2 compares
them, and a comparison against half-missing data is worse than no
comparison.

Chunk provenance columns are denormalised deliberately; see the spec."
```

---

## Task 10: Discovery script and the frozen manifest

**Files:**
- Create: `src/clause/sources/discover.py`, `tests/test_discover.py`, `data/corpus/kyc.manifest.jsonl`
- Modify: `.gitignore` (ensure `data/raw/` ignored, `data/corpus/` tracked)

**Interfaces:**
- Consumes: `ManifestEntry`, `write_manifest`, `Fetcher`, `validate_response`, `canonical_text`, `parse_header`
- Produces:
  - `KYC_TERMS: tuple[str, ...]`
  - `is_kyc_document(title: str, text: str) -> bool`
  - `build_entry(rbi_id: int, url: str, raw: bytes, title: str) -> ManifestEntry`
  - a committed `data/corpus/kyc.manifest.jsonl` with **at least 50 entries**

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_discover.py
import hashlib
from pathlib import Path

from clause.sources.discover import build_entry, is_kyc_document
from clause.sources.manifest import load_manifest

FIXTURES = Path(__file__).parent / "fixtures"
RAW = (FIXTURES / "rbi_13704.html").read_bytes()


def test_recognises_a_kyc_document() -> None:
    assert is_kyc_document("Know Your Customer Amendment Directions, 2026", "") is True


def test_rejects_an_unrelated_document() -> None:
    assert is_kyc_document("Priority Sector Lending targets", "housing loans") is False


def test_build_entry_derives_id_and_hash() -> None:
    entry = build_entry(13704, "https://example.test/d", RAW, "Know Your Customer")
    assert entry.doc_id == "rbi-13704"
    assert entry.sha256 == hashlib.sha256(RAW).hexdigest()
    assert entry.circular_no == "RBI/2026-27/262"


def test_committed_manifest_is_loadable_and_large_enough() -> None:
    entries = load_manifest(Path("data/corpus/kyc.manifest.jsonl"))
    assert len(entries) >= 50, "PROMPT.md Phase 1 requires at least 50 real documents"
    assert len({e.doc_id for e in entries}) == len(entries)
    assert all(len(e.sha256) == 64 for e in entries)
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_discover.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.sources.discover'`

- [ ] **Step 3: Implement**

```python
# src/clause/sources/discover.py
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

from clause.ingest.extract import ExtractionError, canonical_text, parse_header
from clause.ingest.validate import ValidationError, validate_response
from clause.models import ManifestEntry
from clause.sources.manifest import write_manifest

URL_TEMPLATE = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={rbi_id}&Mode=0"

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
    _, _, published = parse_header(text)
    marker = published.strftime("%B %-d, %Y") if sys.platform != "win32" else published.strftime("%B %#d, %Y")
    idx = text.find(marker)
    if idx == -1:
        return ""
    return text[idx + len(marker) : idx + len(marker) + 200].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-id", type=int, required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument(
        "--user-agent", default="clause-research/0.1 (+mailto:you@example.com)"
    )
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
                    min_chars=500,
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
    return 0 if len(entries) >= 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run discovery for real**

This takes roughly 15–25 minutes at a 2-second interval. It is the only long-running step.

```bash
uv run python -m clause.sources.discover \
  --start-id 13704 --count 600 --target 60 \
  --out data/corpus/kyc.manifest.jsonl
wc -l data/corpus/kyc.manifest.jsonl
```
Expected: at least 50 lines. If fewer, widen `--count` and re-run.

- [ ] **Step 5: Review the manifest by hand**

Read it. This review is the freeze. Check that titles are genuinely KYC/AML, that dates look sane, and that no `doc_id` repeats. Remove any entry that is off-topic — a keyword match is not a judgement.

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_discover.py -v && make lint
```
Expected: 4 passed, lint clean.

- [ ] **Step 7: Commit**

```bash
git add src/clause/sources/discover.py tests/test_discover.py data/corpus/kyc.manifest.jsonl
git commit -m "feat: corpus discovery and the frozen KYC/AML manifest

Discovery is deliberately not part of make ingest. The manual review
between running it and committing the result is what freezes the corpus,
and a frozen corpus is the only reason a recall delta in Phase 2 can be
attributed to a code change."
```

---

## Task 11: Ingest CLI and the Phase 1 acceptance test

**Files:**
- Create: `src/clause/cli.py`, `tests/test_ingest_acceptance.py`
- Modify: `PROMPT.md` (§1 robots.txt correction), `STATUS.md`, `README.md`

**Interfaces:**
- Consumes: everything above
- Produces: `python -m clause.cli ingest --manifest <path>` → exit 0 on full success, non-zero if any entry failed

- [ ] **Step 1: Write the failing acceptance tests**

```python
# tests/test_ingest_acceptance.py
import pytest
from sqlalchemy import select

from clause.db.schema import ChunkRow, DocumentRow

pytestmark = pytest.mark.db


def test_corpus_is_ingested(db_session, ingested) -> None:
    docs = db_session.scalars(select(DocumentRow)).all()
    assert len(docs) >= 50, "PROMPT.md Phase 1 requires at least 50 real documents"


def test_every_chunk_slice_roundtrips_against_stored_source(db_session, ingested) -> None:
    """PROMPT.md Phase 1 definition of done."""
    texts = {d.doc_id: d.text for d in db_session.scalars(select(DocumentRow)).all()}
    chunks = db_session.scalars(select(ChunkRow)).all()
    assert chunks, "no chunks were written"
    for c in chunks:
        assert texts[c.doc_id][c.char_start : c.char_end] == c.text, (
            f"{c.doc_id}#{c.ordinal} ({c.strategy}) span does not round-trip"
        )


def test_both_strategies_are_present(db_session, ingested) -> None:
    strategies = {r.strategy for r in db_session.scalars(select(ChunkRow)).all()}
    assert strategies == {"fixed_window", "structural"}


def test_every_chunk_carries_its_provenance(db_session, ingested) -> None:
    for c in db_session.scalars(select(ChunkRow)).all():
        assert c.source_url
        assert c.doc_type
        assert c.char_start < c.char_end
```

Add to `tests/conftest.py`:

```python
@pytest.fixture
def ingested(db_session):
    """Run the real ingest against the committed manifest and the warm cache."""
    from pathlib import Path

    from clause.cli import ingest

    manifest = Path("data/corpus/kyc.manifest.jsonl")
    if not manifest.exists():
        pytest.skip("manifest not present")
    ingest(manifest, session=db_session)
    db_session.commit()
    return True
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_ingest_acceptance.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.cli'`

- [ ] **Step 3: Implement the CLI**

```python
# src/clause/cli.py
import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy.orm import Session

from clause.chunking.base import assert_slices
from clause.chunking.fixed import FixedWindowChunker
from clause.chunking.structural import StructuralChunker
from clause.config import Settings, get_settings
from clause.db.repository import replace_chunks, upsert_document
from clause.db.session import make_engine, session_factory
from clause.ingest.extract import ExtractionError, extract_document
from clause.ingest.fetch import Fetcher, FetchError, RateLimiter
from clause.ingest.validate import ValidationError
from clause.sources.manifest import load_manifest


def _chunkers(settings: Settings) -> list[FixedWindowChunker | StructuralChunker]:
    return [
        FixedWindowChunker(settings.window_chars, settings.overlap_chars),
        StructuralChunker(settings.min_chunk_chars, settings.max_chunk_chars),
    ]


def ingest(manifest_path: Path, *, session: Session, settings: Settings | None = None) -> list[str]:
    """Ingest every manifest entry. Returns the list of doc_ids that failed."""
    settings = settings or get_settings()
    entries = load_manifest(manifest_path)
    failures: list[str] = []

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        fetcher = Fetcher(settings, client, RateLimiter(settings.min_request_interval_s))
        for entry in entries:
            try:
                raw = fetcher.fetch(entry)
                document = extract_document(entry, raw, fetched_at=datetime.now(UTC))
                upsert_document(session, document)
                for chunker in _chunkers(settings):
                    chunks = chunker.chunk(document)
                    # An unresolvable citation is a hard failure, never a warning.
                    assert_slices(document, chunks)
                    replace_chunks(session, document.doc_id, chunker.name, chunks)
            except (FetchError, ValidationError, ExtractionError, AssertionError) as exc:
                print(f"FAILED {entry.doc_id}: {exc}", file=sys.stderr)
                failures.append(entry.doc_id)
                session.rollback()
            else:
                session.commit()

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clause")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    engine = make_engine(settings.database_url)
    with session_factory(engine)() as session:
        failures = ingest(args.manifest, session=session, settings=settings)

    if failures:
        print(f"\n{len(failures)} document(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("ingest complete, all documents succeeded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the real ingest**

```bash
make up
uv run alembic upgrade head
make ingest
```
Expected: exit 0. The first run fetches at 2-second intervals; subsequent runs are cache-served and near-instant.

- [ ] **Step 5: Run the acceptance tests**

```bash
uv run pytest tests/test_ingest_acceptance.py -v
```
Expected: 4 passed.

- [ ] **Step 6: Verify idempotency and the offline property**

```bash
make ingest && make ingest    # second run must perform zero network requests
uv run pytest -q
```
Expected: both runs exit 0; full suite green.

- [ ] **Step 7: Correct `PROMPT.md` §1**

Replace the sentence `Respect robots.txt and rate-limit fetches.` with:

```markdown
`robots.txt` on rbi.org.in returns HTTP 418 to every client tried, so its policy cannot be
read and compliance with it cannot be claimed. In its place the fetcher holds a
conservative floor: single concurrency, a minimum 2-second interval, an identifying
User-Agent with a contact address, and a cache that means each document is fetched exactly
once. The PDF host sits behind a JavaScript challenge, so the HTML page is the canonical
source and PDFs are not fetched at all.
```

- [ ] **Step 8: Update `STATUS.md`**

```markdown
# Status

**Phase:** 1 — complete

| Phase | State | Evidence |
|---|---|---|
| 0 scaffold | done | CI green |
| 1 ingestion | done | `make ingest` + `pytest tests/test_ingest_acceptance.py` |
| 2 retrieval + eval | not started | — |
| 3 citations + refusal | not started | — |
| 4 service | not started | — |
| 5 deploy | not started | — |

## Measured numbers
None yet. Nothing goes in this section that is not produced by `make eval`.
```

Document counts and chunk counts do **not** go in "Measured numbers" — that section is for `make eval` output only.

- [ ] **Step 9: Commit**

```bash
git add src/clause/cli.py tests/test_ingest_acceptance.py tests/conftest.py \
        PROMPT.md STATUS.md
git commit -m "feat: ingest CLI and phase 1 acceptance tests

assert_slices runs inside ingest before any chunk is written, so a
non-slicing chunker fails the ingest rather than quietly populating the
database with citations that do not resolve.

A failed document rolls back and is reported; the command exits non-zero
so a partial ingest can never look like a successful one.

PROMPT.md's robots.txt instruction is corrected rather than left standing:
robots.txt returns 418, so respecting it is not something the code can do."
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §4 Phase 0 scaffold | 1 |
| §5.1 Discovery not part of ingest | 10 |
| §5.2 The invariant | 7 (`make_chunk`), 6 (`canonical_text`), 11 (`assert_slices` in ingest) |
| §5.3 Two chunkers, parameters | 7, 8, 2 |
| §5.4 Data model, denormalisation | 9 |
| §6 Data flow | 11 |
| §7.1 Content validation gate | 4 |
| §7.2 Hash mismatch, retries, idempotency | 5, 9, 11 |
| §8 Politeness + `PROMPT.md` correction | 5, 11 step 7 |
| §9 Testing | every task |
| §10 Dependencies | 2, 4, 5, 9 (each with a `docs/decisions.md` line) |
| §12 Traceability | Task 11 acceptance tests |

No gaps.

**Placeholder scan:** no TBD/TODO. Every code step contains runnable code. The one deliberately unspecified value is `sha256` in the Task 3 test fixture (`"a" * 64`), which is test data, not a placeholder.

**Type consistency:** `Document`, `Chunk`, `ManifestEntry` are defined once in Task 3 and used unchanged in Tasks 5–11. `Chunker.chunk(document: Document) -> list[Chunk]` is used identically in Tasks 7, 8 and 11. This refines the spec's `chunk(text, doc)` signature — `Document` already carries its text, so passing both invited them to disagree. `make_chunk` and `assert_slices` keep their Task 7 signatures throughout.

---

## Known risks

1. **Discovery yield is unverified.** Task 10 walks `Id` backwards and keyword-filters. Whether 600 ids contain 50 KYC/AML documents is not known until it runs. Mitigation: `--count` is a parameter and the step says to widen and re-run. If the yield is very low, the fallback is to seed from RBI's year/month listing pages instead — a change to `discover.py` only, touching nothing else.
2. **`_title_of` is heuristic.** It reads the text following the date. If titles come out poor, it affects `is_kyc_document` precision, and the Task 10 step 5 manual review is the backstop.
3. **RBI may change its block behaviour**, making the Task 4 fixture uncapturable. The step says to stop and report rather than fabricate a fixture, because that test is what protects the corpus from being silently worthless.
