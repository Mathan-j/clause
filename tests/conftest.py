import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from clause.cli import ingest, run_eval
from clause.sources.manifest import load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent


_LOOPBACK = {"localhost", "127.0.0.1", "::1", ""}
_DEFAULT_PG_PORT = 5432


def _resolve_env_url(key: str) -> str | None:
    """Resolve a CLAUSE_* URL the way Settings does: real env var, then `.env`.

    Reading `.env` matters: `.env.example` ships both URLs and the README tells you
    to configure them there, so a guard that only consulted `os.environ` would be
    inert for exactly the setup the docs describe.
    """
    if key in os.environ:
        return os.environ[key]
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            return stripped.split("=", 1)[1].strip()
    return None


def _same_physical_database(left: str, right: str) -> bool:
    """True when two URLs address the same host, port and database.

    Compared on resolved components, not as strings. `localhost` and `127.0.0.1`,
    a `postgresql://` and a `postgresql+psycopg://` spelling, an omitted port and
    an explicit 5432, and a trailing query string all name the same database -- a
    byte comparison waves every one of them through, and the thing on the other
    side of this check is the ingested corpus.
    """
    try:
        a, b = make_url(left), make_url(right)
    except Exception:  # an unparseable URL falls back to exact match
        return left == right

    def host(url: object) -> str:
        raw = (getattr(url, "host", None) or "").lower()
        return "localhost" if raw in _LOOPBACK else raw

    def port(url: object) -> int:
        return getattr(url, "port", None) or _DEFAULT_PG_PORT

    return (host(a), port(a), a.database) == (host(b), port(b), b.database)


# A *distinct* database from the application's, on the same server, by default --
# never the same name as CLAUSE_DATABASE_URL / .env.example's default. The test suite
# tears down its schema on every run (see db_session below); sharing a database with
# the app would mean running `pytest` drops the ingested corpus, which is exactly the
# incident this default now prevents.
DB_URL = (
    _resolve_env_url("CLAUSE_TEST_DATABASE_URL")
    or "postgresql+psycopg://clause:clause@localhost:5434/clause_test"
)

_app_url = _resolve_env_url("CLAUSE_DATABASE_URL")
if _app_url is not None and _same_physical_database(_app_url, DB_URL):
    raise RuntimeError(
        "CLAUSE_TEST_DATABASE_URL resolves to the same physical database as "
        f"CLAUSE_DATABASE_URL ({DB_URL!r}). The test suite builds and tears down its "
        "schema on this URL on every run (via `alembic upgrade head` / `downgrade "
        "base`), so pointing it at the application's own database would destroy "
        "whatever has been ingested there. Point CLAUSE_TEST_DATABASE_URL at a "
        "separate database on the same server, e.g. `clause_test`, and re-run."
    )

# Mirrors Settings.raw_cache_dir's own default and env var name (CLAUSE_ prefix), read
# directly rather than via get_settings() so the warm-cache check below never needs a
# fully valid Settings() (CLAUSE_DATABASE_URL/CLAUSE_USER_AGENT) just to decide to skip.
RAW_CACHE_DIR = Path(os.environ.get("CLAUSE_RAW_CACHE_DIR", "data/raw"))


@contextlib.contextmanager
def _database_url_env(url: str) -> Iterator[None]:
    """Temporarily point `migrations/env.py`'s own CLAUSE_DATABASE_URL lookup at `url`.

    `migrations/env.py` reads `CLAUSE_DATABASE_URL` from the environment directly
    (matching how `make ingest`/the real app resolve it) rather than from Alembic's
    `Config.sqlalchemy.url`, so driving Alembic at a different database in-process
    means overriding that env var for the duration of the call, then restoring it.
    """
    previous = os.environ.get("CLAUSE_DATABASE_URL")
    os.environ["CLAUSE_DATABASE_URL"] = url
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CLAUSE_DATABASE_URL", None)
        else:
            os.environ["CLAUSE_DATABASE_URL"] = previous


def _alembic_config() -> Config:
    return Config(str(REPO_ROOT / "alembic.ini"))


def _ensure_test_database_exists(url: str) -> None:
    """Create the test database if it does not exist yet; skip with a clear message
    if the server itself is unreachable.

    Connects to the server's `postgres` maintenance database to run `CREATE DATABASE`
    (which cannot run inside a transaction, hence AUTOCOMMIT), so a fresh clone gets a
    working test database instead of a raw connection traceback the first time
    `pytest` runs.
    """
    parsed = make_url(url)
    dbname = parsed.database
    admin_url = parsed.set(database="postgres")
    try:
        admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": dbname}
            ).first()
            if exists is None:
                conn.execute(text(f'CREATE DATABASE "{dbname}"'))
        admin_engine.dispose()
    except Exception as exc:
        pytest.skip(
            f"no Postgres server reachable to create test database {dbname!r} "
            f"(via {admin_url}): {exc}. Start it with `docker compose up -d` or "
            f"create {dbname!r} by hand."
        )


@pytest.fixture
def db_session() -> Iterator[Session]:
    _ensure_test_database_exists(DB_URL)
    try:
        create_engine(DB_URL).connect().close()
    except Exception as exc:  # any connection failure means no usable test database
        pytest.skip(f"no Postgres at {DB_URL}: {exc}")

    cfg = _alembic_config()
    with _database_url_env(DB_URL):
        command.upgrade(cfg, "head")

    engine = create_engine(DB_URL)
    with Session(engine) as session:
        yield session
    engine.dispose()

    with _database_url_env(DB_URL):
        command.downgrade(cfg, "base")


@pytest.fixture
def ingested(db_session: Session) -> bool:
    """Run the real ingest against the committed manifest and the warm cache.

    Skips rather than fetching live: if `data/raw/` is not already warm for every
    manifest entry, running the real ingest here would perform a live, ~61-request
    fetch of www.rbi.org.in. That is fine for a human running it locally right after
    `make ingest`, but `data/raw/` is gitignored, so CI always starts cold — and CI runs
    on every push and pull request. Scraping a public regulator on every commit
    contradicts the rate-limiting/politeness commitment this project states in
    PROMPT.md and enforces in RateLimiter, so this fixture refuses to be the thing that
    does it. It skips instead of failing precisely so that skip is legible as
    deliberate, not broken.
    """
    manifest = Path("data/corpus/kyc.manifest.jsonl")
    if not manifest.exists():
        pytest.skip("manifest not present")

    entries = load_manifest(manifest)
    missing = [e.doc_id for e in entries if not (RAW_CACHE_DIR / f"{e.doc_id}.html").exists()]
    if missing:
        pytest.skip(
            "acceptance tests require a warm data/raw/ cache for the whole corpus; "
            "this is deliberate, not a broken test, because a cold cache would make "
            "this fixture fetch www.rbi.org.in live on every CI run, contradicting the "
            "project's own rate-limiting commitment. Populate the cache locally first "
            "with `uv run python -m clause.cli ingest --manifest "
            f"{manifest}`, then re-run. Missing cache entries for {len(missing)} of "
            f"{len(entries)} documents, e.g. {missing[:3]}."
        )

    failures = ingest(manifest, session=db_session)
    assert not failures, f"ingest reported {len(failures)} failed document(s): {failures}"
    db_session.commit()
    return True


@pytest.fixture
def evaluated(db_session: Session, tmp_path: Path) -> tuple[Path, Path]:
    """Run the real evaluation once, against the warm cache and a live Qdrant.

    `db_session` is a dependency only for its skip-if-no-Postgres behaviour --
    `run_eval()` opens its own session against `CLAUSE_DATABASE_URL` (the real,
    frozen corpus), not against this fixture's throwaway `clause_test` schema.

    Writes to `tmp_path`, never to `reports/`. `reports/eval.md` and
    `reports/eval.json` are committed artifacts that Task 11's CI gate exists
    to protect; a test run that wrote over them on every `pytest` invocation
    would silently replace a good report with a bad one, or leave the tree
    dirty after every green test suite. `run_eval` takes output paths for
    exactly this reason.
    """
    md_path = tmp_path / "eval.md"
    json_path = tmp_path / "eval.json"
    run_eval(md_path=md_path, json_path=json_path)
    return md_path, json_path
