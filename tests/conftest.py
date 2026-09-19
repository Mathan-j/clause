import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from clause.cli import ingest
from clause.db.schema import Base
from clause.sources.manifest import load_manifest

DB_URL = os.environ.get(
    "CLAUSE_TEST_DATABASE_URL", "postgresql+psycopg://clause:clause@localhost:5434/clause"
)

# Mirrors Settings.raw_cache_dir's own default and env var name (CLAUSE_ prefix), read
# directly rather than via get_settings() so the warm-cache check below never needs a
# fully valid Settings() (CLAUSE_DATABASE_URL/CLAUSE_USER_AGENT) just to decide to skip.
RAW_CACHE_DIR = Path(os.environ.get("CLAUSE_RAW_CACHE_DIR", "data/raw"))


@pytest.fixture
def db_session() -> Iterator[Session]:
    try:
        create_engine(DB_URL).connect().close()
    except Exception as exc:  # any connection failure means no database
        pytest.skip(f"no Postgres at {DB_URL}: {exc}")
    engine = create_engine(DB_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)
    engine.dispose()


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
