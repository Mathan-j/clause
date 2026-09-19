import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from clause.cli import ingest
from clause.db.schema import Base

DB_URL = os.environ.get(
    "CLAUSE_TEST_DATABASE_URL", "postgresql+psycopg://clause:clause@localhost:5434/clause"
)


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
    """Run the real ingest against the committed manifest and the warm cache."""
    manifest = Path("data/corpus/kyc.manifest.jsonl")
    if not manifest.exists():
        pytest.skip("manifest not present")
    ingest(manifest, session=db_session)
    db_session.commit()
    return True
