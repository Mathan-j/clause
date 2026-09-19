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
    except Exception as exc:  # any connection failure means no database
        pytest.skip(f"no Postgres at {DB_URL}: {exc}")
    engine = create_engine(DB_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)
    engine.dispose()
