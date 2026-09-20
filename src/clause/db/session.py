from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(url: str, *, connect_timeout: float | None = None) -> Engine:
    """Build the SQLAlchemy engine every command shares.

    `connect_timeout` is opt-in and `None` by default, so `ingest`/`index`/
    `eval` keep today's behaviour (no bound -- a developer machine's own
    Postgres is trusted). `clause.cli.run_gate` passes one explicitly: an
    unbounded connection attempt against a misconfigured `CLAUSE_DATABASE_URL`
    in CI would otherwise hang the job on the OS-level TCP timeout instead of
    failing fast into the gate's documented "database unreachable" fallback.
    """
    connect_args: dict[str, Any] = (
        {"connect_timeout": connect_timeout} if connect_timeout is not None else {}
    )
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
