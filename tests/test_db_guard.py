"""The guard that stops the test suite from destroying the application database.

This exists because it already happened once: `conftest` defaulted its test URL to
the same database the app ingests into, and `pytest` dropped the corpus.
"""

import pytest

from conftest import _same_physical_database

APP = "postgresql+psycopg://clause:clause@localhost:5434/clause"


@pytest.mark.parametrize(
    "other",
    [
        APP,
        "postgresql+psycopg://clause:clause@127.0.0.1:5434/clause",  # host alias
        "postgresql://clause:clause@localhost:5434/clause",  # different driver
        "postgresql+psycopg://clause:clause@localhost:5434/clause?sslmode=disable",
        "postgresql+psycopg://other:pw@localhost:5434/clause",  # credentials differ
    ],
)
def test_aliases_of_the_same_database_are_detected(other: str) -> None:
    assert _same_physical_database(APP, other)


@pytest.mark.parametrize(
    "other",
    [
        "postgresql+psycopg://clause:clause@localhost:5434/clause_test",  # the default
        "postgresql+psycopg://clause:clause@localhost:5432/clause",  # different port
        "postgresql+psycopg://clause:clause@db.example.com:5434/clause",  # different host
    ],
)
def test_genuinely_different_databases_are_not_flagged(other: str) -> None:
    assert not _same_physical_database(APP, other)


def test_omitted_port_matches_the_postgres_default() -> None:
    assert _same_physical_database(
        "postgresql+psycopg://clause:clause@localhost/clause",
        "postgresql+psycopg://clause:clause@localhost:5432/clause",
    )
