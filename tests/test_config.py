from pathlib import Path

import pytest

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
    with pytest.raises(ValueError):
        Settings(
            database_url="postgresql+psycopg://x/y",
            user_agent="t",
            window_chars=100,
            overlap_chars=100,
        )
