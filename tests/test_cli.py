"""Wiring tests for `clause.cli.index_all`, fully mocked -- no live Qdrant, no
live Postgres, no model load. `index_strategy` itself is covered against a real
Qdrant/Postgres in tests/test_index.py; these tests exist only to prove
`index_all` calls it correctly for every configured strategy and to prove the
foreign-collection warning path (see clause.index.foreign_collections) fires
without ever refusing to run.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from sqlalchemy.orm import Session

from clause import cli
from clause.config import Settings


def _settings() -> Settings:
    return Settings(
        database_url="postgresql+psycopg://clause:clause@localhost:5434/clause_test",
        user_agent="clause-test/0.1 (+mailto:test@example.com)",
    )


@dataclass
class _StubCollection:
    name: str


@dataclass
class _StubCollectionsResult:
    collections: list[_StubCollection] = field(default_factory=list)


class _StubQdrantClient:
    """Stands in for qdrant_client.QdrantClient: only get_collections() is used
    by index_all itself (index_strategy is monkeypatched away in these tests)."""

    def __init__(self, names: Sequence[str]) -> None:
        self._names = list(names)

    def get_collections(self) -> _StubCollectionsResult:
        return _StubCollectionsResult([_StubCollection(name=n) for n in self._names])


def test_index_all_calls_index_strategy_once_per_configured_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_index_strategy(
        client: object,
        encoder: object,
        session: object,
        strategy: str,
        *,
        batch_size: int = 64,
    ) -> int:
        calls.append(strategy)
        return {"fixed_window": 7, "structural": 9}[strategy]

    monkeypatch.setattr(cli, "QdrantClient", lambda url: _StubQdrantClient([]))
    monkeypatch.setattr(cli, "Encoder", lambda model_name: object())
    monkeypatch.setattr(cli, "index_strategy", fake_index_strategy)

    counts = cli.index_all(session=Session(), settings=_settings())

    assert calls == list(cli.STRATEGIES)
    assert counts == {"fixed_window": 7, "structural": 9}


def test_index_all_proceeds_quietly_when_only_our_collections_are_present(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli, "QdrantClient", lambda url: _StubQdrantClient(["clause_structural"])
    )
    monkeypatch.setattr(cli, "Encoder", lambda model_name: object())
    monkeypatch.setattr(cli, "index_strategy", lambda *a, **k: 0)

    cli.index_all(session=Session(), settings=_settings())

    assert "non-clause" not in capsys.readouterr().err


def test_index_all_warns_but_continues_when_foreign_collections_present(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The production path warns rather than refuses (unlike the test fixture in
    tests/test_index.py): it only ever adds clause_* collections, so the blast
    radius of a misconfigured CLAUSE_QDRANT_URL is far smaller, and refusing
    would block a legitimate first run against a fresh shared instance.
    """
    calls: list[str] = []

    def fake_index_strategy(
        client: object, encoder: object, session: object, strategy: str, **kwargs: object
    ) -> int:
        calls.append(strategy)
        return 0

    monkeypatch.setattr(
        cli,
        "QdrantClient",
        lambda url: _StubQdrantClient(["clause_structural", "someone_elses_data"]),
    )
    monkeypatch.setattr(cli, "Encoder", lambda model_name: object())
    monkeypatch.setattr(cli, "index_strategy", fake_index_strategy)

    counts = cli.index_all(session=Session(), settings=_settings())

    # Warn, don't refuse: both configured strategies still ran.
    assert calls == list(cli.STRATEGIES)
    assert counts == dict.fromkeys(cli.STRATEGIES, 0)
    assert "someone_elses_data" in capsys.readouterr().err
