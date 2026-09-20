"""Wiring tests for `clause.cli.index_all`, fully mocked -- no live Qdrant, no
live Postgres, no model load. `index_strategy` itself is covered against a real
Qdrant/Postgres in tests/test_index.py; these tests exist only to prove
`index_all` calls it correctly for every configured strategy and to prove the
foreign-collection warning path (see clause.index.foreign_collections) fires
without ever refusing to run.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

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


# ---------------------------------------------------------------------------
# _retrieval_code_sha256: the fingerprint field that catches a retrieval
# code regression with no fresh `make eval` (fix round 2, blocker 3).
# ---------------------------------------------------------------------------


def test_retrieval_code_sha256_default_paths_are_real_and_stable() -> None:
    """Every path in RETRIEVAL_CODE_PATHS is a real, readable file, and
    hashing them twice gives the same answer.
    """
    first = cli._retrieval_code_sha256()
    second = cli._retrieval_code_sha256()
    assert first == second
    assert len(first) == 64


def test_retrieval_code_sha256_is_order_independent(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("print('a')\n", encoding="utf-8", newline="\n")
    b.write_text("print('b')\n", encoding="utf-8", newline="\n")

    assert cli._retrieval_code_sha256([a, b]) == cli._retrieval_code_sha256([b, a])


def test_retrieval_code_sha256_changes_when_the_code_does(tmp_path: Path) -> None:
    """This is the property the whole field exists for: editing a retrieval
    file must move the hash, or a code regression with no fresh `make eval`
    would still pass every fingerprint field unchanged.
    """
    path = tmp_path / "a.py"
    path.write_text("print('a')\n", encoding="utf-8", newline="\n")
    before = cli._retrieval_code_sha256([path])

    path.write_text("print('a')  # changed\n", encoding="utf-8", newline="\n")
    after = cli._retrieval_code_sha256([path])

    assert before != after


def test_retrieval_code_sha256_changes_when_a_file_is_renamed(tmp_path: Path) -> None:
    same_content = "print('a')\n"
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text(same_content, encoding="utf-8", newline="\n")
    b.write_text(same_content, encoding="utf-8", newline="\n")

    assert cli._retrieval_code_sha256([a]) != cli._retrieval_code_sha256([b])


def test_retrieval_code_sha256_is_stable_across_line_endings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Windows checkout (CRLF) and a Linux CI checkout (LF) of the
    identical commit must hash identically -- otherwise this field would
    fail every build on whichever platform did not produce the committed
    report. The relative path text is identical in both calls (as it always
    is for RETRIEVAL_CODE_PATHS, which are POSIX-style repo-relative
    literals on every OS); only the on-disk bytes differ, exactly as a real
    cross-platform checkout would.
    """
    win_dir, nix_dir = tmp_path / "win", tmp_path / "nix"
    win_dir.mkdir()
    nix_dir.mkdir()
    (win_dir / "code.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    (nix_dir / "code.py").write_bytes(b"a = 1\nb = 2\n")

    monkeypatch.chdir(win_dir)
    win_hash = cli._retrieval_code_sha256([Path("code.py")])

    monkeypatch.chdir(nix_dir)
    nix_hash = cli._retrieval_code_sha256([Path("code.py")])

    assert win_hash == nix_hash
