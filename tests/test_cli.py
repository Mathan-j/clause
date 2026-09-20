"""Wiring tests for `clause.cli.index_all`, fully mocked -- no live Qdrant, no
live Postgres, no model load. `index_strategy` itself is covered against a real
Qdrant/Postgres in tests/test_index.py; these tests exist only to prove
`index_all` calls it correctly for every configured strategy and to prove the
foreign-collection warning path (see clause.index.foreign_collections) fires
without ever refusing to run.
"""

import ast
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


# ---------------------------------------------------------------------------
# RETRIEVAL_CODE_PATHS membership: dropping a file, or adding a new one
# without listing it, must be caught rather than silently going unhashed
# until the next time someone happens to notice (fix round 3, blocker N3).
# ---------------------------------------------------------------------------


def test_retrieval_code_paths_covers_every_chunking_module() -> None:
    chunking_dir = Path("src/clause/chunking")
    on_disk = set(chunking_dir.glob("*.py"))
    listed = {p for p in cli.RETRIEVAL_CODE_PATHS if p.parent == chunking_dir}
    assert on_disk, "no .py files found under src/clause/chunking -- test itself is broken"
    assert on_disk == listed, (
        f"RETRIEVAL_CODE_PATHS disagrees with the files on disk under "
        f"{chunking_dir}: on disk but not listed: {on_disk - listed}; "
        f"listed but not on disk: {listed - on_disk}"
    )


def test_retrieval_code_paths_covers_every_module_retrieve_directly_imports() -> None:
    """Every `clause.*` module that `clause.retrieve` itself directly imports
    must be listed. Direct imports only, not the full transitive closure --
    `clause.index` (one of retrieve.py's direct imports) itself imports
    `clause.db.schema` for indexing, which cannot move a number this eval
    computes and is deliberately not part of this set; walking the whole
    graph would pull that in too and silently contradict the documented
    exclusions above it.
    """
    retrieve_path = Path("src/clause/retrieve.py")
    tree = ast.parse(retrieve_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("clause")
    }
    assert imported_modules, "retrieve.py imports no clause.* module -- test itself is broken"

    expected = {Path("src") / (mod.replace(".", "/") + ".py") for mod in imported_modules}
    expected.add(retrieve_path)

    listed = set(cli.RETRIEVAL_CODE_PATHS)
    missing = expected - listed
    assert not missing, (
        f"retrieve.py imports {sorted(imported_modules)}, but RETRIEVAL_CODE_PATHS "
        f"is missing: {missing}"
    )


def test_retrieval_code_paths_includes_golden_py() -> None:
    """`golden.py` parses the committed golden set into the `GoldenQuestion`
    objects `evaluate()` scores against; `golden_sha256` pins the file's
    bytes, not what the parser does with them. Fix round 3, blocker 2:
    proved live by the whole-branch reviewer -- shrinking every acceptable
    span in `answer_spans()` to one character (which would collapse recall
    for most questions) left every other fingerprint field unchanged and
    the gate passed.
    """
    assert Path("src/clause/evaluation/golden.py") in cli.RETRIEVAL_CODE_PATHS


def test_retrieval_code_paths_includes_scoring_py() -> None:
    """`evaluate()`/`bucket_result()`/`chance_baseline_for_strategy()` moved
    out of `cli.py` into `clause.evaluation.scoring` specifically so they
    could be hashed -- `cli.py` itself is excluded (see
    `RETRIEVAL_CODE_PATHS`'s own comment). Fix round 3, blocker 3: proved
    live by the whole-branch reviewer -- hardcoding `recall_at_1=1.0` in
    what was then `cli._bucket_result` made every recall@1 in the report a
    lie and the gate still passed.
    """
    assert Path("src/clause/evaluation/scoring.py") in cli.RETRIEVAL_CODE_PATHS
