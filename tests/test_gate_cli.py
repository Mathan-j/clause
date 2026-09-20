"""CLI-level tests for `clause.cli`'s `gate` subcommand: `_gate_chunk_counts`'s
empty-database fallback and the notes it prints, `run_gate`'s no-baseline
note, and `main(["gate", ...])`'s exit codes.

`clause.evaluation.gate.check` itself (the pure comparison) is covered by
tests/test_gate.py without any database. These tests use the real Postgres
the rest of the suite uses (`db_session`), because the whole point of the
behaviour under test is what the gate does with a *real* (here: empty)
database -- a mock would just assert the mock was called.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from clause import cli
from clause.config import Settings
from clause.db.schema import ChunkRow, DocumentRow
from clause.evaluation.gate import GateFailure
from clause.evaluation.metrics import RETRIEVAL_DEPTH


def _settings() -> Settings:
    return Settings(
        database_url="postgresql+psycopg://clause:clause@localhost:5434/clause_test",
        user_agent="clause-test/0.1 (+mailto:test@example.com)",
    )


def _real_fingerprint(chunk_counts: dict[str, int]) -> dict[str, Any]:
    """A fingerprint dict whose every field except `chunk_counts` matches what
    `_rebuild_tree_fingerprint` will compute for real, so a test can control
    only the one field it cares about without every other field also
    (correctly) showing up as a mismatch.
    """
    settings = _settings()
    return {
        "manifest_sha256": cli._sha256_file(cli.MANIFEST_PATH),
        "chunk_counts": chunk_counts,
        "embedding_model": settings.embedding_model,
        "retrieval_depth": RETRIEVAL_DEPTH,
        "golden_path": cli.GOLDEN_PATH.as_posix(),
        "golden_sha256": cli._sha256_file(cli.GOLDEN_PATH),
        "git_commit": cli._git_commit(),
    }


def _write_report(path: Path, *, chunk_counts: dict[str, int], strategies: dict[str, Any]) -> None:
    report = {
        "fingerprint": _real_fingerprint(chunk_counts),
        "golden_provenance": {"drafted": 1},
        "strategies": strategies,
    }
    path.write_text(json.dumps(report), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# _gate_chunk_counts: the empty-database fallback and its notes
# ---------------------------------------------------------------------------


def test_gate_chunk_counts_falls_back_to_report_when_database_is_empty(
    db_session: Session,
) -> None:
    """`db_session` gives a real, freshly migrated Postgres schema with zero
    rows -- exactly CI's situation after `alembic upgrade head` with no
    `make ingest` ever having run.
    """
    report = {"fingerprint": {"chunk_counts": {"fixed_window": 318, "structural": 336}}}

    counts, notes = cli._gate_chunk_counts(db_session, report)

    assert counts == {"fixed_window": 318, "structural": 336}
    assert notes, "an unfireable check must announce that it did not run"
    assert any("chunk-count fingerprint check skipped" in n for n in notes)
    assert any("cannot fail" in n for n in notes)


def test_gate_chunk_counts_uses_the_live_database_when_populated(db_session: Session) -> None:
    """The converse of the fallback test: when the database genuinely has
    chunks, the gate must count them for real rather than always deferring
    to the report -- otherwise the fallback would have swallowed the whole
    check, not just the empty case.
    """
    doc = DocumentRow(
        doc_id="d1",
        rbi_id=1,
        url="https://example.test/d1",
        circular_no="RBI/2026-27/1",
        dept_ref="DOR.X.1",
        title="t",
        doc_type="circular",
        published_date=date(2026, 9, 18),
        effective_date=None,
        sha256="a" * 64,
        fetched_at=datetime(2026, 9, 19, tzinfo=UTC),
        text="hello world",
        regulated_entity=[],
    )
    db_session.add(doc)
    db_session.flush()
    now = datetime(2026, 9, 19, tzinfo=UTC)
    db_session.add_all(
        [
            ChunkRow(
                doc_id="d1",
                strategy="fixed_window",
                ordinal=0,
                char_start=0,
                char_end=5,
                text="hello",
                source_url=doc.url,
                effective_date=None,
                doc_type=doc.doc_type,
                regulated_entity=[],
                created_at=now,
            ),
            ChunkRow(
                doc_id="d1",
                strategy="structural",
                ordinal=0,
                char_start=0,
                char_end=5,
                text="hello",
                source_url=doc.url,
                effective_date=None,
                doc_type=doc.doc_type,
                regulated_entity=[],
                created_at=now,
            ),
            ChunkRow(
                doc_id="d1",
                strategy="structural",
                ordinal=1,
                char_start=6,
                char_end=11,
                text="world",
                source_url=doc.url,
                effective_date=None,
                doc_type=doc.doc_type,
                regulated_entity=[],
                created_at=now,
            ),
        ]
    )
    db_session.commit()

    report = {"fingerprint": {"chunk_counts": {"fixed_window": 999, "structural": 999}}}
    counts, notes = cli._gate_chunk_counts(db_session, report)

    assert counts == {"fixed_window": 1, "structural": 2}
    assert notes == []


# ---------------------------------------------------------------------------
# run_gate: the no-baseline note, and pass/fail behaviour end to end
# ---------------------------------------------------------------------------


def test_run_gate_notes_and_passes_when_baseline_is_missing(
    db_session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "eval.json"
    _write_report(
        report_path,
        chunk_counts={"fixed_window": 1, "structural": 1},
        strategies={},
    )
    missing_baseline = tmp_path / "baseline.json"
    assert not missing_baseline.exists()

    cli.run_gate(report_path=report_path, baseline_path=missing_baseline, settings=_settings())

    err = capsys.readouterr().err
    assert "no baseline" in err.lower()
    assert "nothing to regress against" in err.lower()


def test_run_gate_raises_and_prints_on_recall_regression(
    db_session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "eval.json"
    _write_report(
        report_path,
        chunk_counts={"fixed_window": 1, "structural": 1},
        strategies={"structural": {"overall": {"recall_at_5": 0.10}}},
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"structural": {"recall_at_5": 0.75}}), encoding="utf-8", newline="\n"
    )

    with pytest.raises(GateFailure):
        cli.run_gate(report_path=report_path, baseline_path=baseline_path, settings=_settings())

    err = capsys.readouterr().err
    assert "GATE FAILURE" in err
    assert "structural" in err and "recall@5" in err


def test_run_gate_passes_clean(
    db_session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "eval.json"
    _write_report(
        report_path,
        chunk_counts={"fixed_window": 1, "structural": 1},
        strategies={"structural": {"overall": {"recall_at_5": 0.90}}},
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"structural": {"recall_at_5": 0.75}}), encoding="utf-8", newline="\n"
    )

    cli.run_gate(report_path=report_path, baseline_path=baseline_path, settings=_settings())

    assert "GATE FAILURE" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main(): exit codes for the `gate` subcommand
# ---------------------------------------------------------------------------


def test_main_gate_exits_zero_on_a_clean_pass(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "eval.json"
    _write_report(
        report_path,
        chunk_counts={"fixed_window": 1, "structural": 1},
        strategies={"structural": {"overall": {"recall_at_5": 0.90}}},
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"structural": {"recall_at_5": 0.75}}), encoding="utf-8", newline="\n"
    )
    monkeypatch.setattr(cli, "get_settings", _settings)

    exit_code = cli.main(["gate", "--report", str(report_path), "--baseline", str(baseline_path)])

    assert exit_code == 0


def test_main_gate_exits_one_on_a_regression(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "eval.json"
    _write_report(
        report_path,
        chunk_counts={"fixed_window": 1, "structural": 1},
        strategies={"structural": {"overall": {"recall_at_5": 0.01}}},
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"structural": {"recall_at_5": 0.75}}), encoding="utf-8", newline="\n"
    )
    monkeypatch.setattr(cli, "get_settings", _settings)

    exit_code = cli.main(["gate", "--report", str(report_path), "--baseline", str(baseline_path)])

    assert exit_code == 1


def test_main_gate_exits_one_on_a_stale_fingerprint(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "eval.json"
    _write_report(
        report_path,
        chunk_counts={"fixed_window": 1, "structural": 1},
        strategies={"structural": {"overall": {"recall_at_5": 0.90}}},
    )
    # Corrupt one fingerprint field the empty-database fallback cannot rescue.
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["fingerprint"]["golden_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8", newline="\n")

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"structural": {"recall_at_5": 0.75}}), encoding="utf-8", newline="\n"
    )
    monkeypatch.setattr(cli, "get_settings", _settings)

    exit_code = cli.main(["gate", "--report", str(report_path), "--baseline", str(baseline_path)])

    assert exit_code == 1
