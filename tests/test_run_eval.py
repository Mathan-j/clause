"""Wiring tests for `clause.cli.run_eval`, fully mocked -- no live Qdrant, no
live Postgres, no model load.

`evaluate()` itself and the real span-scoring path are covered against a live
stack by `tests/test_eval_end_to_end.py`. These tests exist to pin down two
behaviours the task brief calls load-bearing and that a mocked-out acceptance
run could not prove by itself:

1. `run_eval` validates the golden set's spans against the stored corpus
   *before* touching the embedding model or Qdrant at all -- a golden set
   that no longer matches the corpus must abort loudly, never silently score
   against stale offsets.
2. `run_eval` writes to the paths it is given, not to a hardcoded
   `reports/eval.md` / `reports/eval.json` -- which is what lets the
   acceptance fixture run the real evaluation without touching the committed
   report.

Each test was verified to fail for the right reason: temporarily reordering
`run_eval` to construct `Encoder`/`QdrantClient` before `validate_spans` made
test 1 fail (the fake encoder's "must not be constructed" assertion fired);
temporarily hardcoding `write_report(report, DEFAULT_REPORT_MD,
DEFAULT_REPORT_JSON)` made test 2 fail (the custom `tmp_path` files were
never created). Both were reverted after observing the failure.
"""

import hashlib
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from clause import cli
from clause.config import Settings
from clause.db.schema import DocumentRow
from clause.evaluation.golden import Answer, GoldenQuestion, GoldenSetError
from clause.evaluation.report import BucketResult, StrategyResult


def _settings() -> Settings:
    return Settings(
        database_url="postgresql+psycopg://clause:clause@localhost:5434/clause_test",
        user_agent="clause-test/0.1 (+mailto:test@example.com)",
    )


class _StubResult:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def all(self) -> list[Any]:
        return self._items


class _StubSession:
    """Answers `session.scalars(...)` calls from a fixed queue, in call order.

    `run_eval` issues exactly `1 + len(STRATEGIES)` such calls: the documents
    query first, then one chunk query per strategy. Order is the only thing
    distinguishing them here, which is fine -- these tests exist to check
    call *ordering* and path plumbing, not query construction.
    """

    def __init__(self, responses: list[list[Any]]) -> None:
        self._responses = list(responses)

    def scalars(self, stmt: sa.Select[Any]) -> _StubResult:
        return _StubResult(self._responses.pop(0))

    def __enter__(self) -> "_StubSession":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _RefusingEncoder:
    """Stands in for `Encoder`; raises if ever constructed."""

    def __init__(self, model_name: str) -> None:
        raise AssertionError(
            "Encoder must not be constructed before span validation has succeeded"
        )


class _RefusingQdrantClient:
    """Stands in for `QdrantClient`; raises if ever constructed."""

    def __init__(self, url: str) -> None:
        raise AssertionError(
            "QdrantClient must not be constructed before span validation has succeeded"
        )


def test_span_validation_aborts_before_any_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    question = GoldenQuestion(
        qid="stale-001",
        question="What went stale?",
        bucket="definitional",
        answers=(
            Answer(doc_id="doc1", char_start=0, char_end=5, content_sha256="0" * 64),
        ),
        provenance="drafted",
        notes="",
    )
    # The stored document's real sha256 disagrees with the golden set's "0" * 64,
    # so validate_spans must raise before Encoder/QdrantClient are ever touched.
    doc = DocumentRow(doc_id="doc1", text="hello world", sha256="f" * 64)

    monkeypatch.setattr(cli, "load_golden", lambda path: [question])
    monkeypatch.setattr(cli, "make_engine", lambda url: object())
    monkeypatch.setattr(
        cli, "session_factory", lambda engine: lambda: _StubSession([[doc]])
    )
    monkeypatch.setattr(cli, "Encoder", _RefusingEncoder)
    monkeypatch.setattr(cli, "QdrantClient", _RefusingQdrantClient)

    with pytest.raises(GoldenSetError):
        cli.run_eval(settings=_settings())


def test_run_eval_writes_to_the_given_paths_not_the_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    doc_text = "hello world, this is the stored document text"
    doc = DocumentRow(
        doc_id="doc1", text=doc_text, sha256=hashlib.sha256(doc_text.encode()).hexdigest()
    )
    question = GoldenQuestion(
        qid="q-001",
        question="What is this?",
        bucket="definitional",
        answers=(Answer(doc_id="doc1", char_start=0, char_end=5, content_sha256=doc.sha256),),
        provenance="drafted",
        notes="",
    )

    class _FakeEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    class _FakeQdrantClient:
        def __init__(self, url: str) -> None:
            pass

    def _fake_evaluate(
        session: object,
        client: object,
        encoder: object,
        questions: object,
        *,
        depth: int = 10,
    ) -> list[StrategyResult]:
        bucket = BucketResult(
            n=1, recall_at_1=1.0, recall_at_5=1.0, recall_at_10=1.0, mrr_at_10=1.0
        )
        return [
            StrategyResult(
                strategy="fixed_window", overall=bucket, per_bucket={"definitional": bucket}
            ),
            StrategyResult(
                strategy="structural", overall=bucket, per_bucket={"definitional": bucket}
            ),
        ]

    monkeypatch.setattr(cli, "load_golden", lambda path: [question])
    monkeypatch.setattr(cli, "make_engine", lambda url: object())
    # documents query, then one (empty) chunk query per strategy
    monkeypatch.setattr(
        cli, "session_factory", lambda engine: lambda: _StubSession([[doc], [], []])
    )
    monkeypatch.setattr(cli, "Encoder", _FakeEncoder)
    monkeypatch.setattr(cli, "QdrantClient", _FakeQdrantClient)
    monkeypatch.setattr(cli, "evaluate", _fake_evaluate)

    md_path = tmp_path / "custom.md"
    json_path = tmp_path / "custom.json"
    default_md_existed = cli.DEFAULT_REPORT_MD.exists()
    default_json_existed = cli.DEFAULT_REPORT_JSON.exists()

    cli.run_eval(md_path=md_path, json_path=json_path, settings=_settings())

    assert md_path.exists()
    assert json_path.exists()
    # The default committed-artifact paths must be untouched by this call.
    assert cli.DEFAULT_REPORT_MD.exists() == default_md_existed
    assert cli.DEFAULT_REPORT_JSON.exists() == default_json_existed
