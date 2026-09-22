from pathlib import Path

import pytest

from clause.answering.base import StubAnswerer
from clause.answering.schema import Answer, Refusal, RefusalReason
from clause.evaluation.answer_run import DEFAULT_ANSWERS_JSON, DEFAULT_ANSWERS_MD, answer_one

pytestmark = pytest.mark.db


def test_a_weak_top_score_refuses_before_any_generation(db_session, monkeypatch) -> None:
    """The refusal gate must not pay for a model call it does not need."""
    called = False

    class ExplodingAnswerer:
        def answer(self, question, hits):
            nonlocal called
            called = True
            raise AssertionError("generation must not run on a refused question")

    result = answer_one("q", [], ExplodingAnswerer(), db_session, threshold=0.35)
    assert isinstance(result, Refusal)
    assert result.reason == RefusalReason.NO_CANDIDATES
    assert called is False


def test_a_strong_hit_produces_an_answer_whose_citations_resolved(
    db_session, seeded_hit
) -> None:
    result = answer_one("q", [seeded_hit], StubAnswerer(), db_session, threshold=0.0)
    assert isinstance(result, Answer)
    assert result.citations
    assert result.citations[0].text


def test_answers_never_carry_an_unresolved_citation(db_session, seeded_hit) -> None:
    """The DoD, asserted directly: no response passes with an unresolvable citation."""
    result = answer_one("q", [seeded_hit], StubAnswerer(), db_session, threshold=0.0)
    assert isinstance(result, Answer)
    for c in result.citations:
        assert c.char_start < c.char_end
        assert len(c.text) == c.char_end - c.char_start


def test_run_writes_to_the_given_paths_not_the_defaults(tmp_path: Path) -> None:
    assert tmp_path / "answers.md" != DEFAULT_ANSWERS_MD
    assert tmp_path / "answers.json" != DEFAULT_ANSWERS_JSON
