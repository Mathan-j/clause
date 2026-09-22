from datetime import date
from pathlib import Path

import pytest

from clause.answering.base import Answerer
from clause.answering.grammar import build_grammar
from clause.answering.llm import DEFAULT_MODEL_FILENAME, LlamaAnswerer
from clause.answering.schema import ModelNotAvailableError
from clause.retrieve import Hit


def test_the_grammar_names_every_valid_index_and_no_other() -> None:
    g = build_grammar(3)
    assert '"1"' in g and '"2"' in g and '"3"' in g
    assert '"4"' not in g


def test_the_grammar_rejects_a_zero_hit_request() -> None:
    """With nothing retrieved there is no citable index, so generation must not start."""
    with pytest.raises(ValueError, match="at least one"):
        build_grammar(0)


def test_an_absent_model_fails_loudly_rather_than_downloading(tmp_path: Path) -> None:
    missing = tmp_path / "not-a-model.gguf"
    with pytest.raises(ModelNotAvailableError, match="not present"):
        LlamaAnswerer(model_path=missing)


def test_the_error_names_the_documented_filename(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError, match=DEFAULT_MODEL_FILENAME):
        LlamaAnswerer(model_path=tmp_path / "not-a-model.gguf")


@pytest.mark.llm
def test_generation_produces_a_draft_whose_indices_are_in_range(answer_model: Path) -> None:
    hits = [
        Hit(
            doc_id="d",
            strategy="structural",
            ordinal=0,
            char_start=0,
            char_end=52,
            score=0.9,
            text="Regulated entities shall verify customer identity.",
            source_url="u",
            published_date=date(2025, 1, 1),
            regulated_entity=(),
        )
    ]
    answerer = LlamaAnswerer(model_path=answer_model)
    assert isinstance(answerer, Answerer)
    draft = answerer.answer("What must regulated entities verify?", hits)
    assert draft.sentences
    for sentence in draft.sentences:
        for index in sentence.citation_indices:
            assert 1 <= index <= len(hits)
