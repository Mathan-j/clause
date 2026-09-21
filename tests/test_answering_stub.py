from datetime import date

from clause.answering.base import Answerer, StubAnswerer
from clause.retrieve import Hit


def _hit(text: str = "Regulated entities shall verify identity.") -> Hit:
    return Hit(
        doc_id="d",
        strategy="structural",
        ordinal=0,
        char_start=0,
        char_end=len(text),
        score=0.8,
        text=text,
        source_url="u",
        published_date=date(2025, 1, 1),
        regulated_entity=(),
    )


def test_the_stub_satisfies_the_protocol() -> None:
    assert isinstance(StubAnswerer(), Answerer)


def test_the_stub_is_deterministic() -> None:
    """A harness whose stub moves is not a harness."""
    a, b = StubAnswerer(), StubAnswerer()
    assert a.answer("q", [_hit()]) == b.answer("q", [_hit()])


def test_the_stub_cites_the_top_hit() -> None:
    draft = StubAnswerer().answer("q", [_hit()])
    assert draft.sentences[0].citation_indices == (1,)
    assert draft.sentences[0].factual is True


def test_the_stub_produces_no_factual_sentence_without_hits() -> None:
    draft = StubAnswerer().answer("q", [])
    assert all(not s.factual for s in draft.sentences)


def test_the_stub_binds_the_draft_to_the_hits_it_was_given() -> None:
    """`AnswerDraft.hits` must be the exact hit list the indices were drafted
    against -- otherwise a citation index resolves against the wrong document.
    """
    hit = _hit()
    draft = StubAnswerer().answer("q", [hit])
    assert draft.hits == (hit,)


def test_the_stub_binds_empty_hits_in_the_no_hits_case() -> None:
    draft = StubAnswerer().answer("q", [])
    assert draft.hits == ()
