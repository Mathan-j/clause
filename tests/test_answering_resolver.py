from datetime import date

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from clause.answering.resolver import resolve_citations
from clause.answering.schema import AnswerDraft, Sentence, UnresolvableCitationError
from clause.db.schema import DocumentRow
from clause.retrieve import Hit

pytestmark = pytest.mark.db


def _hit(doc_id: str, start: int, end: int, text: str) -> Hit:
    return Hit(
        doc_id=doc_id,
        strategy="structural",
        ordinal=0,
        char_start=start,
        char_end=end,
        score=0.5,
        text=text,
        source_url="https://example.invalid/1",
        published_date=date(2025, 1, 1),
        regulated_entity=(),
    )


def _seed(session: Session, doc_id: str, text: str) -> None:
    session.add(
        DocumentRow(
            doc_id=doc_id,
            rbi_id=1,
            url="https://example.invalid/1",
            circular_no="C/1",
            dept_ref="D/1",
            title="t",
            doc_type="notification",
            published_date=date(2025, 1, 1),
            effective_date=None,
            sha256="0" * 64,
            fetched_at=sa.func.now(),
            text=text,
            regulated_entity=[],
        )
    )
    session.flush()


def test_resolution_slices_the_stored_document_rather_than_trusting_the_hit(
    db_session: Session,
) -> None:
    """The citation's text comes from the corpus, never from the model or the hit."""
    _seed(db_session, "doc-a", "0123456789ABCDEFGHIJ")
    hit = _hit("doc-a", 2, 6, "WRONG")  # hit text deliberately disagrees
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="s", citation_indices=(1,), factual=True),)
    )
    cites = resolve_citations(draft, [hit], db_session)
    assert cites[0].text == "2345"


def test_an_index_past_the_presented_hits_is_a_hard_failure(db_session: Session) -> None:
    _seed(db_session, "doc-a", "0123456789")
    hit = _hit("doc-a", 0, 4, "0123")
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="s", citation_indices=(2,), factual=True),)
    )
    with pytest.raises(UnresolvableCitationError, match="index 2"):
        resolve_citations(draft, [hit], db_session)


def test_a_missing_document_is_a_hard_failure(db_session: Session) -> None:
    hit = _hit("doc-absent", 0, 4, "0123")
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="s", citation_indices=(1,), factual=True),)
    )
    with pytest.raises(UnresolvableCitationError, match="doc-absent"):
        resolve_citations(draft, [hit], db_session)


def test_a_span_past_the_end_of_the_document_is_a_hard_failure(db_session: Session) -> None:
    """Re-ingest can shorten a document between retrieval and resolution."""
    _seed(db_session, "doc-a", "short")
    hit = _hit("doc-a", 0, 500, "x" * 500)
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="s", citation_indices=(1,), factual=True),)
    )
    with pytest.raises(UnresolvableCitationError, match="out of range"):
        resolve_citations(draft, [hit], db_session)


def test_each_cited_index_appears_once_in_order(db_session: Session) -> None:
    _seed(db_session, "doc-a", "0123456789ABCDEFGHIJ")
    hits = [_hit("doc-a", 0, 4, "0123"), _hit("doc-a", 4, 8, "4567")]
    draft = AnswerDraft(
        question="q",
        sentences=(
            Sentence(text="s1", citation_indices=(2,), factual=True),
            Sentence(text="s2", citation_indices=(1, 2), factual=True),
        ),
    )
    cites = resolve_citations(draft, hits, db_session)
    assert [(c.char_start, c.char_end) for c in cites] == [(0, 4), (4, 8)]


def test_a_draft_with_no_citations_resolves_to_nothing(db_session: Session) -> None:
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="s", citation_indices=(), factual=False),)
    )
    assert resolve_citations(draft, [], db_session) == ()
