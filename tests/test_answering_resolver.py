from datetime import date

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from clause.answering.resolver import resolve_citations
from clause.answering.schema import AnswerDraft, Sentence, UnresolvableCitationError
from clause.db.schema import ChunkRow, DocumentRow
from clause.retrieve import Hit

pytestmark = pytest.mark.db


def _hit(doc_id: str, start: int, end: int, text: str, *, ordinal: int = 0) -> Hit:
    return Hit(
        doc_id=doc_id,
        strategy="structural",
        ordinal=ordinal,
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


def _seed_chunk(
    session: Session, doc_id: str, ordinal: int, start: int, end: int
) -> None:
    """A chunk row with the authoritative span -- what a hit's span is checked
    against. Always strategy="structural", matching `_hit`'s default."""
    session.add(
        ChunkRow(
            doc_id=doc_id,
            strategy="structural",
            ordinal=ordinal,
            char_start=start,
            char_end=end,
            text="x" * (end - start),
            source_url="https://example.invalid/1",
            effective_date=None,
            doc_type="notification",
            regulated_entity=[],
            created_at=sa.func.now(),
        )
    )
    session.flush()


def test_resolution_slices_the_stored_document_rather_than_trusting_the_hit(
    db_session: Session,
) -> None:
    """The citation's text comes from the corpus, never from the model or the hit."""
    _seed(db_session, "doc-a", "0123456789ABCDEFGHIJ")
    _seed_chunk(db_session, "doc-a", 0, 2, 6)
    hit = _hit("doc-a", 2, 6, "WRONG")  # hit text deliberately disagrees
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="s", citation_indices=(1,), factual=True),),
        hits=(hit,),
    )
    cites = resolve_citations(draft, db_session)
    assert cites[1].text == "2345"


def test_an_index_past_the_presented_hits_is_a_hard_failure(db_session: Session) -> None:
    _seed(db_session, "doc-a", "0123456789")
    hit = _hit("doc-a", 0, 4, "0123")
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="s", citation_indices=(2,), factual=True),),
        hits=(hit,),
    )
    with pytest.raises(UnresolvableCitationError, match="index 2"):
        resolve_citations(draft, db_session)


def test_a_missing_chunk_is_a_hard_failure(db_session: Session) -> None:
    """No document and no chunk row exist for this hit's key -- a chunk can never
    outlive its document (FK cascade), so a missing document surfaces here first."""
    hit = _hit("doc-absent", 0, 4, "0123")
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="s", citation_indices=(1,), factual=True),),
        hits=(hit,),
    )
    with pytest.raises(UnresolvableCitationError, match="doc-absent"):
        resolve_citations(draft, db_session)


def test_a_span_past_the_end_of_the_document_is_a_hard_failure(db_session: Session) -> None:
    """Re-ingest can shorten a document between retrieval and resolution."""
    _seed(db_session, "doc-a", "short")
    _seed_chunk(db_session, "doc-a", 0, 0, 500)
    hit = _hit("doc-a", 0, 500, "x" * 500)
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="s", citation_indices=(1,), factual=True),),
        hits=(hit,),
    )
    with pytest.raises(UnresolvableCitationError, match="out of range"):
        resolve_citations(draft, db_session)


def test_a_hit_span_stale_relative_to_the_chunk_table_is_a_hard_failure(
    db_session: Session,
) -> None:
    """Qdrant can be reindexed with different chunk boundaries while Postgres --
    the authority for what a chunk actually is -- still holds the old ones. The
    document is long enough that both the stale and the authoritative span are
    individually in-range, so only the cross-check against ChunkRow catches this.
    """
    _seed(db_session, "doc-a", "0" * 100)
    _seed_chunk(db_session, "doc-a", 0, 0, 4)  # authoritative span: [0:4]
    hit = _hit("doc-a", 10, 14, "0000")  # Qdrant payload disagrees: [10:14]
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="s", citation_indices=(1,), factual=True),),
        hits=(hit,),
    )
    with pytest.raises(UnresolvableCitationError, match="stale"):
        resolve_citations(draft, db_session)


def test_each_cited_index_appears_once_keyed_by_original_index(db_session: Session) -> None:
    _seed(db_session, "doc-a", "0123456789ABCDEFGHIJ")
    _seed_chunk(db_session, "doc-a", 0, 0, 4)
    _seed_chunk(db_session, "doc-a", 1, 4, 8)
    hits = (
        _hit("doc-a", 0, 4, "0123", ordinal=0),
        _hit("doc-a", 4, 8, "4567", ordinal=1),
    )
    draft = AnswerDraft(
        question="q",
        sentences=(
            Sentence(text="s1", citation_indices=(2,), factual=True),
            Sentence(text="s2", citation_indices=(1, 2), factual=True),
        ),
        hits=hits,
    )
    cites = resolve_citations(draft, db_session)
    assert set(cites) == {1, 2}
    assert (cites[1].char_start, cites[1].char_end) == (0, 4)
    assert (cites[2].char_start, cites[2].char_end) == (4, 8)


def test_a_draft_with_no_citations_resolves_to_nothing(db_session: Session) -> None:
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="s", citation_indices=(), factual=False),),
        hits=(),
    )
    assert resolve_citations(draft, db_session) == {}


def test_hits_are_read_from_the_draft_not_a_separate_argument(db_session: Session) -> None:
    """`resolve_citations` takes no `hits` parameter: the only hit list an index
    can mean is the one that travels with the draft. Two drafts with the same
    sentences but different `hits` must resolve to different documents."""
    _seed(db_session, "doc-a", "0123456789")
    _seed(db_session, "doc-b", "ABCDEFGHIJ")
    _seed_chunk(db_session, "doc-a", 0, 0, 4)
    _seed_chunk(db_session, "doc-b", 0, 0, 4)
    sentences = (Sentence(text="s", citation_indices=(1,), factual=True),)

    draft_a = AnswerDraft(
        question="q", sentences=sentences, hits=(_hit("doc-a", 0, 4, "0123"),)
    )
    draft_b = AnswerDraft(
        question="q", sentences=sentences, hits=(_hit("doc-b", 0, 4, "ABCD"),)
    )
    assert resolve_citations(draft_a, db_session)[1].doc_id == "doc-a"
    assert resolve_citations(draft_b, db_session)[1].doc_id == "doc-b"
