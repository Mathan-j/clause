import pytest
from sqlalchemy import select

from clause.db.schema import ChunkRow, DocumentRow

pytestmark = pytest.mark.db


def test_corpus_is_ingested(db_session, ingested) -> None:
    docs = db_session.scalars(select(DocumentRow)).all()
    assert len(docs) >= 50, "PROMPT.md Phase 1 requires at least 50 real documents"


def test_every_chunk_slice_roundtrips_against_stored_source(db_session, ingested) -> None:
    """PROMPT.md Phase 1 definition of done."""
    texts = {d.doc_id: d.text for d in db_session.scalars(select(DocumentRow)).all()}
    chunks = db_session.scalars(select(ChunkRow)).all()
    assert chunks, "no chunks were written"
    for c in chunks:
        assert texts[c.doc_id][c.char_start : c.char_end] == c.text, (
            f"{c.doc_id}#{c.ordinal} ({c.strategy}) span does not round-trip"
        )


def test_both_strategies_are_present(db_session, ingested) -> None:
    strategies = {r.strategy for r in db_session.scalars(select(ChunkRow)).all()}
    assert strategies == {"fixed_window", "structural"}


def test_every_chunk_carries_its_provenance(db_session, ingested) -> None:
    for c in db_session.scalars(select(ChunkRow)).all():
        assert c.source_url
        assert c.doc_type
        assert c.char_start < c.char_end
