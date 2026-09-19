from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from clause.chunking.fixed import FixedWindowChunker
from clause.db.repository import replace_chunks, upsert_document
from clause.db.schema import ChunkRow, DocumentRow
from clause.models import Document

pytestmark = pytest.mark.db

DOC = Document(
    doc_id="rbi-1", rbi_id=1, url="https://example.test/1",
    circular_no="RBI/2026-27/1", dept_ref="DOR.X.1", title="t",
    doc_type="circular", published_date=date(2026, 9, 18), effective_date=None,
    sha256="a" * 64, fetched_at=datetime(2026, 9, 19, tzinfo=UTC), text="word " * 400,
)


def test_upsert_is_idempotent(db_session) -> None:
    upsert_document(db_session, DOC)
    upsert_document(db_session, DOC)
    db_session.commit()
    assert len(db_session.scalars(select(DocumentRow)).all()) == 1


def test_replace_chunks_swaps_only_that_strategy(db_session) -> None:
    upsert_document(db_session, DOC)
    fixed = FixedWindowChunker(200, 50).chunk(DOC)
    replace_chunks(db_session, DOC.doc_id, "fixed_window", fixed)
    replace_chunks(db_session, DOC.doc_id, "structural", fixed[:2])
    db_session.commit()

    rows = db_session.scalars(select(ChunkRow)).all()
    assert {r.strategy for r in rows} == {"fixed_window", "structural"}

    replace_chunks(db_session, DOC.doc_id, "fixed_window", fixed[:1])
    db_session.commit()
    rows = db_session.scalars(select(ChunkRow)).all()
    assert len([r for r in rows if r.strategy == "fixed_window"]) == 1
    assert len([r for r in rows if r.strategy == "structural"]) == 2


def test_chunk_rows_store_their_own_provenance(db_session) -> None:
    upsert_document(db_session, DOC)
    replace_chunks(db_session, DOC.doc_id, "fixed_window", FixedWindowChunker(200, 50).chunk(DOC))
    db_session.commit()
    row = db_session.scalars(select(ChunkRow)).first()
    assert row is not None
    assert row.source_url == DOC.url
    assert row.doc_type == DOC.doc_type
