import dataclasses
from datetime import UTC, date, datetime, timedelta, timezone

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


def test_upsert_updates_an_existing_document(db_session) -> None:
    """A sha256 mismatch means the content changed; re-ingest must overwrite the row,
    not merely avoid duplicating it.
    """
    upsert_document(db_session, DOC)
    db_session.commit()

    revised = dataclasses.replace(
        DOC, title="Revised title", text="different text " * 50, sha256="b" * 64
    )
    upsert_document(db_session, revised)
    db_session.commit()

    rows = db_session.scalars(select(DocumentRow)).all()
    assert len(rows) == 1
    assert rows[0].title == "Revised title"
    assert rows[0].sha256 == "b" * 64
    assert rows[0].text == revised.text


def test_fetched_at_round_trips_with_offset_intact(db_session) -> None:
    """DateTime(timezone=True) must not silently drop the offset a tz-aware value
    carries; correctness must not depend on the session timezone happening to be UTC.
    """
    ist = timezone(timedelta(hours=5, minutes=30))
    aware = dataclasses.replace(DOC, fetched_at=datetime(2026, 9, 19, 18, 30, tzinfo=ist))
    upsert_document(db_session, aware)
    db_session.commit()

    row = db_session.get(DocumentRow, DOC.doc_id)
    assert row is not None
    assert row.fetched_at.tzinfo is not None
    assert row.fetched_at.astimezone(UTC) == aware.fetched_at.astimezone(UTC)


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
    doc = dataclasses.replace(DOC, regulated_entity=("Commercial Banks", "Payment Banks"))
    upsert_document(db_session, doc)
    replace_chunks(db_session, doc.doc_id, "fixed_window", FixedWindowChunker(200, 50).chunk(doc))
    db_session.commit()
    row = db_session.scalars(select(ChunkRow)).first()
    assert row is not None
    assert row.source_url == doc.url
    assert row.doc_type == doc.doc_type
    assert row.regulated_entity == list(doc.regulated_entity)
