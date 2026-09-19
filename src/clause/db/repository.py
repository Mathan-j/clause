from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from clause.db.schema import ChunkRow, DocumentRow
from clause.models import Chunk, Document


def upsert_document(session: Session, document: Document) -> None:
    values = {
        "doc_id": document.doc_id,
        "rbi_id": document.rbi_id,
        "url": document.url,
        "circular_no": document.circular_no,
        "dept_ref": document.dept_ref,
        "title": document.title,
        "doc_type": document.doc_type,
        "published_date": document.published_date,
        "effective_date": document.effective_date,
        "sha256": document.sha256,
        "fetched_at": document.fetched_at,
        "text": document.text,
    }
    stmt = insert(DocumentRow).values(**values)
    stmt = stmt.on_conflict_do_update(index_elements=[DocumentRow.doc_id], set_=values)
    session.execute(stmt)


def replace_chunks(session: Session, doc_id: str, strategy: str, chunks: list[Chunk]) -> None:
    """Replace this document's chunks for one strategy, leaving others intact.

    Delete-then-insert in the caller's transaction, so a failure mid-ingest
    leaves no partial rows.
    """
    session.execute(
        delete(ChunkRow).where(ChunkRow.doc_id == doc_id, ChunkRow.strategy == strategy)
    )
    now = datetime.now(UTC)
    session.add_all(
        [
            ChunkRow(
                doc_id=c.doc_id,
                strategy=strategy,
                ordinal=c.ordinal,
                char_start=c.char_start,
                char_end=c.char_end,
                text=c.text,
                source_url=c.source_url,
                effective_date=c.effective_date,
                doc_type=c.doc_type,
                created_at=now,
            )
            for c in chunks
        ]
    )
