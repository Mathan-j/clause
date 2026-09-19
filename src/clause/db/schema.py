from datetime import date, datetime

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    rbi_id: Mapped[int] = mapped_column(Integer)
    url: Mapped[str] = mapped_column(Text)
    circular_no: Mapped[str] = mapped_column(String(64))
    dept_ref: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(Text)
    doc_type: Mapped[str] = mapped_column(String(32))
    published_date: Mapped[date]
    effective_date: Mapped[date | None]
    sha256: Mapped[str] = mapped_column(String(64))
    fetched_at: Mapped[datetime]
    text: Mapped[str] = mapped_column(Text)


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id", ondelete="CASCADE"))
    strategy: Mapped[str] = mapped_column(String(32))
    ordinal: Mapped[int] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    # Denormalised from documents so a retrieved chunk carries its own
    # provenance. CLAUDE.md requires every chunk to store these; a citation
    # that needs a second query to say where it came from can be separated
    # from its source.
    source_url: Mapped[str] = mapped_column(Text)
    effective_date: Mapped[date | None]
    doc_type: Mapped[str] = mapped_column(String(32))

    created_at: Mapped[datetime]


Index("ix_chunks_doc_strategy", ChunkRow.doc_id, ChunkRow.strategy)
