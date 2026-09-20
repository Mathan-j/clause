from datetime import date, datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
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
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    text: Mapped[str] = mapped_column(Text)
    regulated_entity: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)


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
    regulated_entity: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # chunk_id is a serial PK and replace_chunks() is delete-then-insert, so
        # every ingest renumbers every chunk; (doc_id, strategy, ordinal) is the
        # stable natural key Phase 2 needs for "ground-truth chunk ids" in the
        # golden set. It also makes a failed/partial delete in replace_chunks()
        # error loudly (duplicate key) instead of silently doubling rows.
        UniqueConstraint(
            "doc_id", "strategy", "ordinal", name="uq_chunks_doc_strategy_ordinal"
        ),
    )


Index("ix_chunks_doc_strategy", ChunkRow.doc_id, ChunkRow.strategy)
