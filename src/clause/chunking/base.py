from typing import Protocol, runtime_checkable

from clause.models import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, document: Document) -> list[Chunk]: ...


def make_chunk(document: Document, strategy: str, ordinal: int, start: int, end: int) -> Chunk:
    """The only sanctioned way to build a Chunk.

    text is sliced from document.text here rather than passed in, so a chunker
    physically cannot emit a chunk whose text differs from its span. That is
    what makes the citation round-trip a structural property rather than a
    convention each chunker is trusted to follow.
    """
    if not 0 <= start < end <= len(document.text):
        raise ValueError(f"span ({start}, {end}) outside document of {len(document.text)} chars")
    return Chunk(
        doc_id=document.doc_id,
        strategy=strategy,
        ordinal=ordinal,
        char_start=start,
        char_end=end,
        text=document.text[start:end],
        source_url=document.url,
        effective_date=document.effective_date,
        doc_type=document.doc_type,
        regulated_entity=document.regulated_entity,
    )


class SliceIntegrityError(Exception):
    """A chunk's stored text does not match document.text[char_start:char_end].

    Raised explicitly, never via a bare ``assert``, because ``assert`` statements are
    stripped entirely under ``python -O``. CLAUDE.md's non-negotiable — "an answer
    containing a citation that does not resolve is a hard failure, never a warning" —
    must hold under an optimised interpreter too, not only in a debug build.
    """


def assert_slices(document: Document, chunks: list[Chunk]) -> None:
    """Assert the invariant. Used by tests and by the ingest CLI before writing.

    Raises SliceIntegrityError (not AssertionError) so this check survives `python -O`.
    """
    for c in chunks:
        actual = document.text[c.char_start : c.char_end]
        if actual != c.text:
            raise SliceIntegrityError(
                f"{c.doc_id}#{c.ordinal} ({c.strategy}): span [{c.char_start}:{c.char_end}] "
                f"yields {actual[:60]!r} but chunk stores {c.text[:60]!r}"
            )
