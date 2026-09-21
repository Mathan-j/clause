"""Turn cited indices into citations, by slicing the corpus.

The model never supplies citation text. It supplies an index into the hits it
was shown; this module reads that hit's `(doc_id, char_start, char_end)`, loads
the stored document, and slices it. A citation's text is therefore derived from
the corpus by construction and cannot disagree with it -- the same reasoning as
`make_chunk`, which slices rather than accepting text.

The span itself is not trusted from the hit either. A `Hit`'s `(char_start,
char_end)` comes off the Qdrant payload, and Qdrant can be reindexed with
different chunk boundaries while Postgres -- the authority for what a chunk
actually is -- still holds the old ones. If that happened, the hit's span could
still land inside the right document, at the right length, and slice cleanly:
sound-looking, but pointing at a span the retriever never actually scored. So
every cited hit is joined against `ChunkRow` by its natural key `(doc_id,
strategy, ordinal)`, and a hit whose span disagrees with the stored chunk's is
treated as a hard failure -- the same kind of staleness this module already
raises on when a document changes shape underneath a hit.
"""

from collections.abc import Mapping

import sqlalchemy as sa
from sqlalchemy.orm import Session

from clause.answering.schema import AnswerDraft, Citation, UnresolvableCitationError
from clause.db.schema import ChunkRow, DocumentRow


def _cited_indices(draft: AnswerDraft) -> list[int]:
    """Every cited index, de-duplicated, in ascending order."""
    return sorted({i for sentence in draft.sentences for i in sentence.citation_indices})


def resolve_citations(draft: AnswerDraft, session: Session) -> Mapping[int, Citation]:
    """Resolve every cited index to a Citation, or raise.

    Reads `draft.hits` -- the same hit list the draft's indices were drafted
    against -- rather than taking a second `hits` argument that could disagree
    with it. Returns a mapping keyed by the **original hit index** (1-based,
    exactly as it appears in a `Sentence.citation_indices`), not a positional
    tuple -- a tuple of only the cited citations would silently change meaning
    depending on which indices happened to be cited, which is precisely the
    ambiguity that must not exist between this function and `validate.enforce`,
    which consumes the result.

    Raises `UnresolvableCitationError` when an index names no presented hit, when
    the hit's chunk is absent from Postgres or its span disagrees with the hit's
    (the retrieval index is stale), when the hit's document is absent from the
    corpus, or when its span no longer fits inside that document. Each is a
    genuine post-retrieval failure: the corpus can change between the search and
    the answer.
    """
    hits = draft.hits
    indices = _cited_indices(draft)
    if not indices:
        return {}

    for index in indices:
        if index > len(hits):
            raise UnresolvableCitationError(
                f"citation index {index} names no presented hit "
                f"(only {len(hits)} were shown for question {draft.question!r})"
            )

    wanted = {hits[i - 1].doc_id for i in indices}
    docs = {
        d.doc_id: d
        for d in session.scalars(
            sa.select(DocumentRow).where(DocumentRow.doc_id.in_(wanted))
        ).all()
    }
    chunks = {
        (c.doc_id, c.strategy, c.ordinal): c
        for c in session.scalars(
            sa.select(ChunkRow).where(ChunkRow.doc_id.in_(wanted))
        ).all()
    }

    resolved: dict[int, Citation] = {}
    for index in indices:
        hit = hits[index - 1]
        chunk = chunks.get((hit.doc_id, hit.strategy, hit.ordinal))
        if chunk is None:
            raise UnresolvableCitationError(
                f"citation index {index} names chunk ({hit.doc_id!r}, "
                f"{hit.strategy!r}, ordinal {hit.ordinal}), which is not in Postgres"
            )
        if (chunk.char_start, chunk.char_end) != (hit.char_start, hit.char_end):
            raise UnresolvableCitationError(
                f"citation index {index}'s span [{hit.char_start}:{hit.char_end}] "
                f"disagrees with the stored chunk's span "
                f"[{chunk.char_start}:{chunk.char_end}] for {hit.doc_id!r} ordinal "
                f"{hit.ordinal} -- the retrieval index is stale relative to Postgres"
            )
        document = docs.get(hit.doc_id)
        if document is None:
            raise UnresolvableCitationError(
                f"citation index {index} points at document {hit.doc_id!r}, "
                "which is not in the corpus"
            )
        if chunk.char_end > len(document.text):
            raise UnresolvableCitationError(
                f"citation index {index} span [{chunk.char_start}:{chunk.char_end}] "
                f"is out of range for {hit.doc_id!r}, which holds "
                f"{len(document.text)} characters -- the document changed since "
                "it was retrieved"
            )
        resolved[index] = Citation(
            doc_id=hit.doc_id,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            source_url=document.url,
            published_date=document.published_date,
            doc_type=document.doc_type,
            text=document.text[chunk.char_start : chunk.char_end],
        )
    return resolved
