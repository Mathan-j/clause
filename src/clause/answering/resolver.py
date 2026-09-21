"""Turn cited indices into citations, by slicing the corpus.

The model never supplies citation text. It supplies an index into the hits it
was shown; this module reads that hit's `(doc_id, char_start, char_end)`, loads
the stored document, and slices it. A citation's text is therefore derived from
the corpus by construction and cannot disagree with it -- the same reasoning as
`make_chunk`, which slices rather than accepting text.
"""

from collections.abc import Mapping, Sequence

import sqlalchemy as sa
from sqlalchemy.orm import Session

from clause.answering.schema import AnswerDraft, Citation, UnresolvableCitationError
from clause.db.schema import DocumentRow
from clause.retrieve import Hit


def _cited_indices(draft: AnswerDraft) -> list[int]:
    """Every cited index, de-duplicated, in first-cited order."""
    seen: dict[int, None] = {}
    for sentence in draft.sentences:
        for index in sentence.citation_indices:
            seen.setdefault(index, None)
    return sorted(seen)


def resolve_citations(
    draft: AnswerDraft, hits: Sequence[Hit], session: Session
) -> Mapping[int, Citation]:
    """Resolve every cited index to a Citation, or raise.

    Returns a mapping keyed by the **original hit index** (1-based, exactly as it
    appears in a `Sentence.citation_indices`), not a positional tuple -- a tuple of
    only the cited citations would silently change meaning depending on which
    indices happened to be cited, which is precisely the ambiguity that must not
    exist between this function and `validate.enforce`, which consumes the result.

    Raises `UnresolvableCitationError` when an index names no presented hit, when
    the hit's document is absent from the corpus, or when its span no longer fits
    inside that document. Each is a genuine post-retrieval failure: the corpus can
    change between the search and the answer.
    """
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

    resolved: dict[int, Citation] = {}
    for index in indices:
        hit = hits[index - 1]
        document = docs.get(hit.doc_id)
        if document is None:
            raise UnresolvableCitationError(
                f"citation index {index} points at document {hit.doc_id!r}, "
                "which is not in the corpus"
            )
        if hit.char_end > len(document.text):
            raise UnresolvableCitationError(
                f"citation index {index} span [{hit.char_start}:{hit.char_end}] is "
                f"out of range for {hit.doc_id!r}, which holds {len(document.text)} "
                "characters -- the document changed since it was retrieved"
            )
        resolved[index] = Citation(
            doc_id=hit.doc_id,
            char_start=hit.char_start,
            char_end=hit.char_end,
            source_url=document.url,
            published_date=document.published_date,
            doc_type=document.doc_type,
            text=document.text[hit.char_start : hit.char_end],
        )
    return resolved
