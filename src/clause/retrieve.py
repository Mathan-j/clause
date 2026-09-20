"""Dense retrieval with server-side metadata filters.

`doc_type` is deliberately not offered as a filter. The predecessor spec's
section 10 records the measurement: zero `circular` labels across the corpus and
near-identical documents split both ways. Exposing a filter on a field known to
be noise would invite Phase 3 to filter on it.

Filters run inside Qdrant (`query_filter=`), not on the client after the top-k
comes back. Post-filtering a top-10 would silently return fewer than ten
results for a filtered query and make recall@10 mean something different for
filtered queries than unfiltered ones -- the top-k has to be drawn from the
filtered set, not filtered out of an unfiltered top-k.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from qdrant_client import QdrantClient, models

from clause.embed import Encoder
from clause.evaluation.metrics import RETRIEVAL_DEPTH, Retrieved
from clause.index import collection_name


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved chunk.

    Carries `source_url`, `published_date` and `regulated_entity` alongside the
    span fields: CLAUDE.md requires every chunk to carry its own provenance so
    that a citation never needs a second query to say where it came from. Task 7
    already writes these into the Qdrant payload and `search()` already fetches
    the whole payload (`with_payload=True`) -- dropping them here would throw
    away data already on the wire for zero runtime saving.
    """

    doc_id: str
    strategy: str
    ordinal: int
    char_start: int
    char_end: int
    score: float
    text: str
    source_url: str
    published_date: date
    regulated_entity: tuple[str, ...]

    def as_retrieved(self) -> Retrieved:
        return Retrieved(
            doc_id=self.doc_id,
            char_start=self.char_start,
            char_end=self.char_end,
            score=self.score,
        )


def _build_filter(
    published_after: date | None,
    published_before: date | None,
    entities: Sequence[str] | None,
) -> models.Filter | None:
    conditions: list[models.Condition] = []
    if published_after is not None or published_before is not None:
        conditions.append(
            models.FieldCondition(
                key="published_date",
                range=models.DatetimeRange(
                    gte=published_after,
                    lte=published_before,
                ),
            )
        )
    if entities:
        conditions.append(
            models.FieldCondition(
                key="regulated_entity", match=models.MatchAny(any=list(entities))
            )
        )
    return models.Filter(must=conditions) if conditions else None


def search(
    client: QdrantClient,
    encoder: Encoder,
    strategy: str,
    query: str,
    *,
    limit: int = RETRIEVAL_DEPTH,
    published_after: date | None = None,
    published_before: date | None = None,
    entities: Sequence[str] | None = None,
) -> list[Hit]:
    """Top `limit` chunks of one strategy, filtered server-side.

    `strategy` selects the collection (`clause_<strategy>`) -- searching
    "structural" only ever queries `clause_structural`, never touching
    `clause_fixed_window`. The two collections hold different ANN indexes over
    different chunk sets, so a query against one strategy must not leak results
    from the other.
    """
    vector = encoder.encode([query])[0]
    response = client.query_points(
        collection_name=collection_name(strategy),
        query=vector,
        limit=limit,
        query_filter=_build_filter(published_after, published_before, entities),
        with_payload=True,
    )
    hits: list[Hit] = []
    for point in response.points:
        payload = point.payload or {}
        hits.append(
            Hit(
                doc_id=payload["doc_id"],
                strategy=payload["strategy"],
                ordinal=int(payload["ordinal"]),
                char_start=int(payload["char_start"]),
                char_end=int(payload["char_end"]),
                score=float(point.score),
                text=payload["text"],
                source_url=payload["source_url"],
                published_date=date.fromisoformat(payload["published_date"]),
                regulated_entity=tuple(payload.get("regulated_entity") or ()),
            )
        )
    return hits
