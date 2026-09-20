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
    """AND-compose a date-range condition and an entity condition, as given.

    `entities=[]` is treated exactly like `entities=None` -- no entity
    constraint at all, not "match none of the zero entities named". Both
    readings are defensible; this one is pinned deliberately because it is the
    less surprising of the two ("nothing selected" reads as "no filter"
    upstream), and because Phase 3 will build queries from exactly this kind
    of caller input, where the distinction between "omitted" and "empty"
    needs to be a stated decision, not an accident of `if entities:`.
    """
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


# PLR0913: this signature is the task's mandated public interface (client,
# encoder, strategy, query, plus four keyword-only filter/limit params), not a
# sign of a function doing too much. Unlike "tests/**" = ["PLR2004"] in
# pyproject.toml -- a durable property of every present and future test file --
# "one mandated signature" is not a property of everything that will ever live
# in this module, so the exemption is pinned to this one definition rather than
# to the whole file: a later `search_multi` or `rerank_within` added here would
# not silently inherit it.
def search(  # noqa: PLR0913
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
    name = collection_name(strategy)
    vector = encoder.encode([query])[0]
    response = client.query_points(
        collection_name=name,
        query=vector,
        limit=limit,
        query_filter=_build_filter(published_after, published_before, entities),
        with_payload=True,
    )
    return [_hit_from_point(point, name) for point in response.points]


def _hit_from_point(point: models.ScoredPoint, collection: str) -> Hit:
    """Build a `Hit` from one scored point, or fail with enough context to find it.

    A point missing a required payload field is kept as a hard failure --
    CLAUDE.md treats a citation that cannot state its own provenance as
    exactly that -- but a bare `KeyError: 'source_url'` gives whoever hits it
    no way to find the offending point. Naming the point id and collection
    turns "which point in which collection was malformed" from a manual
    Qdrant query into a one-line error message.
    """
    payload = point.payload or {}
    try:
        return Hit(
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
    except KeyError as exc:
        raise KeyError(
            f"point {point.id!r} in collection {collection!r} is missing required "
            f"payload field {exc.args[0]!r}; cannot state this chunk's provenance"
        ) from exc
