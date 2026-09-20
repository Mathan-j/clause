"""Qdrant collection lifecycle and chunk indexing.

One collection per chunking strategy, not one shared collection with a strategy
filter. Approximate-nearest-neighbour recall depends on what else is in the
index, so mixing both strategies would make each strategy's number a function of
the other -- destroying the comparison the two strategies exist to enable.
"""

from collections.abc import Iterable, Iterator, Sequence

import sqlalchemy as sa
from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session

from clause.db.schema import ChunkRow, DocumentRow
from clause.embed import Encoder

COLLECTION_PREFIX = "clause_"
DEFAULT_BATCH_SIZE = 64

# Task 8 filters on these two payload fields (published_date as a range,
# regulated_entity as match-any). Without a payload index Qdrant still answers
# a filtered query correctly but does a full collection scan to do it, and --
# more importantly for correctness here -- DatetimeRange/MatchAny filters need
# the field schema declared up front so the index actually restricts results
# rather than silently returning everything. Created for every collection --
# fresh or pre-existing -- so neither a first run nor a collection left over
# from an older version of this code is ever without them.
PUBLISHED_DATE_FIELD = "published_date"
REGULATED_ENTITY_FIELD = "regulated_entity"

_PAYLOAD_SCHEMAS: dict[str, models.PayloadSchemaType] = {
    PUBLISHED_DATE_FIELD: models.PayloadSchemaType.DATETIME,
    REGULATED_ENTITY_FIELD: models.PayloadSchemaType.KEYWORD,
}


def collection_name(strategy: str) -> str:
    return f"{COLLECTION_PREFIX}{strategy}"


def foreign_collections(names: Iterable[str], prefix: str = COLLECTION_PREFIX) -> list[str]:
    """Names in `names` that this project did not create.

    A test fixture that reaches a live Qdrant should not treat "reachable" as
    "ours": on a shared dev machine, two unrelated projects can default to the
    same port, and a test that creates or deletes collections there would be
    writing into a stranger's database -- the same shape of incident that once
    cost this project its own Postgres corpus (see tests/test_db_guard.py).
    Returns the offending names; an empty result means every collection present
    belongs to us (including none at all).
    """
    return [name for name in names if not name.startswith(prefix)]


def _missing_payload_indexes(existing_fields: Iterable[str]) -> list[str]:
    """Which of our payload index fields are absent from `existing_fields`.

    Used both for a brand-new collection (existing_fields=()) and for the
    ensure_collection short-circuit where the dimension already matches: a
    collection created by an older version of this code, or one that survived
    a partial run, could be missing one or both indexes with nothing else
    saying so. Pure and Qdrant-free so it can be unit-tested directly.
    """
    present = set(existing_fields)
    return [field for field in _PAYLOAD_SCHEMAS if field not in present]


def _create_payload_indexes(client: QdrantClient, name: str, fields: Iterable[str]) -> None:
    for field in fields:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=_PAYLOAD_SCHEMAS[field],
        )


def ensure_collection(client: QdrantClient, name: str, dimension: int) -> None:
    """Create the collection, or recreate it when the dimension no longer matches.

    A dimension change means the embedding model changed. Upserting new vectors
    into an index built by a different model would silently mix incomparable
    vectors and produce retrieval numbers that mean nothing.

    Either way, the collection is left with both payload indexes: freshly
    created ones get them immediately, and a pre-existing collection with the
    right dimension gets any it is missing rather than being assumed complete.
    """
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        info = client.get_collection(name)
        current = info.config.params.vectors
        # `vectors` is `VectorParams | dict[str, VectorParams] | None`. This
        # project never creates named-vector collections (a dict), so anything
        # that isn't a plain VectorParams with a matching size is treated as a
        # mismatch and recreated -- narrowing with isinstance rather than
        # reading `.size` off a type that may not have one.
        if isinstance(current, models.VectorParams) and current.size == dimension:
            _create_payload_indexes(client, name, _missing_payload_indexes(info.payload_schema))
            return
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    _create_payload_indexes(client, name, _missing_payload_indexes(()))


def _batches(rows: Sequence[ChunkRow], size: int) -> Iterator[Sequence[ChunkRow]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _existing_point_ids(client: QdrantClient, name: str) -> set[int]:
    """Every point id currently stored in the collection, paginating via scroll.

    No payload or vector needed -- only the ids, to compute what has become
    stale.
    """
    ids: set[int] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=1000,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        ids.update(int(point.id) for point in points)
        if offset is None:
            break
    return ids


def _prune_stale_points(client: QdrantClient, name: str, current_ids: set[int]) -> int:
    """Delete points whose chunk_id no longer exists for this strategy.

    Called only after every current chunk has been upserted, never before:
    upsert-then-prune keeps the collection servable throughout the run, and a
    crash between the two steps leaves exactly today's behaviour (stale points
    linger) rather than a window with nothing indexed at all. Returns the
    number of points removed.
    """
    stale = _existing_point_ids(client, name) - current_ids
    if not stale:
        return 0
    client.delete(collection_name=name, points_selector=models.PointIdsList(points=list(stale)))
    return len(stale)


def index_strategy(
    client: QdrantClient,
    encoder: Encoder,
    session: Session,
    strategy: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Encode and upsert every chunk of one strategy, then prune anything stale.

    `replace_chunks` is delete-then-insert against an autoincrement PK, so
    re-ingesting a document mints fresh chunk_ids for it; without a prune step
    the old points would linger in Qdrant forever, indistinguishable at query
    time from current ones and still holding pre-amendment text. Amendment and
    supersession are the normal case for RBI circulars, not an edge case, so
    this is corrected on every run rather than left for a later phase.

    Returns points written (upserted), not the number pruned.
    """
    name = collection_name(strategy)
    ensure_collection(client, name, encoder.dimension)

    rows = list(
        session.scalars(
            sa.select(ChunkRow).where(ChunkRow.strategy == strategy).order_by(ChunkRow.chunk_id)
        ).all()
    )
    published = {
        d.doc_id: d.published_date for d in session.scalars(sa.select(DocumentRow)).all()
    }

    written = 0
    for batch in _batches(rows, batch_size):
        vectors = encoder.encode([row.text for row in batch])
        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=row.chunk_id,
                    vector=vector,
                    payload={
                        "doc_id": row.doc_id,
                        "strategy": row.strategy,
                        "ordinal": row.ordinal,
                        "char_start": row.char_start,
                        "char_end": row.char_end,
                        "source_url": row.source_url,
                        "regulated_entity": list(row.regulated_entity or []),
                        "published_date": published[row.doc_id].isoformat(),
                        "text": row.text,
                    },
                )
                for row, vector in zip(batch, vectors, strict=True)
            ],
        )
        written += len(batch)

    _prune_stale_points(client, name, {row.chunk_id for row in rows})
    return written
