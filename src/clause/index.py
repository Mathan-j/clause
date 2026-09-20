"""Qdrant collection lifecycle and chunk indexing.

One collection per chunking strategy, not one shared collection with a strategy
filter. Approximate-nearest-neighbour recall depends on what else is in the
index, so mixing both strategies would make each strategy's number a function of
the other -- destroying the comparison the two strategies exist to enable.
"""

from collections.abc import Iterator, Sequence

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
# rather than silently returning everything. Created once per collection,
# immediately after create_collection, so a fresh collection is never without
# them.
PUBLISHED_DATE_FIELD = "published_date"
REGULATED_ENTITY_FIELD = "regulated_entity"


def collection_name(strategy: str) -> str:
    return f"{COLLECTION_PREFIX}{strategy}"


def ensure_collection(client: QdrantClient, name: str, dimension: int) -> None:
    """Create the collection, or recreate it when the dimension no longer matches.

    A dimension change means the embedding model changed. Upserting new vectors
    into an index built by a different model would silently mix incomparable
    vectors and produce retrieval numbers that mean nothing.
    """
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        current = client.get_collection(name).config.params.vectors
        # `vectors` is `VectorParams | dict[str, VectorParams] | None`. This
        # project never creates named-vector collections (a dict), so anything
        # that isn't a plain VectorParams with a matching size is treated as a
        # mismatch and recreated -- narrowing with isinstance rather than
        # reading `.size` off a type that may not have one.
        if isinstance(current, models.VectorParams) and current.size == dimension:
            return
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=name,
        field_name=PUBLISHED_DATE_FIELD,
        field_schema=models.PayloadSchemaType.DATETIME,
    )
    client.create_payload_index(
        collection_name=name,
        field_name=REGULATED_ENTITY_FIELD,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _batches(rows: Sequence[ChunkRow], size: int) -> Iterator[Sequence[ChunkRow]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def index_strategy(
    client: QdrantClient,
    encoder: Encoder,
    session: Session,
    strategy: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Encode and upsert every chunk of one strategy. Returns points written."""
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
    return written
