import os
from datetime import UTC, date, datetime

import pytest
import sqlalchemy as sa
from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session

from clause.db.repository import replace_chunks, upsert_document
from clause.db.schema import ChunkRow
from clause.embed import Encoder
from clause.index import (
    PUBLISHED_DATE_FIELD,
    REGULATED_ENTITY_FIELD,
    _missing_payload_indexes,
    collection_name,
    ensure_collection,
    foreign_collections,
    index_strategy,
)
from clause.models import Chunk, Document

pytestmark = pytest.mark.qdrant

URL = os.environ.get("CLAUSE_QDRANT_URL", "http://localhost:6335")


@pytest.fixture
def client() -> QdrantClient:
    c = QdrantClient(url=URL, timeout=5)
    try:
        names = [col.name for col in c.get_collections().collections]
    except Exception as exc:
        pytest.skip(f"no Qdrant at {URL}: {exc}")
    # "reachable" is not "ours": on a shared dev machine, CLAUSE_QDRANT_URL's
    # default port can just as easily answer for a completely unrelated
    # project. If this instance holds collections we didn't create, skip
    # rather than create/delete test collections in someone else's database.
    foreign = foreign_collections(names)
    if foreign:
        pytest.skip(
            f"Qdrant at {URL} holds non-clause collection(s) {foreign!r}; this looks "
            "like another project's instance sharing the URL, not ours. Refusing to "
            "create or delete collections there. Point CLAUSE_QDRANT_URL at a Qdrant "
            "this project owns and re-run."
        )
    return c


def test_collection_name_is_namespaced_per_strategy() -> None:
    assert collection_name("structural") == "clause_structural"
    assert collection_name("fixed_window") == "clause_fixed_window"


def test_foreign_collections_empty_instance_is_safe() -> None:
    assert foreign_collections([]) == []


def test_foreign_collections_all_ours_is_safe() -> None:
    assert foreign_collections(["clause_fixed_window", "clause_structural"]) == []


def test_foreign_collections_flags_names_outside_our_prefix() -> None:
    assert foreign_collections(["clause_fixed_window", "traefik_docs"]) == ["traefik_docs"]


def test_ensure_collection_is_idempotent(client: QdrantClient) -> None:
    name = "clause_test_idempotent"
    ensure_collection(client, name, 8)
    ensure_collection(client, name, 8)
    assert client.get_collection(name).config.params.vectors.size == 8
    client.delete_collection(name)


def test_changing_dimension_recreates_the_collection(client: QdrantClient) -> None:
    """A dimension change means a different model; upserting into the old index
    would mix incomparable vectors."""
    name = "clause_test_dimension"
    ensure_collection(client, name, 8)
    ensure_collection(client, name, 16)
    assert client.get_collection(name).config.params.vectors.size == 16
    client.delete_collection(name)


def test_ensure_collection_creates_payload_indexes(client: QdrantClient) -> None:
    """Task 8 filters on published_date (range) and regulated_entity (match-any).
    Without a payload index declared on the collection, a DatetimeRange or
    MatchAny filter either errors or silently fails to restrict results -- a
    gap that would otherwise surface as a mysteriously empty result set in
    Task 8 rather than here.
    """
    name = "clause_test_payload_indexes"
    ensure_collection(client, name, 8)
    try:
        info = client.get_collection(name)
        schema = info.payload_schema
        assert PUBLISHED_DATE_FIELD in schema
        assert schema[PUBLISHED_DATE_FIELD].data_type == models.PayloadSchemaType.DATETIME
        assert REGULATED_ENTITY_FIELD in schema
        assert schema[REGULATED_ENTITY_FIELD].data_type == models.PayloadSchemaType.KEYWORD
    finally:
        client.delete_collection(name)


def test_payload_indexes_actually_restrict_filtered_queries(client: QdrantClient) -> None:
    """Prove the indexes are load-bearing, not decorative: index a few points
    with distinct published_date and regulated_entity values, then run a
    filtered query on each field and assert it excludes the non-matching
    point. The payload stores published_date as `date.isoformat()`
    (e.g. "2026-09-18"); this confirms Qdrant's datetime index accepts that
    form directly, with no RFC3339 timestamp conversion needed.
    """
    name = "clause_test_payload_filtering"
    ensure_collection(client, name, 4)
    try:
        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=1,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"published_date": "2026-09-18", "regulated_entity": ["banks"]},
                ),
                models.PointStruct(
                    id=2,
                    vector=[0.4, 0.3, 0.2, 0.1],
                    payload={"published_date": "2020-01-01", "regulated_entity": ["nbfc"]},
                ),
            ],
        )

        by_date, _ = client.scroll(
            collection_name=name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=PUBLISHED_DATE_FIELD,
                        range=models.DatetimeRange(gte="2025-01-01"),
                    )
                ]
            ),
        )
        assert {p.id for p in by_date} == {1}

        by_entity, _ = client.scroll(
            collection_name=name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=REGULATED_ENTITY_FIELD,
                        match=models.MatchAny(any=["nbfc", "cooperative_bank"]),
                    )
                ]
            ),
        )
        assert {p.id for p in by_entity} == {2}
    finally:
        client.delete_collection(name)


def test_missing_payload_indexes_reports_both_when_absent() -> None:
    assert set(_missing_payload_indexes(())) == {PUBLISHED_DATE_FIELD, REGULATED_ENTITY_FIELD}


def test_missing_payload_indexes_reports_none_when_present() -> None:
    assert _missing_payload_indexes([PUBLISHED_DATE_FIELD, REGULATED_ENTITY_FIELD]) == []


def test_missing_payload_indexes_reports_only_the_absent_one() -> None:
    assert _missing_payload_indexes([PUBLISHED_DATE_FIELD]) == [REGULATED_ENTITY_FIELD]


def test_ensure_collection_backfills_missing_payload_indexes(client: QdrantClient) -> None:
    """A collection with the right dimension but missing indexes -- built by an
    older version of this code, or left over from a partial run -- must not be
    treated as complete just because ensure_collection's dimension check passes.
    """
    name = "clause_test_payload_backfill"
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
    )
    try:
        assert client.get_collection(name).payload_schema == {}
        ensure_collection(client, name, 8)
        schema = client.get_collection(name).payload_schema
        assert PUBLISHED_DATE_FIELD in schema
        assert REGULATED_ENTITY_FIELD in schema
    finally:
        client.delete_collection(name)


def _make_document(doc_id: str, published: date) -> Document:
    return Document(
        doc_id=doc_id,
        rbi_id=999000,
        url=f"https://example.com/{doc_id}",
        circular_no="TEST/1",
        dept_ref="TEST",
        title="Synthetic test document",
        doc_type="circular",
        published_date=published,
        effective_date=None,
        sha256="0" * 64,
        fetched_at=datetime.now(UTC),
        text="alpha beta gamma delta epsilon zeta",
        regulated_entity=("banks",),
    )


def _make_chunk(document: Document, ordinal: int, char_start: int, char_end: int) -> Chunk:
    return Chunk(
        doc_id=document.doc_id,
        strategy="unused",  # replace_chunks uses its own `strategy` argument instead
        ordinal=ordinal,
        char_start=char_start,
        char_end=char_end,
        text=document.text[char_start:char_end],
        source_url=document.url,
        effective_date=document.effective_date,
        doc_type=document.doc_type,
        regulated_entity=document.regulated_entity,
    )


@pytest.mark.db
@pytest.mark.model
def test_index_strategy_writes_expected_points_and_payload(
    client: QdrantClient, db_session: Session
) -> None:
    """index_strategy's actual work -- the doc_id -> published_date join,
    chunk_id-as-point-id assignment, and payload construction -- had no test
    coverage; only manual runs verified it. source_url in particular is the
    project's citation-provenance non-negotiable and belongs under a test, not
    just an eyeballed CLI run.
    """
    strategy = "test_payload"
    doc = _make_document("test-payload-doc", date(2025, 6, 1))
    upsert_document(db_session, doc)
    replace_chunks(
        db_session,
        doc.doc_id,
        strategy,
        [_make_chunk(doc, 0, 0, 5), _make_chunk(doc, 1, 6, 10)],
    )
    db_session.commit()

    name = collection_name(strategy)
    try:
        written = index_strategy(client, Encoder(), db_session, strategy)
        assert written == 2

        rows = db_session.scalars(
            sa.select(ChunkRow).where(ChunkRow.doc_id == doc.doc_id).order_by(ChunkRow.ordinal)
        ).all()
        assert len(rows) == 2

        assert client.get_collection(name).points_count == 2

        retrieved = client.retrieve(
            collection_name=name,
            ids=[row.chunk_id for row in rows],
            with_payload=True,
        )
        assert {point.id for point in retrieved} == {row.chunk_id for row in rows}
        for point in retrieved:
            assert point.payload is not None
            assert point.payload["source_url"] == doc.url
            assert point.payload["published_date"] == doc.published_date.isoformat()
            assert point.payload["regulated_entity"] == list(doc.regulated_entity)
    finally:
        if name in {c.name for c in client.get_collections().collections}:
            client.delete_collection(name)


@pytest.mark.db
@pytest.mark.model
def test_index_strategy_prunes_points_whose_chunks_are_gone(
    client: QdrantClient, db_session: Session
) -> None:
    """`replace_chunks` is delete-then-insert against an autoincrement PK, so
    re-ingesting a document mints fresh chunk_ids and strands the old points in
    Qdrant unless index_strategy prunes them. Amendment/supersession is the
    normal case for RBI circulars, so this must hold on every reindex.
    """
    strategy = "test_prune"
    doc = _make_document("test-prune-doc", date(2025, 1, 1))
    upsert_document(db_session, doc)
    replace_chunks(
        db_session,
        doc.doc_id,
        strategy,
        [_make_chunk(doc, 0, 0, 5), _make_chunk(doc, 1, 6, 10)],
    )
    db_session.commit()

    name = collection_name(strategy)
    encoder = Encoder()
    try:
        index_strategy(client, encoder, db_session, strategy)
        old_ids = set(
            db_session.scalars(
                sa.select(ChunkRow.chunk_id).where(ChunkRow.doc_id == doc.doc_id)
            ).all()
        )
        assert len(old_ids) == 2
        assert client.get_collection(name).points_count == 2

        # Simulate a re-ingest that reshapes this document's chunks: delete-then-insert
        # mints a brand-new chunk_id for the single surviving chunk.
        replace_chunks(db_session, doc.doc_id, strategy, [_make_chunk(doc, 0, 0, 10)])
        db_session.commit()

        index_strategy(client, encoder, db_session, strategy)

        new_id = db_session.scalar(
            sa.select(ChunkRow.chunk_id).where(ChunkRow.doc_id == doc.doc_id)
        )
        assert new_id is not None
        assert new_id not in old_ids

        remaining_ids = {
            point.id
            for point in client.scroll(collection_name=name, limit=100, with_payload=False)[0]
        }
        assert remaining_ids == {new_id}, (
            f"expected only the current chunk_id {new_id} to remain, found {remaining_ids} "
            f"(stale ids from before the reindex: {old_ids})"
        )
    finally:
        if name in {c.name for c in client.get_collections().collections}:
            client.delete_collection(name)
