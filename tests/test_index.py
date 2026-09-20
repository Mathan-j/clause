import os

import pytest
from qdrant_client import QdrantClient, models

from clause.index import (
    PUBLISHED_DATE_FIELD,
    REGULATED_ENTITY_FIELD,
    collection_name,
    ensure_collection,
)

pytestmark = pytest.mark.qdrant

URL = os.environ.get("CLAUSE_QDRANT_URL", "http://localhost:6335")


@pytest.fixture
def client() -> QdrantClient:
    c = QdrantClient(url=URL, timeout=5)
    try:
        c.get_collections()
    except Exception as exc:
        pytest.skip(f"no Qdrant at {URL}: {exc}")
    return c


def test_collection_name_is_namespaced_per_strategy() -> None:
    assert collection_name("structural") == "clause_structural"
    assert collection_name("fixed_window") == "clause_fixed_window"


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
