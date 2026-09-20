import os
from datetime import date

import pytest
from qdrant_client import QdrantClient, models

from clause.embed import Encoder
from clause.evaluation.metrics import RETRIEVAL_DEPTH
from clause.index import collection_name, ensure_collection
from clause.retrieve import search

pytestmark = [pytest.mark.qdrant, pytest.mark.model]

URL = os.environ.get("CLAUSE_QDRANT_URL", "http://localhost:6335")


@pytest.fixture
def client() -> QdrantClient:
    c = QdrantClient(url=URL, timeout=5)
    try:
        if not c.get_collections().collections:
            pytest.skip("no indexed collections; run `uv run python -m clause.cli index`")
    except Exception as exc:
        pytest.skip(f"no Qdrant at {URL}: {exc}")
    return c


@pytest.fixture
def encoder() -> Encoder:
    return Encoder()


def test_search_returns_at_most_the_limit(client: QdrantClient, encoder: Encoder) -> None:
    hits = search(client, encoder, "structural", "know your customer", limit=RETRIEVAL_DEPTH)
    assert 0 < len(hits) <= RETRIEVAL_DEPTH


def test_results_are_ordered_by_descending_score(client: QdrantClient, encoder: Encoder) -> None:
    hits = search(client, encoder, "structural", "customer due diligence")
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_every_hit_carries_its_span(client: QdrantClient, encoder: Encoder) -> None:
    for hit in search(client, encoder, "structural", "beneficial owner"):
        assert hit.char_start < hit.char_end
        assert hit.doc_id
        assert hit.text


def test_every_hit_carries_its_provenance(client: QdrantClient, encoder: Encoder) -> None:
    """CLAUDE.md requires every chunk to carry its own provenance, so a citation
    never needs a second query to say where it came from. Assert this straight
    off the `Hit` the caller actually receives, not by scrolling the collection
    back out of Qdrant to recover a payload `search()` already had.
    """
    for hit in search(client, encoder, "structural", "beneficial owner"):
        assert hit.source_url
        assert hit.published_date is not None


def test_a_date_filter_excludes_older_documents(client: QdrantClient, encoder: Encoder) -> None:
    cutoff = date(2026, 1, 1)
    hits = search(client, encoder, "structural", "sanctions list", published_after=cutoff)
    assert hits, "expected at least one document on or after the cutoff"
    for hit in hits:
        assert hit.published_date >= cutoff


def test_an_entity_filter_restricts_to_documents_naming_it(
    client: QdrantClient, encoder: Encoder
) -> None:
    hits = search(client, encoder, "structural", "due diligence", entities=["Commercial Banks"])
    assert hits, "expected at least one chunk from a document naming Commercial Banks"
    for hit in hits:
        assert "Commercial Banks" in hit.regulated_entity


def test_a_filter_matching_nothing_returns_no_results(
    client: QdrantClient, encoder: Encoder
) -> None:
    hits = search(
        client, encoder, "structural", "anything", published_after=date(2099, 1, 1)
    )
    assert hits == []


def test_searching_the_other_strategy_hits_a_different_collection(
    client: QdrantClient, encoder: Encoder
) -> None:
    structural = search(client, encoder, "structural", "know your customer")
    fixed = search(client, encoder, "fixed_window", "know your customer")
    assert {h.strategy for h in structural} == {"structural"}
    assert {h.strategy for h in fixed} == {"fixed_window"}


def test_filter_draws_the_top_k_from_the_filtered_set_not_after_the_fact(
    client: QdrantClient, encoder: Encoder
) -> None:
    """Filters must run server-side inside Qdrant so the top-k is drawn from the
    filtered set, not filtered out of an unfiltered top-k.

    Proven with a synthetic collection where the vectors closest to the query
    all fail the date filter and the vectors farthest from the query all pass
    it. A client-side post-filter implementation (fetch an unfiltered top
    `limit`, then drop the ones that fail the filter) would fetch the close,
    failing points and return nothing at `limit=2`, because it never even
    looks at the far, passing points. `search()`'s server-side filter must
    reach past the close points and return the far, matching ones -- proving
    the top-k is drawn from the filtered set, not carved out of an unfiltered
    one. Same-shape assertions on the production corpus can't distinguish the
    two implementations, because both would happen to return only points that
    satisfy the filter; this test is built so only the server-side one can
    return anything at all.
    """
    strategy = "test_serverside_filter"
    name = collection_name(strategy)
    ensure_collection(client, name, encoder.dimension)
    try:
        query_vector = encoder.encode(["capital adequacy ratio"])[0]
        opposite = [-x for x in query_vector]
        cutoff = date(2026, 1, 1)
        old = date(2020, 1, 1)
        new = date(2026, 6, 1)

        def payload(doc_id: str, published: date) -> dict[str, object]:
            return {
                "doc_id": doc_id,
                "strategy": strategy,
                "ordinal": 0,
                "char_start": 0,
                "char_end": 5,
                "source_url": "https://example.com/x",
                "regulated_entity": [],
                "published_date": published.isoformat(),
                "text": "text",
            }

        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(id=1, vector=query_vector, payload=payload("close-1", old)),
                models.PointStruct(id=2, vector=query_vector, payload=payload("close-2", old)),
                models.PointStruct(id=3, vector=query_vector, payload=payload("close-3", old)),
                models.PointStruct(id=4, vector=opposite, payload=payload("far-1", new)),
                models.PointStruct(id=5, vector=opposite, payload=payload("far-2", new)),
            ],
        )

        hits = search(
            client,
            encoder,
            strategy,
            "capital adequacy ratio",
            limit=2,
            published_after=cutoff,
        )

        assert {h.doc_id for h in hits} == {"far-1", "far-2"}
    finally:
        client.delete_collection(name)
