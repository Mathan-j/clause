import math
import tempfile
import time

import pytest
from sentence_transformers import SentenceTransformer

from clause.embed import DEFAULT_MODEL, Encoder, ModelNotCachedError


@pytest.mark.model
def test_encodes_to_the_declared_dimension() -> None:
    enc = Encoder()
    vectors = enc.encode(["capital adequacy requirements"])
    assert len(vectors) == 1
    assert len(vectors[0]) == enc.dimension


@pytest.mark.model
def test_encoding_is_deterministic() -> None:
    """A measurement harness whose inputs move is not a measurement harness."""
    enc = Encoder()
    assert enc.encode(["know your customer"]) == enc.encode(["know your customer"])


@pytest.mark.model
def test_batch_order_is_preserved() -> None:
    """Batch encoding must not shuffle rows relative to single-item encoding.

    This does not assert bitwise equality between a batch-of-2 encoding and two
    single encodings: ONNX GEMM kernels accumulate in a different order at
    different batch shapes, and "alpha"/"beta" tokenize to different lengths so
    the batch pads, so exact equality is not guaranteed. Order preservation is
    the actual subject of this test, so it is asserted via nearest-neighbor
    identity instead.
    """
    enc = Encoder()
    a, b = enc.encode(["alpha", "beta"])
    a_single = enc.encode(["alpha"])[0]
    b_single = enc.encode(["beta"])[0]
    assert _cos(a, a_single) > _cos(a, b_single)
    assert _cos(b, b_single) > _cos(b, a_single)


@pytest.mark.model
def test_empty_batch_returns_empty() -> None:
    assert Encoder().encode([]) == []


@pytest.mark.model
def test_similar_text_scores_closer_than_unrelated_text() -> None:
    enc = Encoder()
    kyc, kyc2, weather = enc.encode(
        [
            "customer due diligence obligations",
            "know your customer requirements",
            "tomorrow's rainfall",
        ]
    )
    assert _cos(kyc, kyc2) > _cos(kyc, weather)


def test_an_uncached_model_fails_loudly_rather_than_downloading() -> None:
    with pytest.raises(ModelNotCachedError, match="not in the local cache"):
        Encoder(model_name="sentence-transformers/definitely-not-a-real-model-xyz")


def test_default_model_is_the_documented_baseline() -> None:
    assert DEFAULT_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_local_files_only_fails_rather_than_downloads_an_uncached_variant() -> None:
    """local_files_only=True is what actually closes the cache-check gap.

    _is_cached() only proves the repo is present somewhere in the HuggingFace
    cache, not that the specific file a load needs is among the files on disk
    for that repo. This test does not go through Encoder, because Encoder
    deliberately has no way to point at an alternate cache location (that is
    the dropped `cache_folder` parameter) -- it instead calls
    SentenceTransformer the same way Encoder's constructor does, but against an
    empty, temporary cache directory, so the "not on disk" condition is real
    and reproducible without touching the real cache or the network.

    huggingface_hub's local_files_only contract is to never attempt a network
    call and instead raise immediately when a request cannot be satisfied
    locally, so a fast run time here (rather than the tens of seconds a real
    download takes) is itself part of what this test is asserting: this must
    fail fast, not fail after downloading.
    """
    with tempfile.TemporaryDirectory() as empty_cache:
        start = time.monotonic()
        with pytest.raises(OSError):
            SentenceTransformer(
                DEFAULT_MODEL,
                backend="onnx",
                local_files_only=True,
                cache_folder=empty_cache,
            )
        elapsed = time.monotonic() - start
    assert elapsed < 5, (
        f"raised after {elapsed:.1f}s -- too slow to have failed locally; "
        "this may have attempted a network call instead of failing fast"
    )


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)
