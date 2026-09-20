import math

import pytest

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


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)
