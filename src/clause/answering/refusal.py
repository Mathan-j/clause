"""Decide whether retrieval is strong enough to answer at all.

Phase 2 measured recall@5 of 0.446 and 0.338 (reports/eval.md): more than half
of golden questions do not retrieve a correct span in the top five. So the
interesting artifact is not a refusal rate, it is the curve -- how the answer
rate and the false-refusal rate move together as the bar rises. `DEFAULT_THRESHOLD`
is a starting point for the sweep to argue with, not a tuned value.
"""

from collections.abc import Sequence

from clause.answering.schema import Refusal, RefusalReason
from clause.retrieve import Hit

DEFAULT_THRESHOLD = 0.35

SWEEP_THRESHOLDS: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(20))


def decide(question: str, hits: Sequence[Hit], threshold: float) -> Refusal | None:
    """Return a `Refusal`, or `None` to proceed.

    The boundary is inclusive: a score exactly equal to the threshold proceeds.
    A question is not refused for tying the bar.
    """
    if not hits:
        return Refusal(
            question=question, reason=RefusalReason.NO_CANDIDATES, top_score=None
        )
    top = hits[0].score
    if top < threshold:
        return Refusal(question=question, reason=RefusalReason.LOW_SCORE, top_score=top)
    return None


def sweep(
    scores: Sequence[float | None], thresholds: Sequence[float] = SWEEP_THRESHOLDS
) -> dict[float, int]:
    """How many questions each threshold would refuse.

    A `None` score means nothing was retrieved, which is a refusal at every
    threshold including zero.
    """
    return {t: sum(1 for s in scores if s is None or s < t) for t in thresholds}
