"""Score both chunking strategies against the golden set.

Moved out of `cli.py` (fix round 3, blocker 3 of the whole-branch review):
`evaluate()`, `bucket_result()` and `chance_baseline_for_strategy()` are the
code that actually turns a query and a corpus into the numbers
`reports/eval.md` reports, so they belong in `clause.cli.RETRIEVAL_CODE_PATHS`
-- editing any of them can move a recall/MRR/chance figure with no other
fingerprint field moving, exactly like editing `retrieve.py` or `embed.py`.
`cli.py` itself could not simply be added to that tuple: it also holds
ingest/index/gate argument parsing and wiring that cannot move a number in
the report (hashing it would make the field as noisy as `git_commit`, which
`FINGERPRINT_FIELDS` already excludes for that reason), and
`RETRIEVAL_CODE_PATHS` itself lives in `cli.py`, which would make hashing the
whole file self-referential. Moving the scoring logic here turns the
exclusion of `cli.py` from a gap into a principle: `cli.py` parses arguments
and wires modules together; the modules it wires compute the numbers.
"""

import math
from collections.abc import Sequence

from qdrant_client import QdrantClient

from clause.chunking.fixed import FixedWindowChunker
from clause.chunking.structural import StructuralChunker
from clause.embed import Encoder
from clause.evaluation.golden import BUCKETS, GoldenQuestion
from clause.evaluation.metrics import (
    RETRIEVAL_DEPTH,
    Span,
    first_hit_rank,
    mrr,
    overlaps,
    recall_at_k,
)
from clause.evaluation.report import BucketResult, ChanceBaseline, StrategyResult
from clause.retrieve import search

#: The two chunking strategies, indexed into their own Qdrant collection
#: each. Kept as a literal tuple (rather than derived from chunker
#: instances, which need `Settings`) because scoring, indexing and gating
#: only ever need the strategy *names* already stored on `ChunkRow.strategy`.
STRATEGIES = (FixedWindowChunker.name, StructuralChunker.name)

#: The chance-baseline formula considers a random draw of exactly this many
#: chunks, matching `recall_at_5` -- the figure it exists to give a floor for.
_CHANCE_BASELINE_K = 5


def bucket_result(ranks: Sequence[int | None]) -> BucketResult:
    return BucketResult(
        n=len(ranks),
        recall_at_1=recall_at_k(ranks, 1),
        recall_at_5=recall_at_k(ranks, 5),
        recall_at_10=recall_at_k(ranks, 10),
        mrr_at_10=mrr(ranks),
    )


def evaluate(
    client: QdrantClient,
    encoder: Encoder,
    questions: Sequence[GoldenQuestion],
    *,
    depth: int = RETRIEVAL_DEPTH,
) -> list[StrategyResult]:
    """Score both strategies against the golden set."""
    results: list[StrategyResult] = []
    for strategy in STRATEGIES:
        ranks_overall: list[int | None] = []
        ranks_by_bucket: dict[str, list[int | None]] = {b: [] for b in BUCKETS}
        for question in questions:
            hits = search(client, encoder, strategy, question.question, limit=depth)
            rank = first_hit_rank(
                [h.as_retrieved() for h in hits], question.answer_spans()
            )
            ranks_overall.append(rank)
            ranks_by_bucket[question.bucket].append(rank)
        results.append(
            StrategyResult(
                strategy=strategy,
                overall=bucket_result(ranks_overall),
                per_bucket={b: bucket_result(r) for b, r in ranks_by_bucket.items()},
            )
        )
    return results


def chance_baseline_for_strategy(
    questions: Sequence[GoldenQuestion], chunk_spans: Sequence[Span]
) -> ChanceBaseline:
    """The measured probability a uniform random top-5 already contains an
    acceptable chunk, for one strategy's live chunk corpus.

    See `clause.evaluation.report.ChanceBaseline`'s docstring for the method
    this implements exactly: an acceptable chunk for a question is one whose
    span overlaps any acceptable answer span in the same document (not
    double-counted); with `N` chunks in the collection and `m` acceptable
    ones, `P = 1 - C(N-m, 5) / C(N, 5)`. `recall_at_5` is the mean of `P` over
    every question; `worst_question_recall_at_5` is the maximum.
    """
    n = len(chunk_spans)
    total = math.comb(n, _CHANCE_BASELINE_K)
    probabilities: list[float] = []
    for question in questions:
        answers = question.answer_spans()
        m = sum(1 for span in chunk_spans if any(overlaps(span, a) for a in answers))
        # math.comb(a, b) is 0 when b > a, which is exactly right here: once
        # m acceptable chunks account for more than N - 5 of the corpus, a
        # random draw of 5 can no longer avoid one, so P must be 1.
        remaining = math.comb(n - m, _CHANCE_BASELINE_K)
        p = 1.0 if total == 0 else 1 - remaining / total
        probabilities.append(p)
    return ChanceBaseline(
        recall_at_5=sum(probabilities) / len(probabilities),
        worst_question_recall_at_5=max(probabilities),
    )
