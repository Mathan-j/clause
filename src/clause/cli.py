import argparse
import hashlib
import math
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import sqlalchemy as sa
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from clause.chunking.base import SliceIntegrityError, assert_slices
from clause.chunking.fixed import FixedWindowChunker
from clause.chunking.structural import StructuralChunker
from clause.config import Settings, get_settings
from clause.db.repository import replace_chunks, upsert_document
from clause.db.schema import ChunkRow, DocumentRow
from clause.db.session import make_engine, session_factory
from clause.embed import Encoder
from clause.evaluation.golden import (
    BUCKETS,
    GoldenQuestion,
    load_golden,
    provenance_split,
    validate_spans,
)
from clause.evaluation.metrics import (
    RETRIEVAL_DEPTH,
    Span,
    first_hit_rank,
    mrr,
    overlaps,
    recall_at_k,
)
from clause.evaluation.report import (
    BucketResult,
    ChanceBaseline,
    Fingerprint,
    StrategyResult,
    build_report,
    write_report,
)
from clause.index import foreign_collections, index_strategy
from clause.ingest.extract import ExtractionError, extract_document
from clause.ingest.fetch import Fetcher, FetchError, RateLimiter
from clause.ingest.validate import ValidationError
from clause.retrieve import search
from clause.sources.manifest import load_manifest

# The two chunking strategies produced by `_chunkers()` above, indexed into
# their own Qdrant collection each. Kept as a literal tuple here rather than
# derived from `_chunkers()` because that helper builds chunker *instances*
# (it needs `settings`), and indexing only needs the strategy names already
# stored on `ChunkRow.strategy`.
STRATEGIES = (FixedWindowChunker.name, StructuralChunker.name)

GOLDEN_PATH = Path("data/golden/kyc-v1.jsonl")
MANIFEST_PATH = Path("data/corpus/kyc.manifest.jsonl")
DEFAULT_REPORT_MD = Path("reports/eval.md")
DEFAULT_REPORT_JSON = Path("reports/eval.json")

#: The chance-baseline formula considers a random draw of exactly this many
#: chunks, matching `recall_at_5` -- the figure it exists to give a floor for.
_CHANCE_BASELINE_K = 5


def _chunkers(settings: Settings) -> list[FixedWindowChunker | StructuralChunker]:
    return [
        FixedWindowChunker(settings.window_chars, settings.overlap_chars),
        StructuralChunker(settings.min_chunk_chars, settings.max_chunk_chars),
    ]


def ingest(manifest_path: Path, *, session: Session, settings: Settings | None = None) -> list[str]:
    """Ingest every manifest entry. Returns the list of doc_ids that failed."""
    settings = settings or get_settings()
    entries = load_manifest(manifest_path)
    failures: list[str] = []

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        fetcher = Fetcher(settings, client, RateLimiter(settings.min_request_interval_s))
        for entry in entries:
            try:
                raw = fetcher.fetch(entry)
                document = extract_document(entry, raw, fetched_at=datetime.now(UTC))
                upsert_document(session, document)
                for chunker in _chunkers(settings):
                    chunks = chunker.chunk(document)
                    # An unresolvable citation is a hard failure, never a warning.
                    assert_slices(document, chunks)
                    replace_chunks(session, document.doc_id, chunker.name, chunks)
            except (
                FetchError,
                ValidationError,
                ExtractionError,
                ValueError,
                SliceIntegrityError,
            ) as exc:
                print(f"FAILED {entry.doc_id}: {exc}", file=sys.stderr)
                failures.append(entry.doc_id)
                session.rollback()
            else:
                session.commit()

    return failures


def index_all(*, session: Session, settings: Settings | None = None) -> dict[str, int]:
    """Index every chunking strategy into its own Qdrant collection.

    Returns {strategy: points_written}.
    """
    settings = settings or get_settings()
    client = QdrantClient(url=settings.qdrant_url)

    # Unlike the test suite's fixture (tests/test_index.py), this production path
    # warns rather than refuses: it only ever adds clause_* collections, so the
    # blast radius of a misconfigured CLAUSE_QDRANT_URL is far smaller than a test
    # that creates and deletes collections, and refusing outright would block a
    # legitimate first run against a fresh shared instance. Make the
    # misconfiguration visible; leave the decision with the operator.
    foreign = foreign_collections([c.name for c in client.get_collections().collections])
    if foreign:
        print(
            f"warning: Qdrant at {settings.qdrant_url} holds non-clause collection(s) "
            f"{foreign!r}; this may be another project's instance sharing the URL. "
            "Continuing anyway -- this command only ever adds clause_* collections.",
            file=sys.stderr,
        )

    encoder = Encoder(settings.embedding_model)
    return {
        strategy: index_strategy(client, encoder, session, strategy) for strategy in STRATEGIES
    }


def _bucket_result(ranks: Sequence[int | None]) -> BucketResult:
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
                overall=_bucket_result(ranks_overall),
                per_bucket={b: _bucket_result(r) for b, r in ranks_by_bucket.items()},
            )
        )
    return results


def _chance_baseline_for_strategy(
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _print_overall_table(results: Sequence[StrategyResult]) -> None:
    print(f"{'strategy':<14}{'n':>5}{'recall@1':>10}{'recall@5':>10}{'recall@10':>11}{'mrr@10':>9}")
    for r in results:
        o = r.overall
        print(
            f"{r.strategy:<14}{o.n:>5}{o.recall_at_1:>10.2f}{o.recall_at_5:>10.2f}"
            f"{o.recall_at_10:>11.2f}{o.mrr_at_10:>9.3f}"
        )


def run_eval(
    *,
    md_path: Path = DEFAULT_REPORT_MD,
    json_path: Path = DEFAULT_REPORT_JSON,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Score both chunking strategies against the golden set and write the report.

    In order, because it matters: the golden set is loaded and its spans
    validated against the *stored* corpus before any retrieval runs at all.
    A `GoldenSetError` here aborts the whole run -- scoring against offsets
    that no longer point at the labelled text would produce a plausible
    table measuring nothing, which is worse than a crash.
    """
    settings = settings or get_settings()
    questions = load_golden(GOLDEN_PATH)

    engine = make_engine(settings.database_url)
    with session_factory(engine)() as session:
        documents = {
            d.doc_id: (d.text, d.sha256) for d in session.scalars(sa.select(DocumentRow)).all()
        }
        validate_spans(questions, documents)

        # Only after span validation succeeds does this run touch the
        # embedding model or Qdrant -- see the docstring above.
        encoder = Encoder(settings.embedding_model)
        client = QdrantClient(url=settings.qdrant_url)

        results = evaluate(client, encoder, questions)

        chunk_counts: dict[str, int] = {}
        chance_baseline: dict[str, ChanceBaseline] = {}
        for strategy in STRATEGIES:
            rows = session.scalars(
                sa.select(ChunkRow).where(ChunkRow.strategy == strategy)
            ).all()
            chunk_counts[strategy] = len(rows)
            chunk_spans = [
                Span(doc_id=row.doc_id, char_start=row.char_start, char_end=row.char_end)
                for row in rows
            ]
            chance_baseline[strategy] = _chance_baseline_for_strategy(questions, chunk_spans)

    fingerprint = Fingerprint(
        manifest_sha256=_sha256_file(MANIFEST_PATH),
        chunk_counts=chunk_counts,
        embedding_model=settings.embedding_model,
        retrieval_depth=RETRIEVAL_DEPTH,
        golden_path=GOLDEN_PATH.as_posix(),
        golden_sha256=_sha256_file(GOLDEN_PATH),
        git_commit=_git_commit(),
    )
    report = build_report(
        results,
        fingerprint,
        provenance_split(questions),
        questions,
        chance_baseline,
    )
    write_report(report, md_path, json_path)
    _print_overall_table(results)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clause")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--manifest", type=Path, required=True)
    sub.add_parser("index")
    sub.add_parser("eval")
    args = parser.parse_args(argv)

    settings = get_settings()

    if args.command == "index":
        engine = make_engine(settings.database_url)
        with session_factory(engine)() as session:
            counts = index_all(session=session, settings=settings)
        for strategy, count in counts.items():
            print(f"indexed {count} point(s) into clause_{strategy}", file=sys.stderr)
        return 0

    if args.command == "eval":
        run_eval(settings=settings)
        return 0

    engine = make_engine(settings.database_url)
    with session_factory(engine)() as session:
        failures = ingest(args.manifest, session=session, settings=settings)

    if failures:
        print(f"\n{len(failures)} document(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    entry_count = len(load_manifest(args.manifest))
    if entry_count == 0:
        print(
            f"\nmanifest {args.manifest} contains no entries -- ingested 0 documents. "
            "This is not success: PROMPT.md's Phase 1 definition of done requires "
            "at least 50 real documents.",
            file=sys.stderr,
        )
        return 1

    print(f"ingest complete, all {entry_count} document(s) succeeded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
