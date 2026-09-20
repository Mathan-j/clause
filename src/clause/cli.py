import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from clause.chunking.base import SliceIntegrityError, assert_slices
from clause.chunking.fixed import FixedWindowChunker
from clause.chunking.structural import StructuralChunker
from clause.config import Settings, get_settings
from clause.db.repository import replace_chunks, upsert_document
from clause.db.session import make_engine, session_factory
from clause.embed import Encoder
from clause.index import index_strategy
from clause.ingest.extract import ExtractionError, extract_document
from clause.ingest.fetch import Fetcher, FetchError, RateLimiter
from clause.ingest.validate import ValidationError
from clause.sources.manifest import load_manifest

# The two chunking strategies produced by `_chunkers()` above, indexed into
# their own Qdrant collection each. Kept as a literal tuple here rather than
# derived from `_chunkers()` because that helper builds chunker *instances*
# (it needs `settings`), and indexing only needs the strategy names already
# stored on `ChunkRow.strategy`.
STRATEGIES = (FixedWindowChunker.name, StructuralChunker.name)


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
    encoder = Encoder(settings.embedding_model)
    return {
        strategy: index_strategy(client, encoder, session, strategy) for strategy in STRATEGIES
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clause")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--manifest", type=Path, required=True)
    sub.add_parser("index")
    args = parser.parse_args(argv)

    settings = get_settings()

    if args.command == "index":
        engine = make_engine(settings.database_url)
        with session_factory(engine)() as session:
            counts = index_all(session=session, settings=settings)
        for strategy, count in counts.items():
            print(f"indexed {count} point(s) into clause_{strategy}", file=sys.stderr)
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
