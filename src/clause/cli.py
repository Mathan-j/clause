import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy.orm import Session

from clause.chunking.base import assert_slices
from clause.chunking.fixed import FixedWindowChunker
from clause.chunking.structural import StructuralChunker
from clause.config import Settings, get_settings
from clause.db.repository import replace_chunks, upsert_document
from clause.db.session import make_engine, session_factory
from clause.ingest.extract import ExtractionError, extract_document
from clause.ingest.fetch import Fetcher, FetchError, RateLimiter
from clause.ingest.validate import ValidationError
from clause.sources.manifest import load_manifest


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
            except (FetchError, ValidationError, ExtractionError, AssertionError) as exc:
                print(f"FAILED {entry.doc_id}: {exc}", file=sys.stderr)
                failures.append(entry.doc_id)
                session.rollback()
            else:
                session.commit()

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clause")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    engine = make_engine(settings.database_url)
    with session_factory(engine)() as session:
        failures = ingest(args.manifest, session=session, settings=settings)

    if failures:
        print(f"\n{len(failures)} document(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("ingest complete, all documents succeeded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
