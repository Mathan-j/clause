import argparse
import hashlib
import json
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
from clause.config import GateSettings, Settings, get_gate_settings, get_settings
from clause.db.repository import replace_chunks, upsert_document
from clause.db.schema import ChunkRow, DocumentRow
from clause.db.session import make_engine, session_factory
from clause.embed import Encoder
from clause.evaluation.answer_run import DEFAULT_ANSWERS_JSON, DEFAULT_ANSWERS_MD, run_answer_eval
from clause.evaluation.gate import GateFailure, check
from clause.evaluation.golden import load_golden, provenance_split, validate_spans
from clause.evaluation.metrics import RETRIEVAL_DEPTH, Span
from clause.evaluation.report import (
    ChanceBaseline,
    Fingerprint,
    StrategyResult,
    build_report,
    write_report,
)
from clause.evaluation.scoring import STRATEGIES, chance_baseline_for_strategy, evaluate
from clause.index import foreign_collections, index_strategy
from clause.ingest.extract import ExtractionError, extract_document
from clause.ingest.fetch import Fetcher, FetchError, RateLimiter
from clause.ingest.validate import ValidationError
from clause.sources.manifest import load_manifest

GOLDEN_PATH = Path("data/golden/kyc-v1.jsonl")
MANIFEST_PATH = Path("data/corpus/kyc.manifest.jsonl")
DEFAULT_REPORT_MD = Path("reports/eval.md")
DEFAULT_REPORT_JSON = Path("reports/eval.json")
DEFAULT_BASELINE_JSON = Path("reports/baseline.json")


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: The retrieval path: source files whose change can alter a recall, MRR or
#: chance-baseline number in `reports/eval.md` without any *other* fingerprint
#: field moving.
#:
#: - `retrieve.py`, `embed.py`: what `evaluate()` (`clause.evaluation.scoring`)
#:   calls through on every question.
#: - `index.py`, `chunking/*.py` (every `.py` file under that package,
#:   including the empty `__init__.py` -- see the membership guard below):
#:   they produce the corpus `chunk_counts` describes only as a count, not as
#:   content -- two different chunkings can share a chunk count.
#: - `evaluation/metrics.py`: defines what counts as a hit and how
#:   recall/MRR are computed.
#: - `evaluation/golden.py`: parses the committed golden set into the
#:   `GoldenQuestion` objects `evaluate()` scores against and owns
#:   `answer_spans()`/`BUCKETS`/`validate_spans`. `golden_sha256` pins the
#:   *bytes* of `data/golden/kyc-v1.jsonl`; nothing else pinned the *parser*
#:   that turns those bytes into the spans a hit is measured against, and
#:   `run_eval` executes it on every run. Proved during fix round 3: shrinking
#:   every acceptable span in `answer_spans()` to one character (which would
#:   collapse recall for most questions) left every other fingerprint field
#:   unchanged and the gate passed.
#: - `evaluation/scoring.py`: `evaluate()`, `bucket_result()` and
#:   `chance_baseline_for_strategy()` -- the code that turns a query and a
#:   corpus into the numbers in the report. This used to live in `cli.py`;
#:   see below for why it was moved rather than added to this tuple in place.
#:
#: Deliberately *not* every `.py` file in the repo, and not `cli.py` itself,
#: even though `evaluate()`, `bucket_result()` and `chance_baseline_for_strategy()`
#: used to live there: `cli.py` also holds ingest/index/gate argument parsing
#: and wiring that cannot move a number in the report, hashing it would make
#: this field as noisy as `git_commit` (which `FINGERPRINT_FIELDS` already
#: excludes for that exact reason -- it moves on every commit, including ones
#: this eval doesn't depend on), and `RETRIEVAL_CODE_PATHS` itself lives in
#: `cli.py`, which would make hashing it self-referential too. Moving the
#: scoring functions into `clause.evaluation.scoring` (fix round 3) turned
#: that exclusion from a gap -- proved during the same round: patching
#: `_bucket_result` (as it then was) to hardcode `recall_at_1=1.0` made every
#: recall@1 in the report a lie and the gate still passed -- into a principle:
#: `cli.py` parses arguments and wires modules together; the modules it wires
#: compute the numbers.
#:
#: Also *not* `evaluation/report.py`, despite it computing
#: `golden_composition` figures that appear in the report: every one of those
#: is derived from data already pinned elsewhere (`golden_sha256` pins the
#: golden set those figures are computed from; the pinned chance-baseline
#: constants have their own measured-vs-pinned mismatch check, distinct from
#: this fingerprint entirely), and it is the highest-churn file in this
#: package (rendering prose changes constantly; scoring logic does not).
#: Hashing it would make this field fail on doc-only edits with no scoring
#: consequence.
#:
#: Two guards keep this list honest rather than merely plausible:
#: `tests/test_cli.py::test_retrieval_code_sha256_changes_when_the_code_does`
#: proves the listed files are actually hashed (not just readable);
#: `test_retrieval_code_paths_covers_every_chunking_module` and
#: `test_retrieval_code_paths_covers_every_module_retrieve_directly_imports`
#: are the membership guard added in fix round 3 -- they fail if a `.py` file
#: is added under `chunking/` or a new `clause.*` import is added to
#: `retrieve.py` without this tuple being updated to match, so a silent gap
#: here is caught immediately rather than the next time someone happens to
#: edit gate.py's tests. They do not, and cannot, prove *completeness* beyond
#: those two specific properties -- extending this set beyond them (e.g. to
#: `evaluation/scoring.py`'s own other imports) remains a judgement call,
#: made explicitly above rather than silently.
RETRIEVAL_CODE_PATHS: tuple[Path, ...] = (
    Path("src/clause/retrieve.py"),
    Path("src/clause/embed.py"),
    Path("src/clause/index.py"),
    Path("src/clause/chunking/__init__.py"),
    Path("src/clause/chunking/base.py"),
    Path("src/clause/chunking/fixed.py"),
    Path("src/clause/chunking/structural.py"),
    Path("src/clause/evaluation/metrics.py"),
    Path("src/clause/evaluation/golden.py"),
    Path("src/clause/evaluation/scoring.py"),
)


def _retrieval_code_sha256(paths: Sequence[Path] = RETRIEVAL_CODE_PATHS) -> str:
    """One hash over the retrieval path's source, stable across platforms.

    Sorted by POSIX path so file-discovery order never matters. Each file's
    text is decoded and its line endings normalised to `\\n` before hashing --
    without that, a Windows checkout (CRLF) and a Linux CI checkout (LF) of
    the *identical* commit would hash differently, and this field would fail
    every build on whichever platform did not produce the committed report.
    Each file's POSIX path is hashed alongside its content (with a NUL
    separator, which cannot appear in either) so renaming a file with no
    content change still changes the hash.
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.as_posix()):
        normalised = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalised.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


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
            chance_baseline[strategy] = chance_baseline_for_strategy(questions, chunk_spans)

    fingerprint = Fingerprint(
        manifest_sha256=_sha256_file(MANIFEST_PATH),
        chunk_counts=chunk_counts,
        embedding_model=settings.embedding_model,
        retrieval_depth=RETRIEVAL_DEPTH,
        golden_path=GOLDEN_PATH.as_posix(),
        golden_sha256=_sha256_file(GOLDEN_PATH),
        git_commit=_git_commit(),
        retrieval_code_sha256=_retrieval_code_sha256(),
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


def _gate_chunk_counts(
    session: Session, report: dict[str, Any]
) -> tuple[dict[str, int], list[str]]:
    """Chunk counts for the gate's tree fingerprint, tolerating an empty database.

    CI never runs `make ingest`: its Postgres service exists for the test suite,
    not for a real corpus, so by the time the gate runs there is either no
    `chunks` table at all or a table with zero rows for both strategies. Either
    way there is nothing live to count, so this falls back to the *committed
    report's own* `chunk_counts`.

    That fallback makes the `chunk_counts` half of the fingerprint check
    compare `reports/eval.json` to itself -- it cannot fail, in precisely the
    environment where the gate actually runs. A check that cannot fire must at
    minimum announce that it did not run: the second return value is the list
    of notes the caller must print, so a passing gate is never read as having
    verified this field.
    """
    try:
        counts = {
            strategy: (
                session.scalar(
                    sa.select(sa.func.count())
                    .select_from(ChunkRow)
                    .where(ChunkRow.strategy == strategy)
                )
                or 0
            )
            for strategy in STRATEGIES
        }
    except sa.exc.SQLAlchemyError as exc:
        return (
            dict(report.get("fingerprint", {}).get("chunk_counts", {})),
            [
                "chunk-count fingerprint check skipped: the database is unreachable "
                f"or has no corpus ({exc.__class__.__name__}). Falling back to the "
                "committed report's own chunk_counts as a stand-in for the working "
                "tree's -- this means that field's comparison compares "
                "reports/eval.json to itself and cannot fail. This is expected in "
                "CI, which never runs `make ingest`; it is not expected on a "
                "developer machine with a populated database."
            ],
        )

    if sum(counts.values()) == 0:
        return (
            dict(report.get("fingerprint", {}).get("chunk_counts", {})),
            [
                "chunk-count fingerprint check skipped: the database has zero "
                "chunks for every strategy. Falling back to the committed report's "
                "own chunk_counts as a stand-in for the working tree's -- this "
                "means that field's comparison compares reports/eval.json to "
                "itself and cannot fail. This is expected in CI, which never runs "
                "`make ingest`; it is not expected on a developer machine with a "
                "populated database."
            ],
        )
    return counts, []


def _rebuild_tree_fingerprint(
    report: dict[str, Any], *, session: Session, settings: Settings | GateSettings
) -> tuple[dict[str, Any], list[str]]:
    """Rebuild the fingerprint from the working tree, the same way `run_eval`
    does -- but without needing Qdrant, a loaded embedding model or a real
    search, since the gate only ever compares this dict's fields against the
    committed report and never runs retrieval itself. `retrieval_code_sha256`
    is cheap file I/O over `RETRIEVAL_CODE_PATHS`, not a model load, so it
    fits that constraint too.
    """
    chunk_counts, notes = _gate_chunk_counts(session, report)
    fingerprint = {
        "manifest_sha256": _sha256_file(MANIFEST_PATH),
        "chunk_counts": chunk_counts,
        "embedding_model": settings.embedding_model,
        "retrieval_depth": RETRIEVAL_DEPTH,
        "golden_path": GOLDEN_PATH.as_posix(),
        "golden_sha256": _sha256_file(GOLDEN_PATH),
        "git_commit": _git_commit(),
        "retrieval_code_sha256": _retrieval_code_sha256(),
    }
    return fingerprint, notes


def _load_json_or_fail(path: Path, what: str) -> dict[str, Any]:
    """Load a committed JSON artifact for the gate, turning I/O and parse
    failures into a diagnosed `GateFailure` instead of a raw traceback.

    A deleted or corrupted committed report or baseline is precisely the
    failure this gate exists to catch. It must not be the one case where a
    CI reader gets a Python stack trace to parse instead of a sentence
    naming the path and what was wrong with it -- every other failure mode
    here prints `GATE FAILURE: <cause>` before raising, so this does too.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"could not read the {what} at {path}: {exc}"
        print(f"GATE FAILURE: {message}", file=sys.stderr)
        raise GateFailure(message) from exc
    try:
        loaded: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        message = f"the {what} at {path} is not valid JSON: {exc}"
        print(f"GATE FAILURE: {message}", file=sys.stderr)
        raise GateFailure(message) from exc
    return loaded


#: How long the gate will wait to establish a database connection before
#: treating it as unreachable. `_gate_chunk_counts` claims to tolerate an
#: unreachable database -- without a bound here, a misconfigured
#: `CLAUSE_DATABASE_URL` (wrong host, closed port, firewalled) would hang
#: the whole CI job on the OS-level TCP timeout instead of failing fast into
#: that fallback.
_GATE_DB_CONNECT_TIMEOUT_S = 5.0


def run_gate(
    *,
    report_path: Path = DEFAULT_REPORT_JSON,
    baseline_path: Path = DEFAULT_BASELINE_JSON,
    settings: Settings | GateSettings | None = None,
) -> None:
    """Load the committed report and baseline, rebuild the tree fingerprint,
    and raise `GateFailure` if `check()` finds a problem.

    Defaults to `GateSettings`, not the full `Settings` -- the gate never
    needs `CLAUSE_USER_AGENT`, so it must not refuse to start somewhere that
    only sets `CLAUSE_DATABASE_URL` (CI's `gate` step). A caller that already
    has a full `Settings` (e.g. `main()` for every other subcommand, or a
    test) may still pass one; both expose `database_url` and
    `embedding_model`, which is all this needs.

    Every note and every failure is printed to stderr before this returns or
    raises, so a reader watching CI output sees what ran, what was skipped
    and why, and what failed -- never just an exit code.
    """
    settings = settings or get_gate_settings()
    report = _load_json_or_fail(report_path, "report")

    baseline: dict[str, Any] | None = None
    if baseline_path.exists():
        baseline = _load_json_or_fail(baseline_path, "baseline")
    else:
        print(
            f"NOTE: no baseline at {baseline_path} -- nothing to regress against "
            "(a first run, or a baseline deliberately not yet committed). Only "
            "the fingerprint is checked.",
            file=sys.stderr,
        )

    engine = make_engine(settings.database_url, connect_timeout=_GATE_DB_CONNECT_TIMEOUT_S)
    with session_factory(engine)() as session:
        tree_fingerprint, notes = _rebuild_tree_fingerprint(
            report, session=session, settings=settings
        )

    for note in notes:
        print(f"NOTE: {note}", file=sys.stderr)

    problems = check(report, baseline, tree_fingerprint)
    for problem in problems:
        print(f"GATE FAILURE: {problem}", file=sys.stderr)
    if problems:
        raise GateFailure("; ".join(problems))


def _cmd_index(settings: Settings) -> int:
    engine = make_engine(settings.database_url)
    with session_factory(engine)() as session:
        counts = index_all(session=session, settings=settings)
    for strategy, count in counts.items():
        print(f"indexed {count} point(s) into clause_{strategy}", file=sys.stderr)
    return 0


def _cmd_ingest(manifest: Path, settings: Settings) -> int:
    engine = make_engine(settings.database_url)
    with session_factory(engine)() as session:
        failures = ingest(manifest, session=session, settings=settings)

    if failures:
        print(f"\n{len(failures)} document(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    entry_count = len(load_manifest(manifest))
    if entry_count == 0:
        print(
            f"\nmanifest {manifest} contains no entries -- ingested 0 documents. "
            "This is not success: PROMPT.md's Phase 1 definition of done requires "
            "at least 50 real documents.",
            file=sys.stderr,
        )
        return 1

    print(f"ingest complete, all {entry_count} document(s) succeeded", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clause")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--manifest", type=Path, required=True)
    sub.add_parser("index")
    sub.add_parser("eval")
    answer_eval = sub.add_parser("answer-eval")
    answer_eval.add_argument("--md", type=Path, default=DEFAULT_ANSWERS_MD)
    answer_eval.add_argument("--json", type=Path, default=DEFAULT_ANSWERS_JSON)
    gate = sub.add_parser("gate")
    gate.add_argument("--report", type=Path, default=DEFAULT_REPORT_JSON)
    gate.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_JSON)
    args = parser.parse_args(argv)

    if args.command == "gate":
        # Deliberately not get_settings(): the gate compares two committed
        # JSON files and a chunk count, and must not refuse to start
        # somewhere (CI) that never sets CLAUSE_USER_AGENT.
        try:
            run_gate(
                report_path=args.report,
                baseline_path=args.baseline,
                settings=get_gate_settings(),
            )
        except GateFailure:
            return 1
        return 0

    settings = get_settings()

    if args.command == "index":
        return _cmd_index(settings)

    if args.command == "eval":
        run_eval(settings=settings)
        return 0

    if args.command == "answer-eval":
        return run_answer_eval(md_path=args.md, json_path=args.json, settings=settings)

    return _cmd_ingest(args.manifest, settings)


if __name__ == "__main__":
    raise SystemExit(main())
