# clause

Regulatory answer engine over public RBI circulars. Every factual sentence in an
answer resolves to an exact character span in an exact source document, or the system
refuses. See `PROMPT.md` for the full brief, `CLAUDE.md` for the non-negotiables, and
`STATUS.md` for current phase status.

## Running Phase 1 locally

Phase 1 covers ingestion and storage: fetch, extract, chunk, and load a corpus of RBI
KYC/AML documents into Postgres. These are the commands, not the results — per
`CLAUDE.md`, no number belongs in this file unless it comes from a committed file
under `reports/`, and `reports/` does not exist yet.

**Prerequisites**

- Docker, for the Postgres container.
- `uv`, for the Python 3.12 environment (`requires-python` pins 3.12; do not use a
  bare system `python`).
- A copy of `.env` with `CLAUSE_DATABASE_URL` and `CLAUSE_USER_AGENT` set — copy
  `.env.example` and adjust if needed. `CLAUSE_TEST_DATABASE_URL` in the same file
  must name a *different* database than `CLAUSE_DATABASE_URL`; the test suite builds
  and tears down a schema on it, and running that against the application's own
  database would destroy the ingested corpus.

**Setup**

```bash
uv sync                    # install dependencies (make install)
docker compose up -d       # start Postgres on localhost:5434 (make up)
uv run alembic upgrade head   # build the schema (make migrate)
```

**Ingest the corpus**

```bash
uv run python -m clause.cli ingest --manifest data/corpus/kyc.manifest.jsonl
# or: make ingest (which runs the migration above first)
```

The manifest (`data/corpus/kyc.manifest.jsonl`) is a frozen, committed list of RBI
document URLs and header identities produced by `clause.sources.discover` and
reviewed by hand; ingest does not perform discovery itself. Fetched raw HTML is
cached under `data/raw/` and is fetched at most once per document — deleting a
cached file causes it to be re-fetched live from `rbi.org.in` on the next ingest, so
leave the cache alone unless you intend that.

**Run the tests**

```bash
uv run pytest       # make test
uv run ruff check . # make lint (ruff half)
uv run mypy         # make lint (mypy half)
```

Most of the test suite runs without a warm cache or even a database (unit tests
against fixtures). The acceptance tests in `tests/test_ingest_acceptance.py`
(`pytest.mark.db`) additionally require:

- A reachable Postgres server matching `CLAUSE_TEST_DATABASE_URL` (created
  automatically if it doesn't exist yet, given a reachable server).
- A **warm `data/raw/` cache for every entry in the manifest** — i.e. `make ingest`
  (or the ingest command above) has already been run at least once locally. Without
  it, these tests skip deliberately rather than fetching `rbi.org.in` live, which
  would happen on every CI run and contradicts this project's rate-limiting
  commitment (see `PROMPT.md` section 1, and section 8 of
  `docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md`).

The test suite builds and tears down its own schema on `CLAUSE_TEST_DATABASE_URL`
via Alembic on every run; it never touches the database at `CLAUSE_DATABASE_URL`.

## Running the evaluation

Phase 2 covers retrieval and evaluation: embed the ingested corpus into Qdrant, run
the golden set against both chunking strategies, and gate CI on a committed
baseline. As in Phase 1, no number belongs in this file — the actual figures live in
`reports/eval.md`, which `make eval` generates and which is committed alongside this
README.

**Prerequisites**

- Everything in "Running Phase 1 locally" above, with the corpus already ingested
  (`make ingest`).
- `docker compose up -d` also starts Qdrant, on the port `CLAUSE_QDRANT_URL` in
  `.env` names (see `.env.example`).

**Warm the embedding model**

The first call into `Encoder` downloads the configured `sentence-transformers`
model and converts it to ONNX; that has nothing to do with `rbi.org.in` and is safe
to run at any time, but it does make the first `make index` or `make eval` after a
fresh environment noticeably slower than every run after it. There is no separate
warm-up command — running either one once populates the local model cache for
every run that follows.

**Index and evaluate**

```bash
make index   # embed every chunk of every strategy into its own Qdrant collection
make eval    # score both strategies against the golden set; writes reports/eval.md
             # and reports/eval.json
```

`make eval` never writes `reports/baseline.json` — see below.

**The CI gate**

```bash
make gate    # or: uv run python -m clause.cli gate
```

This compares the committed `reports/eval.json` against `reports/baseline.json` and
against a fingerprint of the working tree (the manifest, the golden set, the
embedding model, the retrieval depth, and the indexed chunk counts), and fails if
either recall has regressed below the baseline or the report is stale relative to
the tree. It is what CI runs after the test suite; see `.github/workflows/ci.yml`
and `clause.cli._gate_chunk_counts` for the one field it cannot verify without a
real corpus in CI, and why the gate prints that it is skipping it rather than
silently passing.

**Updating the baseline deliberately**

`reports/baseline.json` is a committed, hand-authored file, never an output of
`make eval` — a harness that promotes its own baseline cannot detect a regression.
Update it only as a deliberate act, after reviewing the new `reports/eval.md`:

```bash
uv run python - <<'PY'
import json, pathlib
report = json.loads(pathlib.Path("reports/eval.json").read_text(encoding="utf-8"))
baseline = {
    s: {"recall_at_5": d["overall"]["recall_at_5"]}
    for s, d in report["strategies"].items()
}
pathlib.Path("reports/baseline.json").write_text(
    json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
)
PY
```
