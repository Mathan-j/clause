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
