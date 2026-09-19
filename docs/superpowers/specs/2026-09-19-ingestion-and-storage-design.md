# Design: Phase 0 scaffold and Phase 1 ingestion + storage

Date: 2026-09-19
Scope: `PROMPT.md` Phase 0 and Phase 1 only. Phases 2-5 are deliberately not designed here.
Status: awaiting author review, then implementation plan.

## 1. What this covers

Phase 0 produces a repository that lints, type-checks, tests and goes green in CI.
Phase 1 produces a reproducible corpus of RBI KYC/AML documents in Postgres, chunked by
two competing strategies, where every chunk's stored character span provably slices the
stored source text.

Phase 1's definition of done, restated from `PROMPT.md`:

> a documented command ingests at least 50 real documents and `pytest` passes, including
> a test asserting that every chunk's `char_start`/`char_end` slice of the source text
> equals the chunk's stored text.

## 2. Evidence and its status

Three findings below came from single manual HTTP requests made on 2026-09-19 while
designing this document. **They are design rationale, not measurements.** No number in
this file is a system measurement. System measurements begin in Phase 2 and live only in
`reports/`, per `CLAUDE.md`. Anything here that later matters as a number gets measured
and committed then.

| Probe | Result |
|---|---|
| `GET /robots.txt` | HTTP 418, with and without a browser User-Agent |
| `GET /Scripts/NotificationUser.aspx?Id=13704&Mode=0` | HTTP 200, HTML, full circular text |
| `GET rbidocs.rbi.org.in/.../NT...PDF` | HTTP 200 but `text/html` - a bot-check page, not a PDF |

Three consequences, each of which changes what `PROMPT.md` assumed:

1. **HTML is the canonical source, not PDF.** `PROMPT.md` section 3 shows
   "raw HTML/PDF kept on disk". The PDF host sits behind a JavaScript challenge, while
   the HTML page serves the complete text. This is fortunate: character offsets into
   extracted HTML text are stable and verifiable, whereas PDF text extraction would have
   made the citation round-trip genuinely hard to guarantee.
2. **`robots.txt` cannot be read, so compliance with it cannot be claimed.**
   `PROMPT.md` section 1 says "Respect robots.txt and rate-limit fetches". The first half
   is not satisfiable. See section 8.
3. **A blocked request returns HTTP 200 with an HTML body.** It does not fail loudly. This
   drives the single most important failure control in the design (section 7.1).

The observed block page contains the literal strings `Unauthorised Access` and
`Support ID:`. The observed circular header line has the shape
`RBI/2026-27/262 DOR.AML.REC.223/14.01.005/2026-27 September 18, 2026`.

**Addendum, 2026-09-20 (Task 10 discovery, fix rounds 1-2).** A fourth finding, from
repeated live fetches rather than a single probe: **no stable whole-document hash
exists against this source**, at either the raw-byte or the canonical-text level.
This directly contradicts the hash-verification guarantee section 7.2 originally
claimed. See section 9 for the evidence and the correction — the same treatment
section 8 already gives the `robots.txt` claim.

## 3. Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Corpus breadth | One regulatory domain, deep | Makes Phase 2's cross-reference and effective-date question buckets real rather than contrived |
| Domain | KYC / AML | Covers all four Phase 2 buckets: definitional, numeric, procedural, cross-reference. Dense amendment traffic makes `effective_date` load-bearing |
| Corpus acquisition | Frozen manifest + hash-verified cache | Holds the corpus still so retrieval deltas are attributable to code, not drift |

The rejected alternatives, recorded so they are not silently revisited:

- **Live discovery on every ingest** was rejected because a drifting corpus makes every
  eval comparison uninterpretable. `CLAUDE.md` requires a fresh `make eval` and a
  committed report for retrieval changes; that discipline is meaningless if recall can
  move because RBI published something overnight.
- **Vendoring extracted text into the repo** was rejected because `PROMPT.md` commits to
  storing extracted text and source URLs without redistributing the corpus in bulk, and
  because it would delete the fetcher from the project's demonstrated scope.

## 4. Phase 0 - scaffold

```
pyproject.toml          ruff, mypy, pytest config in one file
.python-version         3.12 (already committed)
Makefile                install lint test ingest eval serve up
src/clause/
  __init__.py
  config.py             pydantic-settings, environment-driven
tests/
  test_smoke.py
.github/workflows/ci.yml
docs/decisions.md
```

- `requires-python = ">=3.12,<3.13"`, matching the committed `.python-version`.
- ruff and mypy configured in `pyproject.toml`; mypy strict over `src/clause`.
- CI on `ubuntu-latest`: `uv sync --frozen`, then `make lint`, then `make test`, on push
  and pull request.
- Phase 0 adds **no runtime dependencies**. Only ruff, mypy and pytest, each getting its
  line in `docs/decisions.md` per `CLAUDE.md`.

`uv sync` creates `.venv`, which is where `.claude/hooks/gate.sh` already resolves ruff
and pytest from. The Stop-hook gate therefore becomes active the moment Phase 0 lands,
with no further wiring.

**Done when:** CI is green on a real but trivial test suite.

## 5. Phase 1 - units

Six units. Each has one purpose, a stated interface, and can be tested without the others.

| Unit | Purpose | Depends on |
|---|---|---|
| `clause.sources.discover` | Produce a reviewed manifest of candidate documents | network (one-off, manual) |
| `clause.ingest.fetch` | Manifest to `data/raw/` cache, rate-limited, validated, hash-checked | manifest |
| `clause.ingest.extract` | Raw HTML to canonical text plus metadata | cache |
| `clause.chunking` | Canonical text to chunk spans, two strategies | extract |
| `clause.db` | Postgres schema and Alembic migrations | - |
| `clause.cli` | `make ingest` entry point, orchestration and reporting | all of the above |

### 5.1 Discovery is not part of ingest

`clause.sources.discover` is run by hand, reviewed by a human, and its output committed.
It is **not** invoked by `make ingest`. This is the mechanism that freezes the corpus.

Output: `data/corpus/kyc.manifest.jsonl`, one JSON object per line:

One real entry, using the document observed in section 2:

```json
{"doc_id": "rbi-13704",
 "rbi_id": 13704,
 "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13704&Mode=0",
 "circular_no": "RBI/2026-27/262",
 "dept_ref": "DOR.AML.REC.223/14.01.005/2026-27",
 "title": "Reserve Bank of India (Rural Co-operative Banks - Know Your Customer) Amendment Directions, 2026",
 "published_date": "2026-09-18",
 "content_sha256": "<sha256 of canonical_text(raw_html) at discovery time>"}
```

**Corrected, 2026-09-20 (see section 9).** This field was originally named `sha256`
and held the sha256 of the raw HTML response. It is renamed `content_sha256` and now
holds the sha256 of `canonical_text(raw_html)` instead — the raw-byte value could
never be reproduced by a later fetch of the same, unchanged document (section 9), so
the field name and its stated content were both wrong. `content_sha256` is a
best-effort snapshot, not a guaranteed-stable whole-document hash: see section 9 for
why, and for what this manifest can and cannot actually detect.

`doc_id` is stable and derived from the RBI id, so re-running discovery cannot renumber
existing documents.

### 5.2 The invariant

Extraction produces `document.text` exactly once: parse the notification content node,
drop script, style and navigation, decode entities, normalise to Unicode NFC, collapse
whitespace runs. From that moment the string is **immutable**, and every character offset
in the system indexes into precisely it. Nothing downstream re-normalises.

Both chunkers are therefore **slicers**. They choose boundaries; they never transform
characters. Consequently:

```
document.text[chunk.char_start:chunk.char_end] == chunk.text
```

holds by construction rather than by luck. The Phase 1 acceptance test asserts it, but the
design is what makes it true - a chunker that rewrote text (stripped a bullet, re-wrapped
a line, trimmed a span) could pass review and silently break every citation. Slicing is
the property; the test is the alarm.

### 5.3 Chunking

One protocol, two implementations, strategy name stored on every chunk so Phase 2 can
compare them on identical inputs.

```python
class Chunker(Protocol):
    name: str
    def chunk(self, text: str, doc: Document) -> list[Chunk]: ...
```

- **`StructuralChunker`** splits on RBI's own numbering - `4(1)(v)`, `(a)`, numbered
  paragraphs, annex headings. Fragments shorter than `min_chunk_chars` merge forward;
  spans longer than `max_chunk_chars` split at sentence boundaries.
- **`FixedWindowChunker`** is a `window_chars` window with `overlap_chars` overlap, snapped to word
  boundaries. It exists to make the structural chunker's Phase 2 numbers mean something;
  a structure-aware strategy with no baseline to beat is an unfalsifiable claim.

Both emit ordered, non-overlapping-beyond-declared-overlap spans that cover the document
with no dropped characters.

**Initial parameters.** These are starting values, not measurements. They are chosen to be
reasonable and are expected to be wrong; Phase 2 is what tells us by how much. Every one
is configurable, and changing any of them is a retrieval change, so `CLAUDE.md`'s rule
applies - a fresh `make eval` and a committed report before merge.

| Parameter | Initial value | Applies to |
|---|---|---|
| `min_chunk_chars` | 400 | `StructuralChunker` - fragments below this merge forward |
| `max_chunk_chars` | 2000 | `StructuralChunker` - spans above this split at sentence boundaries |
| `window_chars` | 1200 | `FixedWindowChunker` |
| `overlap_chars` | 200 | `FixedWindowChunker` |
| `min_document_chars` | 500 | Fetch validation, section 7.1 |
| `min_request_interval_s` | 2.0 | Fetcher, section 8 |
| `max_retries` | 3 | Fetcher, section 7.2 |

No value here may be quoted as a result. They are inputs.

### 5.4 Data model

```sql
documents(
  doc_id PK, rbi_id, url, circular_no, dept_ref, title,
  doc_type, published_date, effective_date, sha256, fetched_at, text
)

chunks(
  chunk_id PK, doc_id FK -> documents, strategy, ordinal,
  char_start, char_end, text, created_at,
  -- denormalised from documents, see below
  source_url, effective_date, doc_type
)
-- index on (doc_id, strategy)
```

`documents.text` is stored in Postgres, not only on disk, so the round-trip assertion
verifies against the database that retrieval will actually read from.

**On denormalising three columns onto `chunks`.** `CLAUDE.md` states that every chunk
*stores* `(doc_id, char_start, char_end, source_url, effective_date, doc_type)`. Resolving
the last three through a join to `documents` would be the normalised choice, and it would
not satisfy that sentence. The requirement is also the right one on its merits: a
retrieved chunk travels to the answering layer as a citation, and a citation that needs a
second query to say where it came from is a citation that can be separated from its
provenance. The columns are copied at ingest and are immutable thereafter, so the usual
denormalisation hazard - divergence under update - does not arise; a document revision
produces a hash mismatch and a re-ingest, not an in-place edit.

## 6. Data flow

```
manifest (committed)
   |
fetch ---> validate ---> data/raw/<doc_id>.html (cache, hash-verified)
   |                          |
   |                      extract
   |                          |
   |                   document.text  (canonical, immutable)
   |                          |
   |            +-------------+-------------+
   |            |                           |
   |     StructuralChunker          FixedWindowChunker
   |            |                           |
   +------------+---------------------------+
                          |
                      Postgres
```

A warm cache means `make ingest` performs zero network requests. That is what makes the
command reproducible and CI offline.

## 7. Failure handling

### 7.1 Content validation, the control that matters most

The round-trip invariant proves **fidelity**, not **correctness**. It would pass exactly as
happily on fifty identical bot-check pages: the text would be stored faithfully, the
chunks would slice it faithfully, and every test would be green while the corpus was
worthless. Because a blocked request returns HTTP 200 with an HTML body (section 2), the
status code cannot be trusted to catch this.

So every response passes a validation gate **before** it is written to the cache:

1. HTTP 200 and an HTML content type
2. Absence of the observed block markers `Unauthorised Access` and `Support ID:`
3. Presence of a circular signature - an `RBI/<year>/<number>` reference or a department
   code matching the observed shape
4. A minimum visible-text length

A response failing any check is an error. It is never cached, and never becomes a
document.

### 7.2 The rest

**Corrected, 2026-09-20 (see section 9).** The original row here read: "sha256
differs from manifest -> Hard failure naming the document and both hashes. RBI
revises circulars in place; this is what detects it." That assumed a stable
whole-document hash exists against this source. It does not. The corrected table:

| Condition | Behaviour |
|---|---|
| Live fetch: `content_sha256` matches manifest | Accept. |
| Live fetch: `content_sha256` differs, but header identity (`circular_no`, `dept_ref`, `published_date`) still matches manifest | Accept, logging a warning naming the document to stderr. Treated as page-chrome noise (a WAF token, a volatile widget), not a revision - see section 9. |
| Live fetch: `content_sha256` differs AND header identity also differs, or the header no longer parses | Hard failure naming the document, both hashes, and what header field(s) differed. RBI has changed this document's identity. The manifest is never auto-healed. |
| Warm cache: cached file's `content_sha256` differs from manifest | Hard failure, no tolerance. These bytes are ours, not the network's, so a mismatch here means disk corruption or tampering, not WAF/page-chrome noise - see section 9. |
| 4xx, 5xx, timeout | Bounded retry, exponential backoff with jitter. On exhaustion, record the failure, continue with remaining documents, and exit non-zero with a summary. |
| Extraction yields empty or short text | Hard failure |
| Re-run of `make ingest` | Idempotent. Chunks for `(doc_id, strategy)` are replaced transactionally; a failed document leaves no partial rows. |

Partial ingest is always visible. `make ingest` exits non-zero if any manifest entry
failed, and prints which.

## 8. Politeness, and an honest correction to PROMPT.md

`PROMPT.md` section 1 instructs: "Respect robots.txt and rate-limit fetches." The first
half cannot be honoured - `robots.txt` returns 418 to every client tried. Leaving the
sentence as written would put a claim in the repository that the code cannot support,
which is the same failure mode `CLAUDE.md` forbids for numbers.

The fetcher's floor, therefore:

- single concurrency, no parallel requests
- a minimum interval of 2.0 seconds between requests, configurable upward only
- an identifying User-Agent carrying a contact address
- the cache is authoritative, so a document is fetched once and never re-fetched

**Action required:** amend `PROMPT.md` section 1 to state that `robots.txt` is
unreachable and that these conservative defaults stand in its place. This is a
documentation change, not a code change, and it is listed in section 12.

## 9. Correction: no stable whole-document hash exists against this source

Section 7.2 originally claimed that a sha256 mismatch between a live fetch and the
manifest detects RBI revising a circular in place. That claim does not hold, on two
independent counts, both confirmed by repeated live fetches on 2026-09-20 (Task 10,
fix rounds 1-2) rather than the single-probe evidence in section 2. Leaving the
original claim in place would put the same kind of unsupportable statement in this
document that section 8 already had to correct for `robots.txt` - so, in the same
spirit, and just as bluntly:

1. **Raw response bytes are never reproducible.** `www.rbi.org.in` sits behind an F5
   BIG-IP WAF that injects a per-response `<script id="f5_cspm">` tag carrying a
   freshly randomised token into every response. Fetching the identical, unchanged
   URL three times, two seconds apart, produced three different raw-byte sha256
   values, confirmed on two independent documents. A raw-byte hash is therefore not a
   usable "has this document changed" signal here at all - it reports drift on every
   fetch, always, regardless of whether the document changed.
2. **Even the canonical extracted text is not perfectly stable.** Three fetches of one
   unchanged URL, two seconds apart, produced not one but **two** distinct
   `canonical_text` hashes. Diffing the two canonical texts localised the entire
   difference to page furniture: a `": "` inserted at one character offset and a
   `"kb"`/`"KB"` case difference at another, both inside what a PDF-size widget
   renders. The regulatory text itself was byte-identical across all three fetches;
   only the chrome around it varied.

**The fix.** `ManifestEntry`'s hash field is renamed `sha256` -> `content_sha256` and
now holds the sha256 of `canonical_text(raw_html)`, not of raw response bytes
(section 5.1). `Fetcher` verification becomes two-tier (section 7.2): a live fetch
whose `content_sha256` differs from the manifest is not immediately fatal - if the
document's header identity (`circular_no`, `dept_ref`, `published_date`) still
matches, the difference is logged as page-chrome noise and accepted; only a
`content_sha256` mismatch *combined with* a header identity mismatch (or a header that
no longer parses) is treated as a real revision. The warm-cache path keeps a strict,
untolerant check, because bytes already on disk are ours, not the network's, and do
not carry WAF/page-chrome volatility - a mismatch there is corruption or tampering,
not noise, and deserves to fail hard.

**What Phase 1 therefore actually guarantees, stated honestly.** Fetch-time
verification detects **identity drift**: RBI renumbering a circular, re-dating it, or
republishing it under a different department reference, because that changes the
parsed header, which is checked independently of the content hash. It does **not**
detect an **in-place body revision that leaves the header intact**, because no
whole-page hash - raw or canonical - is reproducible enough against this source to
serve as that signal, and no header-independent content hash was designed to survive
page-chrome noise.

A stronger guarantee - a content hash computed over the document body with page
furniture (widgets, WAF chrome) deliberately excluded - is **Phase 2 work**, not
Phase 1: deciding what counts as "furniture" versus "content" is exactly the kind of
judgment call `CLAUDE.md`'s measurement discipline says should be evidenced, not
assumed, and it is not needed to satisfy Phase 1's definition of done.

## 10. Testing

`CLAUDE.md` requires tests before implementation for anything in the retrieval or citation
path. Every unit in section 5 is in that path except `cli`.

| Test | Asserts |
|---|---|
| `test_offsets_roundtrip` | For every chunk of every ingested document, read back from Postgres: `documents.text[char_start:char_end] == chunks.text`. This is Phase 1's definition of done. |
| `test_chunkers_slice_only` | Both strategies, over real fixtures and adversarial generated text (Unicode, whitespace runs, nested numbering): slice identity holds, spans ordered, no characters dropped |
| `test_waf_page_is_rejected` | The captured block page, committed as a fixture, raises rather than parsing |
| `test_chrome_only_mismatch_is_accepted_when_header_matches` | A `content_sha256` divergence with intact header identity is accepted, with a warning, not a failure (section 9) |
| `test_header_identity_mismatch_is_fatal` | A `content_sha256` divergence *and* a header identity divergence fails loudly |
| `test_corrupted_cache_file_is_fatal` | A cached file whose bytes no longer match `content_sha256` fails loudly - no header-identity tolerance on the warm-cache path (section 9) |
| `test_extract_metadata` | Circular number, department reference and date parse from the real header line |
| `test_ingest_is_idempotent` | Two runs produce identical rows and the second performs zero network calls |
| `test_fetcher_rate_limit` | Minimum interval honoured, against a local stub server with a controlled clock |

**No test touches `rbi.org.in`.** Fixtures are two or three real cached pages plus one
captured block page, held in `tests/fixtures/` - test data, not bulk corpus
redistribution.

Postgres comes from `make up` locally and a service container in CI, so database-backed
tests always execute rather than being skipped into irrelevance.

## 11. Dependencies

Each gets one line in `docs/decisions.md` stating what it replaced and why, per
`CLAUDE.md`.

| Dependency | Role |
|---|---|
| httpx | HTTP client with timeouts and retry hooks |
| selectolax | HTML parsing; chosen for speed and a simple text-extraction path |
| SQLAlchemy | Schema definition and queries |
| Alembic | Migrations |
| psycopg | Postgres driver |
| pydantic-settings | Environment-driven configuration |

No embedding, vector or LLM dependency enters in Phase 1. Those belong to Phase 2 and
Phase 3 and are deliberately absent here.

## 12. Explicitly out of scope

Not built in Phase 0 or Phase 1, and not to be added opportunistically:

- Embeddings, Qdrant, retrieval of any kind - Phase 2
- The evaluation harness and golden set - Phase 2
- Answering, citation resolution, refusal - Phase 3
- FastAPI, Redis, tracing - Phase 4
- Hybrid search and reranking - excluded from v1 entirely, on the evidence in
  `plans/market.md`, and only admissible later with a measured before/after

One documentation change is required and is not optional: the `PROMPT.md` section 1
amendment described in section 8.

## 13. Traceability

| `PROMPT.md` requirement | Where it is satisfied |
|---|---|
| CI green on empty test suite | Section 4 |
| Fetcher with on-disk cache and rate limiting | Sections 5.1, 7.2, 8, 9 |
| Text extraction preserving character offsets | Section 5.2 |
| Postgres schema and migrations | Section 5.4 |
| Two chunking strategies behind one interface | Section 5.3 |
| 50+ real documents ingested by a documented command | Sections 5.1, 6 |
| Chunk slice round-trips against source text | Sections 5.2, 10 |
| Every chunk stores the full metadata tuple | Section 5.4 |
