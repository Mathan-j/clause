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

Extraction produces `document.text` exactly once: select the notification's own
content element (`#NotificationUser`, exposed as `CONTENT_SELECTOR` in
`clause.htmltext`), drop `<script>`, `<style>` and `<noscript>`, decode entities,
normalise to Unicode NFC, collapse whitespace runs. From that moment the string is
**immutable**, and every character offset in the system indexes into precisely it.
Nothing downstream re-normalises.

`clause.htmltext` exposes two functions, and the distinction matters:

- `visible_text(html)` returns the **whole page**, site template included. The
  validation gate uses it, because it is judging an untrusted response that may not
  be a document at all.
- `document_text(html)` returns **only the content element**, falling back to the
  whole body when that element is absent. `canonical_text` uses it, so no offset is
  ever assigned to site chrome.

**Correction, 2026-09-20, same register as section 9.** Two earlier versions of this
section were wrong in opposite directions. The first claimed navigation was dropped
when nothing implemented it. The second retracted that claim and recorded
navigation-stripping as the first Phase 2 change, on the reasoning that removing
chrome shifts every character offset and would invalidate a hand-labelled golden
set. That reasoning was right, and it is exactly why the change was made **now**
instead: no golden set exists yet, no retrieval baseline exists yet, and nothing
downstream depends on current offsets — so this was the cheapest moment it will ever
have. Measured over the 61-document cached corpus before the change, the surrounding
template accounted for 371,559 of 668,794 stored characters; the stored corpus is
now the 297,235 characters that are actually regulation.

Two consequences were handled with the change rather than left to be discovered:
every entry's `content_sha256` was re-derived from the warm cache, since all 61 were
computed over the old chrome-inclusive text and would otherwise have silently fallen
through to the header-identity branch forever (see section 9); and the header
identity of all 61 documents was re-parsed and confirmed unchanged first, so
re-deriving a hash could not mask a document that had actually changed.

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
whole-document hash exists against this source. It does not. A second correction,
also 2026-09-20: an intermediate revision of this table applied that check strictly
on a cache read and tolerantly on a live fetch. That asymmetry was itself wrong (see
section 9) — the same check now applies identically to both. The corrected table:

| Condition | Behaviour |
|---|---|
| `content_sha256` matches manifest | Accept. Applies identically whether the bytes came from a live fetch or the on-disk cache. |
| `content_sha256` differs, but header identity (`circular_no`, `dept_ref`, `published_date`) still matches manifest | Accept, logging a warning naming the document to stderr. Treated as page-chrome noise (a WAF token, a volatile widget), not a revision - see section 9. Applies identically on both paths. |
| `content_sha256` differs AND header identity also differs, or the header no longer parses | Hard failure naming the document, both hashes, and what header field(s) differed. RBI has changed this document's identity. The manifest is never auto-healed. Applies identically on both paths. |
| Cached file fails content validation (section 7.1) - truncated, a block page, no circular reference | Hard failure, before the hash check even runs. A cached file is validated exactly as a live fetch is; caching is not a bypass of section 7.1. |
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
documentation change, not a code change, and it is listed in section 13.

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
(section 5.1). `Fetcher` verification becomes two-tier (section 7.2): a fetch whose
`content_sha256` differs from the manifest is not immediately fatal - if the
document's header identity (`circular_no`, `dept_ref`, `published_date`) still
matches, the difference is logged as page-chrome noise and accepted; only a
`content_sha256` mismatch *combined with* a header identity mismatch (or a header that
no longer parses) is treated as a real revision.

**Corrected again, same day.** An intermediate version of this fix applied the
two-tier check only to a live fetch and kept the warm-cache path strict, reasoning
that cached bytes are "ours" and therefore free of WAF/page-chrome noise. That
reasoning was wrong: the bytes sitting in the cache may themselves have been
accepted through the tolerant branch when they were first fetched, in which case
they were never expected to match `content_sha256` exactly. A strict re-check would
then reject a perfectly good, already-accepted document on *every subsequent read*
forever - `make ingest` would succeed once and then fail on every later run, for
documents that are entirely fine, which reads as corruption when none occurred. The
same two-tier check now applies identically on both paths (`Fetcher._verify`,
parameterised only by which failure message to print). To compensate for removing
the cache path's strictness, a cached file is now also run through the section 7.1
content-validation gate before the hash check - the same gate a live fetch already
passes through - which still catches gross on-disk corruption (truncation, a block
page, a lost circular reference) without needing byte-exact hashing to do it.

**What Phase 1 therefore actually guarantees, stated honestly.** Fetch-time
verification detects **identity drift**: RBI renumbering a circular, re-dating it, or
republishing it under a different department reference, because that changes the
parsed header, which is checked independently of the content hash. It does **not**
detect an **in-place body revision that leaves the header intact**, because no
whole-page hash - raw or canonical - is reproducible enough against this source to
serve as that signal, and no header-independent content hash was designed to survive
page-chrome noise. Nor does it detect **cache corruption that happens to preserve
header identity and still passes the content-validation gate** - a narrow case (the
corrupted bytes would need to remain a plausible, sufficiently long circular with an
intact header) but a real gap, accepted rather than closed, because closing it needs
exactly the excluded-furniture content hash described below, not more special-casing
of the cache path.

A stronger guarantee - a content hash computed over the document body with page
furniture (widgets, WAF chrome) deliberately excluded - is **Phase 2 work**, not
Phase 1: deciding what counts as "furniture" versus "content" is exactly the kind of
judgment call `CLAUDE.md`'s measurement discipline says should be evidenced, not
assumed, and it is not needed to satisfy Phase 1's definition of done.

**Addendum, 2026-09-20 (final whole-branch review): a third, larger instability
source, and why it changes the conclusion above.** The two sources above (the WAF
token, the PDF-size widget) are not the only page furniture inside
`canonical_text`. Every page's footer also carries the literal string
`"Website last updated date: <date>"`, and that string is **inside**
`canonical_text`, not stripped by anything upstream of it — confirmed at character
7,980 of a cached document, reading `Sep 19, 2026`, and identical (same date
string) across all 61 documents in the cached corpus. This is a site-wide value,
not a per-document one, so it changes on whatever cadence RBI updates its site as a
whole, independent of whether any individual circular changed at all.

The corpus manifest was frozen the same day this date last rolled over, which is
the only reason a cold re-fetch today mismatches `content_sha256` for just 12 of 61
documents rather than all of them. **Once that site-wide date next rolls over, a
cold fetch will mismatch `content_sha256` for 61 of 61 documents, permanently** —
every document will take the tolerant, header-identity-only branch on every future
fetch, and `content_sha256` stops functioning as a drift signal at all rather than
being merely "best-effort" as stated above. This does not require Phase 2's
excluded-furniture content hash to be *foreseen*; it means that hash is not a
someday-nicer-to-have, it is what makes `content_sha256` mean anything again once
this date turns over.

Related, and worth stating plainly here rather than only in `.gitignore`: `data/raw/`
is gitignored, so what the committed manifest actually freezes is the URL list and
each document's header identity (`circular_no`/`dept_ref`/`published_date`) — **not**
the text. A fresh clone that re-fetches from empty will get different raw HTML (WAF
token, widget, footer date, at minimum) and therefore a different `documents.text`
and different character offsets than whatever `data/raw/` currently holds. That
matters directly for Phase 2: hand-labelled `char_start`/`char_end` ground truth is
only valid against the exact `data/raw/` cache it was labelled from, not against "the
manifest" in the abstract.

**Second addendum, 2026-09-20 (Phase 1 follow-ups): the footer date is no longer
inside the extracted text, and that retires the paragraph above.** Section 5.2's
navigation-stripping change made `canonical_text` select the notification's own
content element instead of the whole page body. `"Website last updated date: ..."`
lives in the page footer, outside that element, so it is no longer part of
`document.text` and no longer part of `content_sha256`. The "61 of 61, permanently"
outcome predicted above therefore cannot happen: a site-wide footer date rolling
over now changes nothing that is hashed.

The remaining known furniture inside the content element is the PDF-size widget
(`"( 275 kb )"`, whose casing varies between fetches). That is smaller and
document-adjacent rather than site-wide, but it is still enough to make a cold fetch
mismatch, so the header-identity fallback remains load-bearing and the two accepted
gaps stated above are unchanged. All 61 `content_sha256` values were re-derived from
the warm cache when extraction changed, and each document's header identity was
re-parsed and confirmed unchanged first, so the re-derivation could not mask a
document that had genuinely changed.

## 10. Correction: `doc_type` classification is unreliable on the real corpus

`_classify` (section 5, `clause.ingest.extract`) reads a fixed 600-character window
of text anchored on the header match (`CLASSIFY_WINDOW_CHARS`) and returns
`master_direction` if "master direction" appears in it, else `circular` if
"circular" appears, else `notification`. This was written without evidence that
the window and keyword choice actually separate the corpus's real document types,
and re-running it over the full 61-document cached corpus on 2026-09-20 (final
whole-branch review) shows it does not, on two independent counts:

1. **Zero `circular` labels.** The real corpus classifies as 29 `master_direction`
   and 32 `notification` — never `circular` — even though several of these
   documents are UAPA/WMD sanctions-list circulars that merely *cite* the KYC
   Master Direction in passing, not master directions themselves.
2. **Near-identical documents split both ways.** `rbi-12922` and `rbi-13310` share
   the same subject line ("Implementation of Section 51A of UAPA, 1967: Updates to
   UNSC's ... Sanctions List") and the same document type in substance, but
   `_classify` labels the first `master_direction` and the second `notification`.

   **Corrected, 2026-09-20.** An earlier version of this section attributed the
   split to the 600-character window boundary. That was asserted, not measured, and
   it is wrong: re-running the classifier over all 61 cached documents with the
   window widened from 600 to 2,000 characters changes **zero** labels. The actual
   discriminator is plain keyword presence in the text `_classify` hashes, which is
   `title + window` — a detail this section previously omitted. `rbi-12922` contains
   "master direction" in both its stored title and its window; `rbi-13310` contains
   it in neither. 18 of the 61 titles contain the phrase, so the manifest title —
   itself a fixed-length window of body prose, see section 5.1 — is doing much of
   the classifying. Tuning `CLASSIFY_WINDOW_CHARS` would not fix this.

`doc_type` is in `CLAUDE.md`'s non-negotiable provenance tuple, is denormalised
onto all 1,314 chunks on the stated grounds that it never changes after ingest
(section 5.4), and is Phase 2's planned metadata filter (`PROMPT.md` section 3).
A metadata filter this unreliable would silently exclude or include the wrong
documents from a retrieval query. **Phase 2 must either fix or retire `_classify`
before using `doc_type` as a retrieval filter** — this is not a Phase 1 fix,
per `CLAUDE.md`'s rule that a retrieval-affecting change needs measured
before/after evidence, which only Phase 2's eval harness can produce.
`tests/test_doc_type_classification.py` pins the current, known-flawed behaviour
(specific `doc_id -> doc_type` mappings, and the zero-`circular` fact) so a future
change to the classifier is visible as a deliberately-updated test, not a silent
behavior change.

## 11. Testing

`CLAUDE.md` requires tests before implementation for anything in the retrieval or citation
path. Every unit in section 5 is in that path except `cli`.

| Test | Asserts |
|---|---|
| `test_every_chunk_slice_roundtrips_against_stored_source` (`tests/test_ingest_acceptance.py`) | For every chunk of every ingested document, read back from Postgres: `documents.text[char_start:char_end] == chunks.text`. This is Phase 1's definition of done. |
| `test_fixed_window_chunks_are_pure_slices` (`tests/test_chunking.py`) / `test_invariant_holds_on_adversarial_input` (`tests/test_chunking_structural.py`) | Both strategies, over real fixtures and adversarial generated text (Unicode, whitespace runs, nested numbering): slice identity holds, spans ordered, no characters dropped |
| `test_block_page_is_rejected_despite_http_200` (`tests/test_validate.py`) | The captured block page, committed as a fixture, raises rather than parsing |
| `test_chrome_only_mismatch_is_accepted_when_header_matches` | A live fetch's `content_sha256` divergence with intact header identity is accepted, with a warning, not a failure (section 9) |
| `test_header_identity_mismatch_is_fatal` | A live fetch's `content_sha256` divergence *and* a header identity divergence fails loudly |
| `test_cached_chrome_only_mismatch_is_accepted_when_header_matches` | The same tolerance applies on the warm-cache path - the regression test for the strict-cache trap (section 9) |
| `test_cached_header_identity_mismatch_is_fatal` | A cached file's header identity divergence fails loudly, same as a live fetch |
| `test_cached_file_failing_content_gate_is_rejected` | A cached file that fails section 7.1 content validation (truncated, a block marker) is rejected before the hash check runs |
| `test_parse_header_reads_the_real_document` (`tests/test_extract.py`) | Circular number, department reference and date parse from the real header line |
| **not implemented** | Two runs produce identical rows and the second performs zero network calls. No test in the suite asserts this; `test_ingest_is_idempotent`, named here in an earlier draft, was never written. |
| `test_rate_limiter_waits_between_calls` (`tests/test_fetch.py`) | Minimum interval honoured, against a local stub server with a controlled clock |

**No test touches `rbi.org.in`.** Fixtures are two or three real cached pages plus one
captured block page, held in `tests/fixtures/` - test data, not bulk corpus
redistribution.

Postgres comes from `make up` locally and a service container in CI, so database-backed
tests always execute rather than being skipped into irrelevance.

## 12. Dependencies

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

## 13. Explicitly out of scope

Not built in Phase 0 or Phase 1, and not to be added opportunistically:

- Embeddings, Qdrant, retrieval of any kind - Phase 2
- The evaluation harness and golden set - Phase 2
- Answering, citation resolution, refusal - Phase 3
- FastAPI, Redis, tracing - Phase 4
- Hybrid search and reranking - excluded from v1 entirely, on the evidence in
  `plans/market.md`, and only admissible later with a measured before/after

One documentation change is required and is not optional: the `PROMPT.md` section 1
amendment described in section 8.

## 14. Traceability

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
