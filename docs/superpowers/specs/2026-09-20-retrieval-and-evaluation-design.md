# Design: Phase 2 retrieval and evaluation

Date: 2026-09-20
Scope: `PROMPT.md` Phase 2 only. Phases 3-5 are deliberately not designed here.
Status: awaiting author review, then implementation plan.
Predecessor: `docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md`

## 1. What this covers

Phase 2 makes retrieval **measurable**. `PROMPT.md` calls it "the important one" and
sequences it deliberately before the answering layer, so that everything built after
it is measured from the start rather than retrofitted.

Phase 2's definition of done, restated from `PROMPT.md`:

> `make eval` prints the table, `reports/eval.md` is committed, and CI fails on a
> deliberate regression.

It requires: an embedding pipeline, Qdrant indexing, dense retrieval with metadata
filters, a golden set of at least 60 questions with ground truth, metrics
(recall@1/5/10, MRR, precision@1) broken down per question bucket, a generated
markdown report, and a CI gate against a committed baseline.

## 2. Evidence and its status

The corpus figures below were measured on 2026-09-20 against the live database and
the warm `data/raw/` cache, by the commands named beside them. They are **design
inputs, not results**. `reports/` does not exist yet — Phase 2 is what creates it —
so per `CLAUDE.md` no figure here may be quoted as a project result. Once `make eval`
runs, the corpus shape it records in `reports/eval.json` supersedes this table.

| Fact | Value | Reproduced by |
|---|---|---|
| Documents | 61 | `select count(*) from documents` |
| Chunks, `fixed_window` | 318 (median 1,194 chars) | `select ... from chunks group by strategy` |
| Chunks, `structural` | 336 (median 829 chars) | same |
| Structural chunks per document | 2 to 24 | same |
| Corpus date span | 2023-07-04 to 2026-09-18 | `select min/max(published_date)` |
| Stored characters | 297,235 | `sum(length(text))` |

Two facts about the environment shape the design:

1. **Qdrant's default port 6333 is already occupied** on the development machine by
   an unrelated project's container. Clause publishes on **6335** (verified free),
   the same collision-avoidance the database made necessary on 5434.
2. **`chunk_id` is a serial primary key that renumbers on every ingest.** Observed
   directly: after one re-ingest the first three rows carried ids 1353, 1373, 1315.
   This is the single most consequential fact for the golden set — see section 5.

## 3. Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Ground-truth production | Drafted by the model, sampled and verified by a human | Fabricated ground truth makes every downstream metric meaningless; full hand-labelling of 60+ items is disproportionate |
| Ground-truth cardinality | A **set** of acceptable spans per question | Much of the corpus is near-identical sanctions-list updates; forcing one right answer would punish a retriever that returns an equally correct sibling |
| Metadata filters | `published_date` + `regulated_entity`; `doc_type` **retired** | `doc_type` is measured noise (predecessor spec section 10); regulated entities are real, structured, and already inside the stored text |
| CI regression gate | Gate the committed report, with a provenance fingerprint | CI has no corpus and must not scrape RBI; this is the only option where the gated number is the headline number |

Rejected alternatives, recorded so they are not silently revisited:

- **Fully model-generated ground truth** was rejected because the report could then
  only claim "retrieval measured against model-generated ground truth", which is a
  materially weaker claim than `PROMPT.md` asks for, and risks questions answerable
  by lexical overlap alone — inflating recall against nothing.
- **One acceptable chunk per question** was rejected because on this corpus it would
  systematically understate recall wherever several circulars say the same thing.
- **Vendoring a micro-corpus so CI recomputes a real number** was rejected for now:
  it means two golden sets and two baselines, and CI would gate a number nobody
  quotes. It remains a reasonable later addition as a pipeline smoke test.
- **Repairing `doc_type`** was rejected as Phase 2 work: its fix cannot be validated
  until the harness that measures it exists, and `CLAUDE.md` forbids retrieval
  changes without measured before/after evidence.

## 4. Units

| Unit | Purpose | Depends on |
|---|---|---|
| `clause.entities` | Parse regulated entities from the addressee block | `clause.ingest.extract` |
| `clause.embed` | Encode chunks and queries via sentence-transformers (ONNX) | - |
| `clause.index` | Qdrant collection lifecycle and upsert | `clause.embed`, `clause.db` |
| `clause.retrieve` | Dense search with metadata filters | `clause.index` |
| `clause.evaluation.golden` | Golden-set schema, loader, span validation | `clause.db` |
| `clause.evaluation.metrics` | recall@k, MRR, the overlap predicate | - (pure) |
| `clause.evaluation.report` | Render `reports/eval.md` and `reports/eval.json` | metrics |
| `clause.evaluation.gate` | Compare a committed report against the baseline | - |
| `scripts/verify_golden.py` | Human verification workflow for the drafted labels | golden |

`clause.evaluation.metrics` has no dependency on the database, Qdrant or the
embedding model by design: it is the one part of Phase 2 that CI can execute, so it
is kept pure and fully unit-tested.

### 4.1 Embedding model

Baseline is **`sentence-transformers/all-MiniLM-L6-v2`** (384 dimensions), run
through the ONNX runtime per `CLAUDE.md`'s stated stack.

It is chosen for being *unsurprising*, not for being best: it is symmetric, so there
is no asymmetric query-prefix convention to get silently wrong, and it is the most
widely reproduced baseline available. `BAAI/bge-small-en-v1.5` is the obvious
upgrade candidate at the same dimensionality. The entire purpose of building the
harness before tuning anything is that the swap can then be **measured** rather than
asserted — the same discipline `CLAUDE.md` already applies to rerankers and hybrid
search.

The model identifier and revision are recorded in every report's fingerprint
(section 10), so a number can always be traced to the model that produced it.

### 4.2 Indexing

**One Qdrant collection per chunking strategy** — `clause_fixed_window` and
`clause_structural` — not one shared collection with a `strategy` payload filter.

Approximate-nearest-neighbour recall depends on what else is in the index. Mixing
both strategies into one collection would let a query retrieve chunks from the other
strategy and would make the two strategies' numbers a function of each other, which
destroys the comparison the two strategies exist to enable.

Each point carries the chunk's payload: `doc_id`, `strategy`, `ordinal`,
`char_start`, `char_end`, `published_date`, `regulated_entity`, `source_url`.

## 5. Ground truth is character spans, not chunk ids

**This deviates from `PROMPT.md`, deliberately.** Section 4 of the brief specifies "a
golden set of at least 60 questions with ground-truth chunk ids". Chunk ids cannot
carry that role in this codebase:

- `chunk_id` is a serial primary key and **renumbers on every ingest** (section 2).
- The natural key `(doc_id, strategy, ordinal)` is worse for this purpose: ordinals
  shift whenever a chunking parameter changes, which is exactly what Phase 2 exists
  to experiment with. Ground truth tied to ordinals would need re-labelling after
  every experiment, and a measurement discipline that expensive is one that gets
  skipped.

Ground truth is therefore a **character span in a document**:

```json
{"doc_id": "rbi-13704", "char_start": 1840, "char_end": 2103, "content_sha256": "..."}
```

A retrieval **hits** when any returned chunk's `[char_start, char_end)` overlaps any
acceptable span for that question, within the same `doc_id`. Overlap, not
containment: a chunk boundary that splits the answer still retrieved the answer.

This is strictly better here for three reasons. It is stable across re-chunking, so
a chunking experiment costs nothing in re-labelling. **One golden set serves both
strategies**, so the comparison is against identical ground truth rather than two
separately-labelled sets. And it is the same span-level primitive the project's
entire citation thesis rests on — the golden set is labelled in the units the
product ultimately promises.

`content_sha256` pins each span to the exact document text it was labelled against.
Phase 1 proved this matters: changing extraction shifted every offset in the corpus.
Section 11 makes a mismatch fatal.

## 6. The golden set

Versioned JSONL at `data/golden/kyc-v1.jsonl`, one question per line, committed.

```json
{"qid": "q001",
 "question": "Which customers require enhanced due diligence under the KYC Directions?",
 "bucket": "definitional",
 "answers": [{"doc_id": "rbi-13704", "char_start": 1840, "char_end": 2103,
              "content_sha256": "..."}],
 "provenance": "drafted",
 "notes": ""}
```

`bucket` is one of `definitional`, `numeric_threshold`, `procedural`,
`cross_reference` — `PROMPT.md`'s four, unchanged.

`provenance` is one of `drafted`, `human_verified`, `human_corrected`. It starts at
`drafted` for every question and is advanced by `scripts/verify_golden.py`. The
report states the resulting split, so a reader can see exactly how much of the
ground truth a human actually touched rather than inferring it.

**Composition:** at least 60 questions, at least 15 per bucket. Questions are drafted
by reading chunks and **paraphrasing**, never by lifting phrasing from the source. A
question that echoes its source verbatim measures lexical overlap rather than
retrieval, and would inflate every number in the report.

**Verification workflow:** `scripts/verify_golden.py` presents a question alongside
its acceptable spans rendered in surrounding document context, and records the
human's judgement — confirm, correct the span, or reject the question. It writes
provenance back into the JSONL. The sample is chosen to cover all four buckets.

**Cross-reference questions** carry spans in more than one document. Hit semantics
remain "any acceptable span" uniformly across buckets; there is deliberately no
all-of-them metric. Cross-reference difficulty then surfaces honestly as lower
recall in that bucket — the query matches the amendment while the definition sits in
the Master Direction — which is the signal the bucket breakdown exists to expose. A
second metric would be machinery ahead of need.

## 7. Metrics

Computed per strategy, and within each strategy per bucket and overall.

**Retrieval depth is fixed at 10** for every evaluation run — the largest k reported.
Every metric below is computed from that single ranked list of 10, so the numbers in
one row are always mutually consistent. The depth is recorded in the report's
fingerprint, so a future change to it is visible rather than silently shifting every
figure.

| Metric | Definition |
|---|---|
| recall@k, k ∈ {1, 5, 10} | Fraction of questions where at least one acceptable span was hit within the top k results |
| MRR (MRR@10) | Mean over all questions of 1/rank of the first hitting result, where rank is 1-based within the top 10; **0 when no result in the top 10 hits**. Reported as MRR@10, never bare "MRR", because the cutoff changes the number |
| precision@1 | See below |

**`precision@1` is arithmetically identical to `recall@1` under these semantics.**
Both are "did the single top result hit". `PROMPT.md` lists them as two metrics; on
a set-of-acceptable-spans golden set they are one number. The report prints it once,
under both names, with a footnote stating the identity. Printing the same figure
twice as though it were two independent pieces of evidence would be exactly the kind
of number-inflation `CLAUDE.md` exists to prevent.

A genuine `precision@k` for k > 1 — what fraction of the top k results are
acceptable — is **not** computable from this golden set, because it would require
every chunk to be labelled acceptable-or-not, not just the answers. The report says
so rather than approximating it.

## 8. Retrieval and filters

`clause.retrieve` performs dense search over one strategy's collection, with
optional filters applied server-side by Qdrant:

- `published_date` — range filter (on or after, on or before)
- `regulated_entity` — match-any against the parsed entity list

`doc_type` is **not** offered as a filter. The predecessor spec's section 10 records
the measurement: zero `circular` labels across the corpus, near-identical documents
split both ways, and the manifest title doing much of the classifying. Exposing a
filter on a field known to be noise would invite Phase 3 to filter on it.

### 8.1 `regulated_entity`

RBI circulars address their audience explicitly, immediately after the header line
and before the salutation:

> The Chairpersons/ CEOs of the Commercial Banks, Small Finance Banks, Payment
> Banks, Urban Co-operative Banks, Rural Co-operative Banks, Regional Rural Banks,
> Local Area Banks, Non-Banking Financial Companies, Asset Reconstruction
> Companies, All India Financial Institutions

`clause.entities` parses that block into a normalised list against a closed
vocabulary of RBI-regulated entity classes. A document whose addressee block does
not parse gets an empty list rather than a guess, and the ingest reports how many
documents that affected — an entity filter is only useful if its absence is visible.

Storage: a new nullable `regulated_entity` column on `documents`, denormalised onto
`chunks` alongside the existing provenance columns, added by migration `0003`.
Denormalisation follows the same reasoning already recorded for `source_url`,
`effective_date` and `doc_type`: a retrieved chunk should carry what a filter needs
without a second query.

## 9. Data flow

```
documents + chunks (Postgres)
      |
  clause.entities ---> regulated_entity (migration 0003)
      |
  clause.embed (ONNX) ---> vectors
      |
  clause.index ---> Qdrant: clause_fixed_window | clause_structural   (port 6335)
      |
data/golden/kyc-v1.jsonl ---> clause.evaluation.golden (validates spans)
      |
  clause.retrieve (dense + date/entity filters)
      |
  clause.evaluation.metrics (pure)
      |
  reports/eval.md (human)  +  reports/eval.json (CI)
      |
  clause.evaluation.gate  <---  reports/baseline.json
```

## 10. The report, the fingerprint and the gate

`make eval` writes two artifacts:

- **`reports/eval.md`** — the human-readable table: per strategy, per bucket, per
  metric, plus the golden set's provenance split and the limits of the measurement
  stated next to the numbers.
- **`reports/eval.json`** — the same numbers for machines, plus a **provenance
  fingerprint**.

The fingerprint records: the manifest's content hash, per-strategy chunk counts, the
embedding model identifier and revision, the retrieval depth (10), the golden
set's path and content hash, and the git commit of the code that produced it.

`clause.evaluation.gate` runs in CI and fails when either:

1. `recall@5` for either strategy is below the corresponding value in
   `reports/baseline.json`, or
2. the fingerprint in `reports/eval.json` does not match the current tree — a stale
   or hand-edited report.

That second check is what makes gating a committed artifact defensible. Without it,
"CI fails on a deliberate regression" would be satisfiable by not re-running the
eval. With it, a deliberate regression fails two ways: lower the numbers and the
baseline check fires; skip the re-run and the staleness check fires.

`reports/baseline.json` is created and updated by an explicit human act, never
automatically by `make eval`. A harness that promotes its own baseline cannot detect
a regression.

## 11. Failure handling

**The failure that matters most.** Every golden-set span carries the
`content_sha256` of the document it was labelled against. On a mismatch, `make eval`
**refuses to run**, naming the affected `qid`s and documents, rather than scoring
against offsets that no longer point where they did. Phase 1 supplied the lesson
directly: when extraction changed, all 61 manifest hashes went stale and the full
test suite still passed, because a tolerant path absorbed every mismatch — a green
result concealing a dead check. This one is not tolerant.

| Condition | Behaviour |
|---|---|
| Golden-set span's `content_sha256` ≠ the stored document's | Hard failure naming the affected questions. Never scored. |
| Qdrant unreachable | `make eval` fails naming the connection target; Qdrant-backed tests skip |
| Embedding model not present in the local cache | `make eval` fails with the command that warms it. It must never download mid-measurement: a silent first-run download makes the first report's timing and reproducibility a lie |
| Vector dimension ≠ collection dimension | Recreate the collection rather than upsert into a mismatched index |
| `reports/baseline.json` absent | `make eval` still writes its artifacts; the gate reports "no baseline committed" and passes, because a first run has nothing to regress against |
| Retrieval returns no results for a question | Scored as a miss, counted, and listed in the report |
| A document's addressee block does not parse | Empty entity list, counted and reported; never a guessed entity |

## 12. Testing

`CLAUDE.md` requires tests first for anything in the retrieval or citation path.
That is all of Phase 2 except the report renderer.

| Test | Asserts |
|---|---|
| `test_metrics_*` | recall@k, MRR and the overlap predicate against synthetic spans, including boundary cases: touching-but-not-overlapping spans, a span exactly at a chunk edge, duplicate acceptable spans, and a question with no hit |
| `test_golden_schema` | Every committed question parses, has a legal bucket, ≥1 answer span, and spans within document bounds |
| `test_golden_spans_resolve` | Every committed span's `content_sha256` matches the stored document — the regression test for section 11's first row |
| `test_entities_*` | Entity parsing against the committed HTML fixtures, including a document whose addressee block is absent |
| `test_index_*`, `test_retrieve_*` | Against a live Qdrant; skipped when unreachable |
| `test_gate_*` | The baseline comparison and the staleness check, against synthetic report JSON — no infrastructure needed |
| `test_eval_end_to_end` | Gated on a warm `data/raw/` cache and a live Qdrant, like the Phase 1 acceptance test |

**What CI can honestly run:** the metric unit tests, the golden-set schema tests and
the gate tests. Those need no corpus, no Qdrant and no model. Retrieval and
end-to-end tests skip in CI for the same reason the Phase 1 acceptance test does —
`data/raw/` is gitignored and this project does not scrape a public regulator on
every push. That is less coverage than it sounds like it should be, and it is the
honest maximum under the constraint.

## 13. Dependencies

Each gets one line in `docs/decisions.md` stating what it replaced and why.

| Dependency | Role |
|---|---|
| `sentence-transformers` | Encoding, with the ONNX backend per `CLAUDE.md`'s stack |
| `onnxruntime` | ONNX execution provider |
| `qdrant-client` | Vector store client |

No LLM client, no reranker, no hybrid-search library enters in Phase 2.

## 14. Explicitly out of scope

- **Reranking and hybrid search.** Still excluded — but note that after this phase
  they become *admissible*, because the harness can finally produce the measured
  before/after `CLAUDE.md` requires. That is the point of building it first.
- **Answering, citation resolution, refusal** — Phase 3.
- **Repairing `doc_type`** — retired as a filter, not fixed. Section 3.
- **Fine-tuning any model.**
- **A second, vendored micro-corpus for CI** — considered and deferred, section 3.

## 15. Carried from Phase 1

- **Navigation stripping is done.** The predecessor spec scheduled it as the first
  Phase 2 change, to land before any hand-labelling. It was completed in the Phase 1
  follow-ups, which is why the golden set can be labelled against the current text.
- **`doc_type` is retired as a filter** rather than repaired, per section 3.
- **Housekeeping:** the predecessor spec's section 11 testing table names four tests
  that exist under different names and one, `test_ingest_is_idempotent`, that does
  not exist at all. Phase 2 corrects that table while it is adding to it.

## 16. Traceability

| `PROMPT.md` Phase 2 requirement | Where it is satisfied |
|---|---|
| Embedding pipeline | Section 4.1 |
| Qdrant indexing | Section 4.2 |
| Dense retrieval with metadata filters | Section 8 |
| Golden set, ≥60 questions, versioned JSONL | Section 6 |
| Ground truth per question | Section 5 — **as spans, not chunk ids; deviation justified there** |
| recall@1/5/10, MRR, precision@1 | Section 7 — reported as MRR@10; **precision@1 ≡ recall@1, stated there** |
| Per-bucket breakdown | Sections 6, 7 |
| Markdown report generated by a script | Section 10 |
| CI gate failing below the committed baseline | Section 10 |
| Done when CI fails on a deliberate regression | Section 10 — both failure routes |
