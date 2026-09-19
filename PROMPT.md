# clause — master brief for Claude Code

Paste this as your first message in a fresh Claude Code session started in the empty
`clause/` repo. Read it end to end before writing any code.

---

## 1. What we are building

`clause` is a **regulatory answer engine with span-level citations**.

A user asks a question about Indian financial regulation. The system returns an answer
where **every factual sentence carries a citation that resolves to an exact character
span in an exact source document** — not a document-level "source: circular.pdf", but
"this claim comes from characters 4,812–5,104 of RBI Master Direction DOR.CRE.REC.04/
21.04.048/2024-25". A reader can click a sentence and land on the words it came from.

When retrieval is weak, the system **refuses** rather than answering. A wrong answer
about a capital adequacy rule is worse than no answer, and refusal quality is a scored
metric here, not an afterthought.

**Corpus:** public RBI circulars, master directions and notifications. These are
government publications, freely readable. We store extracted text plus source URLs and
never redistribute bulk PDFs. Respect robots.txt and rate-limit fetches.

## 2. Why this project, and why built this way

This is not an arbitrary choice. It comes from an analysis of 138 real job postings,
61 of which sit at 1–2 years of experience. The capability demand in those 61:

| Capability | Share of in-band postings |
|---|---:|
| Python | 85% |
| Agentic / tool calling | 54% |
| RAG / retrieval | 51% |
| Cloud (AWS/Azure/GCP) | 51% |
| LangChain / LangGraph | 48% |
| Vector databases | 44% |
| Evaluation / testing | 43% |
| FastAPI | 34% |
| Docker / CI-CD | 33% |
| SQL / Postgres / Redis | 28% |
| Observability | 18% |
| Guardrails / security | 18% |
| Reranking | 3% |
| Hybrid search | 0% |

Design decisions follow directly from that table:

- **Retrieval evaluation is a first-class deliverable, not a README section.** 43% of
  postings ask for it and almost no candidate at this level can show it. It is the
  single strongest differentiator available, so it gets built in phase 2 — before the
  answering layer — so that everything after it is measured from the start rather than
  retrofitted.
- **Deploys to a named cloud (GCP Cloud Run).** 51% of postings name a cloud. A
  Dockerfile alone does not satisfy that; a live URL does.
- **No hybrid search, no cross-encoder reranking in v1.** Zero postings mention hybrid
  search and only 3% mention reranking. They are blog-post topics, not hiring criteria.
  If the evaluation harness shows they improve the numbers, add them in a later phase
  and publish the before/after. Never add them because they sound advanced.
- **Citations at span level.** This is the thing that makes the project not "chat with
  your PDF", which is the most saturated portfolio category in this field.
- **Refusal is measured.** Guardrails appear in 18% of postings and almost nobody
  demonstrates them quantitatively.

Read `plans/market.md` for the full dataset if it is present in the repo.

## 3. Architecture

```
RBI source URLs
      |
  fetch + cache (raw HTML/PDF kept on disk, never re-fetched)
      |
  extract -> normalised text + stable character offsets
      |
  structure-aware chunking (clause / section boundaries, two strategies to compare)
      |
  embed (sentence-transformers, ONNX runtime)
      |
  Qdrant (vectors)  +  PostgreSQL (documents, chunks, offsets, metadata)
      |
  retrieve (dense + metadata filter: date, document type, regulated entity)
      |
  answer with citation contract  ->  refuse if retrieval score below threshold
      |
  FastAPI (streaming)  ->  Docker  ->  GCP Cloud Run
```

Every chunk carries `(doc_id, char_start, char_end, source_url, effective_date,
doc_type)`. The citation contract is enforced in code: the answering layer emits
structured output where each sentence references chunk ids, and a validator rejects any
response whose citations do not resolve. An unresolvable citation is a hard failure, not
a warning.

## 4. Build phases and definition of done

Work through these in order. **Do not start a phase until the previous phase's tests
pass and its eval numbers are committed.**

**Phase 0 — scaffold**
Repo structure, `pyproject.toml`, ruff + mypy configured, pytest running, GitHub Actions
workflow that runs lint and tests on push. Done when CI is green on an empty test suite.

**Phase 1 — ingestion and storage**
Fetcher with on-disk cache and rate limiting. Text extraction preserving character
offsets. Postgres schema and migrations. Two chunking strategies behind one interface.
Done when a documented command ingests at least 50 real documents and `pytest` passes,
including a test asserting that every chunk's `char_start`/`char_end` slice of the source
text equals the chunk's stored text.

**Phase 2 — retrieval and evaluation (the important one)**
Embedding pipeline. Qdrant indexing. Dense retrieval with metadata filters. A golden set
of at least 60 questions with ground-truth chunk ids, hand-labelled and stored as a
versioned JSONL file. Metrics: recall@1/5/10, MRR, precision@1, all broken down per
question bucket (definitional, numeric threshold, procedural, cross-reference). A
markdown eval report generated by a script. A CI gate that fails the build if recall@5
drops below the committed baseline.
Done when `make eval` prints the table, `reports/eval.md` is committed, and CI fails on a
deliberate regression.

**Phase 3 — answering with enforced citations**
Structured-output answering. Citation resolver and validator. Refusal path with a scored
threshold. Metrics added to the harness: citation accuracy, faithfulness, refusal
correctness on an adversarial set of out-of-corpus questions.
Done when no response can pass validation with an unresolvable citation, and the
adversarial set shows the refusal rate.

**Phase 4 — service**
FastAPI with streaming responses, request tracing, token and cost accounting per request,
p50/p95 latency. Health and readiness endpoints. Redis cache for repeated queries.
Done when `docker compose up` serves the API locally and the trace of a single request
can be read end to end.

**Phase 5 — deploy and document**
Cloud Run deployment, CI/CD from GitHub Actions. README written as a product spec:
problem, architecture diagram, measured results with the limits of the measurement stated
next to them, cost per query, latency percentiles, and what does not work yet.
Done when a public URL serves a query and the README numbers match `reports/eval.md`.

## 5. Rules for how you work

- **Never invent numbers.** Every figure in the README, in a docstring, or in a commit
  message must come from a committed artefact under `reports/`. If a number is not
  measured, say it is not measured.
- **Tests before implementation** for anything in the retrieval and citation path.
- **Small commits.** One logical change each, conventional commit messages, and the
  commit body says why, not what.
- **Do not add a dependency** without writing one line in `docs/decisions.md` saying what
  it replaced and why.
- **Do not add hybrid search, reranking, or a fine-tuned model** unless the eval harness
  shows a measured improvement, and then publish the before/after.
- **Ask before** anything that spends money, pushes to a remote, or deploys.
- **When you finish a phase**, update `STATUS.md` with what is done, what the numbers are,
  and what is next.

## 6. Start here

1. Read this file and `CLAUDE.md`.
2. Produce a plan for Phase 0 and Phase 1 only. Do not plan the whole project.
3. Wait for approval before writing code.
