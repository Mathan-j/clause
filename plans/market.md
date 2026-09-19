# Market dataset

The evidence behind `PROMPT.md` section 2. Every figure here is produced by
`plans/market_table.py` from `plans/data/job_market_data.csv`. Nothing in this
file is estimated, remembered, or rounded by hand.

```
python plans/market_table.py           # print the table
python plans/market_table.py --check   # fail if the dataset stops reproducing it
```

## Provenance

| | |
|---|---|
| Source file | `plans/data/job_market_data.csv` (same records also as `.json`) |
| sha256 | `dfe58c57a3342750e53f6b284dee808d939c3417cbfb432afed8eb99ede0c830` |
| Snapshot date | 2026-09-18 (single snapshot, not a time series) |
| Postings | 138 |
| In band for 1-2 years | 61 |
| Job boards / career sites | 31 distinct source labels - 29 after merging variants (`Freehire`/`freehire`, `Foundit`/`foundit.in`). Includes Hirist, LinkedIn, Indeed India, Wellfound, Greenhouse, Ashby, amazon.jobs, Google Careers, Built In, We Work Remotely |
| Territories sampled | 8: indian_tech_boards, indian_product_cos, global_product_careers, services_mnc, consultancy_bigfour, ai_native_startups, staffing_volume, remote_global |
| Company tiers (all 138) | ai_startup 38, services_mnc 23, global_product 21, consultancy 20, other 14, indian_product 12, staffing 10 |

"In band" means the posting's stated minimum experience puts a candidate with
1-2 years in range. Across the 61 in-band postings the stated minimum is
0 years for 6, 0.5 for 1, 1 year for 25, and 2 years for 29.

## Method

Each capability is a case-insensitive regex over a named set of columns. The
rules are not commentary - they are the definition, and they live in `RULES` in
`plans/market_table.py`. Change a rule and the number changes.

| Capability | Columns searched | Pattern |
|---|---|---|
| Python | `languages` | `\bpython\b` |
| Agentic / tool calling | `agent_frameworks` | `agent`, `tool calling`, `function calling`, `langgraph`, `crewai`, `autogen`, `mcp`, `orchestration` |
| RAG / retrieval | `retrieval_vector` | `rag`, `retrieval`, `semantic search`, `embedding`, `grounding`, `chunking` |
| Cloud | `cloud` | `aws`, `azure`, `gcp`, `google cloud` |
| LangChain / LangGraph | `agent_frameworks` | `langchain`, `langgraph` |
| Vector databases | `retrieval_vector` | `vector`, `pinecone`, `faiss`, `weaviate`, `qdrant`, `chroma`, `milvus`, `pgvector`, `opensearch`, `azure ai search` |
| Evaluation / testing | `ops_deploy`, `deliverables` | `eval`, `a/b test`, `unit test`, `testing`, `benchmark` |
| FastAPI | `backend` | `fastapi` |
| Docker / CI-CD | `ops_deploy` | `docker`, `ci/cd`, `github actions`, `jenkins`, `kubernetes` |
| SQL / Postgres / Redis | `data`, `languages` | `sql`, `postgres`, `redis`, `mysql`, `mongo`, `nosql` |
| Observability | `ops_deploy` | `observability`, `monitoring`, `tracing`, `langsmith`, `langfuse`, `telemetry` |
| Guardrails / security | `ops_deploy`, `deliverables`, `soft_asks` | `guardrail`, `security`, `pii`, `compliance`, `safety`, `hallucination` |
| Reranking | all capability columns | `rerank`, `re-rank`, `cross-encoder` |
| Hybrid search | all capability columns | `hybrid search`, `hybrid retrieval`, `bm25`, `sparse retriev` |

The exact anchored forms are in the script; the table above drops the `\b`
word boundaries for readability.

## Capability demand, 61 in-band postings

| Capability | n | Share | `PROMPT.md` says | Delta |
|---|---:|---:|---:|---:|
| Python | 52 | 85% | 85% | 0 |
| Agentic / tool calling | 39 | 64% | 54% | +10 |
| RAG / retrieval | 32 | 52% | 51% | +1 |
| Evaluation / testing | 30 | 49% | 43% | +6 |
| Cloud (AWS/Azure/GCP) | 30 | 49% | 51% | -2 |
| LangChain / LangGraph | 28 | 46% | 48% | -2 |
| Vector databases | 28 | 46% | 44% | +2 |
| FastAPI | 21 | 34% | 34% | 0 |
| Docker / CI-CD | 20 | 33% | 33% | 0 |
| SQL / Postgres / Redis | 19 | 31% | 28% | +3 |
| Observability | 11 | 18% | 18% | 0 |
| Guardrails / security | 9 | 15% | 18% | -3 |
| Reranking | 2 | 3% | 3% | 0 |
| Hybrid search | 0 | 0% | 0% | 0 |

Six of fourteen reproduce exactly. The rest sit within 10 points, and the gaps
are definitional rather than arithmetic - `PROMPT.md`'s table was built with
slightly different matchers, and those matchers were not recorded. The largest
gap, agentic at +10, comes from counting LangChain-as-orchestration and generic
"AI agents" phrasing as agentic demand. Both readings are defensible; this file
states which one produced these numbers.

**Where the two tables disagree, this file is the one that reproduces.** Read
`PROMPT.md` section 2 as the argument and this file as the evidence.

## The two findings the architecture rests on

`PROMPT.md` refuses hybrid search and reranking in v1 on the strength of these
two rows, so they are checked harder than the rest.

**Reranking: 2 of 61 in-band (3%).** Both explicit, both at startups.

- Aruvee Intelligence, "AI Engineer - RAG & LLM" - `re-ranking`, listed
  alongside BGE, E5 and sentence-transformers
- Planso, "Junior AI Engineer" - `reranking`, listed alongside embeddings and
  RAG

Across all 138 postings regardless of band, reranking appears 7 times (5%). It
rises slightly with seniority and never becomes common.

**Hybrid search: 0 of 61 in-band (0%).** This row carries a caveat that a flat
"0%" hides. Across all 138 postings, hybrid search, BM25 or sparse retrieval
appears 6 times (4%), and the distribution is not random:

- Infosys, Capgemini, Capgemini Sogeti, EY - four of the six are services and
  consultancy firms
- Jobbycart Technologies, Kasmoprav - the other two

Hybrid search is not absent from the market. It is absent from the 1-2 year
band and concentrated in consultancies hiring above it. The decision to skip it
in v1 survives, but the honest phrasing is "not asked for at this level", not
"nobody wants it".

## The finding that does not support the brief

`PROMPT.md` argues that span-level citations are what make this project "not
chat with your PDF". That is a bet on differentiation, and the dataset does not
support it as a claim about demand:

| Term, searched across every column of the 61 in-band postings | n |
|---|---:|
| `citation` | 0 |
| `attribution` | 0 |
| `provenance` | 0 |
| `grounding` | 1 |
| `hallucination` | 1 |
| `source` | 1 |

**No in-band posting asks for citations.** The nearest neighbours, grounding
and hallucination, appear once each.

This does not invalidate the design. A portfolio project earns attention by
being what other candidates did not build, and 0% demand is consistent with 0%
supply. But it is a different argument from the one the rest of the table
makes, and it should be made in those words. Every other design decision in
`PROMPT.md` section 2 cites a percentage; this one cannot, and the README must
not imply otherwise.

## Limits of this measurement

State these next to the numbers anywhere they are quoted.

1. **Job ads are not jobs.** They are written by recruiters, copied between
   postings, and list aspirations as requirements. They measure what employers
   advertise, not what the work involves.
2. **One snapshot, one day.** 2026-09-18. No trend, no seasonality control. A
   second snapshot would be needed to claim any movement.
3. **Keyword matching, not reading.** A posting saying "we do not use
   LangChain" counts as a LangChain match. No posting was read to confirm
   intent.
4. **Selection bias in the sample frame.** 32 boards chosen by hand, skewed
   toward Indian boards and AI-native startups (38 of 138 are ai_startup).
   This is not a random sample and no weighting was applied.
5. **Absence of evidence is not evidence of absence.** A capability can be
   assumed rather than stated - testing and SQL are the obvious candidates. A
   0% row means "not written down", not "not wanted". The citations row above
   is the case where this matters most.
6. **Structured fields were extracted, not validated.** The pipe-separated
   capability columns were produced upstream of this repo. Their extraction was
   not independently audited; only the counting on top of them reproduces here.
   The `source` column carries the same board under two labels in two cases
   (`Freehire`/`freehire`, `Foundit`/`foundit.in`), so the other free-text
   columns should be assumed to contain similar inconsistencies. The capability
   regexes are case-insensitive, which absorbs this class of error for the
   table above but not for any per-source cut.
7. **Coverage gaps in the raw fields.** Salary is present for 20 of 61 in-band
   postings and work mode for 29 of 61. Any cut by pay or remote status would
   draw on under half the sample, so none is attempted.

## What this licenses

Supported by the table:

- Retrieval evaluation as a first-class deliverable (49% in-band)
- A named cloud deployment rather than a Dockerfile alone (49%)
- FastAPI over alternatives (34%)
- Postgres and Redis in the stack (31%)
- Skipping hybrid search and reranking in v1 (0% and 3% in-band)

Not supported by the table, to be argued on its own merits:

- Span-level citations as the differentiator (0% in-band)
- Measured refusal quality - guardrails is 15%, but no in-band posting asks for
  a refusal rate specifically

## Rules for citing this file

- `plans/market.md` is a **design input**, not a project result. It describes a
  job market, not the behaviour of `clause`.
- Numbers from this file never appear in `reports/`. `reports/` holds measured
  system performance produced by `make eval`. Mixing the two would let a market
  percentage be mistaken for a recall number.
- Quoting any figure from here means quoting limit 1 and limit 4 with it.
- If the dataset is replaced, run `python plans/market_table.py --check`. It
  fails on a changed sha256 or a changed count, so this document cannot drift
  away from its evidence unnoticed.
