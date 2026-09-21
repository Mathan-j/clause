# Phase 3 — Answering with enforced citations

**Status:** design approved 2026-09-21. Supersedes nothing; extends the Phase 2
harness (`docs/superpowers/specs/2026-09-20-retrieval-and-evaluation-design.md`).

## 1. What this phase delivers

An answering layer over the Phase 2 retriever, where every factual sentence carries
citations that resolve to exact character spans, and where the system refuses rather
than answers when retrieval is weak.

`PROMPT.md`'s definition of done: *"no response can pass validation with an
unresolvable citation, and the adversarial set shows the refusal rate."*

## 2. The constraint that shapes everything

Phase 2 measured `recall@5 = 0.446` for `fixed_window` and `0.338` for `structural`
(`reports/eval.md`). **More than half of golden questions do not retrieve a correct
span in the top five.**

That is the governing fact of this phase. A system that answers whenever it retrieves
something will cite the wrong span more often than the right one. A system that refuses
whenever retrieval looks weak will refuse most questions. Phase 3 does not fix
retrieval; it makes the choice between those two failures explicit, measured, and
tunable.

Consequently this phase publishes a **threshold sweep**, not a single refusal rate. The
deliverable is the curve, not a number someone picked.

## 3. Decisions taken, with their reasons

| Decision | Choice | Why |
|---|---|---|
| Answer engine | Local model, no hosted API | User constraint: no per-run cost. |
| Runtime | `llama-cpp-python==0.3.19` | Latest (0.3.35) is source-only on PyPI; this machine has no MSVC and no cmake. 0.3.19 has a prebuilt `cp312-win_amd64` wheel on the maintainer's index. |
| Model | `Qwen2.5-3B-Instruct` Q4_K_M (~2 GB) | Fits 4.5 GB free RAM. The 1.5B is ~2x faster but materially worse at selecting the correct chunk id, which is the one output that must be right. |
| Schema compliance | GBNF grammar (constrained decoding) | Makes a malformed citation *impossible to emit* rather than something to validate after the fact — the same move as `make_chunk` slicing its own text. |
| Refusal basis | Retrieval score threshold | Deterministic, testable without the model, and calibratable into a published curve. |
| Adversarial set | ~30 questions, near-miss weighted | A set that refuses trivially measures nothing. See §7. |
| Dependency placement | Optional extra `[answer]` | CI never generates answers. The base install must stay free of a 2 GB model and a C++ extension. |

### 3.1 What is deliberately not built

- **No hosted LLM fallback.** Adding one would reintroduce the cost the user excluded.
- **No entailment/NLI faithfulness judge.** See §6.3 — it is not affordable here and a
  3B model grading its own output is worthless.
- **No reranking.** `CLAUDE.md` forbids it without measured before/after evidence.
- **No answering inside `make eval`.** See §8.

## 4. Architecture

```
question
   |
   +--> retrieve.search(...)            -> list[Hit]          (Phase 2, unchanged)
   |
   +--> refusal.decide(hits, threshold) -> Refusal | Proceed   (no model call)
   |
   +--> Answerer.answer(question, hits) -> AnswerDraft         (llama.cpp + GBNF)
   |
   +--> resolver.resolve(draft, session)-> list[Citation]      (raises if unresolvable)
   |
   +--> validate.enforce(draft, cites)  -> Answer | raises
```

Five units. Four need no model:

| Module | Responsibility | Model? |
|---|---|---|
| `answering/schema.py` | `AnswerDraft`, `Answer`, `Refusal`, `Citation`, `Sentence` | no |
| `answering/resolver.py` | chunk_id -> span + provenance; slices stored text | no |
| `answering/validate.py` | enforces the citation contract | no |
| `answering/refusal.py` | threshold decision + calibration sweep | no |
| `answering/llm.py` | the only file that touches `llama_cpp` | **yes** |

`Answerer` is a `Protocol`. `StubAnswerer` (deterministic, in-repo) satisfies it, so the
whole pipeline is exercisable in CI with no model present.

## 5. The citation contract

### 5.1 Three distinct failures, kept distinct

- **Unresolvable** — the cited `chunk_id` does not exist in the corpus.
  **Hard failure.** Raises `UnresolvableCitationError`. No answer is returned.
  This is `CLAUDE.md`'s "an answer containing a citation that does not resolve is a hard
  failure, never a warning", implemented literally.
- **Uncited** — a sentence marked factual carries no citation. **Hard failure.**
- **Unsupported** — resolves, but the span may not support the claim. **Not** a hard
  failure; measured and reported (§6.3). Collapsing this into the first two would mean
  claiming an entailment guarantee the system does not have.

### 5.2 Resolution is by slicing, not by trust

`resolver.resolve` does not accept span text from the model. It takes the `chunk_id`,
reads `(doc_id, char_start, char_end)` from the database, and slices
`DocumentRow.text[char_start:char_end]`. A citation's text is therefore *derived from
the corpus by construction* and cannot disagree with it.

### 5.3 Provenance carried, and its known gap

A `Citation` carries `doc_id`, `char_start`, `char_end`, `source_url`, `published_date`,
`doc_type`, and the sliced `text`.

**`effective_date` is NULL for all 61 documents** (verified against the live corpus).
`CLAUDE.md`'s provenance tuple names it, and this phase cannot populate it without
changing extraction and re-ingesting, which would unfreeze the corpus and invalidate
every committed number. The field is carried where present and the report states the
gap. Resolving it — populate the field, or amend the tuple to name `published_date`,
which is what retrieval actually filters on — is deferred and recorded in §11.

## 6. Metrics

### 6.1 Citation resolution rate

Fraction of returned answers whose every citation resolves. **This is 1.0 by
construction** — an unresolvable citation raises before an answer exists. It is reported
anyway, because a number that must be 1.0 is a live assertion that the contract held,
and a drop below 1.0 would mean the enforcement path itself broke.

### 6.2 Answer / refusal split

Per threshold: how many questions were answered, refused, or failed validation.

### 6.3 Support signal — an explicit proxy, not faithfulness

For each factual sentence, the fraction of its content terms (numbers, capitalised
terms, and quoted strings) that appear in its cited span.

**This is not entailment and the report must say so in those words.** It catches the
gross failure — a sentence citing a span that shares no vocabulary with it — and misses
the subtle one — a fluent paraphrase that reverses the meaning. A real faithfulness
metric needs a judge model, which is excluded by §3.1.

Reported as a distribution, never as a single "faithfulness score", precisely so it
cannot be quoted as one.

### 6.4 Refusal correctness

On the adversarial set (§7): a refusal is **correct**. An answer is **incorrect** —
the question is unanswerable from this corpus by construction.
On the golden set: a refusal on a question whose answer *was* retrievable is a
**false refusal**.

Both are reported per threshold, as a curve.

## 7. The adversarial set

`data/adversarial/out-of-corpus-v1.jsonl`, ~30 questions, each with `qid`, `question`,
`kind`, and `why_unanswerable`.

Weighted toward questions that **retrieve well but cannot be answered**, because a set
that refuses trivially measures nothing:

| kind | what it tests |
|---|---|
| `adjacent_domain` | RBI topics outside KYC/AML — capital adequacy, priority sector lending. Retrieves RBI-shaped text, answer absent. |
| `uncovered_kyc` | KYC/AML questions the frozen corpus genuinely does not cover. The hardest kind. |
| `time_shifted` | Questions about pre-2023 circulars absent from the corpus (corpus spans 2023-07-04 to 2026-09-18). |
| `wrong_jurisdiction` | Non-RBI regulators. Expected to refuse easily; included as a floor, not as the bulk. |

Each question records its retrieval score at authoring time, so the set's own difficulty
is measured rather than asserted.

## 8. Where this plugs in

**`make eval` is unchanged.** Retrieval evaluation stays fast and stays the thing the CI
gate protects. Answering is slow — a 3B model on a 15 W CPU takes roughly 30-60 s per
question, so ~95 questions is a 45-90 minute run — and must not sit inside a command
that is expected to be quick.

New: `make answer-eval` -> `reports/answers.md` and `reports/answers.json`, carrying
their own fingerprint (model file sha256, quantisation, threshold sweep, corpus and
golden hashes).

**The CI gate does not gate answering numbers.** CI cannot run the model. Gating a
number CI cannot reproduce is the "dead check" pattern this project has now corrected
eight times. The answering report is committed evidence, not a gate.

## 9. Testing

| Test | Needs |
|---|---|
| resolver: unresolvable id raises | Postgres |
| resolver: sliced text round-trips against source | Postgres |
| validate: uncited factual sentence raises | nothing |
| validate: resolvable-but-unsupported passes and is counted | nothing |
| refusal: threshold boundary, both sides | nothing |
| refusal: sweep is monotone in threshold | nothing |
| end-to-end with `StubAnswerer` | Postgres + Qdrant |
| `llm.py`: GBNF output parses and ids are in-range | model (`@pytest.mark.llm`) |

The `llm` marker follows the existing `db` / `qdrant` / `model` convention and skips
when the model file is absent. **Every test above the last runs in CI.**

## 10. Failure handling

| Condition | Behaviour |
|---|---|
| Model file absent | `ModelNotAvailableError` naming the expected path and how to fetch it. Never downloads mid-run — same rule as the embedding encoder. |
| Grammar rejects all candidates | Raise. A silently unconstrained fallback would void the schema guarantee. |
| Citation unresolvable | `UnresolvableCitationError`. Hard failure. |
| Factual sentence uncited | `UncitedClaimError`. Hard failure. |
| Retrieval returns nothing | Refuse, recorded as `no_candidates`, distinct from a score-threshold refusal. |
| Postgres unreachable during resolution | Propagate. A citation that cannot be checked must not be reported as checked. |

## 11. Deferred, recorded here so they are not lost

1. `effective_date` NULL across the corpus (§5.3) — needs a decision, then re-ingest.
2. `corpus_sha256` over sorted `(doc_id, sha256)` pairs, to pin corpus *content* in the
   fingerprint rather than only the URL list and a chunk count.
3. Per-question miss list in `reports/eval.md` — spec §11 of the Phase 2 design promised
   it; it is absent.
4. Embedding model *revision* in the fingerprint, not just its name.
5. The mechanism behind `fixed_window`'s raw recall lead — its 200-char overlap gives it
   ~43% more acceptable chunks per question (5.91 vs 4.14). The chance floors encode this;
   the report does not state it.
