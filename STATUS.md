# Status

**Phase:** 3 — complete

| Phase | State | Evidence |
|---|---|---|
| 0 scaffold | done | `.github/workflows/ci.yml` configured (lint + test); not yet executed anywhere -- no remote is configured (`git remote -v` is empty) and nothing has been pushed |
| 1 ingestion | done | `make ingest` + `pytest tests/test_ingest_acceptance.py` |
| 2 retrieval + eval | done | `make eval` writes `reports/eval.md` / `reports/eval.json`; `make gate` is wired into `.github/workflows/ci.yml` to fail the build on a recall regression against the committed `reports/baseline.json` or on a stale report -- configured, not yet executed, for the same reason as above |
| 3 citations + refusal | done | `make answer-eval` writes `reports/answers.md` / `reports/answers.json`, both committed from a completed run. It scores the golden set and an out-of-corpus adversarial set with an enforced citation contract and a refusal-threshold sweep. Deliberately not gated by CI -- it needs a local GGUF model and runs far longer than a push-triggered check should. |
| 4 service | not started | — |
| 5 deploy | not started | — |

## Measured numbers

Not recorded here. Per `CLAUDE.md`, no number belongs in this file unless it comes
from a committed file under `reports/` — read `reports/eval.md` for retrieval,
`reports/baseline.json` for the committed regression reference `make gate` checks
it against, and `reports/answers.md` for the answering and refusal figures.
