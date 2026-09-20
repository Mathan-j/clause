# Status

**Phase:** 2 — complete

| Phase | State | Evidence |
|---|---|---|
| 0 scaffold | done | `.github/workflows/ci.yml` configured (lint + test); not yet executed anywhere -- no remote is configured (`git remote -v` is empty) and nothing has been pushed |
| 1 ingestion | done | `make ingest` + `pytest tests/test_ingest_acceptance.py` |
| 2 retrieval + eval | done | `make eval` writes `reports/eval.md` / `reports/eval.json`; `make gate` is wired into `.github/workflows/ci.yml` to fail the build on a recall regression against the committed `reports/baseline.json` or on a stale report -- configured, not yet executed, for the same reason as above |
| 3 citations + refusal | not started | — |
| 4 service | not started | — |
| 5 deploy | not started | — |

## Measured numbers

Not recorded here. Per `CLAUDE.md`, no number belongs in this file unless it comes
from a committed file under `reports/` — read `reports/eval.md` for what was
actually measured, and `reports/baseline.json` for the committed regression
reference `make gate` checks it against.
