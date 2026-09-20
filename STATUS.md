# Status

**Phase:** 2 — complete

| Phase | State | Evidence |
|---|---|---|
| 0 scaffold | done | CI green |
| 1 ingestion | done | `make ingest` + `pytest tests/test_ingest_acceptance.py` |
| 2 retrieval + eval | done | `make eval` writes `reports/eval.md` / `reports/eval.json`; `make gate` (wired into CI after the test suite) fails the build on a recall regression against the committed `reports/baseline.json` or on a stale report |
| 3 citations + refusal | not started | — |
| 4 service | not started | — |
| 5 deploy | not started | — |

## Measured numbers

Not recorded here. Per `CLAUDE.md`, no number belongs in this file unless it comes
from a committed file under `reports/` — read `reports/eval.md` for what was
actually measured, and `reports/baseline.json` for the committed regression
reference `make gate` checks it against.
