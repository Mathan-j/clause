# clause

Regulatory answer engine over public RBI circulars. Every factual sentence in an answer
must resolve to an exact character span in an exact source document. When retrieval is
weak the system refuses instead of answering.

Full brief: @PROMPT.md
Current state: @STATUS.md

## Stack
Python 3.12 · FastAPI · PostgreSQL · Qdrant · Redis · sentence-transformers (ONNX)
· pytest · ruff · mypy · Docker · GitHub Actions · GCP Cloud Run

## Non-negotiables
- No number appears in documentation unless it came from a committed file under `reports/`.
- Every chunk stores `(doc_id, char_start, char_end, source_url, effective_date, doc_type)`.
  A test asserts the slice round-trips against source text.
- An answer containing a citation that does not resolve is a hard failure, never a warning.
- Retrieval changes require a fresh `make eval` run and a committed report before merge.
- No hybrid search, no reranker, no fine-tuning without measured before/after evidence.

## Commands
```
make install     # uv sync
make lint        # ruff + mypy
make test        # pytest
make ingest      # fetch + extract + chunk + embed + index
make eval        # run the golden set, write reports/eval.md
make serve       # uvicorn, local
make up          # docker compose
```

## Conventions
- Conventional commits. The body explains why, not what.
- New dependency requires one line in `docs/decisions.md`.
- Tests first for anything in the retrieval or citation path.
- Ask before spending money, pushing to a remote, or deploying.
