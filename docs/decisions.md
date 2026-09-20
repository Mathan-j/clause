# Decisions

One line per dependency: what it is, what it replaced, why.

- ruff — lint + format. Replaces flake8 + isort + black; one tool, one config, far faster.
- mypy — static types. Chosen over pyright to keep the toolchain inside the Python package set.
- pytest — test runner. Replaces unittest; fixtures and parametrisation are worth the dependency.
- pydantic-settings — environment-driven config with validation. Replaces hand-rolled os.environ reads; we get type coercion and failure at startup rather than at first use.
- selectolax — HTML parsing. Replaces BeautifulSoup + lxml; a C parser with a direct text() path, and we only need text extraction, not a tree API.
- httpx — HTTP client. Replaces requests; native timeouts, and MockTransport lets every fetcher test run without a socket.
- SQLAlchemy 2.0 — schema and queries. Replaces hand-written SQL; typed declarative models keep the row shape and the domain types in one place.
- Alembic — migrations. Chosen over ad-hoc DDL scripts so the schema has a history; CI builds it from empty via `alembic upgrade head`/`downgrade base` in the test suite's `db_session` fixture (`tests/conftest.py`).
- psycopg (binary) — Postgres driver. Replaces psycopg2; version 3 is maintained and ships wheels.
- sentence-transformers — text embedding. Replaces calling a hosted embedding API; the model runs locally, so an eval run costs nothing per query and is reproducible offline.
- onnxruntime — execution provider for the embedding model, per CLAUDE.md's stated stack. Chosen over raw PyTorch inference for faster CPU encoding. (Not for a smaller install: `torch` is a base dependency of `sentence-transformers` regardless of which backend runs inference, so it is installed either way — only the forward pass changes.)
- optimum[onnxruntime] — sentence-transformers' `backend="onnx"` loads through this; it is not optional despite not being named directly in the brief. Version pinned by uv's resolver to one compatible with sentence-transformers' `transformers` requirement (optimum-onnx currently caps `transformers<4.58`, which forced sentence-transformers down to 5.7.0 rather than the just-released 6.1.0). `pyproject.toml` pins `sentence-transformers<6.0.0` explicitly for the same reason: 6.0.0 is the version where `transformers>=5.0.0` first appears in sentence-transformers' own requirement, so an unbounded `uv lock --upgrade` would re-resolve straight into this conflict. Lift the bound once `optimum-onnx` supports `transformers` 5.x.
- qdrant-client — vector store client. Qdrant is named in CLAUDE.md's stack; the client is its official one and supports the server-side payload filtering the metadata filters need.
