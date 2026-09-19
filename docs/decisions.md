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
