.PHONY: install lint test migrate ingest eval serve up down

install:
	uv sync

lint:
	uv run ruff check .
	uv run mypy

test:
	uv run pytest

migrate:
	uv run alembic upgrade head

ingest: migrate
	uv run python -m clause.cli ingest --manifest data/corpus/kyc.manifest.jsonl

up:
	docker compose up -d

down:
	docker compose down
