.PHONY: install lint test migrate ingest index eval serve up down

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

index:
	uv run python -m clause.cli index

up:
	docker compose up -d

down:
	docker compose down
