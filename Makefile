.PHONY: install lint test migrate ingest index eval gate serve up down

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

eval:
	uv run python -m clause.cli eval

gate:
	uv run python -m clause.cli gate

up:
	docker compose up -d

down:
	docker compose down
