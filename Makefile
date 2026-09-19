.PHONY: install lint test ingest eval serve up down

install:
	uv sync

lint:
	uv run ruff check .
	uv run mypy

test:
	uv run pytest

ingest:
	uv run python -m clause.cli ingest --manifest data/corpus/kyc.manifest.jsonl

up:
	docker compose up -d

down:
	docker compose down
