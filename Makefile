.PHONY: install lint test migrate ingest index eval answer-eval gate serve up down

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

answer-eval:
	uv run python -m clause.cli answer-eval

gate:
	uv run python -m clause.cli gate

up:
	docker compose up -d

down:
	docker compose down

serve:
	uv run uvicorn clause.api:app --reload --port 8000
