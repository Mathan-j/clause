"""A small HTTP surface over the answering pipeline.

This is a thin shell. Everything it exposes -- retrieval, the refusal gate, the
citation contract -- already exists and is already tested; the API adds no
behaviour of its own beyond serialising it. That is deliberate: an endpoint
that reimplements part of the pipeline is an endpoint that can disagree with
the evaluation harness, and then neither number means anything.

The model is loaded once, at startup. Loading it per request would add a
couple of seconds to every call and would let two concurrent requests hold two
copies of a 2 GB file in memory on a machine that does not have room for one
spare.
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from clause.answering.base import Answerer, StubAnswerer
from clause.answering.llm import LlamaAnswerer
from clause.answering.refusal import DEFAULT_THRESHOLD
from clause.answering.schema import Answer, AnsweringError, Refusal
from clause.config import Settings, get_settings
from clause.db.schema import DocumentRow
from clause.db.session import make_engine
from clause.embed import Encoder
from clause.evaluation.answer_run import ANSWER_STRATEGY, answer_one
from clause.retrieve import search

STATIC_DIR = Path(__file__).parent / "static"

#: How many chunks retrieval fetches for the API. The same depth the evaluation
#: uses, so a question asked here scores the way it would score there.
API_RETRIEVAL_DEPTH = 10


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class _State:
    """Everything expensive, built once."""

    settings: Settings
    encoder: Encoder
    client: QdrantClient
    answerer: Answerer
    engine: sa.Engine
    model_name: str


state = _State()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    state.settings = get_settings()
    state.encoder = Encoder(state.settings.embedding_model)
    state.client = QdrantClient(url=state.settings.qdrant_url, timeout=30)
    state.engine = make_engine(state.settings.database_url)
    try:
        state.answerer = LlamaAnswerer(state.settings.answer_model_path, n_ctx=8192)
        state.model_name = state.settings.answer_model_path.name
    except Exception:
        # Without the GGUF the rest of the system is still worth serving: the
        # retrieval, the refusal gate and the citation contract all work, and
        # the stub makes that visible rather than returning a 500 on every ask.
        state.answerer = StubAnswerer()
        state.model_name = "stub (no GGUF model present)"
    yield


app = FastAPI(title="clause", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    with Session(state.engine) as session:
        documents = session.scalar(sa.select(sa.func.count()).select_from(DocumentRow))
    collections = [c.name for c in state.client.get_collections().collections]
    return {
        "documents": documents,
        "collections": collections,
        "model": state.model_name,
        "strategy": ANSWER_STRATEGY,
        "default_threshold": DEFAULT_THRESHOLD,
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    started = time.monotonic()
    hits = search(
        state.client,
        state.encoder,
        ANSWER_STRATEGY,
        request.question,
        limit=API_RETRIEVAL_DEPTH,
    )
    retrieved_ms = int((time.monotonic() - started) * 1000)

    try:
        with Session(state.engine) as session:
            result = answer_one(
                request.question, hits, state.answerer, session, request.threshold
            )
    except AnsweringError as exc:
        # The citation contract refused to let this answer through. Surfacing it
        # as a failure rather than a degraded answer is the whole point: an
        # answer with a broken citation looks exactly like a sound one.
        return {
            "kind": "contract_failure",
            "detail": str(exc),
            "error_type": type(exc).__name__,
            "top_score": hits[0].score if hits else None,
            "timing_ms": {"retrieval": retrieved_ms},
        }

    total_ms = int((time.monotonic() - started) * 1000)
    timing = {"retrieval": retrieved_ms, "total": total_ms}

    if isinstance(result, Refusal):
        return {
            "kind": "refusal",
            "reason": result.reason,
            "top_score": result.top_score,
            "threshold": request.threshold,
            "timing_ms": timing,
        }

    assert isinstance(result, Answer)
    return {
        "kind": "answer",
        "sentences": [asdict(s) for s in result.sentences],
        "citations": [
            {
                "doc_id": c.doc_id,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "source_url": c.source_url,
                "doc_type": c.doc_type,
                "published_date": c.published_date.isoformat(),
                "text": c.text,
            }
            for c in result.citations
        ],
        "top_score": hits[0].score if hits else None,
        "threshold": request.threshold,
        "timing_ms": timing,
    }


@app.get("/api/document/{doc_id}")
def document(doc_id: str) -> dict[str, Any]:
    """The stored text of one document, so a citation's span can be shown in place.

    This is what makes a span-level citation checkable rather than merely
    printed: the page slices the same `[char_start:char_end]` out of the same
    text the resolver did, and the reader sees the claim sitting in its source.
    """
    with Session(state.engine) as session:
        row = session.get(DocumentRow, doc_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no document {doc_id!r}")
        return {
            "doc_id": row.doc_id,
            "title": row.title,
            "url": row.url,
            "doc_type": row.doc_type,
            "published_date": row.published_date.isoformat(),
            "text": row.text,
        }
