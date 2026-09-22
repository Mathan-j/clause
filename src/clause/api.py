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

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
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

#: The committed answering report. The UI reads its limitation figures from
#: here rather than hardcoding them, so the page can only ever state numbers
#: that exist in a committed artifact -- the same rule CLAUDE.md applies to
#: documentation. No report, no numbers shown.
ANSWERS_REPORT = Path("reports/answers.json")

#: The committed retrieval evaluation. Served verbatim so the page renders the
#: same figures `make eval` wrote and CI gates -- the UI cannot show a number
#: that is not in a committed artifact.
EVAL_REPORT = Path("reports/eval.json")

#: How many chunks retrieval fetches for the API. The same depth the evaluation
#: uses, so a question asked here scores the way it would score there.
API_RETRIEVAL_DEPTH = 10


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    strategy: str = Field(default=ANSWER_STRATEGY)
    published_after: date | None = None
    entities: list[str] | None = None


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
        request.strategy,
        request.question,
        limit=API_RETRIEVAL_DEPTH,
        published_after=request.published_after,
        entities=request.entities or None,
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
        "strategy": request.strategy,
        "timing_ms": timing,
    }


@app.get("/api/evidence")
def evidence() -> dict[str, Any]:
    """The committed evaluation reports, served verbatim.

    Both are optional: the page renders whichever exists and says so when one
    does not, rather than showing a figure with no artifact behind it.
    """
    out: dict[str, Any] = {"retrieval": None, "answering": None}
    for key, path in (("retrieval", EVAL_REPORT), ("answering", ANSWERS_REPORT)):
        if path.exists():
            try:
                out[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                out[key] = None
    return out


@app.get("/api/entities")
def entities() -> dict[str, Any]:
    """Regulated-entity values present in the corpus, for the filter control."""
    with Session(state.engine) as session:
        rows = session.scalars(sa.select(DocumentRow.regulated_entity)).all()
    seen: set[str] = set()
    for row in rows:
        seen.update(row or [])
    return {"entities": sorted(seen)}


@app.get("/api/limitations")
def limitations() -> dict[str, Any]:
    """Measured limitations, read from the committed report.

    Returns `{"measured": false}` when no report is committed. The page then
    shows nothing rather than a remembered figure: a number on screen with no
    artifact behind it is exactly the failure this project forbids.
    """
    if not ANSWERS_REPORT.exists():
        return {"measured": False}
    try:
        report = json.loads(ANSWERS_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"measured": False}

    sweep = report.get("sweep", [])
    default = state.settings.answer_threshold if hasattr(state, "settings") else 0.35
    row = min(
        sweep, key=lambda r: abs(float(r.get("threshold", 0)) - default), default=None
    )
    if row is None:
        return {"measured": False}
    return {
        "measured": True,
        "threshold": row.get("threshold"),
        "adversarial_refused": row.get("adversarial_refused"),
        "adversarial_total": (row.get("adversarial_refused", 0) or 0)
        + (row.get("adversarial_answered", 0) or 0),
        "false_refusals": row.get("false_refusals"),
        "answered": row.get("answered"),
        "support_mean": (report.get("support") or {}).get("mean"),
        "resolution_rate": report.get("resolution_rate"),
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
