"""Recompute every number in plans/market.md from plans/data/job_market_data.csv.

Usage:  python plans/market_table.py [--check]

No figure may appear in plans/market.md unless this script prints it. --check
re-derives the table and exits non-zero if the dataset no longer reproduces the
committed counts, so a swapped dataset fails loudly instead of silently
invalidating the document.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data" / "job_market_data.csv"

# Columns holding pipe-separated capability tokens.
CAPABILITY_COLUMNS = [
    "languages", "backend", "frontend", "retrieval_vector", "agent_frameworks",
    "llm_providers", "cloud", "data", "ops_deploy", "ml_deep", "soft_asks",
    "deliverables",
]

# (label, columns searched, regex). Searching is case-insensitive substring
# matching over the joined text of the named columns. These rules ARE the
# definition of each capability — change one and the number changes.
RULES: list[tuple[str, list[str], str]] = [
    ("Python", ["languages"], r"\bpython\b"),
    ("Agentic / tool calling", ["agent_frameworks"],
     r"agent|tool calling|function calling|langgraph|crewai|autogen|mcp\b|orchestration"),
    ("RAG / retrieval", ["retrieval_vector"],
     r"\brag\b|retrieval|semantic search|embedding|grounding|chunking"),
    ("Cloud (AWS/Azure/GCP)", ["cloud"], r"\baws\b|azure|\bgcp\b|google cloud"),
    ("LangChain / LangGraph", ["agent_frameworks"], r"langchain|langgraph"),
    ("Vector databases", ["retrieval_vector"],
     r"vector|pinecone|faiss|weaviate|qdrant|chroma|milvus|pgvector|opensearch|azure ai search"),
    ("Evaluation / testing", ["ops_deploy", "deliverables"],
     r"\beval|a/b test|unit test|testing|benchmark"),
    ("FastAPI", ["backend"], r"fastapi"),
    ("Docker / CI-CD", ["ops_deploy"], r"docker|ci/cd|github actions|jenkins|kubernetes"),
    ("SQL / Postgres / Redis", ["data", "languages"],
     r"\bsql\b|postgres|redis|mysql|mongo|nosql"),
    ("Observability", ["ops_deploy"],
     r"observability|monitoring|tracing|langsmith|langfuse|telemetry"),
    ("Guardrails / security", ["ops_deploy", "deliverables", "soft_asks"],
     r"guardrail|security|\bpii\b|compliance|safety|hallucination"),
    ("Reranking", CAPABILITY_COLUMNS, r"rerank|re-rank|cross-encoder"),
    ("Hybrid search", CAPABILITY_COLUMNS, r"hybrid search|hybrid retrieval|bm25|sparse retriev"),
]

# Counts committed to plans/market.md, as n-of-61 in-band postings.
COMMITTED = {
    "Python": 52, "Agentic / tool calling": 39, "RAG / retrieval": 32,
    "Cloud (AWS/Azure/GCP)": 30, "LangChain / LangGraph": 28, "Vector databases": 28,
    "Evaluation / testing": 30, "FastAPI": 21, "Docker / CI-CD": 20,
    "SQL / Postgres / Redis": 19, "Observability": 11, "Guardrails / security": 9,
    "Reranking": 2, "Hybrid search": 0,
}
COMMITTED_TOTAL = 138
COMMITTED_IN_BAND = 61
COMMITTED_SHA256 = "dfe58c57a3342750e53f6b284dee808d939c3417cbfb432afed8eb99ede0c830"


def load() -> tuple[list[dict], list[dict]]:
    with DATA.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    in_band = [r for r in rows if r["in_band_for_1plus_yrs"].strip().lower() == "true"]
    return rows, in_band


def blob(row: dict, columns: list[str]) -> str:
    return " | ".join((row[c] or "") for c in columns).lower()


def counts(in_band: list[dict]) -> dict[str, int]:
    return {
        label: sum(1 for r in in_band if re.search(pattern, blob(r, columns)))
        for label, columns, pattern in RULES
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the dataset no longer reproduces market.md")
    args = parser.parse_args()

    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    rows, in_band = load()
    n = len(in_band)
    result = counts(in_band)

    print(f"dataset   {DATA.name}  sha256={digest[:16]}...")
    print(f"postings  {len(rows)} total, {n} in band (1-2 yrs)\n")
    print(f"{'Capability':26}{'n':>4}{'share of in-band':>19}")
    print("-" * 49)
    for label, _, _ in RULES:
        c = result[label]
        print(f"{label:26}{c:>4}{round(100 * c / n):>17}%")

    if not args.check:
        return 0

    problems = []
    if digest != COMMITTED_SHA256:
        problems.append(f"dataset sha256 changed: {digest} != {COMMITTED_SHA256}")
    if len(rows) != COMMITTED_TOTAL:
        problems.append(f"total postings {len(rows)} != {COMMITTED_TOTAL}")
    if n != COMMITTED_IN_BAND:
        problems.append(f"in-band postings {n} != {COMMITTED_IN_BAND}")
    for label, expected in COMMITTED.items():
        if result[label] != expected:
            problems.append(f"{label}: {result[label]} != committed {expected}")

    if problems:
        print("\nFAIL - plans/market.md no longer matches the dataset:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("\nOK - plans/market.md reproduces from the committed dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
