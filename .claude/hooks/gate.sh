#!/usr/bin/env bash
# clause — Stop-hook quality gate.
#
# Runs when Claude Code decides its turn is over. Exit 2 refuses the stop and
# hands stderr back to Claude as its next instruction; exit 0 lets it finish.
#
# Order: ruff -> pytest -> retrieval-eval freshness (phase 2 onward).
#
# Safety: this hook keeps its own consecutive-block counter and surrenders
# after CLAUSE_MAX_LOOPS (default 5) so an unfixable failure comes back to you
# instead of burning tokens in a circle. Claude Code may also apply its own
# stop-hook block cap; do not rely on that, this counter is the real floor.
#
# Read this file before trusting it. Hooks run on your machine, as you, with
# your full filesystem and network access.

set -u

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 0

COUNT_FILE="$PROJECT_DIR/.claude/.gate-count"
MAX_LOOPS="${CLAUSE_MAX_LOOPS:-5}"

# Drain the hook JSON on stdin; we do not need any of its fields.
cat >/dev/null 2>&1 || true

pass() { rm -f "$COUNT_FILE"; exit 0; }

surrender() {
  rm -f "$COUNT_FILE"
  printf '{"systemMessage":"clause gate: still failing after %s attempts — stopping the loop and handing back to you. Run `make lint` and `make test` to see it."}\n' "$MAX_LOOPS"
  exit 0
}

# ---- loop guard -------------------------------------------------------------
count=0
[ -f "$COUNT_FILE" ] && count=$(tr -dc '0-9' < "$COUNT_FILE")
[ -z "$count" ] && count=0
if [ "$count" -ge "$MAX_LOOPS" ]; then
  surrender
fi

# ---- nothing to gate yet ----------------------------------------------------
# Phase 0 has not landed: no project file, so there is no lint or test to run.
if [ ! -f "pyproject.toml" ]; then
  pass
fi

# ---- runner ------------------------------------------------------------------
# Prefer `uv run` so we hit the project venv; fall back to bare tools.
MISSING="__CLAUSE_TOOL_MISSING__"

# Resolve a tool from the project venv first, then PATH. We deliberately do not
# shell out to `uv run`: it creates .venv and syncs dependencies as a side
# effect, and a Stop hook must never mutate the environment just to check it.
find_tool() {
  local tool="$1"
  local candidate
  for candidate in ".venv/Scripts/${tool}.exe" ".venv/Scripts/${tool}" ".venv/bin/${tool}"; do
    [ -x "$candidate" ] && { printf '%s' "$candidate"; return 0; }
  done
  command -v "$tool" 2>/dev/null && return 0
  return 1
}

run_tool() {
  local tool="$1"; shift
  local bin out rc
  bin="$(find_tool "$tool")" || { echo "$MISSING"; return 0; }
  out="$("$bin" "$@" 2>&1)"; rc=$?
  printf '%s' "$out"
  return $rc
}

FAILURES=""
add_failure() { FAILURES="${FAILURES}$1"$'\n\n'; }

# ---- 1. ruff -----------------------------------------------------------------
out="$(run_tool ruff check .)"; rc=$?
if [ "$out" != "$MISSING" ] && [ $rc -ne 0 ]; then
  add_failure "ruff check failed:
${out}"
fi

# ---- 2. pytest ---------------------------------------------------------------
# Exit 5 means "no tests collected", which is expected during phase 0.
out="$(run_tool pytest -q)"; rc=$?
if [ "$out" != "$MISSING" ] && [ $rc -ne 0 ] && [ $rc -ne 5 ]; then
  add_failure "pytest failed:
${out}"
fi

# ---- 3. retrieval eval freshness (phase 2 onward) ----------------------------
# Active once a golden set exists. CLAUDE.md: retrieval changes require a fresh
# `make eval` run and a committed report. This checks the report is not stale
# relative to the retrieval path; it does not re-run the eval (too expensive to
# do on every turn) — `make eval` is yours to run.
golden="$(find data/golden evals -name '*.jsonl' -type f 2>/dev/null | head -1)"
if [ -n "$golden" ]; then
  if [ ! -f "reports/eval.md" ]; then
    add_failure "eval gate: a golden set exists at ${golden} but reports/eval.md does not.
Run 'make eval' and commit the report before ending this turn."
  else
    stale="$(find src -type f -name '*.py' \
        \( -path '*retriev*' -o -path '*chunk*' -o -path '*embed*' -o -path '*index*' -o -path '*citation*' \) \
        -newer reports/eval.md 2>/dev/null | head -5)"
    if [ -n "$stale" ]; then
      add_failure "eval gate: retrieval-path files changed after reports/eval.md was last written:
${stale}
Run 'make eval' so the committed numbers match the code, then commit reports/eval.md."
    fi
  fi
fi

# ---- verdict -----------------------------------------------------------------
if [ -n "$FAILURES" ]; then
  echo $((count + 1)) > "$COUNT_FILE"
  {
    echo "The quality gate is red. You may not end this turn. Fix the following, then stop again."
    echo "(attempt $((count + 1)) of ${MAX_LOOPS})"
    echo
    printf '%s' "$FAILURES"
  } >&2
  exit 2
fi

pass
