"""The CI regression gate.

CI has no corpus -- `data/raw/` is gitignored and this project does not scrape a
public regulator on every push -- so it cannot recompute the numbers. It gates
the committed artifact instead, and the fingerprint check is what makes that
defensible: a deliberate regression fails two ways, by lowering the numbers or
by not re-running the eval at all.

One field of that fingerprint, `chunk_counts`, cannot really be verified in
CI: `make ingest` never runs there, so the working tree has no live corpus to
count. The caller (`clause.cli._gate_chunk_counts`) falls back to the
committed report's own `chunk_counts` in that case, which makes this
function's comparison for that one field compare `reports/eval.json` to
itself -- it cannot fail. That is a deliberate, bounded exception, not a
silent one: the caller is responsible for printing why whenever it takes that
fallback, so a passing gate is never read as having verified that field.
"""

from typing import Any

#: Fields whose change invalidates a report. `git_commit` is excluded: it moves
#: on every commit, including ones that touch nothing the eval depends on.
FINGERPRINT_FIELDS = (
    "manifest_sha256",
    "chunk_counts",
    "embedding_model",
    "retrieval_depth",
    "golden_path",
    "golden_sha256",
)


class GateFailure(Exception):
    pass


def check(
    report: dict[str, Any], baseline: dict[str, Any] | None, tree_fingerprint: dict[str, Any]
) -> list[str]:
    """Return failure messages; empty means the gate passes.

    `baseline is None` means there is nothing to regress against (a first
    run, or a baseline deliberately not yet committed): only the fingerprint
    is checked. Announcing *that* fact to a human is the caller's job, not
    this function's -- `check()` reports failures, not notes about what it
    did not check.
    """
    problems: list[str] = []
    reported = report.get("fingerprint", {})

    for field in FINGERPRINT_FIELDS:
        if reported.get(field) != tree_fingerprint.get(field):
            problems.append(
                f"stale report: fingerprint field {field!r} is "
                f"{reported.get(field)!r} in reports/eval.json but "
                f"{tree_fingerprint.get(field)!r} in the working tree. "
                "Re-run `make eval` and commit the result."
            )

    if baseline is None:
        return problems

    for strategy, expected in baseline.items():
        actual = report.get("strategies", {}).get(strategy)
        if actual is None:
            problems.append(f"report is missing strategy {strategy!r}, which the baseline covers")
            continue
        got = actual["overall"]["recall_at_5"]
        want = expected["recall_at_5"]
        if got < want:
            problems.append(
                f"{strategy}: recall@5 regressed to {got:.3f} from a baseline of {want:.3f}"
            )
    return problems
