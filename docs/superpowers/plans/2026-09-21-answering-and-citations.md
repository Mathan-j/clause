# Phase 3 — Answering With Enforced Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An answering layer over the Phase 2 retriever where every factual sentence carries citations that resolve to exact character spans, and where the system refuses rather than answers when retrieval is weak.

**Architecture:** Five units behind a `Answerer` protocol. Four of them — schema, resolver, validator, refusal — need no model and run in CI. Only `answering/llm.py` touches `llama_cpp`. Citations are 1-based indices into the hits presented to the model, constrained by a GBNF grammar so a malformed citation cannot be generated; the resolver maps an index to a span and slices it out of Postgres, so citation text is derived from the corpus rather than trusted from the model.

**Tech Stack:** Python 3.12, `llama-cpp-python==0.3.19` (prebuilt cp312 wheel, optional extra), Qwen2.5-3B-Instruct Q4_K_M GGUF, SQLAlchemy 2.0, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-21-answering-and-citations-design.md`

## Global Constraints

- Python 3.12. `uv` for all commands. `make` is **not installed** on this machine — use `uv run ...` directly; still add Makefile targets, CI has make.
- ruff with `PLC0415` enabled: **no function-level imports**. mypy strict must pass.
- **No test may make a network request to `rbi.org.in`.** The corpus is frozen.
- **`reports/eval.md`, `reports/eval.json`, `reports/baseline.json` must not be modified by this phase.** They are Phase 2's committed artifacts and the CI gate protects them.
- **Do not reindex, re-ingest, or modify the Qdrant collections** `clause_structural` (336 points) / `clause_fixed_window` (318 points).
- **Do not touch `triage-lab-qdrant` on port 6333** — it belongs to another project.
- Postgres on host port **5434**, Qdrant on **6335**.
- No number appears in documentation unless it came from a committed file under `reports/`. Commit messages and task reports may carry numbers; `README.md` and `docs/` prose may not.
- An answer containing a citation that does not resolve is a **hard failure, never a warning**.
- Write committed artifacts with `newline="\n"`.
- `llama-cpp-python` goes in an **optional dependency group**, never the base install. CI must not install it.
- Every behaviour this plan calls load-bearing ships with a test the implementer has watched fail with that behaviour removed.
- Do not `git push`.

---

### Task 1: Answering schema and errors

**Files:**
- Create: `src/clause/answering/__init__.py`, `src/clause/answering/schema.py`, `tests/test_answering_schema.py`

**Interfaces:**
- Consumes: `clause.evaluation.metrics.Span`
- Produces:
  - `MAX_CITATIONS_PER_SENTENCE = 3`
  - `Citation(doc_id, char_start, char_end, source_url, published_date, doc_type, text)` — frozen
  - `Sentence(text, citation_indices: tuple[int, ...], factual: bool)` — frozen
  - `AnswerDraft(question, sentences: tuple[Sentence, ...])` — frozen
  - `Answer(question, sentences, citations: tuple[Citation, ...])` — frozen
  - `Refusal(question, reason: str, top_score: float | None)` — frozen
  - `RefusalReason` constants: `LOW_SCORE = "low_score"`, `NO_CANDIDATES = "no_candidates"`
  - `AnsweringError`, `UnresolvableCitationError`, `UncitedClaimError`, `ModelNotAvailableError`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_answering_schema.py
from datetime import date

import pytest

from clause.answering.schema import (
    Answer,
    AnswerDraft,
    Citation,
    Refusal,
    RefusalReason,
    Sentence,
)


def _citation() -> Citation:
    return Citation(
        doc_id="rbi-12866",
        char_start=100,
        char_end=200,
        source_url="https://example.invalid/12866",
        published_date=date(2025, 6, 12),
        doc_type="master_direction",
        text="x" * 100,
    )


def test_citation_span_must_be_forward_and_non_empty() -> None:
    with pytest.raises(ValueError, match="forward"):
        Citation(
            doc_id="d",
            char_start=5,
            char_end=5,
            source_url="u",
            published_date=date(2025, 1, 1),
            doc_type="notification",
            text="",
        )


def test_citation_text_length_must_match_its_span() -> None:
    """A citation whose text is not its span is not a citation, it is a claim."""
    with pytest.raises(ValueError, match="length"):
        Citation(
            doc_id="d",
            char_start=0,
            char_end=10,
            source_url="u",
            published_date=date(2025, 1, 1),
            doc_type="notification",
            text="too short",
        )


def test_a_factual_sentence_with_no_citation_is_constructible_here() -> None:
    """Deliberately NOT rejected at construction.

    `validate.enforce` is the single place the citation contract is enforced. If
    `Sentence` also rejected this, `enforce`'s branch could never fire in practice
    and its test would have to build the object through `object.__setattr__` to
    reach it -- a check that cannot fail for the reason it exists.
    """
    s = Sentence(text="Banks must verify identity.", citation_indices=(), factual=True)
    assert s.citation_indices == ()


def test_a_non_factual_sentence_may_carry_no_citation() -> None:
    s = Sentence(text="Here is what the circulars say.", citation_indices=(), factual=False)
    assert s.citation_indices == ()


def test_citation_indices_must_be_one_based() -> None:
    with pytest.raises(ValueError, match="1-based"):
        Sentence(text="x", citation_indices=(0,), factual=True)


def test_draft_and_answer_carry_the_question() -> None:
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="a", citation_indices=(1,), factual=True),),
    )
    assert draft.question == "q"
    answer = Answer(question="q", sentences=draft.sentences, citations=(_citation(),))
    assert answer.citations[0].doc_id == "rbi-12866"


def test_refusal_records_why_and_the_score_that_caused_it() -> None:
    r = Refusal(question="q", reason=RefusalReason.LOW_SCORE, top_score=0.21)
    assert r.reason == "low_score"
    assert r.top_score == 0.21


def test_a_no_candidates_refusal_has_no_score() -> None:
    r = Refusal(question="q", reason=RefusalReason.NO_CANDIDATES, top_score=None)
    assert r.top_score is None
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_answering_schema.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.answering'`

- [ ] **Step 3: Implement**

```python
# src/clause/answering/schema.py
"""Structured answer types.

A `Sentence` references citations by **1-based index into the hits presented to
the model**, not by chunk id. `Hit` carries no chunk id, and an index is what a
GBNF grammar can constrain to a valid range at generation time -- so a citation
pointing at nothing cannot be generated. Resolution to an actual span happens in
`resolver.py`, which is where an unresolvable citation is still possible and is
still a hard failure: the document can be absent, or the span out of range,
between retrieval and resolution.
"""

from dataclasses import dataclass
from datetime import date

MAX_CITATIONS_PER_SENTENCE = 3


class AnsweringError(Exception):
    """Base for every failure in the answering path."""


class UnresolvableCitationError(AnsweringError):
    """A citation did not resolve to a span in the corpus.

    CLAUDE.md: an answer containing a citation that does not resolve is a hard
    failure, never a warning. Nothing catches this and returns a degraded answer.
    """


class UncitedClaimError(AnsweringError):
    """A sentence marked factual carried no citation."""


class ModelNotAvailableError(AnsweringError):
    """The GGUF model file is not present, and this process will not download it."""


@dataclass(frozen=True, slots=True)
class Citation:
    doc_id: str
    char_start: int
    char_end: int
    source_url: str
    published_date: date
    doc_type: str
    text: str

    def __post_init__(self) -> None:
        if self.char_start < 0:
            raise ValueError(f"char_start must not be negative: {self.char_start}")
        if self.char_start >= self.char_end:
            raise ValueError(
                f"span must be non-empty and forward: got "
                f"[{self.char_start}:{self.char_end}] in {self.doc_id!r}"
            )
        expected = self.char_end - self.char_start
        if len(self.text) != expected:
            raise ValueError(
                f"citation text length {len(self.text)} does not match its span "
                f"[{self.char_start}:{self.char_end}] (expected {expected}) in "
                f"{self.doc_id!r}: the text must be the slice, not a paraphrase of it"
            )


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    citation_indices: tuple[int, ...]
    factual: bool

    def __post_init__(self) -> None:
        # A factual sentence with no citation is deliberately constructible here.
        # `validate.enforce` is the single enforcement point for the citation
        # contract; rejecting it in both places would leave enforce's branch
        # unreachable and its test would have to fake an object to reach it.
        for i in self.citation_indices:
            if i < 1:
                raise ValueError(f"citation indices are 1-based, got {i}")


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    """What the model produced, before any citation was resolved."""

    question: str
    sentences: tuple[Sentence, ...]


@dataclass(frozen=True, slots=True)
class Answer:
    """A draft whose citations have all resolved against the corpus."""

    question: str
    sentences: tuple[Sentence, ...]
    citations: tuple[Citation, ...]


class RefusalReason:
    LOW_SCORE = "low_score"
    NO_CANDIDATES = "no_candidates"


@dataclass(frozen=True, slots=True)
class Refusal:
    question: str
    reason: str
    top_score: float | None
```

Create `src/clause/answering/__init__.py` empty.

- [ ] **Step 4: Run tests and the gate**

```bash
uv run pytest tests/test_answering_schema.py -v && uv run ruff check . && uv run mypy
```
Expected: 8 passed.

- [ ] **Step 5: Removal proof**

Delete the `len(self.text) != expected` check in `Citation.__post_init__`, run `tests/test_answering_schema.py`, confirm `test_citation_text_length_must_match_its_span` fails. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/clause/answering tests/test_answering_schema.py
git commit -m "feat(answering): structured answer types with citation-by-index

Citations are 1-based indices into the hits shown to the model, not chunk
ids: Hit carries no chunk id, and an index is what a GBNF grammar can
constrain at generation time, so a citation pointing at nothing cannot be
produced. A Citation's text must equal its own span length -- a citation
whose text is not the slice is not a citation, it is a claim."
```

---

### Task 2: Citation resolver

**Files:**
- Create: `src/clause/answering/resolver.py`, `tests/test_answering_resolver.py`

**Interfaces:**
- Consumes: `Citation`, `UnresolvableCitationError` (Task 1); `clause.retrieve.Hit`; `clause.db.schema.DocumentRow`
- Produces: `resolve_citations(draft: AnswerDraft, hits: Sequence[Hit], session: Session) -> tuple[Citation, ...]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_answering_resolver.py
from datetime import date

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from clause.answering.resolver import resolve_citations
from clause.answering.schema import AnswerDraft, Sentence, UnresolvableCitationError
from clause.db.schema import DocumentRow
from clause.retrieve import Hit

pytestmark = pytest.mark.db


def _hit(doc_id: str, start: int, end: int, text: str) -> Hit:
    return Hit(
        doc_id=doc_id,
        strategy="structural",
        ordinal=0,
        char_start=start,
        char_end=end,
        score=0.5,
        text=text,
        source_url="https://example.invalid/1",
        published_date=date(2025, 1, 1),
        regulated_entity=(),
    )


def _seed(session: Session, doc_id: str, text: str) -> None:
    session.add(
        DocumentRow(
            doc_id=doc_id,
            rbi_id=1,
            url="https://example.invalid/1",
            circular_no="C/1",
            dept_ref="D/1",
            title="t",
            doc_type="notification",
            published_date=date(2025, 1, 1),
            effective_date=None,
            sha256="0" * 64,
            fetched_at=sa.func.now(),
            text=text,
            regulated_entity=[],
        )
    )
    session.flush()


def test_resolution_slices_the_stored_document_rather_than_trusting_the_hit(
    db_session: Session,
) -> None:
    """The citation's text comes from the corpus, never from the model or the hit."""
    _seed(db_session, "doc-a", "0123456789ABCDEFGHIJ")
    hit = _hit("doc-a", 2, 6, "WRONG")  # hit text deliberately disagrees
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="s", citation_indices=(1,), factual=True),)
    )
    cites = resolve_citations(draft, [hit], db_session)
    assert cites[0].text == "2345"


def test_an_index_past_the_presented_hits_is_a_hard_failure(db_session: Session) -> None:
    _seed(db_session, "doc-a", "0123456789")
    hit = _hit("doc-a", 0, 4, "0123")
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="s", citation_indices=(2,), factual=True),)
    )
    with pytest.raises(UnresolvableCitationError, match="index 2"):
        resolve_citations(draft, [hit], db_session)


def test_a_missing_document_is_a_hard_failure(db_session: Session) -> None:
    hit = _hit("doc-absent", 0, 4, "0123")
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="s", citation_indices=(1,), factual=True),)
    )
    with pytest.raises(UnresolvableCitationError, match="doc-absent"):
        resolve_citations(draft, [hit], db_session)


def test_a_span_past_the_end_of_the_document_is_a_hard_failure(db_session: Session) -> None:
    """Re-ingest can shorten a document between retrieval and resolution."""
    _seed(db_session, "doc-a", "short")
    hit = _hit("doc-a", 0, 500, "x" * 500)
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="s", citation_indices=(1,), factual=True),)
    )
    with pytest.raises(UnresolvableCitationError, match="out of range"):
        resolve_citations(draft, [hit], db_session)


def test_each_cited_index_appears_once_in_order(db_session: Session) -> None:
    _seed(db_session, "doc-a", "0123456789ABCDEFGHIJ")
    hits = [_hit("doc-a", 0, 4, "0123"), _hit("doc-a", 4, 8, "4567")]
    draft = AnswerDraft(
        question="q",
        sentences=(
            Sentence(text="s1", citation_indices=(2,), factual=True),
            Sentence(text="s2", citation_indices=(1, 2), factual=True),
        ),
    )
    cites = resolve_citations(draft, hits, db_session)
    assert [(c.char_start, c.char_end) for c in cites] == [(0, 4), (4, 8)]


def test_a_draft_with_no_citations_resolves_to_nothing(db_session: Session) -> None:
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="s", citation_indices=(), factual=False),)
    )
    assert resolve_citations(draft, [], db_session) == ()
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_answering_resolver.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.answering.resolver'`

- [ ] **Step 3: Implement**

```python
# src/clause/answering/resolver.py
"""Turn cited indices into citations, by slicing the corpus.

The model never supplies citation text. It supplies an index into the hits it
was shown; this module reads that hit's `(doc_id, char_start, char_end)`, loads
the stored document, and slices it. A citation's text is therefore derived from
the corpus by construction and cannot disagree with it -- the same reasoning as
`make_chunk`, which slices rather than accepting text.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.orm import Session

from clause.answering.schema import AnswerDraft, Citation, UnresolvableCitationError
from clause.db.schema import DocumentRow
from clause.retrieve import Hit


def _cited_indices(draft: AnswerDraft) -> list[int]:
    """Every cited index, de-duplicated, in first-cited order."""
    seen: dict[int, None] = {}
    for sentence in draft.sentences:
        for index in sentence.citation_indices:
            seen.setdefault(index, None)
    return sorted(seen)


def resolve_citations(
    draft: AnswerDraft, hits: Sequence[Hit], session: Session
) -> tuple[Citation, ...]:
    """Resolve every cited index to a Citation, or raise.

    Raises `UnresolvableCitationError` when an index names no presented hit, when
    the hit's document is absent from the corpus, or when its span no longer fits
    inside that document. Each is a genuine post-retrieval failure: the corpus can
    change between the search and the answer.
    """
    indices = _cited_indices(draft)
    if not indices:
        return ()

    for index in indices:
        if index > len(hits):
            raise UnresolvableCitationError(
                f"citation index {index} names no presented hit "
                f"(only {len(hits)} were shown for question {draft.question!r})"
            )

    wanted = {hits[i - 1].doc_id for i in indices}
    docs = {
        d.doc_id: d
        for d in session.scalars(
            sa.select(DocumentRow).where(DocumentRow.doc_id.in_(wanted))
        ).all()
    }

    citations: list[Citation] = []
    for index in indices:
        hit = hits[index - 1]
        document = docs.get(hit.doc_id)
        if document is None:
            raise UnresolvableCitationError(
                f"citation index {index} points at document {hit.doc_id!r}, "
                "which is not in the corpus"
            )
        if hit.char_end > len(document.text):
            raise UnresolvableCitationError(
                f"citation index {index} span [{hit.char_start}:{hit.char_end}] is "
                f"out of range for {hit.doc_id!r}, which holds {len(document.text)} "
                "characters -- the document changed since it was retrieved"
            )
        citations.append(
            Citation(
                doc_id=hit.doc_id,
                char_start=hit.char_start,
                char_end=hit.char_end,
                source_url=document.url,
                published_date=document.published_date,
                doc_type=document.doc_type,
                text=document.text[hit.char_start : hit.char_end],
            )
        )
    return tuple(citations)
```

- [ ] **Step 4: Run tests and the gate**

```bash
uv run pytest tests/test_answering_resolver.py -v && uv run ruff check . && uv run mypy
```
Expected: 6 passed.

- [ ] **Step 5: Removal proof**

Replace the `hit.char_end > len(document.text)` check with `if False:`, run the file, confirm `test_a_span_past_the_end_of_the_document_is_a_hard_failure` fails. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/clause/answering/resolver.py tests/test_answering_resolver.py
git commit -m "feat(answering): resolve citations by slicing the corpus

The model supplies an index, never text. The resolver reads that hit's span
and slices the stored document, so a citation's text is derived from the
corpus rather than trusted from the generator. Three post-retrieval failures
raise rather than degrade: an index naming no hit, a document that has left
the corpus, and a span that no longer fits the document it points into."
```

---

### Task 3: The citation contract validator

**Files:**
- Create: `src/clause/answering/validate.py`, `tests/test_answering_validate.py`

**Interfaces:**
- Consumes: Task 1 types, Task 2's `resolve_citations`
- Produces: `enforce(draft: AnswerDraft, citations: Sequence[Citation]) -> Answer`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_answering_validate.py
from datetime import date

import pytest

from clause.answering.schema import (
    AnswerDraft,
    Citation,
    MAX_CITATIONS_PER_SENTENCE,
    Sentence,
    UncitedClaimError,
    UnresolvableCitationError,
)
from clause.answering.validate import enforce


def _c(start: int = 0, end: int = 4) -> Citation:
    return Citation(
        doc_id="d",
        char_start=start,
        char_end=end,
        source_url="u",
        published_date=date(2025, 1, 1),
        doc_type="notification",
        text="x" * (end - start),
    )


def test_a_clean_draft_becomes_an_answer() -> None:
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="a", citation_indices=(1,), factual=True),)
    )
    answer = enforce(draft, [_c()])
    assert answer.question == "q"
    assert len(answer.citations) == 1


def test_a_factual_sentence_without_a_citation_raises() -> None:
    """This is the single enforcement point, so the object constructs normally."""
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="Banks must verify.", citation_indices=(), factual=True),),
    )
    with pytest.raises(UncitedClaimError, match="Banks must verify"):
        enforce(draft, [])


def test_an_index_with_no_matching_citation_raises() -> None:
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="a", citation_indices=(2,), factual=True),)
    )
    with pytest.raises(UnresolvableCitationError, match="index 2"):
        enforce(draft, [_c()])


def test_more_citations_than_the_cap_raises() -> None:
    n = MAX_CITATIONS_PER_SENTENCE + 1
    draft = AnswerDraft(
        question="q",
        sentences=(
            Sentence(text="a", citation_indices=tuple(range(1, n + 1)), factual=True),
        ),
    )
    with pytest.raises(ValueError, match="at most"):
        enforce(draft, [_c(i, i + 4) for i in range(0, n * 4, 4)])


def test_an_answer_with_no_factual_sentences_needs_no_citations() -> None:
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="I could not find this.", citation_indices=(), factual=False),),
    )
    answer = enforce(draft, [])
    assert answer.citations == ()


def test_unused_citations_are_dropped_not_reported() -> None:
    """A citation nothing references is not evidence; carrying it would inflate the count."""
    draft = AnswerDraft(
        question="q", sentences=(Sentence(text="a", citation_indices=(1,), factual=True),)
    )
    answer = enforce(draft, [_c(0, 4), _c(4, 8)])
    assert len(answer.citations) == 1
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_answering_validate.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.answering.validate'`

- [ ] **Step 3: Implement**

```python
# src/clause/answering/validate.py
"""Enforce the citation contract over a resolved draft.

This is the gate CLAUDE.md describes: "an answer containing a citation that does
not resolve is a hard failure, never a warning". Every failure here raises. There
is deliberately no lenient mode, no `strict=False`, and no path that returns a
partial answer -- an answer that has been allowed through with a broken citation
is worse than no answer, because it looks the same as a sound one.
"""

from collections.abc import Sequence

from clause.answering.schema import (
    MAX_CITATIONS_PER_SENTENCE,
    Answer,
    AnswerDraft,
    Citation,
    UncitedClaimError,
    UnresolvableCitationError,
)


def enforce(draft: AnswerDraft, citations: Sequence[Citation]) -> Answer:
    """Return an `Answer`, or raise. Never returns a degraded result."""
    used: dict[int, None] = {}
    for sentence in draft.sentences:
        if sentence.factual and not sentence.citation_indices:
            raise UncitedClaimError(
                f"factual sentence carries no citation: {sentence.text!r}"
            )
        if len(sentence.citation_indices) > MAX_CITATIONS_PER_SENTENCE:
            raise ValueError(
                f"a sentence may carry at most {MAX_CITATIONS_PER_SENTENCE} citations, "
                f"got {len(sentence.citation_indices)}: {sentence.text!r}"
            )
        for index in sentence.citation_indices:
            if index > len(citations):
                raise UnresolvableCitationError(
                    f"citation index {index} has no resolved citation "
                    f"({len(citations)} resolved)"
                )
            used.setdefault(index, None)

    return Answer(
        question=draft.question,
        sentences=draft.sentences,
        citations=tuple(citations[i - 1] for i in sorted(used)),
    )
```

- [ ] **Step 4: Run tests and the gate**

```bash
uv run pytest tests/test_answering_validate.py -v && uv run ruff check . && uv run mypy
```
Expected: 6 passed.

- [ ] **Step 5: Removal proof**

Delete the `UncitedClaimError` branch, run the file, confirm `test_a_factual_sentence_without_a_citation_raises` fails. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/clause/answering/validate.py tests/test_answering_validate.py
git commit -m "feat(answering): enforce the citation contract, with no lenient mode

Every failure raises. There is deliberately no strict=False and no partial
answer: an answer allowed through with a broken citation looks exactly like
a sound one, which is the failure mode this project exists to prevent."
```

---

### Task 4: Refusal threshold and its sweep

**Files:**
- Create: `src/clause/answering/refusal.py`, `tests/test_answering_refusal.py`

**Interfaces:**
- Consumes: `Refusal`, `RefusalReason` (Task 1); `clause.retrieve.Hit`
- Produces:
  - `DEFAULT_THRESHOLD = 0.35`
  - `SWEEP_THRESHOLDS: tuple[float, ...]` — `0.00` to `0.95` in steps of `0.05`
  - `decide(question: str, hits: Sequence[Hit], threshold: float) -> Refusal | None` — `None` means proceed
  - `sweep(scores: Sequence[float | None], thresholds: Sequence[float] = SWEEP_THRESHOLDS) -> dict[float, int]` — threshold -> number refused

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_answering_refusal.py
from datetime import date

from clause.answering.refusal import DEFAULT_THRESHOLD, SWEEP_THRESHOLDS, decide, sweep
from clause.answering.schema import RefusalReason
from clause.retrieve import Hit


def _hit(score: float) -> Hit:
    return Hit(
        doc_id="d",
        strategy="structural",
        ordinal=0,
        char_start=0,
        char_end=4,
        score=score,
        text="abcd",
        source_url="u",
        published_date=date(2025, 1, 1),
        regulated_entity=(),
    )


def test_no_hits_refuses_with_its_own_reason() -> None:
    """Distinct from a low score: nothing was retrieved at all."""
    r = decide("q", [], DEFAULT_THRESHOLD)
    assert r is not None
    assert r.reason == RefusalReason.NO_CANDIDATES
    assert r.top_score is None


def test_a_score_below_the_threshold_refuses() -> None:
    r = decide("q", [_hit(0.10)], 0.35)
    assert r is not None
    assert r.reason == RefusalReason.LOW_SCORE
    assert r.top_score == 0.10


def test_a_score_exactly_at_the_threshold_proceeds() -> None:
    """The boundary is inclusive; a question is not refused for tying the bar."""
    assert decide("q", [_hit(0.35)], 0.35) is None


def test_a_score_above_the_threshold_proceeds() -> None:
    assert decide("q", [_hit(0.90)], 0.35) is None


def test_only_the_top_hit_decides() -> None:
    assert decide("q", [_hit(0.90), _hit(0.01)], 0.35) is None
    r = decide("q", [_hit(0.01), _hit(0.90)], 0.35)
    assert r is not None and r.top_score == 0.01


def test_sweep_is_monotone_in_the_threshold() -> None:
    """Raising the bar can never refuse fewer questions. A violation is a bug."""
    counts = sweep([0.1, 0.4, 0.8, None])
    values = [counts[t] for t in SWEEP_THRESHOLDS]
    assert values == sorted(values)


def test_sweep_counts_a_missing_score_as_refused_at_every_threshold() -> None:
    counts = sweep([None])
    assert all(counts[t] == 1 for t in SWEEP_THRESHOLDS)


def test_sweep_covers_the_documented_thresholds() -> None:
    assert SWEEP_THRESHOLDS[0] == 0.0
    assert len(set(SWEEP_THRESHOLDS)) == len(SWEEP_THRESHOLDS)
    assert sorted(SWEEP_THRESHOLDS) == list(SWEEP_THRESHOLDS)
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_answering_refusal.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.answering.refusal'`

- [ ] **Step 3: Implement**

```python
# src/clause/answering/refusal.py
"""Decide whether retrieval is strong enough to answer at all.

Phase 2 measured recall@5 of 0.446 and 0.338 (reports/eval.md): more than half
of golden questions do not retrieve a correct span in the top five. So the
interesting artifact is not a refusal rate, it is the curve -- how the answer
rate and the false-refusal rate move together as the bar rises. `DEFAULT_THRESHOLD`
is a starting point for the sweep to argue with, not a tuned value.
"""

from collections.abc import Sequence

from clause.answering.schema import Refusal, RefusalReason
from clause.retrieve import Hit

DEFAULT_THRESHOLD = 0.35

SWEEP_THRESHOLDS: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(20))


def decide(question: str, hits: Sequence[Hit], threshold: float) -> Refusal | None:
    """Return a `Refusal`, or `None` to proceed.

    The boundary is inclusive: a score exactly equal to the threshold proceeds.
    A question is not refused for tying the bar.
    """
    if not hits:
        return Refusal(
            question=question, reason=RefusalReason.NO_CANDIDATES, top_score=None
        )
    top = hits[0].score
    if top < threshold:
        return Refusal(question=question, reason=RefusalReason.LOW_SCORE, top_score=top)
    return None


def sweep(
    scores: Sequence[float | None], thresholds: Sequence[float] = SWEEP_THRESHOLDS
) -> dict[float, int]:
    """How many questions each threshold would refuse.

    A `None` score means nothing was retrieved, which is a refusal at every
    threshold including zero.
    """
    return {
        t: sum(1 for s in scores if s is None or s < t) for t in thresholds
    }
```

- [ ] **Step 4: Run tests and the gate**

```bash
uv run pytest tests/test_answering_refusal.py -v && uv run ruff check . && uv run mypy
```
Expected: 8 passed.

- [ ] **Step 5: Removal proof**

Change `top < threshold` to `top <= threshold`, run the file, confirm `test_a_score_exactly_at_the_threshold_proceeds` fails. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/clause/answering/refusal.py tests/test_answering_refusal.py
git commit -m "feat(answering): refusal threshold and its sweep

The deliverable is the curve, not a refusal rate somebody picked. Phase 2
measured recall@5 of 0.446/0.338, so the choice is between citing the wrong
span often and refusing often -- the sweep is what makes that choice visible
instead of hidden inside a constant."
```

---

### Task 5: The Answerer protocol and a deterministic stub

**Files:**
- Create: `src/clause/answering/base.py`, `tests/test_answering_stub.py`

**Interfaces:**
- Consumes: Task 1 types, `clause.retrieve.Hit`
- Produces:
  - `Answerer` — `Protocol` with `answer(question: str, hits: Sequence[Hit]) -> AnswerDraft`
  - `StubAnswerer` — deterministic, cites hit 1, no model

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_answering_stub.py
from datetime import date

from clause.answering.base import Answerer, StubAnswerer
from clause.retrieve import Hit


def _hit(text: str = "Regulated entities shall verify identity.") -> Hit:
    return Hit(
        doc_id="d",
        strategy="structural",
        ordinal=0,
        char_start=0,
        char_end=len(text),
        score=0.8,
        text=text,
        source_url="u",
        published_date=date(2025, 1, 1),
        regulated_entity=(),
    )


def test_the_stub_satisfies_the_protocol() -> None:
    assert isinstance(StubAnswerer(), Answerer)


def test_the_stub_is_deterministic() -> None:
    """A harness whose stub moves is not a harness."""
    a, b = StubAnswerer(), StubAnswerer()
    assert a.answer("q", [_hit()]) == b.answer("q", [_hit()])


def test_the_stub_cites_the_top_hit() -> None:
    draft = StubAnswerer().answer("q", [_hit()])
    assert draft.sentences[0].citation_indices == (1,)
    assert draft.sentences[0].factual is True


def test_the_stub_produces_no_factual_sentence_without_hits() -> None:
    draft = StubAnswerer().answer("q", [])
    assert all(not s.factual for s in draft.sentences)
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_answering_stub.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.answering.base'`

- [ ] **Step 3: Implement**

```python
# src/clause/answering/base.py
"""The generator interface, and a deterministic implementation of it.

`StubAnswerer` exists so the whole answering pipeline -- resolution, the citation
contract, refusal, the metrics -- is exercisable with no model present. That
matters because CI never installs llama-cpp-python: without a stub, the parts of
this phase that enforce CLAUDE.md's hardest rule would be untested in the only
environment that runs on every push.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from clause.answering.schema import AnswerDraft, Sentence
from clause.retrieve import Hit


@runtime_checkable
class Answerer(Protocol):
    def answer(self, question: str, hits: Sequence[Hit]) -> AnswerDraft: ...


class StubAnswerer:
    """Quotes the top hit verbatim and cites it. Deterministic by construction."""

    def answer(self, question: str, hits: Sequence[Hit]) -> AnswerDraft:
        if not hits:
            return AnswerDraft(
                question=question,
                sentences=(
                    Sentence(
                        text="No candidate passages were retrieved.",
                        citation_indices=(),
                        factual=False,
                    ),
                ),
            )
        return AnswerDraft(
            question=question,
            sentences=(
                Sentence(
                    text=hits[0].text.strip(),
                    citation_indices=(1,),
                    factual=True,
                ),
            ),
        )
```

- [ ] **Step 4: Run tests and the gate**

```bash
uv run pytest tests/test_answering_stub.py -v && uv run ruff check . && uv run mypy
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/clause/answering/base.py tests/test_answering_stub.py
git commit -m "feat(answering): Answerer protocol and a deterministic stub

CI never installs llama-cpp-python. Without a stub implementation, the parts
of this phase that enforce the citation contract would be untested in the
only environment that runs on every push."
```

---

### Task 6: The llama.cpp answerer with a GBNF grammar

**Files:**
- Create: `src/clause/answering/llm.py`, `src/clause/answering/grammar.py`, `tests/test_answering_llm.py`
- Modify: `pyproject.toml`, `docs/decisions.md`, `.env.example`, `src/clause/config.py`

**Interfaces:**
- Consumes: Task 1 types, Task 5's `Answerer`
- Produces:
  - `DEFAULT_MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"`
  - `build_grammar(n_hits: int) -> str` — GBNF constraining citations to `1..n_hits`
  - `LlamaAnswerer(model_path: Path, n_ctx: int = 4096)` implementing `Answerer`
  - `Settings.answer_model_path: Path`, `Settings.answer_threshold: float`

- [ ] **Step 1: Add the optional dependency**

`llama-cpp-python` publishes **no wheels on PyPI** — only an sdist, which would need MSVC and cmake to build. Version `0.3.19` has a prebuilt `cp312-win_amd64` wheel on the maintainer's index. This was verified in an isolated venv before this plan was written: it installs, imports, and exposes `LlamaGrammar`.

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
answer = ["llama-cpp-python==0.3.19"]

[tool.uv]
extra-index-url = ["https://abetlen.github.io/llama-cpp-python/whl/cpu"]
```

Install it into this project:

```bash
uv sync --extra answer
```

Append to `docs/decisions.md`:

```markdown
- llama-cpp-python (optional extra `answer`) — local GGUF inference for the answering layer. Chosen over a hosted API because the project must run with no per-request cost, and over transformers+torch because it is markedly faster on a CPU-only machine and supports GBNF grammars, which make a malformed citation impossible to generate rather than something to validate afterwards. Pinned to 0.3.19: PyPI ships only an sdist and this machine has no MSVC toolchain, so the prebuilt cp312 wheel on the maintainer's index is the only installable build.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_answering_llm.py
import json
from pathlib import Path

import pytest

from clause.answering.base import Answerer
from clause.answering.grammar import build_grammar
from clause.answering.llm import DEFAULT_MODEL_FILENAME, LlamaAnswerer
from clause.answering.schema import ModelNotAvailableError


def test_the_grammar_names_every_valid_index_and_no_other() -> None:
    g = build_grammar(3)
    assert '"1"' in g and '"2"' in g and '"3"' in g
    assert '"4"' not in g


def test_the_grammar_rejects_a_zero_hit_request() -> None:
    """With nothing retrieved there is no citable index, so generation must not start."""
    with pytest.raises(ValueError, match="at least one"):
        build_grammar(0)


def test_an_absent_model_fails_loudly_rather_than_downloading(tmp_path: Path) -> None:
    missing = tmp_path / "not-a-model.gguf"
    with pytest.raises(ModelNotAvailableError, match="not present"):
        LlamaAnswerer(model_path=missing)


def test_the_error_names_the_documented_filename(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError, match=DEFAULT_MODEL_FILENAME):
        LlamaAnswerer(model_path=tmp_path / "not-a-model.gguf")


@pytest.mark.llm
def test_generation_produces_a_draft_whose_indices_are_in_range(answer_model: Path) -> None:
    from datetime import date

    from clause.retrieve import Hit

    hits = [
        Hit(
            doc_id="d",
            strategy="structural",
            ordinal=0,
            char_start=0,
            char_end=52,
            score=0.9,
            text="Regulated entities shall verify customer identity.",
            source_url="u",
            published_date=date(2025, 1, 1),
            regulated_entity=(),
        )
    ]
    answerer = LlamaAnswerer(model_path=answer_model)
    assert isinstance(answerer, Answerer)
    draft = answerer.answer("What must regulated entities verify?", hits)
    assert draft.sentences
    for sentence in draft.sentences:
        for index in sentence.citation_indices:
            assert 1 <= index <= len(hits)
```

Register the marker in `pyproject.toml`'s `[tool.pytest.ini_options]` markers list:

```toml
    "llm: requires the GGUF answering model in the local cache",
```

Add to `tests/conftest.py`:

```python
@pytest.fixture
def answer_model() -> Path:
    from clause.config import get_settings

    path = get_settings().answer_model_path
    if not path.exists():
        pytest.skip(f"answering model not present at {path}")
    return path
```

- [ ] **Step 3: Run and watch them fail**

```bash
uv run pytest tests/test_answering_llm.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.answering.grammar'`

- [ ] **Step 4: Implement the grammar**

```python
# src/clause/answering/grammar.py
"""A GBNF grammar that makes a malformed citation impossible to generate.

Constrained decoding is the same move as `make_chunk` slicing its own text: it
converts a rule that would otherwise be checked after the fact into one the
generator cannot break. The grammar pins the JSON shape AND the citation index
range, so `citation_indices` can only ever name a hit that was actually shown.
"""

_TEMPLATE = """\
root       ::= "{{" ws "\\"sentences\\"" ws ":" ws sentences ws "}}"
sentences  ::= "[" ws sentence (ws "," ws sentence)* ws "]"
sentence   ::= "{{" ws "\\"text\\"" ws ":" ws string ws "," ws \
"\\"citations\\"" ws ":" ws citations ws "}}"
citations  ::= "[" ws (index (ws "," ws index)*)? ws "]"
index      ::= {indices}
string     ::= "\\"" char* "\\""
char       ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]
ws         ::= [ \\t\\n]*
"""


def build_grammar(n_hits: int) -> str:
    """GBNF allowing citation indices `1..n_hits` and nothing else."""
    if n_hits < 1:
        raise ValueError(
            "a grammar needs at least one citable hit; with nothing retrieved the "
            "caller must refuse instead of generating"
        )
    indices = " | ".join(f'"{i}"' for i in range(1, n_hits + 1))
    return _TEMPLATE.format(indices=indices)
```

- [ ] **Step 5: Implement the answerer**

```python
# src/clause/answering/llm.py
"""Local GGUF generation through llama.cpp, constrained by a GBNF grammar.

The model never sees or supplies citation text -- only an index into the hits it
was shown. It also never downloads: an absent model file raises with the path it
expected, the same rule the embedding encoder follows, because a first run that
fetches weights has a different reproducibility story from every run after it.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llama_cpp import Llama, LlamaGrammar

from clause.answering.grammar import build_grammar
from clause.answering.schema import (
    MAX_CITATIONS_PER_SENTENCE,
    AnswerDraft,
    ModelNotAvailableError,
    Sentence,
)
from clause.retrieve import Hit

DEFAULT_MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"

_SYSTEM = (
    "You answer questions about Indian banking regulation using ONLY the numbered "
    "passages provided. Every sentence stating a fact must cite the passage numbers "
    "it came from. If the passages do not answer the question, say so in one "
    "sentence and cite nothing. Never cite a passage number you were not given."
)


def _prompt(question: str, hits: Sequence[Hit]) -> str:
    passages = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, start=1))
    return (
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\nPassages:\n{passages}\n\nQuestion: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class LlamaAnswerer:
    def __init__(self, model_path: Path, n_ctx: int = 4096) -> None:
        if not model_path.exists():
            raise ModelNotAvailableError(
                f"answering model is not present at {model_path}, and this process "
                f"will not download it. Fetch {DEFAULT_MODEL_FILENAME} into that path "
                "first; see README's 'Running the answering evaluation'."
            )
        self._model_path = model_path
        self._llama = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            verbose=False,
            seed=0,
        )

    def answer(self, question: str, hits: Sequence[Hit]) -> AnswerDraft:
        grammar = LlamaGrammar.from_string(build_grammar(len(hits)), verbose=False)
        result: Any = self._llama(
            _prompt(question, hits),
            grammar=grammar,
            max_tokens=512,
            temperature=0.0,
        )
        payload = json.loads(result["choices"][0]["text"])
        sentences: list[Sentence] = []
        for raw in payload["sentences"]:
            indices = tuple(int(i) for i in raw["citations"])[:MAX_CITATIONS_PER_SENTENCE]
            sentences.append(
                Sentence(
                    text=raw["text"],
                    citation_indices=indices,
                    factual=bool(indices),
                )
            )
        return AnswerDraft(question=question, sentences=tuple(sentences))
```

Add to `src/clause/config.py`'s `Settings`:

```python
    # Answering
    answer_model_path: Path = Path("data/models/qwen2.5-3b-instruct-q4_k_m.gguf")
    answer_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
```

Add to `.env.example`:

```bash
CLAUSE_ANSWER_MODEL_PATH=data/models/qwen2.5-3b-instruct-q4_k_m.gguf
CLAUSE_ANSWER_THRESHOLD=0.35
```

Add `data/models/` to `.gitignore`.

- [ ] **Step 6: Fetch the model and run**

```bash
mkdir -p data/models
uv run python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF',
    filename='qwen2.5-3b-instruct-q4_k_m.gguf',
    local_dir='data/models',
)
print(p)
"
uv run pytest tests/test_answering_llm.py -v && uv run ruff check . && uv run mypy
```
Expected: 5 passed. Report the wall-clock time of the `@pytest.mark.llm` test — later tasks depend on knowing per-question cost.

If the repo id or filename 404s, list what the repo actually offers with `huggingface_hub.list_repo_files` and pick the `q4_k_m` variant; report the name you used rather than substituting a different quantisation silently.

- [ ] **Step 7: Removal proof**

Delete the `if not model_path.exists()` guard, run the file, confirm `test_an_absent_model_fails_loudly_rather_than_downloading` fails. Restore.

- [ ] **Step 8: Commit**

```bash
git add src/clause/answering/llm.py src/clause/answering/grammar.py \
        tests/test_answering_llm.py tests/conftest.py pyproject.toml uv.lock \
        docs/decisions.md .env.example src/clause/config.py .gitignore
git commit -m "feat(answering): local GGUF generation constrained by a GBNF grammar

The grammar pins both the JSON shape and the citation index range, so the
model cannot emit a citation naming a passage it was not shown. That converts
the citation contract from something validated after the fact into something
the generator is incapable of breaking.

Pinned to llama-cpp-python 0.3.19: PyPI ships only an sdist and this machine
has no MSVC toolchain, so the prebuilt cp312 wheel is the only installable
build. It goes in an optional extra because CI never generates answers."
```

---

### Task 7: The adversarial out-of-corpus set

**Files:**
- Create: `data/adversarial/out-of-corpus-v1.jsonl`, `src/clause/evaluation/adversarial.py`, `tests/test_adversarial.py`

**Interfaces:**
- Produces:
  - `KINDS: tuple[str, ...]` — `("adjacent_domain", "uncovered_kyc", "time_shifted", "wrong_jurisdiction")`
  - `AdversarialQuestion(qid, question, kind, why_unanswerable)` — frozen
  - `load_adversarial(path: Path) -> list[AdversarialQuestion]`

This is content work. **30 questions**, distributed: `adjacent_domain` 10, `uncovered_kyc` 10, `time_shifted` 6, `wrong_jurisdiction` 4 — weighted toward the kinds that retrieve well, because a set that refuses trivially measures nothing.

Rules for drafting, enforced by the tests below:
- Every question must be answerable-sounding and specific. "What is the capital of France" is worthless here.
- `why_unanswerable` must name what is missing, not merely assert absence.
- No question may be answerable from the corpus. Verify each by running retrieval and reading the top hit.
- `time_shifted` questions must name a date before **2023-07-04**, the corpus's earliest document.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_adversarial.py
from pathlib import Path

import pytest

from clause.evaluation.adversarial import KINDS, load_adversarial

ADVERSARIAL = Path("data/adversarial/out-of-corpus-v1.jsonl")


def test_the_set_exists_and_is_the_documented_size() -> None:
    qs = load_adversarial(ADVERSARIAL)
    assert len(qs) == 30


def test_every_kind_is_represented_and_recognised() -> None:
    qs = load_adversarial(ADVERSARIAL)
    used = {q.kind for q in qs}
    assert used == set(KINDS)


def test_the_set_is_weighted_toward_the_hard_kinds() -> None:
    """A set dominated by wrong_jurisdiction refuses trivially and measures nothing."""
    qs = load_adversarial(ADVERSARIAL)
    hard = sum(1 for q in qs if q.kind in {"adjacent_domain", "uncovered_kyc"})
    assert hard >= 20


def test_every_question_explains_why_it_cannot_be_answered() -> None:
    for q in load_adversarial(ADVERSARIAL):
        assert len(q.why_unanswerable) >= 30, q.qid


def test_qids_are_unique() -> None:
    qs = load_adversarial(ADVERSARIAL)
    assert len({q.qid for q in qs}) == len(qs)


def test_a_malformed_row_names_its_line() -> None:
    bad = Path("tests/fixtures/adversarial-bad.jsonl")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text('{"qid": "x", "question": "q"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_adversarial(bad)
    bad.unlink()
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_adversarial.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.evaluation.adversarial'`

- [ ] **Step 3: Implement the loader**

```python
# src/clause/evaluation/adversarial.py
"""Out-of-corpus questions, for measuring refusal.

Weighted toward near-misses that retrieve well. A question the retriever scores
near zero refuses at every threshold, so a set built from those would report a
high refusal rate while testing nothing about where the threshold should sit.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

KINDS: tuple[str, ...] = (
    "adjacent_domain",
    "uncovered_kyc",
    "time_shifted",
    "wrong_jurisdiction",
)


@dataclass(frozen=True, slots=True)
class AdversarialQuestion:
    qid: str
    question: str
    kind: str
    why_unanswerable: str


def _rows(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if line.strip():
                yield lineno, json.loads(line)


def load_adversarial(path: Path) -> list[AdversarialQuestion]:
    questions: list[AdversarialQuestion] = []
    for lineno, row in _rows(path):
        missing = {"qid", "question", "kind", "why_unanswerable"} - set(row)
        if missing:
            raise ValueError(f"line {lineno}: missing {sorted(missing)}")
        kind = str(row["kind"])
        if kind not in KINDS:
            raise ValueError(f"line {lineno}: unknown kind {kind!r}, expected one of {KINDS}")
        questions.append(
            AdversarialQuestion(
                qid=str(row["qid"]),
                question=str(row["question"]),
                kind=kind,
                why_unanswerable=str(row["why_unanswerable"]),
            )
        )
    return questions
```

- [ ] **Step 4: Draft the 30 questions**

Write `data/adversarial/out-of-corpus-v1.jsonl`, one JSON object per line with keys `qid`, `question`, `kind`, `why_unanswerable`. Use qid prefixes `adj-`, `unc-`, `time-`, `juris-`.

Two worked examples to match in specificity:

```json
{"qid": "adj-001", "question": "What minimum capital conservation buffer must a scheduled commercial bank maintain under the RBI's Basel III capital framework?", "kind": "adjacent_domain", "why_unanswerable": "The corpus is KYC/AML circulars only; capital adequacy directions were never ingested, so no span in any of the 61 documents states this ratio."}
{"qid": "time-001", "question": "What did the RBI's Master Direction on KYC require for periodic updation of low-risk customer records as it stood in March 2019?", "kind": "time_shifted", "why_unanswerable": "The frozen corpus begins at 2023-07-04; the 2019 text of the Master Direction is not present, and answering from the current text would misstate what the rule said then."}
```

**Verify each question does not accidentally have an answer.** For every question, run retrieval and read the top hit:

```bash
uv run python -c "
from pathlib import Path
from qdrant_client import QdrantClient
from clause.embed import Encoder
from clause.retrieve import search
from clause.evaluation.adversarial import load_adversarial

c, e = QdrantClient(url='http://localhost:6335', timeout=30), Encoder()
for q in load_adversarial(Path('data/adversarial/out-of-corpus-v1.jsonl')):
    hits = search(c, e, 'structural', q.question, limit=3)
    top = hits[0] if hits else None
    print(f'{q.qid} {q.kind:18s} score={top.score:.3f} {top.doc_id if top else \"-\"}')
    print('   ', (top.text[:140] if top else 'no hits').replace(chr(10), ' '))
"
```

Report the score distribution. If most questions score below 0.2 the set is too easy and must be rewritten — say so rather than proceeding.

- [ ] **Step 5: Run tests and commit**

```bash
uv run pytest tests/test_adversarial.py -v && uv run ruff check . && uv run mypy
git add data/adversarial src/clause/evaluation/adversarial.py tests/test_adversarial.py
git commit -m "feat(eval): adversarial out-of-corpus set, weighted toward near-misses

A question the retriever scores near zero refuses at every threshold, so a
set built from those reports a high refusal rate while testing nothing about
where the threshold belongs. Two thirds of this set is adjacent-domain and
uncovered-KYC questions that retrieve RBI-shaped text and still cannot be
answered."
```

---

### Task 8: Answering metrics

**Files:**
- Create: `src/clause/evaluation/answer_metrics.py`, `tests/test_answer_metrics.py`

**Interfaces:**
- Consumes: Task 1 types
- Produces:
  - `content_terms(text: str) -> frozenset[str]`
  - `support_signal(sentence: Sentence, citations: Sequence[Citation]) -> float`
  - `SupportDistribution(n, mean, at_or_above_half, zero)` — frozen
  - `summarise_support(values: Sequence[float]) -> SupportDistribution`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_answer_metrics.py
from datetime import date

from clause.answering.schema import Citation, Sentence
from clause.evaluation.answer_metrics import (
    content_terms,
    summarise_support,
    support_signal,
)


def _c(text: str) -> Citation:
    return Citation(
        doc_id="d",
        char_start=0,
        char_end=len(text),
        source_url="u",
        published_date=date(2025, 1, 1),
        doc_type="notification",
        text=text,
    )


def test_content_terms_keeps_numbers_and_capitalised_terms() -> None:
    terms = content_terms("Banks must report within 7 days to the Reserve Bank.")
    assert "7" in terms
    assert "reserve" in terms
    assert "banks" in terms


def test_content_terms_drops_stopwords_and_punctuation() -> None:
    terms = content_terms("the and to of a")
    assert terms == frozenset()


def test_a_sentence_fully_covered_by_its_citation_scores_one() -> None:
    s = Sentence(text="Report within 7 days.", citation_indices=(1,), factual=True)
    assert support_signal(s, [_c("Entities shall report within 7 days of detection.")]) == 1.0


def test_a_sentence_sharing_nothing_with_its_citation_scores_zero() -> None:
    s = Sentence(text="Capital adequacy is 11 percent.", citation_indices=(1,), factual=True)
    assert support_signal(s, [_c("Customer identification requires a valid document.")]) == 0.0


def test_a_non_factual_sentence_is_not_scored() -> None:
    s = Sentence(text="Here is a summary.", citation_indices=(), factual=False)
    assert support_signal(s, []) == 1.0


def test_the_summary_reports_a_distribution_not_a_single_score() -> None:
    """Named so it cannot be quoted as a faithfulness score, because it is not one."""
    d = summarise_support([0.0, 0.5, 1.0, 1.0])
    assert d.n == 4
    assert d.zero == 1
    assert d.at_or_above_half == 3
    assert 0.6 < d.mean < 0.65


def test_an_empty_summary_is_not_a_perfect_one() -> None:
    d = summarise_support([])
    assert d.n == 0
    assert d.mean == 0.0
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_answer_metrics.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'clause.evaluation.answer_metrics'`

- [ ] **Step 3: Implement**

```python
# src/clause/evaluation/answer_metrics.py
"""A support signal, which is a proxy for faithfulness and is not faithfulness.

It measures what fraction of a sentence's content terms appear in the spans it
cites. That catches the gross failure -- a sentence citing a passage it shares no
vocabulary with -- and misses the subtle one, a fluent paraphrase that reverses
the meaning. Real entailment needs a judge model, which this project excludes on
cost, and a 3B model grading its own output would be worse than no metric.

It is reported as a distribution and never as a single "faithfulness score",
specifically so it cannot be quoted as one.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from clause.answering.schema import Citation, Sentence

_WORD = re.compile(r"[A-Za-z0-9]+")

_STOPWORDS = frozenset(
    """a an and are as at be been by for from has have in is it its of on or shall
    that the their to was were will with within must may any such other than these
    this those under upon
    """.split()
)


def content_terms(text: str) -> frozenset[str]:
    """Lower-cased tokens that carry content: numbers and non-stopword words."""
    return frozenset(
        token.lower()
        for token in _WORD.findall(text)
        if token.lower() not in _STOPWORDS and len(token) > 1
    )


def support_signal(sentence: Sentence, citations: Sequence[Citation]) -> float:
    """Fraction of the sentence's content terms present in the spans it cites.

    A non-factual sentence scores 1.0: it asserts nothing, so there is nothing
    for a citation to support.
    """
    if not sentence.factual:
        return 1.0
    terms = content_terms(sentence.text)
    if not terms:
        return 1.0
    cited: set[str] = set()
    for i in sentence.citation_indices:
        if i <= len(citations):
            cited |= content_terms(citations[i - 1].text)
    return len(terms & cited) / len(terms)


@dataclass(frozen=True, slots=True)
class SupportDistribution:
    n: int
    mean: float
    at_or_above_half: int
    zero: int


def summarise_support(values: Sequence[float]) -> SupportDistribution:
    if not values:
        return SupportDistribution(n=0, mean=0.0, at_or_above_half=0, zero=0)
    return SupportDistribution(
        n=len(values),
        mean=sum(values) / len(values),
        at_or_above_half=sum(1 for v in values if v >= 0.5),
        zero=sum(1 for v in values if v == 0.0),
    )
```

- [ ] **Step 4: Run tests and the gate**

```bash
uv run pytest tests/test_answer_metrics.py -v && uv run ruff check . && uv run mypy
```
Expected: 7 passed.

- [ ] **Step 5: Removal proof**

Make `summarise_support([])` return `mean=1.0`, run the file, confirm `test_an_empty_summary_is_not_a_perfect_one` fails. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/clause/evaluation/answer_metrics.py tests/test_answer_metrics.py
git commit -m "feat(eval): term-overlap support signal, named as the proxy it is

It catches a sentence citing a passage it shares no vocabulary with, and
misses a fluent paraphrase that reverses the meaning. Reported as a
distribution rather than a score so it cannot be quoted as faithfulness,
which it is not."
```

---

### Task 9: The answering report

**Files:**
- Create: `src/clause/evaluation/answer_report.py`, `tests/test_answer_report.py`

**Interfaces:**
- Consumes: Tasks 1, 4, 7, 8
- Produces:
  - `AnswerFingerprint(model_file, model_sha256, n_ctx, threshold, golden_sha256, adversarial_sha256, corpus_documents, git_commit)` — frozen, `.to_dict()`
  - `ThresholdRow(threshold, answered, refused, false_refusals, adversarial_refused, adversarial_answered)` — frozen
  - `build_answer_report(rows, fingerprint, support, resolution_rate) -> dict`
  - `render_answer_markdown(report: dict) -> str`
  - `write_answer_report(report, md_path, json_path) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_answer_report.py
import json
from pathlib import Path

from clause.evaluation.answer_metrics import SupportDistribution
from clause.evaluation.answer_report import (
    AnswerFingerprint,
    ThresholdRow,
    build_answer_report,
    render_answer_markdown,
    write_answer_report,
)

FP = AnswerFingerprint(
    model_file="qwen2.5-3b-instruct-q4_k_m.gguf",
    model_sha256="m" * 64,
    n_ctx=4096,
    threshold=0.35,
    golden_sha256="g" * 64,
    adversarial_sha256="a" * 64,
    corpus_documents=61,
    git_commit="abc1234",
)

ROWS = (
    ThresholdRow(0.0, answered=65, refused=0, false_refusals=0, adversarial_refused=0, adversarial_answered=30),
    ThresholdRow(0.5, answered=20, refused=45, false_refusals=8, adversarial_refused=22, adversarial_answered=8),
)

SUPPORT = SupportDistribution(n=65, mean=0.61, at_or_above_half=44, zero=6)


def test_the_report_carries_its_fingerprint() -> None:
    r = build_answer_report(ROWS, FP, SUPPORT, resolution_rate=1.0)
    assert r["fingerprint"]["model_sha256"] == "m" * 64


def test_the_report_states_the_support_signal_is_not_faithfulness() -> None:
    md = render_answer_markdown(build_answer_report(ROWS, FP, SUPPORT, resolution_rate=1.0))
    assert "not entailment" in md
    assert "faithfulness score" not in md.replace("not a faithfulness score", "")


def test_the_report_renders_the_whole_sweep() -> None:
    md = render_answer_markdown(build_answer_report(ROWS, FP, SUPPORT, resolution_rate=1.0))
    assert "0.00" in md and "0.50" in md


def test_a_resolution_rate_below_one_is_called_out_loudly() -> None:
    """It is 1.0 by construction; anything else means the enforcement path broke."""
    md = render_answer_markdown(build_answer_report(ROWS, FP, SUPPORT, resolution_rate=0.98))
    assert "CITATION CONTRACT BREACH" in md


def test_a_resolution_rate_of_one_says_so_without_alarm() -> None:
    md = render_answer_markdown(build_answer_report(ROWS, FP, SUPPORT, resolution_rate=1.0))
    assert "CITATION CONTRACT BREACH" not in md


def test_write_emits_both_artifacts(tmp_path: Path) -> None:
    r = build_answer_report(ROWS, FP, SUPPORT, resolution_rate=1.0)
    md, js = tmp_path / "answers.md", tmp_path / "answers.json"
    write_answer_report(r, md, js)
    assert "sweep" in md.read_text(encoding="utf-8").lower()
    assert json.loads(js.read_text(encoding="utf-8"))["fingerprint"]["git_commit"] == "abc1234"
```

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_answer_report.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Build `build_answer_report` returning `{"fingerprint": ..., "resolution_rate": ..., "support": asdict(...), "sweep": [asdict(r) for r in rows]}`, and `render_answer_markdown` emitting, in order:

1. A header saying every figure comes from `make answer-eval` and nothing is estimated.
2. **The citation-resolution line.** When `resolution_rate < 1.0`, emit a line beginning `**CITATION CONTRACT BREACH:**` naming the rate, because a value below 1.0 means the enforcement path itself failed rather than that some answers were weaker.
3. **The sweep table** — one row per threshold: `threshold | answered | refused | false refusals | adversarial refused | adversarial answered`.
4. **The support-signal section**, which must contain the sentence: *"This is a term-overlap proxy, not entailment — it catches a sentence citing a passage it shares no vocabulary with, and misses a fluent paraphrase that reverses the meaning."*
5. **Limitations**, carrying: the golden set is model-drafted; `effective_date` is absent for every document so citations cannot state when a rule took effect; the support signal is not faithfulness; and the adversarial set's size, so its refusal rate is read with the right error bars.
6. The fingerprint table.

`write_answer_report` writes both with `newline="\n"`, JSON with `indent=2, sort_keys=True` and a trailing newline.

- [ ] **Step 4: Run tests, gate, commit**

```bash
uv run pytest tests/test_answer_report.py -v && uv run ruff check . && uv run mypy
git add src/clause/evaluation/answer_report.py tests/test_answer_report.py
git commit -m "feat(eval): answering report carrying its own fingerprint

Citation resolution is 1.0 by construction, so it is reported as an
assertion: anything below 1.0 is rendered as a contract breach, because it
would mean the enforcement path broke rather than that answers got weaker."
```

---

### Task 10: `make answer-eval` end to end

**Files:**
- Create: `src/clause/evaluation/answer_run.py`, `tests/test_answer_run.py`
- Modify: `src/clause/cli.py`, `Makefile`

**Interfaces:**
- Produces:
  - `answer_one(question, hits, answerer, session, threshold) -> Answer | Refusal`
  - `run_answer_eval(md_path=DEFAULT_ANSWERS_MD, json_path=DEFAULT_ANSWERS_JSON) -> int`
  - `python -m clause.cli answer-eval`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_answer_run.py
from pathlib import Path

import pytest

from clause.answering.base import StubAnswerer
from clause.answering.schema import Answer, Refusal, RefusalReason
from clause.evaluation.answer_run import answer_one

pytestmark = pytest.mark.db


def test_a_weak_top_score_refuses_before_any_generation(db_session, monkeypatch) -> None:
    """The refusal gate must not pay for a model call it does not need."""
    called = False

    class ExplodingAnswerer:
        def answer(self, question, hits):
            nonlocal called
            called = True
            raise AssertionError("generation must not run on a refused question")

    result = answer_one("q", [], ExplodingAnswerer(), db_session, threshold=0.35)
    assert isinstance(result, Refusal)
    assert result.reason == RefusalReason.NO_CANDIDATES
    assert called is False


def test_a_strong_hit_produces_an_answer_whose_citations_resolved(
    db_session, seeded_hit
) -> None:
    result = answer_one("q", [seeded_hit], StubAnswerer(), db_session, threshold=0.0)
    assert isinstance(result, Answer)
    assert result.citations
    assert result.citations[0].text


def test_answers_never_carry_an_unresolved_citation(db_session, seeded_hit) -> None:
    """The DoD, asserted directly: no response passes with an unresolvable citation."""
    result = answer_one("q", [seeded_hit], StubAnswerer(), db_session, threshold=0.0)
    assert isinstance(result, Answer)
    for c in result.citations:
        assert c.char_start < c.char_end
        assert len(c.text) == c.char_end - c.char_start


def test_run_writes_to_the_given_paths_not_the_defaults(tmp_path: Path) -> None:
    from clause.evaluation.answer_run import DEFAULT_ANSWERS_JSON, DEFAULT_ANSWERS_MD

    assert DEFAULT_ANSWERS_MD != tmp_path / "answers.md"
    assert DEFAULT_ANSWERS_JSON != tmp_path / "answers.json"
```

Add a `seeded_hit` fixture to `tests/conftest.py` that inserts a `DocumentRow` with known text and returns a matching `Hit`.

- [ ] **Step 2: Run and watch them fail**

```bash
uv run pytest tests/test_answer_run.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`answer_one` in order: `refusal.decide(...)` → return the `Refusal` if any (**no model call**) → `answerer.answer(...)` → `resolve_citations(...)` → `validate.enforce(...)`.

`run_answer_eval` in order:

1. Load the golden set and the adversarial set.
2. Build `Encoder` and `QdrantClient`; build `LlamaAnswerer` from `settings.answer_model_path`.
3. For each golden question and each adversarial question: retrieve once at `RETRIEVAL_DEPTH`, record the top score, and answer at `settings.answer_threshold`.
4. Compute the sweep from the recorded scores — **retrieval runs once per question, not once per threshold**; the sweep is arithmetic over stored scores.
5. A *false refusal* is a golden question refused at a threshold whose answer *was* retrievable (its `first_hit_rank` is not `None`).
6. Compute the support distribution over every factual sentence produced at the default threshold.
7. Compute resolution rate: answers returned without an `UnresolvableCitationError` ÷ answers attempted.
8. Write both artifacts. Print the sweep table to stdout.

**Output paths are parameters defaulting to `reports/answers.md` and `reports/answers.json`.** Tests pass `tmp_path`. The suite must never rewrite a committed artifact — this project has already had a test fixture destroy its application database.

Add the `answer-eval` subcommand to `cli.py` and to the `Makefile`:

```make
answer-eval:
	uv run python -m clause.cli answer-eval
```

- [ ] **Step 4: Run the real thing**

```bash
uv run python -m clause.cli answer-eval
cat reports/answers.md
```

Expect roughly 45-90 minutes for 95 questions. **Report the numbers; do not tune anything.** A first honest number is the point. If the model produces poor citations, that is a finding about a 3B model on this task, not a failure to fix by prompt-fiddling.

- [ ] **Step 5: Run tests and commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy
git add src/clause/evaluation/answer_run.py tests/test_answer_run.py tests/conftest.py \
        src/clause/cli.py Makefile reports/answers.md reports/answers.json
git commit -m "feat(eval): answer-eval scores the golden and adversarial sets

Retrieval runs once per question and the threshold sweep is arithmetic over
the stored scores, so the curve costs one retrieval pass rather than twenty.
The refusal gate runs before generation, so a refused question never pays for
a model call."
```

---

### Task 11: Documentation and housekeeping

**Files:**
- Modify: `README.md`, `STATUS.md`, `.github/workflows/ci.yml`

- [ ] **Step 1: README**

Add a "Running the answering evaluation" section: `uv sync --extra answer`, fetching the GGUF into `data/models/`, `make answer-eval`, and the expected wall-clock. State that it is separate from `make eval` because it is slow and CI cannot run it. **No numbers** — point at `reports/answers.md`.

- [ ] **Step 2: STATUS.md**

Mark Phase 3 done, pointing at `reports/answers.md`. **No numbers.** Keep the existing honest wording about CI never having executed.

- [ ] **Step 3: CI**

Confirm CI installs **without** the `answer` extra, so `llama-cpp-python` is never built there. Add a step asserting the base install does not pull it:

```yaml
      - run: uv run python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('llama_cpp') is None else 1)"
```

This is a real guard: if someone moves the dependency into the base install, CI starts trying to compile a C++ extension and the failure is confusing. Watch it fail by temporarily moving the dependency, then restore.

- [ ] **Step 4: Verify the whole suite and commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy
uv run python -m clause.cli gate   # Phase 2's gate must still pass untouched
git add README.md STATUS.md .github/workflows/ci.yml
git commit -m "docs: record phase 3 and keep the answering runtime out of CI

CI asserts llama_cpp is absent from the base install. Moving that dependency
out of its optional extra would make CI try to compile a C++ extension, and
the resulting failure would not obviously point at the cause."
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §4 Architecture, five units | 1-6 |
| §5.1 Three distinct failures | 1 (types), 2 (unresolvable), 3 (uncited, unsupported not fatal) |
| §5.2 Resolution by slicing | 2 |
| §5.3 Provenance and the `effective_date` gap | 2 (carried), 9 (stated in Limitations) |
| §6.1 Citation resolution rate | 9, 10 |
| §6.2 Answer/refusal split | 4, 9, 10 |
| §6.3 Support signal as a proxy | 8, 9 |
| §6.4 Refusal correctness | 9, 10 |
| §7 Adversarial set | 7 |
| §8 Separate from `make eval`; not gated | 10, 11 |
| §9 Testing, `llm` marker | 6, and every task's tests |
| §10 Failure handling | 1 (errors), 2, 3, 6 (absent model), 10 (`no_candidates`) |
| §11 Deferred items | carried forward, not implemented |

**Placeholder scan:** no TBD/TODO. Task 7 is content work and says so, with two worked examples and a verification command. Task 9's step 3 describes required *content* rather than pasting a 200-line renderer; every element it must contain is enumerated and each is asserted by a test in step 1.

**Type consistency:** `Citation`, `Sentence`, `AnswerDraft`, `Answer`, `Refusal` are defined once in Task 1 and used unchanged in 2, 3, 5, 8, 9, 10. `Hit` comes from Phase 2 and is never redefined. `resolve_citations` returns `tuple[Citation, ...]`, which is what `enforce` consumes. `SupportDistribution` is produced in Task 8 and consumed in Task 9. `Answerer` is defined in Task 5 and implemented in Tasks 5 and 6.

## Known risks

1. **A 3B model may cite badly.** The grammar guarantees a well-formed, in-range citation; nothing guarantees a *relevant* one. If the support distribution is poor, that is the finding — report it, do not tune the prompt until the number improves.
2. **Recall@5 is 0.446/0.338.** At any useful threshold a large fraction of golden questions will refuse. This is inherited from retrieval, not caused here, and the sweep is what makes it legible.
3. **Wall clock.** ~95 questions at 30-60 s is 45-90 minutes per run. Budget for one or two full runs, not iterative experimentation.
4. **The adversarial set is authored by the same model family being evaluated** — the same caveat the golden set carries. The report states it.
