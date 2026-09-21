from collections.abc import Callable
from datetime import date

import pytest

from clause.answering.schema import (
    MAX_CITATIONS_PER_SENTENCE,
    Answer,
    AnswerDraft,
    AnsweringError,
    Citation,
    InvalidAnswerError,
    Sentence,
    TooManyCitationsError,
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
        question="q",
        sentences=(Sentence(text="a", citation_indices=(1,), factual=True),),
        hits=(),
    )
    answer = enforce(draft, {1: _c()})
    assert answer.question == "q"
    assert len(answer.citations) == 1


def test_a_factual_sentence_without_a_citation_raises() -> None:
    """This is the single enforcement point, so the object constructs normally."""
    draft = AnswerDraft(
        question="the-question",
        sentences=(Sentence(text="Banks must verify.", citation_indices=(), factual=True),),
        hits=(),
    )
    with pytest.raises(UncitedClaimError, match="Banks must verify") as exc_info:
        enforce(draft, {})
    assert "the-question" in str(exc_info.value)


def test_an_index_with_no_matching_citation_raises() -> None:
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="a", citation_indices=(2,), factual=True),),
        hits=(),
    )
    with pytest.raises(UnresolvableCitationError, match="index 2") as exc_info:
        enforce(draft, {1: _c()})
    # the message names the keys that actually resolved, not a vestigial count
    assert "resolved: [1]" in str(exc_info.value)


def test_more_citations_than_the_cap_raises() -> None:
    n = MAX_CITATIONS_PER_SENTENCE + 1
    draft = AnswerDraft(
        question="q",
        sentences=(
            Sentence(text="a", citation_indices=tuple(range(1, n + 1)), factual=True),
        ),
        hits=(),
    )
    resolved = {i: _c(i * 4, i * 4 + 4) for i in range(1, n + 1)}
    with pytest.raises(TooManyCitationsError, match="at most"):
        enforce(draft, resolved)


def test_an_answer_with_no_factual_sentences_needs_no_citations() -> None:
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="I could not find this.", citation_indices=(), factual=False),),
        hits=(),
    )
    answer = enforce(draft, {})
    assert answer.citations == ()


def test_unused_citations_are_dropped_not_reported() -> None:
    """A citation nothing references is not evidence; carrying it would inflate the count."""
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="a", citation_indices=(1,), factual=True),),
        hits=(),
    )
    answer = enforce(draft, {1: _c(0, 4), 2: _c(4, 8)})
    assert len(answer.citations) == 1


def test_citing_only_a_non_first_hit_produces_an_answer() -> None:
    """Pins the seam bug: a draft citing only hit 2 must not be treated as citing a
    citation that does not exist just because the resolved mapping's *size* is 1.
    """
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="a", citation_indices=(2,), factual=True),),
        hits=(),
    )
    answer = enforce(draft, {2: _c()})
    assert len(answer.citations) == 1
    assert answer.sentences[0].citation_indices == (1,)


def test_citation_indices_are_renumbered_to_positions_in_answer_citations() -> None:
    """Every index any sentence in the answer carries must resolve positionally:
    `answer.citations[i - 1]` is correct for any `i` in any sentence's indices.
    """
    hit1 = _c(0, 4)
    hit3 = _c(8, 12)
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="a", citation_indices=(3, 1), factual=True),),
        hits=(),
    )
    answer = enforce(draft, {1: hit1, 3: hit3})
    new_indices = answer.sentences[0].citation_indices
    assert len(new_indices) == 2
    for i in new_indices:
        assert answer.citations[i - 1] in (hit1, hit3)
    # relative order preserved: original (3, 1) -> renumbered positions of hit3, hit1
    resolved_citations = [answer.citations[i - 1] for i in new_indices]
    assert resolved_citations == [hit3, hit1]


def test_an_index_absent_from_the_resolved_mapping_raises() -> None:
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="a", citation_indices=(5,), factual=True),),
        hits=(),
    )
    with pytest.raises(UnresolvableCitationError, match="index 5"):
        enforce(draft, {1: _c()})


def test_an_empty_draft_does_not_become_an_answer() -> None:
    """A draft with no sentences says nothing; `enforce` must not hand it back as
    an answered contract with a vacuous, always-1.0 resolution rate."""
    draft = AnswerDraft(question="q", sentences=(), hits=())
    with pytest.raises(InvalidAnswerError, match="at least one sentence"):
        enforce(draft, {})


def _direct_answer_out_of_range_index() -> None:
    Answer(
        question="q",
        sentences=(Sentence(text="Banks must verify.", citation_indices=(7,), factual=True),),
        citations=(),
    )


def _empty_draft_via_enforce() -> None:
    enforce(AnswerDraft(question="q", sentences=(), hits=()), {})


def _cap_exceeded_via_enforce() -> None:
    n = MAX_CITATIONS_PER_SENTENCE + 1
    draft = AnswerDraft(
        question="q",
        sentences=(
            Sentence(text="a", citation_indices=tuple(range(1, n + 1)), factual=True),
        ),
        hits=(),
    )
    resolved = {i: _c(i * 4, i * 4 + 4) for i in range(1, n + 1)}
    enforce(draft, resolved)


def _uncited_factual_sentence_via_enforce() -> None:
    draft = AnswerDraft(
        question="q",
        sentences=(Sentence(text="Banks must verify.", citation_indices=(), factual=True),),
        hits=(),
    )
    enforce(draft, {})


@pytest.mark.parametrize(
    "reject",
    [
        _direct_answer_out_of_range_index,
        _empty_draft_via_enforce,
        _cap_exceeded_via_enforce,
        _uncited_factual_sentence_via_enforce,
    ],
    ids=[
        "direct-answer-out-of-range-index",
        "empty-draft-via-enforce",
        "cap-exceeded-via-enforce",
        "uncited-factual-sentence-via-enforce",
    ],
)
def test_every_contract_rejection_path_is_catchable_as_answering_error(
    reject: Callable[[], None],
) -> None:
    """Task 10 writes `except AnsweringError:` to record a contract breach. Every
    way the contract can be rejected -- whether raised directly by `Answer`'s own
    invariant or by `enforce` -- must be catchable through that one hierarchy, or
    a future fifth path that forgets it discovers this in Task 10 instead of here.

    Also asserts the raised exception is a `ValueError`: `AnsweringError` inherits
    from it precisely so every existing `except ValueError` / `pytest.raises
    (ValueError)` call site keeps working alongside the new `except
    AnsweringError`. If `AnsweringError` ever stopped inheriting from `ValueError`,
    this assertion is what would catch it here, not a mismatched call site
    discovered later.
    """
    with pytest.raises(AnsweringError) as exc_info:
        reject()
    assert isinstance(exc_info.value, ValueError)
