from pathlib import Path

import pytest

from clause.evaluation.adversarial import KINDS, load_adversarial

ADVERSARIAL = Path("data/adversarial/out-of-corpus-v1.jsonl")


def test_the_set_exists_and_is_the_documented_size() -> None:
    qs = load_adversarial(ADVERSARIAL)
    assert len(qs) == 15


def test_every_kind_is_represented_and_recognised() -> None:
    qs = load_adversarial(ADVERSARIAL)
    used = {q.kind for q in qs}
    assert used == set(KINDS)


def test_the_set_is_weighted_toward_the_hard_kinds() -> None:
    """A set dominated by wrong_jurisdiction refuses trivially and measures nothing."""
    qs = load_adversarial(ADVERSARIAL)
    hard = sum(1 for q in qs if q.kind in {"adjacent_domain", "uncovered_kyc"})
    assert hard >= 10


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
