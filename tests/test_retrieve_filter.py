"""Unit tests for `_build_filter`, a pure function.

No Qdrant, no embedding model, no fixtures -- so no `qdrant`/`model` marker is
needed here, unlike the rest of `tests/test_retrieve.py`. Filed separately so
that CI can run this coverage on every push regardless of whether a live
Qdrant/model is available.
"""

from datetime import date

from qdrant_client import models

from clause.retrieve import _build_filter


def _conditions(filter_: models.Filter | None) -> list[models.Condition]:
    assert filter_ is not None
    assert filter_.must is not None
    return list(filter_.must)


def test_no_bounds_and_no_entities_is_no_filter_at_all() -> None:
    assert _build_filter(None, None, None) is None


def test_published_after_alone_sets_only_gte() -> None:
    conditions = _conditions(_build_filter(date(2025, 1, 1), None, None))
    assert len(conditions) == 1
    condition = conditions[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.key == "published_date"
    assert isinstance(condition.range, models.DatetimeRange)
    assert condition.range.gte == date(2025, 1, 1)
    assert condition.range.lte is None


def test_published_before_alone_sets_only_lte() -> None:
    conditions = _conditions(_build_filter(None, date(2025, 12, 31), None))
    assert len(conditions) == 1
    condition = conditions[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.key == "published_date"
    assert isinstance(condition.range, models.DatetimeRange)
    assert condition.range.gte is None
    assert condition.range.lte == date(2025, 12, 31)


def test_both_bounds_together_sets_gte_and_lte() -> None:
    conditions = _conditions(_build_filter(date(2025, 1, 1), date(2025, 12, 31), None))
    assert len(conditions) == 1
    condition = conditions[0]
    assert isinstance(condition, models.FieldCondition)
    assert isinstance(condition.range, models.DatetimeRange)
    # This is the swap that finding 1 warns about: gte must be the lower bound
    # (published_after) and lte the upper bound (published_before). Swapping
    # them would pass every other test in this suite silently.
    assert condition.range.gte == date(2025, 1, 1)
    assert condition.range.lte == date(2025, 12, 31)


def test_one_entity_sets_match_any_with_that_entity() -> None:
    conditions = _conditions(_build_filter(None, None, ["Commercial Banks"]))
    assert len(conditions) == 1
    condition = conditions[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.key == "regulated_entity"
    assert isinstance(condition.match, models.MatchAny)
    assert condition.match.any == ["Commercial Banks"]


def test_several_entities_sets_match_any_with_all_of_them() -> None:
    entities = ["Commercial Banks", "Small Finance Banks", "Payment Banks"]
    conditions = _conditions(_build_filter(None, None, entities))
    assert len(conditions) == 1
    condition = conditions[0]
    assert isinstance(condition, models.FieldCondition)
    assert isinstance(condition.match, models.MatchAny)
    assert condition.match.any == entities


def test_an_empty_entities_list_is_treated_as_no_constraint() -> None:
    """Pinned decision (finding 2): `entities=[]` and `entities=None` behave
    identically -- both mean "no entity constraint", not "match none of the
    zero entities named". This is a deliberate choice, not an accident: "no
    entities selected" is the less surprising reading of an empty selection,
    and Phase 3 will build queries from exactly this kind of upstream input.
    If this reading is ever reversed, it must be a decision made here, not a
    side effect of some other change -- hence the explicit test.
    """
    assert _build_filter(None, None, []) is None


def test_date_and_entity_conditions_are_and_composed() -> None:
    """Both a date bound and an entity constraint together produce one Filter
    with both conditions in `must` -- Qdrant's AND, not either alone.
    """
    conditions = _conditions(
        _build_filter(date(2025, 1, 1), None, ["Commercial Banks"])
    )
    assert len(conditions) == 2
    keys = {c.key for c in conditions if isinstance(c, models.FieldCondition)}
    assert keys == {"published_date", "regulated_entity"}
