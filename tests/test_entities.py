from pathlib import Path

from clause.entities import ENTITY_VOCABULARY, parse_regulated_entities
from clause.ingest.extract import canonical_text

FIXTURE = Path(__file__).parent / "fixtures" / "rbi_13704.html"
REAL = canonical_text(FIXTURE.read_text(encoding="utf-8", errors="replace"))

ADDRESSEE = (
    "RBI/2026-27/253 DOR.AML.REC.1/14.01.005/2026-27 September 18, 2026 "
    "The Chairpersons/ CEOs of the Commercial Banks, Small Finance Banks, "
    "Payment Banks, Urban Co-operative Banks, Regional Rural Banks, "
    "Non-Banking Financial Companies Madam/Dear Sir, Subject line here"
)


def test_parses_the_entities_from_an_addressee_block() -> None:
    found = parse_regulated_entities(ADDRESSEE)
    assert "Commercial Banks" in found
    assert "Small Finance Banks" in found
    assert "Non-Banking Financial Companies" in found


def test_result_is_ordered_and_deduplicated() -> None:
    doubled = ADDRESSEE.replace("Commercial Banks", "Commercial Banks, Commercial Banks")
    found = parse_regulated_entities(doubled)
    assert len(found) == len(set(found))
    assert list(found) == sorted(found)


def test_every_result_comes_from_the_closed_vocabulary() -> None:
    assert set(parse_regulated_entities(ADDRESSEE)) <= set(ENTITY_VOCABULARY)


def test_a_document_without_an_addressee_block_yields_nothing() -> None:
    """Empty, never a guess -- an entity filter is only useful if absence is visible."""
    assert parse_regulated_entities("RBI/2026-27/1 DOR.X.1/1 January 1, 2026 no addressee") == ()


def test_text_without_a_header_yields_nothing() -> None:
    assert parse_regulated_entities("nothing resembling a circular") == ()


def test_the_real_fixture_parses_without_guessing() -> None:
    """rbi_13704 is a short "Amendment Directions" notice: it amends an existing
    Master Direction and signs off with the Chief General Manager, but carries no
    addressee paragraph at all in the source HTML (verified directly against the
    raw fixture: zero occurrences of "Madam", "Dear Sir", or "Chairperson"). The
    brief's original assertion here ("yields at least one entity") does not hold
    against this real fixture's actual content. Per the rule that matters most --
    never guess -- the correct result for a document with no addressee block is
    the empty tuple, not a forced non-empty one.
    """
    assert parse_regulated_entities(REAL) == ()


def test_only_the_addressee_block_is_scanned() -> None:
    """A body mention of an entity class must not become a filter value."""
    body_mention = (
        "RBI/2026-27/1 DOR.X.1/1 January 1, 2026 "
        "The Chairpersons of the Commercial Banks Madam/Dear Sir, "
        "This circular also concerns Payment Banks in passing."
    )
    found = parse_regulated_entities(body_mention)
    assert "Commercial Banks" in found
    assert "Payment Banks" not in found
