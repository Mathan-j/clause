from pathlib import Path

import pytest

from clause.entities import ADDRESSEE_WINDOW_CHARS, ENTITY_VOCABULARY, parse_regulated_entities
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


@pytest.mark.parametrize(
    "spacing",
    ["/ CEOs", "/CEOs", " / CEOs"],
    ids=["slash-space-ceos", "slash-ceos", "space-slash-space-ceos"],
)
def test_blanket_addressee_expands_to_the_full_vocabulary(spacing: str) -> None:
    """"The Chairpersons/ CEOs of all the Regulated Entities" is not a vocabulary
    gap -- it is the document stating its own audience as universal, observed
    verbatim (with these three spacing variants) across 27 of the 61 corpus
    documents. Recording it as empty would assert the circular governs nobody,
    the opposite of what it says, and would make a per-entity filter wrongly
    exclude a document that legitimately governs that entity.
    """
    blanket = (
        "RBI/2025-26/97 DOR.AML.REC.61/14.06.001/2025-26 November 14, 2025 "
        f"The Chairpersons{spacing} of all the Regulated Entities Madam/Dear Sir, "
        "Implementation of Section 51A of UAPA, 1967"
    )
    assert parse_regulated_entities(blanket) == tuple(sorted(ENTITY_VOCABULARY))


def test_blanket_addressee_matches_regardless_of_internal_whitespace_and_case() -> None:
    blanket = (
        "RBI/2025-26/1 DOR.X.1/1 January 1, 2026 "
        "The Chairpersons/CEOs of  ALL   THE regulated  entities Madam/Dear Sir, Subject"
    )
    assert parse_regulated_entities(blanket) == tuple(sorted(ENTITY_VOCABULARY))


def test_explicit_list_addressee_does_not_expand_to_the_full_vocabulary() -> None:
    """The explicit-list form names specific classes; only those are returned --
    the blanket expansion must not fire just because some entities are named.
    """
    found = parse_regulated_entities(ADDRESSEE)
    assert found != tuple(sorted(ENTITY_VOCABULARY))
    assert "Asset Reconstruction Companies" not in found


def test_no_addressee_still_yields_nothing() -> None:
    """A document with neither an explicit list nor the blanket phrase gets an
    empty list -- the correct result for 18 of the corpus's 61 documents, which
    run straight from the header into the title with no addressee block at all.
    """
    assert parse_regulated_entities(
        "RBI/2026-27/1 DOR.X.1/1 January 1, 2026 Amendment Directions, 2026 Text follows."
    ) == ()


def test_addressee_within_the_window_parses() -> None:
    header = "RBI/2026-27/1 DOR.X.1/1 January 1, 2026 "
    padding = "x " * ((ADDRESSEE_WINDOW_CHARS - 200) // 2)
    text = f"{header}The Chairpersons of the Commercial Banks {padding}Madam/Dear Sir, Subject"
    assert "Commercial Banks" in parse_regulated_entities(text)


def test_addressee_pushed_past_the_window_yields_nothing() -> None:
    """Pins today's behaviour at the `ADDRESSEE_WINDOW_CHARS` boundary: an
    addressee block long enough to push the salutation past the window parses
    empty rather than erroring. Not observed in the current corpus -- this test
    exists so a future change to the constant (or an unusually long addressee
    block) becomes a visible test failure instead of a silent coverage loss.
    """
    header = "RBI/2026-27/1 DOR.X.1/1 January 1, 2026 "
    padding = "x " * ((ADDRESSEE_WINDOW_CHARS + 200) // 2)
    text = f"{header}The Chairpersons of the Commercial Banks {padding}Madam/Dear Sir, Subject"
    assert parse_regulated_entities(text) == ()
