"""`visible_text` sees the whole page; `document_text` sees only the document.

The distinction is load-bearing. Before it existed, extraction stored the whole
page, and more than half of every stored document's characters were RBI's site
template rather than regulation.
"""

from pathlib import Path

import pytest
from selectolax.parser import HTMLParser

from clause.htmltext import CONTENT_SELECTOR, document_text, visible_text

FIXTURE = Path(__file__).parent / "fixtures" / "rbi_13704.html"
REAL = FIXTURE.read_text(encoding="utf-8", errors="replace")

PAGE = (
    "<html><body>"
    "<nav>Skip to main content About Us Notifications</nav>"
    '<div id="NotificationUser">RBI/2026-27/262 the actual circular text</div>'
    "<footer>Website last updated date: Sep 19, 2026</footer>"
    "</body></html>"
)


def test_document_text_returns_only_the_content_node() -> None:
    assert document_text(PAGE) == "RBI/2026-27/262 the actual circular text"


def test_visible_text_still_returns_the_whole_page() -> None:
    whole = visible_text(PAGE)
    assert "Skip to main content" in whole
    assert "Website last updated" in whole
    assert "the actual circular text" in whole


def test_document_text_falls_back_to_the_body_when_the_node_is_absent() -> None:
    """Degrading to the old behaviour beats returning nothing.

    An empty result would fail extraction's length check and silently drop a real
    circular out of the corpus.
    """
    without = PAGE.replace('id="NotificationUser"', 'id="SomethingElse"')
    out = document_text(without)
    assert "the actual circular text" in out
    assert "Skip to main content" in out  # fallback is the whole body


def test_the_real_fixture_loses_its_chrome_but_keeps_its_header() -> None:
    whole, doc = visible_text(REAL), document_text(REAL)
    assert "Skip to main content" in whole
    assert "Skip to main content" not in doc
    assert "Website last updated" in whole
    assert "Website last updated" not in doc
    assert "RBI/2026-27/262" in doc
    assert len(doc) < len(whole) / 2


def test_content_selector_is_present_in_every_committed_document() -> None:
    """The selector is an anchor, not a guess — if RBI renames it, this fails."""
    cached = sorted(Path("data/raw").glob("*.html"))
    if not cached:
        pytest.skip("no warm data/raw/ cache; run the ingest first")
    missing = [
        f.name
        for f in cached
        if HTMLParser(f.read_text(encoding="utf-8", errors="replace")).css_first(CONTENT_SELECTOR)
        is None
    ]
    assert not missing, f"{len(missing)} cached documents lack {CONTENT_SELECTOR}: {missing[:5]}"
