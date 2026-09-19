from pathlib import Path

import pytest

from clause.htmltext import visible_text
from clause.ingest.validate import ValidationError, validate_response

FIXTURES = Path(__file__).parent / "fixtures"
CIRCULAR = (FIXTURES / "rbi_13704.html").read_text(encoding="utf-8", errors="replace")
BLOCK = (FIXTURES / "rbi_block_page.html").read_text(encoding="utf-8", errors="replace")


def _ok(body: str) -> None:
    validate_response(status_code=200, content_type="text/html", body=body, min_chars=500)


def test_real_circular_passes() -> None:
    _ok(CIRCULAR)


def test_block_page_is_rejected_despite_http_200() -> None:
    with pytest.raises(ValidationError, match="bot-check"):
        _ok(BLOCK)


def test_non_200_is_rejected() -> None:
    with pytest.raises(ValidationError, match="status"):
        validate_response(status_code=503, content_type="text/html", body=CIRCULAR, min_chars=500)


def test_non_html_is_rejected() -> None:
    with pytest.raises(ValidationError, match="content type"):
        validate_response(
            status_code=200, content_type="application/pdf", body=CIRCULAR, min_chars=500
        )


def test_html_without_a_circular_reference_is_rejected() -> None:
    with pytest.raises(ValidationError, match="circular reference"):
        _ok("<html><body>" + "padding text " * 200 + "</body></html>")


def test_short_document_is_rejected() -> None:
    with pytest.raises(ValidationError, match="too short"):
        validate_response(
            status_code=200,
            content_type="text/html",
            body="<html><body>RBI/2026-27/262 short</body></html>",
            min_chars=500,
        )


def test_visible_text_strips_scripts_and_styles() -> None:
    html = (
        "<html><head><style>p{color:red}</style></head>"
        "<body><script>x=1</script><p>Hello</p></body></html>"
    )
    out = visible_text(html)
    assert "Hello" in out
    assert "color" not in out
    assert "x=1" not in out


def test_visible_text_decodes_entities_and_collapses_whitespace() -> None:
    assert visible_text("<p>a &amp;   b\n\n\nc</p>") == "a & b c"
