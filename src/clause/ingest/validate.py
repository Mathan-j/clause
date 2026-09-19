import re

from clause.htmltext import visible_text

BLOCK_MARKERS: tuple[str, ...] = ("Unauthorised Access", "Support ID:")
CIRCULAR_REFERENCE = re.compile(r"RBI/\d{4}-\d{2}/\d+")
HTTP_OK = 200


class ValidationError(Exception):
    """A response that must never become a cached document."""


def validate_response(*, status_code: int, content_type: str, body: str, min_chars: int) -> None:
    """Raise unless this response is plausibly a real RBI circular.

    The offset round-trip proves fidelity, not correctness: it would pass just
    as happily on fifty identical bot-check pages. Because a blocked request
    returns HTTP 200 with an HTML body, the status code cannot stand in for
    this check.
    """
    if status_code != HTTP_OK:
        raise ValidationError(f"bad status: {status_code}")

    if "html" not in content_type.lower():
        raise ValidationError(f"unexpected content type: {content_type!r}")

    for marker in BLOCK_MARKERS:
        if marker in body:
            raise ValidationError(f"bot-check page detected (marker {marker!r})")

    if not CIRCULAR_REFERENCE.search(body):
        raise ValidationError("no RBI circular reference found in body")

    text = visible_text(body)
    if len(text) < min_chars:
        raise ValidationError(f"document too short: {len(text)} < {min_chars} chars")
