"""Parse the regulated entities a circular is addressed to.

RBI names its audience explicitly between the header line and the salutation:

    The Chairpersons/ CEOs of the Commercial Banks, Small Finance Banks, ...
    Madam/Dear Sir,

That block is structured data the corpus already contains, and it is what makes
`PROMPT.md`'s "regulated entity" metadata filter real rather than aspirational.
"""

import re

from clause.ingest.extract import HEADER

#: Closed vocabulary. A value outside it is not emitted, because a filter whose
#: values are open-ended cannot be offered in a UI or asserted in a test.
ENTITY_VOCABULARY: tuple[str, ...] = (
    "All India Financial Institutions",
    "Asset Reconstruction Companies",
    "Commercial Banks",
    "Cooperative Banks",
    "Credit Information Companies",
    "Housing Finance Companies",
    "Local Area Banks",
    "Non-Banking Financial Companies",
    "Payment Banks",
    "Payments Banks",
    "Primary (Urban) Co-operative Banks",
    "Regional Rural Banks",
    "Rural Co-operative Banks",
    "Small Finance Banks",
    "State Co-operative Banks",
    "Urban Co-operative Banks",
)

SALUTATION = re.compile(
    r"(?:Madam\s*/?\s*Dear Sir|Dear Sir\s*/?\s*Madam|Dear Madam|Madam|Dear Sir)\s*,",
    re.IGNORECASE,
)

#: The addressee block sits between the header and the salutation. Scanning only
#: that window is what keeps a passing body mention of "Payment Banks" out of the
#: filter values.
ADDRESSEE_WINDOW_CHARS = 800


def parse_regulated_entities(text: str) -> tuple[str, ...]:
    """Entities named in the addressee block, sorted and deduplicated.

    Returns an empty tuple when there is no header, no salutation, or no known
    entity in between. Never guesses: an entity filter is only useful if a
    document's absence from it is trustworthy.
    """
    header = HEADER.search(text)
    if header is None:
        return ()
    window = text[header.end() : header.end() + ADDRESSEE_WINDOW_CHARS]
    salutation = SALUTATION.search(window)
    if salutation is None:
        return ()
    block = window[: salutation.start()]
    found = {name for name in ENTITY_VOCABULARY if name.lower() in block.lower()}
    return tuple(sorted(found))
