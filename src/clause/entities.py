"""Parse the regulated entities a circular is addressed to.

RBI names its audience explicitly between the header line and the salutation:

    The Chairpersons/ CEOs of the Commercial Banks, Small Finance Banks, ...
    Madam/Dear Sir,

That block is structured data the corpus already contains, and it is what makes
`PROMPT.md`'s "regulated entity" metadata filter real rather than aspirational.
"""

import re

from clause.rbi_format import HEADER, SALUTATION

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

#: The addressee block sits between the header and the salutation. Scanning only
#: that window is what keeps a passing body mention of "Payment Banks" out of the
#: filter values.
ADDRESSEE_WINDOW_CHARS = 800

#: RBI addresses roughly a quarter of this corpus's circulars to every regulated
#: entity class at once -- "The Chairpersons/ CEOs of all the Regulated Entities" --
#: rather than naming classes individually. Spacing varies in the corpus: "/ CEOs",
#: "/CEOs", " / CEOs". This is not a vocabulary gap to fill by guessing; it is the
#: document stating its own audience as universal. Recording it as an empty list
#: would assert the opposite of what the text says, and would make a per-entity
#: filter silently *exclude* a document that legitimately governs that entity --
#: worse than an undiscriminating filter, a wrong one. So this one case is expanded
#: to the full vocabulary by an explicit rule, not invented: the document said "all",
#: and "all" is read plainly, from `ENTITY_VOCABULARY` as it stands today.
_BLANKET_ADDRESSEE = re.compile(r"all\s+the\s+regulated\s+entities", re.IGNORECASE)


def parse_regulated_entities(text: str) -> tuple[str, ...]:
    """Entities named in the addressee block, sorted and deduplicated.

    Returns an empty tuple when there is no header, no salutation, or no known
    entity in between. Never guesses: an entity filter is only useful if a
    document's absence from it is trustworthy. The one exception is the blanket
    "all the Regulated Entities" addressee (see `_BLANKET_ADDRESSEE`), which is
    expanded to the full vocabulary because the document itself says "all" --
    that is honouring a stated audience, not inventing one.
    """
    header = HEADER.search(text)
    if header is None:
        return ()
    window = text[header.end() : header.end() + ADDRESSEE_WINDOW_CHARS]
    salutation = SALUTATION.search(window)
    if salutation is None:
        return ()
    block = window[: salutation.start()]
    if _BLANKET_ADDRESSEE.search(block):
        return tuple(sorted(ENTITY_VOCABULARY))
    found = {name for name in ENTITY_VOCABULARY if name.lower() in block.lower()}
    return tuple(sorted(found))
