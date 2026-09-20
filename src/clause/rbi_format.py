"""Regex constants describing RBI's own circular text format.

Three modules each need to find the same structural landmarks in an RBI
circular's extracted text -- the header line, and the salutation that closes
the addressee block -- for different reasons:

- `clause.ingest.extract` anchors header parsing and doc-type classification
  on `HEADER`.
- `clause.entities` scans the window between `HEADER` and `SALUTATION` for the
  addressee block.
- `clause.sources.discover` skips past the same window to approximate a
  circular's subject line for the manifest.

These are RBI document-format constants, not ingestion logic or domain
parsing -- they belong to neither of the modules that need them, so they live
here instead. That keeps the dependency graph a plain DAG (all three import
from here; none of them import each other for this), rather than the module
that happened to define `HEADER` first owning it and the others reaching
into that module's internals.
"""

import re

#: `circular_no`, `dept_ref`, `published` -- the identity line every RBI
#: circular / master direction / notification opens with.
HEADER = re.compile(
    r"(?P<circular_no>RBI/\d{4}-\d{2}/\d+)\s+"
    r"(?P<dept_ref>[A-Z]{2,}(?:\.[A-Z0-9]+)+[A-Z0-9./-]*)\s+"
    r"(?P<published>[A-Z][a-z]+ \d{1,2}, \d{4})"
)

# Many RBI circulars address a salutation to regulated entities between the header
# line and the actual subject line, e.g.:
#   "...September 18, 2026  The Chairpersons/ CEOs of ... All India Financial
#   Institutions  Madam/Dear Sir,  Implementation of Section 51A of UAPA, 1967: ..."
# Spellings observed in the corpus: "Madam/Dear Sir,", "Madam/ Dear Sir,",
# "Dear Sir/Madam,", "Dear Sir / Madam,", "Dear Sir/ Madam,", "Dear Madam,",
# "Madam,", "Dear Sir," -- this pattern is not guaranteed exhaustive against
# spellings RBI has not yet used.
SALUTATION = re.compile(
    r"(?:Madam\s*/?\s*Dear Sir|Dear Sir\s*/?\s*Madam|Dear Madam|Madam|Dear Sir)\s*,",
    re.IGNORECASE,
)
