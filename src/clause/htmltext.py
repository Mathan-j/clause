import re

from selectolax.parser import HTMLParser

_WS = re.compile(r"\s+")


def visible_text(html: str) -> str:
    """Visible text with `<script>`, `<style>` and `<noscript>` elements removed.

    Page chrome — navigation, breadcrumbs, skip links and the rest of the site
    template — is NOT removed: only those three tag types are stripped. Every
    document's canonical text currently begins with RBI's site navigation (e.g.
    "Skip to main content Not Pressed Not Pressed ..."), and site chrome accounts
    for a large share of every stored document's characters. Stripping navigation
    is planned but not yet implemented; see
    `docs/superpowers/specs/2026-09-19-ingestion-and-storage-design.md` section 5.2.

    Entities are decoded and whitespace runs collapsed to single spaces. Used by
    both the validation gate and the extractor so they agree on what 'the text'
    means.
    """
    tree = HTMLParser(html)
    for tag in ("script", "style", "noscript"):
        for node in tree.css(tag):
            node.decompose()
    body = tree.body or tree.root
    if body is None:
        return ""
    return _WS.sub(" ", body.text(separator=" ")).strip()
