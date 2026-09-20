import re

from selectolax.parser import HTMLParser

_WS = re.compile(r"\s+")

_STRIPPED_TAGS = ("script", "style", "noscript")

#: RBI renders the notification itself inside this element; everything outside it
#: is site template. Present in all 61 documents of the committed corpus, and always
#: containing the circular header line, so it is a reliable anchor rather than a
#: guess. `document_text` falls back to the whole body when it is absent.
CONTENT_SELECTOR = "#NotificationUser"


def _parse(html: str) -> HTMLParser:
    tree = HTMLParser(html)
    for tag in _STRIPPED_TAGS:
        for node in tree.css(tag):
            node.decompose()
    return tree


def _collapse(raw: str) -> str:
    return _WS.sub(" ", raw).strip()


def visible_text(html: str) -> str:
    """All visible page text, with `<script>`, `<style>` and `<noscript>` removed.

    This is the whole page, site template included. It answers "how much text did
    this response contain", which is what the validation gate needs when judging an
    untrusted response that may not be a document at all. Use `document_text` for
    the document itself.
    """
    tree = _parse(html)
    body = tree.body or tree.root
    if body is None:
        return ""
    return _collapse(body.text(separator=" "))


def document_text(html: str) -> str:
    """Visible text of the notification itself, with the site template excluded.

    Selects `CONTENT_SELECTOR` and returns only what is inside it. Measured over
    the committed corpus, the surrounding template — the mega-menu before the
    document and the footer block after it — accounted for more than half of every
    stored document's characters, so a chunk drawn from the old whole-page text was
    as likely to be site navigation as regulation.

    Falls back to `visible_text` when the element is absent, which degrades to the
    previous behaviour rather than returning nothing: an empty document would fail
    extraction's length check and silently drop a real circular from the corpus.
    """
    tree = _parse(html)
    node = tree.css_first(CONTENT_SELECTOR)
    if node is None:
        body = tree.body or tree.root
        return "" if body is None else _collapse(body.text(separator=" "))
    return _collapse(node.text(separator=" "))
