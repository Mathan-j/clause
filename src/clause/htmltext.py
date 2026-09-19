import re

from selectolax.parser import HTMLParser

_WS = re.compile(r"\s+")


def visible_text(html: str) -> str:
    """Visible text with scripts, styles and navigation removed.

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
