"""A support signal, which is a proxy for faithfulness and is not faithfulness.

It measures what fraction of a sentence's content terms appear in the spans it
cites. That catches the gross failure -- a sentence citing a passage it shares no
vocabulary with -- and misses the subtle one, a fluent paraphrase that reverses
the meaning. Real entailment needs a judge model, which this project excludes on
cost, and a 3B model grading its own output would be worse than no metric.

It is reported as a distribution and never as a single "faithfulness score",
specifically so it cannot be quoted as one.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from clause.answering.schema import Citation, Sentence

_WORD = re.compile(r"[A-Za-z0-9]+")

_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
        "has", "have", "in", "is", "it", "its", "of", "on", "or", "shall",
        "that", "the", "their", "to", "was", "were", "will", "with", "within",
        "must", "may", "any", "such", "other", "than", "these", "this",
        "those", "under", "upon",
    ]
)


def content_terms(text: str) -> frozenset[str]:
    """Lower-cased tokens that carry content: numbers and non-stopword words.

    A number is kept regardless of length -- "7" in "7 days" is exactly the
    kind of content term this signal exists to catch. A word is dropped only
    if it is a stopword or a single letter, which numbers are exempt from.
    """
    terms: set[str] = set()
    for token in _WORD.findall(text):
        lowered = token.lower()
        if lowered.isdigit() or (len(lowered) > 1 and lowered not in _STOPWORDS):
            terms.add(lowered)
    return frozenset(terms)


def support_signal(sentence: Sentence, citations: Sequence[Citation]) -> float:
    """Fraction of the sentence's content terms present in the spans it cites.

    A non-factual sentence scores 1.0: it asserts nothing, so there is nothing
    for a citation to support.
    """
    if not sentence.factual:
        return 1.0
    terms = content_terms(sentence.text)
    if not terms:
        return 1.0
    cited: set[str] = set()
    for i in sentence.citation_indices:
        if i <= len(citations):
            cited |= content_terms(citations[i - 1].text)
    return len(terms & cited) / len(terms)


_HALF = 0.5


@dataclass(frozen=True, slots=True)
class SupportDistribution:
    n: int
    mean: float
    at_or_above_half: int
    zero: int


def summarise_support(values: Sequence[float]) -> SupportDistribution:
    if not values:
        return SupportDistribution(n=0, mean=0.0, at_or_above_half=0, zero=0)
    return SupportDistribution(
        n=len(values),
        mean=sum(values) / len(values),
        at_or_above_half=sum(1 for v in values if v >= _HALF),
        zero=sum(1 for v in values if v == 0.0),
    )
