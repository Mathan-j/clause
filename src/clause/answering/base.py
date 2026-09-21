"""The generator interface, and a deterministic implementation of it.

`StubAnswerer` exists so the whole answering pipeline -- resolution, the citation
contract, refusal, the metrics -- is exercisable with no model present. That
matters because CI never installs llama-cpp-python: without a stub, the parts of
this phase that enforce CLAUDE.md's hardest rule would be untested in the only
environment that runs on every push.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from clause.answering.schema import AnswerDraft, Sentence
from clause.retrieve import Hit


@runtime_checkable
class Answerer(Protocol):
    def answer(self, question: str, hits: Sequence[Hit]) -> AnswerDraft: ...


class StubAnswerer:
    """Quotes the top hit verbatim and cites it. Deterministic by construction."""

    def answer(self, question: str, hits: Sequence[Hit]) -> AnswerDraft:
        hit_tuple = tuple(hits)
        if not hit_tuple:
            return AnswerDraft(
                question=question,
                sentences=(
                    Sentence(
                        text="No candidate passages were retrieved.",
                        citation_indices=(),
                        factual=False,
                    ),
                ),
                hits=(),
            )
        return AnswerDraft(
            question=question,
            sentences=(
                Sentence(
                    text=hit_tuple[0].text.strip(),
                    citation_indices=(1,),
                    factual=True,
                ),
            ),
            hits=hit_tuple,
        )
