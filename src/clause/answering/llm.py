"""Local GGUF generation through llama.cpp, constrained by a GBNF grammar.

The model never sees or supplies citation text -- only an index into the hits it
was shown. It also never downloads: an absent model file raises with the path it
expected, the same rule the embedding encoder follows, because a first run that
fetches weights has a different reproducibility story from every run after it.
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# LlamaGrammar is imported from its defining submodule, not the top-level `llama_cpp`
# package, deliberately -- do not "simplify" this back. `llama_cpp/__init__.py` does
# `from .llama import *`, and `llama.py` itself only *imports* LlamaGrammar from
# `llama_grammar.py` (no `__all__`, no `as` re-export). mypy strict's
# `no_implicit_reexport` therefore does not recognise `llama_cpp.LlamaGrammar` as a
# valid public re-export, even though it exists at runtime. `Llama` needs no such
# workaround because it is defined directly in `llama.py`, not imported into it.
from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar

from clause.answering.grammar import build_grammar
from clause.answering.schema import (
    MAX_CITATIONS_PER_SENTENCE,
    AnswerDraft,
    ModelNotAvailableError,
    Sentence,
)
from clause.retrieve import Hit

DEFAULT_MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"

#: Generation budget. Observed answers run 76-250 tokens; 512 was headroom
#: nothing used, and an unused budget still costs nothing until the model
#: reaches for it -- but a runaway under grammar constraint spends all of
#: it, which is exactly the whitespace-attractor failure this grammar was
#: bounded to prevent. 320 covers the observed range with room to spare.
MAX_ANSWER_TOKENS = 320


def _physical_cores() -> int:
    """Physical cores, falling back to half the logical count.

    llama.cpp gains nothing from SMT siblings on this workload and loses a
    little to contention, so the useful thread count is the physical core
    count rather than `os.cpu_count()`.
    """
    logical = os.cpu_count() or 4
    return max(1, logical // 2)


_SYSTEM = (
    "You answer questions about Indian banking regulation using ONLY the numbered "
    "passages provided. Every sentence stating a fact must cite the passage numbers "
    "it came from. If the passages do not answer the question, say so in one "
    "sentence and cite nothing. Never cite a passage number you were not given."
)


def _prompt(question: str, hits: Sequence[Hit]) -> str:
    passages = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, start=1))
    return (
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\nPassages:\n{passages}\n\nQuestion: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class LlamaAnswerer:
    """`Answerer` backed by a local GGUF model, grammar-constrained at decode time.

    `temperature=0.0` and a fixed `seed` make generation deterministic: a
    harness whose generator moves between runs cannot be trusted to reproduce
    the numbers it reports.
    """

    def __init__(self, model_path: Path, n_ctx: int = 4096) -> None:
        if not model_path.exists():
            raise ModelNotAvailableError(
                f"answering model is not present at {model_path}, and this process "
                f"will not download it. Fetch {DEFAULT_MODEL_FILENAME} into that path "
                "first; see README's 'Running the answering evaluation'."
            )
        self._model_path = model_path
        self._llama = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            verbose=False,
            seed=0,
            # Prompt processing, not token generation, dominates the wall clock
            # here: a five-passage prompt is a few thousand tokens and each one
            # must be run through the model before the first output token
            # appears. llama.cpp's defaults leave cores idle on this 6-core
            # part, and prefill parallelises across them almost linearly.
            n_threads=_physical_cores(),
            n_batch=512,
        )

    def answer(self, question: str, hits: Sequence[Hit]) -> AnswerDraft:
        hit_tuple = tuple(hits)
        grammar = LlamaGrammar.from_string(build_grammar(len(hit_tuple)), verbose=False)
        result: Any = self._llama(
            _prompt(question, hit_tuple),
            grammar=grammar,
            max_tokens=MAX_ANSWER_TOKENS,
            temperature=0.0,
            seed=0,
        )
        payload = json.loads(result["choices"][0]["text"])
        sentences: list[Sentence] = []
        for raw in payload["sentences"]:
            indices = tuple(int(i) for i in raw["citations"])[:MAX_CITATIONS_PER_SENTENCE]
            sentences.append(
                Sentence(
                    text=raw["text"],
                    citation_indices=indices,
                    factual=bool(indices),
                )
            )
        return AnswerDraft(question=question, sentences=tuple(sentences), hits=hit_tuple)
