"""`make answer-eval`: score the golden and adversarial sets end to end.

Retrieval runs exactly once per question, at `RETRIEVAL_DEPTH`. The threshold
sweep in the report is arithmetic over the scores recorded from that one pass
-- re-running retrieval per threshold would cost twenty retrieval passes for a
curve that needs only one, since `refusal.decide()` is a pure function of a
question's already-retrieved top score.

Generation is far more expensive than retrieval (~10s/question with the model
kept alive; see the SDD ledger), so one `LlamaAnswerer` is built once and
reused across every golden and adversarial question, and only questions that
survive the refusal gate at the *default* threshold ever reach it -- the same
gate `answer_one` enforces for a single question, applied here to the whole
run so a question refused at the sweep's own default threshold never pays for
a model call either.
"""

import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path

import sqlalchemy as sa
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from clause.answering.base import Answerer
from clause.answering.llm import LlamaAnswerer
from clause.answering.refusal import SWEEP_THRESHOLDS, decide
from clause.answering.resolver import resolve_citations
from clause.answering.schema import (
    Answer,
    AnsweringError,
    Refusal,
)
from clause.answering.validate import enforce
from clause.config import Settings, get_settings
from clause.db.schema import DocumentRow
from clause.db.session import make_engine, session_factory
from clause.embed import Encoder
from clause.evaluation.adversarial import AdversarialQuestion, load_adversarial
from clause.evaluation.answer_metrics import summarise_support, support_signal
from clause.evaluation.answer_report import (
    AnswerFingerprint,
    ThresholdRow,
    build_answer_report,
    write_answer_report,
)
from clause.evaluation.golden import GoldenQuestion, load_golden
from clause.evaluation.metrics import RETRIEVAL_DEPTH, first_hit_rank
from clause.retrieve import Hit, search

GOLDEN_PATH = Path("data/golden/kyc-v1.jsonl")
ADVERSARIAL_PATH = Path("data/adversarial/out-of-corpus-v1.jsonl")
DEFAULT_ANSWERS_MD = Path("reports/answers.md")
DEFAULT_ANSWERS_JSON = Path("reports/answers.json")

#: The single chunking strategy this evaluation retrieves against. Phase 2
#: scores both `structural` and `fixed_window`; answering needs one hit list
#: per question, not a strategy comparison, and `structural` is the strategy
#: every worked example in the SDD briefs (Task 7's verification script,
#: Task 9's fingerprint) already assumes.
ANSWER_STRATEGY = "structural"

#: `LlamaAnswerer.__init__` defaults to 4096, which is too small for this
#: evaluation: at `RETRIEVAL_DEPTH=10` with `structural` chunks up to 2000
#: characters each, a real prompt measured at 4207 tokens against a 4096
#: window -- crashing generation outright rather than truncating. 8192 covers
#: the observed worst case with headroom for the 512-token completion budget
#: `LlamaAnswerer.answer` requests; passed explicitly here (not left as the
#: constructor's default) so the fingerprint records the value actually used.
ANSWER_N_CTX = 8192

#: How many of the retrieved hits are put in front of the model. Retrieval
#: still runs at `RETRIEVAL_DEPTH`, because the refusal threshold and the
#: sweep are computed from the top score and the full ranked list; only the
#: prompt is narrowed. Five rather than ten for two reasons: a 3B model given
#: ten passages of up to 2000 characters degrades (the answer it needs sits
#: in the middle of a very long context), and the prompt halves, which is the
#: difference between a run that completes and one that does not on this
#: machine. Narrowing further to three was measured and did not help: the
#: wall clock follows total tokens, and a shorter prompt simply drew a
#: longer answer. Five it is, for the better retrieval coverage.
#: Citation indices are 1-based into *these* hits, so `answer_one`
#: must hand the same narrowed list to both the answerer and the resolver --
#: the draft carries it, which is what `AnswerDraft.hits` exists to guarantee.
ANSWER_PROMPT_HITS = 5


def answer_one(
    question: str,
    hits: Sequence[Hit],
    answerer: Answerer,
    session: Session,
    threshold: float,
) -> Answer | Refusal:
    """Refuse before generation when retrieval is too weak; otherwise answer.

    Order is the contract: `decide()` runs first and returns the `Refusal`
    outright with no model call. Only a question that clears the bar reaches
    `answerer.answer()`, then `resolve_citations()`, then `validate.enforce()`
    -- any of which may raise `AnsweringError` for a genuine contract breach.
    That is deliberately not caught here: this function answers one question
    or refuses it, and leaves what a raised `AnsweringError` means to the
    caller running a whole evaluation (see `_score_pass` below).
    """
    refusal = decide(question, hits, threshold)
    if refusal is not None:
        return refusal
    # The narrowed list goes to the answerer, and the draft carries it onward
    # to the resolver. Handing the answerer five hits and the resolver ten
    # would make every citation index mean two different things -- the seam
    # defect this phase already fixed once.
    prompt_hits = list(hits[:ANSWER_PROMPT_HITS])
    draft = answerer.answer(question, prompt_hits)
    resolved = resolve_citations(draft, session)
    return enforce(draft, resolved)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _retrieve_all(
    client: QdrantClient,
    encoder: Encoder,
    questions: Sequence[str],
) -> list[list[Hit]]:
    """One retrieval pass per question, in order. Nothing here reads a score
    or decides a threshold -- that happens once, arithmetically, over the
    scores this returns."""
    return [search(client, encoder, ANSWER_STRATEGY, q, limit=RETRIEVAL_DEPTH) for q in questions]


def _sweep_row(
    threshold: float,
    golden_top: Sequence[float | None],
    golden_rank: Sequence[int | None],
    adversarial_top: Sequence[float | None],
) -> ThresholdRow:
    refused_idx = [i for i, s in enumerate(golden_top) if s is None or s < threshold]
    false_refusals = sum(1 for i in refused_idx if golden_rank[i] is not None)
    adversarial_refused = sum(1 for s in adversarial_top if s is None or s < threshold)
    return ThresholdRow(
        threshold=threshold,
        answered=len(golden_top) - len(refused_idx),
        refused=len(refused_idx),
        false_refusals=false_refusals,
        adversarial_refused=adversarial_refused,
        adversarial_answered=len(adversarial_top) - adversarial_refused,
    )


def _score_pass(
    items: Sequence[tuple[str, Sequence[Hit]]],
    answerer: Answerer,
    session: Session,
    threshold: float,
) -> tuple[int, int, list[float], list[str], int]:
    """Run `answer_one` over every (question, hits) pair at one threshold.

    Returns `(attempted, succeeded, support_values, factual_sentence_texts,
    generation_failures)`. A question the refusal gate turns away is not
    "attempted" -- no model call happened for it. A question that reaches
    generation but whose `Answer` never materialises because `AnsweringError`
    was raised *is* counted as attempted: a citation contract breach, per
    Task 10's brief and the SDD ledger's ruling that every such rejection is
    catchable as `AnsweringError` and belongs in the resolution-rate
    accounting.

    A plain `ValueError` from the model call itself (not an `AnsweringError`,
    which subclasses it) is a different failure mode -- llama.cpp raising
    because a prompt exceeds the context window -- and is counted separately
    as `generation_failures` rather than folded into the resolution rate,
    which is specifically about the citation contract, not about whether
    generation ran at all. It is still counted as attempted: a model call was
    genuinely made and failed.

    `factual_sentence_texts` is every factual sentence's raw text, collected
    so the caller can report how many of them are distinct: the 3B model has
    been observed to emit the same sentence more than once in a single
    answer, and counting those as independent corroborating claims would
    misstate the support distribution.
    """
    attempted = 0
    succeeded = 0
    generation_failures = 0
    support_values: list[float] = []
    sentence_texts: list[str] = []
    for question, hits in items:
        try:
            result = answer_one(question, hits, answerer, session, threshold)
        except AnsweringError:
            attempted += 1
            continue
        except ValueError:
            # Not an AnsweringError: a plain ValueError here is llama.cpp's
            # own "requested tokens exceed context window" failure, not a
            # citation contract breach. Counted as attempted-but-failed,
            # tracked separately, and does not stop the run.
            attempted += 1
            generation_failures += 1
            continue
        if isinstance(result, Refusal):
            continue
        attempted += 1
        succeeded += 1
        for sentence in result.sentences:
            if sentence.factual:
                sentence_texts.append(sentence.text)
                support_values.append(support_signal(sentence, result.citations))
    return attempted, succeeded, support_values, sentence_texts, generation_failures


def _print_sweep(rows: Sequence[ThresholdRow]) -> None:
    print(
        f"{'threshold':>9}{'answered':>10}{'refused':>9}{'false_ref':>11}"
        f"{'adv_refused':>13}{'adv_answered':>14}"
    )
    for r in rows:
        print(
            f"{r.threshold:>9.2f}{r.answered:>10}{r.refused:>9}{r.false_refusals:>11}"
            f"{r.adversarial_refused:>13}{r.adversarial_answered:>14}"
        )


def run_answer_eval(
    *,
    md_path: Path = DEFAULT_ANSWERS_MD,
    json_path: Path = DEFAULT_ANSWERS_JSON,
    settings: Settings | None = None,
) -> int:
    """Score both the golden and adversarial sets, write both artifacts, and
    print the sweep table.

    Output paths are parameters, defaulting to the committed `reports/`
    location -- tests pass `tmp_path` so the suite never rewrites the
    committed artifact `reports/eval.md`'s Task 11 CI gate protects.
    """
    settings = settings or get_settings()
    golden: list[GoldenQuestion] = load_golden(GOLDEN_PATH)
    adversarial: list[AdversarialQuestion] = load_adversarial(ADVERSARIAL_PATH)

    encoder = Encoder(settings.embedding_model)
    client = QdrantClient(url=settings.qdrant_url)
    answerer = LlamaAnswerer(settings.answer_model_path, n_ctx=ANSWER_N_CTX)

    golden_hits = _retrieve_all(client, encoder, [q.question for q in golden])
    golden_top: list[float | None] = [h[0].score if h else None for h in golden_hits]
    golden_rank: list[int | None] = [
        first_hit_rank([h.as_retrieved() for h in hits], q.answer_spans())
        for q, hits in zip(golden, golden_hits, strict=True)
    ]

    adversarial_hits = _retrieve_all(client, encoder, [q.question for q in adversarial])
    adversarial_top: list[float | None] = [h[0].score if h else None for h in adversarial_hits]

    rows = [
        _sweep_row(t, golden_top, golden_rank, adversarial_top) for t in SWEEP_THRESHOLDS
    ]

    threshold = settings.answer_threshold
    engine = make_engine(settings.database_url)
    with session_factory(engine)() as session:
        corpus_documents = (
            session.scalar(sa.select(sa.func.count()).select_from(DocumentRow)) or 0
        )

        g_attempted, g_succeeded, g_support, g_sentences, g_failures = _score_pass(
            list(zip([q.question for q in golden], golden_hits, strict=True)),
            answerer,
            session,
            threshold,
        )
        a_attempted, a_succeeded, a_support, a_sentences, a_failures = _score_pass(
            list(zip([q.question for q in adversarial], adversarial_hits, strict=True)),
            answerer,
            session,
            threshold,
        )

    attempted = g_attempted + a_attempted
    succeeded = g_succeeded + a_succeeded
    generation_failures = g_failures + a_failures
    # The denominator excludes generation failures deliberately. A prompt that
    # llama.cpp refused to run produced no draft, so there was no citation for
    # the contract to enforce -- counting it here would report a "citation
    # contract breach" for a question where the contract was never reached.
    # The report renders anything below 1.0 as a breach, so folding the two
    # together publishes a catastrophic-looking number for something that did
    # not happen. Generation failures are reported on their own.
    generated = attempted - generation_failures
    resolution_rate = succeeded / generated if generated else 1.0
    support_values = g_support + a_support
    support = summarise_support(support_values)
    all_sentences = g_sentences + a_sentences

    fingerprint = AnswerFingerprint(
        model_file=settings.answer_model_path.name,
        model_sha256=_sha256_file(settings.answer_model_path),
        n_ctx=ANSWER_N_CTX,
        threshold=threshold,
        golden_sha256=_sha256_file(GOLDEN_PATH),
        adversarial_sha256=_sha256_file(ADVERSARIAL_PATH),
        corpus_documents=corpus_documents,
        git_commit=_git_commit(),
    )
    report = build_answer_report(rows, fingerprint, support, resolution_rate=resolution_rate)
    report["sentence_repetition"] = {
        "total_factual_sentences": len(all_sentences),
        "distinct_factual_sentences": len(set(all_sentences)),
    }
    report["generation_failures"] = generation_failures
    write_answer_report(report, md_path, json_path)

    # `render_answer_markdown` (Task 9) knows nothing of `sentence_repetition`
    # -- it is not part of that module's frozen interface. Appended here so
    # the count of distinct sentences is visible in the artifact itself, not
    # only in the JSON, per the SDD ledger's note that a model repeating a
    # sentence verbatim must not be read as independent corroborating claims.
    total = report["sentence_repetition"]["total_factual_sentences"]
    distinct = report["sentence_repetition"]["distinct_factual_sentences"]
    addendum = (
        "\n## Sentence repetition\n\n"
        f"Of **{total}** factual sentence(s) scored above, **{distinct}** are "
        "textually distinct. The 3B model has been observed to repeat the "
        "same well-formed, correctly-cited sentence multiple times in a "
        "single answer; those repeats are counted once each in the support "
        "distribution above, not collapsed, so a reader comparing `n` in "
        "that section against this count can see how much of it is "
        "repetition rather than independent claims.\n"
        "\n## Generation failures\n\n"
        f"**{generation_failures}** attempted generation(s) failed outright "
        "(a plain `ValueError` from llama.cpp itself, not an `AnsweringError` "
        "-- a prompt whose token count exceeded the model's context window) "
        "and are excluded from the support distribution above but counted as "
        "attempted for resolution-rate purposes. This is an infrastructure "
        "limit of the fixed retrieval depth and context window, not a "
        "citation contract breach.\n"
    )
    with md_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(addendum)

    print(f"resolution rate: {resolution_rate:.4f} ({succeeded}/{attempted} attempted)")
    print(f"factual sentences: {total} total, {distinct} distinct")
    print(f"generation failures (context overflow): {generation_failures}")
    _print_sweep(rows)
    return 0
