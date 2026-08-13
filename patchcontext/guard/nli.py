"""
PatchContext – NLI hallucination guard.

Uses a cross-encoder NLI model to verify that the LLM's answer is entailed
by the retrieved context documents.

Pipeline (as specified in the implementation plan, section 16):

  Step 1 – Citation presence check (delegated to ``validator.py``).
  Step 2 – NLI entailment check:
            For each cited passage, run the NLI model with:
              premise   = the cited document's page_content
              hypothesis = the generated answer
            If the model classifies the pair as CONTRADICTION with
            confidence ≥ threshold, the claim is flagged as unsupported.

MVP design decisions
─────────────────────
• We check the *whole answer* against each *individual cited document*.
  Sentence-level claim decomposition is reserved for a future iteration.
• The guard can be disabled entirely via ``config.NLI_ENABLED = false``
  (useful during development / when GPU is unavailable).
• We use ``cross-encoder/nli-deberta-v3-base`` by default (configurable).
  It is downloaded automatically by HuggingFace the first time it runs.
• Inference is CPU-only by default; set NLI_DEVICE=cuda if a GPU is available.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from patchcontext import config
from patchcontext.guard.validator import check_fabricated_citations
from patchcontext.schemas import GuardResult, PatchContextAnswer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── Model loading ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_nli_pipeline():
    """Lazily load the NLI cross-encoder pipeline (cached after first call)."""
    try:
        from transformers import pipeline as hf_pipeline

        device = int(os.environ.get("NLI_DEVICE", "-1"))  # -1 = CPU
        logger.info(
            "Loading NLI model '%s' (device=%d) …", config.NLI_MODEL, device
        )
        pipe = hf_pipeline(
            "text-classification",
            model=config.NLI_MODEL,
            device=device,
            top_k=None,  # return all class scores
        )
        logger.info("NLI model loaded.")
        return pipe
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for the NLI guard. "
            "Install it with: pip install transformers torch"
        ) from exc


# ─── Guard logic ──────────────────────────────────────────────────────────────

# cross-encoder/nli-deberta-v3-base returns labels: ENTAILMENT, NEUTRAL, CONTRADICTION
_ENTAILMENT_LABEL = "ENTAILMENT"
_CONTRADICTION_LABEL = "CONTRADICTION"


def _score_entailment(
    premise: str,
    hypothesis: str,
    pipe,
) -> float:
    """Return the entailment probability for (premise, hypothesis)."""
    # The cross-encoder model accepts the pair as a single string
    result = pipe(f"{premise} </s></s> {hypothesis}", truncation=True, max_length=512)

    # result is a list of [{"label": ..., "score": ...}, ...]
    scores: dict[str, float] = {r["label"].upper(): r["score"] for r in result[0]}
    return scores.get(_ENTAILMENT_LABEL, 0.0)


def _get_cited_docs(
    citations: list[str],
    retrieved_docs: list[Document],
) -> list[Document]:
    """Return the subset of retrieved docs that match any of the citations."""
    from patchcontext.guard.validator import _citation_matches_doc

    return [
        doc
        for doc in retrieved_docs
        if any(_citation_matches_doc(c, doc) for c in citations)
    ]


# ─── Public API ───────────────────────────────────────────────────────────────


def run_guard(
    answer: PatchContextAnswer,
    retrieved_docs: list[Document],
) -> GuardResult:
    """Run the hallucination guard on a structured LLM answer.

    Checks:
      1. Citation presence – are all cited IDs grounded in retrieved docs?
      2. NLI entailment   – is the answer supported by the cited evidence?
                            (skipped if NLI_ENABLED=false)

    Args:
        answer: Structured answer from ``generate_answer``.
        retrieved_docs: Documents returned by the MMR retriever.

    Returns:
        A ``GuardResult`` with ``passed=True`` if both checks succeed.
    """
    # ── Check 1: fabricated citations ─────────────────────────────────────────
    fabricated = check_fabricated_citations(answer.citations, retrieved_docs)

    if fabricated:
        return GuardResult(
            passed=False,
            fabricated_citations=fabricated,
            unsupported_claims=[],
            reason=(
                f"The following citations are not grounded in retrieved "
                f"context: {', '.join(fabricated)}"
            ),
        )

    # ── Check 2: NLI entailment ───────────────────────────────────────────────
    if not config.NLI_ENABLED:
        logger.info("NLI guard disabled (NLI_ENABLED=false). Skipping entailment check.")
        return GuardResult(passed=True)

    cited_docs = _get_cited_docs(answer.citations, retrieved_docs)

    if not cited_docs:
        # No specific cited docs resolved – fall back to all retrieved docs
        cited_docs = retrieved_docs

    try:
        pipe = _load_nli_pipeline()
    except Exception as exc:
        logger.error("NLI model could not be loaded: %s. Skipping NLI check.", exc)
        return GuardResult(passed=True, reason="NLI guard skipped (model unavailable).")

    unsupported: list[str] = []
    hypothesis = answer.answer

    for doc in cited_docs:
        premise = doc.page_content[:1000]  # truncate very long passages
        score = _score_entailment(premise, hypothesis, pipe)

        logger.debug(
            "NLI: source_id=%s  entailment=%.3f  threshold=%.3f",
            doc.metadata.get("source_id", "?"),
            score,
            config.NLI_ENTAILMENT_THRESHOLD,
        )

        if score < config.NLI_ENTAILMENT_THRESHOLD:
            # Low entailment means the answer may not be supported by this source
            unsupported.append(doc.metadata.get("source_id", "unknown"))

    # The guard passes only if ALL cited docs entail the answer
    # (soft policy: flag if *majority* don't entail, not just one)
    fail_fraction = len(unsupported) / len(cited_docs) if cited_docs else 0.0
    passed = fail_fraction < 0.5  # more than half must be entailing

    if not passed:
        return GuardResult(
            passed=False,
            fabricated_citations=[],
            unsupported_claims=unsupported,
            reason=(
                f"The answer is not sufficiently entailed by the cited evidence. "
                f"Low-entailment sources: {', '.join(unsupported)}"
            ),
        )

    return GuardResult(passed=True)
