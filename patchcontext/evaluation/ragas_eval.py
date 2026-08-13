"""
PatchContext – RAGAS evaluation runner.

Takes a list of benchmark result records (as produced by ``benchmark.py``)
and evaluates them using the RAGAS framework.

Metrics evaluated
──────────────────
  faithfulness            – Is the answer grounded in the retrieved context?
  answer_relevancy        – Is the answer relevant to the question?
  context_precision       – Are the retrieved chunks high-signal?
  context_recall          – Did we retrieve all the needed information?

RAGAS requires an LLM (gpt-4o-mini) and embeddings for its own internals,
so we reuse the same LangChain components already configured in PatchContext.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset

from patchcontext import config
from patchcontext.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _build_ragas_dataset(results: list[dict[str, Any]]) -> Dataset:
    """Convert benchmark results into a HuggingFace Dataset for RAGAS.

    RAGAS expects columns: question, answer, contexts, ground_truth.
    """
    rows = []
    for r in results:
        if not r.get("answer"):  # skip failed questions
            continue
        rows.append(
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],  # list[str]
                "ground_truth": r["ground_truth"],
            }
        )

    if not rows:
        raise ValueError("No successful answers to evaluate.")

    return Dataset.from_list(rows)


def run_ragas_evaluation(
    results: list[dict[str, Any]],
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Run RAGAS evaluation on benchmark results and return a metrics DataFrame.

    Args:
        results: Output from ``benchmark.run_benchmark``.
        output_csv: If provided, save the per-question metrics to this CSV file.

    Returns:
        A ``pd.DataFrame`` with per-question RAGAS scores, plus a summary row.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "RAGAS evaluation requires: pip install ragas datasets"
        ) from exc

    dataset = _build_ragas_dataset(results)
    logger.info("Running RAGAS evaluation on %d questions …", len(dataset))

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    )
    embeddings = get_embeddings()

    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    df: pd.DataFrame = ragas_result.to_pandas()

    # Print summary
    summary_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available_cols = [c for c in summary_cols if c in df.columns]
    if available_cols:
        means = df[available_cols].mean()
        logger.info("─── RAGAS Summary ───")
        for col, val in means.items():
            logger.info("  %-25s %.4f", col, val)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info("RAGAS results saved to %s", output_csv)

    return df


def print_ragas_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary table of RAGAS scores."""
    metric_cols = [
        c for c in df.columns
        if c in {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    ]
    if not metric_cols:
        print("No RAGAS metric columns found in DataFrame.")
        return

    print("\n" + "═" * 50)
    print("RAGAS Evaluation Summary")
    print("═" * 50)
    for col in metric_cols:
        mean_val = df[col].mean()
        print(f"  {col:<30} {mean_val:.4f}")
    print(f"  {'Total questions':<30} {len(df)}")
    print("═" * 50 + "\n")
