"""
PatchContext – Benchmark runner.

Loads the benchmark JSON file and runs each question through the full
PatchContext pipeline (retrieval → generation → guard), collecting the
data needed for RAGAS evaluation.

Output record format (matches RAGAS expected input):
    {
        "question"          : str,
        "ground_truth"      : str,
        "answer"            : str,          ← generated
        "contexts"          : list[str],    ← retrieved page_content strings
        "ground_truth_sources": list[str],  ← from benchmark file
        "generated_citations" : list[str],  ← from LLM
        "guard_passed"        : bool,
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.vectorstores import VectorStoreRetriever
from tqdm import tqdm

from patchcontext import config
from patchcontext.guard.nli import run_guard
from patchcontext.rag.generator import generate_answer

logger = logging.getLogger(__name__)


def load_benchmark(benchmark_file: Path | None = None) -> list[dict[str, Any]]:
    """Load the benchmark question set from JSON.

    Expected format per entry:
        {
            "question"             : str,
            "ground_truth"         : str,
            "ground_truth_sources" : list[str]   (e.g. ["pr:1234"])
        }
    """
    path = benchmark_file or config.BENCHMARK_FILE
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d benchmark questions from %s", len(data), path)
    return data


def run_benchmark(
    retriever: VectorStoreRetriever,
    benchmark_file: Path | None = None,
    max_questions: int | None = None,
    skip_guard: bool = False,
) -> list[dict[str, Any]]:
    """Run the full PatchContext pipeline over benchmark questions.

    Args:
        retriever: The configured LangChain MMR retriever.
        benchmark_file: Path to the benchmark JSON. Defaults to config.
        max_questions: If set, only run the first N questions (useful for quick smoke tests).
        skip_guard: Skip the NLI guard for faster evaluation runs.

    Returns:
        A list of result records in RAGAS-compatible format.
    """
    questions = load_benchmark(benchmark_file)

    if max_questions is not None:
        questions = questions[:max_questions]

    results: list[dict[str, Any]] = []

    for item in tqdm(questions, desc="Benchmarking"):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        ground_truth_sources = item.get("ground_truth_sources", [])

        try:
            answer_obj, retrieved_docs = generate_answer(question, retriever)

            guard_passed = True
            if not skip_guard:
                guard_result = run_guard(answer_obj, retrieved_docs)
                guard_passed = guard_result.passed

            results.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": answer_obj.answer,
                    # RAGAS expects contexts as a list of strings
                    "contexts": [doc.page_content for doc in retrieved_docs],
                    "ground_truth_sources": ground_truth_sources,
                    "generated_citations": answer_obj.citations,
                    "guard_passed": guard_passed,
                }
            )
        except Exception as exc:
            logger.error("Question failed: %s — %s", question[:60], exc)
            results.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": "",
                    "contexts": [],
                    "ground_truth_sources": ground_truth_sources,
                    "generated_citations": [],
                    "guard_passed": False,
                    "error": str(exc),
                }
            )

    logger.info(
        "Benchmark complete: %d/%d questions succeeded.",
        sum(1 for r in results if r.get("answer")),
        len(results),
    )
    return results


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Persist benchmark results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Benchmark results saved to %s", output_path)
