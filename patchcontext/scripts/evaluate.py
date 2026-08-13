"""
PatchContext – RAGAS evaluation runner script.

Loads the FAISS index, runs all benchmark questions through the pipeline,
then evaluates with RAGAS and prints a summary.

Usage:
    python -m patchcontext.scripts.evaluate [--questions N] [--skip-guard]
                                             [--output PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from patchcontext import config
from patchcontext.evaluation.benchmark import run_benchmark, save_results
from patchcontext.evaluation.ragas_eval import print_ragas_summary, run_ragas_evaluation
from patchcontext.rag.retriever import get_retriever
from patchcontext.rag.vectorstore import load_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PatchContext – RAGAS evaluation")
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Number of benchmark questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--skip-guard",
        action="store_true",
        default=False,
        help="Skip the NLI guard during evaluation (faster)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "eval_results.json",
        help="Path to save raw benchmark results JSON",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_ROOT / "data" / "ragas_scores.csv",
        help="Path to save per-question RAGAS scores CSV",
    )
    args = parser.parse_args()

    # Load index
    logger.info("Loading FAISS index …")
    try:
        vectorstore = load_vectorstore()
    except FileNotFoundError as exc:
        logger.error("%s\nRun the indexing pipeline first.", exc)
        sys.exit(1)

    retriever = get_retriever(vectorstore)

    # Run benchmark
    logger.info("Running benchmark …")
    results = run_benchmark(
        retriever=retriever,
        max_questions=args.questions,
        skip_guard=args.skip_guard,
    )

    # Save raw results
    save_results(results, args.output)

    # RAGAS evaluation
    logger.info("Running RAGAS evaluation …")
    df = run_ragas_evaluation(results, output_csv=args.csv)
    print_ragas_summary(df)

    logger.info("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
