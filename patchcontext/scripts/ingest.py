"""
PatchContext – Ingestion script.

Fetches GitHub artifacts from the FastAPI repository and serialises them to
disk as a list of LangChain Documents (JSON Lines format).

Usage:
    python -m patchcontext.scripts.ingest [--output PATH] [--no-commit-files]

Output file default: data/raw_documents.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from patchcontext.github.client import GitHubClient
from patchcontext.github.loaders import load_all_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PatchContext – GitHub ingestion")
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "raw_documents.jsonl",
        help="Output JSONL file path (default: data/raw_documents.jsonl)",
    )
    parser.add_argument(
        "--commit-files",
        action="store_true",
        default=False,
        help="Also ingest per-file diffs from commits (large!)",
    )
    args = parser.parse_args()

    logger.info("Starting GitHub ingestion …")
    client = GitHubClient()
    docs = load_all_documents(client=client, include_commit_files=args.commit_files)

    # Serialise to JSONL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for doc in docs:
            record = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d documents to %s", len(docs), args.output)


if __name__ == "__main__":
    main()
