"""
PatchContext – FAISS index builder.

Reads raw Documents from JSONL (produced by ingest.py), splits them, embeds
them using text-embedding-ada-002, and saves the FAISS index to disk.

Usage:
    python -m patchcontext.scripts.index [--input PATH] [--index-dir PATH]

Typical workflow:
    python -m patchcontext.scripts.ingest   # → data/raw_documents.jsonl
    python -m patchcontext.scripts.index    # → indexes/fastapi/
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

from langchain_core.documents import Document

from patchcontext import config
from patchcontext.rag.splitter import split_documents
from patchcontext.rag.vectorstore import build_vectorstore, save_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[Document]:
    """Load Documents from a JSONL file produced by ingest.py."""
    docs: list[Document] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            docs.append(
                Document(
                    page_content=record["page_content"],
                    metadata=record.get("metadata", {}),
                )
            )
    return docs


def main():
    parser = argparse.ArgumentParser(description="PatchContext – FAISS index builder")
    parser.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "data" / "raw_documents.jsonl",
        help="Input JSONL file (default: data/raw_documents.jsonl)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=config.FAISS_INDEX_DIR,
        help=f"FAISS index output directory (default: {config.FAISS_INDEX_DIR})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(
            "Input file not found: %s\n"
            "Run `python -m patchcontext.scripts.ingest` first.",
            args.input,
        )
        sys.exit(1)

    logger.info("Loading documents from %s …", args.input)
    raw_docs = load_jsonl(args.input)
    logger.info("Loaded %d raw documents.", len(raw_docs))

    logger.info("Splitting documents …")
    chunks = split_documents(raw_docs)
    logger.info("Split into %d chunks.", len(chunks))

    logger.info("Building FAISS index (this will call the OpenAI Embeddings API) …")
    vectorstore = build_vectorstore(chunks)

    logger.info("Saving FAISS index to %s …", args.index_dir)
    save_vectorstore(vectorstore, index_dir=args.index_dir)

    logger.info("✅ Index built successfully.  %d vectors.", vectorstore.index.ntotal)


if __name__ == "__main__":
    main()
