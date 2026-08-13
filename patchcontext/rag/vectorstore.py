"""
PatchContext – FAISS vectorstore management.

Two public functions:

  build_vectorstore(docs)    – embed documents and return a FAISS instance
  load_vectorstore()         – load a previously saved FAISS index from disk
  save_vectorstore(vs)       – persist the FAISS index to disk

The index directory is set by ``config.FAISS_INDEX_DIR``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from patchcontext import config
from patchcontext.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def build_vectorstore(documents: list[Document]) -> FAISS:
    """Embed *documents* and return a new FAISS vectorstore.

    Args:
        documents: Split, ready-to-index LangChain Documents.

    Returns:
        An in-memory ``FAISS`` vectorstore.
    """
    if not documents:
        raise ValueError("Cannot build a vectorstore from an empty document list.")

    logger.info("Building FAISS index over %d chunks …", len(documents))
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    logger.info("FAISS index built.  Total vectors: %d", vectorstore.index.ntotal)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, index_dir: Path | None = None) -> None:
    """Persist the FAISS index to disk.

    Args:
        vectorstore: The FAISS instance to save.
        index_dir: Target directory. Defaults to ``config.FAISS_INDEX_DIR``.
    """
    index_dir = index_dir or config.FAISS_INDEX_DIR
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(index_dir))
    logger.info("FAISS index saved to %s", index_dir)


def load_vectorstore(index_dir: Path | None = None) -> FAISS:
    """Load a previously saved FAISS index from disk.

    Args:
        index_dir: Source directory. Defaults to ``config.FAISS_INDEX_DIR``.

    Returns:
        A loaded ``FAISS`` vectorstore ready for retrieval.

    Raises:
        FileNotFoundError: If no index exists at *index_dir*.
    """
    index_dir = index_dir or config.FAISS_INDEX_DIR
    index_dir = Path(index_dir)

    index_file = index_dir / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {index_dir}. "
            "Run 'python -m patchcontext.scripts.index' to build it first."
        )

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,  # required for local files
    )
    logger.info(
        "FAISS index loaded from %s. Total vectors: %d",
        index_dir,
        vectorstore.index.ntotal,
    )
    return vectorstore
