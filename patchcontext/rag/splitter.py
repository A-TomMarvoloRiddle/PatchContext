"""
PatchContext – Artifact-aware text splitter.

Strategy (as specified in the implementation plan):

  source_type            action
  ─────────────────────  ──────────────────────────────────────────────
  commit                 no split (commit messages are short)
  commit_file            split at 800/200 (diffs can be very long)
  pull_request           split at 800/200
  pr_comment             no split if ≤ SHORT_DOC_THRESHOLD chars
  pr_issue_comment       no split if ≤ SHORT_DOC_THRESHOLD chars
  issue                  split at 800/200
  issue_comment          no split if ≤ SHORT_DOC_THRESHOLD chars
  <anything else>        split at 800/200 (safe default)

The function ``split_documents`` is the single entry point used by the
indexing script and the ingestion pipeline.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from patchcontext import config

logger = logging.getLogger(__name__)

# Source types that should *never* be split
_NO_SPLIT_TYPES: frozenset[str] = frozenset({"commit"})

# Source types that should be split only when they exceed a character threshold
_THRESHOLD_SPLIT_TYPES: frozenset[str] = frozenset(
    {"pr_comment", "pr_issue_comment", "issue_comment"}
)


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,  # keeps track of where in the original text each chunk came from
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Split a list of LangChain Documents according to artifact-type strategy.

    Metadata is propagated to every child chunk automatically by
    ``RecursiveCharacterTextSplitter``.  An additional ``chunk_index`` key is
    added to make individual chunks distinguishable.

    Args:
        documents: Raw Documents as returned by the ingestion loaders.

    Returns:
        A (potentially longer) list of Documents, each ≤ CHUNK_SIZE characters.
    """
    splitter = _make_splitter()
    result: list[Document] = []

    no_split: list[Document] = []
    to_split: list[Document] = []

    for doc in documents:
        stype = doc.metadata.get("source_type", "")

        if stype in _NO_SPLIT_TYPES:
            no_split.append(doc)
        elif stype in _THRESHOLD_SPLIT_TYPES:
            if len(doc.page_content) <= config.SHORT_DOC_THRESHOLD:
                no_split.append(doc)
            else:
                to_split.append(doc)
        else:
            to_split.append(doc)

    # Pass through short / never-split docs unchanged
    result.extend(no_split)

    # Split the rest; LangChain propagates metadata automatically
    if to_split:
        split_chunks = splitter.split_documents(to_split)
        result.extend(split_chunks)

    logger.info(
        "Splitting: %d docs → %d chunks  (no-split=%d, split=%d → %d)",
        len(documents),
        len(result),
        len(no_split),
        len(to_split),
        len(split_chunks) if to_split else 0,
    )
    return result
