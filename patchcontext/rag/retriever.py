"""
PatchContext – LangChain MMR retriever.

A single function ``get_retriever`` wraps a FAISS vectorstore in LangChain's
built-in MMR retriever.  All parameters come from ``config`` so they can be
tuned via environment variables without code changes.

MMR (Maximal Marginal Relevance) ensures that the retrieved chunks are both
relevant *and* diverse – important when a query could match many near-duplicate
comment threads.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from patchcontext import config


def get_retriever(vectorstore: FAISS) -> VectorStoreRetriever:
    """Return an MMR-configured LangChain retriever for *vectorstore*.

    Configuration (all tunable via .env):
        MMR_K           – number of final documents returned          (default 6)
        MMR_FETCH_K     – candidate pool size before MMR re-ranking   (default 20)
        MMR_LAMBDA_MULT – diversity/relevance trade-off  0=diverse, 1=relevant
                          (default 0.65 = lean towards relevance)

    Args:
        vectorstore: A loaded or freshly built FAISS vectorstore.

    Returns:
        A ``VectorStoreRetriever`` using ``search_type="mmr"``.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.MMR_K,
            "fetch_k": config.MMR_FETCH_K,
            "lambda_mult": config.MMR_LAMBDA_MULT,
        },
    )
