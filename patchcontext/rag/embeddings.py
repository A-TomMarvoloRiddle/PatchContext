"""
PatchContext – Embeddings configuration.

Returns a pre-configured ``OpenAIEmbeddings`` instance using the model
specified in ``config.EMBEDDING_MODEL`` (default: text-embedding-ada-002).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from patchcontext import config


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings instance.

    Using ``lru_cache`` ensures we never accidentally create multiple
    embedding clients (which would each warm up their own HTTP connection pool).
    """
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )
