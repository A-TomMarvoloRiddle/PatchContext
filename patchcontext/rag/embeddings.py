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
def get_embeddings():
    """Return a cached embeddings instance (OpenAI or HuggingFace).

    Using ``lru_cache`` ensures we never accidentally create multiple
    embedding clients.
    """
    if config.EMBEDDING_PROVIDER == "huggingface":
        try:
            try:
                from langchain_huggingface.embeddings import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HuggingFace embeddings '{config.HF_EMBEDDING_MODEL}': {exc}"
            ) from exc

    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )
