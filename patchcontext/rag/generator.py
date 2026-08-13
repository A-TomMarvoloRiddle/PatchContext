"""
PatchContext – Answer generation pipeline.

``generate_answer`` is the primary entry point used by the Streamlit UI and
the benchmark runner.  It implements the explicit, step-by-step flow described
in the implementation plan (section 12 and 19):

    query
    → retriever.invoke(query)         # LangChain MMR retriever
    → format_docs(docs)               # custom context formatter
    → PATCHCONTEXT_PROMPT             # LangChain ChatPromptTemplate
    → structured_llm.invoke(...)      # gpt-4o-mini with structured output
    → PatchContextAnswer              # Pydantic schema

No LangChain chain composition (SequentialChain / LCEL) is used here;
the flow is explicit Python so it stays debuggable.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from patchcontext import config
from patchcontext.rag.prompts import PATCHCONTEXT_PROMPT
from patchcontext.schemas import PatchContextAnswer

logger = logging.getLogger(__name__)


# ─── Context formatter ────────────────────────────────────────────────────────


def format_docs(docs: list[Document]) -> str:
    """Format retrieved Documents into a structured context string for the LLM.

    Each passage includes its source type, source ID, and GitHub URL so the
    LLM can construct valid citations that can later be resolved by the
    application layer.

    Args:
        docs: Retrieved LangChain Documents from the MMR retriever.

    Returns:
        A multi-passage string ready to be inserted into the prompt.
    """
    passages = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        passage = (
            f"--- PASSAGE {i} ---\n"
            f"SOURCE TYPE : {meta.get('source_type', 'unknown')}\n"
            f"SOURCE ID   : {meta.get('source_id', 'unknown')}\n"
            f"URL         : {meta.get('github_url', '')}\n"
            f"AUTHOR      : {meta.get('author', '')}\n"
            f"DATE        : {meta.get('created_at', '')}\n"
            f"\n"
            f"CONTENT:\n{doc.page_content}"
        )
        passages.append(passage)
    return "\n\n".join(passages)


# ─── LLM factory ─────────────────────────────────────────────────────────────


def _get_structured_llm() -> object:
    """Return a gpt-4o-mini instance configured for structured output."""
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY,
    )
    return llm.with_structured_output(PatchContextAnswer)


# ─── Main pipeline function ───────────────────────────────────────────────────


def generate_answer(
    query: str,
    retriever: VectorStoreRetriever,
) -> tuple[PatchContextAnswer, list[Document]]:
    """Run the full retrieval-augmented generation pipeline.

    Steps:
        1. Retrieve diverse, relevant chunks using LangChain MMR retriever.
        2. Format the retrieved chunks into a structured context string.
        3. Fill the ChatPromptTemplate with the context and question.
        4. Invoke gpt-4o-mini with structured output.

    Args:
        query: The user's natural-language question.
        retriever: A LangChain ``VectorStoreRetriever`` (MMR-configured).

    Returns:
        A tuple of:
          - ``PatchContextAnswer`` – structured answer + citation IDs from LLM
          - ``list[Document]``    – the retrieved context documents (for guard/UI)
    """
    logger.info("Retrieving documents for query: %s", query[:80])
    retrieved_docs: list[Document] = retriever.invoke(query)
    logger.info("Retrieved %d documents.", len(retrieved_docs))

    context_str = format_docs(retrieved_docs)
    prompt_value = PATCHCONTEXT_PROMPT.invoke(
        {"context": context_str, "question": query}
    )

    logger.info("Invoking LLM (%s) …", config.LLM_MODEL)
    structured_llm = _get_structured_llm()
    answer: PatchContextAnswer = structured_llm.invoke(prompt_value)
    logger.info(
        "LLM returned answer with %d citation(s).", len(answer.citations)
    )

    return answer, retrieved_docs
