"""
PatchContext – Citation validator.

Two responsibilities:
  1. ``check_fabricated_citations`` – verifies that every citation returned
     by the LLM corresponds to a ``source_id`` that actually appears in the
     retrieved documents.  This catches hallucinated PR/commit/issue references.

  2. ``resolve_citation`` – maps an opaque source ID like ``pr:1234`` to its
     canonical GitHub URL.  This is kept deterministic and outside the LLM so
     the application controls where links point.

Citation ID format (as instructed in the prompt):
    pr:<number>        → https://github.com/<repo>/pull/<number>
    commit:<sha>       → https://github.com/<repo>/commit/<sha>
    issue:<number>     → https://github.com/<repo>/issues/<number>

Anything else is treated as an unknown/invalid citation.
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document

from patchcontext import config

logger = logging.getLogger(__name__)

# ─── Citation presence check ──────────────────────────────────────────────────

# We need to match citations against source_ids in retrieved docs.
# A citation like "pr:1234" should match source_id "pr:1234" OR any source_id
# that *starts with* the citation prefix (e.g. "pr_comment:1234:5678" for PR 1234).

_PR_RE = re.compile(r"^pr:(\d+)$")
_COMMIT_RE = re.compile(r"^commit:([0-9a-f]+)$", re.IGNORECASE)
_ISSUE_RE = re.compile(r"^issue:(\d+)$")


def _citation_matches_doc(citation: str, doc: Document) -> bool:
    """Return True if *citation* is grounded in *doc*'s metadata."""
    source_id: str = doc.metadata.get("source_id", "")

    # Exact match
    if source_id == citation:
        return True

    # pr:1234 → matches any pr_comment, pr_issue_comment with pr_number=1234
    m = _PR_RE.match(citation)
    if m:
        pr_num = int(m.group(1))
        return doc.metadata.get("pr_number") == pr_num

    # commit:<sha> → matches commit or commit_file with that sha (prefix match)
    m = _COMMIT_RE.match(citation)
    if m:
        sha = m.group(1).lower()
        doc_sha = (doc.metadata.get("commit_sha") or "").lower()
        return doc_sha.startswith(sha) or sha.startswith(doc_sha)

    # issue:982 → matches issue or issue_comment with issue_number=982
    m = _ISSUE_RE.match(citation)
    if m:
        issue_num = int(m.group(1))
        return doc.metadata.get("issue_number") == issue_num

    return False


def check_fabricated_citations(
    citations: list[str],
    retrieved_docs: list[Document],
) -> list[str]:
    """Return the subset of *citations* that cannot be grounded in *retrieved_docs*.

    A citation is considered grounded if at least one retrieved document
    matches it according to ``_citation_matches_doc``.

    Args:
        citations: Source IDs emitted by the LLM (e.g. ["pr:1234", "commit:abc"]).
        retrieved_docs: Documents returned by the MMR retriever.

    Returns:
        A list of fabricated citation IDs (ideally empty).
    """
    fabricated: list[str] = []
    for citation in citations:
        grounded = any(
            _citation_matches_doc(citation, doc) for doc in retrieved_docs
        )
        if not grounded:
            logger.warning("Fabricated citation detected: %s", citation)
            fabricated.append(citation)
    return fabricated


# ─── Citation resolver ────────────────────────────────────────────────────────


def resolve_citation(citation: str, repository: str | None = None) -> str | None:
    """Map an opaque source ID to a canonical GitHub URL.

    Args:
        citation: A source ID like ``pr:1234``, ``commit:abc123``, or ``issue:982``.
        repository: GitHub repo slug (e.g. ``tiangolo/fastapi``).
                    Defaults to ``config.GITHUB_REPO``.

    Returns:
        A full GitHub URL string, or ``None`` if the format is unrecognised.
    """
    repo = repository or config.GITHUB_REPO

    m = _PR_RE.match(citation)
    if m:
        return f"https://github.com/{repo}/pull/{m.group(1)}"

    m = _COMMIT_RE.match(citation)
    if m:
        return f"https://github.com/{repo}/commit/{m.group(1)}"

    m = _ISSUE_RE.match(citation)
    if m:
        return f"https://github.com/{repo}/issues/{m.group(1)}"

    logger.debug("Unknown citation format: %s", citation)
    return None


def resolve_all_citations(
    citations: list[str],
    repository: str | None = None,
) -> dict[str, str | None]:
    """Resolve a list of citation IDs to GitHub URLs.

    Returns:
        A dict mapping ``citation_id → url_or_None``.
    """
    return {c: resolve_citation(c, repository) for c in citations}
