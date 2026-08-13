"""
PatchContext – GitHub ingestion orchestrator.

``load_all_documents`` is the single public entry point.  It calls the
GitHub client, converts every artifact into LangChain Documents (via the
factory functions in ``documents.py``), and returns a flat ``List[Document]``.

Ingestion order
───────────────
  1. Commits  (+ per-file diffs)
  2. Pull requests  (body + review comments + general comments)
  3. Issues  (body + comments)

Progress is logged at INFO level.  A ``tqdm`` progress-bar is shown when
running interactively.
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.documents import Document
from tqdm import tqdm

from patchcontext import config
from patchcontext.github.client import GitHubClient
from patchcontext.github.documents import (
    commit_files_to_documents,
    commit_to_document,
    issue_comment_to_document,
    issue_to_document,
    pr_issue_comment_to_document,
    pr_review_comment_to_document,
    pr_to_documents,
)

logger = logging.getLogger(__name__)


def load_all_documents(
    client: GitHubClient | None = None,
    include_commit_files: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> list[Document]:
    """Fetch all GitHub artifacts and return them as LangChain Documents.

    Args:
        client: A ``GitHubClient`` instance. Created from config if not provided.
        include_commit_files: If *True*, emit one Document per changed file
            (per-file diffs). These can be large – disabled by default.
        on_progress: Optional callback invoked with status strings, useful for
            updating a Streamlit progress widget.

    Returns:
        A flat ``list[Document]`` ready for splitting and indexing.
    """
    if client is None:
        client = GitHubClient()

    repo = client.repo_full_name
    docs: list[Document] = []

    def _report(msg: str) -> None:
        logger.info(msg)
        if on_progress:
            on_progress(msg)

    # ── 1. Commits ────────────────────────────────────────────────────────────
    _report(f"Ingesting commits from {repo} …")
    commits = list(
        tqdm(
            client.iter_commits(max_items=config.MAX_COMMITS),
            desc="Commits",
            total=config.MAX_COMMITS,
            leave=False,
        )
    )
    for commit in commits:
        try:
            docs.append(commit_to_document(commit, repo))
            if include_commit_files:
                docs.extend(commit_files_to_documents(commit, repo))
        except Exception as exc:
            logger.warning("Skipping commit %s – %s", getattr(commit, "sha", "?"), exc)

    _report(f"  → {len(docs)} commit documents ingested.")

    # ── 2. Pull Requests ──────────────────────────────────────────────────────
    _report(f"Ingesting pull requests from {repo} …")
    pr_doc_count_start = len(docs)

    prs = list(
        tqdm(
            client.iter_pull_requests(max_items=config.MAX_PRS),
            desc="Pull Requests",
            total=config.MAX_PRS,
            leave=False,
        )
    )

    for pr in prs:
        try:
            docs.extend(pr_to_documents(pr, repo))

            # General comments on the PR thread
            for ic in client.safe_get_pr_issue_comments(pr):
                if ic.body and ic.body.strip():
                    docs.append(pr_issue_comment_to_document(ic, pr, repo))

            # Review comments (inline code comments)
            for rc in client.safe_get_pr_comments(pr):
                if rc.body and rc.body.strip():
                    docs.append(pr_review_comment_to_document(rc, pr, repo))

        except Exception as exc:
            logger.warning("Skipping PR #%s – %s", getattr(pr, "number", "?"), exc)

    _report(
        f"  → {len(docs) - pr_doc_count_start} PR documents ingested "
        f"({len(prs)} PRs)."
    )

    # ── 3. Issues ─────────────────────────────────────────────────────────────
    _report(f"Ingesting issues from {repo} …")
    issue_doc_count_start = len(docs)

    issues = list(
        tqdm(
            client.iter_issues(max_items=config.MAX_ISSUES),
            desc="Issues",
            total=config.MAX_ISSUES,
            leave=False,
        )
    )

    for issue in issues:
        try:
            docs.append(issue_to_document(issue, repo))

            for comment in client.safe_get_issue_comments(issue):
                if comment.body and comment.body.strip():
                    docs.append(issue_comment_to_document(comment, issue, repo))

        except Exception as exc:
            logger.warning(
                "Skipping issue #%s – %s", getattr(issue, "number", "?"), exc
            )

    _report(
        f"  → {len(docs) - issue_doc_count_start} issue documents ingested "
        f"({len(issues)} issues)."
    )

    _report(f"Ingestion complete. Total documents: {len(docs)}")
    return docs
