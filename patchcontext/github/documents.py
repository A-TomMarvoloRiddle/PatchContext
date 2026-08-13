"""
PatchContext – GitHub artifact → LangChain Document factory functions.

Each public function accepts a raw PyGithub object (or a pair of them when
context is needed) and returns one or more ``langchain_core.documents.Document``
objects whose ``metadata`` carries the full GitHub provenance required for
citation resolution downstream.

Metadata field conventions
───────────────────────────
  source_type   : str   – one of the 7 types listed below
  source_id     : str   – globally unique, used as the citation key
  repository    : str   – e.g. "tiangolo/fastapi"
  pr_number     : int|None
  issue_number  : int|None
  commit_sha    : str|None
  parent_pr     : int|None  – for comments that belong to a PR
  author        : str
  created_at    : str   – ISO-8601
  github_url    : str   – deep link to the exact artifact on GitHub

Source types
─────────────
  commit          – commit message + stats summary
  commit_file     – per-file patch/diff from a commit
  pull_request    – PR title + body
  pr_comment      – PR review comment (inline code comment)
  pr_issue_comment– general (non-review) comment on a PR thread
  issue           – issue title + body
  issue_comment   – comment on an issue thread
"""

from __future__ import annotations

import logging
from typing import Optional

from github.Commit import Commit
from github.CommitComment import CommitComment
from github.Issue import Issue
from github.IssueComment import IssueComment
from github.PullRequest import PullRequest
from github.PullRequestComment import PullRequestComment
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_REPO = "tiangolo/fastapi"  # overridden via loaders


def _base_meta(
    repository: str,
    source_type: str,
    source_id: str,
    github_url: str,
    author: str,
    created_at: str,
    pr_number: Optional[int] = None,
    issue_number: Optional[int] = None,
    commit_sha: Optional[str] = None,
    parent_pr: Optional[int] = None,
) -> dict:
    return {
        "source_type": source_type,
        "source_id": source_id,
        "repository": repository,
        "pr_number": pr_number,
        "issue_number": issue_number,
        "commit_sha": commit_sha,
        "parent_pr": parent_pr,
        "author": author,
        "created_at": created_at,
        "github_url": github_url,
    }


# ─── Commits ─────────────────────────────────────────────────────────────────


def commit_to_document(commit: Commit, repository: str) -> Document:
    """Produce a single Document from a commit's message and summary stats."""
    sha = commit.sha
    author_name = (
        commit.commit.author.name
        if commit.commit.author
        else (commit.author.login if commit.author else "unknown")
    )
    created_at = (
        commit.commit.author.date.isoformat()
        if commit.commit.author and commit.commit.author.date
        else ""
    )

    # Build a concise summary for the body
    stats = commit.stats
    stats_line = (
        f"Files changed: {stats.total}  +{stats.additions}  -{stats.deletions}"
        if stats
        else ""
    )

    content = f"{commit.commit.message.strip()}\n\n{stats_line}".strip()

    return Document(
        page_content=content,
        metadata=_base_meta(
            repository=repository,
            source_type="commit",
            source_id=f"commit:{sha}",
            github_url=f"https://github.com/{repository}/commit/{sha}",
            author=author_name,
            created_at=created_at,
            commit_sha=sha,
        ),
    )


def commit_files_to_documents(commit: Commit, repository: str) -> list[Document]:
    """One Document per changed file carrying its patch / diff text."""
    sha = commit.sha
    docs: list[Document] = []

    try:
        files = commit.files
    except Exception:
        return docs

    for f in files:
        patch = getattr(f, "patch", None) or ""
        if not patch:
            continue  # binary or too-large diffs have no patch text

        content = f"File: {f.filename}\n\n{patch}"
        doc = Document(
            page_content=content,
            metadata=_base_meta(
                repository=repository,
                source_type="commit_file",
                source_id=f"commit_file:{sha}:{f.filename}",
                github_url=f"https://github.com/{repository}/commit/{sha}",
                author="",
                created_at="",
                commit_sha=sha,
            ),
        )
        doc.metadata["filename"] = f.filename
        docs.append(doc)

    return docs


# ─── Pull requests ────────────────────────────────────────────────────────────


def pr_to_documents(pr: PullRequest, repository: str) -> list[Document]:
    """Produce Documents for the PR title+body (one Document for the PR itself)."""
    body = (pr.body or "").strip()
    content = f"PR #{pr.number}: {pr.title}\n\n{body}" if body else f"PR #{pr.number}: {pr.title}"

    author = pr.user.login if pr.user else "unknown"
    created_at = pr.created_at.isoformat() if pr.created_at else ""

    doc = Document(
        page_content=content,
        metadata=_base_meta(
            repository=repository,
            source_type="pull_request",
            source_id=f"pr:{pr.number}",
            github_url=f"https://github.com/{repository}/pull/{pr.number}",
            author=author,
            created_at=created_at,
            pr_number=pr.number,
        ),
    )
    # Extra fields useful for filtering
    doc.metadata["pr_state"] = pr.state
    doc.metadata["pr_title"] = pr.title
    return [doc]


def pr_review_comment_to_document(
    comment: PullRequestComment, pr: PullRequest, repository: str
) -> Document:
    """Convert a single PR *review* comment (inline code comment) to a Document."""
    author = comment.user.login if comment.user else "unknown"
    created_at = comment.created_at.isoformat() if comment.created_at else ""

    content = f"Review comment on PR #{pr.number} at {comment.path}:\n\n{comment.body or ''}"

    return Document(
        page_content=content,
        metadata=_base_meta(
            repository=repository,
            source_type="pr_comment",
            source_id=f"pr_comment:{pr.number}:{comment.id}",
            github_url=comment.html_url,
            author=author,
            created_at=created_at,
            pr_number=pr.number,
            parent_pr=pr.number,
        ),
    )


def pr_issue_comment_to_document(
    comment: IssueComment, pr: PullRequest, repository: str
) -> Document:
    """Convert a general (non-review) comment on a PR thread to a Document."""
    author = comment.user.login if comment.user else "unknown"
    created_at = comment.created_at.isoformat() if comment.created_at else ""

    content = f"Comment on PR #{pr.number}:\n\n{comment.body or ''}"

    return Document(
        page_content=content,
        metadata=_base_meta(
            repository=repository,
            source_type="pr_issue_comment",
            source_id=f"pr_issue_comment:{pr.number}:{comment.id}",
            github_url=comment.html_url,
            author=author,
            created_at=created_at,
            pr_number=pr.number,
            parent_pr=pr.number,
        ),
    )


# ─── Issues ──────────────────────────────────────────────────────────────────


def issue_to_document(issue: Issue, repository: str) -> Document:
    """Convert an issue title + body into a Document."""
    body = (issue.body or "").strip()
    content = (
        f"Issue #{issue.number}: {issue.title}\n\n{body}"
        if body
        else f"Issue #{issue.number}: {issue.title}"
    )

    author = issue.user.login if issue.user else "unknown"
    created_at = issue.created_at.isoformat() if issue.created_at else ""

    doc = Document(
        page_content=content,
        metadata=_base_meta(
            repository=repository,
            source_type="issue",
            source_id=f"issue:{issue.number}",
            github_url=issue.html_url,
            author=author,
            created_at=created_at,
            issue_number=issue.number,
        ),
    )
    doc.metadata["issue_state"] = issue.state
    doc.metadata["issue_title"] = issue.title
    return doc


def issue_comment_to_document(
    comment: IssueComment, issue: Issue, repository: str
) -> Document:
    """Convert a single issue comment into a Document."""
    author = comment.user.login if comment.user else "unknown"
    created_at = comment.created_at.isoformat() if comment.created_at else ""

    content = f"Comment on Issue #{issue.number}:\n\n{comment.body or ''}"

    return Document(
        page_content=content,
        metadata=_base_meta(
            repository=repository,
            source_type="issue_comment",
            source_id=f"issue_comment:{issue.number}:{comment.id}",
            github_url=comment.html_url,
            author=author,
            created_at=created_at,
            issue_number=issue.number,
        ),
    )
