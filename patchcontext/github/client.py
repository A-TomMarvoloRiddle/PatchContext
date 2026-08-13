"""
PatchContext – GitHub API client.

Thin wrapper around PyGithub that provides authenticated access to the
FastAPI repository and exposes iterators for the three primary artifact
types: commits, pull requests, and issues (including their sub-resources).

All heavy pagination is kept here so the rest of the codebase never has to
import PyGithub directly.
"""

from __future__ import annotations

import logging
from typing import Iterator

from github import Github, GithubException
from github.Commit import Commit
from github.Issue import Issue
from github.PullRequest import PullRequest
from tenacity import retry, stop_after_attempt, wait_exponential

from patchcontext import config

logger = logging.getLogger(__name__)


class GitHubClient:
    """Authenticated PyGithub client scoped to a single repository."""

    def __init__(
        self,
        token: str | None = None,
        repo: str | None = None,
    ) -> None:
        token = token or config.GITHUB_TOKEN
        repo = repo or config.GITHUB_REPO

        if not token:
            raise ValueError(
                "GITHUB_TOKEN is required.  Set it in your .env file."
            )

        self._gh = Github(token, per_page=100)
        self._repo = self._gh.get_repo(repo)
        logger.info("Connected to GitHub repo: %s", repo)

    # ─── Public iterators ─────────────────────────────────────────────────────

    def iter_commits(self, max_items: int | None = None) -> Iterator[Commit]:
        """Yield commits from most-recent to oldest, up to *max_items*."""
        max_items = max_items or config.MAX_COMMITS
        logger.info("Fetching up to %d commits …", max_items)
        for i, commit in enumerate(self._repo.get_commits()):
            if i >= max_items:
                break
            yield commit

    def iter_pull_requests(
        self,
        state: str = "all",
        max_items: int | None = None,
    ) -> Iterator[PullRequest]:
        """Yield pull requests (open + merged + closed) up to *max_items*."""
        max_items = max_items or config.MAX_PRS
        logger.info("Fetching up to %d pull requests (state=%s) …", max_items, state)
        for i, pr in enumerate(self._repo.get_pulls(state=state, sort="created", direction="desc")):
            if i >= max_items:
                break
            yield pr

    def iter_issues(
        self,
        state: str = "all",
        max_items: int | None = None,
    ) -> Iterator[Issue]:
        """Yield issues (excluding PRs, which GitHub API mixes in) up to *max_items*.

        Note: GitHub's ``get_issues`` returns both Issues and PRs. We skip
        entries that have a ``pull_request`` attribute so only true issues
        are returned.
        """
        max_items = max_items or config.MAX_ISSUES
        logger.info("Fetching up to %d issues (state=%s) …", max_items, state)
        count = 0
        for item in self._repo.get_issues(state=state, sort="created", direction="desc"):
            if count >= max_items:
                break
            if item.pull_request is not None:
                # This is a PR surfaced as an issue – skip it.
                continue
            yield item
            count += 1

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @property
    def repo_full_name(self) -> str:
        return self._repo.full_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def safe_get_pr_comments(self, pr: PullRequest):
        """Return all review comments on a PR with auto-retry on transient errors."""
        return list(pr.get_review_comments())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def safe_get_issue_comments(self, issue: Issue):
        """Return all comments on an issue with auto-retry on transient errors."""
        return list(issue.get_comments())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def safe_get_pr_issue_comments(self, pr: PullRequest):
        """Return general (non-review) comments on a PR."""
        return list(pr.get_issue_comments())
