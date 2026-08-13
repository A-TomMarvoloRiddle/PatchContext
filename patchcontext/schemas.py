"""
PatchContext – Shared Pydantic schemas.

PatchContextAnswer  : structured output produced by gpt-4o-mini.
GuardResult         : result from the NLI hallucination guard.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PatchContextAnswer(BaseModel):
    """Structured answer produced by the LLM.

    The LLM is instructed to return opaque source IDs like ``pr:1234``,
    ``commit:abc123``, ``issue:982`` – never raw GitHub URLs.
    The application resolves those IDs into real links afterwards.
    """

    answer: str = Field(
        description="The narrative answer to the user's question, written in "
        "plain English.  Must be grounded in the supplied context passages."
    )
    citations: list[str] = Field(
        description=(
            "List of source identifiers that support the answer.  "
            "Each entry must be one of: "
            "'pr:<number>', 'commit:<sha>', or 'issue:<number>'.  "
            "Only cite sources that appear in the context."
        )
    )


class GuardResult(BaseModel):
    """Outcome of the NLI hallucination-guard step."""

    passed: bool = Field(
        description="True if all citations are grounded and the answer is "
        "entailed by the retrieved context."
    )
    fabricated_citations: list[str] = Field(
        default_factory=list,
        description="Citation IDs that were NOT found in the retrieved documents.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Sentence-level claims that the NLI model found unsupported.",
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation when the guard fails.",
    )
