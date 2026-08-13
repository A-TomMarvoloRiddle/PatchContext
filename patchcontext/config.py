"""
PatchContext – Central configuration.

All runtime parameters are loaded from environment variables (via .env).
Import `settings` from this module anywhere in the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ─── GitHub ──────────────────────────────────────────────────────────────────

GITHUB_TOKEN: str = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO: str = os.environ.get("GITHUB_REPO", "tiangolo/fastapi")

# ─── OpenAI ──────────────────────────────────────────────────────────────────

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0"))

# ─── FAISS index ─────────────────────────────────────────────────────────────

FAISS_INDEX_DIR: Path = Path(
    os.environ.get("FAISS_INDEX_DIR", str(_PROJECT_ROOT / "indexes" / "fastapi"))
)

# ─── Chunking ─────────────────────────────────────────────────────────────────

CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "200"))

# Character threshold below which we skip splitting (e.g. short comments)
SHORT_DOC_THRESHOLD: int = int(os.environ.get("SHORT_DOC_THRESHOLD", "600"))

# ─── MMR retriever ────────────────────────────────────────────────────────────

MMR_K: int = int(os.environ.get("MMR_K", "6"))
MMR_FETCH_K: int = int(os.environ.get("MMR_FETCH_K", "20"))
MMR_LAMBDA_MULT: float = float(os.environ.get("MMR_LAMBDA_MULT", "0.65"))

# ─── NLI guard ────────────────────────────────────────────────────────────────

NLI_MODEL: str = os.environ.get(
    "NLI_MODEL", "cross-encoder/nli-deberta-v3-base"
)
NLI_ENTAILMENT_THRESHOLD: float = float(
    os.environ.get("NLI_ENTAILMENT_THRESHOLD", "0.5")
)
NLI_ENABLED: bool = os.environ.get("NLI_ENABLED", "true").lower() == "true"

# ─── Ingestion limits (to avoid rate-limit / cost blowout) ───────────────────

MAX_COMMITS: int = int(os.environ.get("MAX_COMMITS", "500"))
MAX_PRS: int = int(os.environ.get("MAX_PRS", "300"))
MAX_ISSUES: int = int(os.environ.get("MAX_ISSUES", "300"))

# ─── Benchmark ────────────────────────────────────────────────────────────────

BENCHMARK_FILE: Path = Path(
    os.environ.get(
        "BENCHMARK_FILE", str(_PROJECT_ROOT / "data" / "benchmark.json")
    )
)
