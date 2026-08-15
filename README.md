# PatchContext

> **Understand the "why" behind a codebase.**
>
> PatchContext is a provenance-aware Retrieval-Augmented Generation (RAG) system that enables engineers to query the historical evolution of the **FastAPI** repository. Instead of answering from documentation alone, it retrieves evidence from **commit history, pull requests, issue discussions, and review comments** to explain the design rationale behind implementation decisions with **verifiable GitHub citations**.

---

> ⚠️ **PROTOTYPE NOTE & RATE-LIMIT DESIGN DECISIONS**
>
> 1. **OpenAI Credit & Token Constraints → Switched to Hugging Face + Groq**:
>    To avoid OpenAI API rate limits (`429 credit_balance_exhausted`), this prototype was configured to use **free local Hugging Face embeddings** (`all-MiniLM-L6-v2` via `sentence-transformers`) for FAISS vector indexing, and **Groq's fast open-source Llama 3.3 70B** (`llama-3.3-70b-versatile`) for answer generation. OpenAI embeddings (`text-embedding-ada-002`) and models remain fully supported via `.env` toggles.
>
> 2. **GitHub API Rate Limits → Hardcoded Ingestion Caps**:
>    The FastAPI repository contains over 7,600 commits, 6,100 PRs, and 3,500 issues. Fetching all items along with their nested review comments requires tens of thousands of API calls, easily exceeding GitHub's API rate limits (5,000 requests/hour). To prevent rate-limit blocks and keep ingestion fast for this prototype, ingestion limits were hardcoded in `config.py` to **500 commits, 300 PRs, and 300 issues** (yielding ~2,725 raw documents and 6,052 vector chunks). These limits can be adjusted or removed in `.env`.

---

## Overview

Modern codebases contain years of architectural decisions scattered across commits, pull requests, issue threads, and review discussions. Traditional code search tools answer **what changed**, but rarely explain **why** those changes were made.

PatchContext bridges this gap by indexing the historical development artifacts of the FastAPI repository and providing evidence-grounded answers using Retrieval-Augmented Generation (RAG).

Given a question such as:

> **"Why was this designed this way?"**

PatchContext retrieves the most relevant discussions from GitHub, diversifies the retrieved context using **Maximum Marginal Relevance (MMR)**, generates an answer using **Llama 3.3 70B / GPT-4o-mini**, validates the generated response with an **NLI-based hallucination guard**, and returns the explanation alongside clickable GitHub references.

---

## Features

- **GitHub History RAG**: Index commit messages, PR descriptions, review comments, issues, and discussion threads.
- **LangChain Integration**: Built cleanly on LangChain Document abstractions, Text Splitters, FAISS Vector Stores, and MMR retrievers.
- **Flexible Embeddings**: Supports both free local Hugging Face embeddings (`all-MiniLM-L6-v2`) and OpenAI (`text-embedding-ada-002`).
- **Flexible LLM Endpoints**: Supports Groq (`llama-3.3-70b-versatile`), OpenAI (`gpt-4o-mini`), OpenRouter, or local Ollama / LM Studio servers.
- **MMR Retrieval**: Maximum Marginal Relevance ensures diverse evidence retrieval without redundant discussion chunks.
- **Deterministic Citations**: Resolves opaque source IDs (`pr:1234`, `commit:abc123`, `issue:982`) to live GitHub links.
- **NLI Hallucination Guard**: Uses Hugging Face DeBERTa NLI (`cross-encoder/nli-deberta-v3-base`) to block unsupported claims.
- **Streamlit UI**: Dark-themed interactive interface with clickable citation pills, source passage inspection, and debug metadata viewers.
- **Automated Evaluation**: Evaluates faithfulness, relevancy, and context metrics across a 50-question benchmark using RAGAS.

---

# System Architecture

```text
                      FastAPI GitHub Repository
                                │
                                ▼
                     GitHub API (PyGithub)
                                │ (Ingestion capped for rate limits)
                                ▼
                      LangChain Documents
                                │
                                ▼
                 Artifact-Aware Text Chunking
                                │
                                ▼
            Hugging Face Embeddings / OpenAI Ada-002
                                │
                                ▼
                     LangChain FAISS Index
                                │
                                ▼
                  MMR Retriever (LangChain)
                                │
                                ▼
                  Retrieved LangChain Documents
                                │
                                ▼
                     Prompt Construction
                                │
                                ▼
               Groq Llama 3.3 70B / GPT-4o-mini
                                │
                                ▼
                  Structured PatchContextAnswer
                                │
                                ▼
                 Citation Validation + NLI Guard
                                │
                                ▼
                    Clickable GitHub Citations
                                │
                                ▼
                          Streamlit UI
```

---

# Project Structure

```text
PatchContext/
├── .env.example              # Template for environment variables
├── .gitignore
├── requirements.txt
├── Detailed Implementation Plan.md
│
├── data/
│   ├── benchmark.json        # 50 FastAPI design-rationale questions
│   ├── raw_documents.jsonl   # Exported raw GitHub documents
│   └── ragas_scores.csv      # Evaluation results
│
├── indexes/
│   └── fastapi/              # Saved FAISS vector index
│
└── patchcontext/
    ├── config.py             # Central env-driven configuration & caps
    ├── schemas.py            # Pydantic schemas (PatchContextAnswer, GuardResult)
    ├── app.py                # Main application entry point
    │
    ├── github/               # Ingestion Layer
    │   ├── client.py         # PyGithub client with retry logic & rate-limit handling
    │   ├── documents.py      # Artifact -> LangChain Document converters
    │   └── loaders.py        # Ingestion orchestrator
    │
    ├── rag/                  # RAG Pipeline
    │   ├── splitter.py       # Artifact-aware chunking strategy
    │   ├── embeddings.py     # Embeddings factory (Hugging Face / OpenAI)
    │   ├── vectorstore.py    # FAISS index build, save, load
    │   ├── retriever.py      # LangChain MMR retriever
    │   ├── prompts.py        # ChatPromptTemplate with strict citation rules
    │   └── generator.py      # Answer generation pipeline
    │
    ├── guard/                # Hallucination & Citation Guard
    │   ├── validator.py      # Citation grounding check & URL resolver
    │   └── nli.py            # DeBERTa NLI entailment guard
    │
    ├── evaluation/           # RAGAS Benchmarking
    │   ├── benchmark.py      # Pipeline benchmark harness
    │   └── ragas_eval.py     # RAGAS evaluation runner
    │
    ├── scripts/              # CLI Executables
    │   ├── ingest.py         # Step 1: Download GitHub history
    │   ├── index.py          # Step 2: Build FAISS vector store
    │   └── evaluate.py       # Step 3: Run benchmark evaluation
    │
    └── ui/
        └── streamlit_app.py  # Streamlit UI
```

---

# Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.11+ |
| Framework | LangChain |
| UI | Streamlit |
| Default LLM | Groq `llama-3.3-70b-versatile` (or OpenAI `gpt-4o-mini`) |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` (or OpenAI `text-embedding-ada-002`) |
| Vector Store | FAISS |
| GitHub Access | PyGithub |
| Hallucination Guard | DeBERTa NLI (`cross-encoder/nli-deberta-v3-base`) |
| Evaluation | RAGAS |
| Data Validation | Pydantic |

---

# Pipeline Details

## 1. Repository Ingestion & Rate-Limit Strategy
Artifacts (commits, PRs, review comments, issues, and issue comments) are fetched via PyGithub. Because GitHub imposes a rate limit of 5,000 requests/hour, ingestion caps are defined in `config.py`:
- `MAX_COMMITS=500`
- `MAX_PRS=300`
- `MAX_ISSUES=300`

## 2. Artifact-Aware Chunking
- Short comments and commit messages are preserved whole.
- PR bodies, issue descriptions, and long diffs are split using `RecursiveCharacterTextSplitter` (chunk size 800, overlap 200).

## 3. Embeddings & Vector Store
- Document chunks are converted into dense vector embeddings using local Hugging Face models (`all-MiniLM-L6-v2`) or OpenAI embeddings.
- Indexed into a local **FAISS** vector store.

## 4. MMR Retrieval
- Uses LangChain's Maximum Marginal Relevance (`search_type="mmr"`, `k=6`, `fetch_k=20`, `lambda_mult=0.65`) to retrieve diverse, non-redundant context passages.

## 5. Structured Answer Generation
- Prompts instruct the LLM to return answers grounded *only* in the retrieved context passages, accompanied by opaque citation keys (`pr:1234`, `commit:abc123`, `issue:982`).

## 6. Citation Resolution & NLI Guard
- Validates that every citation returned by the LLM was actually present in the retrieved documents.
- Deterministically maps citation keys to exact GitHub URLs (`https://github.com/tiangolo/fastapi/pull/1234`).
- Uses a Hugging Face DeBERTa cross-encoder NLI model to check for unsupported claims.

---

# Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/<username>/PatchContext.git
cd PatchContext
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set your API keys:

```env
# Groq (Free & Fast LLM)
GROQ_API_KEY=gsk_your_groq_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1

# Embeddings (Free local Hugging Face)
EMBEDDING_PROVIDER=huggingface
HF_EMBEDDING_MODEL=all-MiniLM-L6-v2

# GitHub Token
GITHUB_TOKEN=ghp_your_github_token_here
```

---

# Running the Project

### Step 1 — Ingest GitHub Artifacts
```bash
python -m patchcontext.scripts.ingest
```
*Saves raw documents to `data/raw_documents.jsonl`.*

### Step 2 — Build FAISS Vector Index
```bash
python -m patchcontext.scripts.index
```
*Generates FAISS vector files in `indexes/fastapi/`.*

### Step 3 — Launch Streamlit Web UI
```bash
streamlit run patchcontext/app.py
```
*Opens interactive interface at `http://localhost:8501`.*

### Step 4 — Run RAGAS Benchmark Evaluation
```bash
python -m patchcontext.scripts.evaluate
```
*Runs the 50-question benchmark dataset in `data/benchmark.json` and outputs scores to `data/ragas_scores.csv`.*

---
