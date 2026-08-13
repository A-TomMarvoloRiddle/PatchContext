# PatchContext

> **Understand the "why" behind a codebase.**
>
> PatchContext is a provenance-aware Retrieval-Augmented Generation (RAG) system that enables engineers to query the historical evolution of the **FastAPI** repository. Instead of answering from documentation alone, it retrieves evidence from **commit history, pull requests, issue discussions, and review comments** to explain the design rationale behind implementation decisions with **verifiable GitHub citations**.

---

## Overview

Modern codebases contain years of architectural decisions scattered across commits, pull requests, issue threads, and review discussions. Traditional code search tools answer **what changed**, but rarely explain **why** those changes were made.

PatchContext bridges this gap by indexing the historical development artifacts of the FastAPI repository and providing evidence-grounded answers using Retrieval-Augmented Generation (RAG).

Given a question such as:

> **"Why was this designed this way?"**

PatchContext retrieves the most relevant discussions from GitHub, diversifies the retrieved context using **Maximum Marginal Relevance (MMR)**, generates an answer using **GPT-4o-mini**, validates the generated response with an **NLI-based hallucination guard**, and returns the explanation alongside clickable GitHub references.

---

## Features

- Retrieval-Augmented Generation over FastAPI GitHub history
- Indexes:
  - Commits
  - Pull Requests
  - PR Review Comments
  - Issues
  - Issue Discussions
- LangChain-powered RAG pipeline
- OpenAI Embeddings (`text-embedding-ada-002`)
- FAISS Vector Store
- MMR Retrieval for diverse evidence
- Structured citations to:
  - Commit SHAs
  - Pull Requests
  - Issues
- NLI-based hallucination detection using DeBERTa
- Interactive Streamlit interface
- Quantitative evaluation using RAGAS

---

# System Architecture

```
                      FastAPI GitHub Repository
                               │
                               ▼
                    GitHub API (PyGithub)
                               │
                               ▼
                     LangChain Documents
                               │
                               ▼
                Artifact-aware Text Chunking
                               │
                               ▼
            OpenAI text-embedding-ada-002
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
                        GPT-4o-mini
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

```
PatchContext/
├── .env.example
├── .gitignore
├── requirements.txt
│
├── data/
│   └── benchmark.json
│
└── patchcontext/
    ├── config.py
    ├── schemas.py
    ├── app.py
│
    ├── github/
    │   ├── client.py
    │   ├── documents.py
    │   └── loaders.py
│
    ├── rag/
    │   ├── splitter.py
    │   ├── embeddings.py
    │   ├── vectorstore.py
    │   ├── retriever.py
    │   ├── prompts.py
    │   └── generator.py
│
    ├── guard/
    │   ├── validator.py
    │   └── nli.py
│
    ├── evaluation/
    │   ├── benchmark.py
    │   └── ragas_eval.py
│
    ├── scripts/
    │   ├── ingest.py
    │   ├── index.py
    │   └── evaluate.py
│
    └── ui/
        └── streamlit_app.py
```

---

# Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Framework | LangChain |
| UI | Streamlit |
| LLM | GPT-4o-mini |
| Embeddings | text-embedding-ada-002 |
| Vector Store | FAISS |
| GitHub Access | PyGithub |
| Hallucination Guard | DeBERTa NLI |
| Evaluation | RAGAS |
| Data Validation | Pydantic |

---

# Pipeline

## 1. Repository Ingestion

The GitHub loader extracts repository history using the GitHub REST API through **PyGithub**.

Artifacts collected include:

- Commits
- Pull Requests
- PR Reviews
- Review Comments
- Issues
- Issue Comments

Each artifact is converted into a LangChain `Document`.

---

## 2. Document Construction

Every GitHub artifact becomes a LangChain `Document` containing:

- Page content
- Artifact metadata
- Repository metadata
- GitHub URL
- Commit SHA / PR Number / Issue ID

Metadata is preserved throughout the retrieval pipeline to enable deterministic citation generation.

---

## 3. Chunking

Documents are split using an artifact-aware chunking strategy.

Examples:

- Commit messages remain intact
- PR descriptions are recursively split
- Long issue discussions are chunked
- Short comments are indexed without splitting

---

## 4. Embedding

Document chunks are embedded using:

```
text-embedding-ada-002
```

Embeddings are cached to avoid unnecessary API calls.

---

## 5. Vector Store

Embeddings are stored in a local FAISS index using LangChain.

```
LangChain Documents
        │
        ▼
OpenAI Embeddings
        │
        ▼
FAISS
```

---

## 6. Retrieval

PatchContext performs semantic retrieval using LangChain's MMR retriever.

Configuration:

```
search_type = "mmr"

k = 6

fetch_k = 20

lambda_mult = 0.65
```

MMR improves result diversity by reducing redundant context while maintaining semantic relevance.

---

## 7. Answer Generation

Retrieved documents are formatted into a grounded prompt and passed to GPT-4o-mini.

The model is instructed to:

- Use only retrieved evidence
- Never fabricate citations
- Reference source IDs instead of URLs
- Abstain when evidence is insufficient

The response is returned as a structured `PatchContextAnswer`.

---

## 8. Citation Validation

Every generated citation is validated against the retrieved documents.

Invalid or fabricated references are rejected before the response reaches the user.

---

## 9. Hallucination Guard

Each generated answer is verified using a DeBERTa Natural Language Inference (NLI) model.

Claims unsupported by retrieved evidence are blocked.

The system can:

- Regenerate the response
- Remove unsupported claims
- Abstain if evidence is insufficient

---

## 10. User Interface

The Streamlit interface allows engineers to:

- Ask repository history questions
- View retrieved evidence
- Inspect GitHub citations
- Open commits, PRs, and issues directly
- Understand the reasoning behind architectural decisions

---

# Evaluation

PatchContext is evaluated using a curated benchmark of **50 FastAPI design-rationale questions**.

Evaluation metrics include:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

using **RAGAS**.

Benchmark execution automatically generates:

```
data/ragas_scores.csv
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<username>/PatchContext.git

cd PatchContext
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Copy the example environment file.

```bash
cp .env.example .env
```

Configure:

```env
OPENAI_API_KEY=...

GITHUB_TOKEN=...
```

---

# Build the Knowledge Base

## Step 1 — Download GitHub History

```bash
python patchcontext/scripts/ingest.py
```

Produces:

```
data/raw_documents.jsonl
```

---

## Step 2 — Build the Vector Store

```bash
python patchcontext/scripts/index.py
```

Produces:

```
indexes/fastapi/
```

---

# Launch the Application

```bash
streamlit run patchcontext/ui/streamlit_app.py
```

---

# Run the Benchmark

```bash
python patchcontext/scripts/evaluate.py
```

Outputs:

```
data/ragas_scores.csv
```

---

# Example Questions

- Why was this feature implemented this way?
- What discussion introduced this API?
- Why was support for Python 3.7 removed?
- What motivated this refactoring?
- Which issue led to this architectural change?
- Why did maintainers reject the original implementation?
- What design trade-offs were discussed in this pull request?
- Which commit introduced dependency injection changes?

---

# Acknowledgements

Built using:

- LangChain
- OpenAI
- FAISS
- Streamlit
- PyGithub
- Hugging Face Transformers
- RAGAS
- FastAPI GitHub Repository