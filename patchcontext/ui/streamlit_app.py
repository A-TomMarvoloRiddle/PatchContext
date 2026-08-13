"""
PatchContext – Streamlit UI.

Run with:
    streamlit run patchcontext/ui/streamlit_app.py

This file contains the entire Streamlit application logic. It:
  1. Loads (or prompts to build) the FAISS index.
  2. Accepts a user query.
  3. Runs retrieval → generation → guard → citation resolution.
  4. Renders the answer with clickable GitHub citations and source passages.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# ─── Ensure the project root is on sys.path when running directly ─────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from patchcontext import config
from patchcontext.guard.nli import run_guard
from patchcontext.guard.validator import resolve_all_citations
from patchcontext.rag.generator import generate_answer
from patchcontext.rag.retriever import get_retriever
from patchcontext.rag.vectorstore import load_vectorstore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PatchContext",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Base ────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ──────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%);
        min-height: 100vh;
    }

    /* ── Main content area ───────────────────────────────────────── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 900px;
    }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(22, 27, 34, 0.95);
        border-right: 1px solid rgba(48, 54, 61, 0.7);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* ── Hero header ─────────────────────────────────────────────── */
    .pc-hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .pc-logo {
        font-size: 3rem;
        margin-bottom: 0.4rem;
    }
    .pc-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #e6edf3;
        margin: 0;
    }
    .pc-subtitle {
        font-size: 1.05rem;
        color: #8b949e;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* ── Answer card ─────────────────────────────────────────────── */
    .answer-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin: 1.25rem 0;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .answer-card h3 {
        color: #58a6ff;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0;
        margin-bottom: 0.75rem;
    }
    .answer-text {
        color: #e6edf3;
        font-size: 1.05rem;
        line-height: 1.75;
    }

    /* ── Citations ───────────────────────────────────────────────── */
    .citations-section {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(48, 54, 61, 0.6);
    }
    .citations-section h4 {
        color: #8b949e;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .citation-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(88, 166, 255, 0.1);
        border: 1px solid rgba(88, 166, 255, 0.25);
        border-radius: 6px;
        padding: 0.25rem 0.65rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #58a6ff;
        text-decoration: none;
        transition: all 0.18s ease;
    }
    .citation-pill:hover {
        background: rgba(88, 166, 255, 0.18);
        border-color: rgba(88, 166, 255, 0.5);
        color: #79b8ff;
    }
    .citation-pill-invalid {
        background: rgba(248, 81, 73, 0.1);
        border-color: rgba(248, 81, 73, 0.25);
        color: #f85149;
    }

    /* ── Guard badge ─────────────────────────────────────────────── */
    .guard-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .guard-pass {
        background: rgba(63, 185, 80, 0.12);
        border: 1px solid rgba(63, 185, 80, 0.3);
        color: #3fb950;
    }
    .guard-fail {
        background: rgba(248, 81, 73, 0.12);
        border: 1px solid rgba(248, 81, 73, 0.3);
        color: #f85149;
    }
    .guard-skip {
        background: rgba(139, 148, 158, 0.12);
        border: 1px solid rgba(139, 148, 158, 0.3);
        color: #8b949e;
    }

    /* ── Source passages ─────────────────────────────────────────── */
    .source-card {
        background: rgba(13, 17, 23, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-left: 3px solid rgba(88, 166, 255, 0.4);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.7rem;
        font-size: 0.9rem;
    }
    .source-card-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #8b949e;
        margin-bottom: 0.5rem;
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .source-type-tag {
        background: rgba(139, 148, 158, 0.1);
        border: 1px solid rgba(139, 148, 158, 0.2);
        border-radius: 4px;
        padding: 0.1rem 0.4rem;
        font-size: 0.72rem;
        color: #8b949e;
    }
    .source-card-content {
        color: #c9d1d9;
        line-height: 1.6;
        white-space: pre-wrap;
        font-size: 0.875rem;
        max-height: 180px;
        overflow-y: auto;
    }

    /* ── Search bar ──────────────────────────────────────────────── */
    .stTextArea textarea {
        background: rgba(22, 27, 34, 0.9) !important;
        border: 1px solid rgba(48, 54, 61, 0.9) !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        resize: vertical !important;
        transition: border-color 0.18s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(88, 166, 255, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.08) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.18s ease !important;
        box-shadow: 0 2px 8px rgba(31, 111, 235, 0.3) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%) !important;
        box-shadow: 0 4px 16px rgba(56, 139, 253, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Expanders ───────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(22, 27, 34, 0.7) !important;
        border-radius: 8px !important;
        color: #8b949e !important;
        font-size: 0.9rem !important;
    }

    /* ── Divider ─────────────────────────────────────────────────── */
    hr {
        border-color: rgba(48, 54, 61, 0.5) !important;
    }

    /* ── Spinner ─────────────────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: #58a6ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Vectorstore loader (cached) ───────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _load_retriever():
    """Load FAISS index and return the MMR retriever (cached for the session)."""
    try:
        vs = load_vectorstore()
        return get_retriever(vs)
    except FileNotFoundError:
        return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────


def _render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ PatchContext")
        st.markdown("---")

        st.markdown("**Repository**")
        st.code(config.GITHUB_REPO, language=None)

        st.markdown("**Index path**")
        st.code(str(config.FAISS_INDEX_DIR), language=None)

        st.markdown("**Model**")
        st.code(config.LLM_MODEL, language=None)

        st.markdown("**Embedding**")
        st.code(config.EMBEDDING_MODEL, language=None)

        st.markdown("**Retriever**")
        st.markdown(
            f"MMR · k={config.MMR_K} · fetch_k={config.MMR_FETCH_K} · "
            f"λ={config.MMR_LAMBDA_MULT}"
        )

        st.markdown("---")
        nli_status = "✅ Enabled" if config.NLI_ENABLED else "⚠️ Disabled"
        st.markdown(f"**NLI Guard** {nli_status}")
        if config.NLI_ENABLED:
            st.caption(f"Model: `{config.NLI_MODEL}`")

        st.markdown("---")
        st.caption("Built with LangChain · FAISS · GPT-4o-mini · Streamlit")


# ─── Citation rendering ────────────────────────────────────────────────────────


def _render_citations(
    citations: list[str],
    fabricated: list[str],
) -> str:
    """Build HTML for citation pills."""
    if not citations:
        return "<span style='color:#8b949e;font-size:0.85rem;'>No citations returned.</span>"

    resolved = resolve_all_citations(citations)
    parts: list[str] = []

    for cid in citations:
        url = resolved.get(cid)
        is_fabricated = cid in fabricated

        if url and not is_fabricated:
            parts.append(
                f'<a href="{url}" target="_blank" class="citation-pill">🔗 {cid}</a>'
            )
        else:
            extra_cls = " citation-pill-invalid" if is_fabricated else ""
            tooltip = " (fabricated – not in retrieved context)" if is_fabricated else " (unknown format)"
            parts.append(
                f'<span class="citation-pill{extra_cls}" title="{tooltip}">⚠️ {cid}</span>'
            )

    return "".join(parts)


# ─── Source passage cards ──────────────────────────────────────────────────────


def _render_source_card(doc, index: int) -> str:
    meta = doc.metadata
    stype = meta.get("source_type", "unknown")
    source_id = meta.get("source_id", "")
    url = meta.get("github_url", "")
    author = meta.get("author", "")
    date = (meta.get("created_at", "") or "")[:10]
    content = doc.page_content[:600].replace("<", "&lt;").replace(">", "&gt;")
    if len(doc.page_content) > 600:
        content += "…"

    url_html = f'<a href="{url}" target="_blank" style="color:#58a6ff;">{url}</a>' if url else ""

    return f"""
<div class="source-card">
  <div class="source-card-meta">
    <span><strong style="color:#c9d1d9;">#{index}</strong></span>
    <span class="source-type-tag">{stype}</span>
    <span style="color:#58a6ff;font-family:'JetBrains Mono',monospace;">{source_id}</span>
    {f'<span>👤 {author}</span>' if author else ''}
    {f'<span>📅 {date}</span>' if date else ''}
    {f'<span>{url_html}</span>' if url_html else ''}
  </div>
  <div class="source-card-content">{content}</div>
</div>
"""


# ─── Main app ─────────────────────────────────────────────────────────────────


def main():
    _render_sidebar()

    # Hero
    st.markdown(
        """
        <div class="pc-hero">
          <div class="pc-logo">🔍</div>
          <h1 class="pc-title">PatchContext</h1>
          <p class="pc-subtitle">
            Ask <em>why</em> FastAPI was designed this way — answers grounded in
            actual commit messages, pull requests, and issue threads.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Index status check ─────────────────────────────────────────────────────
    retriever = _load_retriever()

    if retriever is None:
        st.error(
            "⚠️ **No FAISS index found.**\n\n"
            f"Expected index at: `{config.FAISS_INDEX_DIR}`\n\n"
            "Run the ingestion + indexing pipeline first:\n"
            "```bash\n"
            "python patchcontext/scripts/ingest.py\n"
            "python patchcontext/scripts/index.py\n"
            "```"
        )
        st.stop()

    # ── Query input ────────────────────────────────────────────────────────────
    st.markdown("#### 💬 Ask a question")

    example_questions = [
        "Why does FastAPI use Pydantic instead of marshmallow?",
        "What motivated the dependency injection design?",
        "Why was async/await chosen over synchronous handlers?",
        "Why does FastAPI generate OpenAPI docs automatically?",
        "What was the reasoning behind response_model?",
    ]

    with st.expander("💡 Example questions", expanded=False):
        for q in example_questions:
            if st.button(q, key=f"ex_{q[:20]}"):
                st.session_state["query_text"] = q

    query = st.text_area(
        label="Your question",
        value=st.session_state.get("query_text", ""),
        height=100,
        placeholder="e.g. Why does FastAPI use Pydantic for validation?",
        label_visibility="collapsed",
        key="query_input",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ask_clicked = st.button("🔍 Ask PatchContext", use_container_width=True)

    if not ask_clicked or not query.strip():
        if ask_clicked and not query.strip():
            st.warning("Please enter a question.")
        st.stop()

    # ── Pipeline execution ─────────────────────────────────────────────────────
    with st.spinner("Searching commit history, PRs, and issues …"):
        try:
            answer_obj, retrieved_docs = generate_answer(query.strip(), retriever)
        except Exception as exc:
            st.error(f"❌ Generation failed: {exc}")
            logger.exception("generate_answer failed")
            st.stop()

    # ── Guard ──────────────────────────────────────────────────────────────────
    guard_result = None
    fabricated: list[str] = []

    if config.NLI_ENABLED:
        with st.spinner("Running hallucination guard …"):
            try:
                guard_result = run_guard(answer_obj, retrieved_docs)
                fabricated = guard_result.fabricated_citations
            except Exception as exc:
                logger.warning("Guard failed: %s", exc)

    # ── Answer card ────────────────────────────────────────────────────────────
    st.markdown("---")
    citations_html = _render_citations(answer_obj.citations, fabricated)

    st.markdown(
        f"""
        <div class="answer-card">
          <h3>Answer</h3>
          <div class="answer-text">{answer_obj.answer}</div>
          <div class="citations-section">
            <h4>Citations</h4>
            {citations_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Guard status ───────────────────────────────────────────────────────────
    if guard_result is not None:
        if guard_result.passed:
            st.markdown(
                '<div class="guard-badge guard-pass">✅ Hallucination guard: PASSED</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="guard-badge guard-fail">⚠️ Hallucination guard: FAILED — {guard_result.reason}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="guard-badge guard-skip">⬜ Hallucination guard: SKIPPED (NLI_ENABLED=false)</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Retrieved sources expander ─────────────────────────────────────────────
    with st.expander(
        f"📄 View {len(retrieved_docs)} retrieved source passages", expanded=False
    ):
        for i, doc in enumerate(retrieved_docs, start=1):
            st.markdown(
                _render_source_card(doc, i),
                unsafe_allow_html=True,
            )

    # ── Metadata debug expander ────────────────────────────────────────────────
    with st.expander("🔧 Debug: raw metadata", expanded=False):
        for i, doc in enumerate(retrieved_docs, start=1):
            st.caption(f"Passage {i}")
            st.json(doc.metadata)


if __name__ == "__main__":
    main()
