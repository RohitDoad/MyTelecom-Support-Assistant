"""Streamlit chat UI for the telecom RAG chatbot.

Implements the conversational interface requirements: free-text questions,
clickable sample questions, in-session history, clear-conversation, and
token-by-token streaming (FR-01..FR-05).

Product note — two audiences, one UI:
- End users (customers) see a clean answer plus a subtle trust signal. Internal
  ticket content is used for grounding but never shown verbatim.
- Reviewers can flip "Show retrieval details" in the sidebar to inspect the
  FAQ / TICKETS / GUIDES documents that grounded each answer. This mirrors how
  sources would be exposed to *support ops*, not to customers.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import config
from chain import build_generation_chain
from retriever import format_context, group_sources, retrieve

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "How do I activate roaming before traveling abroad?",
    "My SIM is not being recognised — what should I do?",
    "There's an unexpected charge on my bill. What can I do?",
    "How do I set up eSIM on my phone?",
    "I hear an echo on my calls. How do I fix it?",
]

TRUST_SIGNAL = "✓ Grounded in MyTelecom's verified help content"

st.set_page_config(page_title="MyTelecom Support Assistant", page_icon="📶")


@st.cache_resource(show_spinner="Loading knowledge base and model…")
def get_chain():
    """Build the generation chain once and reuse it across reruns (NFR-05).

    On a fresh deploy (e.g. Streamlit Community Cloud) the vector store does not
    exist yet, so build it from the committed source data on first launch. This
    keeps the repo free of the generated store while staying self-contained.
    """
    if not config.CHROMA_DIR.exists():
        import ingest_all

        with st.spinner("First run: building the knowledge base (one-time)…"):
            ingest_all.main()
    return build_generation_chain()


def ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None


def submit(question: str) -> None:
    """Queue a user question for processing on the next rerun."""
    st.session_state.pending = question


def render_sources(sources: dict[str, list[dict]], show_details: bool) -> None:
    """Render grounding info under an answer.

    Default: a subtle trust signal only. Reviewer mode: an expandable panel
    that breaks down the retrieved documents by source.
    """
    if not sources:
        return
    if not show_details:
        st.caption(TRUST_SIGNAL)
        return

    total = sum(len(v) for v in sources.values())
    with st.expander(f"🔍 Retrieval details — {total} documents (reviewer view)"):
        st.caption(
            "Documents retrieved as grounding context. Internal TICKETS ground "
            "the answer but their case notes are **redacted everywhere** (they "
            "may contain customer PII) — only the ticket ID + issue type is shown."
        )
        for label, items in sources.items():
            st.markdown(f"**{label}** ({len(items)})")
            for item in items:
                if item["snippet"] is None:
                    st.markdown(f"- *{item['title']}* — _case notes redacted_")
                else:
                    st.markdown(f"- *{item['title']}* — {item['snippet']}")


def main() -> None:
    ensure_state()
    chain = get_chain()

    # --- Sidebar: samples, reviewer toggle, clear (FR-02, FR-04) ----------
    with st.sidebar:
        st.header("📶 MyTelecom Assistant")
        st.caption("Grounded in FAQ, resolved tickets, and official guides.")

        show_details = st.toggle(
            "🔍 Show retrieval details (reviewer view)",
            value=False,
            help="Off = customer view. On = inspect the FAQ/TICKETS/GUIDES "
            "documents that grounded each answer.",
        )

        st.subheader("Try a sample question")
        for q in SAMPLE_QUESTIONS:
            st.button(q, key=f"sample::{q}", use_container_width=True,
                      on_click=submit, args=(q,))
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending = None
            st.rerun()

    st.title("MyTelecom Support Assistant")

    # --- Replay history (FR-03) ------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", {}), show_details)

    # --- Input: free-text box OR a clicked sample (FR-01, FR-02) ---------
    typed = st.chat_input("Ask a telecom support question…")
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                # Retrieve once, on the main thread, so we can both ground the
                # answer and display the sources (FR-06..FR-08).
                docs = retrieve(question)
                context = format_context(docs)
                sources = group_sources(docs)
                # Escape '$' so Streamlit's Markdown doesn't read price pairs
                # (e.g. "$15 ... $105") as LaTeX math and render them in a
                # serif math font.
                answer = st.write_stream(
                    token.replace("$", "\\$")
                    for token in chain.stream(
                        {"question": question, "context": context}
                    )
                )
            except Exception as exc:  # surface API/config errors gracefully
                answer = (
                    "Sorry, something went wrong reaching the assistant. "
                    f"Please try again or call 611.\n\n`{exc}`"
                )
                sources = {}
                st.markdown(answer)
            render_sources(sources, show_details)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
