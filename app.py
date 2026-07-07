"""Phase 5 — Streamlit chat UI: upload PDFs, ask questions, get cited answers.

Streamlit reruns this whole script on every user interaction, so:
- chat history lives in st.session_state (survives reruns),
- the embedding model is loaded via @st.cache_resource (loaded once, reused).
"""

import tempfile
from pathlib import Path

import streamlit as st
from groq import APIError, RateLimitError

from src.generate import answer
from src.ingest import get_collection, get_embedder, ingest_pdf

st.set_page_config(page_title="Chat With Your Docs", page_icon=":books:")

if "docs_ready" not in st.session_state:
    # The vector DB persists on disk, so documents ingested in a previous
    # session are still searchable — no need to force a re-upload.
    st.session_state.docs_ready = get_collection().count() > 0


@st.cache_resource
def load_embedder():
    """Load the sentence-transformers model once per server, not per rerun."""
    return get_embedder()


# --- Sidebar: document upload ---
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )
    if uploaded and st.button("Process documents", type="primary"):
        load_embedder()  # warm up the model before ingesting
        with st.status("Ingesting...", expanded=True) as status:
            total = 0
            for file in uploaded:
                # file_uploader gives bytes in memory; parse_pdf needs a path,
                # so write to a temp file that keeps the original filename.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir) / file.name
                    tmp_path.write_bytes(file.getbuffer())
                    n = ingest_pdf(str(tmp_path))
                st.write(f"{file.name}: {n} chunks")
                total += n
            status.update(label=f"Ready — {total} chunks indexed", state="complete")
        st.session_state.docs_ready = True

    st.caption(
        "Answers come only from the uploaded documents, with file and page "
        "citations. If the answer is not in them, the assistant says so."
    )

# --- Main area: chat ---
st.title("Chat With Your Docs")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw the whole conversation on every rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- `{s['file']}`, p. {s['page']} — similarity {s['score']}"
                    )

question = st.chat_input("Ask a question about your documents")

if question:
    if not st.session_state.get("docs_ready"):
        st.warning("Upload and process at least one PDF first (sidebar).")
        st.stop()

    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating..."):
            try:
                result = answer(question)
            except RateLimitError:
                st.error(
                    "The free LLM tier hit its rate limit. "
                    "Wait a minute and ask again."
                )
                st.stop()
            except APIError as e:
                st.error(f"LLM API error: {e}")
                st.stop()
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for s in result["sources"]:
                match_label = (
                    f"similarity {s['score']:.0%}"
                    if s["score"] is not None
                    else "exact page match"
                )
                st.markdown(f"**`{s['file']}`, p. {s['page']}** — {match_label}")
                st.caption(s["text"][:350].replace("\n", " ") + "...")

    st.session_state.messages.append(
        {"role": "assistant", "text": result["answer"], "sources": result["sources"]}
    )
