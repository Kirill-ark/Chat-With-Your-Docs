"""Streamlit chat UI: upload PDFs, ask questions, get cited answers.

Streamlit reruns this whole script on every user interaction, so:
- chat history lives in st.session_state (survives reruns),
- heavy resources (embedding model, Chroma client) are process-wide
  singletons in src.ingest, loaded once and shared across sessions.
"""

import tempfile
from pathlib import Path

import streamlit as st
from groq import APIError, RateLimitError

from src.generate import answer
from src.ingest import clear_collection, get_collection, get_embedder, ingest_pdf

st.set_page_config(page_title="Chat With Your Docs", page_icon=":books:")

if "docs_ready" not in st.session_state:
    # The vector DB persists on disk, so documents ingested in a previous
    # session are still searchable — no need to force a re-upload.
    st.session_state.docs_ready = get_collection().count() > 0


# --- Sidebar: document upload ---
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )
    if uploaded and st.button("Process documents", type="primary"):
        get_embedder()  # load the model up front (process-wide singleton)
        with st.status("Ingesting...", expanded=True) as status:
            total = 0
            for file in uploaded:
                # file_uploader gives bytes in memory; parse_pdf needs a path,
                # so write to a temp file that keeps the original filename.
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_path = Path(tmp_dir) / file.name
                        tmp_path.write_bytes(file.getbuffer())
                        n = ingest_pdf(str(tmp_path))
                except Exception as e:  # corrupt / mis-renamed / unreadable file
                    st.write(f"{file.name}: could not be read ({e})")
                    continue
                st.write(f"{file.name}: {n} chunks")
                total += n
            if total > 0:
                status.update(label=f"Ready — {total} chunks indexed", state="complete")
                st.session_state.docs_ready = True
            else:
                status.update(label="No text extracted", state="error")
        if total == 0:
            st.warning(
                "No text could be extracted. Scanned or image-only PDFs have "
                "no text layer to search — try a text-based PDF."
            )

    if st.session_state.get("docs_ready") and st.button("Clear all documents"):
        clear_collection()
        st.session_state.docs_ready = False
        st.session_state.messages = []
        st.success("Index cleared — upload new documents to continue.")

    st.caption(
        "Answers come only from the uploaded documents, with file and page "
        "citations. If the answer is not in them, the assistant says so. "
        "Note: this demo keeps one shared index — documents stay searchable "
        "until cleared."
    )

# --- Main area: chat ---
st.title("Chat With Your Docs")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list[dict]) -> None:
    """Render the sources block; a caption when nothing relevant was found."""
    if not sources:
        st.caption("No relevant passages found in the documents.")
        return
    with st.expander("Sources"):
        for s in sources:
            match_label = (
                f"similarity {s['score']:.0%}"
                if s["score"] is not None
                else "exact page match"
            )
            st.markdown(f"**`{s['file']}`, p. {s['page']}** — {match_label}")
            preview = s["text"][:350].replace("\n", " ")
            if len(s["text"]) > 350:
                preview += "..."
            st.caption(preview)


# Redraw the whole conversation on every rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []))

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
                    "The free LLM tier hit its rate limit. Wait a minute and ask again."
                )
                st.stop()
            except APIError as e:
                st.error(f"LLM API error: {e}")
                st.stop()
            except RuntimeError as e:  # e.g. GROQ_API_KEY not configured
                st.error(str(e))
                st.stop()
        st.markdown(result["answer"])
        show_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "text": result["answer"], "sources": result["sources"]}
    )
