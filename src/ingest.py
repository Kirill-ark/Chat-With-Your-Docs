"""Phase 2 — PDF ingestion: parse PDF -> chunk -> embed -> write to Chroma.

Parsing lives in its own function (an "adapter"): anything that yields
(text, file, page) records can feed the same chunk -> embed -> store pipeline,
so adding .txt/.docx support later means writing one new parser, nothing else.
"""

import threading
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

_embedder = None
_client = None
_client_lock = threading.Lock()


def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per process and reuse it.

    Retrieval imports this same function, which guarantees documents and
    queries are embedded by the SAME model — a mismatch would put their
    vectors in different spaces and make similarity search return garbage.
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_collection() -> chromadb.Collection:
    """Open (or create) the persisted Chroma collection, cosine similarity.

    The client is created once per process behind a lock: Streamlit Cloud runs
    sessions in parallel threads, and concurrent PersistentClient creation for
    the same path races inside Chroma's client registry (KeyError on deploy).
    """
    global _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def parse_pdf(pdf_path: str) -> list[dict]:
    """Extract text page by page, keeping filename + page for citations.

    Page numbers are captured here, at parse time, because after chunking
    there is no way to recover which page a piece of text came from.
    """
    pages = []
    filename = Path(pdf_path).name
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:  # skip blank / image-only pages
                pages.append({"text": text, "file": filename, "page": page_number})
    return pages


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks with a sliding window.

    The window moves by (size - overlap) each step, so neighbouring chunks
    share `overlap` characters — a sentence cut at one chunk's boundary
    survives intact at the start of the next one.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk)
        if start + size >= len(text):
            break  # the window already covered the tail; avoid tiny duplicates
    return chunks


def ingest_pdf(pdf_path: str) -> int:
    """Parse -> chunk -> embed -> store one PDF. Returns the number of chunks.

    Chunk ids are deterministic (file_page_index), so re-ingesting the same
    file overwrites its chunks (upsert) instead of duplicating them.
    """
    documents, metadatas, ids = [], [], []
    for page in parse_pdf(pdf_path):
        for i, chunk in enumerate(chunk_text(page["text"])):
            documents.append(chunk)
            metadatas.append({"file": page["file"], "page": page["page"]})
            ids.append(f"{page['file']}_p{page['page']}_c{i}")

    if not documents:
        return 0

    embeddings = get_embedder().encode(documents, show_progress_bar=False).tolist()
    get_collection().upsert(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )
    return len(documents)


if __name__ == "__main__":
    # Sanity check: python -m src.ingest path/to/file.pdf [more.pdf ...]
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.ingest <file.pdf> [<file.pdf> ...]")
    for path in sys.argv[1:]:
        n = ingest_pdf(path)
        print(f"{Path(path).name}: {n} chunks ingested")
