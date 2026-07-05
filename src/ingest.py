"""Phase 2 — PDF ingestion: parse PDF -> chunk -> embed -> write to Chroma.

Flow:
  1. Extract text per page with PyMuPDF, keeping (filename, page) as metadata
     so answers can later cite "notes.pdf, p. 4".
  2. Split each page into overlapping chunks (see CHUNK_SIZE / CHUNK_OVERLAP).
  3. Embed chunks with sentence-transformers (same model as the query).
  4. Store vectors + text + metadata in ChromaDB (persisted to CHROMA_DIR).

TODO (Phase 2): implement the functions below.
"""

# TODO: implement in Phase 2
