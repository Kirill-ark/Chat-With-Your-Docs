"""Phase 3 — Retrieval: query -> embed -> similarity search -> top-k chunks.

Embed the question with the SAME model used at ingest time, then ask Chroma for
the top-k nearest chunks by cosine similarity. Returns the chunks together with
their metadata (filename, page) and similarity scores — the scores are surfaced
in the UI for transparency and used in the Evaluation phase.

TODO (Phase 3): implement the retrieval function below.
"""

# TODO: implement in Phase 3
