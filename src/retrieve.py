"""Phase 3 — Retrieval: query -> embed -> similarity search -> top-k chunks.

The question is embedded by the SAME model that embedded the documents
(imported from ingest — single source of truth), then Chroma returns the
top-k nearest chunks by cosine similarity, with metadata and scores.
"""

from src.config import TOP_K
from src.ingest import get_collection, get_embedder


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k chunks most similar to the question, best first.

    Each result is {"text", "file", "page", "score"}. Score is cosine
    similarity (~1 = same meaning, ~0 = unrelated). Chroma returns cosine
    *distance* (smaller = closer), so we convert: similarity = 1 - distance.
    """
    query_embedding = get_embedder().encode(question).tolist()
    result = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": text,
            "file": meta["file"],
            "page": meta["page"],
            "score": round(1 - distance, 3),
        }
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


if __name__ == "__main__":
    # Sanity check: python -m src.retrieve your question here
    import sys

    question = " ".join(sys.argv[1:])
    if not question:
        sys.exit("usage: python -m src.retrieve <question>")
    print(f"Q: {question}\n")
    for i, c in enumerate(retrieve(question), start=1):
        preview = c["text"][:160].replace("\n", " ")
        print(f"{i}. [score {c['score']}] {c['file']}, p.{c['page']}")
        print(f"   {preview}...\n")
