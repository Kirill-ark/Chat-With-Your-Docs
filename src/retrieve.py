"""Phase 3 — Retrieval: query -> embed -> similarity search -> top-k chunks.

The question is embedded by the SAME model that embedded the documents
(imported from ingest — single source of truth), then Chroma returns the
top-k nearest chunks by cosine similarity, with metadata and scores.
"""

import re

from src.config import TOP_K
from src.ingest import get_collection, get_embedder

# "page 5", "p. 5", "стр. 5", "страница 5" — a question about a specific page
# is a STRUCTURAL query: semantic search can't answer it (embeddings encode
# content meaning, not document structure), but chunk metadata can.
# The negative lookbehind keeps Russian words that merely END in "стр" from
# triggering routing ("оркестр 5", "регистр 8") — \b can't do this reliably
# for Cyrillic because the preceding letter is also a word character.
PAGE_REFERENCE = re.compile(
    r"(?:\bpage\b|\bp\.|(?<![а-яёa-z])стр(?:аниц\w*|\.)?)\s*(\d+)", re.IGNORECASE
)


def detect_page_reference(question: str) -> int | None:
    """Return the page number mentioned in the question, if any."""
    match = PAGE_REFERENCE.search(question)
    return int(match.group(1)) if match else None


def fetch_page(page: int) -> list[dict]:
    """Fetch ALL chunks of one page via metadata filter — no vector search.

    Used for structural questions ("what is on page 5?"). Score is None:
    these chunks are an exact page match, not a similarity guess.
    """
    result = get_collection().get(
        where={"page": page}, include=["documents", "metadatas"]
    )
    chunks = [
        {
            "text": text,
            "file": meta["file"],
            "page": meta["page"],
            "score": None,
            "_id": chunk_id,
        }
        for chunk_id, text, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]
    # get() returns arbitrary order; restore reading order by chunk index in id
    chunks.sort(key=lambda c: (c["file"], int(c["_id"].rsplit("_c", 1)[1])))
    for c in chunks:
        del c["_id"]
    return chunks


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
