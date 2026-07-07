"""Phase 4 — Generation: grounded prompt -> LLM -> answer + sources.

The grounding prompt is the anti-hallucination mechanism: the model is told to
answer ONLY from the retrieved context and that "the documents do not contain
this" is a correct, expected answer. The source list under each answer comes
from retrieval metadata (deterministic), never from the model's own text —
generated citations could be wrong, metadata cannot.
"""

from groq import Groq

from src.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MIN_SIMILARITY
from src.retrieve import detect_page_reference, fetch_page, retrieve

SYSTEM_PROMPT = """\
You are an assistant that answers questions strictly from the provided context.

Rules:
- Use ONLY the information from the context below. Never use your own knowledge.
- If the context contains partial or related information, answer with what is \
available and clearly state what is missing. Do not refuse entirely when a \
partial answer exists.
- If the context contains nothing relevant to the question, reply that the \
documents do not contain this information. That is a correct answer, not a failure.
- If the user's message is not a question at all (a greeting, thanks, small \
talk), reply politely and briefly. Do not mention the context.
- When you rely on a passage, mention its source in the text, e.g. (file.pdf, p. 4).
- Answer in the same language the question is asked in.
- Be concise and factual."""


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble the augmented prompt: labelled context passages + the question."""
    if chunks:
        context = "\n\n".join(
            f"[Source: {c['file']}, p.{c['page']}]\n{c['text']}" for c in chunks
        )
    else:
        context = "(no relevant passages were found in the documents)"
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question: str) -> dict:
    """Full RAG answer: retrieve -> grounded prompt -> LLM.

    Returns {"answer": str, "sources": [{"file", "page", "score"}, ...]}.
    Sources are the retrieved chunks' metadata — what the model was actually
    shown — ordered best-match first.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    # Query routing: a question about a specific page is answered from that
    # page's chunks directly (metadata filter); anything else goes through
    # semantic search. Falls back to semantic search if the page is empty.
    page = detect_page_reference(question)
    chunks = fetch_page(page) if page is not None else []
    if not chunks:
        # Drop passages below the relevance floor: top-k always returns
        # *something*, and garbage context both invites hallucination and
        # shows meaningless "sources" under small-talk replies.
        chunks = [c for c in retrieve(question) if c["score"] >= MIN_SIMILARITY]

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, chunks)},
        ],
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "file": c["file"],
                "page": c["page"],
                "score": c["score"],
                "text": c["text"],
            }
            for c in chunks
        ],
    }


if __name__ == "__main__":
    # Sanity check: python -m src.generate your question here
    import sys

    question = " ".join(sys.argv[1:])
    if not question:
        sys.exit("usage: python -m src.generate <question>")
    result = answer(question)
    print(f"Q: {question}\n")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['file']}, p.{s['page']} (score {s['score']})")
