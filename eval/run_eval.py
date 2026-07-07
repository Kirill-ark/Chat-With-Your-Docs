"""Phase 6 — Evaluation: retrieval accuracy + per-stage latency.

Usage:
    python -m eval.run_eval          # retrieval eval (free, deterministic, no API)
    python -m eval.run_eval --llm    # + LLM latency (one Groq call per question)

Measures, per the project plan:
- hit-rate@1/3/5: did the expected page land in the top-k? Split by question
  type (direct vs paraphrased) — paraphrased questions stress the embeddings.
- unanswerable questions: how many chunks leak through the MIN_SIMILARITY
  floor (ideally zero — the floor is what makes the assistant refuse cleanly).
- latency medians per stage: query embedding / vector search / LLM call.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from src.config import MIN_SIMILARITY, TOP_K
from src.ingest import get_collection, get_embedder
from src.retrieve import retrieve

QUESTIONS_PATH = Path(__file__).parent / "questions.json"


def evaluate(run_llm: bool = False) -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    answerable = [q for q in questions if q["expected_source"]]
    unanswerable = [q for q in questions if not q["expected_source"]]

    embedder = get_embedder()
    collection = get_collection()
    embedder.encode("warm-up")  # first call loads weights; keep it out of timings

    embed_ms, search_ms, llm_ms = [], [], []
    hits = {}  # question -> rank of the expected page (None if not in top-k)

    for q in answerable:
        t0 = time.perf_counter()
        query_embedding = embedder.encode(q["question"]).tolist()
        t1 = time.perf_counter()
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["metadatas"],
        )
        t2 = time.perf_counter()
        embed_ms.append((t1 - t0) * 1000)
        search_ms.append((t2 - t1) * 1000)

        expected = q["expected_source"]
        rank = None
        for i, meta in enumerate(result["metadatas"][0], start=1):
            if meta["file"] == expected["file"] and meta["page"] == expected["page"]:
                rank = i
                break
        hits[q["question"]] = rank

    # --- hit-rate table by question type ---
    print(f"Retrieval hit-rate ({len(answerable)} answerable questions, top-{TOP_K}):")
    print(f"{'type':<14}{'n':>3}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}")
    types = ["direct", "paraphrased"]
    for qtype in types + ["all"]:
        subset = [
            q for q in answerable if qtype == "all" or q["type"] == qtype
        ]
        ranks = [hits[q["question"]] for q in subset]
        row = [
            sum(1 for r in ranks if r is not None and r <= k) for k in (1, 3, 5)
        ]
        print(
            f"{qtype:<14}{len(subset):>3}"
            + "".join(f"{h}/{len(subset)}".rjust(8) for h in row)
        )

    misses = [q for q in answerable if hits[q["question"]] is None]
    if misses:
        print("\nMissed (expected page not in top-k):")
        for q in misses:
            print(f"  [{q['type']}] {q['question']}")

    # --- unanswerable: does the relevance floor hold? ---
    print(f"\nUnanswerable questions vs MIN_SIMILARITY={MIN_SIMILARITY}:")
    for q in unanswerable:
        chunks = retrieve(q["question"])
        leaked = [c for c in chunks if c["score"] >= MIN_SIMILARITY]
        top = max(c["score"] for c in chunks)
        status = "OK (refused)" if not leaked else f"LEAKED {len(leaked)} chunks"
        print(f"  top score {top:.3f} — {status} | {q['question'][:60]}")

    # --- optional: LLM latency ---
    if run_llm:
        from groq import Groq

        from src.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
        from src.generate import SYSTEM_PROMPT, build_user_prompt

        client = Groq(api_key=GROQ_API_KEY)
        for q in answerable:
            chunks = retrieve(q["question"])
            chunks = [c for c in chunks if c["score"] >= MIN_SIMILARITY]
            t0 = time.perf_counter()
            client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(q["question"], chunks)},
                ],
            )
            llm_ms.append((time.perf_counter() - t0) * 1000)

    # --- latency table ---
    print(f"\nLatency medians over {len(answerable)} questions (ms):")
    print(f"  query embedding: {statistics.median(embed_ms):8.1f}")
    print(f"  vector search:   {statistics.median(search_ms):8.1f}")
    if llm_ms:
        med_llm = statistics.median(llm_ms)
        total = statistics.median(embed_ms) + statistics.median(search_ms) + med_llm
        print(f"  LLM call:        {med_llm:8.1f}")
        print(f"  total (approx):  {total:8.1f}")
    else:
        print("  LLM call:        (skipped — run with --llm)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm", action="store_true", help="also measure LLM call latency (Groq API)"
    )
    args = parser.parse_args()
    sys.exit(evaluate(run_llm=args.llm))
