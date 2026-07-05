"""Phase 6 — Evaluation: retrieval accuracy + latency measurements.

Reads questions.json (question + expected source), then for each question:
  - Retrieval accuracy: did the correct chunk appear in the top-k? -> hit-rate.
  - Latency: time the stages separately (query embedding / vector search /
    LLM call / total) and report medians.
  - Chunk-size experiment: run the same eval with small vs large chunks and
    report the accuracy/latency tradeoff.

This phase is what turns "built a demo" into "built and validated a system".

TODO (Phase 6): implement after the core path works end-to-end.
"""

# TODO: implement in Phase 6
