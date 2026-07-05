"""Central configuration: model names, chunk settings, paths, env keys.

Everything tunable lives here so the rest of the code never hard-codes a value.
The chunk settings in particular get tuned in the Evaluation phase (Phase 6),
so keeping them in one place makes the experiment easy to run.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # pull GROQ_API_KEY etc. from .env before any os.getenv below

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")  # persisted vectors (gitignored)
COLLECTION_NAME = "documents"

# --- Embeddings (local, free, CPU-friendly) ---
# Same model MUST be used for ingesting documents AND embedding the query,
# otherwise the vectors live in different spaces and retrieval is garbage.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking (chars) ---
# Long documents are split into small overlapping pieces so retrieval returns
# focused context; overlap avoids cutting a sentence's meaning at a boundary.
# These defaults get compared small-vs-large in Phase 6 (chunk-size experiment).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Retrieval ---
TOP_K = 5  # how many nearest chunks to pull for the LLM

# --- LLM (generation) ---
# Key comes from the environment ONLY, never hard-coded. On deploy it is a hosting secret.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
# Low temperature = near-deterministic wording; we want factual QA, not creativity.
LLM_TEMPERATURE = 0.1
