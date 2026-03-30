"""
config.py — Central configuration for Pāli RAG
All settings read from .env or environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
DB_PATH       = os.getenv("DB_PATH", "data/nissaya.db")
PROGRESS_DIR  = os.getenv("PROGRESS_DIR", "./progress")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "./embeddings_cache")

NISSAYA_DB_PATH = 'data/nissaya.db'
SC_DATA_DB_PATH = 'data/sc-data.db'

# ── Gemini ─────────────────────────────────────────────────────────────────
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))

# Free-tier limits per key
GEMINI_FREE_RPM = int(os.getenv("GEMINI_FREE_RPM", "15"))
GEMINI_FREE_RPD = int(os.getenv("GEMINI_FREE_RPD", "1500"))

# Batch size for embedding via Gemini API
# Gemini text-embedding-004 allows up to 100 texts per batch call
GEMINI_EMBED_BATCH_SIZE = int(os.getenv("GEMINI_EMBED_BATCH_SIZE", "80"))

# gemini-embedding-2-preview
# gemini-embedding-001
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")
GEMINI_EMBED_DIMENSION = 768   # text-embedding-004 fixed dimension

# ── Embedding Configuration ───────────────────────────────────────────────
# Set to True to use a local model instead of Gemini for embeddings
USE_LOCAL_EMBEDDING    = os.getenv("USE_LOCAL_EMBEDDING", "false").lower() == "true"
# Example: "sentence-transformers/all-MiniLM-L6-v2" or "all-distilroberta-v1"
LOCAL_EMBEDDING_MODEL  = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Fixed dimension for text-embedding-004 is 768. 
# If using local, ensure this matches your model's output (e.g., 384 for MiniLM).
GEMINI_EMBED_DIMENSION = 768

# Enrichment batch (how many chunks per Gemini enrichment call)
ENRICH_BATCH_SIZE = int(os.getenv("ENRICH_BATCH_SIZE", "10"))

# Quota-wait interval (seconds) when all keys exhausted
QUOTA_EXHAUSTED_WAIT = int(os.getenv("QUOTA_EXHAUSTED_WAIT", "60"))

# ── Pinecone ───────────────────────────────────────────────────────────────
PINECONE_API      = os.getenv("PINECONE_API", "")
PINECONE_REGION   = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_CLOUD    = os.getenv("PINECONE_CLOUD", "aws")

# One Pinecone index per nikaya
PINECONE_INDEXES = {
    "sutta"  : "pali-sutta",
    "vinaya"    : "pali-vinaya",
    "abhidhamma": "pali-abhidhamma",
    "anna"      : "pali-anna",   # añña / miscellaneous
}

# ── Chunking ───────────────────────────────────────────────────────────────
MAX_TOKENS_PER_CHUNK     = int(os.getenv("MAX_TOKENS_PER_CHUNK", "350"))
SENTENCES_PER_CHUNK_MAX  = int(os.getenv("SENTENCES_PER_CHUNK_MAX", "8"))
SENTENCES_PER_CHUNK_MIN  = int(os.getenv("SENTENCES_PER_CHUNK_MIN", "2"))
CHUNK_OVERLAP_SENTENCES  = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "1"))

# ── Model names ────────────────────────────────────────────────────────────────
# Primary model: configurable via GEMINI_MODEL in .env (default: gemini-2.5-flash)
# Lite model: fixed flash-lite — used for lightweight structural tasks (chunking)
# GEMINI_LITE_MODEL   = "gemini-2.5-flash-lite"
# gemini-3.1-flash-lite-preview (Newest, highest throughput)
# gemini-2.5-flash-lite (Current stable workhorse)
# gemini-2.5-flash (Balanced speed and intelligence)
# gemini-3-flash-preview (Fastest preview model)
GEMINI_LITE_MODEL   = "gemini-2-flash"
CHUNKING_THINKING_BUDGET = 512   # tokens of thinking for the lite/chunking model. between 512 and 24576.
SPLIT_MAX_TOKENS   = 2048   # Gemini output: just a JSON array of ints
ENRICH_MAX_TOKENS  = 1024   # Gemini output: role + concepts + summary

# ── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVAL_CANDIDATES  = int(os.getenv("RETRIEVAL_CANDIDATES", "20"))
CONTEXT_TOP_K         = int(os.getenv("CONTEXT_TOP_K", "8"))
QUERY_EXPANSION_COUNT = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))
USE_HYDE              = os.getenv("USE_HYDE", "true").lower() == "true"

# ── Nikaya categories ──────────────────────────────────────────────────────
# Maps the exact DB nikaya string → Pinecone index key.
#
# DB values (from books.nikaya column):
#   "Sutta Piṭaka"      → sutta index
#   "Vinaya Piṭaka"     → vinaya index
#   "Abhidhamma Piṭaka" → abhidhamma index
#   anything else       → anna index  (aññā = "other")
#
# Matching is case-insensitive and checks containment, so
# "Sutta Piṭaka", "sutta pitaka", "sutta" all resolve correctly.

NIKAYA_INDEX_MAP = {
    # Full canonical strings as stored in the DB
    "sutta piṭaka"      : "sutta",
    "vinaya piṭaka"     : "vinaya",
    "abhidhamma piṭaka" : "abhidhamma",
    # Short-form fallbacks (in case some rows use abbreviated values)
    "sutta"             : "sutta",
    "vinaya"            : "vinaya",
    "abhidhamma"        : "abhidhamma",
}
DEFAULT_NIKAYA_INDEX = "anna"   # aññā — catch-all for anything not in the map

SUPPORTED_LANGUAGES = ["english", "pali", "vietnamese"]

# ── External links ─────────────────────────────────────────────────────────
EPITAKA_BASE_URL = os.getenv("EPITAKA_BASE_URL", "https://epitaka.org/book")

SEMANTIC_ROLES = [
    "Sutta Teaching",
    "Vinaya Rule",
    "Abhidhamma Matrix",
    "Nidana Context",
    "Verse Gatha",
    "Exegesis Vannana",
    "Doctrinal Clarification",
    "Jataka Narrative",
    "Matika Outline",
    "Grammar Philology",
    "Summary Parivara",
    "Devotional Formal",
]

# ── Collect Gemini keys ────────────────────────────────────────────────────
def get_gemini_keys() -> list[str]:
    keys = []
    for i in range(1, 30):
        k = os.getenv(f"GEMINI_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    # Also accept plain GEMINI_API_KEY
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k and k not in keys:
        keys.append(k)
    return keys