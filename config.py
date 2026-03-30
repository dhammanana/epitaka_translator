"""
config.py — Central configuration for AI Runner.
All paths, timeouts, and defaults live here.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR  = BASE_DIR / "data"
LOG_DIR   = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Database paths ──────────────────────────────────────────────
RUNNER_DB   = DATA_DIR / "runner.db"       # API keys, task queue, logs
GLOSSARY_DB = DATA_DIR / "glossary.db"     # Job output
NISSAYA_DB  = DATA_DIR / "nissaya.db"
SC_DATA_DB  = DATA_DIR / "sc-data.db"

# ── Source DB — path may be overridden via env or web settings ──
SOURCE_DB = os.environ.get(
    "SOURCE_DB",
    str(BASE_DIR / "data" / "epitaka.db")  # adjust to your actual path
)

# ── Queue / worker ───────────────────────────────────────────────
QUERY_TIMEOUT_SEC    = 30 * 60   # 30 min — kill stalled queries
API_COOLDOWN_SEC     = 60 * 60  # 1 hr  — retry exhausted keys
WORKER_POLL_SEC      = 5        # how often worker checks the queue
MAX_CONCURRENT_TASKS = 5        # parallel workers (1 = sequential)

# ── Default AI provider / model ──────────────────────────────────
DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL    = "gemini-2.5-flash"   # free-tier default

AVAILABLE_MODELS = {
    "gemini": [
        # --- Generation 3.1 (Latest Stable) ---
        "gemini-3.1-pro-preview",        
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-pro-latest",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
    ],
    "anthropic": [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
    ],
}

# ── Flask ─────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("FLASK_SECRET", "change-me-in-production")
DEBUG      = os.environ.get("FLASK_DEBUG", "1") == "1"
PORT       = int(os.environ.get("PORT", 5555))
