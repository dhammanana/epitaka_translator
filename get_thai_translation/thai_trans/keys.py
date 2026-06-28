"""
keys.py — Gemini + Groq key rotation (shared by all modules).

Reads GEMINI_KEY_1, GEMINI_KEY_2, ... and GROQ_API_1, GROQ_API_2, ...
(also accepts bare GROQ_API) from .env / environment.

Groq keys are tried first (higher daily quota). Gemini keys are used
as fallback if all Groq keys are exhausted.

Usage:
    from keys import call_with_rotation, log, Colors
"""

import os
import re
import time
import logging
import datetime
import random
from itertools import cycle
from dotenv import load_dotenv

# Silence SDK internal retry loggers — we handle all rotation ourselves
logging.getLogger("google.genai._api_client").setLevel(logging.CRITICAL)
logging.getLogger("groq").setLevel(logging.CRITICAL)

load_dotenv()


# ── Colour logging ────────────────────────────────────────────────────────────

class Colors:
    RESET         = "\033[0m"
    BOLD          = "\033[1m"
    DIM           = "\033[2m"
    RED           = "\033[31m"
    GREEN         = "\033[32m"
    YELLOW        = "\033[33m"
    BLUE          = "\033[34m"
    MAGENTA       = "\033[35m"
    CYAN          = "\033[36m"
    BRIGHT_RED    = "\033[91m"
    BRIGHT_GREEN  = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE   = "\033[94m"
    BRIGHT_CYAN   = "\033[96m"

TAG_COLORS = {
    "START":      (Colors.BRIGHT_CYAN,   "⚙️ "),
    "TURN":       (Colors.BRIGHT_BLUE,   "🔄"),
    "RESPONSE":   (Colors.BRIGHT_YELLOW, "📤"),
    "MODEL":      (Colors.CYAN,          "💬"),
    "TOOL_CALL":  (Colors.BRIGHT_YELLOW, "🔧"),
    "TOOL_ERR":   (Colors.BRIGHT_RED,    "❌"),
    "TOOL_RESULT":(Colors.GREEN,         "✓ "),
    "SAVED":      (Colors.BRIGHT_GREEN,  "💾"),
    "DONE":       (Colors.BRIGHT_GREEN,  "✨"),
    "RATE_LIMIT": (Colors.BRIGHT_RED,    "⏱️ "),
    "SLEEP":      (Colors.DIM,           "⏸️ "),
    "KEY_ROTATE": (Colors.BRIGHT_YELLOW, "🔑"),
    "ERROR":      (Colors.BRIGHT_RED,    "⚠️ "),
    "INFO":       (Colors.CYAN,          "ℹ️ "),
    "SKIP":       (Colors.DIM,           "⏭️ "),
    "WARN":       (Colors.YELLOW,        "⚡"),
    "TOKENS":     (Colors.BRIGHT_CYAN,   "🔢"),
}

def log(tag: str, msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    color, emoji = TAG_COLORS.get(tag, (Colors.CYAN, "• "))
    colored_tag = f"{color}{emoji}{tag.ljust(12)}{Colors.RESET}"
    print(f"[{ts}] {colored_tag} {msg}", flush=True)


# ── Key loading ───────────────────────────────────────────────────────────────

def _load_groq_keys() -> list[str]:
    """Load GROQ_API, GROQ_API_1, GROQ_API_2, ... from environment."""
    keys = {}
    for k, v in os.environ.items():
        if k == "GROQ_API" and v.strip():
            keys[0] = v.strip()
        else:
            m = re.match(r"GROQ_API_(\d+)$", k)
            if m and v.strip():
                keys[int(m.group(1))] = v.strip()
    ordered = [keys[i] for i in sorted(keys)]
    random.shuffle(ordered)
    return ordered

def _load_gemini_keys() -> list[str]:
    """Load GEMINI_KEY_1, GEMINI_KEY_2, ... from environment."""
    keys = {}
    for k, v in os.environ.items():
        m = re.match(r"GEMINI_KEY_(\d+)$", k)
        if m and v.strip():
            keys[int(m.group(1))] = v.strip()
    ordered = [keys[i] for i in sorted(keys)]
    random.shuffle(ordered)
    return ordered


_groq_keys   = _load_groq_keys()
_gemini_keys = _load_gemini_keys()

if not _groq_keys and not _gemini_keys:
    raise RuntimeError("No GROQ_API[_N] or GEMINI_KEY_[N] found in .env / environment")

if _groq_keys:
    log("START", f"Loaded {len(_groq_keys)} Groq key(s) + {len(_gemini_keys)} Gemini key(s) — Groq preferred")
else:
    log("START", f"No Groq keys found — using {len(_gemini_keys)} Gemini key(s) only")

# Separate cycles for each provider
_groq_cycle   = cycle(range(len(_groq_keys)))   if _groq_keys   else None
_gemini_cycle = cycle(range(len(_gemini_keys))) if _gemini_keys else None
_groq_index   = 0
_gemini_index = 0

# Prime cycles
if _groq_cycle:   next(_groq_cycle)
if _gemini_cycle: next(_gemini_cycle)


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _rotate_groq(reason: str = ""):
    global _groq_index
    _groq_index = next(_groq_cycle)
    log("KEY_ROTATE",
        f"{Colors.BRIGHT_YELLOW}Groq KEY_{_groq_index + 1} ...{_groq_keys[_groq_index][-4:]}{Colors.RESET}"
        + (f" — {reason}" if reason else ""))

def _rotate_gemini(reason: str = ""):
    global _gemini_index
    _gemini_index = next(_gemini_cycle)
    log("KEY_ROTATE",
        f"{Colors.BRIGHT_YELLOW}Gemini KEY_{_gemini_index + 1} ...{_gemini_keys[_gemini_index][-4:]}{Colors.RESET}"
        + (f" — {reason}" if reason else ""))

# Keep rotate_key as a public alias (rotates whichever provider is active)
def rotate_key(reason: str = ""):
    if _groq_keys:
        _rotate_groq(reason)
    elif _gemini_keys:
        _rotate_gemini(reason)

def current_groq_key() -> str:
    return _groq_keys[_groq_index]

def current_gemini_key() -> str:
    return _gemini_keys[_gemini_index]


# ── LangChain LLM factories ───────────────────────────────────────────────────

def make_groq_llm(model: str, tools):
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model,
        api_key=current_groq_key(),
        temperature=0,
        max_retries=0,
    ).bind_tools(tools)

def make_gemini_llm(model: str, tools):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=current_gemini_key(),
        temperature=0,
        max_retries=0,
    ).bind_tools(tools)


# ── Unified retry wrapper ─────────────────────────────────────────────────────

def call_with_rotation(fn, tag: str = ""):
    """
    Call fn() and retry across all Groq keys, then all Gemini keys on
    429 / 503 / 504. Rotates to next key immediately on quota errors.
    Raises only when every key is exhausted.
    """
    is_quota   = lambda e: "429" in e or "RESOURCE_EXHAUSTED" in e or "rate_limit_exceeded" in e
    is_server  = lambda e: "503" in e or "UNAVAILABLE" in e
    is_timeout = lambda e: "504" in e or "DEADLINE_EXCEEDED" in e
    is_retryable = lambda e: is_quota(e) or is_server(e) or is_timeout(e)

    total_keys  = len(_groq_keys) + len(_gemini_keys)
    max_retries = total_keys * 2

    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            if not is_retryable(err):
                log("ERROR", f"{Colors.BRIGHT_RED}Non-retryable: {err[:120]}{Colors.RESET}")
                raise

            if attempt >= max_retries:
                log("ERROR", f"Gave up after {max_retries} attempts. Last: {err[:80]}")
                raise

            if is_quota(err):
                # Rotate immediately — no sleep needed when switching key
                if _groq_keys:
                    _rotate_groq(reason=f"429 — {tag}")
                elif _gemini_keys:
                    _rotate_gemini(reason=f"429 — {tag}")
                wait = 0
            else:
                wait = 20 * attempt
                log("RATE_LIMIT",
                    f"Attempt {attempt}/{max_retries}: sleeping {Colors.YELLOW}{wait}s{Colors.RESET} "
                    f"({err[:60].strip()})")
                time.sleep(wait)