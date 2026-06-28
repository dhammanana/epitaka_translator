"""
Tipitaka Translation Pipeline
==============================
For each row in thaimm.matching:
  1. Fetch Thai text from thaimm.main (volume + page range)
  2. Fetch Pali sentences from nissaya.sentences (book_id + para range)
  3. Send both to Gemini for Thai translation
  4. Save result to epitaka_th.db (same schema as nissaya.sentences)

Usage:
  python translate_tipitaka.py \
      --thaimm data/thaimm.sqlite \
      --nissaya data/nissaya.db \
      --output data/epitaka_th.db \
      [--start-row 0] [--workers 4] [--dry-run]
"""

import os
import re
import sys
import time
import json
import sqlite3
import logging
import argparse
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Logging — colorized console + plain file
# ─────────────────────────────────────────────

class ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG:    "\033[37m",    # white
        logging.INFO:     "\033[36m",    # cyan
        logging.WARNING:  "\033[33m",    # yellow
        logging.ERROR:    "\033[31m",    # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"
    _GREY  = "\033[90m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        ts    = self.formatTime(record, "%H:%M:%S")
        level = f"{color}{record.levelname:<8}{self._RESET}"
        msg   = record.getMessage()
        # Dim the timestamp, color the level, normal message
        return f"{self._GREY}{ts}{self._RESET} {level} {msg}"

_plain_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(ColorFormatter())

_file = logging.FileHandler("translate_tipitaka.log", encoding="utf-8")
_file.setFormatter(_plain_fmt)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_console)
logging.root.addHandler(_file)

# logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# API key rotation
# ─────────────────────────────────────────────

class AllKeysExhaustedError(RuntimeError):
    """Raised when every API key has been permanently removed due to quota exhaustion."""
class KeyRotator:
    """
    Thread-safe round-robin key pool with permanent key removal.
    Implemented as a thread-safe Singleton for library usage.

    - Keys are removed permanently when they hit 429 RESOURCE_EXHAUSTED.
    - When the pool is empty, next() raises AllKeysExhaustedError.
    """
    
    _instance = None
    _class_lock = threading.Lock()  # Prevents race conditions during first creation

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._class_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Guard: Python calls __init__ every time KeyRotator() is called.
        # This prevents resetting the pool on subsequent access.
        if getattr(self, "_initialized", False):
            return

        self._lock = threading.Lock()
        pattern = re.compile(r"^GEMINI_KEY_\d+$")
        keys = [
            v.strip()
            for k, v in os.environ.items()
            if pattern.match(k) and v.strip()
        ]
        if not keys:
            raise RuntimeError(
                "No GEMINI_KEY_<N> environment variables found. "
                "Set GEMINI_KEY_1, GEMINI_KEY_2, … before running."
            )
        log.info(f"Loaded {len(keys)} Gemini API key(s).")
        self._keys = list(keys)
        self._index = 0          # always points to the next key to hand out
        
        self._initialized = True  # Mark initialization as complete

    def next(self) -> str:
        """Return next key in round-robin order. Raises AllKeysExhaustedError if pool is empty."""
        with self._lock:
            if not self._keys:
                raise AllKeysExhaustedError("All API keys have been exhausted (429). Exiting.")
            key = self._keys[self._index % len(self._keys)]
            self._index = (self._index + 1) % len(self._keys)
            return key

    def remove(self, key: str):
        """Permanently remove a key from the pool (called on 429)."""
        with self._lock:
            if key in self._keys:
                idx = self._keys.index(key)
                self._keys.remove(key)
                log.warning(
                    f"Key …{key[-6:]} removed from pool (quota exhausted). "
                    f"{len(self._keys)} key(s) remaining."
                )
                # Adjust index so we don't skip the key that slid into this slot
                if self._keys and self._index > idx:
                    self._index -= 1
                if self._keys:
                    self._index %= len(self._keys)
                else:
                    self._index = 0

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._keys)


# Optional: You can still pre-instantiate or leave it to lazy-loading.
# Because it's a Singleton, ROTATOR = KeyRotator() anywhere will yield the exact same instance.
ROTATOR = KeyRotator()

DB_WRITE_LOCK = threading.Lock()  # serialises all writes to the output DB

# ─────────────────────────────────────────────
# Gemini call with timeout + per-call key rotation
# ─────────────────────────────────────────────

def call_groq(prompt: str) -> str | None:
    """Fallback to Groq API when Gemini returns None."""
    for env_key in ("GROQ_API_1", "GROQ_API_2"):
        api_key = os.environ.get(env_key)
        if not api_key:
            continue
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning(f"Groq fallback failed with {env_key}: {e}")
    return None

def call_gemini(prompt: str, timeout_seconds: int = 300, model='gemini-3.1-flash-lite1') -> str:
    """
    Call Gemini, retrying forever until success or all keys are exhausted.

    Key lifecycle:
    - 429 RESOURCE_EXHAUSTED → key removed from pool permanently; try next key immediately.
    - 401 / 403              → invalid key; skip and try next key.
    - Timeout (>timeout_s)   → skip and try next key.
    - Any other error        → log and retry with next key after a brief pause.

    Raises AllKeysExhaustedError when the key pool becomes empty.
    """
    global ROTATOR

    attempt = 0
    while True:
        attempt += 1
        key = ROTATOR.next()          # raises AllKeysExhaustedError if pool empty
        key_suffix = key[-6:] if len(key) >= 6 else key

        result: dict = {"response": None, "error": None}

        def gemini_call():
            try:
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        thinking_config=genai.types.ThinkingConfig(
                            thinking_budget=512,
                        ),
                    ),
                )
                
                result["response"] = response.text

            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=gemini_call, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            log.error(
                f"Gemini timeout (>{timeout_seconds}s) on key …{key_suffix} "
                f"(attempt {attempt}). Retrying with next key."
            )
            continue

        if result["error"] is not None:
            e = result["error"]
            status = getattr(e, "status_code", None) or getattr(e, "code", None)

            if status == 429:
                log.warning(
                    f"429 RESOURCE_EXHAUSTED on key …{key_suffix} (attempt {attempt}). "
                    f"Removing key from pool permanently."
                )
                ROTATOR.remove(key)
                time.sleep(20)
                # AllKeysExhaustedError will be raised on the next ROTATOR.next() if pool is now empty
                continue

            if status in (401, 403):
                log.warning(
                    f"{status} on key …{key_suffix} (attempt {attempt}). "
                    f"Invalid/unauthorised key, skipping."
                )
                time.sleep(20)
                continue

            log.warning(
                f"Gemini error on key …{key_suffix} (attempt {attempt}): {e}. Retrying."
            )
            time.sleep(20)
            continue

        if result["response"] is not None:
            log.debug(f"Gemini call succeeded on key …{key_suffix} (attempt {attempt}).")
            return result["response"]

        # Gemini returned None — try Groq fallback
        log.warning(
            f"Gemini returned None on key …{key_suffix} (attempt {attempt}). Trying Groq fallback."
        )
        groq_response = call_groq(prompt)
        if groq_response is not None:
            log.debug(f"Groq fallback succeeded with length {len(groq_response)}")
            return groq_response
        log.warning("Groq fallback also failed. Retrying Gemini.")
        time.sleep(20)





# ─────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────

def open_db(path: str, read_only: bool = False) -> sqlite3.Connection:
    uri = f"file:{path}{'?mode=ro' if read_only else ''}".replace("\\", "/")
    con = sqlite3.connect(uri, uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    if not read_only:
        con.execute("PRAGMA journal_mode=WAL")
    return con
