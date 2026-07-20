"""
ai_client.py — Everything related to talking to the AI (currently Gemini) lives
HERE and only here.

Why this file exists
---------------------
book_translator.py and glossary_builder.py both need to "send a prompt, get
JSON back" — but neither of them should know or care *how* that happens.
This module is the single seam between "the rest of the pipeline" and "the
actual AI provider". If you ever need to:

  - switch models / bump max_output_tokens / change timeouts
  - swap Gemini for a different provider entirely
  - change retry, backoff, or API-key rotation behaviour
  - change per-key rpm/tpm throttling behaviour
  - change how prompts/responses get logged to disk for debugging
  - change how a malformed/truncated AI JSON reply gets salvaged

Key throttling
--------------
Each API key is only allowed AI_RPM_LIMIT requests and AI_TPM_LIMIT tokens
per rolling 60s window (defaults: 3 rpm / 250,000 tpm — override via env
vars). Usage is tracked in a small JSON file (KEY_STATE_FILE, an OS-locked
file so it's safe across processes) keyed by a hash of each key, so several
sessions of book_translator.py / glossary_builder.py running at the same
time and sharing a key pool will correctly throttle each other instead of
independently blowing through the provider's real rate limit.

When every key is currently over budget, KeyRotator.acquire() just sleeps
until the earliest one frees up — it does not remove keys or give up.
Keys are only ever permanently removed (via .remove()) when the provider
itself rejects them (e.g. a persistent HTTP 429/401/403), and once the pool
is completely empty, call_gemini() sends a Telegram alert and exits the
process — a run with zero usable keys should stop loudly, not spin forever
silently skipping chunks.

...you only ever need to edit THIS file. book_translator.py and
glossary_builder.py call the functions below by name and never import
`google.genai` themselves.

Public API (this is the contract the two callers rely on):
  - estimate_tokens(text)                         -> int
  - make_rotator(api_keys)                        -> KeyRotator
  - AllKeysExhaustedError                         (exception)
  - call_ai_with_logging(rotator, prompt, book_id, chunk_id, log_dir,
                          model, system_prompt)    -> str | None
  - parse_ai_json_response(raw, keys)              -> dict
  - send_telegram(message)                         -> None

As long as call_ai_with_logging keeps this signature and still returns either
the raw response text or None, nothing else in the codebase needs to change.
"""

import fcntl
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Telegram notifications (used for "the whole run just died" alerts)
# ══════════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram(message: str) -> None:
    """
    Fire-and-forget Telegram notification.
    Reads TELEGRAM_TOKEN / TELEGRAM_CHAT_ID from env (.env supported via
    python-dotenv, loaded by whichever script imports this module first).
    Silently swallows all errors so a notification failure never breaks a run.
    """
    try:
        token   = TELEGRAM_TOKEN.strip()
        chat_id = TELEGRAM_CHAT_ID.strip()
        if not token or not chat_id:
            return
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id":    chat_id,
            "text":       message,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as exc:
        print(f"[telegram] notification failed: {exc}")


# ══════════════════════════════════════════════════════════════════
# Token estimation
# ══════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """
    Cheap token estimate (~4 characters per token). Good enough for prompt
    size budgeting; not meant to match the provider's real tokenizer exactly.
    """
    return len(text) // 4


# ══════════════════════════════════════════════════════════════════
# API key rotation + rpm/tpm throttling
# ══════════════════════════════════════════════════════════════════

# Per-key budget over a rolling 60s window. Override via env vars if a key
# has different quota (e.g. a paid tier), without touching code.
RATE_WINDOW_SECONDS = 60
DEFAULT_RPM_LIMIT    = int(os.environ.get("AI_RPM_LIMIT", "3"))
DEFAULT_TPM_LIMIT    = int(os.environ.get("AI_TPM_LIMIT", "250000"))

# Where per-key "last used" state is persisted. This is a small JSON file
# guarded by an OS-level file lock (fcntl.flock), so it stays correct even
# when several separate processes (e.g. book_translator.py and
# glossary_builder.py running at the same time, possibly sharing the same
# keys) are all reading/writing it concurrently. Defaults to a fixed path
# under the system temp dir so independent sessions find the same file
# without any extra configuration; override with AI_KEY_STATE_FILE if you
# want per-project isolation.
KEY_STATE_FILE = os.environ.get(
    "AI_KEY_STATE_FILE",
    os.path.join(tempfile.gettempdir(), "gemini_key_rate_state.json"),
)


def _key_id(key: str) -> str:
    """Short, non-secret fingerprint of a key — safe to persist to disk."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class AllKeysExhaustedError(RuntimeError):
    """Raised when every configured Gemini API key has been permanently removed."""


class KeyRotator:
    """
    Round-robins across a pool of Gemini API keys, while enforcing a per-key
    rpm/tpm budget so callers never exceed what the provider actually allows.

    Two distinct situations, handled differently:
      - A key is temporarily over its rpm/tpm budget -> `.acquire()` just
        waits until it (or another key) frees up. This is the *normal*,
        expected case under load and is not an error.
      - A key is permanently unusable (the provider rejected it, e.g. a
        persistent 429/401/403) -> it's dropped via `.remove()`. Once the
        pool is empty, `.acquire()` raises AllKeysExhaustedError instead of
        waiting forever with no usable key — callers should catch that,
        notify, and stop the run.
    """

    def __init__(
        self,
        keys:      list[str],
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        tpm_limit: int = DEFAULT_TPM_LIMIT,
    ):
        self._lock = threading.Lock()
        if not keys:
            raise RuntimeError(
                "No Gemini API keys configured. "
                "Set GEMINI_KEY_<N> env vars or pass --api-keys."
            )
        self._keys      = list(keys)
        self._index     = 0
        self._rpm_limit = rpm_limit
        self._tpm_limit = tpm_limit
        log.info(
            f"Loaded {len(self._keys)} Gemini key(s). "
            f"Budget: {rpm_limit} rpm / {tpm_limit} tpm per key."
        )

    def next(self) -> str:
        """
        Return the next key in round-robin order, with NO rpm/tpm check.
        Prefer `.acquire()` for real calls — this is kept only for callers
        that don't need throttling (e.g. tests).
        """
        with self._lock:
            if not self._keys:
                raise AllKeysExhaustedError("All Gemini API keys exhausted.")
            k = self._keys[self._index % len(self._keys)]
            self._index = (self._index + 1) % len(self._keys)
            return k

    def acquire(self, estimated_tokens: int = 0) -> str:
        """
        Block until some key in the pool has rpm/tpm budget for one more
        request of `estimated_tokens`, reserve that budget, and return the
        key. Loops trying every key (round-robin) before sleeping, so it
        picks up whichever key frees up soonest rather than always waiting
        on the same one.
        """
        while True:
            with self._lock:
                if not self._keys:
                    raise AllKeysExhaustedError("All Gemini API keys exhausted.")
                keys_snapshot = list(self._keys)
                start_index   = self._index

            key, wait_seconds = self._try_reserve(keys_snapshot, start_index, estimated_tokens)
            if key is not None:
                with self._lock:
                    if key in self._keys:
                        self._index = (self._keys.index(key) + 1) % len(self._keys)
                return key

            wait_seconds = max(wait_seconds, 1.0)
            log.info(
                f"[Gemini] All {len(keys_snapshot)} key(s) at rpm/tpm budget — "
                f"waiting {wait_seconds:.0f}s."
            )
            time.sleep(wait_seconds)

    def _try_reserve(
        self,
        keys:              list[str],
        start_index:       int,
        estimated_tokens:  int,
    ) -> tuple[str | None, float]:
        """
        Try to reserve budget for one request on the first key (starting at
        start_index, round-robin) that has room under its rpm/tpm limit in
        the current rolling 60s window.

        Returns (key, 0) and records the usage if a key was reserved, or
        (None, seconds_until_the_soonest_key_frees_up) if every key is
        currently over budget.
        """
        now = time.time()
        soonest_wait = None
        n = len(keys)

        with open(KEY_STATE_FILE, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                state = self._read_state(fh)
                for offset in range(n):
                    key   = keys[(start_index + offset) % n]
                    kid   = _key_id(key)
                    events = [
                        e for e in state.get(kid, [])
                        if now - e[0] < RATE_WINDOW_SECONDS
                    ]
                    req_count   = len(events)
                    token_count = sum(e[1] for e in events)

                    if (req_count < self._rpm_limit
                            and token_count + estimated_tokens <= self._tpm_limit):
                        events.append([now, estimated_tokens])
                        state[kid] = events
                        self._write_state(fh, state)
                        return key, 0.0

                    state[kid] = events  # keep pruned even when over budget
                    if events:
                        oldest   = min(e[0] for e in events)
                        key_wait = RATE_WINDOW_SECONDS - (now - oldest)
                        if soonest_wait is None or key_wait < soonest_wait:
                            soonest_wait = key_wait

                self._write_state(fh, state)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

        return None, (soonest_wait if soonest_wait is not None else RATE_WINDOW_SECONDS)

    @staticmethod
    def _read_state(fh) -> dict:
        fh.seek(0)
        raw = fh.read()
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _write_state(fh, state: dict) -> None:
        fh.seek(0)
        fh.truncate()
        json.dump(state, fh)
        fh.flush()
        os.fsync(fh.fileno())

    def remove(self, key: str):
        """Permanently drop a key (e.g. after a persistent 429/401/403) so it's never handed out again."""
        with self._lock:
            if key in self._keys:
                self._keys.remove(key)
                log.warning(f"Key removed. {len(self._keys)} remaining.")


def make_rotator(api_keys: list[str]) -> KeyRotator:
    """
    Build a KeyRotator from an explicit --api-keys list, or, if that's empty,
    from every GEMINI_KEY_<N> environment variable that's set.
    """
    if api_keys:
        return KeyRotator(api_keys)
    env_keys = [
        v.strip()
        for k, v in os.environ.items()
        if re.match(r"^GEMINI_KEY_\d+$", k) and v.strip()
    ]
    return KeyRotator(env_keys)


# ══════════════════════════════════════════════════════════════════
# Gemini call
# ══════════════════════════════════════════════════════════════════

def call_gemini(
    rotator:            KeyRotator,
    prompt:             str,
    system_prompt:      str,
    model:              str,
    max_output_tokens:  int = 65_000,
    timeout:            int = 300,
) -> str | None:
    """
    Send one prompt to Gemini and return the raw response text, or None if
    every retry attempt failed.

    Behaviour:
      - Runs the actual network call in a background thread so a hung
        request can be abandoned after `timeout` seconds instead of blocking
        forever.
      - On HTTP 429 (rate limit): removes that key from the rotator, waits
        20s, and retries with the next key.
      - On HTTP 401/403 or any other error: waits 20s and retries with the
        same rotation (does not remove the key — those are usually transient
        or auth hiccups, not proof the key itself is bad).
      - Retries up to 10 attempts total before giving up and returning None.
      - If the key pool is ever fully exhausted, sends a Telegram alert and
        exits the whole process — a translation run with zero usable keys
        should stop loudly, not silently produce empty output.
    """
    from google import genai
    from google.genai import types as genai_types

    estimated_tokens = estimate_tokens(prompt) + estimate_tokens(system_prompt)

    for attempt in range(10):
        try:
            key = rotator.acquire(estimated_tokens)
        except AllKeysExhaustedError:
            log.error("[Gemini] All API keys exhausted (rate-limited/invalid). Exiting.")
            send_telegram(
                "<b>AI run FATAL</b>\n"
                "All Gemini API keys are exhausted (rate-limited or invalid).\n"
                "The run has stopped — add fresh keys and restart."
            )
            sys.exit(1)

        result: dict = {"response": None, "error": None}

        def _call():
            try:
                client = genai.Client(api_key=key)
                r = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                result["response"] = r.text
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            log.error(f"[Gemini] Timeout on attempt {attempt + 1}")
            continue

        if result["error"]:
            e = result["error"]
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status == 429:
                rotator.remove(key)
                time.sleep(20)
                continue
            if status in (401, 403):
                time.sleep(20)
                continue
            log.warning(f"[Gemini] Error attempt {attempt + 1}: {e}")
            time.sleep(20)
            continue

        if result["response"] is not None:
            return result["response"]

    log.error("[Gemini] All retry attempts exhausted.")
    return None


def call_ai_with_logging(
    rotator:       KeyRotator,
    prompt:        str,
    book_id:       str,
    chunk_id:      str,
    log_dir:       str,
    model:         str,
    system_prompt: str,
) -> str | None:
    """
    Thin wrapper around call_gemini() that also writes the exact prompt sent
    and the raw response received to `log_dir`, so any run can be audited or
    replayed later. Filenames are timestamped + book/chunk-tagged, e.g.:

        20260716_143012_Sp-i_p10-42_prompt.txt
        20260716_143012_Sp-i_p10-42_response.txt

    Returns the raw response text, or None if the call ultimately failed.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_id   = re.sub(r"[^\w\-]", "_", f"{book_id}_{chunk_id}")
    base_name = f"{timestamp}_{safe_id}"

    prompt_path = os.path.join(log_dir, f"{base_name}_prompt.txt")
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("=== SYSTEM ===\n")
            f.write(system_prompt)
            f.write("\n\n=== USER ===\n")
            f.write(prompt)
    except OSError as exc:
        print(f"[LOG] could not write prompt log: {exc}")

    n_tokens = estimate_tokens(prompt)
    print(f"[AI] calling: book={book_id} chunk={chunk_id} "
          f"{len(prompt)} chars (~{n_tokens} tokens)")

    raw = call_gemini(rotator, prompt, system_prompt, model=model)
    if raw is None:
        return None

    response_path = os.path.join(log_dir, f"{base_name}_response.txt")
    try:
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(raw)
    except OSError as exc:
        print(f"[LOG] could not write response log: {exc}")

    print(f"[AI] response: {len(raw)} chars")
    return raw


# ══════════════════════════════════════════════════════════════════
# Response parsing
# ══════════════════════════════════════════════════════════════════

def parse_ai_json_response(raw: str, keys: tuple[str, ...]) -> dict:
    """
    Defensively parse the AI's reply into a dict containing exactly `keys`,
    each mapped to a list (defaulting to [] when absent).

    `keys` lets the same parser serve both callers:
      - book_translator.py wants  ("translations", "glossary", "remarks")
      - glossary_builder.py wants ("glossary", "remarks")  [or just
        ("remarks",) in --check-only mode]

    Strategy, in order:
      1. Strip any ``` markdown code fences the model added despite instructions.
      2. Try a straight json.loads() on the outermost {...} span. This
         succeeds for the vast majority of well-formed responses.
      3. If that fails (e.g. the response got cut off mid-array because it
         hit max_output_tokens), fall back to manually scanning for each
         `"key": [` marker and re-parsing individual `{...}` objects inside
         that array one at a time. This salvages every complete item even
         when the array as a whole is truncated or has one broken entry,
         instead of throwing the whole chunk away.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^\s*```[a-zA-Z]*\s*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?\s*```\s*$",           "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start:end + 1])
            for key in keys:
                obj.setdefault(key, [])
            return obj
        except json.JSONDecodeError:
            pass

    obj = {key: [] for key in keys}
    for key in keys:
        m = re.search(rf'"{key}"\s*:\s*\[', cleaned)
        if not m:
            continue
        array_start = m.end() - 1
        depth, obj_start, items = 0, None, []
        for i, ch in enumerate(cleaned[array_start:], start=array_start):
            if ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        items.append(json.loads(cleaned[obj_start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
            elif ch == "]" and depth == 0:
                break
        obj[key] = items

    return obj