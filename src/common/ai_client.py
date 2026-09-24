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
def _key_state_file(model: str) -> str:
    """
    Per-model key-state JSON path, e.g. gemini_key_rate_state_gemini-2.5-pro.json.
    Keeps rpm/tpm/dead-key tracking separate per model (each model has its
    own quota). Override the base path with AI_KEY_STATE_FILE if needed.
    """
    safe_model = re.sub(r"[^\w\-.]", "_", model or "default")
    base = os.environ.get("AI_KEY_STATE_FILE")
    if base:
        root, ext = os.path.splitext(base)
        return f"{root}_{safe_model}{ext or '.json'}"
    return os.path.join(
        tempfile.gettempdir(), f"gemini_key_rate_state_{safe_model}.json"
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
        labels:    dict[str, str] | None = None,
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
        # 429 strikes are tracked per (model, key) — NOT globally — because
        # a key's quota is exhausted per model-tier, not for every model at
        # once. A key 429'd 3x on gemini-3.1-flash-lite may still be fine on
        # gemini-3-flash-preview. In-memory only (per process), same as the
        # old ModelPool strike counter it replaces.
        self._strikes429:     dict[str, dict[str, int]] = {}  # model -> {kid: count}
        self._dead_for_model: dict[str, set[str]]       = {}  # model -> {kid, ...}
        # key -> human-readable label (e.g. "GEMINI_KEY_23"), so log lines
        # and the state file can say *which* configured key something is
        # about, instead of only an opaque hash. Falls back to a 1-based
        # position label when no explicit labels are given (e.g. --api-keys).
        self._labels = dict(labels) if labels else {
            k: f"key#{i+1}" for i, k in enumerate(self._keys)
        }
        log.info(
            f"Loaded {len(self._keys)} Gemini key(s): "
            f"{', '.join(self._labels.get(k, '?') for k in self._keys)}. "
            f"Budget: {rpm_limit} rpm / {tpm_limit} tpm per key."
        )

    def _label(self, key: str) -> str:
        return self._labels.get(key, _key_id(key))

    def remaining_key_count(self) -> int:
        """How many keys are left in the pool at all (any model)."""
        with self._lock:
            return len(self._keys)

    def mark_429(self, key: str, model: str) -> bool:
        """
        Record one HTTP 429 for `key` on `model`. After 3 strikes on that
        SAME key for that SAME model, the key is parked for that model only
        (it stays available for other models) and this returns True. The
        key keeps rotating normally in the meantime — a single 429 doesn't
        remove anything, it just counts a strike for whichever key actually
        got it.
        """
        kid    = _key_id(key)
        label  = self._label(key)
        counts = self._strikes429.setdefault(model, {})
        counts[kid] = counts.get(kid, 0) + 1
        n = counts[kid]
        if n >= 3:
            self._dead_for_model.setdefault(model, set()).add(kid)
            log.warning(
                f"[Gemini] {label} hit 3x 429 on {model} — parking it for "
                f"this model (still usable on other models / other keys "
                f"keep rotating normally)."
            )
            return True
        log.info(f"[Gemini] {label} 429 strike {n}/3 on {model}.")
        return False

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

    def acquire(self, estimated_tokens: int = 0, model: str = "default") -> str:
        """
        Block until some key in the pool has rpm/tpm budget for one more
        request of `estimated_tokens`, reserve that budget, and return the
        key. Loops trying every key (round-robin) before sleeping, so it
        picks up whichever key frees up soonest rather than always waiting
        on the same one.

        Before trying, syncs against any keys that OTHER processes have
        marked permanently dead (see `remove()` / `_sync_dead_keys`), so
        several scripts sharing one key pool don't keep hammering a key
        that's already known to be exhausted/invalid elsewhere.
        """
        self._sync_dead_keys(model)
        while True:
            with self._lock:
                if not self._keys:
                    raise AllKeysExhaustedError("All Gemini API keys exhausted.")
                dead_here     = self._dead_for_model.get(model, set())
                keys_snapshot = [k for k in self._keys if _key_id(k) not in dead_here]
                start_index   = self._index
            if not keys_snapshot:
                raise AllKeysExhaustedError(
                    f"All keys 429-parked for model {model} (keys still live "
                    f"for other models)."
                )

            key, wait_seconds = self._try_reserve(keys_snapshot, start_index, estimated_tokens, model)
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
            self._sync_dead_keys(model)

    def _try_reserve(
        self,
        keys:              list[str],
        start_index:       int,
        estimated_tokens:  int,
        model:             str = "default",
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

        with open(_key_state_file(model), "a+") as fh:
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
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())

    def remove(self, key: str, reason: str | None = None, model: str = "default"):
        """
        Permanently drop a key (e.g. after a persistent 429/401/403) so it's
        never handed out again — both from this process's own pool AND from
        KEY_STATE_FILE's shared "_dead_keys" list, so every other process
        sharing this key pool (e.g. a second script running at the same
        time) finds out and stops trying that key too, instead of each
        process independently re-discovering the same dead key the hard way.

        Full identifying info (label, the key itself, why, and when) is
        written to KEY_STATE_FILE["_dead_keys_info"] so you can later answer
        "which of my configured keys got removed, and why" just by reading
        the state file — not just an opaque hash with no history attached.
        """
        kid   = _key_id(key)
        label = self._label(key)
        now   = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        with self._lock:
            if key in self._keys:
                self._keys.remove(key)
                log.warning(
                    f"Key removed: {label} (id={kid}) — reason: {reason}. "
                    f"{len(self._keys)} remaining."
                )

        with open(_key_state_file(model), "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                state = self._read_state(fh)
                dead = set(state.get("_dead_keys", []))
                dead.add(kid)
                state["_dead_keys"] = sorted(dead)

                dead_info = state.setdefault("_dead_keys_info", {})
                dead_info[kid] = {
                    "label":      label,
                    "key":        key,
                    "reason":     reason,
                    "removed_at": now,
                }
                self._write_state(fh, state)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _sync_dead_keys(self, model: str = "default") -> None:
        """
        Pull the shared "_dead_keys" list from that model's key-state file
        and drop any matching keys from this process's own pool. Cheap (one
        locked read), called at the start of every `acquire()` wait-loop so
        a key another process just had rejected doesn't keep getting tried
        here too.
        """
        with open(_key_state_file(model), "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                state = self._read_state(fh)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

        dead = set(state.get("_dead_keys", []))
        if not dead:
            return
        with self._lock:
            still_alive = [k for k in self._keys if _key_id(k) not in dead]
            n_dropped = len(self._keys) - len(still_alive)
            if n_dropped:
                log.warning(
                    f"[Gemini] Dropping {n_dropped} key(s) marked dead by "
                    f"another process. {len(still_alive)} remaining."
                )
            self._keys = still_alive

    def record_result(self, key: str, success: bool, error: str | None = None, model: str = "default") -> None:
        """
        Record a completed call's outcome (success/failure) for `key` into
        KEY_STATE_FILE, under a "_stats" section separate from the rpm/tpm
        rolling-window events, plus a "_totals" section across all keys.
        This is purely informational (for humans reading the state file /
        debugging which keys are healthy) and does not affect throttling.
        """
        kid   = _key_id(key)
        label = self._label(key)
        now   = time.time()
        with open(_key_state_file(model), "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                state = self._read_state(fh)
                stats = state.setdefault("_stats", {})
                entry = stats.setdefault(kid, {
                    "label":           label,
                    "key":             key,
                    "success_count":   0,
                    "failure_count":   0,
                    "last_success_at": None,
                    "last_failure_at": None,
                    "last_error":      None,
                })
                entry["label"] = label  # keep in sync if labels change across runs
                if success:
                    entry["success_count"] += 1
                    entry["last_success_at"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(now)
                    )
                else:
                    entry["failure_count"] += 1
                    entry["last_failure_at"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(now)
                    )
                    if error:
                        entry["last_error"] = error[:300]

                totals = state.setdefault(
                    "_totals", {"success_count": 0, "failure_count": 0}
                )
                totals["success_count" if success else "failure_count"] += 1

                self._write_state(fh, state)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def make_rotator(api_keys: list[str]) -> KeyRotator:
    """
    Build a KeyRotator from an explicit --api-keys list, or, if that's empty,
    from every GEMINI_KEY_<N> environment variable that's set.

    When loading from env vars, each key is labeled with its env var name
    (e.g. "GEMINI_KEY_23"), so later logs/state ("Key removed: GEMINI_KEY_23
    ...") tell you exactly which of your configured keys it was, instead of
    just an opaque hash.
    """
    if api_keys:
        return KeyRotator(api_keys)
    env_items = [
        (k, v.strip())
        for k, v in os.environ.items()
        if re.match(r"^GEMINI_KEY_\d+$", k) and v.strip()
    ]
    # Sort by the numeric suffix so labels/logs come out in a sane order
    # (GEMINI_KEY_2 before GEMINI_KEY_10), not alphabetical/env-dict order.
    env_items.sort(key=lambda kv: int(kv[0].rsplit("_", 1)[1]))
    env_keys = [v for _, v in env_items]
    labels   = {v: k for k, v in env_items}
    return KeyRotator(env_keys, labels=labels)


# ══════════════════════════════════════════════════════════════════
# Gemini call
# ══════════════════════════════════════════════════════════════════

class ModelPool:
    """
    Ordered list of Gemini models for one run, with fallback when a model's
    quota is exhausted.

    The first enabled model is used for every call. 429 strikes are counted
    per (model, key) in KeyRotator (see `KeyRotator.mark_429`) — a single key
    getting parked doesn't touch the model, since other keys keep rotating
    normally. Only once EVERY key is parked for the current model (KeyRotator
    raises AllKeysExhaustedError for that model) does `_generate_with_retry`
    call `disable_current()` here, moving on to the next model in the list.
    Disabled models stay out for the rest of the run.
    """

    def __init__(self, models: list[str]):
        self._models = [m for m in models if m]
        self._disabled: set[str] = set()

    def current(self) -> str | None:
        """The first enabled model, or None if every model has been removed."""
        for m in self._models:
            if m not in self._disabled:
                return m
        return None

    def remaining(self) -> list[str]:
        return [m for m in self._models if m not in self._disabled]

    def is_disabled(self, model: str) -> bool:
        return model in self._disabled

    def disable_current(self) -> None:
        """Remove the current model from the pool (all its keys are 429-parked)."""
        m = self.current()
        if m is None:
            return
        self._disabled.add(m)
        rest = self.remaining()
        log.warning(
            f"[ModelPool] {m} — every key is 429-parked for it. Removing it "
            f"from the model list. Remaining: {', '.join(rest) if rest else 'NONE'}."
        )


def _fatal_all_keys_exhausted() -> None:
    """Alert and exit when every configured key has been permanently removed."""
    log.error("[Gemini] All API keys exhausted (rate-limited/invalid). Exiting.")
    send_telegram(
        "<b>AI run FATAL</b>\n"
        "All Gemini API keys are exhausted (rate-limited or invalid).\n"
        "The run has stopped — add fresh keys and restart."
    )
    sys.exit(1)


def _generate_with_retry(
    rotator:           KeyRotator,
    pool:              ModelPool,
    contents:          list,
    config:            object,
    estimated_tokens:  int,
    timeout:           int,
) -> tuple[object | None, str | None, str | None]:
    """
    One Gemini generateContent request with the standard retry/backoff loop:
    acquire a throttled key for the pool's current model, run the call in a
    background thread so a hung request can be abandoned after `timeout`,
    handle 429 / invalid-key / generic errors, retry up to 10 attempts.

    Model fallback: when the current model is 429'd enough times to be
    removed from the pool (see ModelPool), the next model in the pool is
    picked up on the next attempt automatically, with a fresh retry budget.

    Shared by plain text calls and the function-calling loop below.
    Returns (response, key, model) on success, or (None, None, None) after
    all retries fail.
    """
    from google import genai

    attempt = 0
    while attempt < 10:
        model = pool.current()
        if model is None:
            # Every model in the pool has been disabled — i.e. every key is
            # 429-parked on every model. This is just as fatal as
            # remaining_key_count()==0 below: there is no key/model
            # combination left to try, so the run cannot make progress.
            # Previously this just logged and returned None, which meant
            # every chunk from here on silently "failed" one at a time
            # while the script kept running to the end doing nothing —
            # it needs to stop loudly here instead.
            log.error("[Gemini] No models left in the pool (all keys 429-parked "
                      "on every model). Treating as fatal.")
            _fatal_all_keys_exhausted()
        attempt += 1

        try:
            key = rotator.acquire(estimated_tokens, model=model)
        except AllKeysExhaustedError:
            if rotator.remaining_key_count() == 0:
                # Truly nothing left — every key was permanently removed
                # (invalid/expired/blocked). Nothing a model switch can fix.
                _fatal_all_keys_exhausted()
            # Keys still exist, they're just all 429-parked for THIS model —
            # fall back to the next model in the pool instead of exiting.
            pool.disable_current()
            attempt = 0  # fresh retry budget for the next model
            continue

        result: dict = {"response": None, "error": None}

        def _call():
            try:
                client = genai.Client(api_key=key)
                r = client.models.generate_content(
                    model=model, contents=contents, config=config,
                )
                result["response"] = r
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            log.error(f"[Gemini] Timeout on attempt {attempt + 1} ({model})")
            rotator.record_result(key, success=False, error="timeout", model=model)
            continue

        if result["error"]:
            e = result["error"]
            err_str = str(e)
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            err_msg = f"HTTP {status}: {e}"
            rotator.record_result(key, success=False, error=err_msg, model=model)
            is_429 = status == 429 or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str.upper() or "RATE_LIMIT" in err_str.upper()
            if is_429:
                log.warning(f"[Gemini] Rate limited (429) on {model}, attempt {attempt + 1}.")

                # Record the 429 against the specific key + model.
                # KeyRotator owns 429 strike tracking.
                if rotator.mark_429(key, model):
                    attempt = 0  # this key/model reached 3 strikes
                else:
                    time.sleep(20)

                continue

            is_invalid_key = status in (400, 401, 403) and any(
                term in err_str.upper() for term in ("API_KEY_INVALID", "UNAUTHENTICATED", "PERMISSION_DENIED", "KEY_EXPIRED", "INVALID_API_KEY")
            )
            if is_invalid_key:
                rotator.remove(key, reason=err_msg, model=model)
                time.sleep(5)
                continue
            log.warning(f"[Gemini] Error attempt {attempt + 1} ({model}): {e}")
            time.sleep(20)
            continue

        if result["response"] is not None:
            rotator.record_result(key, success=True, model=model)
            return result["response"], key, model

    log.error("[Gemini] All retry attempts exhausted.")
    return None, None, None


def call_gemini(
    rotator:            KeyRotator,
    prompt:             str,
    system_prompt:      str,
    model:              str | None = None,
    models:             list[str] | None = None,
    max_output_tokens:  int = 65_536,
    timeout:            int = 300,
    used_model:         list | None = None,
) -> str | None:
    """
    Send one prompt to Gemini and return the raw response text, or None if
    every retry attempt failed.

    Models: pass a single `model` or an ordered `models` list (the first
    model is preferred; a model is removed from the pool after 3× HTTP 429
    and the next one takes over — see ModelPool). If `used_model` is given,
    the model that actually produced the response is appended to it.

    Behaviour:
      - Runs the actual network call in a background thread so a hung
        request can be abandoned after `timeout` seconds instead of blocking
        forever.
      - On HTTP 429 (rate limit): counts a strike for the current model
        (removed after 3), waits 20s, and retries — with the next key, or
        the next model if this one was removed.
      - On HTTP 401/403 or any other error: waits 20s and retries with the
        same rotation (does not remove the key — those are usually transient
        or auth hiccups, not proof the key itself is bad).
      - Retries up to 10 attempts total before giving up and returning None.
      - If the key pool is ever fully exhausted, sends a Telegram alert and
        exits the whole process — a translation run with zero usable keys
        should stop loudly, not silently produce empty output.
    """
    from google.genai import types as genai_types

    pool = ModelPool(models if models else ([model] if model else []))
    if pool.current() is None:
        log.error("[Gemini] call_gemini called with no usable models.")
        return None

    estimated_tokens = estimate_tokens(prompt) + estimate_tokens(system_prompt)
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_output_tokens,
    )
    contents = [genai_types.Content(
        role="user", parts=[genai_types.Part.from_text(text=prompt)],
    )]
    response, _, model_used = _generate_with_retry(
        rotator, pool, contents, config, estimated_tokens, timeout,
    )
    if response is None:
        return None
    if used_model is not None and model_used:
        used_model.append(model_used)
    return response.text


def call_gemini_with_tools(
    rotator:           KeyRotator,
    prompt:            str,
    system_prompt:     str,
    model:             str | None = None,
    models:            list[str] | None = None,
    tools:             list | None = None,
    tool_executor=None,  # callable (name, args_dict) -> dict (JSON-serialisable response)
    max_output_tokens: int = 65_536,
    timeout:           int = 300,
    max_rounds:        int = 12,
    log:               callable = lambda msg: None,
    used_model:        list | None = None,
) -> str | None:
    """
    Function-calling loop on top of _generate_with_retry.

    Sends `prompt` with `tools` declared; whenever the model answers with
    function calls, executes them through `tool_executor(name, args)` and feeds
    the results back as a follow-up user turn, then asks again. Repeats until
    the model returns plain text.

    `max_rounds` is the maximum number of TOOL rounds, not total requests:
    the final text always costs one more request, so a call uses at most
    `max_rounds + 1` requests (e.g. max_rounds=1 → initial request + one tool
    round where the model may issue MANY parallel calls, then the final
    answer). One extra tool round beyond the cap is tolerated so the results
    can be fed back and a final answer forced; if the model still keeps
    calling tools after that, the call gives up and returns None.

    Models: pass a single `model` or an ordered `models` list (first model
    preferred; removed after 3× 429, next takes over). If `used_model` is
    given, the model that produced the final text is appended to it.

    Each tool round is a separate API request (counts against rpm/tpm), so the
    tool description handed to the model should push it to fetch generously
    in one call rather than many small ones. `log` receives a line per round.
    Returns the final response text, or None on failure.
    """
    from google.genai import types as genai_types

    pool = ModelPool(models if models else ([model] if model else []))
    if pool.current() is None:
        log.error("[Gemini] call_gemini_with_tools called with no usable models.")
        return None

    contents = [genai_types.Content(
        role="user", parts=[genai_types.Part.from_text(text=prompt)],
    )]
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools or [],
        max_output_tokens=max_output_tokens,
    )
    estimated = estimate_tokens(prompt) + estimate_tokens(system_prompt)

    tool_rounds = 0
    while True:
        response, _, model_used = _generate_with_retry(
            rotator, pool, contents, config, estimated, timeout,
        )
        if response is None:
            return None

        fcs = getattr(response, "function_calls", None)
        if fcs:
            tool_rounds += 1
            if tool_rounds > max_rounds + 1:
                log(f"[tools] model kept calling tools past the {max_rounds}-round "
                    f"cap; forcing a final answer with tools disabled.")
                forced_config = genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[],
                    max_output_tokens=max_output_tokens,
                )
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=(
                        "You are out of tool rounds. Do not call any more tools "
                        "— write your final answer now using only the text and "
                        "tool results already provided above."
                    ))],
                ))
                response, _, model_used = _generate_with_retry(
                    rotator, pool, contents, forced_config, estimated, timeout,
                )
                if response is None:
                    return None
                try:
                    text = response.text
                except (ValueError, TypeError):
                    text = None
                if text:
                    if used_model is not None and model_used:
                        used_model.append(model_used)
                    return text
                log("[tools] forced final answer still returned no text. Giving up.")
                return None
            call_desc = ", ".join(
                f"{fc.name}({json.dumps(fc.args or {}, ensure_ascii=False)[:160]})"
                for fc in fcs
            )
            log(f"[tools] round {tool_rounds}/{max_rounds} ({model_used}): "
                f"{len(fcs)} call(s): {call_desc}")
            # Echo the model's function-call parts back VERBATIM — they carry
            # the thought_signature the API requires on round-tripped calls
            # (rebuilding via Part.from_function_call drops it and the API
            # rejects the request with 400 INVALID_ARGUMENT).
            cand = (response.candidates or [None])[0]
            model_parts = []
            if cand is not None and cand.content is not None:
                model_parts = [
                    p for p in cand.content.parts
                    if p.function_call is not None
                ]
            if not model_parts:
                model_parts = [
                    genai_types.Part.from_function_call(
                        name=fc.name, args=fc.args or {},
                    ) for fc in fcs
                ]
            contents.append(genai_types.Content(role="model", parts=model_parts))
            tool_parts = []
            for fc in fcs:
                out = tool_executor(fc.name, dict(fc.args or {}))
                tool_parts.append(genai_types.Part.from_function_response(
                    name=fc.name, response=out,
                ))
                estimated += estimate_tokens(json.dumps(out, ensure_ascii=False))
            contents.append(genai_types.Content(role="user", parts=tool_parts))
            continue

        try:
            text = response.text
        except (ValueError, TypeError):
            text = None
        if text:
            if used_model is not None and model_used:
                used_model.append(model_used)
            return text
        finish = getattr(
            getattr(getattr(response, "candidates", [None])[0] or None,
                    "finish_reason", None), None, None,
        )
        log(f"[tools] round {tool_rounds + 1}: no text and no function calls "
            f"(finish_reason={finish}). Stopping.")
        return None


def make_get_text_range_tool(book_ids: list[str], single_round: bool = False):
    """
    Function declaration for the `get_text_range` tool: lets the model fetch
    more lines (Pāli + English) from the given books when it needs context
    beyond what was included in the prompt. The description pushes it to fetch
    whole paragraphs / large ranges in ONE call, since each call costs a full
    API request (rpm/tpm budget).

    `single_round` (used by study_builder.py): tells the model it gets AT MOST
    ONE tool round, so it must issue every fetch it needs as PARALLEL calls in
    that single round — the model cannot fetch iteratively.
    """
    from google.genai import types as genai_types

    books = ", ".join(sorted(set(book_ids)))
    desc = (
        "Fetch additional lines of the source texts (Pāli with English "
        f"translation) for the books: {books}. Use it when you need to see "
        "more text than was provided — e.g. the passage a ṭīkā discussion "
        "refers to, surrounding context of a debate, or an earlier/later part "
        "of the text. Every line is labelled [book_id:para:line]. "
        "RATE LIMITS: each tool round costs a full API request, so fetch "
        "GENEROUSLY — whole paragraphs or large ranges per call, and issue "
        "EVERY call you need in a single response (parallel calls) rather "
        "than one at a time. Prefer a few large calls over many small ones."
    )
    if single_round:
        desc += (
            " You get AT MOST ONE tool round: make all the fetches you need "
            "now, in parallel, then after the results come back you must "
            "write the final answer — you will not get another round."
        )
    return genai_types.Tool(function_declarations=[
        genai_types.FunctionDeclaration(
            name="get_text_range",
            description=desc,
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "book_id": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description=(
                            f"A SINGLE book to read from — one of: {books}. "
                            "Never combine multiple book ids in one string; to "
                            "fetch from several books, issue one call per book "
                            "(in parallel, same round)."
                        ),
                    ),
                    "para_start": genai_types.Schema(
                        type=genai_types.Type.INTEGER,
                        description="First paragraph (para_id) of the range, inclusive.",
                    ),
                    "para_end": genai_types.Schema(
                        type=genai_types.Type.INTEGER,
                        description="Last paragraph of the range, inclusive. At most 200 paragraphs per call.",
                    ),
                    "line_start": genai_types.Schema(
                        type=genai_types.Type.INTEGER,
                        description="Optional first line within each paragraph (default 1).",
                    ),
                    "line_end": genai_types.Schema(
                        type=genai_types.Type.INTEGER,
                        description="Optional last line within each paragraph (default: all lines).",
                    ),
                },
                required=["book_id", "para_start", "para_end"],
            ),
        )
    ])


def call_ai_with_logging(
    rotator:           KeyRotator,
    prompt:            str,
    book_id:           str,
    chunk_id:          str,
    log_dir:           str,
    model:             str | None = None,
    system_prompt:     str = "",
    tools:             list | None = None,
    tool_executor=None,
    max_output_tokens: int = 65_536,
    max_tool_rounds:   int = 12,
    models:            list[str] | None = None,
    used_model:        list | None = None,
) -> str | None:
    """
    Thin wrapper around call_gemini() / call_gemini_with_tools() that also
    writes the exact prompt sent and the raw response received to `log_dir`,
    so any run can be audited or replayed later. Filenames are timestamped +
    book/chunk-tagged, e.g.:

        20260716_143012_Sp-i_p10-42_prompt.txt
        20260716_143012_Sp-i_p10-42_response.txt

    When `tools` and `tool_executor` are given, uses the function-calling
    loop instead of a single-shot call.

    `model` (single) or `models` (ordered list, first preferred, removed
    after 3× 429) select which Gemini model(s) to use; `used_model`, if
    given, receives the model that actually produced the response.

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
            if tools:
                f.write("\n\n=== TOOLS ===\n")
                f.write(repr(tools)[:4000])
    except OSError as exc:
        print(f"[LOG] could not write prompt log: {exc}")

    n_tokens = estimate_tokens(prompt)
    print(f"[AI] calling: book={book_id} chunk={chunk_id} "
          f"{len(prompt)} chars (~{n_tokens} tokens)"
          + (" [tools on]" if tools else ""))

    if tools and tool_executor:
        raw = call_gemini_with_tools(
            rotator, prompt, system_prompt, model=model, models=models,
            tools=tools, tool_executor=tool_executor,
            max_output_tokens=max_output_tokens, max_rounds=max_tool_rounds,
            log=lambda msg: print(f"[AI] {msg}"), used_model=used_model,
        )
    else:
        raw = call_gemini(rotator, prompt, system_prompt, model=model,
                          models=models, max_output_tokens=max_output_tokens,
                          used_model=used_model)
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
