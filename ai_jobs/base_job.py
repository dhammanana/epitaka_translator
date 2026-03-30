"""
jobs/base_job.py — Abstract base for all AI jobs.

Subclasses must implement `run()`. They get:
  - self.task_id      — row id in task_queue
  - self.params       — dict from job_params JSON
  - self.provider     — AIProvider instance (first key; kept for back-compat)
  - self.log(msg)     — write to task_logs + Python logger
  - self.timeout      — seconds before a query is considered stale
  - self.ask_ai(...)  — calls AI with automatic key rotation across the pool

Key pool behaviour
------------------
  - On construction, BaseJob loads ALL active, non-exhausted keys for the
    current provider (or the explicit list in params["key_ids"]).
  - ask_ai() tries each key in round-robin order.
  - If a key returns a rate-limit error it is marked exhausted in the DB and
    the next key in the pool is tried immediately — no task failure.
  - If every key in the pool is exhausted, ask_ai() raises NoKeyAvailable so
    the worker can mark the task failed cleanly.
  - After each successful call the key's total_calls counter is incremented.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from ai_provider import AIProvider, build_provider
from database import log_task, get_runner_conn, get_setting
from key_manager import (
    list_keys, mark_exhausted, mark_success, mark_error,
    NoKeyAvailable, _maybe_restore_keys,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Key pool — internal helper
# ─────────────────────────────────────────────────────────────────

class _KeyPool:
    """
    Round-robin pool of API keys for one provider/model pair.
    Thread-safe enough for a single task thread.
    """

    def __init__(self, provider_name: str, model: str,
                 timeout: int, allowed_key_ids: Optional[list] = None):
        self.provider_name   = provider_name
        self.model           = model
        self.timeout         = timeout
        self.allowed_key_ids = set(allowed_key_ids) if allowed_key_ids else None

        self._pool: list  = []   # list of key rows currently in pool
        self._index: int  = 0    # round-robin cursor
        self._reload()

    # ── Pool management ──────────────────────────────────────────

    def _reload(self):
        """Fetch all usable keys from DB (re-enables cooled-down ones first)."""
        _maybe_restore_keys(self.provider_name)
        rows = list_keys(self.provider_name)
        pool = [
            r for r in rows
            if r["is_active"]
            and not r.get("exhausted_at")
            and (self.allowed_key_ids is None or r["id"] in self.allowed_key_ids)
        ]
        self._pool  = pool
        self._index = 0

    def _remove_exhausted(self, key_id: int):
        """Drop a key from the in-memory pool after marking it exhausted."""
        self._pool = [k for k in self._pool if k["id"] != key_id]
        if self._index >= len(self._pool):
            self._index = 0

    def size(self) -> int:
        return len(self._pool)

    # ── Main call ────────────────────────────────────────────────

    def call(self, prompt: str, system: str = "") -> tuple:
        """
        Try keys in round-robin order until one succeeds.
        Returns (response_text, key_id_used).
        Raises NoKeyAvailable if all keys are exhausted.
        """
        attempts = 0
        while self._pool:
            if attempts >= len(self._pool):
                # Tried every remaining key — attempt a reload in case
                # some cooldowns expired during a long-running job.
                self._reload()
                if not self._pool:
                    break
                attempts = 0

            key_row  = self._pool[self._index % len(self._pool)]
            self._index = (self._index + 1) % len(self._pool)
            key_id   = key_row["id"]
            attempts += 1

            provider = build_provider(self.provider_name, key_row["api_key"], self.model)

            try:
                result = provider.complete(prompt, system=system, timeout=self.timeout)
                mark_success(key_id)
                return result, key_id

            except Exception as exc:
                if provider.is_rate_limit_error(exc):
                    mark_exhausted(key_id)
                    self._remove_exhausted(key_id)
                    # loop → try next key
                else:
                    mark_error(key_id)
                    raise   # non-rate-limit errors bubble up immediately

        raise NoKeyAvailable(
            f"All API keys for provider '{self.provider_name}' are exhausted or inactive."
        )


# ─────────────────────────────────────────────────────────────────
# BaseJob
# ─────────────────────────────────────────────────────────────────

class BaseJob(ABC):
    # Human-readable name shown in the UI
    display_name: str = "Unnamed Job"
    # JSON schema for params (used to render the web form — optional)
    param_schema: dict = {}

    def __init__(self, task_id: int, params: dict, provider: AIProvider, timeout: int = 300):
        self.task_id  = task_id
        self.params   = params
        self.provider = provider   # kept for back-compat; pool is the real engine
        self.timeout  = timeout

        # ── Build the key pool ───────────────────────────────────
        # params["key_ids"] can be a list of ints to restrict which keys are used.
        # If absent/empty, all active keys for the provider are used.
        provider_name = get_setting("provider", "gemini")
        model         = get_setting("model",    "gemini-1.5-flash")

        allowed_key_ids = None
        raw_ids = params.get("key_ids")
        if raw_ids:
            try:
                # Accept either a list (JSON array) or a comma-separated string
                if isinstance(raw_ids, list):
                    allowed_key_ids = [int(i) for i in raw_ids if str(i).strip()]
                else:
                    allowed_key_ids = [int(i.strip()) for i in str(raw_ids).split(",") if i.strip()]
            except (TypeError, ValueError):
                pass

        self._pool = _KeyPool(
            provider_name   = provider_name,
            model           = model,
            timeout         = timeout,
            allowed_key_ids = allowed_key_ids,
        )

        self.log_info(
            f"Key pool initialised: {self._pool.size()} key(s) available "
            f"for {provider_name}/{model}"
            + (f" (restricted to ids={allowed_key_ids})" if allowed_key_ids else "")
        )

    # ── Logging helpers ──────────────────────────────────────────

    def log(self, message: str, level: str = "INFO"):
        log_task(self.task_id, message, level)

    def log_info(self, msg: str):
        self.log(msg, "INFO")

    def log_sucess(self, msg: str):
        self.log(msg, "SUCESS")

    def log_warn(self, msg: str):
        self.log(msg, "WARNING")

    def log_error(self, msg: str):
        self.log(msg, "ERROR")

    def log_debug(self, msg: str):
        self.log(msg, "DEBUG")

    # ── Heartbeat (feeds the watchdog in app.py) ─────────────────

    def heartbeat(self):
        """Touch heartbeat_at so the stale-task watchdog stays calm."""
        try:
            with get_runner_conn() as conn:
                conn.execute(
                    "UPDATE task_queue SET heartbeat_at=? WHERE id=?",
                    (time.time(), self.task_id)
                )
        except Exception:
            pass   # never let a heartbeat failure kill a job

    # ── AI call wrapper (pool rotation) ──────────────────────────

    def ask_ai(self, prompt: str, system: str = "") -> str:
        """
        Send a prompt to the AI, rotating through the key pool transparently
        on rate-limit errors.  Also updates the heartbeat before each call.
        """
        self.heartbeat()
        self.log_debug(
            f"Sending prompt ({len(prompt)} chars) — "
            f"pool has {self._pool.size()} active key(s)."
        )

        result, key_id = self._pool.call(prompt, system=system)

        self.log_debug(f"AI responded ({len(result)} chars) via key_id={key_id}.")

        # Keep the task row pointing at the last-used key for visibility in the UI
        try:
            with get_runner_conn() as conn:
                conn.execute(
                    "UPDATE task_queue SET api_key_id=? WHERE id=?",
                    (key_id, self.task_id)
                )
        except Exception:
            pass

        return result

    # ── Entry point ──────────────────────────────────────────────

    @abstractmethod
    def run(self) -> Any:
        """
        Execute the job. Should call self.log_*() throughout.
        Return value is stored nowhere — use DB side-effects.
        Raise an exception to mark the task failed.
        """