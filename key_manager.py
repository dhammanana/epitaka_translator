"""
key_manager.py — Rotating API key pool with cooldown tracking.

Rules
-----
- Keys are tried in round-robin order.
- If a key hits a rate-limit error it is marked exhausted_at = now.
- Exhausted keys are re-enabled after API_COOLDOWN_SEC (default 1 h).
- get_next_key() raises NoKeyAvailable if all keys are exhausted/inactive.
"""

import time
import logging
import threading
from typing import Optional

from config import API_COOLDOWN_SEC
from database import get_runner_conn

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class NoKeyAvailable(Exception):
    pass


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def get_next_key(provider: str) -> dict:
    """
    Return the next usable api_key row for *provider*.
    Raises NoKeyAvailable if none are ready.
    """
    with _lock:
        _maybe_restore_keys(provider)
        conn = get_runner_conn()
        row  = conn.execute(
            """SELECT * FROM api_keys
               WHERE provider=? AND is_active=1 AND exhausted_at IS NULL
               ORDER BY total_calls ASC, id ASC
               LIMIT 1""",
            (provider,)
        ).fetchone()
        conn.close()

    if not row:
        raise NoKeyAvailable(
            f"No active API keys for provider '{provider}'. "
            f"Add keys in Settings or wait for cooldown."
        )
    logger.debug("Using API key id=%d alias=%s", row["id"], row["alias"])
    return dict(row)


def mark_exhausted(key_id: int):
    """Call when the provider returns a rate-limit error."""
    logger.warning("API key id=%d marked exhausted — cooldown %ds", key_id, API_COOLDOWN_SEC)
    with get_runner_conn() as conn:
        conn.execute(
            "UPDATE api_keys SET exhausted_at=?, total_errors=total_errors+1 WHERE id=?",
            (time.time(), key_id)
        )


def mark_success(key_id: int):
    """Increment call counter on success."""
    with get_runner_conn() as conn:
        conn.execute(
            "UPDATE api_keys SET total_calls=total_calls+1 WHERE id=?",
            (key_id,)
        )


def mark_error(key_id: int):
    with get_runner_conn() as conn:
        conn.execute(
            "UPDATE api_keys SET total_errors=total_errors+1 WHERE id=?",
            (key_id,)
        )


def _maybe_restore_keys(provider: str):
    """Un-exhaust keys whose cooldown period has passed."""
    cutoff = time.time() - API_COOLDOWN_SEC
    with get_runner_conn() as conn:
        restored = conn.execute(
            """UPDATE api_keys SET exhausted_at=NULL
               WHERE provider=? AND exhausted_at IS NOT NULL AND exhausted_at < ?""",
            (provider, cutoff)
        ).rowcount
    if restored:
        logger.info("Restored %d exhausted key(s) for provider '%s'", restored, provider)


# ─────────────────────────────────────────────────────────────────
# CRUD helpers (used by Flask routes)
# ─────────────────────────────────────────────────────────────────

def list_keys(provider: str = None) -> list[dict]:
    conn = get_runner_conn()
    if provider:
        rows = conn.execute(
            "SELECT * FROM api_keys WHERE provider=? ORDER BY id", (provider,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM api_keys ORDER BY provider, id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_key(provider: str, alias: str, api_key: str) -> int:
    with get_runner_conn() as conn:
        cur = conn.execute(
            "INSERT INTO api_keys(provider, alias, api_key) VALUES (?,?,?)",
            (provider, alias, api_key)
        )
        logger.info("Added API key id=%d alias=%s provider=%s", cur.lastrowid, alias, provider)
        return cur.lastrowid


def delete_key(key_id: int):
    with get_runner_conn() as conn:
        conn.execute("DELETE FROM api_keys WHERE id=?", (key_id,))
    logger.info("Deleted API key id=%d", key_id)


def toggle_key(key_id: int, active: bool):
    with get_runner_conn() as conn:
        conn.execute(
            "UPDATE api_keys SET is_active=? WHERE id=?",
            (1 if active else 0, key_id)
        )


def reset_exhausted(key_id: int):
    """Manually clear cooldown from the UI."""
    with get_runner_conn() as conn:
        conn.execute("UPDATE api_keys SET exhausted_at=NULL WHERE id=?", (key_id,))
    logger.info("Manually reset exhausted state for key id=%d", key_id)
