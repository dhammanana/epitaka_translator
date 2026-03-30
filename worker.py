"""
worker.py — Background task worker.

- Polls task_queue for 'pending' tasks.
- Picks up the next task, instantiates the job class, runs it.
- Key rotation is handled inside BaseJob._pool — the worker no longer
  needs its own rotation loop.
- Marks task done/failed; each task runs in its own daemon thread.
- On failure caused by 503 / quota exhaustion the task is rescheduled
  as a new 'pending' task after retry_delay seconds (default 1800 = 30 min),
  respecting the max_retries counter stored in the task row.
- Sends a Telegram message when a task finishes (done or failed).
"""

import time
import json
import logging
import threading
import urllib.request
import urllib.parse

from config import WORKER_POLL_SEC, QUERY_TIMEOUT_SEC
from database import get_runner_conn, update_task_status, log_task, get_setting, create_task
from key_manager import get_next_key, NoKeyAvailable
from ai_provider import build_provider
from ai_jobs import get_job_class

logger = logging.getLogger(__name__)

# ── Patterns that indicate a transient server-side failure ────────
_RETRYABLE_PATTERNS = (
    "503", "unavailable", "resource_exhausted", "quota", "rate",
    "limit", "429", "exhausted", "overloaded", "service unavailable",
    "too many requests",
)

def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _RETRYABLE_PATTERNS)


# ── Telegram helper ───────────────────────────────────────────────

def _send_telegram(message: str) -> None:
    """
    Fire-and-forget Telegram notification.
    Reads telegram_token and telegram_chat_id from settings DB.
    Silently swallows all errors so it never breaks the worker.
    """
    try:
        token   = get_setting("telegram_token",   "").strip()
        chat_id = get_setting("telegram_chat_id", "").strip()
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
        logger.warning("Telegram notification failed: %s", exc)


def _notify(task_id: int, job_type: str, status: str, detail: str = "") -> None:
    icon = "✅" if status == "done" else "❌"
    lines = [f"{icon} <b>Task #{task_id}</b> [{job_type}] — <b>{status.upper()}</b>"]
    if detail:
        lines.append(detail)
    _send_telegram("\n".join(lines))

_shutdown_event = threading.Event()


def request_shutdown():
    """Called by service_runner to stop the worker loop."""
    _shutdown_event.set()


# ─────────────────────────────────────────────────────────────────
# Reschedule helper
# ─────────────────────────────────────────────────────────────────

def _reschedule_task(
    task_id:     int,
    job_type:    str,
    job_params:  dict,
    priority:    int,
    retries_left: int,
    retry_delay: int,
) -> int | None:
    """
    Create a new pending task that will be picked up after `retry_delay` seconds.
    Stores retry metadata in the params so it survives re-queueing.
    Returns the new task_id, or None on failure.
    """
    try:
        params = dict(job_params)
        params["_retries_left"] = retries_left - 1
        params["_retry_delay"]  = retry_delay
        # Schedule: set run_after epoch in DB via create_task
        new_id = create_task(
            job_type, params, priority,
            run_after=int(time.time()) + retry_delay,
        )
        logger.info(
            "Task %d rescheduled as task %d (retries_left=%d, delay=%ds).",
            task_id, new_id, retries_left - 1, retry_delay,
        )
        return new_id
    except Exception as exc:
        logger.error("Failed to reschedule task %d: %s", task_id, exc)
        return None


# ─────────────────────────────────────────────────────────────────
# Task executor (runs in its own thread)
# ─────────────────────────────────────────────────────────────────

def _run_task(task_id: int, job_type: str, job_params: dict):
    """
    Execute one task. Key rotation is delegated to BaseJob._pool.

    On transient failure (503 / quota exhausted):
      - If retries_left > 0: reschedule as new pending task after retry_delay.
      - Otherwise: mark failed and send Telegram notification.
    On success: mark done and send Telegram notification.
    """
    log_task(task_id, f"Worker picked up task — job_type={job_type}", "INFO")
    update_task_status(task_id, "running")

    with get_runner_conn() as conn:
        conn.execute(
            "UPDATE task_queue SET started_at=strftime('%s','now') WHERE id=?",
            (task_id,)
        )
        row = conn.execute(
            "SELECT priority, max_retries, retry_delay FROM task_queue WHERE id=?",
            (task_id,)
        ).fetchone()

    priority     = row["priority"]     if row else 5
    # retries_left: prefer the runtime counter stored in params, else the DB column
    retries_left = int(job_params.pop("_retries_left", None)
                       or (row["max_retries"] if row else 0))
    retry_delay  = int(job_params.pop("_retry_delay",  None)
                       or (row["retry_delay"] if row else 1800))

    # Resolve settings
    provider_name = get_setting("provider", "gemini")
    model         = get_setting("model",    "gemini-1.5-flash")
    timeout_sec   = int(get_setting("query_timeout", str(QUERY_TIMEOUT_SEC)))

    try:
        key_row  = get_next_key(provider_name)
        provider = build_provider(provider_name, key_row["api_key"], model)
    except NoKeyAvailable as exc:
        log_task(task_id, f"No API key available: {exc}", "ERROR")
        update_task_status(task_id, "failed", str(exc))
        _notify(task_id, job_type, "failed", f"No API key: {exc}")
        return
    except Exception as exc:
        log_task(task_id, f"Could not build provider: {exc}", "ERROR")
        update_task_status(task_id, "failed", str(exc))
        _notify(task_id, job_type, "failed", str(exc))
        return

    try:
        job_cls = get_job_class(job_type)
        job     = job_cls(task_id, job_params, provider, timeout=timeout_sec)
        job.run()

        log_task(task_id, "Task completed successfully.", "INFO")
        update_task_status(task_id, "done")
        _notify(task_id, job_type, "done")

    except (NoKeyAvailable, Exception) as exc:
        is_retryable = (
            isinstance(exc, NoKeyAvailable) or _is_retryable_error(exc)
        )

        if is_retryable and retries_left > 0:
            log_task(
                task_id,
                f"Transient failure ({exc}). "
                f"Rescheduling in {retry_delay}s "
                f"({retries_left} retr{'y' if retries_left == 1 else 'ies'} left).",
                "WARN",
            )
            update_task_status(
                task_id, "failed",
                f"Transient — rescheduled in {retry_delay}s. Error: {exc}",
            )
            new_id = _reschedule_task(
                task_id, job_type, job_params, priority, retries_left, retry_delay
            )
            detail = (
                f"Transient error: {exc}\n"
                f"Rescheduled as task #{new_id} in {retry_delay // 60} min."
                if new_id else f"Reschedule failed. Original error: {exc}"
            )
            _notify(task_id, job_type, "failed (rescheduled)", detail)
        else:
            level = "ERROR"
            if isinstance(exc, TimeoutError):
                msg = f"Task timed out: {exc}"
            elif isinstance(exc, NoKeyAvailable):
                msg = f"All API keys exhausted during run: {exc}"
            else:
                msg = f"Task failed: {exc}"

            log_task(task_id, msg, level)
            update_task_status(task_id, "failed", str(exc))
            _notify(task_id, job_type, "failed", msg)


# ─────────────────────────────────────────────────────────────────
# Poll loop
# ─────────────────────────────────────────────────────────────────

def _fetch_next_pending() -> dict | None:
    """Return the highest-priority pending task whose run_after time has passed."""
    try:
        with get_runner_conn() as conn:
            row = conn.execute(
                """SELECT * FROM task_queue
                   WHERE status='pending'
                     AND (run_after IS NULL OR run_after <= strftime('%s','now'))
                   ORDER BY priority ASC, created_at ASC
                   LIMIT 1"""
            ).fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Polling failed: {e}")
        return None

def _active_thread_count() -> int:
    active_threads = [t.name for t in threading.enumerate() if t.is_alive()]
    # This will print ALL threads to your log so you can see if 'task-32' is still there
    logger.debug(f"Current System Threads: {active_threads}")
    return sum(1 for name in active_threads if name.startswith("task-"))

def run_worker(max_concurrent: int = 2):
    """Main blocking loop. Call from service_runner.py."""
    logger.info("Worker started (poll_interval=%ds, max_concurrent=%d).",
                WORKER_POLL_SEC, max_concurrent)

    while not _shutdown_event.is_set():
        try:
            active = _active_thread_count()
            if active < max_concurrent:
                task = _fetch_next_pending()
                if task:
                    task_id    = task["id"]
                    job_type   = task["job_type"]
                    job_params = json.loads(task["job_params"] or "{}")

                    logger.info("Dispatching task id=%d type=%s", task_id, job_type)
                    t = threading.Thread(
                        target=_run_task,
                        args=(task_id, job_type, job_params),
                        name=f"task-{task_id}",
                        daemon=True,
                    )
                    t.start()
                else:
                    logger.debug("No pending tasks.")
            else:
                logger.debug("Max concurrent tasks reached (%d). Waiting…", active)

        except Exception as exc:
            logger.exception("Unexpected error in worker poll loop: %s", exc)

        _shutdown_event.wait(timeout=WORKER_POLL_SEC)

    logger.info("Worker shut down gracefully.")