"""
database.py — SQLite schema creation and shared helpers.

Tables
------
api_keys   — rotating API credentials with rate-limit tracking
task_queue — job instances with status / result
task_logs  — per-task log lines (streamed to web)
settings   — key/value store for web-configurable options
"""

import sqlite3
import logging
from pathlib import Path
from config import RUNNER_DB, GLOSSARY_DB

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Connection helpers
# ─────────────────────────────────────────────────────────────────

def get_runner_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(RUNNER_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_glossary_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(GLOSSARY_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ─────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────

RUNNER_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    provider      TEXT    NOT NULL DEFAULT 'gemini',
    alias         TEXT    NOT NULL,
    api_key       TEXT    NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 1,
    exhausted_at  REAL,
    total_calls   INTEGER NOT NULL DEFAULT 0,
    total_errors  INTEGER NOT NULL DEFAULT 0,
    created_at    REAL    NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS task_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type     TEXT    NOT NULL,
    job_params   TEXT    NOT NULL DEFAULT '{}',
    status       TEXT    NOT NULL DEFAULT 'pending',
    priority     INTEGER NOT NULL DEFAULT 5,
    api_key_id   INTEGER,
    started_at   REAL,
    finished_at  REAL,
    error_msg    TEXT,
    max_retries INTEGER  DEFAULT 0,
    retry_delay  INTEGER  DEFAULT 1800,   -- seconds; 1800 = 30 min
    run_after    REAL,                    -- unix epoch; NULL = run immediately
    created_at   REAL    NOT NULL DEFAULT (strftime('%s','now')),
    FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
);

CREATE TABLE IF NOT EXISTS task_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    INTEGER NOT NULL,
    level      TEXT    NOT NULL DEFAULT 'INFO',
    message    TEXT    NOT NULL,
    ts         REAL    NOT NULL DEFAULT (strftime('%s','now')),
    FOREIGN KEY (task_id) REFERENCES task_queue(id)
);

CREATE INDEX IF NOT EXISTS idx_task_logs_task ON task_logs(task_id);
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
"""

GLOSSARY_SCHEMA = """
CREATE TABLE IF NOT EXISTS glossary (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    pali       TEXT    NOT NULL,
    english    TEXT    NOT NULL,
    domain     TEXT,
    sub_domain TEXT,
    context    TEXT,
    note       TEXT,
    source_id  TEXT,
    created_at REAL    NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE(pali, english)
);
CREATE INDEX IF NOT EXISTS idx_glossary_pali ON glossary(pali);
"""


def init_databases():
    """Create all tables on first run."""
    logger.info("Initialising runner DB at %s", RUNNER_DB)
    with get_runner_conn() as conn:
        conn.executescript(RUNNER_SCHEMA)

    logger.info("Initialising glossary DB at %s", GLOSSARY_DB)
    with get_glossary_conn() as conn:
        conn.executescript(GLOSSARY_SCHEMA)

    _seed_default_settings()
    logger.info("Database init complete.")


def _seed_default_settings():
    """Write defaults only when the key doesn't exist yet."""
    defaults = {
        "provider":       "gemini",
        "model":          "gemini-2.5-flash",
        "source_db":      "",          # path to epitaka.db
        "query_timeout":  "300",       # seconds
        "worker_poll":    "5",
        "max_concurrent": "1",
    }
    with get_runner_conn() as conn:
        for k, v in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)",
                (k, v)
            )


# ─────────────────────────────────────────────────────────────────
# Settings helpers
# ─────────────────────────────────────────────────────────────────

def get_setting(key: str, default=None):
    with get_runner_conn() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str):
    with get_runner_conn() as conn:
        conn.execute(
            """INSERT INTO settings(key, value, updated_at)
               VALUES (?, ?, strftime('%s','now'))
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                                              updated_at=strftime('%s','now')""",
            (key, value)
        )


# ─────────────────────────────────────────────────────────────────
# Task helpers
# ─────────────────────────────────────────────────────────────────

# def create_task(job_type: str, job_params: dict, priority: int = 5) -> int:
#     import json
#     with get_runner_conn() as conn:
#         cur = conn.execute(
#             "INSERT INTO task_queue(job_type, job_params, priority) VALUES (?,?,?)",
#             (job_type, json.dumps(job_params), priority)
#         )
#         return cur.lastrowid

def create_task(job_type, params, priority=5, max_retries=0, retry_delay=1800, run_after:   int  = None,):
    import json, time
    with get_runner_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO task_queue
                       (job_type, job_params, priority,
                        max_retries, retry_delay, run_after)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    job_type,
                    json.dumps(params),
                    priority,
                    max_retries,
                    retry_delay,
                    run_after,
                )
        )
        tid = cur.lastrowid
        conn.commit()
        return tid


def log_task(task_id: int, message: str, level: str = "INFO"):
    """Write a log line for a task (also echoes to Python logger)."""
    python_level = getattr(logging, level, logging.INFO)
    logger.log(python_level, "[task %d] %s", task_id, message)
    with get_runner_conn() as conn:
        conn.execute(
            "INSERT INTO task_logs(task_id, level, message) VALUES (?,?,?)",
            (task_id, level, message)
        )


def update_task_status(task_id: int, status: str, error_msg: str = None):
    finished = status in ("done", "failed", "cancelled")
    with get_runner_conn() as conn:
        if finished:
            conn.execute(
                """UPDATE task_queue SET status=?, finished_at=strftime('%s','now'),
                   error_msg=? WHERE id=?""",
                (status, error_msg, task_id)
            )
        else:
            conn.execute(
                "UPDATE task_queue SET status=? WHERE id=?",
                (status, task_id)
            )