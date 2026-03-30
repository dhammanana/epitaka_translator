"""
service_runner.py — Run Flask web + background worker together.

Usage:
    python service_runner.py

The script:
  1. Initialises the databases.
  2. Starts the background worker in a daemon thread.
  3. Starts the Flask dev server in the main thread.

Press Ctrl+C to stop both.
"""

import os
import sys
import time
import signal
import logging
import threading
from pathlib import Path

# ── Make sure project root is on sys.path ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Logging — console + file ──────────────────────────────────────
LOG_FILE = Path(__file__).parent / "logs" / "service.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ]
)

# Quieten noisy third-party loggers
for noisy in ("werkzeug", "urllib3", "httpcore", "httpx", "google"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("service_runner")

# ─────────────────────────────────────────────────────────────────
# Boot sequence
# ─────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("AI Runner — service starting")
    logger.info("=" * 60)

    # 1. Initialise DBs
    from database import init_databases, get_setting
    init_databases()
    logger.info("Databases ready.")

    # 2. Start worker thread
    max_concurrent = int(get_setting("max_concurrent", "1"))
    from worker import run_worker

    worker_thread = threading.Thread(
        target=run_worker,
        args=(max_concurrent,),
        name="worker-loop",   # NOT "worker-*" so it won't be counted as a task thread
        daemon=True,
    )
    worker_thread.start()
    logger.info("Background worker started (max_concurrent=%d).", max_concurrent)

    # 3. Give worker a moment to settle, then start Flask
    time.sleep(0.5)

    from config import PORT
    from app import app

    logger.info("Flask web UI starting on http://127.0.0.1:%d", PORT)
    logger.info("Log file: %s", LOG_FILE)
    logger.info("-" * 60)

    # Handle Ctrl+C cleanly — Flask's dev server swallows SIGINT internally,
    # so we register our own handler that calls os._exit after a short grace period.
    def _shutdown(sig, frame):
        logger.info("Shutdown requested (signal %d) — exiting.", sig)
        # Give log handlers a moment to flush
        time.sleep(0.3)
        os._exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        app.run(
            host="0.0.0.0",
            port=PORT,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    except SystemExit:
        pass
    finally:
        logger.info("Service stopped.")


if __name__ == "__main__":
    main()