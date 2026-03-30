"""
app.py — Flask control panel for AI Runner.

Blueprints
----------
  /            — Dashboard (task list, status)
  /tasks/*     — Task management (create, cancel)
  /keys/*      — API key CRUD
  /settings/*  — Provider/model/path settings
  /logs/*      — Streaming task logs
  /glossary/*  — Glossary viewer
"""

import json
import time
import logging
import threading
import bcrypt
from functools import wraps
from flask import (Flask, render_template, request, redirect, session,
                   url_for, flash, jsonify, Response, stream_with_context)

import config
from database import (init_databases, get_runner_conn, get_glossary_conn,
                      create_task, update_task_status, get_setting, set_setting)
from key_manager import (list_keys, add_key, delete_key,
                         toggle_key, reset_exhausted)
from ai_jobs import JOB_REGISTRY
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# ─────────────────────────────────────────────────────────────────
# Auth
# Users: add/remove entries here. Generate a hash with:
#   python3 -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"
# ─────────────────────────────────────────────────────────────────

users_json = os.getenv('APP_USERS', '{}')
USERS = json.loads(users_json)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").encode()
        stored   = USERS.get(username)
        if stored and bcrypt.checkpw(password, stored.encode()):
            session["user"] = username
            next_url = request.args.get("next") or url_for("dashboard")
            return redirect(next_url)
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ─────────────────────────────────────────────────────────────────
# Init on first request
# ─────────────────────────────────────────────────────────────────

_db_ready = False

# ─────────────────────────────────────────────────────────────────
# Stale-task watchdog
# ─────────────────────────────────────────────────────────────────

_STALE_THRESHOLD = 300   # seconds with no heartbeat → mark failed
_watchdog_started = False

def _watchdog_loop():
    """
    Background thread: every 60 s, find tasks stuck in 'running' whose
    last heartbeat (or started_at) is older than _STALE_THRESHOLD and
    mark them 'failed'.  Handles worker crashes / OOM kills that leave
    status dangling.
    """
    import time as _time
    while True:
        _time.sleep(60)
        try:
            conn = get_runner_conn()
            now  = _time.time()
            cutoff = now - _STALE_THRESHOLD
            # heartbeat_at falls back to started_at when NULL
            stale = conn.execute(
                """
                SELECT id FROM task_queue
                WHERE status = 'running'
                  AND COALESCE(heartbeat_at, started_at) < ?
                """,
                (cutoff,)
            ).fetchall()
            for row in stale:
                tid = row["id"]
                conn.execute(
                    """
                    UPDATE task_queue
                       SET status      = 'failed',
                           error_msg   = 'Watchdog: no heartbeat — worker may have crashed',
                           finished_at = ?
                     WHERE id = ?
                    """,
                    (now, tid)
                )
                logger.warning("Watchdog marked task %d as failed (stale).", tid)
            if stale:
                conn.commit()
            conn.close()
        except Exception as exc:
            logger.exception("Watchdog error: %s", exc)


@app.before_request
def _ensure_db():
    global _db_ready, _watchdog_started
    if not _db_ready:
        init_databases()
        # Migrate: add new columns to existing DBs (each ALTER is idempotent via try/except)
        for alter_sql in (
            "ALTER TABLE task_queue ADD COLUMN heartbeat_at REAL",
            "ALTER TABLE task_queue ADD COLUMN max_retries INTEGER DEFAULT 0",
            "ALTER TABLE task_queue ADD COLUMN retry_delay INTEGER DEFAULT 1800",
            "ALTER TABLE task_queue ADD COLUMN run_after REAL",
        ):
            try:
                conn = get_runner_conn()
                conn.execute(alter_sql)
                conn.commit()
                conn.close()
            except Exception:
                pass  # column already exists — fine
        _db_ready = True
        t = threading.Thread(target=_watchdog_loop, daemon=True, name="task-watchdog")
        t.start()
        _watchdog_started = True
        logger.info("Task watchdog started (stale threshold=%ds).", _STALE_THRESHOLD)


@app.template_filter('ctime')
def timectime(s):
    return datetime.fromtimestamp(float(s)).strftime('%Y-%m-%d %H:%M:%S')

# ─────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def dashboard():
    conn  = get_runner_conn()
    tasks = conn.execute(
        "SELECT * FROM task_queue ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    keys  = conn.execute("SELECT * FROM api_keys ORDER BY provider, id").fetchall()
    conn.close()

    g_conn = get_glossary_conn()
    gloss_count = g_conn.execute("SELECT COUNT(*) FROM glossary").fetchone()[0]
    g_conn.close()

    return render_template("dashboard.html",
                           tasks=tasks,
                           keys=keys,
                           gloss_count=gloss_count,
                           job_types=list(JOB_REGISTRY.keys()))


# ─────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────

@app.route("/tasks/create", methods=["GET", "POST"])
@login_required
def task_create():
    if request.method == "POST":
        job_type   = request.form.get("job_type", "")
        priority   = int(request.form.get("priority", 5))
        raw_params = request.form.get("params_json", "{}")
        try:
            params = json.loads(raw_params)
        except json.JSONDecodeError as e:
            flash(f"Invalid JSON params: {e}", "error")
            return redirect(url_for("task_create"))

        max_retries = int(request.form.get("max_retries", 0))
        retry_delay = int(request.form.get("retry_delay", 7200))
        task_id = create_task(job_type, params, priority,
                              max_retries=max_retries, retry_delay=retry_delay)
        flash(f"Task #{task_id} created.", "success")
        logger.info("Task created via web: id=%d type=%s", task_id, job_type)
        return redirect(url_for("task_detail", task_id=task_id))

    job_schemas = {k: v.param_schema for k, v in JOB_REGISTRY.items()}
    return render_template("task_create.html",
                           job_types=list(JOB_REGISTRY.keys()),
                           job_schemas=job_schemas)


@app.route("/tasks/<int:task_id>")
@login_required
def task_detail(task_id):
    conn = get_runner_conn()
    task = conn.execute(
        "SELECT t.*, k.alias as key_alias FROM task_queue t "
        "LEFT JOIN api_keys k ON t.api_key_id=k.id WHERE t.id=?",
        (task_id,)
    ).fetchone()
    logs = conn.execute(
        "SELECT * FROM task_logs WHERE task_id=? ORDER BY ts ASC",
        (task_id,)
    ).fetchall()
    conn.close()

    if not task:
        flash(f"Task #{task_id} not found.", "error")
        return redirect(url_for("dashboard"))

    return render_template("task_detail.html", task=task, logs=logs)


@app.route("/tasks/<int:task_id>/cancel", methods=["POST"])
@login_required
def task_cancel(task_id):
    conn = get_runner_conn()
    task = conn.execute(
        "SELECT status FROM task_queue WHERE id=?", (task_id,)
    ).fetchone()
    conn.close()

    if task and task["status"] in ("pending", "running"):
        update_task_status(task_id, "cancelled")
        flash(f"Task #{task_id} cancelled.", "info")
        logger.info("Task %d cancelled via web.", task_id)
    else:
        flash("Cannot cancel — task is not pending/running.", "warning")

    return redirect(url_for("task_detail", task_id=task_id))


@app.route("/tasks/<int:task_id>/delete", methods=["POST"])
@login_required
def task_delete(task_id):
    conn = get_runner_conn()
    task = conn.execute(
        "SELECT status FROM task_queue WHERE id=?", (task_id,)
    ).fetchone()

    if not task:
        conn.close()
        flash("Task not found.", "error")
        return redirect(url_for("dashboard"))

    if task["status"] in ("pending", "running"):
        conn.close()
        flash("Cannot delete a pending/running task — cancel it first.", "warning")
        return redirect(url_for("task_detail", task_id=task_id))

    conn.execute("DELETE FROM task_logs WHERE task_id=?", (task_id,))
    conn.execute("DELETE FROM task_queue WHERE id=?", (task_id,))
    conn.commit()
    conn.close()
    flash(f"Task #{task_id} deleted.", "info")
    logger.info("Task %d deleted via web.", task_id)
    return redirect(url_for("dashboard"))


@app.route("/tasks/<int:task_id>/retry", methods=["POST"])
@login_required
def task_retry(task_id):
    conn = get_runner_conn()
    task = conn.execute(
        "SELECT * FROM task_queue WHERE id=?", (task_id,)
    ).fetchone()
    conn.close()

    if not task:
        flash("Task not found.", "error")
        return redirect(url_for("dashboard"))

    new_id = create_task(task["job_type"], json.loads(task["job_params"]), task["priority"])
    flash(f"Retried as Task #{new_id}.", "success")
    return redirect(url_for("task_detail", task_id=new_id))


# ── Streaming log endpoint (SSE) ─────────────────────────────────

@app.route("/tasks/<int:task_id>/stream")
@login_required
def task_log_stream(task_id):
    """
    Server-Sent Events stream: pushes new log lines as they appear.
    The browser JS reconnects automatically.
    """
    def generate():
        last_id = 0
        conn    = get_runner_conn()
        consecutive_empty = 0

        while True:
            rows = conn.execute(
                "SELECT * FROM task_logs WHERE task_id=? AND id>? ORDER BY id",
                (task_id, last_id)
            ).fetchall()

            for row in rows:
                last_id = row["id"]
                data    = json.dumps({
                    "id":      row["id"],
                    "level":   row["level"],
                    "message": row["message"],
                    "ts":      row["ts"],
                })
                yield f"data: {data}\n\n"
                consecutive_empty = 0

            # Check if task is finished
            task = conn.execute(
                "SELECT status FROM task_queue WHERE id=?", (task_id,)
            ).fetchone()
            if task and task["status"] in ("done", "failed", "cancelled"):
                yield "data: {\"__done__\": true}\n\n"
                break

            consecutive_empty += 1
            if consecutive_empty > 360:   # 30 min safety
                break
            time.sleep(5)

        conn.close()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── JSON API for dashboard polling ───────────────────────────────

@app.route("/api/tasks")
@login_required
def api_tasks():
    conn  = get_runner_conn()
    tasks = conn.execute(
        "SELECT id, job_type, status, created_at, started_at, finished_at "
        "FROM task_queue ORDER BY created_at DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return jsonify([dict(t) for t in tasks])


@app.route("/api/tasks/<int:task_id>/logs")
@login_required
def api_task_logs(task_id):
    since = float(request.args.get("since_id", 0))
    conn  = get_runner_conn()
    rows  = conn.execute(
        "SELECT * FROM task_logs WHERE task_id=? AND id>? ORDER BY id",
        (task_id, since)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ─────────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────────

@app.route("/keys")
@login_required
def keys_list():
    keys = list_keys()
    return render_template("keys.html", keys=keys,
                           providers=list(config.AVAILABLE_MODELS.keys()))


@app.route("/keys/add", methods=["POST"])
@login_required
def key_add():
    provider = request.form.get("provider", "gemini")
    alias    = request.form.get("alias", "").strip()
    api_key  = request.form.get("api_key", "").strip()

    if not alias or not api_key:
        flash("Alias and API key are required.", "error")
        return redirect(url_for("keys_list"))

    kid = add_key(provider, alias, api_key)
    flash(f"API key '{alias}' added (id={kid}).", "success")
    return redirect(url_for("keys_list"))


@app.route("/keys/<int:key_id>/delete", methods=["POST"])
@login_required
def key_delete(key_id):
    delete_key(key_id)
    flash(f"Key #{key_id} deleted.", "info")
    return redirect(url_for("keys_list"))


@app.route("/keys/<int:key_id>/toggle", methods=["POST"])
@login_required
def key_toggle(key_id):
    active = request.form.get("active") == "1"
    toggle_key(key_id, active)
    return redirect(url_for("keys_list"))


@app.route("/keys/<int:key_id>/reset", methods=["POST"])
@login_required
def key_reset(key_id):
    reset_exhausted(key_id)
    flash(f"Key #{key_id} cooldown cleared.", "success")
    return redirect(url_for("keys_list"))


@app.route("/keys/reset_all", methods=["POST"])
@login_required
def key_reset_all():
    """Reset rate-limit cooldowns on every key (useful after model change)."""
    keys = list_keys()
    for k in keys:
        reset_exhausted(k["id"])
    flash(f"All {len(keys)} key cooldowns cleared.", "success")
    logger.info("All key cooldowns reset via web.")
    return redirect(url_for("keys_list"))


# ─────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        old_model = get_setting("model", "")
        for key in ("provider", "model", "source_db", "query_timeout",
                    "worker_poll", "max_concurrent",
                    "telegram_token", "telegram_chat_id"):
            val = request.form.get(key, "").strip()
            if val:
                set_setting(key, val)
        new_model = get_setting("model", "")
        if new_model != old_model:
            keys = list_keys()
            for k in keys:
                reset_exhausted(k["id"])
            flash(f"Settings saved. Model changed → all {len(keys)} key cooldowns auto-reset.", "success")
            logger.info("Model changed (%s → %s): all key cooldowns reset.", old_model, new_model)
        else:
            flash("Settings saved.", "success")
        return redirect(url_for("settings"))

    current = {k: get_setting(k, "") for k in
               ("provider", "model", "source_db", "query_timeout",
                "worker_poll", "max_concurrent",
                "telegram_token", "telegram_chat_id")}
    return render_template("settings.html",
                           current=current,
                           available_models=config.AVAILABLE_MODELS)


# ─────────────────────────────────────────────────────────────────
# Glossary viewer
# ─────────────────────────────────────────────────────────────────

@app.route("/glossary")
@login_required
def glossary_view():
    page   = int(request.args.get("page", 1))
    search = request.args.get("q", "").strip()
    domain = request.args.get("domain", "").strip()
    limit  = 50
    offset = (page - 1) * limit

    g_conn = get_glossary_conn()

    where_clauses = []
    params        = []
    if search:
        where_clauses.append("(pali LIKE ? OR english LIKE ?)")
        params += [f"%{search}%", f"%{search}%"]
    if domain:
        where_clauses.append("domain=?")
        params.append(domain)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    total = g_conn.execute(
        f"SELECT COUNT(*) FROM glossary {where_sql}", params
    ).fetchone()[0]

    rows = g_conn.execute(
        f"SELECT * FROM glossary {where_sql} ORDER BY pali LIMIT ? OFFSET ?",
        params + [limit, offset]
    ).fetchall()

    domains = [r[0] for r in g_conn.execute(
        "SELECT DISTINCT domain FROM glossary WHERE domain!='' ORDER BY domain"
    ).fetchall()]
    g_conn.close()

    return render_template("glossary.html",
                           rows=rows, total=total,
                           page=page, limit=limit,
                           search=search, domain=domain,
                           domains=domains,
                           pages=((total - 1) // limit + 1) if total else 1)


if __name__ == "__main__":
    init_databases()
    app.run(debug=config.DEBUG, port=config.PORT, use_reloader=False)