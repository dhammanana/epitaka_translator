# AI Runner

A Flask-based control panel + background worker for running AI jobs in rotation
across multiple API keys. Built for E-Piṭaka Theravāda text processing.

---

## Quick Start

```bash
cd ai_runner
pip install -r requirements.txt

# 1. Point to your source DB
export SOURCE_DB=/path/to/epitaka.db   # or set it in the web Settings page

# 2. Start everything (Flask web + background worker)
python service_runner.py
```

Open **http://localhost:5555** in your browser.

---

## First-time Setup (web UI)

1. **Settings** → set `source_db` path to your `epitaka.db`
2. **API Keys** → add one or more Gemini keys
3. **New Task** → pick `glossary_builder`, adjust params, click Enqueue

---

## Project Structure

```
ai_runner/
├── service_runner.py   ← Entry point: run this to start everything
├── app.py              ← Flask web UI
├── worker.py           ← Background task worker (polling loop)
├── ai_provider.py      ← AI backend abstraction (Gemini / OpenAI / Anthropic)
├── key_manager.py      ← API key rotation + cooldown tracking
├── database.py         ← SQLite schema, helpers
├── config.py           ← All constants, paths, defaults
├── jobs/
│   ├── __init__.py         ← JOB_REGISTRY — add new jobs here
│   ├── base_job.py         ← BaseJob abstract class
│   └── glossary_builder.py ← GlossaryBuilder implementation
├── templates/          ← Jinja2 HTML templates
├── data/               ← SQLite databases (runner.db, glossary.db)
└── logs/               ← service.log
```

---

## Adding a New Job

1. Create `jobs/my_job.py`:

```python
from jobs.base_job import BaseJob

class MyJob(BaseJob):
    display_name = "My Job"
    param_schema = {
        "input_path": {"type": "string", "label": "Input path", "default": ""},
        "batch_size": {"type": "integer", "label": "Batch size", "default": 5},
    }

    def run(self):
        self.log_info("Starting MyJob")
        # ... do work, call self.ask_ai(prompt) for AI calls
        self.log_info("Done")
```

2. Register in `jobs/__init__.py`:

```python
from jobs.my_job import MyJob

JOB_REGISTRY = {
    "glossary_builder": GlossaryBuilderJob,
    "my_job": MyJob,         # ← add here
}
```

That's it — the web UI picks it up automatically.

---

## API Key Rotation

- Keys rotate round-robin (fewest calls first).
- A `429 / quota exhausted` error marks a key with `exhausted_at = now`.
- After **1 hour** the key is automatically restored.
- You can manually reset a key in **API Keys** → **Reset**.

---

## Timeout Handling

Gemini can silently hang. The worker enforces a **hard thread-level timeout**
(default 5 minutes, configurable in Settings). A stalled query is abandoned
and the task is marked `failed`. The next task in the queue then runs.

---

## Databases

| File | Contents |
|------|----------|
| `data/runner.db` | API keys, task queue, task logs, settings |
| `data/glossary.db` | Built glossary (pali, english, domain, ...) |

---

## Planned Jobs (add when needed)

| Job type | Description |
|----------|-------------|
| `glossary_builder` | ✅ Extract Pāli/English glossary from suttas |
| `translator` | Translate Pāli text with AI |
| `line_splitter` | Split sutta into lines aligned with Pāli |
| `term_validator` | Cross-check glossary entries |
