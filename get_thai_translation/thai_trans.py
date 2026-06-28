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

    - Keys are removed permanently when they hit 429 RESOURCE_EXHAUSTED.
    - When the pool is empty, next() raises AllKeysExhaustedError.
    """

    def __init__(self):
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


ROTATOR: Optional[KeyRotator] = None
MODEL_NAME: str = "gemini-2.5-flash-preview-05-20"  # overridden by --model arg
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

def call_gemini(prompt: str, timeout_seconds: int = 300) -> str:
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
                    model=MODEL_NAME,
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


def ensure_output_db(path: str) -> sqlite3.Connection:
    """Create epitaka_th.db with same schema as nissaya.sentences."""
    con = open_db(path)

    con.executescript("""
        CREATE TABLE IF NOT EXISTS sentences (
            book_id          TEXT    NOT NULL,
            para_id          INTEGER NOT NULL,
            line_id          INTEGER NOT NULL,
            pali_sentence    TEXT,
            thai_translation TEXT,
            PRIMARY KEY (book_id, para_id, line_id)
        );

        CREATE TABLE IF NOT EXISTS _progress (
            matching_rowid INTEGER PRIMARY KEY,
            done           INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS _chunk_progress (
            matching_rowid INTEGER NOT NULL,
            chunk_index    INTEGER NOT NULL,
            done           INTEGER DEFAULT 0,
            PRIMARY KEY (matching_rowid, chunk_index)
        );
    """)
    con.commit()
    return con


# ─────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────

def get_thai_text(thaimm_con: sqlite3.Connection,
                  volume_id: str,
                  page_start: int,
                  page_end: int) -> str:
    """Concatenate content rows from thaimm.main for the given volume & page range."""
    # volume stored as zero-padded string e.g. "01"
    vol_str = str(volume_id).zfill(2)
    rows = thaimm_con.execute(
        """
        SELECT content FROM main
        WHERE volume = ?
          AND CAST(page AS INTEGER) >= ?
          AND CAST(page AS INTEGER) <= ?
        ORDER BY CAST(page AS INTEGER), CAST(items AS INTEGER)
        """,
        (vol_str, page_start, page_end),
    ).fetchall()
    return "\n".join(r["content"] for r in rows if r["content"])


def get_pali_sentences(nissaya_con: sqlite3.Connection,
                       book_id: str,
                       para_start: int,
                       para_end: int) -> list[sqlite3.Row]:
    """Fetch all sentences in the para range."""
    return nissaya_con.execute(
        """
        SELECT * FROM sentences
        WHERE book_id = ?
          AND para_id >= ?
          AND para_id <= ?
        ORDER BY para_id, line_id
        """,
        (book_id, para_start, para_end),
    ).fetchall()


# ─────────────────────────────────────────────
# Translation prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Pali-to-Thai translator specializing in Theravāda Buddhist scripture (Tipiṭaka).

You will receive:
1. A Thai reference text from the Mahāmakut edition — use this as your primary translation guide.
2. A JSON array of Pali sentences, each with para_id and line_id.

Translation rules — follow them strictly:
- For each Pali sentence, look for the corresponding passage in the Thai reference text.
  If you can find a matching or closely related passage, use that Thai wording as the base
  and adjust only as needed to fit the Pali precisely.
- If you CANNOT find any corresponding passage or clue in the Thai reference text for a
  sentence, set thai_translation to "" (empty string). Do NOT guess or translate from
  scratch without a reference — accuracy matters more than completeness.
- Do not add commentary, explanations, or translator notes inside the translation field.
- Return ONLY a JSON array. No preamble, no markdown fences, no text outside the array.

Output format:
[
  {"para_id": 1, "line_id": 1, "thai_translation": "..."},
  {"para_id": 1, "line_id": 2, "thai_translation": ""}
]
"""

def build_prompt(thai_text: str, sentences: list) -> str:
    pali_list = [
        {
            "para_id": s["para_id"],
            "line_id": s["line_id"],
            "pali_sentence": s["pali_sentence"] or "",
        }
        for s in sentences
    ]
    thai_section = (
        thai_text.strip()
        if thai_text.strip()
        else "(ไม่มีข้อความอ้างอิงภาษาไทยสำหรับส่วนนี้ — ให้ใส่ thai_translation เป็นค่าว่างทั้งหมด)"
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Thai Reference Text\n{thai_section}\n\n"
        f"## Pali Sentences (JSON)\n{json.dumps(pali_list, ensure_ascii=False, indent=2)}"
    )


def parse_translations(response_text: str) -> dict[tuple, str]:
    """Parse Gemini JSON response into {(para_id, line_id): thai_translation}.

    Handles three failure modes:
    1. Markdown fences around valid JSON — strip and parse.
    2. Truncated array (ran out of tokens) — recover all complete objects.
    3. Completely unparseable — return empty dict.
    """
    # Strip markdown fences
    text = re.sub(r"```[a-z]*", "", response_text).strip().rstrip("`").strip()

    # ── Attempt 1: clean parse ────────────────
    try:
        items = json.loads(text)
        return _extract_pairs(items)
    except json.JSONDecodeError:
        pass

    # ── Attempt 2: truncated array recovery ──
    # Find every complete {...} object in the text using a simple brace tracker
    recovered = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                fragment = text[start:i+1]
                try:
                    obj = json.loads(fragment)
                    recovered.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    if recovered:
        log.warning(
            f"Partial JSON recovered: {len(recovered)} object(s) extracted from truncated response."
        )
        return _extract_pairs(recovered)

    log.error(f"Could not parse response at all. Snippet: {text[:300]}")
    return {}


def _extract_pairs(items: list) -> dict[tuple, str]:
    result = {}
    for item in items:
        try:
            key = (int(item.get("para_id", 0)), int(item.get("line_id", 0)))
            result[key] = item.get("thai_translation", "")
        except (TypeError, ValueError):
            pass
    return result


# ─────────────────────────────────────────────
# Progress tracking
# ─────────────────────────────────────────────

def get_done_rows(out_con: sqlite3.Connection) -> set[int]:
    rows = out_con.execute("SELECT matching_rowid FROM _progress WHERE done=1").fetchall()
    return {r[0] for r in rows}


def mark_done(out_con: sqlite3.Connection, rowid: int):
    with DB_WRITE_LOCK:
        out_con.execute(
            "INSERT OR REPLACE INTO _progress (matching_rowid, done) VALUES (?, 1)", (rowid,)
        )
        out_con.commit()


def get_done_chunks(out_con: sqlite3.Connection, rowid: int) -> set[int]:
    rows = out_con.execute(
        "SELECT chunk_index FROM _chunk_progress WHERE matching_rowid=? AND done=1", (rowid,)
    ).fetchall()
    return {r[0] for r in rows}


def mark_chunk_done(out_con: sqlite3.Connection, rowid: int, chunk_index: int):
    with DB_WRITE_LOCK:
        out_con.execute(
            "INSERT OR REPLACE INTO _chunk_progress (matching_rowid, chunk_index, done) VALUES (?,?,1)",
            (rowid, chunk_index),
        )
        out_con.commit()


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def process_row(
    rowid: int,
    row: sqlite3.Row,
    thaimm_path: str,
    nissaya_path: str,
    out_path: str,
    dry_run: bool,
):
    """
    Translate one matching row.  Opens its own DB connections so it is safe
    to call from multiple threads simultaneously.
    """
    thaimm_con  = open_db(thaimm_path,  read_only=True)
    nissaya_con = open_db(nissaya_path, read_only=True)
    out_con     = open_db(out_path)
    try:
        thai_vol   = row["thai_volume_id"]
        niss_book  = row["nissaya_book_id"]
        page_start = row["thai_page_start"]
        page_end   = row["thai_page_end"]
        para_start = row["nissaya_para_start"]
        para_end   = row["nissaya_para_end"]
        thai_title = row["thai_title"]

        log.info(
            f"Row {rowid}: '{thai_title}' | "
            f"vol={thai_vol} pages {page_start}-{page_end} | "
            f"{niss_book} paras {para_start}-{para_end}"
        )

        # 1. Thai reference text
        thai_text = get_thai_text(thaimm_con, thai_vol, page_start, page_end)
        if not thai_text:
            log.warning(f"  No Thai text found for vol={thai_vol} pages {page_start}-{page_end}")

        # 2. Pali sentences
        sentences = get_pali_sentences(nissaya_con, niss_book, para_start, para_end)
        if not sentences or len(sentences) < 5:
            log.warning(f"  No Pali sentences for {niss_book} paras {para_start}-{para_end}. Skipping.")
            mark_done(out_con, rowid)
            return


        log.info(f"  {len(sentences)} Pali sentences to translate.")

        if dry_run:
            log.info("  [DRY RUN] Skipping Gemini call.")
            mark_done(out_con, rowid)
            return

        # 3. Translate in chunks, saving each chunk immediately
        CHUNK = 100
        chunks = [sentences[i:i+CHUNK] for i in range(0, len(sentences), CHUNK)]
        done_chunks = get_done_chunks(out_con, rowid)

        for ci, chunk in enumerate(chunks):
            if ci in done_chunks:
                log.info(f"  Chunk {ci+1}/{len(chunks)} already done, skipping.")
                continue

            prompt = build_prompt(thai_text, chunk)
            log.info(f"  Chunk {ci+1}/{len(chunks)} ({len(chunk)} sentences)…")
            try:
                open('input.txt', 'wt').write(prompt)
                response = call_gemini(prompt)
                open('output.txt', 'wt').write(f"{prompt}\n==================\n{response}")
                translations = parse_translations(response)
            except AllKeysExhaustedError:
                raise   # propagate immediately — no keys left, stop everything
            except Exception as e:
                log.error(f"  Chunk {ci+1} failed: {e}. Leaving chunk untranslated.")
                translations = {}

            # Save this chunk immediately
            with DB_WRITE_LOCK:
                for s in chunk:
                    key = (s["para_id"], s["line_id"])
                    thai_tr = translations.get(key, "")
                    out_con.execute(
                        """
                        INSERT OR REPLACE INTO sentences
                          (book_id, para_id, line_id, pali_sentence, thai_translation)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (s["book_id"], s["para_id"], s["line_id"], s["pali_sentence"], thai_tr),
                    )
                out_con.commit()
            mark_chunk_done(out_con, rowid, ci)
            log.info(f"  ✓ Chunk {ci+1}/{len(chunks)} saved ({len(chunk)} rows).")

        mark_done(out_con, rowid)
        log.info(f"  ✓ Row {rowid} complete ({len(sentences)} total sentences).")
    finally:
        thaimm_con.close()
        nissaya_con.close()
        out_con.close()


def run(args):
    global ROTATOR, MODEL_NAME
    ROTATOR = KeyRotator()
    MODEL_NAME = args.model

    workers = args.workers
    log.info(f"Starting pipeline with {workers} worker thread(s).")

    # Load matching rows from a short-lived connection (main thread only)
    thaimm_con_main = open_db(args.thaimm, read_only=True)
    matching_rows = thaimm_con_main.execute(
        "SELECT rowid, * FROM matching ORDER BY rowid"
    ).fetchall()
    thaimm_con_main.close()

    total = len(matching_rows)
    log.info(f"Total matching rows: {total}")

    # Read done-set from a short-lived output connection
    out_con_main = ensure_output_db(args.output)
    done_set = get_done_rows(out_con_main)
    out_con_main.close()
    log.info(f"Already completed: {len(done_set)}")

    # Build the work list (respects --start-row and existing progress)
    pending = [
        (row["rowid"], row)
        for idx, row in enumerate(matching_rows)
        if idx >= args.start_row and row["rowid"] not in done_set
    ]
    log.info(f"Rows to process: {len(pending)}")

    completed = 0
    failed    = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_row,
                rowid, row,
                args.thaimm, args.nissaya, args.output,
                args.dry_run,
            ): rowid
            for rowid, row in pending
        }

        for future in as_completed(futures):
            rowid = futures[future]
            try:
                future.result()
                completed += 1
            except AllKeysExhaustedError as e:
                log.critical(f"ALL API KEYS EXHAUSTED — shutting down. ({e})")
                executor.shutdown(wait=False, cancel_futures=True)
                break
            except KeyboardInterrupt:
                log.info("Interrupted by user.")
                executor.shutdown(wait=False, cancel_futures=True)
                break
            except Exception as e:
                failed += 1
                log.error(f"Row {rowid} failed: {e}", exc_info=True)

    log.info(
        f"Pipeline finished. "
        f"Completed: {completed}, Failed: {failed}, "
        f"Total pending: {len(pending)}"
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tipitaka Pali→Thai translation pipeline")
    parser.add_argument("--thaimm",   default="../data/thaimm.sqlite", help="Path to thaimm.sqlite")
    parser.add_argument("--nissaya",  default="../data/nissaya.db",    help="Path to nissaya.db")
    parser.add_argument("--output",   default="../data/epitaka_th.db", help="Output DB path")
    parser.add_argument("--start-row", type=int, default=0,         help="Skip first N matching rows (0-based index)")
    parser.add_argument("--workers",   type=int, default=4,          help="Number of parallel translation threads (default: 4)")
    parser.add_argument("--model",     default="gemini-2.5-flash-preview-05-20", help="Gemini model name")
    parser.add_argument("--dry-run",  action="store_true",          help="Fetch data but don't call Gemini")
    args = parser.parse_args()

    # Validate inputs
    for p in [args.thaimm, args.nissaya]:
        if not Path(p).exists():
            sys.exit(f"ERROR: Database not found: {p}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run(args)