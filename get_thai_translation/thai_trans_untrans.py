"""
Tipitaka Translation Pipeline with Untranslated Detection
===========================================================
For each row in thaimm.matching:
  1. Check epitaka_th.db for untranslated sentences (NULL or '') in that range
  2. If >5 untranslated sentences found:
     - Fetch Thai text from thaimm.main (volume + page range, optionally expanded)
     - Fetch untranslated Pali sentences from epitaka_th.db
     - Send both to Gemini for Thai translation
     - Save result to epitaka_th.db

Usage:
  python thai_trans_untrans.py \
      --thaimm data/thaimm.sqlite \
      --output data/epitaka_th.db \
      [--start-row 0] \
      [--page-expand N] \
      [--dry-run]
"""

import os
import re
import sys
import time
import json
import sqlite3
import logging
import argparse
import itertools
import threading
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
        return f"{self._GREY}{ts}{self._RESET} {level} {msg}"

_plain_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(ColorFormatter())

_file = logging.FileHandler("translate_tipitaka.log", encoding="utf-8")
_file.setFormatter(_plain_fmt)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_console)
logging.root.addHandler(_file)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# API key rotation
# ─────────────────────────────────────────────

class KeyRotator:
    """Thread-safe round-robin key pool (no key removal, simple cycling)."""

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
        self._keys = keys
        self._cycle = itertools.cycle(self._keys)

    def next(self) -> str:
        """Return next key in round-robin order."""
        with self._lock:
            return next(self._cycle)


ROTATOR: Optional[KeyRotator] = None
MODEL_NAME: str = "gemini-2.5-flash-preview-05-20"

# ─────────────────────────────────────────────
# Gemini call with timeout + per-call key rotation
# ─────────────────────────────────────────────

def call_gemini(prompt: str, max_retries: int = 3, timeout_seconds: int = 300) -> str:
    """
    Call Gemini with per-call key rotation and timeout.

    - Each call uses the next key in rotation (round-robin, no key exhaustion tracking).
    - If a call times out (5 min), it's marked as failed and retried with next key.
    - Invalid keys (401, 403) are logged but don't stop the pipeline.
    """
    global ROTATOR

    for attempt in range(max_retries):
        key = ROTATOR.next()
        key_suffix = key[-6:] if len(key) >= 6 else key

        result = {"response": None, "error": None}

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
                f"(attempt {attempt+1}/{max_retries}). Retrying with next key."
            )
            continue

        if result["error"] is not None:
            e = result["error"]
            status = getattr(e, "status_code", None) or getattr(e, "code", None)

            if status in (401, 403):
                log.warning(
                    f"{status} on key …{key_suffix} (attempt {attempt+1}/{max_retries}). "
                    f"Invalid key, trying next key."
                )
            else:
                log.warning(
                    f"Gemini error on key …{key_suffix} (attempt {attempt+1}/{max_retries}): {e}"
                )
            continue

        if result["response"] is not None:
            log.debug(f"Gemini call succeeded on key …{key_suffix}.")
            return result["response"]

    raise RuntimeError(
        f"Gemini failed after {max_retries} attempts (max timeout: {timeout_seconds}s)."
    )


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
    """Create epitaka_th.db with the required schema."""
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


def get_untranslated_sentences(out_con: sqlite3.Connection,
                               book_id: str,
                               para_start: int,
                               para_end: int) -> list[sqlite3.Row]:
    """Fetch untranslated sentences (NULL or empty string) in the para range."""
    return out_con.execute(
        """
        SELECT * FROM sentences
        WHERE book_id = ?
          AND para_id >= ?
          AND para_id <= ?
          AND (thai_translation IS NULL OR thai_translation = '')
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


RESPOND ONLY with a valid JSON array. Each object must contain:
  - "para_id": integer
  - "line_id": integer
  - "thai_translation": string (your Thai translation, or empty string if unable)

Example response format (minimal):
[
  {"para_id": 300, "line_id": 1, "thai_translation": "..."},
  {"para_id": 300, "line_id": 2, "thai_translation": "..."}
]

Do NOT include markdown, explanations, or any text outside the JSON array."""


def build_prompt(thai_text: str, sentences: list[sqlite3.Row]) -> str:
    """Build prompt for Gemini with Thai reference text and Pali sentences."""
    pali_json = json.dumps(
        [
            {
                "para_id":      s["para_id"],
                "line_id":      s["line_id"],
                "pali_sentence": s["pali_sentence"],
            }
            for s in sentences
        ],
        ensure_ascii=False,
        indent=2,
    )

    return f"""{SYSTEM_PROMPT}

─────────────────────────────────────────────
THAI REFERENCE TEXT:
─────────────────────────────────────────────
{thai_text}

─────────────────────────────────────────────
PALI SENTENCES TO TRANSLATE:
─────────────────────────────────────────────
{pali_json}"""


def parse_translations(response_text: str) -> dict[tuple, str]:
    """Parse Gemini JSON response into {(para_id, line_id): thai_translation}.

    Handles three failure modes:
    1. Markdown fences around valid JSON — strip and parse.
    2. Truncated array (ran out of tokens) — recover all complete objects.
    3. Completely unparseable — return empty dict.
    """
    text = re.sub(r"```[a-z]*", "", response_text).strip().rstrip("`").strip()

    # ── Attempt 1: clean parse ────────────────
    try:
        items = json.loads(text)
        return _extract_pairs(items)
    except json.JSONDecodeError:
        pass

    # ── Attempt 2: truncated array recovery ──
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
# Neighbour page expansion
# ─────────────────────────────────────────────

def get_neighbour_page_range(
    thaimm_con: sqlite3.Connection,
    volume_id: str,
    page_start: int,
    page_end: int,
    expand: int,
) -> tuple[int, int]:
    """
    Expand the page window by `expand` pages in each direction, clamped to
    whatever pages actually exist in thaimm.main for that volume.
    """
    vol_str = str(volume_id).zfill(2)
    row = thaimm_con.execute(
        """
        SELECT MIN(CAST(page AS INTEGER)) AS min_p,
               MAX(CAST(page AS INTEGER)) AS max_p
        FROM main
        WHERE volume = ?
        """,
        (vol_str,),
    ).fetchone()

    if row is None or row["min_p"] is None:
        return page_start, page_end   # no data — leave unchanged

    min_p = row["min_p"]
    max_p = row["max_p"]
    return (
        max(min_p, page_start - expand),
        min(max_p, page_end   + expand),
    )


# ─────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────

def process_row(
    rowid: int,
    row: sqlite3.Row,
    thaimm_con: sqlite3.Connection,
    out_con: sqlite3.Connection,
    dry_run: bool,
    page_expand: int,
):
    """
    Check for untranslated sentences in this matching range.
    If >5 are found, fetch Thai context and re-translate them.
    """
    niss_book  = row["nissaya_book_id"]
    page_start = row["thai_page_start"]
    page_end   = row["thai_page_end"]
    para_start = row["nissaya_para_start"]
    para_end   = row["nissaya_para_end"]
    thai_vol   = row["thai_volume_id"]
    thai_title = row["thai_title"]

    # ── 1. Find untranslated sentences ───────
    untranslated = get_untranslated_sentences(out_con, niss_book, para_start, para_end)
    untrans_count = len(untranslated)

    log.info(
        f"Row {rowid} ({thai_title} | {niss_book} para {para_start}–{para_end}): "
        f"{untrans_count} untranslated sentence(s)."
    )

    if untrans_count == 0:
        log.info("  Nothing to do.")
        return

    if untrans_count <= 5:
        log.info(f"  Skipping (≤5 untranslated, below threshold).")
        return

    log.info(f"  ✓ Above threshold — re-translating {untrans_count} sentences…")

    # ── 2. Determine page window ─────────────
    fetch_start, fetch_end = page_start, page_end
    if page_expand > 0:
        fetch_start, fetch_end = get_neighbour_page_range(
            thaimm_con, thai_vol, page_start, page_end, page_expand
        )
        if (fetch_start, fetch_end) != (page_start, page_end):
            log.info(
                f"  Page window expanded by ±{page_expand}: "
                f"{page_start}–{page_end} → {fetch_start}–{fetch_end}"
            )

    # ── 3. Fetch Thai reference text ─────────
    thai_text = get_thai_text(thaimm_con, thai_vol, fetch_start, fetch_end)
    if not thai_text:
        log.warning(
            f"  No Thai text found for vol={thai_vol} pages {fetch_start}–{fetch_end}"
        )

    if dry_run:
        log.info("  [DRY RUN] Skipping Gemini call.")
        return

    # ── 4. Translate in chunks of 100 ────────
    CHUNK = 100
    chunks = [untranslated[i:i+CHUNK] for i in range(0, len(untranslated), CHUNK)]

    for ci, chunk in enumerate(chunks):
        prompt = build_prompt(thai_text, chunk)
        log.info(f"    Chunk {ci+1}/{len(chunks)} ({len(chunk)} sentences)…")
        try:
            response     = call_gemini(prompt)
            translations = parse_translations(response)
        except Exception as e:
            log.error(f"    Chunk {ci+1} failed: {e}. Leaving chunk untranslated.")
            translations = {}

        for s in chunk:
            key     = (s["para_id"], s["line_id"])
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
        log.info(f"    ✓ Chunk {ci+1}/{len(chunks)} saved ({len(chunk)} rows).")

    log.info(f"  ✓ Row {rowid} complete.")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run(args):
    global ROTATOR, MODEL_NAME
    ROTATOR    = KeyRotator()
    MODEL_NAME = args.model

    thaimm_con = open_db(args.thaimm, read_only=True)
    out_con    = ensure_output_db(args.output)

    matching_rows = thaimm_con.execute(
        "SELECT rowid, * FROM matching ORDER BY rowid"
    ).fetchall()

    total = len(matching_rows)
    log.info(f"Total matching rows: {total}")

    for idx, row in enumerate(matching_rows):
        rowid = row["rowid"]
        if idx < args.start_row:
            continue
        try:
            process_row(
                rowid, row,
                thaimm_con, out_con,
                dry_run=args.dry_run,
                page_expand=args.page_expand,
            )
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
            break
        except Exception as e:
            log.error(f"Row {rowid} failed: {e}", exc_info=True)

    log.info("Pipeline finished.")
    thaimm_con.close()
    out_con.close()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tipitaka untranslated detection & re-translation pipeline"
    )
    parser.add_argument("--thaimm",      default="data/thaimm.sqlite",
                        help="Path to thaimm.sqlite")
    parser.add_argument("--output",      default="data/epitaka_th.db",
                        help="Output DB path")
    parser.add_argument("--start-row",   type=int, default=0,
                        help="Skip first N matching rows (0-based index)")
    parser.add_argument("--model",       default="gemini-2.5-flash-preview-05-20",
                        help="Gemini model name")
    parser.add_argument("--page-expand", type=int, default=0,
                        help=(
                            "Expand Thai page window by N pages in each direction. "
                            "E.g. --page-expand 3 fetches 3 extra pages before and after "
                            "the matched range to give Gemini broader context. "
                            "Clamped to the actual page bounds of that volume."
                        ))
    parser.add_argument("--dry-run",     action="store_true",
                        help="Fetch data but don't call Gemini")
    args = parser.parse_args()

    for p in [args.thaimm, args.output]:
        if not Path(p).exists():
            sys.exit(f"ERROR: Database not found: {p}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run(args)