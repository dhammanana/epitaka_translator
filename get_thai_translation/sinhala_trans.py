"""
Tipitaka Translation Pipeline — Pali → Sinhala
===============================================
For each row in epitaka_link (in sinhala.db):
  1. Fetch Sinhala reference text from sinhala.db (entries table)
     matched via epitaka_link → sinhala_filename + page/entry range
  2. Fetch Pali sentences from nissaya.db (sentences table)
     using epitaka_book_id + epitaka_para_id range
  3. Send both to Gemini for Sinhala translation
  4. Save result to epitaka_si.db

Usage:
  python sinhala_trans.py \\
      --sinhala  data/sinhala.db \\
      --nissaya  data/nissaya.db \\
      --output   data/epitaka_si.db \\
      [--start-row 0] [--dry-run] [--rerun]
"""

import os
import re
import sys
import time
import json
import sqlite3
import logging
import argparse
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
        logging.DEBUG:    "\033[37m",
        logging.INFO:     "\033[36m",
        logging.WARNING:  "\033[33m",
        logging.ERROR:    "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    _RESET = "\033[0m"
    _GREY  = "\033[90m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        ts    = self.formatTime(record, "%H:%M:%S")
        level = f"{color}{record.levelname:<8}{self._RESET}"
        return f"{self._GREY}{ts}{self._RESET} {level} {record.getMessage()}"

_plain_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(ColorFormatter())

_file = logging.FileHandler("sinhala_trans.log", encoding="utf-8")
_file.setFormatter(_plain_fmt)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_console)
logging.root.addHandler(_file)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# AI request / response file logger
# ─────────────────────────────────────────────

_LOG_DIR = Path("log")
_LOG_DIR.mkdir(exist_ok=True)

_STATS_CSV = _LOG_DIR / "translation_stats.csv"


def _ensure_stats_csv() -> None:
    if not _STATS_CSV.exists():
        import csv
        with _STATS_CSV.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "batch_num", "chunk_idx",
                "title", "total", "translated", "blank", "pct_translated",
            ])


def log_translation_stats(
    batch_num: int,
    chunk_idx: int,
    title: str,
    total: int,
    translated: int,
) -> None:
    import csv, datetime
    blank = total - translated
    pct   = f"{translated / total * 100:.1f}" if total else "0.0"
    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.info(
        f"  Stats  batch={batch_num} chunk={chunk_idx+1} | "
        f"total={total}  translated={translated}  blank={blank}  ({pct}%)"
    )

    _ensure_stats_csv()
    with _STATS_CSV.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([ts, batch_num, chunk_idx, title, total, translated, blank, pct])


def dump_ai_exchange(batch_num: int, ci: int, prompt: str, response: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = _LOG_DIR / f"{ts}_row{batch_num}_c{ci}.txt"
    try:
        with fname.open("w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"BATCH {batch_num}  CHUNK {ci}  {ts}\n")
            f.write("=" * 60 + "\n\n")
            f.write("── PROMPT ──────────────────────────────────────────────\n\n")
            f.write(prompt)
            f.write("\n\n── RESPONSE ────────────────────────────────────────────\n\n")
            f.write(response)
            f.write("\n")
        log.debug(f"  AI exchange logged → {fname.name}")
    except Exception as exc:
        log.warning(f"  Could not write AI log file {fname}: {exc}")


# ─────────────────────────────────────────────
# API key rotation
# ─────────────────────────────────────────────

class AllKeysExhaustedError(RuntimeError):
    """Raised when every API key has been permanently removed due to quota exhaustion."""


class KeyRotator:
    """Round-robin key pool with permanent key removal on 429."""

    def __init__(self):
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
        self._index = 0

    def next(self) -> str:
        if not self._keys:
            raise AllKeysExhaustedError("All API keys have been exhausted (429). Exiting.")
        key = self._keys[self._index % len(self._keys)]
        self._index = (self._index + 1) % len(self._keys)
        return key

    def remove(self, key: str):
        if key in self._keys:
            idx = self._keys.index(key)
            self._keys.remove(key)
            log.warning(
                f"Key …{key[-6:]} removed from pool (quota exhausted). "
                f"{len(self._keys)} key(s) remaining."
            )
            if self._keys and self._index > idx:
                self._index -= 1
            if self._keys:
                self._index %= len(self._keys)
            else:
                self._index = 0

    @property
    def count(self) -> int:
        return len(self._keys)


ROTATOR: Optional[KeyRotator] = None
MODEL_NAME: str = "gemini-2.5-flash-preview-05-20"

# ─────────────────────────────────────────────
# Gemini call
# ─────────────────────────────────────────────

def call_gemini(prompt: str, timeout_seconds: int = 300) -> str:
    global ROTATOR

    attempt = 0
    while True:
        attempt += 1
        key = ROTATOR.next()
        key_suffix = key[-6:] if len(key) >= 6 else key

        try:
            client = genai.Client(
                api_key=key,
                http_options=genai.types.HttpOptions(timeout=timeout_seconds * 1000),
            )
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_budget=512,
                    ),
                ),
            )
            log.debug(f"Gemini call succeeded on key …{key_suffix} (attempt {attempt}).")
            return response.text

        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)

            if status == 429:
                log.warning(
                    f"429 RESOURCE_EXHAUSTED on key …{key_suffix} (attempt {attempt}). "
                    f"Removing key from pool permanently."
                )
                ROTATOR.remove(key)
                time.sleep(20)
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


# ─────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────

def open_db(path: str, read_only: bool = False) -> sqlite3.Connection:
    uri = f"file:{path}{'?mode=ro' if read_only else ''}".replace("\\", "/")
    con = sqlite3.connect(uri, uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    if not read_only:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA cache_size=-32768")
    return con


def ensure_output_db(path: str) -> sqlite3.Connection:
    con = open_db(path)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS sentences (
            book_id               TEXT    NOT NULL,
            para_id               INTEGER NOT NULL,
            line_id               INTEGER NOT NULL,
            pali_sentence         TEXT,
            sinhala_translation   TEXT,
            PRIMARY KEY (book_id, para_id, line_id)
        );
    """)
    con.commit()
    return con


# ─────────────────────────────────────────────
# Data fetching — Sinhala side
# ─────────────────────────────────────────────

def get_sinhala_text(
    sinhala_con: sqlite3.Connection,
    filename: str,
    page_start: int,
    entry_start: int,
    page_end: Optional[int],
    entry_end: Optional[int],
) -> str:
    if page_end is None:
        rows = sinhala_con.execute(
            """
            SELECT palitext, sinhalatext FROM entries
            WHERE filename = ?
              AND (
                    page_num > ?
                 OR (page_num = ? AND entry_order >= ?)
              )
            ORDER BY page_num, page_order, entry_order
            """,
            (filename, page_start, page_start, entry_start),
        ).fetchall()
    else:
        rows = sinhala_con.execute(
            """
            SELECT palitext, sinhalatext FROM entries
            WHERE filename = ?
              AND (
                    page_num > ?
                 OR (page_num = ? AND entry_order >= ?)
              )
              AND (
                    page_num < ?
                 OR (page_num = ? AND entry_order < ?)
              )
            ORDER BY page_num, page_order, entry_order
            """,
            (
                filename,
                page_start, page_start, entry_start,
                page_end,   page_end,   entry_end,
            ),
        ).fetchall()

    parts = []
    for r in rows:
        if r["palitext"]:
            parts.append(r["palitext"])
        if r["sinhalatext"]:
            parts.append(r["sinhalatext"])
    return "\n".join(parts)


_sinhala_file_cache: dict[str, str] = {}


def _get_full_file_text(sinhala_con: sqlite3.Connection, filename: str) -> str:
    if filename in _sinhala_file_cache:
        return _sinhala_file_cache[filename]

    rows = sinhala_con.execute(
        """
        SELECT palitext, sinhalatext
        FROM   entries
        WHERE  filename = ?
        ORDER  BY page_num, page_order, entry_order
        """,
        (filename,),
    ).fetchall()

    parts = []
    for r in rows:
        if r["palitext"]:
            parts.append(r["palitext"])
        if r["sinhalatext"]:
            parts.append(r["sinhalatext"])
    text = "\n".join(parts)
    _sinhala_file_cache[filename] = text
    return text


def get_sinhala_text_expanded(
    sinhala_con: sqlite3.Connection,
    filename: str,
    page_start: int,
    entry_start: int,
    page_end: Optional[int],
    entry_end: Optional[int],
    max_chars: int = 50_000,
) -> str:
    full_text = _get_full_file_text(sinhala_con, filename)

    if len(full_text) <= max_chars:
        return full_text

    exact = get_sinhala_text(
        sinhala_con, filename,
        page_start, entry_start,
        page_end, entry_end,
    )

    if len(exact) >= max_chars:
        return exact[:max_chars]

    exact_stripped = exact.strip()
    pos = full_text.find(exact_stripped[:min(200, len(exact_stripped))]) if exact_stripped else -1

    if pos == -1:
        log.debug(
            f"  Sinhala context: exact section not located in full text; "
            f"using first {max_chars} chars of '{filename}'"
        )
        return full_text[:max_chars]

    half  = max_chars // 2
    start = max(0, pos - half)
    end   = min(len(full_text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)

    expanded = full_text[start:end]
    log.debug(
        f"  Sinhala context expanded: {len(exact)} → {len(expanded)} chars "
        f"(centred at pos {pos}) for '{filename}'"
    )
    return expanded


# ─────────────────────────────────────────────
# Data fetching — Pali / Nissaya side
# ─────────────────────────────────────────────

def get_pali_sentences(
    nissaya_con: sqlite3.Connection,
    book_id: str,
    para_start: int,
    para_end: int,
) -> list:
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
# Load epitaka_link rows
# ─────────────────────────────────────────────

def load_link_rows(sinhala_con: sqlite3.Connection) -> list[dict]:
    rows = sinhala_con.execute(
        """
        SELECT
            id                  AS rowid,
            epitaka_book_id,
            epitaka_para_id,
            nissaya_title,
            sinhala_filename,
            sinhala_page_num,
            sinhala_entry_order,
            sinhala_palitext
        FROM   epitaka_link
        ORDER  BY sinhala_filename, sinhala_page_num, sinhala_entry_order
        """
    ).fetchall()
    return [dict(r) for r in rows]


def compute_para_end(
    nissaya_con: sqlite3.Connection,
    book_id: str,
    para_start: int,
    next_link_row: Optional[dict],
) -> int:
    if (
        next_link_row is not None
        and next_link_row["epitaka_book_id"] == book_id
    ):
        return next_link_row["epitaka_para_id"] - 1

    row = nissaya_con.execute(
        "SELECT MAX(para_id) AS max_para FROM sentences WHERE book_id = ?",
        (book_id,),
    ).fetchone()
    return row["max_para"] if row and row["max_para"] is not None else para_start


def compute_sinhala_end(
    next_link_row: Optional[dict],
    current_filename: str,
) -> tuple[Optional[int], Optional[int]]:
    if (
        next_link_row is not None
        and next_link_row["sinhala_filename"] == current_filename
    ):
        return next_link_row["sinhala_page_num"], next_link_row["sinhala_entry_order"]
    return None, None


# ─────────────────────────────────────────────
# Translation prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Pali-to-Sinhala translator specializing in Theravāda Buddhist scripture (Tipiṭaka).

You will receive:
1. A Sinhala reference text from the e-Piṭaka edition — each entry contains the original Pali followed by its Sinhala rendering. Use this as your primary translation guide.
2. A JSON array of Pali sentences, each with para_id and line_id.

Translation rules — follow them strictly:
- For each Pali sentence, find the corresponding passage in the Sinhala reference text.
  If you can find a matching or closely related passage, use that Sinhala wording as the base
  and adjust only as needed to fit the Pali precisely.
- If you CANNOT find any corresponding passage or clue in the Sinhala reference text for a
  sentence, set sinhala_translation to "" (empty string). Do NOT guess or translate from
  scratch without a reference — accuracy matters more than completeness.
- Do not add commentary, explanations, or translator notes inside the translation field.
- Return ONLY a JSON array. No preamble, no markdown fences, no text outside the array.

Output format:
[
  {"para_id": 1, "line_id": 1, "sinhala_translation": "..."},
  {"para_id": 1, "line_id": 2, "sinhala_translation": ""}
]
"""


def build_prompt(sinhala_text: str, sentences: list) -> str:
    pali_list = [
        {
            "para_id":       s["para_id"],
            "line_id":       s["line_id"],
            "pali_sentence": s["pali_sentence"] or "",
        }
        for s in sentences
    ]
    sinhala_section = (
        sinhala_text.strip()
        if sinhala_text.strip()
        else "(සිංහල යොමු පාඨයක් නොමැත — sinhala_translation හිස් දෙන්න)"
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Sinhala Reference Text\n{sinhala_section}\n\n"
        f"## Pali Sentences (JSON)\n{json.dumps(pali_list, ensure_ascii=False, indent=2)}"
    )


def parse_translations(response_text: str) -> dict[tuple, str]:
    text = re.sub(r"```[a-z]*", "", response_text).strip().rstrip("`").strip()

    try:
        items = json.loads(text)
        return _extract_pairs(items)
    except json.JSONDecodeError:
        pass

    # Truncated array recovery
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
            result[key] = item.get("sinhala_translation", "")
        except (TypeError, ValueError):
            pass
    return result


# ─────────────────────────────────────────────
# Section resolver
# ─────────────────────────────────────────────

def resolve_section(
    row: dict,
    next_row: Optional[dict],
    sinhala_con: sqlite3.Connection,
    nissaya_con: sqlite3.Connection,
) -> tuple[str, list]:
    si_filename = row["sinhala_filename"]
    si_page     = row["sinhala_page_num"]
    si_entry    = row["sinhala_entry_order"]
    niss_book   = row["epitaka_book_id"]
    niss_para   = row["epitaka_para_id"]

    si_page_end, si_entry_end = compute_sinhala_end(next_row, si_filename)
    niss_para_end = compute_para_end(nissaya_con, niss_book, niss_para, next_row)

    sinhala_text = get_sinhala_text_expanded(
        sinhala_con, si_filename,
        si_page, si_entry,
        si_page_end, si_entry_end,
    )
    sentences = get_pali_sentences(nissaya_con, niss_book, niss_para, niss_para_end)
    return sinhala_text, sentences


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CHUNK_MIN        = 100
CHUNK_MAX        = 150
SINHALA_CONTEXT_MAX = 50_000
RERUN_MIN_BLANK  = 5


# ─────────────────────────────────────────────
# Process one batch of sentences
# ─────────────────────────────────────────────

def process_batch(
    batch_num: int,
    rowids: list[int],
    title: str,
    sinhala_text: str,
    sentences: list,
    out_con: sqlite3.Connection,
    dry_run: bool,
):
    import math

    n = len(sentences)
    log.info(f"Batch {batch_num}: '{title}' | rowids={rowids} | {n} sentences")

    if not sentences:
        log.info(f"  Batch {batch_num}: no sentences to translate, skipping.")
        return

    if dry_run:
        log.info(f"  [DRY RUN] Batch {batch_num} skipped.")
        return

    if n <= CHUNK_MAX:
        chunks = [sentences]
    else:
        num_chunks = math.ceil(n / CHUNK_MAX)
        base, extra = divmod(n, num_chunks)
        chunks, start = [], 0
        for i in range(num_chunks):
            size = base + (1 if i < extra else 0)
            chunks.append(sentences[start:start + size])
            start += size

    for ci, chunk in enumerate(chunks):
        prompt = build_prompt(sinhala_text, chunk)
        log.info(
            f"  Chunk {ci+1}/{len(chunks)} "
            f"({len(chunk)} sentences, sinhala_ref={len(sinhala_text):,} chars)…"
        )
        try:
            response = call_gemini(prompt)
            dump_ai_exchange(batch_num, ci, prompt, response)
            translations = parse_translations(response)
        except AllKeysExhaustedError:
            raise
        except Exception as e:
            log.error(f"  Chunk {ci+1} failed: {e}. Leaving untranslated.")
            translations = {}

        translated_count = 0
        for s in chunk:
            key   = (s["para_id"], s["line_id"])
            si_tr = translations.get(key, "")
            if si_tr:
                translated_count += 1
            out_con.execute(
                """
                INSERT OR REPLACE INTO sentences
                  (book_id, para_id, line_id, pali_sentence, sinhala_translation)
                VALUES (?, ?, ?, ?, ?)
                """,
                (s["book_id"], s["para_id"], s["line_id"], s["pali_sentence"], si_tr),
            )
        out_con.commit()

        log_translation_stats(batch_num, ci, title, len(chunk), translated_count)
        log.info(f"  ✓ Chunk {ci+1}/{len(chunks)} saved ({len(chunk)} rows).")

    log.info(f"  ✓ Batch {batch_num} complete ({n} sentences).")


# ─────────────────────────────────────────────
# Main entry — run()
# ─────────────────────────────────────────────

def run(args):
    global ROTATOR, MODEL_NAME
    ROTATOR    = KeyRotator()
    MODEL_NAME = args.model

    log.info("Starting Sinhala pipeline (single-threaded).")

    sinhala_con = open_db(args.sinhala, read_only=True)
    nissaya_con = open_db(args.nissaya, read_only=True)
    out_con     = ensure_output_db(args.output)

    link_rows = load_link_rows(sinhala_con)
    log.info(f"Total epitaka_link rows: {len(link_rows)}")

    mode = "RERUN (untranslated-only)" if args.rerun else "FULL"
    log.info(f"Mode: {mode}")

    # Pre-load translated keys for rerun mode
    translated_keys: Optional[set] = None
    if args.rerun:
        rows = out_con.execute(
            """
            SELECT book_id, para_id, line_id
            FROM   sentences
            WHERE  sinhala_translation IS NOT NULL
              AND  sinhala_translation != ''
            """
        ).fetchall()
        translated_keys = {(r["book_id"], r["para_id"], r["line_id"]) for r in rows}
        log.info(f"Rerun mode: {len(translated_keys):,} sentences already translated.")

    n_rows    = len(link_rows)
    batch_num = 0
    completed = 0
    failed    = 0

    # Accumulator for the current batch
    cur_rowids   : list[int] = []
    cur_sinhala  : list[str] = []
    cur_sentences: list      = []
    cur_titles   : list[str] = []

    def dispatch():
        nonlocal batch_num, completed, failed
        nonlocal cur_rowids, cur_sinhala, cur_sentences, cur_titles

        if not cur_rowids:
            return

        sinhala_text = "\n\n".join(filter(None, cur_sinhala))
        if len(sinhala_text) > SINHALA_CONTEXT_MAX:
            sinhala_text = sinhala_text[:SINHALA_CONTEXT_MAX]
            log.debug(f"  Sinhala context clamped to {SINHALA_CONTEXT_MAX} chars for batch {batch_num}")

        title = " + ".join(cur_titles)

        try:
            process_batch(
                batch_num,
                list(cur_rowids),
                title,
                sinhala_text,
                list(cur_sentences),
                out_con,
                args.dry_run,
            )
            completed += 1
        except AllKeysExhaustedError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failed += 1
            log.error(f"Batch {batch_num} failed: {e}", exc_info=True)

        batch_num += 1
        cur_rowids.clear()
        cur_sinhala.clear()
        cur_sentences.clear()
        cur_titles.clear()

    try:
        for idx in range(args.start_row, n_rows):
            row      = link_rows[idx]
            next_row = link_rows[idx + 1] if idx + 1 < n_rows else None
            rowid    = row["rowid"]
            title    = row["nissaya_title"] or row["sinhala_palitext"] or f"row {rowid}"

            si_text, sents = resolve_section(row, next_row, sinhala_con, nissaya_con)

            if not sents:
                log.debug(f"Row {rowid} ('{title}'): 0 sentences, skipping.")
                continue

            if args.rerun:
                sents = [
                    s for s in sents
                    if (s["book_id"], s["para_id"], s["line_id"]) not in translated_keys
                ]
                blank_count = len(sents)
                if blank_count <= RERUN_MIN_BLANK:
                    log.info(
                        f"Row {rowid} ('{title}'): {blank_count} blank sentence(s) "
                        f"<= threshold ({RERUN_MIN_BLANK}), skipping."
                    )
                    continue
                log.info(
                    f"Row {rowid} ('{title}'): {blank_count} untranslated sentence(s) — queuing."
                )

            projected = len(cur_sentences) + len(sents)

            if cur_sentences and projected > CHUNK_MAX:
                dispatch()

            cur_rowids.append(rowid)
            cur_sinhala.append(si_text)
            cur_sentences.extend(sents)
            cur_titles.append(title)

            if len(cur_sentences) >= CHUNK_MIN:
                dispatch()

        # Flush any remaining sentences
        dispatch()

    except AllKeysExhaustedError as e:
        log.critical(f"ALL API KEYS EXHAUSTED — shutting down. ({e})")
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        # Flush what we have
        try:
            dispatch()
        except Exception:
            pass

    sinhala_con.close()
    nissaya_con.close()
    out_con.close()

    log.info(
        f"Pipeline finished. "
        f"Batches completed: {completed}, failed: {failed}, total dispatched: {batch_num}."
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tipitaka Pali→Sinhala translation pipeline")
    parser.add_argument("--sinhala",   default="data/sinhala.db",    help="Path to sinhala.db")
    parser.add_argument("--nissaya",   default="data/nissaya.db",    help="Path to nissaya.db")
    parser.add_argument("--output",    default="data/epitaka_si.db", help="Output DB path")
    parser.add_argument("--start-row", type=int, default=0,          help="Skip first N link rows (0-based index)")
    parser.add_argument("--model",     default="gemini-2.5-flash-preview-05-20", help="Gemini model name")
    parser.add_argument("--dry-run",   action="store_true",          help="Fetch data but don't call Gemini")
    parser.add_argument("--rerun",     action="store_true",          help="Only translate sentences still blank in the output DB")
    args = parser.parse_args()

    for p in [args.sinhala, args.nissaya]:
        if not Path(p).exists():
            sys.exit(f"ERROR: Database not found: {p}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run(args)