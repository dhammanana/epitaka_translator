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
from common import KeyRotator, call_groq, call_gemini, open_db

DB_WRITE_LOCK = threading.Lock()  # serialises all writes to the output DB


log = logging.getLogger(__name__)

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
                response = call_gemini(prompt, MODEL_NAME)
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
    parser.add_argument("--thaimm",   default="data/thaimm.sqlite", help="Path to thaimm.sqlite")
    parser.add_argument("--nissaya",  default="data/nissaya.db",    help="Path to nissaya.db")
    parser.add_argument("--output",   default="data/epitaka_th.db", help="Output DB path")
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