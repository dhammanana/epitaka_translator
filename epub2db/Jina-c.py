"""
Jina-c Translator: Read English passages from Jina-c-anandajoti.csv,
group rows in chunks of 3, find the Pali para_id range via vripara lookup
in nissaya.db, fetch sentences for book_id='Jina-c', translate, and save
translations back into an output SQLite database.
"""

import os
import re
import sys
import csv
import json
import time
import sqlite3
import logging
import threading
from pathlib import Path

from google import genai
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Key Rotation
# ─────────────────────────────────────────────

class KeyRotator:
    def __init__(self):
        self._lock = threading.Lock()
        keys = [
            v.strip()
            for k, v in os.environ.items()
            if re.match(r"^GEMINI_KEY_\d+$", k) and v.strip()
        ]
        if not keys:
            raise RuntimeError("No GEMINI_KEY_<N> env vars found.")
        log.info(f"Loaded {len(keys)} Gemini key(s).")
        self._keys = list(keys)
        self._index = 0

    def next(self) -> str:
        with self._lock:
            if not self._keys:
                raise RuntimeError("All keys exhausted.")
            k = self._keys[self._index % len(self._keys)]
            self._index = (self._index + 1) % len(self._keys)
            return k

    def remove(self, key: str):
        with self._lock:
            if key in self._keys:
                self._keys.remove(key)
                log.warning(f"Key removed. {len(self._keys)} remaining.")

ROTATOR = KeyRotator()
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
BOOK_ID = "Jina-c"
DB_WRITE_LOCK = threading.Lock()
OVERWRITE_LOCK = threading.Lock()


def _overwrite(path: str, content: str):
    """Overwrite a file with the given content (UTF-8). Thread-safe."""
    with OVERWRITE_LOCK:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            log.warning(f"Could not write {path}: {e}")

# ─────────────────────────────────────────────
# API Calls
# ─────────────────────────────────────────────

def call_gemini(prompt: str, timeout: int = 300) -> str:
    """Call Gemini API with key rotation and timeout."""
    attempt = 0
    while attempt < 10:
        attempt += 1
        key = ROTATOR.next()
        result = {"response": None, "error": None}

        def _call():
            try:
                client = genai.Client(api_key=key)
                r = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(),
                )
                result["response"] = r.text
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            log.error(f"Timeout attempt {attempt}")
            continue

        if result["error"]:
            e = result["error"]
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status == 429:
                ROTATOR.remove(key)
                time.sleep(20)
                continue
            if status in (401, 403):
                time.sleep(20)
                continue
            log.warning(f"Error attempt {attempt}: {e}")
            time.sleep(20)
            continue

        if result["response"] is not None:
            return result["response"]

    raise RuntimeError("All retry attempts exhausted")


def parse_json_response(response: str, label: str = "") -> list:
    """Strip markdown fences and parse JSON array."""
    text = re.sub(r"```[a-z]*", "", response).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Object-by-object recovery
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
                try:
                    recovered.append(json.loads(text[start : i + 1]))
                except Exception:
                    pass
                start = None

    log.warning(f"Partial JSON ({label}): recovered {len(recovered)} items.")
    return recovered

# ─────────────────────────────────────────────
# Database Helpers
# ─────────────────────────────────────────────

def open_db(path: str, read_only: bool = False) -> sqlite3.Connection:
    """Open database with WAL and Row factory."""
    uri = f"file:{path}{'?mode=ro' if read_only else ''}".replace("\\", "/")
    con = sqlite3.connect(uri, uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    if not read_only:
        con.execute("PRAGMA journal_mode=WAL")
    return con


def ensure_sentences_table(con: sqlite3.Connection):
    """Create sentences table in the output DB if it doesn't exist."""
    con.executescript("""
        CREATE TABLE IF NOT EXISTS sentences (
            book_id TEXT NOT NULL,
            para_id INTEGER NOT NULL,
            line_id INTEGER NOT NULL,
            pali_sentence TEXT,
            english_translation TEXT,
            PRIMARY KEY (book_id, para_id, line_id)
        );
    """)
    con.commit()

# ─────────────────────────────────────────────
# CSV Reading & Chunking
# ─────────────────────────────────────────────

def load_csv_rows(csv_path: str) -> list[dict]:
    """Load Jina-c-anandajoti.csv; return list of dicts with int verse numbers."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "file":        row["file"],
                    "start_verse": int(row["start_verse"]),
                    "end_verse":   int(row["end_verse"]),
                    "content":     row["content"] or "",
                })
            except (KeyError, ValueError) as e:
                log.warning(f"  Skipping malformed CSV row: {e} — {dict(row)}")
    log.info(f"  Loaded {len(rows)} rows from CSV")
    return rows


def chunk_rows(rows: list[dict], chunk_size: int = 3) -> list[list[dict]]:
    """Group CSV rows into chunks of *chunk_size*."""
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

# ─────────────────────────────────────────────
# Pali Sentence Retrieval via vripara
# ─────────────────────────────────────────────

def get_para_id_for_vripara(nissaya_con: sqlite3.Connection,
                             book_id: str, vripara: int) -> int | None:
    """Look up the para_id in nissaya.db whose vripara equals *vripara*
    for the given book_id.

    Assumes a table or view that associates vripara with para_id, e.g.:

        SELECT para_id FROM sentences
        WHERE book_id = ? AND vripara = ?
        LIMIT 1

    Adjust the query below if the schema differs (e.g. separate verses table).
    """
    row = nissaya_con.execute("""
        SELECT para_id FROM sentences
        WHERE book_id = ? AND vripara = ?
        ORDER BY para_id, line_id
        LIMIT 1
    """, (book_id, vripara)).fetchone()
    return row["para_id"] if row else None


def get_sentences_by_para_range(nissaya_con: sqlite3.Connection,
                                 book_id: str,
                                 para_start: int,
                                 para_end: int) -> list:
    """Fetch all Pali sentences in [para_start, para_end] for book_id."""
    return nissaya_con.execute("""
        SELECT book_id, para_id, line_id, pali_sentence
        FROM sentences
        WHERE book_id = ?
          AND para_id >= ?
          AND para_id <= ?
        ORDER BY para_id, line_id
    """, (book_id, para_start, para_end)).fetchall()

# ─────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────

TRANSLATE_SYSTEM = """\
You are an assistant specialized in aligning Pali sentences with their corresponding English translations.

You will be provided with:
1. ENGLISH PASSAGE: The published English translation of a text section.
2. PALI SENTENCES: A JSON array of Pali sentences, each identified by a "para_id" and "line_id".

Your Task:
For each Pali sentence, locate the corresponding segment within the ENGLISH PASSAGE and extract that exact text as the "english_translation" value.

Strict Rules:
- Verbatim Extraction: Use only the text that is explicitly present in the provided ENGLISH PASSAGE. Do not translate the Pali independently, paraphrase the English, or infer meaning.
- Handling Unmatched Sentences: If a Pali sentence has no clear corresponding translation in the English passage, assign an empty string ("") to the "english_translation" field.
- Completeness: Process every input sentence. Every "para_id" and "line_id" from the input must be included in the output in the same order. Do not omit any entries.
- Technical Terms: Retain any Pali technical terms exactly as they are written in the provided ENGLISH PASSAGE.

Output Format:
Return only a valid JSON array. Do not include any introductory text, explanatory remarks, or markdown code block formatting (such as ```json ... ```).

[
  {"para_id": 1, "line_id": 1, "english_translation": "..."},
  ...
]
"""


def translate_chunk(pali_sentences: list, eng_passage: str,
                    chunk_label: str, book_id: str) -> dict:
    """Translate one chunk of sentences; returns {(para_id, line_id): english_text}."""
    if not pali_sentences or not eng_passage:
        return {}

    CHUNK = 150
    sub_chunks = [pali_sentences[i:i+CHUNK] for i in range(0, len(pali_sentences), CHUNK)]
    translations = {}

    for ci, sub in enumerate(sub_chunks):
        payload = [
            {
                "para_id":      s["para_id"],
                "line_id":      s["line_id"],
                "pali_sentence": s["pali_sentence"] or "",
            }
            for s in sub
        ]

        prompt = (
            f"{TRANSLATE_SYSTEM}\n\n"
            f"## Section\n{chunk_label}\n\n"
            f"## English Passage\n{eng_passage}\n\n"
            f"## Pali Sentences (sub-chunk {ci + 1}/{len(sub_chunks)})\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

        log.info(f"    Sub-chunk {ci + 1}/{len(sub_chunks)}: {len(sub)} sentences…")

        _overwrite(
            "input.txt",
            f"=== TRANSLATION CALL: {book_id} '{chunk_label}' sub-chunk {ci + 1}/{len(sub_chunks)} ===\n\n"
            f"{prompt}\n"
        )

        try:
            response = call_gemini(prompt)

            _overwrite(
                "output.txt",
                f"=== TRANSLATION RESPONSE: {book_id} '{chunk_label}' sub-chunk {ci + 1}/{len(sub_chunks)} ===\n\n"
                f"{response}\n"
            )

            items = parse_json_response(response, f"translation [{book_id}]")
            total    = len(items)
            non_empty = 0

            for item in items:
                try:
                    key = (int(item["para_id"]), int(item["line_id"]))
                    translation = item.get("english_translation", "")
                    translations[key] = translation
                    if translation.strip():
                        non_empty += 1
                except Exception:
                    pass

            log.info(f"      Translated: {non_empty}/{total} (non-empty/total)")

        except RuntimeError as e:
            log.error(f"Runtime failed: {e}")
            sys.exit(1)
        except Exception as e:
            log.error(f"      Translation failed: {e}")

        time.sleep(1)

    return translations

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def process_jinac(nissaya_path: str, csv_path: str, out_db_path: str,
                  chunk_size: int = 3):
    """Main pipeline for Jina-c.

    For every chunk of *chunk_size* CSV rows:
      1. Merge their English content into one passage.
      2. Determine verse range: first row's start_verse … last row's end_verse.
      3. Look up the para_id for both boundary verses in nissaya.db via vripara.
      4. Fetch all Pali sentences in [para_start, para_end] for book_id='Jina-c'.
      5. Translate by aligning Pali with the merged English passage.
      6. Save results into the output DB's sentences table.
    """
    log.info(f"Processing Jina-c  (CSV chunk size = {chunk_size})")

    nissaya_con = open_db(nissaya_path, read_only=True)
    out_con     = open_db(out_db_path)
    ensure_sentences_table(out_con)

    # ── Load & chunk CSV ──────────────────────────────────────────────────────
    rows   = load_csv_rows(csv_path)
    chunks = chunk_rows(rows, chunk_size)
    log.info(f"  {len(rows)} CSV rows → {len(chunks)} chunks of ≤{chunk_size}")

    for chunk_idx, chunk in enumerate(chunks):
        start_verse = chunk[0]["start_verse"]
        end_verse   = chunk[-1]["end_verse"]
        chunk_label = (
            f"vv. {start_verse}–{end_verse}  "
            f"({', '.join(r['file'] for r in chunk)})"
        )
        log.info(f"  Chunk {chunk_idx + 1}/{len(chunks)}: {chunk_label}")

        # Merge English passages from all rows in the chunk
        eng_passage = "\n\n".join(r["content"] for r in chunk if r["content"].strip())
        if not eng_passage.strip():
            log.warning(f"    No English content — skipping")
            continue

        # ── Find para_id boundaries via vripara ───────────────────────────────
        para_start_row = get_para_id_for_vripara(nissaya_con, BOOK_ID, start_verse)
        para_end_row   = get_para_id_for_vripara(nissaya_con, BOOK_ID, end_verse)

        if para_start_row is None:
            log.warning(f"    vripara={start_verse} not found in nissaya.db — skipping")
            continue
        if para_end_row is None:
            log.warning(f"    vripara={end_verse} not found in nissaya.db — skipping")
            continue

        para_start = para_start_row
        para_end   = para_end_row
        log.info(f"    para range: {para_start} … {para_end}")

        # ── Fetch Pali sentences ───────────────────────────────────────────────
        sentences = get_sentences_by_para_range(nissaya_con, BOOK_ID, para_start, para_end)
        if not sentences:
            log.warning(f"    No Pali sentences in para range — skipping")
            continue

        # ── Skip if already translated ────────────────────────────────────────
        done_keys = {
            (row[0], row[1])
            for row in out_con.execute("""
                SELECT para_id, line_id FROM sentences
                WHERE book_id = ?
                  AND para_id >= ? AND para_id <= ?
                  AND english_translation IS NOT NULL
                  AND english_translation != ''
            """, (BOOK_ID, para_start, para_end)).fetchall()
        }
        sentence_keys = {(s["para_id"], s["line_id"]) for s in sentences}

        if len(done_keys) > 5:
            log.info(f"    ✓ Skipping (already has {len(done_keys)} translations)")
            continue
        if sentence_keys == done_keys:
            log.info(f"    ✓ Already done ({len(sentence_keys)} sentences) — skipping")
            continue

        log.info(f"    {len(sentences)} sentences | passage {len(eng_passage)} chars "
                 f"| {len(done_keys)} already translated")

        # ── Translate ─────────────────────────────────────────────────────────
        translations = translate_chunk(sentences, eng_passage, chunk_label, BOOK_ID)

        # ── Persist ───────────────────────────────────────────────────────────
        with DB_WRITE_LOCK:
            for s in sentences:
                key = (s["para_id"], s["line_id"])
                eng_trans = translations.get(key, "")
                out_con.execute("""
                    INSERT OR REPLACE INTO sentences
                      (book_id, para_id, line_id, pali_sentence, english_translation)
                    VALUES (?, ?, ?, ?, ?)
                """, (BOOK_ID, s["para_id"], s["line_id"],
                      s["pali_sentence"], eng_trans))
            out_con.commit()

        log.info(f"    ✓ Saved {len(sentences)} sentences")

    nissaya_con.close()
    out_con.close()
    log.info("✓ Jina-c pipeline complete")

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Jina-c translator — reads English from CSV, aligns with Pali "
                    "sentences from nissaya.db via vripara lookup, saves translations."
    )
    parser.add_argument("--nissaya", default="../data/nissaya.db",
                        help="Path to nissaya.db (read-only Pali source)")
    parser.add_argument("--csv", default="Jina-c-anandajoti.csv",
                        help="Path to Jina-c-anandajoti.csv")
    parser.add_argument("--db", default="jinac_out.db",
                        help="Output SQLite database for translated sentences")
    parser.add_argument("--chunk-size", type=int, default=3,
                        help="Number of CSV rows per translation chunk (default: 3)")
    parser.add_argument("--model", default="gemini-2.5-flash-preview-05-20",
                        help="Gemini model name")
    args = parser.parse_args()

    for p in [args.nissaya, args.csv]:
        if not Path(p).exists():
            sys.exit(f"ERROR: path not found: {p}")

    MODEL_NAME = args.model

    process_jinac(args.nissaya, args.csv, args.db, chunk_size=args.chunk_size)

    log.info("Pipeline complete")