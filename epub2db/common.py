"""
Dhammapada Translator: Read sections from dhammapada.db, fetch Pali sentences
from nissaya.db using book_id/para_id/chap_len, translate using English content,
then save translations back into dhammapada.db's sentences table.
"""

import os
import re
import sys
import json
import time
import sqlite3
import logging
import threading
from pathlib import Path
from typing import Optional

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
    """Create sentences table in dhammapada.db if it doesn't exist."""
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
# Section & Sentence Retrieval
# ─────────────────────────────────────────────

def get_dhammapada_sections(dhammapada_con: sqlite3.Connection) -> list:
    """Fetch all sections from dhammapada.db that have a book_id and content."""
    rows = dhammapada_con.execute("""
        SELECT id, file, book_id, para_id, chap_len, chapter,
               h2_title, h3_title, content
        FROM sections
        WHERE book_id IS NOT NULL
          AND para_id IS NOT NULL
          AND chap_len IS NOT NULL
          AND content IS NOT NULL
          AND content != ''
        ORDER BY id
    """).fetchall()
    log.info(f"  Found {len(rows)} sections in dhammapada.db")
    return rows


def get_sentences_for_section(nissaya_con: sqlite3.Connection,
                               book_id: str,
                               para_id: int,
                               chap_len: int) -> list:
    """Fetch Pali sentences from nissaya.db for the given book_id,
    in the para range [para_id, para_id + chap_len - 1]."""
    para_end = para_id + chap_len - 1
    sentences = nissaya_con.execute("""
        SELECT book_id, para_id, line_id, pali_sentence
        FROM sentences
        WHERE book_id = ?
          AND para_id >= ?
          AND para_id <= ?
        ORDER BY para_id, line_id
    """, (book_id, para_id, para_end)).fetchall()
    return sentences

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

def translate_section(pali_sentences: list, eng_passage: str,
                      section_title: str='', book_id: str) -> dict:
    """Translate sentences for one section; returns {(para_id, line_id): english_text}."""
    if not pali_sentences or not eng_passage:
        return {}

    CHUNK = 150
    chunks = [pali_sentences[i:i+CHUNK] for i in range(0, len(pali_sentences), CHUNK)]
    translations = {}

    for ci, chunk in enumerate(chunks):
        payload = [
            {
                "para_id": s["para_id"],
                "line_id": s["line_id"],
                "pali_sentence": s["pali_sentence"] or "",
            }
            for s in chunk
        ]

        prompt = (
            f"{TRANSLATE_SYSTEM}\n\n"
            f"## Section\n{section_title}\n\n"
            f"## English Passage\n{eng_passage}\n\n"
            f"## Pali Sentences (chunk {ci + 1}/{len(chunks)})\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

        log.info(f"    Chunk {ci + 1}/{len(chunks)}: {len(chunk)} sentences…")

        _overwrite(
            "input.txt",
            f"=== TRANSLATION CALL: {book_id} '{section_title}' chunk {ci + 1}/{len(chunks)} ===\n\n"
            f"{prompt}\n"
        )

        try:
            response = call_gemini(prompt)

            _overwrite(
                "output.txt",
                f"=== TRANSLATION RESPONSE: {book_id} '{section_title}' chunk {ci + 1}/{len(chunks)} ===\n\n"
                f"{response}\n"
            )

            items = parse_json_response(response, f"translation [{book_id}]")
            total = len(items)
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

            log.info(f"      Sentences translated: {non_empty}/{total} (non-empty/total)")

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

def process_dhammapada(nissaya_path: str, dhammapada_path: str):
    """Main pipeline for Dhammapada.

    For each section in dhammapada.db:
      - Read book_id, para_id, chap_len, and content (English translation).
      - Fetch Pali sentences from nissaya.db for the para range
        [para_id, para_id + chap_len - 1].
      - Translate by aligning Pali sentences with the English content.
      - Save translations back into dhammapada.db's sentences table.
    """
    log.info("Processing Dhammapada")

    nissaya_con  = open_db(nissaya_path, read_only=True)
    dhamm_con    = open_db(dhammapada_path)           # read + write
    ensure_sentences_table(dhamm_con)

    sections = get_dhammapada_sections(dhamm_con)

    for sec in sections:
        book_id    = sec["book_id"]
        para_id    = sec["para_id"]
        chap_len   = sec["chap_len"]
        eng_passage = sec["content"] or ""

        # Build a descriptive title for logging / prompts
        h2 = sec["h2_title"] or ""
        h3 = sec["h3_title"] or ""
        chapter = sec["chapter"] or ""
        section_title = " — ".join(filter(None, [chapter, h2, h3])) or f"{book_id}:{para_id}"

        log.info(f"  Section '{section_title}' [{book_id} para {para_id}..{para_id + chap_len - 1}]")

        # ── Fetch Pali sentences from nissaya.db ─────────────────────────────
        sentences = get_sentences_for_section(nissaya_con, book_id, para_id, chap_len)
        if not sentences:
            log.warning(f"    No Pali sentences found — skipping")
            continue

        # ── Skip if already fully translated ─────────────────────────────────
        para_end = para_id + chap_len - 1
        done_keys = {
            (row[0], row[1])
            for row in dhamm_con.execute("""
                SELECT para_id, line_id FROM sentences
                WHERE book_id = ?
                  AND para_id >= ? AND para_id <= ?
                  AND english_translation IS NOT NULL
                  AND english_translation != ''
            """, (book_id, para_id, para_end)).fetchall()
        }

        sentence_keys = {(s["para_id"], s["line_id"]) for s in sentences}

        if len(done_keys) > 5:
            log.info(f"    ✓ Skipping (already has {len(done_keys)} translations)")
            continue
        if sentence_keys == done_keys:
            log.info(f"    ✓ Already done ({len(sentence_keys)} sentences) — skipping")
            continue

        log.info(f"    {len(sentences)} Pali sentences | passage {len(eng_passage)} chars "
                 f"| {len(done_keys)} already translated")

        # ── Translate ─────────────────────────────────────────────────────────
        translations = translate_section(sentences, eng_passage, section_title, book_id)

        # ── Persist translations into dhammapada.db sentences ─────────────────
        with DB_WRITE_LOCK:
            for s in sentences:
                key = (s["para_id"], s["line_id"])
                eng_trans = translations.get(key, "")
                dhamm_con.execute("""
                    INSERT OR REPLACE INTO sentences
                      (book_id, para_id, line_id, pali_sentence, english_translation)
                    VALUES (?, ?, ?, ?, ?)
                """, (s["book_id"], s["para_id"], s["line_id"],
                      s["pali_sentence"], eng_trans))
            dhamm_con.commit()

        log.info(f"    ✓ Saved {len(sentences)} sentences to dhammapada.db")

    nissaya_con.close()
    dhamm_con.close()
    log.info("✓ Dhammapada pipeline complete")

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dhammapada translator — reads sections from dhammapada.db, "
                    "fetches Pali sentences from nissaya.db, and saves translations "
                    "back into dhammapada.db."
    )
    parser.add_argument("--nissaya", default="../data/nissaya.db",
                        help="Path to nissaya.db (Pali source, read-only)")
    parser.add_argument("--dhammapada", default="dhammapada.db",
                        help="Path to dhammapada.db (sections + output sentences)")
    parser.add_argument("--model", default="gemini-2.5-flash-preview-05-20",
                        help="Gemini model name")
    args = parser.parse_args()

    for p in [args.nissaya, args.dhammapada]:
        if not Path(p).exists():
            sys.exit(f"ERROR: path not found: {p}")

    MODEL_NAME = args.model

    process_dhammapada(args.nissaya, args.dhammapada)

    log.info("Pipeline complete")