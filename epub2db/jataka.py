"""
Abhidhamma Translator: Match Pali headings from nissaya.db with Markdown chapters,
then translate Pali sentences using English passage context.
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
# Configure loggers to suppress INFO logs
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

def ensure_abhidh_tables(con: sqlite3.Connection):
    """Create output tables if they don't exist."""
    con.executescript("""
        CREATE TABLE IF NOT EXISTS headings (
            book_id TEXT NOT NULL,
            para_id INTEGER NOT NULL,
            title TEXT,
            level INTEGER,
            parent INTEGER,
            PRIMARY KEY (book_id, para_id)
        );

        CREATE TABLE IF NOT EXISTS sentences (
            book_id TEXT NOT NULL,
            para_id INTEGER NOT NULL,
            line_id INTEGER NOT NULL,
            pali_sentence TEXT,
            english_translation TEXT,
            PRIMARY KEY (book_id, para_id, line_id)
        );

        CREATE TABLE IF NOT EXISTS md_chapters (
            book_id TEXT,
            chapter_num INTEGER,
            chapter_title TEXT,
            content TEXT,
            PRIMARY KEY (book_id, chapter_num)
        );

        CREATE TABLE IF NOT EXISTS matches (
            book_id TEXT,
            para_id INTEGER,
            chapter_num INTEGER,
            confidence TEXT,
            PRIMARY KEY (book_id, para_id)
        );
    """)
    con.commit()

# ─────────────────────────────────────────────
# EPUB HTML Parsing  (English side)
# ─────────────────────────────────────────────

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # will raise a clear error at runtime if needed


def _require_bs4():
    if BeautifulSoup is None:
        raise RuntimeError(
            "beautifulsoup4 is required. Install it with: pip install beautifulsoup4"
        )


def collect_epub_html_files(jataka_root: str) -> list[Path]:
    """Return all .htm/.html/.xhtml files under *jataka_root*, sorted by path."""
    root = Path(jataka_root)
    files = [
        p for p in root.rglob("*")
        if p.suffix.lower() in {".htm", ".html", ".xhtml"} and p.is_file()
    ]
    return sorted(files)


def _extract_number(text: str) -> Optional[int]:
    """Return the leading integer from a string like '264. Mahā…' or '264 Mahā…'."""
    m = re.match(r"^\s*(\d+)", text)
    return int(m.group(1)) if m else None


def parse_epub_english_chapters(jataka_root: str) -> dict[int, dict]:
    """Scan every HTML file in *jataka_root* (sorted), extract the <h2> title,
    parse its leading tale-number, and collect the full visible text of that file
    as the English passage.

    Returns {tale_number: {"title": str, "content": str, "file": str}}
    """
    _require_bs4()
    html_files = collect_epub_html_files(jataka_root)
    log.info(f"  Found {len(html_files)} HTML files under '{jataka_root}'")

    chapters: dict[int, dict] = {}
    for html_path in html_files:
        try:
            soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        except Exception as e:
            log.warning(f"  Could not parse {html_path}: {e}")
            continue

        h2_tag = soup.find("h2")
        if not h2_tag:
            continue  # no h2 → skip (front-matter, navigation pages, etc.)

        h2_text = h2_tag.get_text(separator=" ", strip=True)
        tale_num = _extract_number(h2_text)
        if tale_num is None:
            log.debug(f"  Skipped (no leading number in h2): {html_path.name} — '{h2_text}'")
            continue

        # Full visible text of the file (strip all tags)
        full_text = soup.get_text(separator="\n", strip=True)

        if tale_num in chapters:
            log.warning(f"  Duplicate tale number {tale_num} in {html_path}; keeping first occurrence")
            continue

        chapters[tale_num] = {
            "title": h2_text,
            "content": full_text,
            "file": str(html_path),
        }

    log.info(f"  Parsed {len(chapters)} English tale chapters from EPUB HTML")
    return chapters


# ─────────────────────────────────────────────
# Pali Heading Retrieval  (Pali side)
# ─────────────────────────────────────────────

# Pattern: [264] 4. Mahāpanādajātakavaṇṇanā
_PALI_TITLE_RE = re.compile(r"^\[(\d+)\]")


def get_pali_jataka_chapters(nissaya_con: sqlite3.Connection) -> dict[int, dict]:
    """Fetch headings from nissaya.db with book_id LIKE 'J-a-%' and level < 10,
    filter those whose title starts with [<number>], and return
    {tale_number: {"book_id": str, "para_id": int, "title": str, "level": int}}.
    """
    rows = nissaya_con.execute("""
        SELECT book_id, para_id, title, level, parent
        FROM headings
        WHERE book_id LIKE 'Ja-a-%'
          AND level < 10
        ORDER BY book_id, para_id
    """).fetchall()

    chapters: dict[int, dict] = {}
    for row in rows:
        title = row["title"] or ""
        m = _PALI_TITLE_RE.match(title.strip())
        if not m:
            continue
        tale_num = int(m.group(1))
        if tale_num in chapters:
            log.warning(f"  Duplicate Pali tale number {tale_num} — keeping first occurrence")
            continue
        chapters[tale_num] = {
            "book_id": row["book_id"],
            "para_id": row["para_id"],
            "title": title,
            "level": row["level"],
        }

    log.info(f"  Found {len(chapters)} Pali Jātaka chapters in nissaya.db")
    return chapters


def get_sentences_for_chapter(nissaya_con: sqlite3.Connection,
                               book_id: str, chapter_para_id: int) -> tuple[list, int, int]:
    """Get sentences for a chapter and return para range."""
    row = nissaya_con.execute(
        "SELECT para_id, level FROM headings WHERE book_id=? AND para_id=?",
        (book_id, chapter_para_id)
    ).fetchone()

    if not row:
        return [], chapter_para_id, chapter_para_id

    next_row = nissaya_con.execute("""
        SELECT para_id FROM headings
        WHERE book_id = ? AND para_id > ? AND level <= ?
        ORDER BY para_id LIMIT 1
    """, (book_id, chapter_para_id, row["level"])).fetchone()

    para_start = chapter_para_id
    para_end = (next_row["para_id"] - 1) if next_row else 999_999

    sentences = nissaya_con.execute("""
        SELECT * FROM sentences
        WHERE book_id = ? AND para_id >= ? AND para_id <= ?
        ORDER BY para_id, line_id
    """, (book_id, para_start, para_end)).fetchall()

    return sentences, para_start, para_end

# ─────────────────────────────────────────────
# Matching & Translation
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

def translate_chapter(pali_sentences: list, eng_passage: str,
                     chapter_title: str, book_id: str) -> dict:
    """Translate sentences and return {(para_id, line_id): english_text}."""
    if not pali_sentences or not eng_passage:
        return {}
    
    # Chunk into groups of 150
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
            f"## Chapter\n{chapter_title}\n\n"
            f"## English Passage\n{eng_passage}\n\n"
            f"## Pali Sentences (chunk {ci + 1}/{len(chunks)})\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        
        log.info(f"    Chunk {ci + 1}/{len(chunks)}: {len(chunk)} sentences…")
        
        # Log prompt
        _overwrite(
            "input.txt",
            f"=== TRANSLATION CALL: {book_id} {chapter_title} chunk {ci + 1}/{len(chunks)} ===\n\n"
            f"{prompt}\n"
        )
        
        try:
            response = call_gemini(prompt)
            
            # Log response
            _overwrite(
                "output.txt",
                f"=== TRANSLATION RESPONSE: {book_id} {chapter_title} chunk {ci + 1}/{len(chunks)} ===\n\n"
                f"{response}\n"
            )
            
            items = parse_json_response(response, f"translation [{book_id}]")
            total_sentences = len(items)
            non_empty_sentences = 0

            for item in items:
                try:
                    key = (int(item["para_id"]), int(item["line_id"]))
                    translation = item.get("english_translation", "")
                    translations[key] = translation
                    # Log the proportion of non-empty sentences
                    # Check if the translation is not empty
                    if translation.strip() != "":
                        non_empty_sentences += 1
                except Exception:
                    pass
            log.info(f"      Sentences translated: {non_empty_sentences}/{total_sentences} (non-empty/total)")
        except RuntimeError as e:
            log.error(f"Runtime failed: {e}")
            exit()
        except Exception as e:
            log.error(f"      Translation failed: {e}")
        
        time.sleep(1)
    
    return translations

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def process_jataka(nissaya_path: str, jataka_root: str, out_db_path: str):
    """Main pipeline for Jātaka: match tale chapters by number and translate.

    English source  → EPUB HTML files under *jataka_root*, matched by leading
                      number extracted from each file's <h2>.
    Pali source     → nissaya.db headings with book_id LIKE 'Ja-a-%', level < 10,
                      and title starting with [<number>].
    Matching        → by tale number (the leading integer present in both sides).
    """
    log.info("Processing Jātaka")

    nissaya_con = open_db(nissaya_path, read_only=True)
    out_con = open_db(out_db_path)
    ensure_abhidh_tables(out_con)

    # ── English side ──────────────────────────────────────────────────────────
    eng_chapters = parse_epub_english_chapters(jataka_root)

    # ── Pali side ─────────────────────────────────────────────────────────────
    pali_chapters = get_pali_jataka_chapters(nissaya_con)

    # ── Match by tale number ──────────────────────────────────────────────────
    common_nums = sorted(set(pali_chapters) & set(eng_chapters))
    only_pali   = sorted(set(pali_chapters) - set(eng_chapters))
    only_eng    = sorted(set(eng_chapters)  - set(pali_chapters))

    log.info(f"  Matched {len(common_nums)} tale(s) | "
             f"Pali-only {len(only_pali)} | English-only {len(only_eng)}")
    if only_pali:
        log.warning(f"  Tales in Pali but not English: {only_pali[:20]}")
    if only_eng:
        log.warning(f"  Tales in English but not Pali: {only_eng[:20]}")

    # ── Translate matched tales ────────────────────────────────────────────────
    for tale_num in common_nums:
        pali_ch  = pali_chapters[tale_num]
        eng_ch   = eng_chapters[tale_num]

        p_book_id  = pali_ch["book_id"]
        p_para_id  = pali_ch["para_id"]
        p_title    = pali_ch["title"]
        eng_title  = eng_ch["title"]
        eng_passage = eng_ch["content"]

        log.info(f"  Tale {tale_num}: '{p_title}' ↔ '{eng_title}'")

        # Pali sentences for this heading's range
        sentences, para_start, para_end = get_sentences_for_chapter(
            nissaya_con, p_book_id, p_para_id
        )

        if not sentences:
            log.warning(f"    No Pali sentences found — skipping")
            continue

        # ── Skip if already fully translated ─────────────────────────────────
        sentence_keys = {(s["para_id"], s["line_id"]) for s in sentences}
        done_keys = {
            (row[0], row[1])
            for row in out_con.execute("""
                SELECT para_id, line_id FROM sentences
                WHERE book_id = ?
                  AND para_id >= ? AND para_id <= ?
                  AND english_translation IS NOT NULL
                  AND english_translation != ''
            """, (p_book_id, min(s["para_id"] for s in sentences),
                  max(s["para_id"] for s in sentences))).fetchall()
        }
        if len(done_keys) > 5:
            log.info(f"    ✓ Skipping")
            continue
        if sentence_keys == done_keys:
            log.info(f"    ✓ Already done ({len(sentence_keys)} sentences) — skipping")
            continue

        log.info(f"    {len(sentences)} sentences, passage {len(eng_passage)} chars "
                 f"({len(done_keys)} already translated)")

        # Translate
        translations = translate_chapter(sentences, eng_passage, p_title, p_book_id)

        # Persist to output DB
        with DB_WRITE_LOCK:
            for s in sentences:
                key = (s["para_id"], s["line_id"])
                eng_trans = translations.get(key, "")
                out_con.execute("""
                    INSERT OR REPLACE INTO sentences
                      (book_id, para_id, line_id, pali_sentence, english_translation)
                    VALUES (?, ?, ?, ?, ?)
                """, (s["book_id"], s["para_id"], s["line_id"],
                      s["pali_sentence"], eng_trans))
            out_con.commit()

        log.info(f"    ✓ Saved {len(sentences)} sentences")

    nissaya_con.close()
    out_con.close()
    log.info("✓ Jātaka pipeline complete")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Jātaka translator — matches EPUB HTML chapters with Pali "
                    "headings from nissaya.db by tale number."
    )
    parser.add_argument("--nissaya", default="../data/nissaya.db",
                        help="Path to nissaya.db")
    parser.add_argument("--jataka", required=True,
                        help="Root folder of the extracted Jātaka EPUB "
                             "(e.g. ./jataka)")
    parser.add_argument("--db", default="jataka_out.db",
                        help="Output SQLite database")
    parser.add_argument("--model", default="gemini-2.5-flash-preview-05-20",
                        help="Gemini model name")
    args = parser.parse_args()

    for p in [args.nissaya, args.jataka]:
        if not Path(p).exists():
            sys.exit(f"ERROR: path not found: {p}")

    MODEL_NAME = args.model

    process_jataka(args.nissaya, args.jataka, args.db)

    log.info("Pipeline complete")