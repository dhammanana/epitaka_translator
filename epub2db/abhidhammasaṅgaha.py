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
                continue
            if status in (401, 403):
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
# Markdown Parsing
# ─────────────────────────────────────────────

def parse_md_chapters(md_file: str) -> dict:
    """Parse markdown into chapters (level 2 headings).
    Returns {chapter_num: {"title": str, "content": str}}
    """
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chapters = {}
    current_chapter = None
    chapter_num = 0
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        match = re.match(r'^#\s+(.+)$', line)
        if match:
            # New level 2 heading
            chapter_title = match.group(1).strip()
            current_chapter = {
                'title': chapter_title,
                'lines': [],
                'start_line': i
            }
            chapters[chapter_num] = current_chapter
            chapter_num += 1
        elif current_chapter is not None and not re.match(r'^##', line):
            # Collect content until next level 2 heading
            current_chapter['lines'].append(line)
    
    # Convert lines to content string
    result = {}
    for ch_num, ch_data in chapters.items():
        content_str = '\n'.join(ch_data['lines']).strip()
        result[ch_num] = {
            'title': ch_data['title'],
            'content': content_str
        }
    
    return result

# ─────────────────────────────────────────────
# Pali Heading Retrieval
# ─────────────────────────────────────────────

def get_pali_chapters(nissaya_con: sqlite3.Connection, book_id: str) -> list:
    """Get all level 2 headings (chapters) for a book."""
    return nissaya_con.execute("""
        SELECT book_id, para_id, title, level, parent
        FROM headings
        WHERE book_id = ? AND level = 2
        ORDER BY para_id
    """, (book_id,)).fetchall()

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
                    log.info(f"      Sentences translated: {non_empty_sentences}/{total_sentences} (non-empty/total)")
                except Exception:
                    pass
        except Exception as e:
            log.error(f"      Translation failed: {e}")
        
        time.sleep(1)
    
    return translations

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def process_book(nissaya_path: str, md_path: str, book_id: str, out_db_path: str):
    """Main pipeline: match chapters and translate."""
    log.info(f"Processing {book_id}")
    
    nissaya_con = open_db(nissaya_path, read_only=True)
    out_con = open_db(out_db_path)
    ensure_abhidh_tables(out_con)
    
    # Parse Pali chapters from nissaya.db
    pali_chapters = get_pali_chapters(nissaya_con, book_id)
    log.info(f"  Found {len(pali_chapters)} Pali chapters (level 2)")
    
    # Parse markdown chapters
    md_chapters = parse_md_chapters(md_path)
    log.info(f"  Found {len(md_chapters)} Markdown chapters")
    
    # Simple matching: Pali chapter i ↔ MD chapter i
    for pali_idx, pali_ch in enumerate(pali_chapters):
        if pali_idx not in md_chapters:
            log.warning(f"  Chapter {pali_idx}: no matching MD chapter, skipping")
            continue
        
        p_para_id = pali_ch["para_id"]
        p_title = pali_ch["title"]
        md_ch = md_chapters[pali_idx]
        md_title = md_ch["title"]
        eng_passage = md_ch["content"]
        
        log.info(f"  Chapter {pali_idx}: '{p_title}' ↔ '{md_title}'")
        
        # Get Pali sentences for this chapter
        sentences, para_start, para_end = get_sentences_for_chapter(
            nissaya_con, book_id, p_para_id
        )
        
        if not sentences:
            log.warning(f"    No sentences found")
            continue
        
        log.info(f"    {len(sentences)} sentences, passage {len(eng_passage)} chars")
        
        # Translate
        translations = translate_chapter(sentences, eng_passage, p_title, book_id)
        
        # Save to output DB
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
    log.info(f"✓ {book_id} complete")

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Abhidhamma translator")
    parser.add_argument("--nissaya", default="../data/nissaya.db")
    parser.add_argument("--abhidh-s", required=True, help="Path to Abhidh-s.md")
    parser.add_argument("--abhidh-s-t", required=True, help="Path to Abhidh-s-t.md")
    parser.add_argument("--db", default="abhis.db", help="Output database")
    parser.add_argument("--model", default="gemini-2.5-flash-preview-05-20")
    args = parser.parse_args()
    
    for p in [args.nissaya, args.abhidh_s, args.abhidh_s_t]:
        if not Path(p).exists():
            sys.exit(f"ERROR: file not found: {p}")
    
    MODEL_NAME = args.model
    
    # Process both books
    process_book(args.nissaya, args.abhidh_s, "Abhidh-s", args.db)
    process_book(args.nissaya, args.abhidh_s_t, "Abhidh-s-t", args.db)
    
    log.info("Pipeline complete")