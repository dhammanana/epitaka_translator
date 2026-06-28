"""
Visuddhimagga Match + Translate Pipeline
=========================================

Overview
--------
nissaya.db  — Pali source, split across two books:
                  headings  (book_id, para_id, title, level, parent)
                  sentences (book_id, para_id, line_id, pali_sentence, …)
              Vism-i and Vism-ii are the two physical volumes of ONE book.
              They are treated as a single ordered sequence (Vism-i first).

vism.db     — English published translation:
                  sections  (id, chapter, chapter_title, heading, heading_level, content)
                  matches   (book_id, para_id, title, id, chapter, heading, confidence)
              The chapter/chapter_title columns identify top-level chapters;
              heading / heading_level identify sub-sections within a chapter.

Pipeline
--------
Stage 1  Match Pali chapters (level=2 headings from combined Vism-i + Vism-ii)
         to English chapters in vism.db → saved to matches (level=2).

Stage 2  For each matched chapter, match Pali inner headings (level=3)
         to English headings (heading_level=2 rows in that chapter)
         → saved to matches (level=3, chapter_match_id = stage-1 row).

Stage 3  For each matched heading pair, fetch the corresponding English passage
         (sections.content) from vism.db and the Pali sentences from nissaya.db.
         The AI is given the English passage as a reference and must map each Pali
         sentence to the corresponding portion of that passage.
         If a sentence cannot be found in the passage, "" is returned — the AI
         must NOT infer or fabricate a translation.
         Results stored in vism.db → sentences table.
         Progress is tracked as a single row (last completed match_id) in
         _trans_progress so the run can be resumed after interruption.

Stage retranslate
         For each section in matches, if the number of untranslated sentences
         (english_translation IS NULL or '') exceeds 10, those sentences are
         sent to the AI again. The AI must attempt to translate each one; if it
         cannot, it must return a structured error object explaining why.
         Errors are saved to translation_errors for manual review.

Logs
----
  headings.log    — overwritten on every AI call during stages 1+2
  translation.log — overwritten on every AI call during stage 3 / retranslate
  vism_pipeline.log — append-only full run log

Usage
-----
    python vism_match_translate.py \\
        --nissaya data/nissaya.db \\
        --vism    data/vism.db   \\
        [--workers 4] [--dry-run] [--stage 12|3|retranslate|all]
"""

import os, re, sys, json, time, sqlite3, logging, threading, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from google import genai
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Overwriting log files (one per stage group)
# ─────────────────────────────────────────────

HEADINGS_LOG     = "headings.log"
TRANSLATION_LOG  = "translation.log"


def _overwrite(path: str, content: str):
    """Overwrite a file with the given content (UTF-8). Thread-safe via a lock."""
    with _OVERWRITE_LOCK:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            log.warning(f"Could not write {path}: {e}")


_OVERWRITE_LOCK = threading.Lock()

# ─────────────────────────────────────────────
# Logging (console + append-only pipeline log)
# ─────────────────────────────────────────────

class ColorFormatter(logging.Formatter):
    _C = {
        logging.DEBUG:    "\033[37m",
        logging.INFO:     "\033[36m",
        logging.WARNING:  "\033[33m",
        logging.ERROR:    "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    _R = "\033[0m"
    _G = "\033[90m"

    def format(self, r):
        c = self._C.get(r.levelno, "")
        return (f"{self._G}{self.formatTime(r, '%H:%M:%S')}{self._R} "
                f"{c}{r.levelname:<8}{self._R} {r.getMessage()}")


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

_con = logging.StreamHandler(sys.stdout)
_con.setFormatter(ColorFormatter())
_fil = logging.FileHandler("vism_pipeline.log", encoding="utf-8")
_fil.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_con)
logging.root.addHandler(_fil)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Gemini key rotator
# ─────────────────────────────────────────────

class AllKeysExhaustedError(RuntimeError):
    pass


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
                raise AllKeysExhaustedError("All keys exhausted.")
            k = self._keys[self._index % len(self._keys)]
            self._index = (self._index + 1) % len(self._keys)
            return k

    def remove(self, key: str):
        with self._lock:
            if key in self._keys:
                idx = self._keys.index(key)
                self._keys.remove(key)
                log.warning(f"Key …{key[-6:]} removed. {len(self._keys)} remaining.")
                if self._keys:
                    if self._index > idx:
                        self._index -= 1
                    self._index %= len(self._keys)
                else:
                    self._index = 0


ROTATOR: Optional[KeyRotator] = None
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
DB_WRITE_LOCK = threading.Lock()

# ─────────────────────────────────────────────
# Gemini call
# ─────────────────────────────────────────────

def call_gemini(prompt: str, timeout: int = 300) -> str:
    attempt = 0
    while True:
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
            log.error(f"Timeout attempt {attempt}, retrying.")
            continue

        if result["error"]:
            e = result["error"]
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status == 429:
                ROTATOR.remove(key)
                continue
            if status in (401, 403):
                log.warning(f"Auth error key …{key[-6:]}, skipping.")
                continue
            log.warning(f"Error attempt {attempt}: {e}. Retrying in 20s.")
            time.sleep(20)
            continue

        if result["response"] is not None:
            return result["response"]


def parse_json_response(response: str, label: str = "") -> list:
    """Strip markdown fences and parse JSON array; fall back to object-by-object recovery."""
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
# DB helpers
# ─────────────────────────────────────────────

def open_db(path: str, read_only: bool = False) -> sqlite3.Connection:
    uri = f"file:{path}{'?mode=ro' if read_only else ''}".replace("\\", "/")
    con = sqlite3.connect(uri, uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    if not read_only:
        con.execute("PRAGMA journal_mode=WAL")
    return con


def ensure_vism_tables(vism_con: sqlite3.Connection):
    """Create pipeline tables in vism.db if they don't exist yet."""
    vism_con.executescript("""
        CREATE TABLE IF NOT EXISTS sentences (
            book_id              TEXT    NOT NULL,
            para_id              INTEGER NOT NULL,
            line_id              INTEGER NOT NULL,
            section_id           INTEGER,
            vripara              TEXT,
            thaipage             TEXT,
            vripage              TEXT,
            ptspage              TEXT,
            mypage               TEXT,
            pali_sentence        TEXT,
            english_translation  TEXT,
            PRIMARY KEY (book_id, para_id, line_id)
        );

        -- Single-row progress table: stores only the last completed match_id.
        -- A fresh run starts from the beginning (no row / last_match_id = 0).
        -- On resume, processing continues from last_match_id + 1.
        CREATE TABLE IF NOT EXISTS _trans_progress (
            id            INTEGER PRIMARY KEY CHECK (id = 1),  -- enforces single row
            last_match_id INTEGER NOT NULL DEFAULT 0
        );

        -- Errors from retranslation passes: sentences the AI could not align.
        CREATE TABLE IF NOT EXISTS translation_errors (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id     TEXT    NOT NULL,
            para_id     INTEGER NOT NULL,
            line_id     INTEGER NOT NULL,
            match_id    INTEGER NOT NULL,
            error_type  TEXT,   -- e.g. "unmatched", "ambiguous", "passage_missing"
            reason      TEXT,   -- AI's explanation
            created_at  TEXT DEFAULT (datetime('now'))
        );
    """)
    vism_con.commit()

# ─────────────────────────────────────────────
# Progress helpers  (single-row _trans_progress)
# ─────────────────────────────────────────────

def get_last_completed_match_id(vism_con: sqlite3.Connection) -> int:
    """Return the last successfully completed match_id, or 0 if none."""
    row = vism_con.execute(
        "SELECT last_match_id FROM _trans_progress WHERE id = 1"
    ).fetchone()
    return row["last_match_id"] if row else 0


def save_last_completed_match_id(vism_con: sqlite3.Connection, match_id: int):
    """Upsert the single progress row with the latest completed match_id."""
    with DB_WRITE_LOCK:
        vism_con.execute("""
            INSERT INTO _trans_progress (id, last_match_id) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET last_match_id = excluded.last_match_id
        """, (match_id,))
        vism_con.commit()

# ─────────────────────────────────────────────
# Pali heading helpers  (nissaya.db)
# ─────────────────────────────────────────────

PALI_BOOKS = ("Vism-i", "Vism-ii")


def get_pali_chapters(nissaya_con: sqlite3.Connection) -> list:
    rows = []
    for book_id in PALI_BOOKS:
        book_rows = nissaya_con.execute("""
            SELECT book_id, para_id, title, level, parent
            FROM headings
            WHERE book_id = ? AND level = 2
            ORDER BY para_id
        """, (book_id,)).fetchall()
        rows.extend(book_rows)
    return rows


def get_pali_inner_headings(nissaya_con: sqlite3.Connection,
                             book_id: str, chapter_para_id: int) -> list:
    return nissaya_con.execute("""
        SELECT book_id, para_id, title, level, parent
        FROM headings
        WHERE book_id = ? AND level = 3 AND parent = ?
        ORDER BY para_id
    """, (book_id, chapter_para_id)).fetchall()

# ─────────────────────────────────────────────
# English section helpers  (vism.db)
# ─────────────────────────────────────────────

def get_eng_chapters(vism_con: sqlite3.Connection) -> list:
    return vism_con.execute("""
        SELECT MIN(id) AS id, chapter, chapter_title
        FROM sections
        GROUP BY chapter
        ORDER BY MIN(id)
    """).fetchall()


def get_eng_headings_for_chapter(vism_con: sqlite3.Connection, chapter: str) -> list:
    return vism_con.execute("""
        SELECT id, chapter, chapter_title, heading, heading_level
        FROM sections
        WHERE chapter = ? AND heading IS NOT NULL AND heading != ''
        ORDER BY id
    """, (chapter,)).fetchall()


def get_eng_section_content(vism_con: sqlite3.Connection, section_id: int) -> str:
    """Return the full text content of a section row, or '' if missing."""
    row = vism_con.execute(
        "SELECT content FROM sections WHERE id = ?", (section_id,)
    ).fetchone()
    if row and row["content"]:
        return row["content"].strip()
    return ""


def get_eng_chapter_content(vism_con: sqlite3.Connection, section_id: int) -> str:
    """
    Return the combined content of all sections in the same chapter as section_id,
    joined in order. Falls back to the single section content if chapter is not found.
    """
    row = vism_con.execute(
        "SELECT chapter FROM sections WHERE id = ?", (section_id,)
    ).fetchone()
    if not row or not row["chapter"]:
        return get_eng_section_content(vism_con, section_id)

    chapter = row["chapter"]
    rows = vism_con.execute("""
        SELECT content FROM sections
        WHERE chapter = ? AND content IS NOT NULL AND content != ''
        ORDER BY id
    """, (chapter,)).fetchall()

    if not rows:
        return ""
    return "\n\n".join(r["content"].strip() for r in rows)

# ─────────────────────────────────────────────
# matches table helpers  (replaces heading_matches)
# ─────────────────────────────────────────────
# Schema (already in vism.db, not created here):
#   matches (book_id TEXT, para_id INTEGER, title TEXT,
#            id INTEGER, chapter TEXT, heading TEXT, confidence TEXT)
#
# • book_id, para_id, title  — from nissaya.db sentences / headings
# • id, chapter, heading     — from vism.db sections
# • confidence               — AI-assigned match confidence

def get_all_matches(vism_con: sqlite3.Connection) -> list:
    """Return all rows from matches ordered by their vism section id."""
    return vism_con.execute(
        "SELECT rowid AS match_rowid, * FROM heading_matches ORDER BY rowid"
    ).fetchall()

# ─────────────────────────────────────────────
# AI matching helpers
# ─────────────────────────────────────────────

def ai_match_chapters(pali_chapters: list, eng_chapters: list) -> list[dict]:
    pali_list = [
        {"book_id": r["book_id"], "para_id": r["para_id"], "title": r["title"]}
        for r in pali_chapters
    ]
    eng_list = [
        {"section_id": r["id"], "chapter": r["chapter"],
         "chapter_title": r["chapter_title"]}
        for r in eng_chapters
    ]

    prompt = f"""You are matching chapter headings between two editions of the same text:
the Pali Visuddhimagga (combined Vism-i + Vism-ii volumes) and an English translation.

Pali chapters (combined, in order):
{json.dumps(pali_list, ensure_ascii=False, indent=2)}

English chapters:
{json.dumps(eng_list, ensure_ascii=False, indent=2)}

Match each Pali chapter to its corresponding English chapter.
The titles may differ in language and wording but refer to the same content.
Omit any entry that has no clear match.

Return ONLY a JSON array, no preamble, no markdown:
[{{"pali_book_id": "Vism-i", "pali_para_id": 10, "eng_section_id": 1}}, ...]
"""
    _overwrite(HEADINGS_LOG,
               f"=== CHAPTER MATCHING PROMPT ===\n\n{prompt}\n")
    response = call_gemini(prompt, timeout=180)
    _overwrite(HEADINGS_LOG,
               f"=== CHAPTER MATCHING PROMPT ===\n\n{prompt}\n\n"
               f"=== RESPONSE ===\n\n{response}\n")
    return parse_json_response(response, "chapter matching")


def ai_match_headings(pali_headings: list, eng_headings: list,
                      chapter_title: str) -> list[dict]:
    pali_list = [{"para_id": r["para_id"], "title": r["title"]}
                 for r in pali_headings]
    eng_list  = [{"section_id": r["id"], "heading": r["heading"]}
                 for r in eng_headings]

    prompt = f"""You are matching sub-section headings within the chapter:
"{chapter_title}"

This is from the Pali Visuddhimagga (one volume) vs its English translation.

Pali headings:
{json.dumps(pali_list, ensure_ascii=False, indent=2)}

English headings:
{json.dumps(eng_list, ensure_ascii=False, indent=2)}

Match each Pali heading to its corresponding English heading.
Omit entries with no clear match.

Return ONLY a JSON array, no preamble, no markdown:
[{{"pali_para_id": 10, "eng_section_id": 3}}, ...]
"""
    _overwrite(HEADINGS_LOG,
               f"=== HEADING MATCHING: {chapter_title} ===\n\n{prompt}\n")
    response = call_gemini(prompt, timeout=120)
    _overwrite(HEADINGS_LOG,
               f"=== HEADING MATCHING: {chapter_title} ===\n\n{prompt}\n\n"
               f"=== RESPONSE ===\n\n{response}\n")
    return parse_json_response(response, f"heading matching [{chapter_title}]")

# ─────────────────────────────────────────────
# Stage 1 + 2: matching
# ─────────────────────────────────────────────

def run_stage_1_2(nissaya_con: sqlite3.Connection, vism_con: sqlite3.Connection):
    # NOTE: stages 1+2 write into the pre-existing `matches` table in vism.db.
    # The table schema is owned by vism.db; we only INSERT here.
    # If you want a clean re-run, truncate `matches` manually before starting.

    log.info("Stage 1: matching Pali chapters → English chapters…")
    pali_chapters = get_pali_chapters(nissaya_con)
    eng_chapters  = get_eng_chapters(vism_con)
    log.info(f"  Pali chapters (combined): {len(pali_chapters)}")
    log.info(f"  English chapters:         {len(eng_chapters)}")

    chapter_pairs = ai_match_chapters(pali_chapters, eng_chapters)
    log.info(f"  AI matched {len(chapter_pairs)} chapter pairs.")

    pali_ch_by_key = {(r["book_id"], r["para_id"]): r for r in pali_chapters}
    eng_ch_by_id   = {r["id"]: r for r in eng_chapters}
    chapter_match_ids = []

    for pair in chapter_pairs:
        p_book = pair.get("pali_book_id")
        p_para = pair.get("pali_para_id")
        e_sid  = pair.get("eng_section_id")

        p_row = pali_ch_by_key.get((p_book, p_para))
        e_row = eng_ch_by_id.get(e_sid)

        if not p_row or not e_row:
            log.warning(f"  Skipping invalid pair: {pair}")
            continue

        with DB_WRITE_LOCK:
            cur = vism_con.execute("""
                INSERT INTO matches
                  (book_id, para_id, title, id, chapter, heading, confidence)
                VALUES (?, ?, ?, ?, ?, NULL, 'high')
            """, (p_book, p_para, p_row["title"],
                  e_sid, e_row["chapter"]))
            mid = cur.lastrowid
            vism_con.commit()

        chapter_match_ids.append((mid, p_book, p_para, e_row["chapter"],
                                   e_row["chapter_title"]))

    log.info(f"  Saved {len(chapter_match_ids)} chapter matches.")

    log.info("Stage 2: matching inner headings per chapter…")
    total_inner = 0

    for match_rowid, p_book, p_para, eng_chapter, chapter_title in chapter_match_ids:
        pali_inner = get_pali_inner_headings(nissaya_con, p_book, p_para)
        eng_inner  = get_eng_headings_for_chapter(vism_con, eng_chapter)

        if not pali_inner or not eng_inner:
            log.info(f"  '{chapter_title}': no inner headings on one side, skipping.")
            continue

        log.info(f"  '{chapter_title}': {len(pali_inner)} Pali ↔ {len(eng_inner)} English headings")
        pairs = ai_match_headings(pali_inner, eng_inner, chapter_title)

        pali_inner_by_para = {r["para_id"]: r for r in pali_inner}
        eng_inner_by_id    = {r["id"]: r    for r in eng_inner}

        saved = 0
        for pair in pairs:
            p_p = pair.get("pali_para_id")
            e_s = pair.get("eng_section_id")
            p_row = pali_inner_by_para.get(p_p)
            e_row = eng_inner_by_id.get(e_s)
            if not p_row or not e_row:
                continue

            with DB_WRITE_LOCK:
                vism_con.execute("""
                    INSERT INTO matches
                      (book_id, para_id, title, id, chapter, heading, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, 'high')
                """, (p_book, p_p, p_row["title"],
                      e_s, e_row["chapter"], e_row["heading"]))
                vism_con.commit()
            saved += 1

        total_inner += saved
        log.info(f"    Saved {saved} heading matches.")

    log.info(f"Stage 1+2 done. Chapter: {len(chapter_match_ids)}, Inner headings: {total_inner}")

# ─────────────────────────────────────────────
# Stage 3: reference-passage translation
# ─────────────────────────────────────────────

TRANSLATE_SYSTEM = """\
You are a scholarly assistant helping to align a Pali text with its published English translation.

You will be given:
1. ENGLISH PASSAGE — the published English translation of a section of the Visuddhimagga.
2. PALI SENTENCES  — a JSON array of Pali sentences from the same section, each identified
   by para_id and line_id.

Your task:
  For each Pali sentence, find the part of the ENGLISH PASSAGE that corresponds to it and
  return that English text as the "english_translation" value.

STRICT RULES:
  • Only use text that actually appears in the ENGLISH PASSAGE provided.
  • Do NOT translate, paraphrase, infer, or fabricate anything.
  • If you cannot find a clear match for a sentence in the passage, return "" for that entry.
  • Do not skip any sentence — every para_id/line_id must appear in your output.
  • Keep Pali technical terms exactly as they appear in the passage (e.g. sīla, samādhi).
  • Translate if some line's meaning can be inferred based on the provided text.

Return ONLY a JSON array, no preamble, no markdown fences:
[{"para_id": 1, "line_id": 1, "english_translation": "..."}, ...]
"""


def get_para_range_for_match(nissaya_con: sqlite3.Connection,
                              book_id: str, para_id: int) -> tuple[int, int]:
    """
    Determine the para_id range covered by the heading at (book_id, para_id).
    The range ends just before the next heading at the same or higher level.
    """
    row = nissaya_con.execute(
        "SELECT para_id, parent, level FROM headings WHERE book_id=? AND para_id=?",
        (book_id, para_id)
    ).fetchone()
    if not row:
        return para_id, para_id

    next_row = nissaya_con.execute("""
        SELECT para_id FROM headings
        WHERE book_id = ? AND para_id > ? AND level <= ?
        ORDER BY para_id LIMIT 1
    """, (book_id, para_id, row["level"])).fetchone()

    end = (next_row["para_id"] - 1) if next_row else 999_999
    return para_id, end


def get_sentences(nissaya_con: sqlite3.Connection,
                  book_id: str, para_start: int, para_end: int) -> list:
    return nissaya_con.execute("""
        SELECT * FROM sentences
        WHERE book_id = ? AND para_id >= ? AND para_id <= ?
        ORDER BY para_id, line_id
    """, (book_id, para_start, para_end)).fetchall()


def parse_translations(response: str) -> dict:
    """Parse translation response → {(para_id, line_id): english_translation}."""
    items = parse_json_response(response, "translation")
    result = {}
    for item in items:
        try:
            key = (int(item["para_id"]), int(item["line_id"]))
            result[key] = item.get("english_translation", "")
        except Exception:
            pass
    return result


def process_match_row(match_rowid: int, match: sqlite3.Row,
                      nissaya_path: str, vism_path: str, dry_run: bool):
    """
    For one row in `matches`:
      1. Fetch the English passage from vism.db sections.content.
      2. Fetch all Pali sentences for the heading's para range.
      3. Send both to the AI in chunks; AI maps each sentence to a span of the passage.
      4. Save results to vism.db sentences.
      5. Update the single-row progress entry with this match_rowid.
    """
    nissaya_con = open_db(nissaya_path, read_only=True)
    vism_con    = open_db(vism_path)

    try:
        p_book     = match["book_id"]
        p_para     = match["para_id"]
        p_title    = match["title"]
        section_id = match["id"]        # vism.db sections.id

        # ── Fetch English reference passage ──────────────────────────────────
        eng_passage = get_eng_section_content(vism_con, section_id)

        if not eng_passage:
            log.warning(f"  Match {match_rowid} '{p_title}': no English passage, "
                        f"all translations will be empty.")

        # ── Fetch Pali sentences ──────────────────────────────────────────────
        para_start, para_end = get_para_range_for_match(nissaya_con, p_book, p_para)
        sentences = get_sentences(nissaya_con, p_book, para_start, para_end)

        if not sentences:
            log.warning(f"  Match {match_rowid} '{p_title}': no Pali sentences, skipping.")
            save_last_completed_match_id(vism_con, match_rowid)
            return

        log.info(f"  Match {match_rowid} '{p_title}': {len(sentences)} sentences, "
                 f"passage {len(eng_passage)} chars (section_id={section_id})")

        if dry_run:
            log.info("  [DRY RUN] skipping translation.")
            save_last_completed_match_id(vism_con, match_rowid)
            return

        # ── If there is no English passage, store empty strings immediately ───
        if not eng_passage:
            with DB_WRITE_LOCK:
                for s in sentences:
                    vism_con.execute("""
                        INSERT OR REPLACE INTO sentences
                          (book_id, para_id, line_id, section_id,
                           vripara, thaipage, vripage, ptspage, mypage,
                           pali_sentence, english_translation)
                        VALUES (?,?,?,?, ?,?,?,?,?, ?,?)
                    """, (
                        s["book_id"], s["para_id"], s["line_id"], section_id,
                        s["vripara"], s["thaipage"], s["vripage"],
                        s["ptspage"], s["mypage"],
                        s["pali_sentence"], "",
                    ))
                vism_con.commit()
            save_last_completed_match_id(vism_con, match_rowid)
            log.info(f"  ✓ Match {match_rowid} complete (no passage → all empty).")
            return

        # ── Build section label for prompt context ────────────────────────────
        section_label = match["chapter"] or ""
        if match["heading"]:
            section_label += f" — {match['heading']}"

        # ── Chunked translation ───────────────────────────────────────────────
        CHUNK = 150
        chunks = [sentences[i : i + CHUNK] for i in range(0, len(sentences), CHUNK)]

        for ci, chunk in enumerate(chunks):
            payload = [
                {
                    "para_id":       s["para_id"],
                    "line_id":       s["line_id"],
                    "pali_sentence": s["pali_sentence"] or "",
                }
                for s in chunk
            ]

            prompt = (
                f"{TRANSLATE_SYSTEM}\n\n"
                f"## Section\n{section_label}\n\n"
                f"## English Passage (reference — use ONLY this text)\n"
                f"{eng_passage}\n\n"
                f"## Pali Sentences to align (chunk {ci + 1}/{len(chunks)})\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            )

            _overwrite(
                TRANSLATION_LOG,
                f"=== TRANSLATION CALL: match={match_rowid} "
                f"chunk={ci + 1}/{len(chunks)} '{p_title}' ===\n\n"
                f"PROMPT\n{'─'*60}\n{prompt}\n\n"
                f"[waiting for response…]\n"
            )

            try:
                response     = call_gemini(prompt)
                translations = parse_translations(response)
            except AllKeysExhaustedError:
                raise
            except Exception as e:
                log.error(f"  Chunk {ci + 1} failed: {e}")
                translations = {}
                response = f"ERROR: {e}"

            _overwrite(
                TRANSLATION_LOG,
                f"=== TRANSLATION CALL: match={match_rowid} "
                f"chunk={ci + 1}/{len(chunks)} '{p_title}' ===\n\n"
                f"PROMPT\n{'─'*60}\n{prompt}\n\n"
                f"RESPONSE\n{'─'*60}\n{response}\n\n"
                f"PARSED: {len(translations)} entries\n"
            )

            with DB_WRITE_LOCK:
                for s in chunk:
                    key = (s["para_id"], s["line_id"])
                    eng = translations.get(key, "")
                    vism_con.execute("""
                        INSERT OR REPLACE INTO sentences
                          (book_id, para_id, line_id, section_id,
                           vripara, thaipage, vripage, ptspage, mypage,
                           pali_sentence, english_translation)
                        VALUES (?,?,?,?, ?,?,?,?,?, ?,?)
                    """, (
                        s["book_id"], s["para_id"], s["line_id"], section_id,
                        s["vripara"], s["thaipage"], s["vripage"],
                        s["ptspage"], s["mypage"],
                        s["pali_sentence"], eng,
                    ))
                vism_con.commit()

            log.info(f"    Chunk {ci + 1}/{len(chunks)} saved "
                     f"({len(chunk)} rows, {len(translations)} matched).")

        # ── Mark progress: record this match_rowid as the last completed ──────
        save_last_completed_match_id(vism_con, match_rowid)
        log.info(f"  ✓ Match {match_rowid} complete.")

    finally:
        nissaya_con.close()
        vism_con.close()


def run_stage_3(nissaya_path: str, vism_path: str, workers: int, dry_run: bool):
    log.info("Stage 3: aligning Pali sentences to English passage…")

    vism_con = open_db(vism_path)
    matches  = get_all_matches(vism_con)

    # Resume: skip everything up to and including the last completed match_rowid.
    last_done = get_last_completed_match_id(vism_con)
    vism_con.close()

    pending = [(r["match_rowid"], r) for r in matches if r["match_rowid"] > last_done]
    skipped = len(matches) - len(pending)
    log.info(f"  {len(pending)} matches pending "
             f"({skipped} already done, last_match_id={last_done}).")

    # Stage 3 MUST be sequential so the single-row progress stays meaningful.
    # (If parallel execution is desired in future, switch to per-row tracking.)
    completed = failed = 0
    for match_rowid, row in pending:
        try:
            process_match_row(match_rowid, row, nissaya_path, vism_path, dry_run)
            completed += 1
        except AllKeysExhaustedError as e:
            log.critical(f"ALL KEYS EXHAUSTED: {e}")
            break
        except Exception as e:
            failed += 1
            log.error(f"Match {match_rowid} failed: {e}", exc_info=True)

    log.info(f"Stage 3 done. Completed: {completed}, Failed: {failed}")

# ─────────────────────────────────────────────
# Stage retranslate: fix untranslated sentences
# ─────────────────────────────────────────────

# When the AI cannot match a sentence it returns a structured error:
#   {"para_id": N, "line_id": M, "english_translation": null,
#    "error": {"type": "unmatched|ambiguous|passage_missing|other", "reason": "…"}}
# Sentences with a valid string translation are saved normally.
# Error entries are saved to translation_errors for manual review.

RETRANSLATE_SYSTEM = """\
You are a scholarly assistant helping to align a Pali text with its published English translation.

You will be given:
1. ENGLISH PASSAGE — the published English translation of a section of the Visuddhimagga.
2. PALI SENTENCES  — a JSON array of Pali sentences that were NOT yet translated,
   each identified by para_id and line_id.

Your task:
  For each Pali sentence, find the part of the ENGLISH PASSAGE that corresponds to it and
  return that English text as the "english_translation" value.

STRICT RULES:
  • Only use text that actually appears in the ENGLISH PASSAGE provided.
  • Do NOT paraphrase or fabricate anything.
  • If you CAN find the matching passage, return it in "english_translation"
    and omit the "error" key entirely.
  • If you CANNOT find a clear match, return "english_translation": null and include
    an "error" object with:
      - "type": one of "unmatched" | "ambiguous" | "passage_missing" | "other"
      - "reason": a brief explanation of why you could not align this sentence
  • Do not skip any sentence — every para_id/line_id must appear in your output.

Return ONLY a JSON array, no preamble, no markdown fences:
[
  {"para_id": 1, "line_id": 1, "english_translation": "The translated text…"},
  {"para_id": 1, "line_id": 2, "english_translation": null,
   "error": {"type": "unmatched", "reason": "No matching phrase found in passage."}}
]
"""

RETRANSLATE_THRESHOLD = 10   # minimum untranslated sentences to trigger a re-run


def get_untranslated_sentences(vism_con: sqlite3.Connection,
                                nissaya_con: sqlite3.Connection,
                                match: sqlite3.Row) -> list:
    """
    Return sentences for this match that have no english_translation yet
    (NULL or empty string) and exist in the sentences table.
    """
    p_book = match["book_id"]
    p_para = match["para_id"]

    # Determine the para_id range the same way stage 3 does.
    para_start, para_end = get_para_range_for_match(nissaya_con, p_book, p_para)

    return vism_con.execute("""
        SELECT * FROM sentences
        WHERE book_id = ? AND para_id >= ? AND para_id <= ?
          AND (english_translation IS NULL OR english_translation = '')
        ORDER BY para_id, line_id
    """, (p_book, para_start, para_end)).fetchall()


def parse_retranslations(response: str) -> tuple[dict, list]:
    """
    Parse retranslation response.
    Returns:
        translations — {(para_id, line_id): english_translation}  (non-null entries)
        errors       — [{"para_id": N, "line_id": M, "error": {...}}]
    """
    items = parse_json_response(response, "retranslation")
    translations = {}
    errors = []
    for item in items:
        try:
            key = (int(item["para_id"]), int(item["line_id"]))
            eng = item.get("english_translation")
            err = item.get("error")
            if eng is not None and eng != "":
                translations[key] = eng
            elif err:
                errors.append({
                    "para_id":  int(item["para_id"]),
                    "line_id":  int(item["line_id"]),
                    "error":    err,
                })
            # If both are absent/empty, treat as error with type "other"
            else:
                errors.append({
                    "para_id":  int(item["para_id"]),
                    "line_id":  int(item["line_id"]),
                    "error":    {"type": "other", "reason": "Empty response from AI."},
                })
        except Exception:
            pass
    return translations, errors


def process_retranslate_match(match_rowid: int, match: sqlite3.Row,
                               nissaya_path: str, vism_path: str, dry_run: bool):
    nissaya_con = open_db(nissaya_path, read_only=True)
    vism_con    = open_db(vism_path)

    try:
        p_book     = match["book_id"]
        p_para     = match["para_id"]
        p_title    = match["title"]
        section_id = match["id"]

        # ── Find untranslated sentences ───────────────────────────────────────
        untranslated = get_untranslated_sentences(vism_con, nissaya_con, match)

        if len(untranslated) <= RETRANSLATE_THRESHOLD:
            # Not enough to bother (should have been filtered by caller, but be safe)
            return

        eng_passage = get_eng_chapter_content(vism_con, section_id)
        if not eng_passage:
            log.warning(f"  Match {match_rowid}: no English chapter content, cannot retranslate.")
            return

        log.info(f"  Retranslate match {match_rowid} '{p_title}': "
                 f"{len(untranslated)} untranslated sentences.  "
                 f"Chapter English passage: {len(eng_passage)} chars")

        section_label = match["chapter"] or ""
        if match["heading"]:
            section_label += f" — {match['heading']}"

        if dry_run:
            log.info("  [DRY RUN] skipping retranslation.")
            return

        # ── Chunked retranslation ─────────────────────────────────────────────
        CHUNK = 150
        chunks = [untranslated[i : i + CHUNK] for i in range(0, len(untranslated), CHUNK)]
        total_fixed = total_errors = 0

        for ci, chunk in enumerate(chunks):
            payload = [
                {
                    "para_id":       s["para_id"],
                    "line_id":       s["line_id"],
                    "pali_sentence": s["pali_sentence"] or "",
                }
                for s in chunk
            ]

            prompt = (
                f"{RETRANSLATE_SYSTEM}\n\n"
                f"## Section\n{section_label}\n\n"
                f"## English Passage (reference — use ONLY this text)\n"
                f"{eng_passage}\n\n"
                f"## Untranslated Pali Sentences (chunk {ci + 1}/{len(chunks)})\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            )

            _overwrite(
                TRANSLATION_LOG,
                f"=== RETRANSLATE CALL: match={match_rowid} "
                f"chunk={ci + 1}/{len(chunks)} '{p_title}' ===\n\n"
                f"PROMPT\n{'─'*60}\n{prompt}\n\n"
                f"[waiting for response…]\n"
            )

            try:
                response = call_gemini(prompt)
                translations, errors = parse_retranslations(response)
            except AllKeysExhaustedError:
                raise
            except Exception as e:
                log.error(f"  Retranslate chunk {ci + 1} failed: {e}")
                translations, errors = {}, []
                response = f"ERROR: {e}"

            _overwrite(
                TRANSLATION_LOG,
                f"=== RETRANSLATE CALL: match={match_rowid} "
                f"chunk={ci + 1}/{len(chunks)} '{p_title}' ===\n\n"
                f"PROMPT\n{'─'*60}\n{prompt}\n\n"
                f"RESPONSE\n{'─'*60}\n{response}\n\n"
                f"PARSED: {len(translations)} fixed, {len(errors)} errors\n"
            )

            with DB_WRITE_LOCK:
                # Save successful translations
                for s in chunk:
                    key = (s["para_id"], s["line_id"])
                    if key in translations:
                        vism_con.execute("""
                            UPDATE sentences
                            SET english_translation = ?
                            WHERE book_id = ? AND para_id = ? AND line_id = ?
                        """, (translations[key], s["book_id"],
                              s["para_id"], s["line_id"]))

                # Save error records
                for err_item in errors:
                    err_obj = err_item["error"]
                    vism_con.execute("""
                        INSERT OR REPLACE INTO translation_errors
                          (book_id, para_id, line_id, match_id, error_type, reason)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        p_book,
                        err_item["para_id"],
                        err_item["line_id"],
                        match_rowid,
                        err_obj.get("type", "other"),
                        err_obj.get("reason", ""),
                    ))

                vism_con.commit()

            total_fixed  += len(translations)
            total_errors += len(errors)
            log.info(f"    Chunk {ci + 1}/{len(chunks)}: "
                     f"{len(translations)} fixed, {len(errors)} errors logged.")

        log.info(f"  ✓ Match {match_rowid} retranslate done. "
                 f"Fixed: {total_fixed}, Errors: {total_errors}.")

    finally:
        nissaya_con.close()
        vism_con.close()


def run_stage_retranslate(nissaya_path: str, vism_path: str,
                           workers: int, dry_run: bool):
    log.info(f"Stage retranslate: scanning for sections with >{RETRANSLATE_THRESHOLD} "
             f"untranslated sentences…")

    nissaya_con = open_db(nissaya_path, read_only=True)
    vism_con    = open_db(vism_path)
    matches     = get_all_matches(vism_con)

    # Find matches worth re-processing
    pending = []
    for m in matches:
        p_book = m["book_id"]
        p_para = m["para_id"]
        para_start, para_end = get_para_range_for_match(nissaya_con, p_book, p_para)

        count = vism_con.execute("""
            SELECT COUNT(*) FROM sentences
            WHERE book_id = ? AND para_id >= ? AND para_id <= ?
              AND (english_translation IS NULL OR english_translation = '')
        """, (p_book, para_start, para_end)).fetchone()[0]

        if count > RETRANSLATE_THRESHOLD:
            pending.append((m["match_rowid"], m, count))

    nissaya_con.close()
    vism_con.close()

    log.info(f"  {len(pending)} sections qualify for retranslation.")

    completed = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_retranslate_match,
                      mid, row, nissaya_path, vism_path, dry_run): (mid, cnt)
            for mid, row, cnt in pending
        }
        for future in as_completed(futures):
            mid, cnt = futures[future]
            try:
                future.result()
                completed += 1
            except AllKeysExhaustedError as e:
                log.critical(f"ALL KEYS EXHAUSTED: {e}")
                ex.shutdown(wait=False, cancel_futures=True)
                break
            except Exception as e:
                failed += 1
                log.error(f"Retranslate match {mid} failed: {e}", exc_info=True)

    log.info(f"Stage retranslate done. Completed: {completed}, Failed: {failed}.")

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    global ROTATOR, MODEL_NAME

    parser = argparse.ArgumentParser(description="Vism match + translate pipeline")
    parser.add_argument("--nissaya", default="../data/nissaya.db")
    parser.add_argument("--vism",    default="vism.db")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model",   default="gemini-2.5-flash-preview-05-20")
    parser.add_argument("--stage",   default="all",
                        choices=["12", "3", "retranslate", "all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for p in [args.nissaya, args.vism]:
        if not Path(p).exists():
            sys.exit(f"ERROR: file not found: {p}")

    ROTATOR    = KeyRotator()
    MODEL_NAME = args.model

    nissaya_con = open_db(args.nissaya, read_only=True)
    vism_con    = open_db(args.vism)
    ensure_vism_tables(vism_con)

    if args.stage in ("12", "all"):
        run_stage_1_2(nissaya_con, vism_con)

    nissaya_con.close()
    vism_con.close()

    if args.stage in ("3", "all"):
        run_stage_3(args.nissaya, args.vism, args.workers, args.dry_run)

    if args.stage == "retranslate":
        run_stage_retranslate(args.nissaya, args.vism, args.workers, args.dry_run)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()