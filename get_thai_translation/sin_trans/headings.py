"""
headings.py — Build heading-level links between nissaya and sinhala editions.

New behaviour (v2):
  - For each nissaya heading, count untranslated sentences in its section.
  - If > UNTRANSLATED_THRESHOLD lines are untranslated, the heading is flagged
    for (re-)matching even if a link already exists in epitaka_link.
  - Flagged headings are batched and sent to Gemini for fresh matching.
  - Successful re-matches are saved with INSERT OR REPLACE, overwriting stale links.
"""

import logging
import json
import re
import os
import time
import threading
from db import SinhalaDB, NissayaDB, OutputDB
from google import genai


# ---------------------------------------------------------------------------
# Tuneable constant
# ---------------------------------------------------------------------------
UNTRANSLATED_THRESHOLD = 5   # sections with more than this many untranslated
                              # sentences are re-matched

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")


class ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG:    "\033[37m",
        logging.INFO:     "\033[36m",
        SUCCESS:          "\033[32m",
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


class ColorLogger(logging.Logger):
    def success(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, msg, args, **kwargs)


logging.setLoggerClass(ColorLogger)


def _make_logger() -> ColorLogger:
    logger = logging.getLogger("app")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger  # type: ignore[return-value]


logger = _make_logger()


# ---------------------------------------------------------------------------
# Gemini key rotation
# ---------------------------------------------------------------------------
class AllKeysExhaustedError(RuntimeError):
    pass


class KeyRotator:
    def __init__(self):
        self._lock = threading.Lock()
        pattern = re.compile(r"^GEMINI_KEY_\d+$")
        keys = [v.strip() for k, v in os.environ.items() if pattern.match(k) and v.strip()]
        if not keys:
            raise RuntimeError("No GEMINI_KEY_<N> environment variables found.")
        logger.info(f"Loaded {len(keys)} Gemini API key(s).")
        self._keys = list(keys)
        self._index = 0

    def next(self) -> str:
        with self._lock:
            if not self._keys:
                raise AllKeysExhaustedError("All API keys exhausted.")
            key = self._keys[self._index % len(self._keys)]
            self._index = (self._index + 1) % len(self._keys)
            return key

    def remove(self, key: str):
        with self._lock:
            if key in self._keys:
                idx = self._keys.index(key)
                self._keys.remove(key)
                logger.warning(f"Key …{key[-6:]} removed (quota). {len(self._keys)} remaining.")
                if self._keys and self._index > idx:
                    self._index -= 1
                self._index = self._index % len(self._keys) if self._keys else 0


ROTATOR: KeyRotator | None = None
MODEL_NAME: str = "gemini-2.0-flash-preview"


def call_gemini(prompt: str, timeout_seconds: int = 120) -> str:
    global ROTATOR
    attempt = 0
    while True:
        attempt += 1
        key = ROTATOR.next()
        key_suffix = key[-6:]
        result: dict = {"response": None, "error": None}

        def gemini_call():
            try:
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                )
                result["response"] = response.text
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=gemini_call, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            logger.error(f"Gemini timeout (>{timeout_seconds}s) on key …{key_suffix} (attempt {attempt}).")
            continue

        if result["error"] is not None:
            e = result["error"]
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                ROTATOR.remove(key)
                continue
            if "401" in err_str or "403" in err_str:
                logger.warning(f"Key …{key_suffix} invalid — skipping.")
                ROTATOR.remove(key)
                continue
            logger.warning(f"Gemini error attempt {attempt}: {e} — retrying.")
            time.sleep(5)
            continue

        if result["response"] is None:
            logger.warning(f"Gemini returned None on key …{key_suffix} — retrying.")
            time.sleep(5)
            continue

        return result["response"]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
LINK_SYSTEM = """You are an expert in Pali Buddhist canonical texts.
You will be given two lists of headings from the same book in two different editions:
  A) Nissaya (CST) headings: each has para_id, level (lower number = higher/parent), and title in Pali
  B) Sinhala (BJT) headings: each has id, page_num, entry_order, level (lower number = leaf/child), and palitext + sinhalatext

Your task: match each Nissaya heading to the corresponding Sinhala heading entry.

Rules:
- Match by Pali title similarity.
- Nissaya level 1 is the top (root); Sinhala level 1 is the leaf (deepest)
- Only match when confident. Omit doubtful pairs.
- One nissaya para_id → one sinhala entry id maximum.

Return ONLY a JSON array, no explanation, no markdown fences:
[
  {
    "nissaya_para_id": 12,
    "sinhala_id": 345,
    "sinhala_page_num": 5,
    "sinhala_entry_order": 2
  }
]
"""

LINK_USER = """
Book pair: Nissaya [{ni_book_id}]  ↔  Sinhala [{si_filename}]

=== NISSAYA HEADINGS (CST) ===
{ni_headings}

=== SINHALA HEADINGS (BJT) ===
{si_headings}

Produce the JSON matching array now.
"""

REMATCH_SYSTEM = """You are an expert in Pali Buddhist canonical texts.
The following Nissaya headings currently have missing or incorrect Sinhala matches —
their content sections contain many untranslated sentences.

Match EACH of these Nissaya headings to the best Sinhala heading from the full list below.
Be especially careful: previous matching was wrong or absent for these entries.

Rules:
- Match by Pali title similarity (and context if needed).
- Nissaya level 1 is the top (root); Sinhala level 1 is the leaf (deepest).
- Only match when confident. Omit doubtful pairs.
- One nissaya para_id → one sinhala entry id maximum.

Return ONLY a JSON array, no explanation, no markdown fences:
[
  {
    "nissaya_para_id": 12,
    "sinhala_id": 345,
    "sinhala_page_num": 5,
    "sinhala_entry_order": 2
  }
]
"""

REMATCH_USER = """
Book pair: Nissaya [{ni_book_id}]  ↔  Sinhala [{si_filename}]

=== NISSAYA HEADINGS TO RE-MATCH (untranslated sections) ===
{ni_headings}

=== ALL SINHALA HEADINGS (BJT) ===
{si_headings}

Produce the JSON matching array now.
"""


def init_rotator(model: str):
    global ROTATOR, MODEL_NAME
    ROTATOR = KeyRotator()
    MODEL_NAME = model


# ---------------------------------------------------------------------------
# Schema / DB helpers
# ---------------------------------------------------------------------------
def init_link_schema(out_db: OutputDB):
    out_db.conn.executescript("""
        CREATE TABLE IF NOT EXISTS epitaka_link (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            epitaka_book_id     TEXT    NOT NULL,
            epitaka_para_id     INTEGER NOT NULL,
            nissaya_title       TEXT,
            sinhala_filename    TEXT    NOT NULL,
            sinhala_page_num    INTEGER NOT NULL,
            sinhala_entry_order INTEGER NOT NULL,
            sinhala_palitext    TEXT,
            sinhala_text        TEXT,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(epitaka_book_id, epitaka_para_id, sinhala_filename)
        );
    """)
    out_db.conn.commit()


def save_links(out_db: OutputDB, book_id: str, filename: str,
               matches: list[dict], ni_headings: list[dict], si_headings: list[dict],
               replace: bool = False):
    """Persist matched heading pairs.

    When *replace* is True the rows are written with INSERT OR REPLACE so that
    stale links (from a previous run) are overwritten with the fresh match.
    """
    ni_map = {h["para_id"]: h for h in ni_headings}
    si_map = {h["id"]: h for h in si_headings}

    rows = []
    for m in matches:
        ni = ni_map.get(m["nissaya_para_id"], {})
        si = si_map.get(m["sinhala_id"], {})
        rows.append({
            "epitaka_book_id":     book_id,
            "epitaka_para_id":     m["nissaya_para_id"],
            "nissaya_title":       ni.get("title"),
            "sinhala_filename":    filename,
            "sinhala_page_num":    m["sinhala_page_num"],
            "sinhala_entry_order": m["sinhala_entry_order"],
            "sinhala_palitext":    (si.get("palitext") or "").strip() or None,
            "sinhala_text":        (si.get("sinhalatext") or "").strip() or None,
        })

    if replace:
        # DELETE the stale rows first, then re-insert cleanly.
        # (INSERT OR REPLACE would reset the auto-increment id unnecessarily.)
        for r in rows:
            out_db.conn.execute("""
                DELETE FROM epitaka_link
                WHERE epitaka_book_id = :epitaka_book_id
                  AND epitaka_para_id = :epitaka_para_id
                  AND sinhala_filename = :sinhala_filename
            """, r)

    out_db.conn.executemany("""
        INSERT OR IGNORE INTO epitaka_link
            (epitaka_book_id, epitaka_para_id, nissaya_title,
             sinhala_filename, sinhala_page_num, sinhala_entry_order,
             sinhala_palitext, sinhala_text)
        VALUES
            (:epitaka_book_id, :epitaka_para_id, :nissaya_title,
             :sinhala_filename, :sinhala_page_num, :sinhala_entry_order,
             :sinhala_palitext, :sinhala_text)
    """, rows)
    out_db.conn.commit()


def get_nissaya_headings(ni_db: NissayaDB, book_id: str) -> list[dict]:
    cur = ni_db.conn.execute("""
        SELECT para_id, level, title, chapter_len
        FROM headings
        WHERE book_id = ? AND level < 10
        ORDER BY para_id
    """, (book_id,))
    return [dict(r) for r in cur.fetchall()]


def get_sinhala_headings(si_db: SinhalaDB, filename: str) -> list[dict]:
    cur = si_db.conn.execute("""
        SELECT id, page_num, page_order, entry_order, level, palitext, sinhalatext
        FROM entries
        WHERE filename = ? AND type = 'heading'
        ORDER BY page_num, page_order, entry_order
    """, (filename,))
    return [dict(r) for r in cur.fetchall()]


def is_already_linked(out_db: OutputDB, book_id: str, filename: str) -> bool:
    cur = out_db.conn.execute(
        "SELECT COUNT(*) FROM epitaka_link WHERE epitaka_book_id = ? AND sinhala_filename = ?",
        (book_id, filename)
    )
    return cur.fetchone()[0] > 0


# ---------------------------------------------------------------------------
# NEW: untranslated-sentence audit
# ---------------------------------------------------------------------------
def get_untranslated_counts(out_db: OutputDB, book_id: str,
                            ni_headings: list[dict]) -> dict[int, int]:
    """Return {para_id: untranslated_sentence_count} for every heading section.

    A "section" for heading at index i spans para_ids from ni_headings[i]['para_id']
    up to (but not including) ni_headings[i+1]['para_id'].  The last heading runs
    to the end of the book.

    Counts rows in the *sentences* table (which lives in the output DB) where
    sinhala_translation IS NULL or empty.
    """
    result: dict[int, int] = {}
    n = len(ni_headings)
    for i, heading in enumerate(ni_headings):
        start_para = heading["para_id"]
        end_para   = ni_headings[i + 1]["para_id"] if i + 1 < n else 2_147_483_647

        cur = out_db.conn.execute("""
            SELECT COUNT(*) FROM sentences
            WHERE book_id  = ?
              AND para_id >= ?
              AND para_id <  ?
              AND (sinhala_translation IS NULL OR TRIM(sinhala_translation) = '')
        """, (book_id, start_para, end_para))
        result[start_para] = cur.fetchone()[0]

    return result


def find_headings_needing_rematch(
    out_db: OutputDB,
    book_id: str,
    filename: str,
    ni_headings: list[dict],
    threshold: int = UNTRANSLATED_THRESHOLD,
) -> list[dict]:
    """Return the subset of ni_headings whose sections have > threshold untranslated
    sentences.  These are candidates for re-matching regardless of existing links."""

    counts = get_untranslated_counts(out_db, book_id, ni_headings)
    flagged = []
    for h in ni_headings:
        pid   = h["para_id"]
        count = counts.get(pid, 0)
        if count > threshold:
            already = out_db.conn.execute("""
                SELECT COUNT(*) FROM epitaka_link
                WHERE epitaka_book_id = ? AND epitaka_para_id = ? AND sinhala_filename = ?
            """, (book_id, pid, filename)).fetchone()[0]
            status = "re-match (existing link)" if already else "new match needed"
            logger.info(
                f"    para_id={pid} '{h.get('title','?')}' — "
                f"{count} untranslated sentences → {status}"
            )
            flagged.append(h)

    return flagged


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _format_ni_block(ni_headings: list[dict]) -> str:
    return "\n".join(
        f"  para_id={h['para_id']} level={h['level']} | {h['title']}"
        for h in ni_headings
    )


def _format_si_block(si_headings: list[dict]) -> str:
    return "\n".join(
        f"  id={h['id']} page={h['page_num']} entry_order={h['entry_order']} level={h['level']} "
        f"| pali: {(h['palitext'] or '').strip()}"
        for h in si_headings
    )


def build_link_prompt(ni_book_id: str, si_filename: str,
                      ni_headings: list[dict], si_headings: list[dict]) -> str:
    return LINK_SYSTEM + "\n\n" + LINK_USER.format(
        ni_book_id=ni_book_id,
        si_filename=si_filename,
        ni_headings=_format_ni_block(ni_headings),
        si_headings=_format_si_block(si_headings),
    )


def build_rematch_prompt(ni_book_id: str, si_filename: str,
                         ni_headings: list[dict], si_headings: list[dict]) -> str:
    """Prompt variant used for headings flagged as needing re-matching."""
    return REMATCH_SYSTEM + "\n\n" + REMATCH_USER.format(
        ni_book_id=ni_book_id,
        si_filename=si_filename,
        ni_headings=_format_ni_block(ni_headings),
        si_headings=_format_si_block(si_headings),
    )


# ---------------------------------------------------------------------------
# Response parser (shared)
# ---------------------------------------------------------------------------
def parse_response(raw: str) -> list[dict]:
    """Extract a JSON array from Gemini's raw text, tolerating markdown fences."""
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        # Try to extract the first [...] block
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_link_building(sinhala_path: str, nissaya_path: str, output_path: str,
                      call_gemini=None, parse_response=None,
                      init_rotator=None,
                      # also accept the _fn-suffixed names for forward compat
                      call_gemini_fn=None, parse_response_fn=None,
                      init_rotator_fn=None,
                      model: str = "gemini-2.0-flash-preview",
                      untranslated_threshold: int = UNTRANSLATED_THRESHOLD):
    """
    Entry point.

    Accepts both old-style keyword args (call_gemini, parse_response, init_rotator)
    and new-style _fn-suffixed names — whichever is provided wins.
    """
    import sys
    _call   = call_gemini_fn   or call_gemini   or sys.modules[__name__].call_gemini
    _parse  = parse_response_fn or parse_response or sys.modules[__name__].parse_response
    _init   = init_rotator_fn  or init_rotator  or sys.modules[__name__].init_rotator

    _init(model)

    si_db  = SinhalaDB(sinhala_path)
    ni_db  = NissayaDB(nissaya_path)
    out_db = OutputDB(output_path)

    init_link_schema(out_db)

    pairs = out_db.get_book_pairs()
    if not pairs:
        logger.error("No book pairs found — run pair_books first.")
        si_db.close(); ni_db.close(); out_db.close()
        return

    logger.info(f"Building heading links for {len(pairs)} book pair(s).")

    for bp in pairs:
        ni_book_id  = bp.nissaya_book_id
        si_filename = bp.sinhala_filename

        ni_headings = get_nissaya_headings(ni_db, ni_book_id)
        si_headings = get_sinhala_headings(si_db, si_filename)

        if not ni_headings:
            logger.warning(f"  [{ni_book_id}] No nissaya headings — skipping.")
            continue
        if not si_headings:
            logger.warning(f"  [{si_filename}] No sinhala headings — skipping.")
            continue

        logger.info(
            f"  [{ni_book_id} ↔ {si_filename}] "
            f"{len(ni_headings)} ni / {len(si_headings)} si headings."
        )

        already_linked = is_already_linked(out_db, ni_book_id, si_filename)

        # ------------------------------------------------------------------ #
        #  PHASE 1 — full initial link pass (only if not done yet)            #
        # ------------------------------------------------------------------ #
        if not already_linked:
            logger.info(f"  Phase 1: initial full link pass.")
            _run_full_link_pass(
                ni_book_id, si_filename,
                ni_headings, si_headings,
                out_db, _call, _parse,
            )
        else:
            logger.info(f"  [{ni_book_id} ↔ {si_filename}] already linked — skipping Phase 1.")

        # ------------------------------------------------------------------ #
        #  PHASE 2 — re-match headings with too many untranslated sentences   #
        # ------------------------------------------------------------------ #
        logger.info(f"  Phase 2: auditing for untranslated sections (threshold={untranslated_threshold}).")
        flagged = find_headings_needing_rematch(
            out_db, ni_book_id, si_filename, ni_headings, untranslated_threshold
        )

        if not flagged:
            logger.info(f"  Phase 2: no headings need re-matching.")
        else:
            logger.info(f"  Phase 2: {len(flagged)} heading(s) flagged — sending to Gemini.")
            _run_rematch_pass(
                ni_book_id, si_filename,
                flagged, ni_headings, si_headings,
                out_db, _call, _parse,
            )

        print()

    si_db.close(); ni_db.close(); out_db.close()
    logger.info("Link building complete.")


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------
CHUNK_SIZE = 150
check  = False

def _run_full_link_pass(ni_book_id, si_filename,
                        ni_headings, si_headings,
                        out_db, _call, _parse):
    global check
    if not check and si_filename != 'atta-an-2':
        return
    else:
        check = True
        
    ni_chunks = [ni_headings[i:i + CHUNK_SIZE] for i in range(0, len(ni_headings), CHUNK_SIZE)]
    n_chunks  = len(ni_chunks)
    if n_chunks > 1:
        logger.info(f"    Splitting {len(ni_headings)} nissaya headings into {n_chunks} chunks.")

    matches: list[dict] = []
    seen_ni_ids: set[int] = set()
    combined_log = ""

    for chunk_idx, ni_chunk in enumerate(ni_chunks):
        prompt = build_link_prompt(ni_book_id, si_filename, ni_chunk, si_headings)
        if chunk_idx == 0:
            open("input.txt", "wt").write(prompt)

        raw = _call(prompt)
        combined_log += f"\n===== CHUNK {chunk_idx + 1}/{n_chunks} =====\n\n{raw}"

        chunk_matches = _parse(raw)
        if not chunk_matches:
            logger.warning(f"    Chunk {chunk_idx + 1}: no matches parsed. Raw[:400]:\n{raw[:400]}")
            continue

        for m in chunk_matches:
            ni_id = m.get("nissaya_para_id")
            if ni_id not in seen_ni_ids:
                seen_ni_ids.add(ni_id)
                matches.append(m)

        logger.info(f"    Chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_matches)} matches.")

    open("output.txt", "wt").write(combined_log)

    if not matches:
        logger.warning(f"    No matches parsed across all chunks.")
        return

    save_links(out_db, ni_book_id, si_filename, matches, ni_headings, si_headings, replace=False)
    logger.success(f"    Saved {len(matches)} heading links (full pass).")


def _run_rematch_pass(ni_book_id, si_filename,
                      flagged_headings, all_ni_headings, si_headings,
                      out_db, _call, _parse):
    """Send flagged headings in chunks to Gemini using the re-match prompt,
    then overwrite any existing links with the fresh results."""

    ni_chunks = [flagged_headings[i:i + CHUNK_SIZE]
                 for i in range(0, len(flagged_headings), CHUNK_SIZE)]
    n_chunks  = len(ni_chunks)

    matches: list[dict] = []
    seen_ni_ids: set[int] = set()
    combined_log = ""

    for chunk_idx, ni_chunk in enumerate(ni_chunks):
        prompt = build_rematch_prompt(ni_book_id, si_filename, ni_chunk, si_headings)
        raw    = _call(prompt)
        combined_log += f"\n===== REMATCH CHUNK {chunk_idx + 1}/{n_chunks} =====\n\n{raw}"

        chunk_matches = _parse(raw)
        if not chunk_matches:
            logger.warning(
                f"    Rematch chunk {chunk_idx + 1}: no matches parsed. Raw[:400]:\n{raw[:400]}"
            )
            continue

        for m in chunk_matches:
            ni_id = m.get("nissaya_para_id")
            if ni_id not in seen_ni_ids:
                seen_ni_ids.add(ni_id)
                matches.append(m)

        logger.info(f"    Rematch chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_matches)} matches.")

    open("output_rematch.txt", "wt").write(combined_log)

    if not matches:
        logger.warning(f"    Rematch: no matches parsed.")
        return

    # Pass all_ni_headings so ni_map can resolve titles for any para_id
    save_links(out_db, ni_book_id, si_filename, matches, all_ni_headings, si_headings, replace=True)
    logger.success(f"    Rematch: saved/replaced {len(matches)} heading link(s).")