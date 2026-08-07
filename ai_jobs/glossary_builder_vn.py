"""
glossary_builder_paragraphs.py — Build / refresh the glossary for a
             single-table DB shaped like:

    CREATE TABLE paragraphs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id     TEXT NOT NULL,
        page        INTEGER NOT NULL,
        para_num    INTEGER NOT NULL,
        pali        TEXT,
        vietnamese  TEXT,
        UNIQUE(book_id, page, para_num)
    )

Glossary-only variant: no quality-check / remarks pass, just glossary
extraction. This is a standalone variant of glossary_builder.py for DBs
that keep the Pāli and the translation in ONE table/row (no separate
epitaka_<lang>.db / sentences split, no line_id — the unit is a
paragraph, addressed by (book_id, page, para_num)).

    # Build glossary for explicit books:
    python glossary_builder_paragraphs.py --db epitaka_vi.db --books Sp-i,Sp-ii

    # Auto-discover all books that have a Vietnamese translation:
    python glossary_builder_paragraphs.py --db epitaka_vi.db

    # Resume from last saved position (default behaviour — state is always saved):
    python glossary_builder_paragraphs.py --db epitaka_vi.db

    # Force restart from the beginning:
    python glossary_builder_paragraphs.py --db epitaka_vi.db --restart

What it does
------------
For each book it:

  1. Reads every (pali, vietnamese) paragraph pair from the `paragraphs`
     table that has a non-empty vietnamese translation.

  2. Sends chunks to Gemini with:
       • the current ESTABLISHED GLOSSARY (so the AI can skip duplicates)
       • the pali+translation pairs for this chunk

  3. Gemini returns "glossary" — new terms whose meaning is NOT yet
     covered in the established glossary (context and note in Vietnamese).

  4. New glossary terms are saved to a `glossary` table in glossary.db
     (next to the source DB, or --glossary-db).

Progress tracking
-----------------
A JSON state file is saved next to --db as
  glossary_build_<dbname>_state.json
It stores the last completed (book_id, page, para_num) triple so the
script can resume exactly where it left off after interruption.

File layout
-----------
This script is self-contained aside from the AI-calling infrastructure
(Gemini calls, key rotation, retries, response-JSON parsing), which is
reused from ai_client.py — change that file, not this one, when the AI
logic needs to change. Everything schema-specific (reading the
`paragraphs` table, the glossary table, resume state) lives here since
it doesn't match the multi-table epitaka_<lang>.db layout that
common_utils.py / glossary_builder.py assume.
"""

import argparse
import json
import os
import re
import sys
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

import common.ai_client as ai

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

LANG_NAME    = "Vietnamese"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
DEFAULT_LOG_DIR = "/tmp/build_glossary_logs"
DEFAULT_CHUNK_SIZE = 60   # paragraph pairs per Gemini call


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def glossary_db_path(db: str) -> str:
    return str(Path(db).with_name("glossary_vn.db"))


def state_file_path(db: str) -> str:
    stem = Path(db).stem
    return str(Path(db).with_name(f"glossary_build_{stem}_state.json"))


# ══════════════════════════════════════════════════════════════════
# Schema setup
# ══════════════════════════════════════════════════════════════════

def ensure_glossary_db(glossary_db: str):
    with connect(glossary_db) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pali        TEXT NOT NULL,
                translation TEXT NOT NULL,
                domain      TEXT,
                sub_domain  TEXT,
                context     TEXT,
                note        TEXT,
                source_id   TEXT,
                page_start  INTEGER,
                page_end    INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_glossary_pali ON glossary(pali)")
        conn.commit()


# ══════════════════════════════════════════════════════════════════
# Progress state (resume support)
# ══════════════════════════════════════════════════════════════════

class BuildState:
    """
    Persists progress to a JSON file so the script can resume after
    interruption.

    Schema:
      {
        "completed_books": ["Sp-i", "Sp-ii"],
        "current_book": "Sp-iii",
        "current_page_end": 12,
        "current_para_num_end": 4
      }
    """

    def __init__(self, path: str):
        self._path = path
        self._data: dict = {}
        self._load()

    def _load(self):
        if Path(self._path).exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                print(f"[state] Loaded from {self._path}")
            except Exception as exc:
                print(f"[state] Could not load state file: {exc}. Starting fresh.")
                self._data = {}

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[state] Could not save state: {exc}")

    def reset(self):
        self._data = {
            "completed_books": [],
            "current_book": None,
            "current_page_end": 0,
            "current_para_num_end": 0,
        }
        self._save()
        print("[state] Reset.")

    @property
    def completed_books(self) -> list[str]:
        return self._data.get("completed_books", [])

    def is_book_done(self, book_id: str) -> bool:
        return book_id in self.completed_books

    def resume_cursor_for(self, book_id: str) -> tuple[int, int]:
        """Return (page, para_num) to resume from (exclusive — start AFTER this)."""
        if self._data.get("current_book") == book_id:
            return (
                self._data.get("current_page_end", 0),
                self._data.get("current_para_num_end", 0),
            )
        return (0, 0)

    def mark_chunk_done(self, book_id: str, page_end: int, para_num_end: int):
        self._data["current_book"]          = book_id
        self._data["current_page_end"]      = page_end
        self._data["current_para_num_end"]  = para_num_end
        self._save()

    def mark_book_done(self, book_id: str):
        done = self._data.get("completed_books", [])
        if book_id not in done:
            done.append(book_id)
        self._data["completed_books"]       = done
        self._data["current_book"]          = None
        self._data["current_page_end"]      = 0
        self._data["current_para_num_end"]  = 0
        self._save()
        print(f"[state] Book '{book_id}' marked complete.")


# ══════════════════════════════════════════════════════════════════
# Book / pair discovery
# ══════════════════════════════════════════════════════════════════

def discover_books(db: str) -> list[str]:
    """Return distinct book_ids that have at least one Vietnamese translation."""
    with connect(db) as conn:
        rows = conn.execute(
            """SELECT DISTINCT book_id FROM paragraphs
               WHERE vietnamese IS NOT NULL AND vietnamese != ''
               ORDER BY book_id"""
        ).fetchall()
    books = [r["book_id"] for r in rows]
    print(f"[discover] Found {len(books)} book(s) with translations in {db}.")
    return books


def fetch_translated_pairs(
    db:            str,
    book_id:       str,
    page_start:    int,
    para_num_start: int,
    chunk_size:    int = DEFAULT_CHUNK_SIZE,
) -> list[list[dict]]:
    """
    Fetch (pali, vietnamese) paragraph rows for a book, ordered by
    (page, para_num). Only rows strictly after the (page_start,
    para_num_start) cursor are returned, so pass the last completed
    cursor (0, 0 = from the very beginning).
    Returns a list of chunks (each chunk is a list of paragraph dicts).
    """
    with connect(db) as conn:
        rows = conn.execute(
            """SELECT book_id, page, para_num, pali, vietnamese
               FROM paragraphs
               WHERE book_id = ?
                 AND (page > ? OR (page = ? AND para_num > ?))
                 AND vietnamese IS NOT NULL AND vietnamese != ''
                 AND pali IS NOT NULL AND pali != ''
               ORDER BY page, para_num""",
            (book_id, page_start, page_start, para_num_start),
        ).fetchall()

    pairs = [
        {
            "page":       r["page"],
            "para_num":   r["para_num"],
            "pali":       r["pali"].strip(),
            "translation": r["vietnamese"],
        }
        for r in rows
    ]

    if not pairs:
        return []
    return [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]


def fetch_established_glossary_block(glossary_db: str, pali_text: str) -> str:
    """
    Build a compact ESTABLISHED GLOSSARY block for the prompt, limited to
    terms whose stems appear in the current pali_text. Falls back to the
    top-500 most-recent entries if stem matching returns nothing useful.
    """
    if not Path(glossary_db).exists():
        return ""

    try:
        tokens = set(re.findall(r"[a-zāīūṃṅñṭḍṇḷṣśḥ]+", pali_text.lower()))

        with connect(glossary_db) as conn:
            rows = conn.execute(
                "SELECT pali, translation, domain, context, note FROM glossary "
                "ORDER BY id DESC LIMIT 2000"
            ).fetchall()

        if not rows:
            return ""

        matched = [r for r in rows if r["pali"] in tokens]
        if not matched:
            matched = rows[:500]  # fallback: most recent 500

        lines = ["══════════════════════════════",
                 "ESTABLISHED GLOSSARY",
                 "══════════════════════════════"]
        for r in matched:
            line = f"  {r['pali']} → {r['translation']}"
            if r["domain"]:
                line += f"  [{r['domain']}]"
            if r["context"]:
                line += f"  ctx: {r['context']}"
            if r["note"]:
                line += f"  note: {r['note']}"
            lines.append(line)
        return "\n".join(lines)
    except Exception as exc:
        print(f"[glossary_block] Could not build: {exc}")
        return ""


# ══════════════════════════════════════════════════════════════════
# Save helper
# ══════════════════════════════════════════════════════════════════

def save_glossary_terms(
    glossary_db:  str,
    new_terms:    list[dict],
    source_id:    str,
    page_start:   int,
    page_end:     int,
) -> int:
    if not new_terms:
        return 0
    rows = []
    for t in new_terms:
        pali = str(t.get("pali") or "").strip()
        translation = str(t.get("translation") or "").strip()
        if not pali or not translation:
            continue
        rows.append((
            pali,
            translation,
            str(t.get("domain") or ""),
            str(t.get("sub_domain") or ""),
            str(t.get("context") or ""),
            str(t.get("note") or ""),
            source_id,
            page_start,
            page_end,
        ))
    if not rows:
        return 0
    with connect(glossary_db) as conn:
        conn.executemany(
            """INSERT INTO glossary
               (pali, translation, domain, sub_domain, context, note,
                source_id, page_start, page_end)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
    return len(rows)


# ══════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════

def _build_system_prompt() -> str:
    lang_name = LANG_NAME
    return f"""You are an expert in Pāli Buddhist terminology and {lang_name} translation.

You will receive Pāli paragraphs paired with their {lang_name} translations from
Buddhist scriptures (canonical texts, commentaries, sub-commentaries).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK — "glossary": Extract NEW glossary terms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract significant Pāli technical terms and their {lang_name} renderings
to help future translators choose the correct, consistent translation.

Return one object per NEW term:
  {{ "pali": "…", "translation": "…", "domain": "…",
    "sub_domain": "…", "context": "…", "note": "…" }}

domain ∈ {{sutta, vinaya, abhidhamma, grammar, story}}

CRITICAL — "translation" field:
  • Must be in {lang_name}.
  • Extract the rendering that appears in the supplied translations.

CRITICAL — "context" and "note" fields:
  • Write BOTH in {lang_name} (not in English or Pāli).
  • "context": describe in {lang_name} when/where this term appears and how
    it is used — enough that a future translator understands the doctrinal
    or textual setting.
  • "note": write in {lang_name} any guidance that helps a future translator
    DECIDE the correct {lang_name} rendering — e.g. how this term differs from
    a similar one, why this particular {lang_name} word was chosen, what
    nuance would be lost with an alternative, or common pitfalls to avoid.
    This is the most important field: make it genuinely useful for decision-making.

CRITICAL — DO NOT add duplicates:
  • Check the ESTABLISHED GLOSSARY block carefully before adding any term.
  • If the pali stem (or a very close variant) already appears in the
    established glossary AND the {lang_name} meaning is substantially the same,
    DO NOT add it again — even if the wording differs slightly.
  • Only add a term if its meaning or usage is NOT already covered.

What TO include:
  • Technical terms easily rendered inconsistently or confused with similar
    terms (e.g. "samādhi" vs "samāpatti", "sīla" vs "vinaya").
  • Named doctrinal concepts, proper nouns, set Vinaya/Abhidhamma terms.
  • Terms whose {lang_name} rendering is non-obvious or debatable.

What NOT to include:
  • Grammatical particles / conjunctions:
    ca, va, vā, pi, api, tu, pana, hi, eva, kho, ti, iti, atha, yeva,
    hoti, hotu, honti, ahosi, atthi, natthi, kacci, kiṃ, na, mā, evaṃ,
    seyyathā, tattha, tatra, tato, yathā, tathā, idaṃ, ayaṃ, so, sā, yo, yā.
  • Plain verbs of being/doing with no doctrinal significance.
  • Any term already covered in the ESTABLISHED GLOSSARY.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — critical
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown fences, no prose outside the JSON.
{{ "glossary": [...] }}
"""


def _build_user_prompt(
    book_id:          str,
    page_start:       int,
    page_end:         int,
    glossary_block:   str,
    pairs:            list[dict],
) -> str:
    lang_name = LANG_NAME
    pairs_text = "\n".join(
        f"  page={s['page']} para_num={s['para_num']}\n"
        f"  Pāli: {s['pali']}\n"
        f"  {lang_name}: {s['translation']}"
        for s in pairs
    )

    return (
        f"Book: {book_id}  —  pages {page_start}–{page_end}\n\n"
        f"{glossary_block}\n\n"
        f"══════════════════════════════\n"
        f"PĀLI + {lang_name.upper()} TRANSLATION PAIRS\n"
        f"══════════════════════════════\n"
        f"{pairs_text}\n\n"
        f"Extract new glossary terms as instructed.\n"
        f"Do NOT add any glossary term already present in the ESTABLISHED GLOSSARY above.\n"
        f"Return only JSON: {{ \"glossary\": [...] }}"
    )


# ══════════════════════════════════════════════════════════════════
# Per-book processing
# ══════════════════════════════════════════════════════════════════

def process_book(
    book_id:     str,
    args,
    rotator:     ai.KeyRotator,
    db:          str,
    glossary_db: str,
    state:       BuildState,
) -> int:
    """
    Process one book. Returns glossary_inserted.
    """
    system_prompt = _build_system_prompt()

    print("=" * 60)
    print(f"BOOK: {book_id}")
    print("=" * 60)

    # Determine where to resume
    resume_page, resume_para_num = state.resume_cursor_for(book_id)
    if resume_page > 0 or resume_para_num > 0:
        print(f"  [resume] Continuing from after page={resume_page}, para_num={resume_para_num}")

    chunks = fetch_translated_pairs(
        db             = db,
        book_id        = book_id,
        page_start     = resume_page,
        para_num_start = resume_para_num,
        chunk_size     = args.chunk_size,
    )

    if not chunks:
        print("  No translated pairs found (or all already processed). Skipping.")
        state.mark_book_done(book_id)
        return 0

    print(f"  {sum(len(c) for c in chunks)} pairs in {len(chunks)} chunk(s).")
    total_glossary = 0

    for c_idx, chunk in enumerate(chunks, 1):
        page_start = chunk[0]["page"]
        page_end   = chunk[-1]["page"]
        print(f"  Chunk {c_idx}/{len(chunks)}: {len(chunk)} pairs "
              f"(pages {page_start}-{page_end})")

        pali_text      = "\n".join(s["pali"] for s in chunk)
        glossary_block = fetch_established_glossary_block(glossary_db, pali_text)

        prompt = _build_user_prompt(
            book_id        = book_id,
            page_start     = page_start,
            page_end       = page_end,
            glossary_block = glossary_block,
            pairs          = chunk,
        )

        if args.dry_run:
            print(prompt[:500], "...")
            state.mark_chunk_done(book_id, chunk[-1]["page"], chunk[-1]["para_num"])
            continue

        raw = ai.call_ai_with_logging(
            rotator       = rotator,
            prompt        = prompt,
            book_id       = book_id,
            chunk_id      = f"glos_p{page_start}-{page_end}_c{c_idx}",
            log_dir       = args.log_dir,
            model         = args.model,
            system_prompt = system_prompt,
        )
        if raw is None:
            print(f"  Chunk {c_idx} returned no response. Skipping.")
            continue

        try:
            result = ai.parse_ai_json_response(raw, ("glossary",))
        except Exception as exc:
            print(f"  parse_response failed for chunk {c_idx}: {exc}. Skipping.")
            continue

        new_terms = result.get("glossary", [])
        print(f"  Parsed: {len(new_terms)} glossary term(s)")

        if new_terms:
            inserted = save_glossary_terms(
                glossary_db = glossary_db,
                new_terms   = new_terms,
                source_id   = book_id,
                page_start  = page_start,
                page_end    = page_end,
            )
            total_glossary += inserted
            print(f"  Saved {inserted} new glossary term(s). Running total: {total_glossary}.")

        # Persist progress
        state.mark_chunk_done(book_id, chunk[-1]["page"], chunk[-1]["para_num"])

    state.mark_book_done(book_id)
    return total_glossary


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db", required=True,
        help="Path to the sqlite DB containing the `paragraphs` table.",
    )
    parser.add_argument(
        "--books", default="",
        help=(
            'Comma-separated book_ids, e.g. "Sp-i,Sp-ii". '
            'Omit to auto-discover all books with Vietnamese translations in --db.'
        ),
    )
    parser.add_argument("--chunk-size",  type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Paragraph pairs per Gemini call (default {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--restart",     action="store_true",
                        help="Ignore saved state and restart from the beginning.")
    parser.add_argument("--glossary-db", default="",
                        help="Override path to glossary.db (default: glossary.db next to --db).")
    parser.add_argument("--model",       default=GEMINI_MODEL)
    parser.add_argument("--api-keys",    default="")
    parser.add_argument("--log-dir",     default=DEFAULT_LOG_DIR)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    db          = args.db
    glossary_db = args.glossary_db or glossary_db_path(db)
    state_path  = state_file_path(db)

    print(f"[db] Source DB    : {db}")
    print(f"[db] Glossary DB  : {glossary_db}")
    print(f"[db] State file   : {state_path}")

    if not Path(db).exists():
        print(f"[ERROR] DB not found: {db}")
        return 1

    # Ensure DB structures
    if not args.dry_run:
        ensure_glossary_db(glossary_db)

    # Load (or reset) progress state
    state = BuildState(state_path)
    if args.restart:
        state.reset()
        print("[state] Restarted from scratch.")

    # Resolve book list
    if args.books.strip():
        book_list = [b.strip() for b in args.books.split(",") if b.strip()]
        print(f"[books] Explicit list: {len(book_list)} book(s).")
    else:
        book_list = discover_books(db)
        if not book_list:
            print("No books found with translations. Nothing to do.")
            return 1

    # Filter already-completed books (unless --restart already cleared state)
    pending = [b for b in book_list if not state.is_book_done(b)]
    skipped = len(book_list) - len(pending)
    if skipped:
        print(f"[state] Skipping {skipped} already-completed book(s).")
    book_list = pending

    if not book_list:
        print("All books already processed. Use --restart to reprocess.")
        return 0

    rotator = ai.make_rotator([k.strip() for k in args.api_keys.split(",") if k.strip()])

    grand_glossary = 0

    for book_idx, book_id in enumerate(book_list, 1):
        print(f"\n{'#' * 60}")
        print(f"# BOOK {book_idx}/{len(book_list)}: {book_id}")
        print(f"{'#' * 60}")

        try:
            g = process_book(
                book_id     = book_id,
                args        = args,
                rotator     = rotator,
                db          = db,
                glossary_db = glossary_db,
                state       = state,
            )
        except Exception as exc:
            print(f"[ERROR] Book {book_id} failed: {exc}. Continuing.")
            continue

        grand_glossary += g
        print(f"Book {book_id} done — glossary: {g}.")

    print("\n" + "=" * 60)
    print(f"ALL DONE.  Books processed: {len(book_list)}")
    print(f"  Total glossary terms saved : {grand_glossary}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())