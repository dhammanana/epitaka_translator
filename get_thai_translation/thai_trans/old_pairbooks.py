"""
pair_books_thai.py — Pair Thai Tipitaka volumes (thaimm.sqlite) with
Nissaya books (nissaya.db, category Mūla / Aṭṭhakathā).

Strategy
--------
Each Thai volume contains exactly one or two texts (Mūla + its Aṭṭhakathā,
or just one of them).  Each Nissaya book may span 1-3 Thai volumes.

For every Thai volume the agent:
  1. Gets the volume's headings (first ~30) as a fingerprint.
  2. Gets a sample of Thai text (first page).
  3. Calls the LLM with:
       - Volume id / name
       - Thai headings sample
       - Full list of candidate Nissaya books (Mūla + Aṭṭhakathā)
  4. The LLM may call get_nissaya_headings(book_id) for up to 6 books
     to compare chapter structure before returning a JSON decision.

Output is saved to the output DB: table thai_nissaya_pairs.

Run:
    python pair_books_thai.py                        # pair all volumes
    python pair_books_thai.py --from-volume 10       # resume
    python pair_books_thai.py --volume-id 42         # single volume (test)
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Reuse keys.py from sibling sinhala project ────────────────────────────────
# Assumes this file lives in the same directory as keys.py, OR that
# ../sin_trans is on the path.  Adjust SIN_TRANS_DIR to taste.
SIN_TRANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sin_trans")
if os.path.isdir(SIN_TRANS_DIR):
    sys.path.insert(0, SIN_TRANS_DIR)

from keys import call_with_rotation, log, Colors  # type: ignore

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-3-flash-preview"
BATCH_SIZE      = 5     # Thai volumes per LLM call
MAX_TOOL_ROUNDS = 8     # per volume
NISSAYA_CATS    = ("Mūla", "Aṭṭhakathā")   # only these categories
HEADINGS_LIMIT  = 40    # nissaya headings returned per tool call
THAI_HEAD_LIMIT = 30    # thai headings used as fingerprint
THAI_TEXT_CHARS = 800   # chars of Thai text sent as sample


# ══════════════════════════════════════════════════════════════════════════════
# Database helpers
# ══════════════════════════════════════════════════════════════════════════════

class ThaiDB:
    """Read-only access to thaimm.sqlite."""

    def __init__(self, path: str):
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    # ── Volume list ──────────────────────────────────────────────────────────
    def get_all_volumes(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT book_id AS volume_id, book_name AS volume_name FROM books ORDER BY book_id"
        )
        return [dict(r) for r in cur.fetchall()]

    # ── Headings for one volume ──────────────────────────────────────────────
    def get_headings(self, volume_id: int, limit: int = THAI_HEAD_LIMIT) -> list[dict]:
        cur = self.conn.execute(
            """
            SELECT id, page, title
            FROM   headings
            WHERE  volume_id = ?
            ORDER  BY page, id
            LIMIT  ?
            """,
            (str(volume_id), limit),
        )
        return [dict(r) for r in cur.fetchall()]

    # ── First page of text for a volume ─────────────────────────────────────
    def get_first_text(self, volume_id: int, chars: int = THAI_TEXT_CHARS) -> str:
        cur = self.conn.execute(
            """
            SELECT content
            FROM   main
            WHERE  volume = ?
            ORDER  BY CAST(page AS INTEGER)
            LIMIT  3
            """,
            (f"{int(volume_id):02d}",),
        )
        rows = cur.fetchall()
        combined = " ".join((r["content"] or "") for r in rows)
        return combined[:chars]

    def close(self):
        self.conn.close()


class NissayaDB:
    """Read-only access to nissaya.db."""

    def __init__(self, path: str):
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    # ── All Mūla + Aṭṭhakathā books ─────────────────────────────────────────
    def get_candidate_books(self) -> list[dict]:
        placeholders = ", ".join("?" for _ in NISSAYA_CATS)
        cur = self.conn.execute(
            f"""
            SELECT book_id, book_name, category, nikaya, sub_nikaya,
                   mula_ref, attha_ref, chapter_len
            FROM   books
            WHERE  category IN ({placeholders})
            ORDER  BY ref_id
            """,
            NISSAYA_CATS,
        )
        return [dict(r) for r in cur.fetchall()]

    # ── Headings for one book (used by the tool) ─────────────────────────────
    def get_book_headings(self, book_id: str, limit: int = HEADINGS_LIMIT) -> list[dict]:
        cur = self.conn.execute(
            """
            SELECT para_id, level, title, chapter_len
            FROM   headings
            WHERE  book_id = ?
            ORDER  BY para_id
            LIMIT  ?
            """,
            (book_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    # ── Book metadata ────────────────────────────────────────────────────────
    def get_book(self, book_id: str) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT * FROM books WHERE book_id = ?", (book_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def close(self):
        self.conn.close()


class OutputDB:
    """Output database — stores pairing results and progress."""

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS thai_nissaya_pairs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                volume_id       INTEGER NOT NULL,
                volume_name     TEXT,
                nissaya_book_id TEXT NOT NULL,
                confidence      REAL DEFAULT 0.0,
                notes           TEXT,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(volume_id, nissaya_book_id)
            );

            CREATE TABLE IF NOT EXISTS thai_pair_progress (
                volume_id   INTEGER PRIMARY KEY,
                status      TEXT DEFAULT 'pending',
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS thai_pair_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                volume_id  INTEGER,
                level      TEXT,
                message    TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    # ── Pairs ────────────────────────────────────────────────────────────────
    def save_pairs(self, pairs: list[dict]):
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO thai_nissaya_pairs
                (volume_id, volume_name, nissaya_book_id, confidence, notes)
            VALUES
                (:volume_id, :volume_name, :nissaya_book_id, :confidence, :notes)
            """,
            pairs,
        )
        self.conn.commit()

    def get_pairs_for_volume(self, volume_id: int) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM thai_nissaya_pairs WHERE volume_id = ?", (volume_id,)
        )
        return [dict(r) for r in cur.fetchall()]

    def has_pairs_for_volume(self, volume_id: int) -> bool:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM thai_nissaya_pairs WHERE volume_id = ?", (volume_id,)
        )
        return cur.fetchone()[0] > 0

    # ── Progress ─────────────────────────────────────────────────────────────
    def mark_done(self, volume_id: int):
        self.conn.execute(
            """
            INSERT OR REPLACE INTO thai_pair_progress (volume_id, status, updated_at)
            VALUES (?, 'done', CURRENT_TIMESTAMP)
            """,
            (volume_id,),
        )
        self.conn.commit()

    def mark_failed(self, volume_id: int):
        self.conn.execute(
            """
            INSERT OR REPLACE INTO thai_pair_progress (volume_id, status, updated_at)
            VALUES (?, 'failed', CURRENT_TIMESTAMP)
            """,
            (volume_id,),
        )
        self.conn.commit()

    def get_status(self, volume_id: int) -> str:
        cur = self.conn.execute(
            "SELECT status FROM thai_pair_progress WHERE volume_id = ?", (volume_id,)
        )
        row = cur.fetchone()
        return row["status"] if row else "pending"

    # ── Log ──────────────────────────────────────────────────────────────────
    def write_log(self, volume_id: int, level: str, message: str):
        self.conn.execute(
            "INSERT INTO thai_pair_log (volume_id, level, message) VALUES (?, ?, ?)",
            (volume_id, level, message),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Tool (nissaya heading lookup, used by the LLM during pairing)
# ══════════════════════════════════════════════════════════════════════════════

_ni_db_ref: Optional[NissayaDB] = None   # set before tool calls


def _set_tool_db(ni_db: NissayaDB):
    global _ni_db_ref
    _ni_db_ref = ni_db


@tool
def get_nissaya_headings(book_id: str) -> str:
    """
    Return the first headings of a Nissaya book so you can compare its
    chapter/section structure with Thai headings to confirm a match.

    Args:
        book_id: The nissaya book_id (e.g. 'D-i', 'Sv-i', 'A-iii').
    """
    if _ni_db_ref is None:
        return json.dumps({"error": "Database not available"})
    headings = _ni_db_ref.get_book_headings(book_id)
    meta = _ni_db_ref.get_book(book_id)
    if not headings:
        return json.dumps({"found": False, "book_id": book_id})
    return json.dumps({
        "found": True,
        "book_id": book_id,
        "book_name": meta["book_name"] if meta else "",
        "category": meta["category"] if meta else "",
        "chapter_len": meta["chapter_len"] if meta else 0,
        "headings": headings,
    })


PAIRING_TOOLS = [get_nissaya_headings]


# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════

PAIRING_SYSTEM = """\
You are an expert in Pali Buddhist texts, the Thai Tipiṭaka (Mahāmakut edition),
and the Nissaya commentarial tradition.

Your task: given a Thai Tipiṭaka volume, identify which Nissaya book(s)
(category Mūla or Aṭṭhakathā) it corresponds to.

Key facts:
- Each Thai volume contains text from exactly ONE main work, but that work
  may be Mūla *or* Aṭṭhakathā (never both in the same volume).
- A single Nissaya book may span 1, 2, or 3 Thai volumes (parts i, ii, iii).
- Match using: book/nikaya names, heading titles, section counts (chapter_len).
- If you are uncertain, call get_nissaya_headings(book_id) for promising
  candidates to compare section structure.

Respond ONLY with valid JSON (no markdown fences):
{
  "pairs": [
    {
      "nissaya_book_id": "D-i",
      "confidence": 0.97,
      "notes": "DN vol 1; heading 'Sīlakkhandhavagga' matches exactly"
    }
  ],
  "unmatched": false,
  "reasoning": "brief"
}

If you cannot find a match set "unmatched": true and "pairs": [].
"""


def _make_pairing_prompt(volume: dict, thai_headings: list[dict], thai_text: str,
                          candidate_books: list[dict]) -> str:
    head_lines = "\n".join(
        f"  [p{h['page']}] {h['title']}" for h in thai_headings
    ) or "  (no headings found)"

    book_lines = "\n".join(
        f"  {b['book_id']:20s} | {b['category']:15s} | {b['nikaya']:30s} | "
        f"{b['book_name']} (len={b['chapter_len']})"
        for b in candidate_books
    )

    return f"""\
## Thai volume to match

volume_id   : {volume['volume_id']}
volume_name : {volume['volume_name']}

### First {THAI_HEAD_LIMIT} headings (page, title)
{head_lines}

### Sample text (first ~{THAI_TEXT_CHARS} chars)
{thai_text or '(empty)'}

---

## Nissaya candidate books (Mūla + Aṭṭhakathā only)

{book_lines}

---

Use get_nissaya_headings(book_id) on promising candidates if needed,
then return the JSON result.
"""


# ══════════════════════════════════════════════════════════════════════════════
# LLM pairing logic (one volume at a time, with tool-call loop)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_text(response) -> str:
    if hasattr(response, "content"):
        c = response.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in c
            )
    return str(response)


def _parse_json(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def _invoke_tool(name: str, args: dict) -> str:
    tool_map = {t.name: t for t in PAIRING_TOOLS}
    if name not in tool_map:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = tool_map[name].invoke(args)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def pair_volume(
    volume: dict,
    llm,
    thai_db: ThaiDB,
    ni_db: NissayaDB,
    candidate_books: list[dict],
    out_db: OutputDB,
) -> list[dict]:
    """
    Ask the LLM to match one Thai volume to Nissaya books.
    Returns a list of pair dicts ready for OutputDB.save_pairs().
    """
    vid = volume["volume_id"]
    thai_headings = thai_db.get_headings(vid)
    thai_text     = thai_db.get_first_text(vid)

    user_content = _make_pairing_prompt(volume, thai_headings, thai_text, candidate_books)

    messages = [
        SystemMessage(content=PAIRING_SYSTEM),
        HumanMessage(content=user_content),
    ]

    tag = f"vol-{vid}"

    response = call_with_rotation(lambda: llm.invoke(messages), tag=tag)
    if response is None:
        log("ERROR", f"LLM returned None for volume {vid}")
        return []

    # ── Tool-call loop ────────────────────────────────────────────────────────
    rounds = 0
    while response.tool_calls and rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        messages.append(response)
        tool_messages = []

        for tc in response.tool_calls:
            name      = tc["name"]
            args      = tc["args"]
            log("TOOL_CALL",
                f"{Colors.BRIGHT_YELLOW}{name}{Colors.RESET}({json.dumps(args)[:80]})")
            result    = _invoke_tool(name, args)
            log("TOOL_RESULT",
                f"{Colors.GREEN}{name}{Colors.RESET} → {result[:120]}")
            tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        messages.extend(tool_messages)
        response = call_with_rotation(lambda: llm.invoke(messages), tag=tag)
        if response is None:
            log("ERROR", f"LLM returned None in tool loop for volume {vid}")
            return []

    # ── Parse final answer ────────────────────────────────────────────────────
    raw  = _extract_text(response)
    data = _parse_json(raw)

    if data is None:
        log("WARN", f"Could not parse JSON for volume {vid}. Raw: {raw[:300]}")
        out_db.write_log(vid, "WARN", f"Unparseable response: {raw[:300]}")
        return []

    if data.get("unmatched"):
        log("SKIP", f"Volume {vid} — no match found (LLM said unmatched)")
        out_db.write_log(vid, "INFO", "LLM: unmatched")
        return []

    pairs_raw = data.get("pairs", [])
    pairs_out = []
    for p in pairs_raw:
        book_id = p.get("nissaya_book_id", "").strip()
        if not book_id:
            continue
        pairs_out.append({
            "volume_id":       vid,
            "volume_name":     volume["volume_name"],
            "nissaya_book_id": book_id,
            "confidence":      float(p.get("confidence", 0.0)),
            "notes":           p.get("notes", ""),
        })

    reasoning = data.get("reasoning", "")
    if reasoning:
        out_db.write_log(vid, "INFO", f"reasoning: {reasoning}")

    log("SAVED",
        f"{Colors.BRIGHT_GREEN}Volume {vid}{Colors.RESET} → "
        + ", ".join(f"{p['nissaya_book_id']} ({p['confidence']:.2f})" for p in pairs_out))

    return pairs_out


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_pairing(
    thai_path:    str,
    nissaya_path: str,
    output_path:  str,
    model:        str = GEMINI_MODEL,
    volume_id:    Optional[int] = None,
    from_volume:  Optional[int] = None,
):
    thai_db = ThaiDB(thai_path)
    ni_db   = NissayaDB(nissaya_path)
    out_db  = OutputDB(output_path)

    _set_tool_db(ni_db)

    # Build LLM with tools bound
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    llm = llm.bind_tools(PAIRING_TOOLS)

    # ── Candidate Nissaya books (fixed for all volumes) ──────────────────────
    candidate_books = ni_db.get_candidate_books()
    log("START", f"Nissaya candidates: {len(candidate_books)} books (Mūla + Aṭṭhakathā)")

    # ── Thai volumes ─────────────────────────────────────────────────────────
    all_volumes = thai_db.get_all_volumes()
    log("START", f"Thai volumes: {len(all_volumes)}")

    # Filter
    if volume_id is not None:
        all_volumes = [v for v in all_volumes if v["volume_id"] == volume_id]
        if not all_volumes:
            log("ERROR", f"Volume id={volume_id} not found")
            return
    if from_volume is not None:
        all_volumes = [v for v in all_volumes if v["volume_id"] >= from_volume]

    skipped = done = failed = 0

    for vol in all_volumes:
        vid = vol["volume_id"]

        # Skip already done
        if out_db.get_status(vid) == "done":
            logger.info(f"[{vid}] already done — skipping")
            skipped += 1
            continue

        log("TURN", f"[{vid}] {vol['volume_name']}")

        try:
            pairs = pair_volume(vol, llm, thai_db, ni_db, candidate_books, out_db)
            if pairs:
                out_db.save_pairs(pairs)
            out_db.mark_done(vid)
            done += 1
        except Exception as e:
            logger.exception(f"[{vid}] Unhandled error: {e}")
            out_db.write_log(vid, "ERROR", str(e))
            out_db.mark_failed(vid)
            failed += 1

        time.sleep(1)   # be polite to the API

    log("DONE",
        f"Volumes processed={done}  skipped={skipped}  failed={failed}")

    thai_db.close()
    ni_db.close()
    out_db.close()


# ══════════════════════════════════════════════════════════════════════════════
# Status helper
# ══════════════════════════════════════════════════════════════════════════════

def print_status(output_path: str):
    out_db = OutputDB(output_path)
    cur = out_db.conn.execute("""
        SELECT p.volume_id, p.status,
               GROUP_CONCAT(t.nissaya_book_id || ' (' || ROUND(t.confidence,2) || ')',
                            ', ') AS books
        FROM   thai_pair_progress p
        LEFT   JOIN thai_nissaya_pairs t ON t.volume_id = p.volume_id
        GROUP  BY p.volume_id
        ORDER  BY p.volume_id
    """)
    print(f"\n{'Vol':>4}  {'Status':<12}  Books")
    print("-" * 80)
    for r in cur.fetchall():
        print(f"{r['volume_id']:>4}  {r['status']:<12}  {r['books'] or '—'}")
    out_db.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pair Thai Tipiṭaka volumes with Nissaya books"
    )
    parser.add_argument("--thai",        default="../data/thaimm.sqlite",
                        help="Path to thaimm.sqlite")
    parser.add_argument("--nissaya",     default="../data/nissaya.db",
                        help="Path to nissaya.db")
    parser.add_argument("--output",      default="../data/thai_pairs.db",
                        help="Path to output SQLite DB")
    parser.add_argument("--model",       default=GEMINI_MODEL)
    parser.add_argument("--volume-id",   type=int, default=None,
                        help="Process only this Thai volume id")
    parser.add_argument("--from-volume", type=int, default=None,
                        help="Start from this volume id (skip earlier ones)")
    parser.add_argument("--status",      action="store_true",
                        help="Print pairing status and exit")
    args = parser.parse_args()

    if args.status:
        print_status(args.output)
        sys.exit(0)

    run_pairing(
        thai_path=args.thai,
        nissaya_path=args.nissaya,
        output_path=args.output,
        model=args.model,
        volume_id=args.volume_id,
        from_volume=args.from_volume,
    )