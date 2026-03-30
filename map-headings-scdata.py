"""
match_headings_to_sc.py  —  Map nissaya.db headings → sc_id from sc-data.db
============================================================================

For every Mūla book:
  1. Pull headings (level < 10) from nissaya.db
  2. Pull all (sc_id, sutta_name) rows for that book_id from en_translation
  3. Ask Gemini to return a JSON mapping  { "para_id": "sc_id" | null }
  4. ALTER/UPDATE headings.sc_id  (idempotent — safe to re-run)

Resumable: already-mapped headings are skipped unless --reset is given.
Progress is saved per-book in:
    progress/heading_match_{book_id}.jsonl

Usage
-----
    python match_headings_to_sc.py                        # all Mūla books
    python match_headings_to_sc.py --book-id D-i          # single book
    python match_headings_to_sc.py --reset                # clear cache & redo
    python match_headings_to_sc.py --dry-run              # print, don't write
"""

import argparse
import datetime
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from gemini_client import GeminiClient
from config_tmp import PROGRESS_DIR, NISSAYA_DB_PATH, SC_DATA_DB_PATH

# ─── If your config doesn't export these two DB paths, define them here: ───
# NISSAYA_DB_PATH  = "data/nissaya.db"
# SC_DATA_DB_PATH  = "data/sc-data.db"
# ───────────────────────────────────────────────────────────────────────────


# =============================================================================
#  LOGGING / PROGRESS   (identical helpers to indexer.py)
# =============================================================================

def _log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)


class Progress:
    def __init__(self, phase: str, total: int, extra: str = ""):
        self.phase = phase
        self.total = total
        self.done  = 0
        self.start = time.time()
        self.extra = extra
        self._last = 0.0
        self._print()

    def update(self, done: int, extra: str = ""):
        self.done = done
        if extra:
            self.extra = extra
        now = time.time()
        if now - self._last >= 0.5 or done >= self.total:
            self._print()
            self._last = now

    def _print(self):
        elapsed = max(time.time() - self.start, 0.001)
        rate    = self.done / elapsed
        pct     = self.done / max(self.total, 1) * 100
        bar_len = 25
        filled  = int(bar_len * pct / 100)
        bar     = "█" * filled + "░" * (bar_len - filled)
        eta     = (str(datetime.timedelta(seconds=int((self.total - self.done) / rate)))
                   if rate > 0 and self.done < self.total else "done")
        rate_s  = f"{rate:.1f}/s" if rate >= 1 else f"{rate*60:.1f}/min"
        line    = (f"\r[{self.phase}] |{bar}| "
                   f"{self.done:>5,}/{self.total:,} ({pct:5.1f}%) "
                   f"{rate_s}  ETA {eta}")
        if self.extra:
            line += f"  {self.extra}"
        sys.stdout.write(line.ljust(120))
        sys.stdout.flush()

    def finish(self, msg: str = ""):
        self.update(self.total)
        elapsed = time.time() - self.start
        print(f"\n  ✓ {self.phase} done in {elapsed:.1f}s"
              + (f" — {msg}" if msg else ""))


# =============================================================================
#  JSONL CACHE HELPERS   (identical to indexer.py)
# =============================================================================

def _cache_path(book_id: str) -> Path:
    Path(PROGRESS_DIR).mkdir(parents=True, exist_ok=True)
    safe = book_id.replace("/", "_").replace(".", "_")
    return Path(PROGRESS_DIR) / f"heading_match_{safe}.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _append_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =============================================================================
#  JSON REPAIR   (same approach as indexer.py's _repair_json)
# =============================================================================

def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    for prefix in ("```json", "```"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def _extract_json_object(raw: str) -> dict | None:
    """
    Parse the model's response as a JSON *object* (not array).
    Handles fenced output and finds the outermost { … } block.
    Falls back to a character-walk salvage on parse failure.
    """
    text = _strip_fences(raw)

    # Fast path
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start == -1 or brace_end == -1:
        return None
    candidate = text[brace_start : brace_end + 1]
    try:
        result = json.loads(candidate)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    # Slow-path: salvage complete key-value pairs with a character walk
    # (mirrors indexer.py's object-extraction loop)
    depth, in_str, escape = 0, False, False
    obj_start = None
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(text[obj_start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                obj_start = None
    return None


# =============================================================================
#  DB HELPERS
# =============================================================================

def ensure_sc_id_column(conn: sqlite3.Connection) -> None:
    """Add sc_id column to headings if it doesn't exist yet (idempotent)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(headings)")}
    if "sc_id" not in cols:
        conn.execute("ALTER TABLE headings ADD COLUMN sc_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_headings_sc_id ON headings(sc_id)"
        )
        conn.commit()
        _log("Added sc_id column + index to headings.")


def get_mula_books(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT book_id, book_name FROM books "
        "WHERE category = 'Mūla' ORDER BY book_id"
    ).fetchall()


def get_top_headings(conn: sqlite3.Connection, book_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT para_id, level, title, parent
        FROM   headings
        WHERE  book_id = ?
        ORDER  BY para_id
        """,
        (book_id,),
    ).fetchall()

    by_para_id = {
        r[0]: {"para_id": r[0], "level": r[1], "title": r[2], "parent": r[3]}
        for r in rows
    }

    def build_path(heading: dict) -> str:
        parts = []
        current = heading
        while current is not None:
            parts.append(current["title"])
            parent_id = current["parent"]
            current = by_para_id.get(parent_id) if parent_id != -1 else None
        parts.reverse()
        return " > ".join(parts)

    return [
        {"para_id": h["para_id"], "level": h["level"], "title": build_path(h)}
        for h in by_para_id.values()
        if h["level"] < 10
    ]


def get_sc_candidates(sc_conn: sqlite3.Connection, book_id: str) -> list[dict]:
    """All (sc_id, sutta_name) rows from en_translation for this book."""
    rows = sc_conn.execute(
        "SELECT sc_id, sutta_name FROM en_translation "
        "WHERE book_id = ? ORDER BY sc_id",
        (book_id,),
    ).fetchall()
    return [{"sc_id": r[0], "sutta_name": r[1]} for r in rows]


def save_mapping(
    conn: sqlite3.Connection,
    book_id: str,
    mapping: dict,  # { "para_id_str": sc_id | None }
    dry_run: bool,
) -> int:
    updated = 0

    for para_id_str, sc_id in mapping.items():
        if sc_id is None:
            continue

        try:
            para_id = int(para_id_str)
        except ValueError:
            continue

        if dry_run:
            updated += 1
            continue

        # Try to find the correct para_id by incrementing if the original doesn't exist
        current_para_id = para_id
        max_attempts = 1  # Safety limit to prevent infinite loops

        for attempt in range(max_attempts):
            # Check if this para_id exists for the given book_id
            check_cursor = conn.execute(
                "SELECT 1 FROM headings "
                "WHERE book_id = ? AND para_id = ?",
                (book_id, current_para_id),
            )
            exists = check_cursor.fetchone() is not None

            if exists:
                # Update the existing row
                conn.execute(
                    "UPDATE headings SET sc_id = ? "
                    "WHERE book_id = ? AND para_id = ?",
                    (sc_id, book_id, current_para_id),
                )
                updated += 1
                break
            else:
                # Try next para_id
                print("=== check next row")
                current_para_id += 1
        else:
            # This runs if we exhausted max_attempts without finding a match
            print(f"==== Warning: Could not find para_id near {para_id} for book_id {book_id} "
                  f"after {max_attempts} attempts. Skipping.")

    if not dry_run:
        conn.commit()

    return updated


# =============================================================================
#  PROMPT
# =============================================================================

SYSTEM_PROMPT = """\
You are an expert in Pali Buddhist texts and SuttaCentral identifiers.
Given a list of chapter headings from a Burmese nissaya commentary and a list
of SuttaCentral suttas for the same book, map each heading to the best sc_id.

Rules:
- A heading maps to exactly ONE sc_id, or null if no clear match exists.
- Multiple headings inside the same sutta all share the same sc_id.
- Generic structural headings (Nidāna, Uddesa, page refs) → null.
- sutta_name contains the full TOC path, e.g.
  "Dīgha Nikāya > Sīlakkhandhavagga > Brahmajāla Sutta" — use all levels.
- Respond ONLY with a valid JSON object.
  Format:  { "PARA_ID": "sc_id_value_or_null", ... }
  No markdown fences. No explanation. No extra keys."""


def _build_prompt(
    book_id: str,
    book_name: str,
    headings: list[dict],
    candidates: list[dict],
) -> str:
    h_lines = "\n".join(
        f'  {{"para_id":{h["para_id"]},'
        f'"lvl":{h["level"]},'
        f'"title":{json.dumps(h["title"], ensure_ascii=False)},'
        f'}}'
        for h in headings
    )
    c_lines = "\n".join(
        f'  {{"sc_id":{json.dumps(c["sc_id"])},'
        f'"name":{json.dumps(c["sutta_name"] or "", ensure_ascii=False)}}}'
        for c in candidates
    )
    return (
        f"Book: {book_id} — {book_name}\n\n"
        f"=== HEADINGS (level < 10) ===\n[\n{h_lines}\n]\n\n"
        f"=== SUTTACENTRAL CANDIDATES ===\n[\n{c_lines}\n]\n\n"
        "Return the JSON object mapping each para_id (as a string key) "
        "to its sc_id or null."
    )


# =============================================================================
#  PER-BOOK MATCHING   (resumable via JSONL cache)
# =============================================================================

# Gemini 1.5 Flash: 1 M-token context, so we can send a whole book at once.
# We still chunk as a safety net for books with hundreds of headings.
HEADING_CHUNK_SIZE = 150   # headings per call


def _validate(mapping: dict, valid_sc_ids: set) -> dict:
    """Sanitise model output: numeric keys only, values in valid set or None."""
    clean = {}
    for k, v in mapping.items():
        k = str(k).strip()
        if not k.lstrip("-").isdigit():
            continue
        if v is not None and v not in valid_sc_ids:
            _log(f"  [WARN] unknown sc_id {v!r} → null")
            v = None
        clean[k] = v
    return clean


def match_book(
    book_id    : str,
    book_name  : str,
    nissaya_conn: sqlite3.Connection,
    sc_conn    : sqlite3.Connection,
    gemini     : GeminiClient,
    dry_run    : bool,
    reset      : bool,
) -> int:
    """
    Match all headings for one book.  Returns number of sc_ids assigned.
    """
    headings   = get_top_headings(nissaya_conn, book_id)
    candidates = get_sc_candidates(sc_conn, book_id)

    if not headings:
        _log(f"  [{book_id}] No headings (level < 10) — skip")
        return 0
    if not candidates:
        _log(f"  [{book_id}] No sc_id candidates in en_translation — skip")
        return 0

    valid_sc_ids = {c["sc_id"] for c in candidates}

    # ── Resume: load already-processed para_ids from JSONL cache ──────────
    cache   = _cache_path(book_id)
    if reset and cache.exists():
        cache.unlink()
        _log(f"  [{book_id}] Cache cleared.")

    cached_map: dict[str, str | None] = {}
    for rec in _load_jsonl(cache):
        cached_map.update(rec)          # each JSONL line is a partial mapping

    already_done = set(cached_map.keys())
    todo_headings = [h for h in headings
                     if str(h["para_id"]) not in already_done]

    _log(
        f"  [{book_id}]  {len(headings)} headings | "
        f"{len(candidates)} sc candidates | "
        f"{len(already_done)} cached | {len(todo_headings)} remaining"
    )

    if not todo_headings:
        _log(f"  [{book_id}] Fully cached — writing to DB.")
        return save_mapping(nissaya_conn, book_id, cached_map, dry_run)

    # ── Process in chunks ──────────────────────────────────────────────────
    prog   = Progress(f"Match {book_id}", len(todo_headings))
    done   = 0
    errors = 0

    for chunk_start in range(0, len(todo_headings), HEADING_CHUNK_SIZE):
        chunk = todo_headings[chunk_start : chunk_start + HEADING_CHUNK_SIZE]
        prompt = _build_prompt(book_id, book_name, chunk, candidates)

        # ── Call Gemini (up to 3 retries, mirrors indexer.py pattern) ─────
        raw = "{}"
        for attempt in range(1, 4):
            try:
                raw = gemini.generate(prompt, max_tokens=10000,  thinking=512)
                break
            except Exception as exc:
                errors += 1
                _log(f"  [{book_id}] attempt {attempt} failed: {exc}")
                if attempt < 3:
                    time.sleep(5 * attempt)
                else:
                    _log(f"  [{book_id}] giving up on chunk {chunk_start}; saving nulls.")

        # ── Parse & validate ───────────────────────────────────────────────
        partial = _extract_json_object(raw)
        if partial is None:
            _log(f"  [{book_id}] JSON parse failed — chunk {chunk_start}:"
                 f"\n    {raw[:300]}")
            # Save nulls so this chunk is not retried on resume
            partial = {str(h["para_id"]): None for h in chunk}
        else:
            partial = _validate(partial, valid_sc_ids)
            # Fill in any headings the model silently omitted
            for h in chunk:
                k = str(h["para_id"])
                if k not in partial:
                    partial[k] = None

        # ── Persist to JSONL cache (one line per chunk) ────────────────────
        _append_jsonl(cache, [partial])
        cached_map.update(partial)

        done += len(chunk)
        mapped = sum(1 for v in partial.values() if v is not None)
        prog.update(done, f"errors={errors} mapped={mapped}")

        # Polite inter-chunk delay
        if chunk_start + HEADING_CHUNK_SIZE < len(todo_headings):
            time.sleep(1.0)

    prog.finish(f"errors={errors}")

    # ── Write merged mapping to DB ─────────────────────────────────────────
    n = save_mapping(nissaya_conn, book_id, cached_map, dry_run)
    return n


# =============================================================================
#  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Map nissaya headings → sc_id via Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python match_headings_to_sc.py                     # all Mūla books
  python match_headings_to_sc.py --book-id D-i       # single book
  python match_headings_to_sc.py --reset             # clear cache & redo all
  python match_headings_to_sc.py --dry-run           # print mappings, no DB write
        """,
    )
    ap.add_argument("--nissaya",  default=NISSAYA_DB_PATH)
    ap.add_argument("--scdata",   default=SC_DATA_DB_PATH)
    ap.add_argument("--book-id",  default=None,
                    help="Process only this book_id (default: all Mūla books)")
    ap.add_argument("--reset",    action="store_true",
                    help="Clear JSONL cache and re-match from scratch")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print mappings without writing to nissaya.db")
    args = ap.parse_args()

    nissaya_conn = sqlite3.connect(args.nissaya)
    sc_conn      = sqlite3.connect(args.scdata)
    nissaya_conn.execute("PRAGMA journal_mode=WAL")

    ensure_sc_id_column(nissaya_conn)

    gemini = GeminiClient()     # reads keys from .env via config.py

    # ── Book list ────────────────────────────────────────────────────────
    if args.book_id:
        row = nissaya_conn.execute(
            "SELECT book_id, book_name FROM books WHERE book_id = ?",
            (args.book_id,),
        ).fetchone()
        if not row:
            _log(f"[ERROR] book_id {args.book_id!r} not found.")
            sys.exit(1)
        books = [row]
    else:
        books = get_mula_books(nissaya_conn)

    _log(f"[START] {len(books)} book(s) to process.")

    total_assigned = 0
    for book_id, book_name in books:
        _log(f"── {book_id}  {book_name}")
        n = match_book(
            book_id      = book_id,
            book_name    = book_name,
            nissaya_conn = nissaya_conn,
            sc_conn      = sc_conn,
            gemini       = gemini,
            dry_run      = args.dry_run,
            reset        = args.reset,
        )
        tag = "  [DRY RUN]" if args.dry_run else ""
        _log(f"  → {n} sc_id assignments written for {book_id}{tag}")
        total_assigned += n

    _log(
        f"\n✓ Done.  Total sc_id assignments: {total_assigned}"
        + ("  (DRY RUN — nothing written)" if args.dry_run else "")
    )

    # Print Gemini key status (mirrors `indexer.py keys` command)
    print("\n── Gemini Key Status ───────────────────────────────────────")
    for k in gemini.status():
        status = "❌ exhausted" if k["exhausted"] else "✅ available"
        print(f"  {k['key_suffix']}  "
              f"used {k['requests_today']:>5}/1500  "
              f"remaining {k['remaining']:>5}  {status}")
    print()

    nissaya_conn.close()
    sc_conn.close()


if __name__ == "__main__":
    main()