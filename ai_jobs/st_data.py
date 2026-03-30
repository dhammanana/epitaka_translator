"""
jobs/st_data.py  —  Data-access layer for SentenceTranslatorJob.

Design rule: every function is self-contained.
  • It resolves the DB path from config / params.
  • Opens a connection with a context manager (WITH … AS conn).
  • Fetches or writes what it needs.
  • The connection is closed automatically when the WITH block exits.

No connection object is ever passed in or stored between calls.
This eliminates the risk of long-held locks causing deadlocks.

Public API
----------
  # path resolvers (pure helpers, no I/O)
  nissaya_path(params) → str
  sc_path(params)      → str

  # reads
  fetch_headings(params, log_info)                              → list[dict]
  fetch_paragraphs_for_heading(params, heading, overwrite,
                               log_info, log_warn)             → list[dict]
  fetch_sc_reference(params, sc_id, log_info, log_warn)        → dict
  fetch_nissaya_map(params, book_id, para_id, log_info)         → dict[int, str]
  build_nissaya_block(sentences, nissaya_map)                   → str
  extract_pali_tokens(text)                                     → list[str]
  fetch_glossary_block(pali_tokens, log_info, log_warn)         → str
  fetch_commentary_block(params, line_ids, max_chars,
                         log_info, log_warn)                    → str
  fetch_pali_definitions_block(pali_text, params,
                               log_info, log_warn)              → str
  fetch_previous_paragraph_translation(params, book_id, para_id,
                                       log_info)                → str

  # writes
  save_translations(params, book_id, para_id,
                    translations, log_info, log_warn, log_error) → int
  upsert_glossary(terms, sc_id, log_info, log_warn, log_error)  → int

  # startup validation (called once at job start, does NOT keep conn open)
  validate_nissaya_db(params, log_info, log_warn)
  validate_sc_db(params, log_info, log_warn)
"""

import re
import sqlite3
import logging
from contextlib import contextmanager
from typing import Callable

from database import get_glossary_conn
from config import SC_DATA_DB, NISSAYA_DB

logger = logging.getLogger(__name__)

_Log = Callable[[str], None]


# ══════════════════════════════════════════════════════════════════
# Path resolvers
# ══════════════════════════════════════════════════════════════════

def nissaya_path(params: dict) -> str:
    path = params.get("nissaya_db") or NISSAYA_DB
    if not path:
        raise RuntimeError(
            "nissaya_db path is not configured. "
            "Set NISSAYA_DB in config.py or pass it as a task param."
        )
    return str(path)


def sc_path(params: dict) -> str:
    path = params.get("source_db") or SC_DATA_DB
    if not path:
        raise RuntimeError(
            "SC_DATA_DB path is not configured. "
            "Set SC_DATA_DB in config.py or pass it as a task param."
        )
    return str(path)


# ══════════════════════════════════════════════════════════════════
# Internal connection helper
# ══════════════════════════════════════════════════════════════════

@contextmanager
def _connect(path: str):
    """Open a SQLite connection, yield it, always close on exit."""
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")   # wait up to 10 s on lock
    try:
        yield conn
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════
# Startup validation  (one-shot, no persistent connection)
# ══════════════════════════════════════════════════════════════════

def validate_nissaya_db(params: dict, log_info: _Log, log_warn: _Log) -> None:
    path = nissaya_path(params)
    with _connect(path) as conn:
        tables = {
            r[0] for r in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    log_info(f"[DB] nissaya.db at {path!r}. Tables: {sorted(tables)}")
    for t in ("books", "headings", "nissaya", "sentences", "pali_definition"):
        if t not in tables:
            log_warn(f"[DB] nissaya.db MISSING expected table: '{t}'")


def validate_sc_db(params: dict, log_info: _Log, log_warn: _Log) -> None:
    path = sc_path(params)
    with _connect(path) as conn:
        tables = {
            r[0] for r in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    log_info(f"[DB] sc-data.db at {path!r}. Tables: {sorted(tables)}")
    if "en_translation" not in tables:
        log_warn("[DB] sc-data.db MISSING expected table: 'en_translation'")


# ══════════════════════════════════════════════════════════════════
# Headings
# ══════════════════════════════════════════════════════════════════

def fetch_headings(params: dict, log_info: _Log) -> list[dict]:
    """
    Return headings that have a non-null sc_id and still have pending work.
    Params consumed: book_id, sc_id, batch_size, overwrite.
    """
    book_id_filter = params.get("book_id", "").strip()
    sc_id_filter   = params.get("sc_id",   "").strip()
    batch_size     = int(params.get("batch_size", 5))
    overwrite      = bool(params.get("overwrite", False))

    log_info(
        f"[DB] fetch_headings called — "
        f"book_id={book_id_filter!r}, sc_id={sc_id_filter!r}, "
        f"batch_size={batch_size}, overwrite={overwrite}"
    )

    path = nissaya_path(params)
    log_info(f"[DB] fetch_headings: opening connection to {path!r} ...")

    try:
        with _connect(path) as conn:
            log_info("[DB] fetch_headings: connection opened.")

            # Step 1 — fetch candidate headings (no subquery, no correlated scan)
            if sc_id_filter:
                log_info(f"[DB] fetch_headings: sc_id filter={sc_id_filter!r}")
                rows = conn.execute(
                    "SELECT * FROM headings WHERE sc_id = ?",
                    (sc_id_filter,)
                ).fetchall()
                candidates = [dict(r) for r in rows]

            elif book_id_filter:
                log_info(f"[DB] fetch_headings: book_id filter={book_id_filter!r}")
                rows = conn.execute(
                    """SELECT * FROM headings
                       WHERE book_id = ?
                         AND sc_id IS NOT NULL AND sc_id != ''
                       ORDER BY para_id""",
                    (book_id_filter,)
                ).fetchall()
                candidates = [dict(r) for r in rows]

            else:
                log_info("[DB] fetch_headings: fetching all headings with sc_id ...")
                rows = conn.execute(
                    """SELECT * FROM headings
                       WHERE sc_id IS NOT NULL AND sc_id != ''
                       ORDER BY book_id, para_id"""
                ).fetchall()
                candidates = [dict(r) for r in rows]

            log_info(f"[DB] fetch_headings: {len(candidates)} candidate heading(s) found.")

            # Step 2 — if not overwrite, filter out headings whose paragraph range
            #           is fully translated.  We check one quick COUNT per heading
            #           (targeted index lookup, not a correlated subquery over all rows).
            if overwrite or not candidates:
                result = candidates[:batch_size]
            else:
                log_info("[DB] fetch_headings: filtering already-completed headings ...")
                result = []
                for h in candidates:
                    if len(result) >= batch_size:
                        break
                    start = h["para_id"]
                    end   = start + int(h.get("chapter_len") or 1)
                    row = conn.execute(
                        """SELECT COUNT(*) FROM sentences
                           WHERE book_id = ?
                             AND para_id >= ? AND para_id < ?
                             AND (english_translation IS NULL
                                  OR english_translation = '')""",
                        (h["book_id"], start, end)
                    ).fetchone()
                    pending_count = row[0]
                    if pending_count > 0:
                        result.append(h)

            log_info(f"[DB] fetch_headings: {len(result)} heading(s) need work.")

    except Exception as exc:
        log_info(f"[DB] fetch_headings: EXCEPTION — {type(exc).__name__}: {exc}")
        raise

    log_info(f"[DB] fetch_headings → {len(result)} heading(s) returned.")
    return result


# ══════════════════════════════════════════════════════════════════
# Paragraphs & sentences
# ══════════════════════════════════════════════════════════════════

def fetch_paragraphs_for_heading(
    params:    dict,
    heading:   dict,
    overwrite: bool,
    log_info:  _Log,
    log_warn:  _Log,
) -> list[dict]:
    """
    A heading covers para_id .. para_id + chapter_len - 1.

    Returns a list of paragraph dicts:
      {
        "book_id":   str,
        "para_id":   int,
        "sentences": [ {"line_id": int, "pali_sentence": str} ],
        "pending":   [ {"line_id": int, "pali_sentence": str} ]  — only if not overwrite
      }
    """
    path = nissaya_path(params)
    start_para = heading["para_id"]
    chapter_len = int(heading.get("chapter_len") or 1)
    end_para = start_para + chapter_len
    book_id = heading["book_id"]

    log_info(
        f"[DB] fetch_paragraphs_for_heading: {book_id} "
        f"para_id={start_para}..{end_para - 1}"
    )

    result = []
    with _connect(path) as conn:
        for pid in range(start_para, end_para):
            rows = conn.execute(
                """SELECT line_id, pali_sentence, english_translation
                   FROM sentences
                   WHERE book_id = ? AND para_id = ?
                   ORDER BY line_id""",
                (book_id, pid)
            ).fetchall()

            sentences = [dict(r) for r in rows]
            if not sentences:
                continue

            # Filter pending (untranslated)
            pending = [
                s for s in sentences
                if not s["english_translation"] or s["english_translation"].strip() == ""
            ] if not overwrite else sentences

            # If there is nothing to translate and we aren't in overwrite mode,
            # do not add this paragraph to the workload.
            if not pending and not overwrite:
                continue

            result.append({
                "book_id":   book_id,
                "para_id":   pid,
                "sentences": sentences,
                "pending":   pending,
            })

    log_info(
        f"[DB] fetch_paragraphs_for_heading: {len(result)} para(s) with "
        f"{sum(len(p['pending']) for p in result)} pending sentence(s)."
    )
    return result


def fetch_sc_reference(
    params:   dict,
    sc_id:    str,
    log_info: _Log,
    log_warn: _Log,
) -> dict:
    """Return {"sutta_name", "pali_text", "en_text"} for the given sc_id."""
    with _connect(sc_path(params)) as conn:
        row = conn.execute(
            "SELECT * FROM en_translation WHERE sc_id = ?", (sc_id,)
        ).fetchone()

    if not row:
        log_warn(f"[DB] No en_translation row for sc_id={sc_id!r}.")
        return {"sutta_name": sc_id, "pali_text": "", "en_text": ""}

    ref = {
        "sutta_name": row["sutta_name"] or sc_id,
        "pali_text":  row["palitext"]   or "",
        "en_text":    row["entext"]     or "",
    }
    log_info(
        f"[DB] SC ref for {sc_id!r}: name={ref['sutta_name']!r}, "
        f"pali={len(ref['pali_text'])} chars, en={len(ref['en_text'])} chars."
    )
    return ref


# ══════════════════════════════════════════════════════════════════
# Nissaya
# ══════════════════════════════════════════════════════════════════

def fetch_nissaya_map(
    params:   dict,
    book_id:  str,
    para_id:  int,
    log_info: _Log,
) -> dict[int, str]:
    """
    Return { line_id: content } for the given paragraph.
    Schema: nissaya(book_id, para_id, line_id, content, channel_id)
    """
    with _connect(nissaya_path(params)) as conn:
        rows = conn.execute(
            """SELECT line_id, content
               FROM nissaya
               WHERE book_id = ? AND para_id = ?
               ORDER BY line_id""",
            (book_id, para_id)
        ).fetchall()

    result = {r["line_id"]: (r["content"] or "") for r in rows}
    log_info(f"[DB]   nissaya rows for para_id={para_id}: {len(result)}")
    return result


def build_nissaya_block(sentences: list[dict], nissaya_map: dict[int, str]) -> str:
    """
    Build a human-readable nissaya block aligned exactly by line_id.

      [line_id=1] <pali_sentence>
        Nissaya: <myanmar gloss>
    """
    if not nissaya_map:
        return "(no nissaya available for this paragraph)"

    lines = []
    for s in sentences:
        lid       = s["line_id"]
        niss_text = (nissaya_map.get(lid) or "").strip() or "(none)"
        lines.append(
            f"[line_id={lid}] {s.get('pali_sentence', '')}\n"
            f"  Nissaya: {niss_text}"
        )
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Glossary — read
# ══════════════════════════════════════════════════════════════════

def extract_pali_tokens(text: str) -> list[str]:
    tokens = re.split(r"[\s,;.\u2018\u2019\"'()\[\]]+", text)
    return list({t.lower() for t in tokens if 2 < len(t) < 40})


def fetch_glossary_block(
    pali_tokens: list[str],
    log_info:    _Log,
    log_warn:    _Log,
) -> str:
    """
    Open glossary.db, query matching terms, close, return formatted string.
    """
    if not pali_tokens:
        return "(no existing glossary entries)"

    try:
        # get_glossary_conn() returns a plain connection; we close it manually.
        conn = get_glossary_conn()
        try:
            placeholders = ",".join("?" * len(pali_tokens))
            rows = conn.execute(
                f"SELECT pali, english, context FROM glossary "
                f"WHERE pali IN ({placeholders})",
                pali_tokens
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        log_warn(f"[DB] Glossary lookup failed (non-fatal): {exc}")
        return "(glossary unavailable)"

    if not rows:
        return "(no matching glossary entries yet)"

    log_info(f"[DB] Glossary: {len(rows)} matching term(s) loaded.")
    lines = []
    for r in rows:
        context = r["context"] or ""
        line = f"  {r['pali']} → {r['english']}"
        if context:
            line += f"  [{context}]"
        lines.append(line)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Commentary & Sub-commentary (via book_links)
# ══════════════════════════════════════════════════════════════════

def fetch_commentary_block(
    params:    dict,
    src_lines: list[tuple],   # (book_id, para_id, line_id) — full composite key
    max_chars: int = 3000,
    log_info:  _Log = None,
    log_warn:  _Log = None,
) -> str:
    """
    Fetch commentary (atthakatha) and sub-commentary (tika) text linked to
    the given source sentences via the book_links table.

    src_lines : list of (src_book, src_para, src_line) tuples.
                line_id alone is NOT globally unique — it is only the row
                order within one paragraph, so we must match all three columns.

    Three-tier fallback when result exceeds max_chars:

      Tier 1 — Full paragraphs
        Fetch every sentence in every dst paragraph that has at least one
        link. Duplicate paragraphs are collapsed.

      Tier 2 — 3-line window (i-1, i, i+1)
        For every directly-linked dst_line, fetch that line plus its
        immediate neighbours. Overlapping windows are deduplicated.

      Tier 3 — Exact linked lines only (i)
        Keep only the single dst_line that is directly referenced.

    All bulk DB lookups use TEMP TABLE JOINs to avoid SQLite's
    bound-variable limit.
    """
    if log_info is None:
        log_info = lambda x: None
    if log_warn is None:
        log_warn = lambda x: None

    if not src_lines:
        return "(no commentary available)"

    path = nissaya_path(params)

    try:
        with _connect(path) as conn:

            # 0. Guard: check book_links exists
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='book_links'"
            ).fetchone()
            if not has_table:
                log_warn("[DB] book_links table not found. Skipping commentary fetch.")
                return "(no commentary available)"

            # 1. Load source keys into a temp table, then JOIN book_links.
            #    Matching on all three columns (src_book, src_para, src_line)
            #    because line_id is only unique within its own paragraph.
            conn.execute("DROP TABLE IF EXISTS _tmp_src")
            conn.execute(
                "CREATE TEMP TABLE _tmp_src"
                " (book_id TEXT, para_id INTEGER, line_id INTEGER)"
            )
            conn.executemany("INSERT INTO _tmp_src VALUES (?,?,?)", src_lines)

            link_rows = conn.execute(
                """SELECT bl.src_book, bl.src_para, bl.src_line,
                          bl.dst_book, bl.dst_para, bl.dst_line
                   FROM   book_links bl
                   JOIN   _tmp_src s
                          ON  bl.src_book = s.book_id
                          AND bl.src_para = s.para_id
                          AND bl.src_line = s.line_id
                   ORDER  BY bl.dst_book, bl.dst_para, bl.dst_line"""
            ).fetchall()
            conn.execute("DROP TABLE IF EXISTS _tmp_src")

            if not link_rows:
                log_info(
                    f"[DB] Commentary: no book_links for {len(src_lines)} source line(s)."
                )
                return "(no commentary available)"

            log_info(
                f"[DB] Commentary: {len(link_rows)} link row(s) for "
                f"{len(src_lines)} source line(s)."
            )

            # Unique (dst_book, dst_para, dst_line) targets, order preserved.
            seen_targets: dict[tuple, None] = {}
            for r in link_rows:
                seen_targets[(r["dst_book"], r["dst_para"], r["dst_line"])] = None
            targets = list(seen_targets.keys())
            log_info(f"[DB] Commentary: {len(targets)} unique target(s) after dedup.")

            # Helpers: bulk fetches via TEMP TABLE

            def _fetch_exact(triples: list[tuple]) -> list[sqlite3.Row]:
                """Fetch sentences for explicit (book_id, para_id, line_id) triples."""
                if not triples:
                    return []
                conn.execute("DROP TABLE IF EXISTS _tmp_targets")
                conn.execute(
                    "CREATE TEMP TABLE _tmp_targets"
                    " (book_id TEXT, para_id INTEGER, line_id INTEGER)"
                )
                conn.executemany("INSERT INTO _tmp_targets VALUES (?,?,?)", triples)
                rows = conn.execute(
                    """SELECT s.book_id, s.para_id, s.line_id, s.pali_sentence
                       FROM   sentences s
                       JOIN   _tmp_targets t
                              ON  s.book_id = t.book_id
                              AND s.para_id = t.para_id
                              AND s.line_id = t.line_id
                       ORDER  BY s.book_id, s.para_id, s.line_id"""
                ).fetchall()
                conn.execute("DROP TABLE IF EXISTS _tmp_targets")
                return rows

            def _fetch_paragraphs(pairs: list[tuple]) -> list[sqlite3.Row]:
                """Fetch all sentences for (book_id, para_id) pairs."""
                if not pairs:
                    return []
                conn.execute("DROP TABLE IF EXISTS _tmp_paras")
                conn.execute(
                    "CREATE TEMP TABLE _tmp_paras (book_id TEXT, para_id INTEGER)"
                )
                conn.executemany("INSERT INTO _tmp_paras VALUES (?,?)", pairs)
                rows = conn.execute(
                    """SELECT s.book_id, s.para_id, s.line_id, s.pali_sentence
                       FROM   sentences s
                       JOIN   _tmp_paras t
                              ON  s.book_id = t.book_id
                              AND s.para_id = t.para_id
                       ORDER  BY s.book_id, s.para_id, s.line_id"""
                ).fetchall()
                conn.execute("DROP TABLE IF EXISTS _tmp_paras")
                return rows

            def _render(rows: list[sqlite3.Row], label: str) -> str:
                if not rows:
                    return ""
                sections: dict[tuple, list[str]] = {}
                for r in rows:
                    key = (r["book_id"], r["para_id"])
                    sections.setdefault(key, []).append(
                        f"  [{r['line_id']}] {r['pali_sentence'] or ''}"
                    )
                parts = [label]
                for (book, para), sent_lines in sections.items():
                    parts.append(f"\n[{book} §{para}]")
                    parts.extend(sent_lines)
                return "\n".join(parts)

            # Tier 1: full paragraphs
            unique_paras: dict[tuple, None] = {}
            for dst_book, dst_para, _ in targets:
                unique_paras[(dst_book, dst_para)] = None
            log_info(
                f"[DB] Commentary tier-1: fetching {len(unique_paras)} unique paragraph(s)."
            )
            tier1_rows = _fetch_paragraphs(list(unique_paras.keys()))
            tier1_text = _render(tier1_rows, "[Commentary & Sub-commentary - full paragraphs]")

            if max_chars <= 0 or len(tier1_text) <= max_chars:
                log_info(
                    f"[DB] Commentary tier-1 OK: "
                    f"{len(tier1_text)} chars, {len(tier1_rows)} sentence(s)."
                )
                return tier1_text or "(no commentary available)"

            # Tier 2: 3-line window (i-1, i, i+1)
            log_info(
                f"[DB] Commentary tier-1 too long ({len(tier1_text)} > {max_chars}); "
                f"falling back to 3-line window."
            )
            window_set: dict[tuple, None] = {}
            for dst_book, dst_para, dst_line in targets:
                for offset in (-1, 0, 1):
                    window_set[(dst_book, dst_para, dst_line + offset)] = None
            log_info(f"[DB] Commentary tier-2: {len(window_set)} window line(s) to fetch.")
            tier2_rows = _fetch_exact(list(window_set.keys()))
            tier2_text = _render(tier2_rows, "[Commentary & Sub-commentary - 3-line window]")

            if len(tier2_text) <= max_chars:
                log_info(
                    f"[DB] Commentary tier-2 OK: "
                    f"{len(tier2_text)} chars, {len(tier2_rows)} sentence(s)."
                )
                return tier2_text or "(no commentary available)"

            # Tier 3: exact linked lines only
            log_info(
                f"[DB] Commentary tier-2 too long ({len(tier2_text)} > {max_chars}); "
                f"falling back to exact linked lines only."
            )
            log_info(f"[DB] Commentary tier-3: {len(targets)} exact target(s).")
            tier3_rows = _fetch_exact(targets)
            tier3_text = _render(tier3_rows, "[Commentary & Sub-commentary - linked lines only]")

            log_info(
                f"[DB] Commentary tier-3 final: "
                f"{len(tier3_text)} chars, {len(tier3_rows)} sentence(s)."
            )
            return tier3_text or "(no commentary available)"

    except Exception as exc:
        log_warn(f"[DB] Commentary fetch failed: {exc}")
        return "(no commentary available)"


# ══════════════════════════════════════════════════════════════════
# Pali Definitions — lookup with context sentences
# ══════════════════════════════════════════════════════════════════

def _get_usages_for_stem(conn, stem: str, limit: int = 3) -> list[dict]:
    """
    Find sentences in pali_definition where the stem matches, then return
    the definition sentence plus 1 before and 1 after for context.
    Returns list of dicts with the Pali text (already in HTML format).
    """
    # Find the definition sentence for this stem
    rows = conn.execute("""
        SELECT DISTINCT
            pd.book_id,
            pd.para_id,
            pd.line_id
        FROM pali_definition pd
        WHERE pd.stem = ?
        LIMIT 1
    """, (stem,)).fetchall()
    
    if not rows:
        return []
    
    book_id, para_id, line_id = rows[0]
    
    # Get the definition sentence and context (1 before, 1 after)
    context_rows = conn.execute("""
        SELECT line_id, pali_sentence
        FROM sentences
        WHERE book_id = ? AND para_id = ?
        AND line_id BETWEEN ? AND ?
        ORDER BY line_id
    """, (book_id, para_id, line_id - 1, line_id + 1)).fetchall()
    
    usages = []
    for row in context_rows:
        usages.append({
            "book_id": book_id,
            "para_id": para_id,
            "line_id": row["line_id"],
            "pali": row["pali_sentence"] or "",
        })
    
    return usages


def fetch_pali_definitions_block(
    pali_text: str,
    params:    dict,
    log_info:  _Log,
    log_warn:  _Log,
) -> str:
    """
    Extract all words longer than 4 chars from pali_text, look up their stems
    in pali_definition table, and return example sentences (definition line + context).
    
    Returns formatted string for inclusion in prompt.
    """
    # Extract words longer than 4 chars
    words = re.findall(r"\b[\w\u0900-\u097F]{5,}\b", pali_text)
    words = list(set(w.lower() for w in words))  # deduplicate
    
    if not words:
        return "(no word definitions available)"
    
    try:
        path = nissaya_path(params)
        with _connect(path) as conn:
            # Check if pali_definition table exists
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pali_definition'"
            ).fetchone()
            if not tables:
                log_warn("[DB] pali_definition table not found.")
                return "(no word definitions available)"
            
            # For each word, try to find its stem and example usages
            defs_block = []
            for word in words[:30]:  # Limit to first 30 unique words to avoid huge output
                # Try to find stems for this word (it could be inflected)
                stem_rows = conn.execute("""
                    SELECT DISTINCT stem FROM pali_definition
                    WHERE plain = ? OR word = ?
                    LIMIT 1
                """, (word, word)).fetchall()
                
                if stem_rows:
                    for (stem,) in stem_rows:
                        usages = _get_usages_for_stem(conn, stem, limit=1)
                        if usages:
                            pali_sentences = " … ".join(u["pali"] for u in usages)
                            defs_block.append(f"  {word} (stem: {stem}): {pali_sentences}")
            
            if defs_block:
                result = "Pali Word Definitions:\n" + "\n".join(defs_block)
                log_info(f"[DB] Pali definitions: {len(defs_block)} word(s) defined.")
                return result
            else:
                return "(no word definitions found)"
    
    except Exception as exc:
        log_warn(f"[DB] Pali definitions lookup failed: {exc}")
        return "(word definitions unavailable)"


# ══════════════════════════════════════════════════════════════════
# Previous Paragraph Context
# ══════════════════════════════════════════════════════════════════

def fetch_previous_paragraph_translation(
    params:   dict,
    book_id:  str,
    para_id:  int,
    log_info: _Log,
) -> str:
    """
    Fetch the previous paragraph (para_id - 1) if it exists and is translated.
    Used to provide context to the AI for consistent translation style.
    
    Returns: (para_id, joined_english_translation) or empty string if not available.
    """
    if para_id <= 0:
        return ""
    
    path = nissaya_path(params)
    try:
        with _connect(path) as conn:
            rows = conn.execute(
                """SELECT line_id, english_translation FROM sentences
                   WHERE book_id = ? AND para_id = ?
                   AND english_translation IS NOT NULL
                   AND english_translation != ''
                   ORDER BY line_id""",
                (book_id, para_id - 1)
            ).fetchall()
        
        if rows:
            prev_text = " ".join(r["english_translation"].strip() for r in rows)
            log_info(f"[DB] Previous paragraph (para_id={para_id - 1}) found: {len(prev_text)} chars.")
            return f"[Previous Paragraph {para_id - 1}]\n{prev_text}"
        else:
            return ""
    except Exception as exc:
        log_info(f"[DB] Could not fetch previous paragraph: {exc}")
        return ""


# ══════════════════════════════════════════════════════════════════
# Translations — write
# ══════════════════════════════════════════════════════════════════

def save_translations(
    params:       dict,
    book_id:      str,
    para_id:      int,
    translations: list[dict],
    log_info:     _Log,
    log_warn:     _Log,
    log_error:    _Log,
) -> int:
    """
    Open nissaya.db, UPDATE sentences.english_translation, commit, close.
    Returns the number of rows changed.
    """
    updated = 0

    with _connect(nissaya_path(params)) as conn:
        for entry in translations:
            line_id = entry.get("line_id")
            text    = str(entry.get("english_translation") or "").strip()
            if line_id is None or not text:
                log_warn(f"[DB] Skipping empty/malformed translation entry: {entry}")
                continue
            try:
                conn.execute(
                    """UPDATE sentences
                       SET english_translation = ?
                       WHERE book_id = ? AND para_id = ? AND line_id = ?""",
                    (text, book_id, para_id, line_id)
                )
                updated += conn.execute("SELECT changes()").fetchone()[0]
            except Exception as exc:
                log_error(f"[DB] Error updating line_id={line_id}: {exc}")

        conn.commit()

    log_info(f"[DB] save_translations: {updated} row(s) updated for para_id={para_id}.")
    return updated


# ══════════════════════════════════════════════════════════════════
# Glossary — write
# ══════════════════════════════════════════════════════════════════

def upsert_glossary(
    terms:     list[dict],
    sc_id:     str,
    log_info:  _Log,
    log_warn:  _Log,
    log_error: _Log,
) -> int:
    """
    Open glossary.db, insert new terms (ON CONFLICT DO NOTHING), commit, close.
    Returns count of rows actually inserted.
    """
    required = {"pali", "english"}
    inserted = 0

    try:
        conn = get_glossary_conn()
    except Exception as exc:
        log_warn(f"[DB] Cannot open glossary DB: {exc}")
        return 0

    try:
        for term in terms:
            if not required.issubset(term):
                log_warn(f"[DB] Skipping malformed glossary entry: {term}")
                continue
            pali    = str(term.get("pali",    "")).strip()
            english = str(term.get("english", "")).strip()
            if not pali or not english:
                continue
            try:
                conn.execute(
                    """INSERT INTO glossary
                         (pali, english, domain, sub_domain, context, note, source_id)
                       VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(pali, english) DO NOTHING""",
                    (
                        pali, english,
                        str(term.get("domain",     "") or ""),
                        str(term.get("sub_domain", "") or ""),
                        str(term.get("context",    "") or ""),
                        str(term.get("note",       "") or ""),
                        sc_id,
                    )
                )
                inserted += conn.execute("SELECT changes()").fetchone()[0]
            except Exception as exc:
                log_error(f"[DB] Glossary insert error for '{pali}': {exc}")

        conn.commit()
    finally:
        conn.close()

    log_info(f"[DB] upsert_glossary: {inserted} new term(s) (sc_id={sc_id!r}).")
    return inserted