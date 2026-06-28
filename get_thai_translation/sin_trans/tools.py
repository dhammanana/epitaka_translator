"""
tools.py — LangChain tools for the translation agent.

Search strategy (layered, best-first):
  1. Exact LIKE
  2. FTS5 AND query
  3. FTS5 OR query  (relaxed)
  4. Normalised word-fragment LIKE (strips diacritics)
"""

import json
import logging
import re
import unicodedata
import sqlite3
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_sinhala_db  = None
_nissaya_db  = None
_current_si_filename = None
_current_ni_book_id  = None


def bind_databases(sinhala_db, nissaya_db):
    global _sinhala_db, _nissaya_db
    _sinhala_db = sinhala_db
    _nissaya_db = nissaya_db


def bind_current_book(si_filename: str, ni_book_id: str):
    global _current_si_filename, _current_ni_book_id
    _current_si_filename = si_filename
    _current_ni_book_id  = ni_book_id


# ── Text normalisation ─────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    stripped = re.sub(r"[^\w\s]", " ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def _keywords(text: str) -> list[str]:
    return [w for w in _normalize(text).split() if len(w) >= 3]


# ── FTS5 setup ─────────────────────────────────────────────────────────────────

def _ensure_fts_sinhala(conn: sqlite3.Connection):
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
        USING fts5(
            id UNINDEXED, filename UNINDEXED,
            page_num UNINDEXED, page_order UNINDEXED, entry_order UNINDEXED,
            palitext, sinhalatext,
            content='entries', content_rowid='id'
        )
    """)
    cur = conn.execute("SELECT COUNT(*) FROM entries_fts")
    if cur.fetchone()[0] == 0:
        logger.info("Building FTS5 index for sinhala entries (one-time)…")
        conn.execute("""
            INSERT INTO entries_fts(rowid, id, filename, page_num, page_order,
                                    entry_order, palitext, sinhalatext)
            SELECT id, id, filename, page_num, page_order,
                   entry_order, palitext, sinhalatext FROM entries
        """)
        conn.commit()


def _ensure_fts_nissaya(conn: sqlite3.Connection):
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts
        USING fts5(
            book_id UNINDEXED, para_id UNINDEXED, line_id UNINDEXED,
            pali_sentence,
            content='sentences', content_rowid='rowid'
        )
    """)
    cur = conn.execute("SELECT COUNT(*) FROM sentences_fts")
    if cur.fetchone()[0] == 0:
        logger.info("Building FTS5 index for nissaya sentences (one-time)…")
        conn.execute("""
            INSERT INTO sentences_fts(rowid, book_id, para_id, line_id, pali_sentence)
            SELECT rowid, book_id, para_id, line_id, pali_sentence FROM sentences
        """)
        conn.commit()


# ── Layered search helpers ─────────────────────────────────────────────────────

def _search_entries_layered(conn, filename, keyword, limit=15):
    cols = "id, filename, page_num, page_order, entry_order, palitext, sinhalatext"

    # 1. Exact LIKE
    rows = _entry_like(conn, filename, keyword, cols, limit)
    if rows: return _entry_offsets(conn, filename, rows)

    # 2 & 3. FTS5
    _ensure_fts_sinhala(conn)
    words = _keywords(keyword)
    if words:
        for joiner in (" AND ", " OR "):
            fts_q = joiner.join(f'"{w}"' for w in words[:6])
            rows = _entry_fts(conn, filename, fts_q, cols, limit)
            if rows: return _entry_offsets(conn, filename, rows)

    # 4. Normalised fragments
    for word in _keywords(keyword)[:4]:
        rows = _entry_like(conn, filename, word, cols, limit)
        if rows: return _entry_offsets(conn, filename, rows)

    return []


def _search_sentences_layered(conn, book_id, keyword, limit=15):
    cols = "book_id, para_id, line_id, pali_sentence"

    # 1. Exact LIKE
    rows = _sent_like(conn, book_id, keyword, cols, limit)
    if rows: return _sent_offsets(conn, book_id, rows)

    # 2 & 3. FTS5
    _ensure_fts_nissaya(conn)
    words = _keywords(keyword)
    if words:
        for joiner in (" AND ", " OR "):
            fts_q = joiner.join(f'"{w}"' for w in words[:6])
            rows = _sent_fts(conn, book_id, fts_q, cols, limit)
            if rows: return _sent_offsets(conn, book_id, rows)

    # 4. Normalised fragments
    for word in _keywords(keyword)[:4]:
        rows = _sent_like(conn, book_id, word, cols, limit)
        if rows: return _sent_offsets(conn, book_id, rows)

    return []


def _entry_like(conn, filename, kw, cols, limit):
    cur = conn.execute(f"""
        SELECT {cols} FROM entries
        WHERE filename=? AND (palitext LIKE ? OR sinhalatext LIKE ?)
        ORDER BY page_num, page_order, entry_order LIMIT ?
    """, (filename, f"%{kw}%", f"%{kw}%", limit))
    return [dict(r) for r in cur.fetchall()]


def _entry_fts(conn, filename, fts_q, cols, limit):
    try:
        cur = conn.execute(f"""
            SELECT {cols} FROM entries
            WHERE filename=?
              AND id IN (SELECT id FROM entries_fts WHERE entries_fts MATCH ? AND filename=?)
            ORDER BY page_num, page_order, entry_order LIMIT ?
        """, (filename, fts_q, filename, limit))
        return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


def _entry_offsets(conn, filename, rows):
    for r in rows:
        cur = conn.execute("""
            SELECT COUNT(*) FROM entries WHERE filename=?
              AND (page_num, page_order, entry_order) <
                  (SELECT page_num, page_order, entry_order FROM entries WHERE id=?)
        """, (filename, r["id"]))
        r["row_offset"] = cur.fetchone()[0]
    return rows


def _sent_like(conn, book_id, kw, cols, limit):
    cur = conn.execute(f"""
        SELECT {cols} FROM sentences
        WHERE book_id=? AND pali_sentence LIKE ?
        ORDER BY para_id, line_id LIMIT ?
    """, (book_id, f"%{kw}%", limit))
    return [dict(r) for r in cur.fetchall()]


def _sent_fts(conn, book_id, fts_q, cols, limit):
    try:
        cur = conn.execute(f"""
            SELECT {cols} FROM sentences
            WHERE book_id=?
              AND rowid IN (SELECT rowid FROM sentences_fts WHERE sentences_fts MATCH ? AND book_id=?)
            ORDER BY para_id, line_id LIMIT ?
        """, (book_id, fts_q, book_id, limit))
        return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


def _sent_offsets(conn, book_id, rows):
    for r in rows:
        cur = conn.execute("""
            SELECT COUNT(*) FROM sentences
            WHERE book_id=? AND (para_id, line_id) < (?, ?)
        """, (book_id, r["para_id"], r["line_id"]))
        r["row_offset"] = cur.fetchone()[0]
    return rows


# ── LangChain tools ────────────────────────────────────────────────────────────

@tool
def search_sinhala_entries(keyword: str) -> str:
    """
    Search Sinhala entries of the current book by Pali or Sinhala keyword.
    Handles diacritics (ā/a, ṭ/t), case, punctuation automatically.
    Returns entries with row_offset for reposition_window(si_offset=...).

    Args:
        keyword: Any Pali or Sinhala text fragment.
    """
    if not _sinhala_db or not _current_si_filename:
        return json.dumps({"error": "Database not bound"})
    rows = _search_entries_layered(_sinhala_db.conn, _current_si_filename, keyword)
    if not rows:
        return json.dumps({"found": False, "message": f"No entries for '{keyword}'. Try shorter keyword."})
    out = [{"id": r["id"], "row_offset": r["row_offset"], "page_num": r["page_num"],
            "palitext": (r.get("palitext") or "")[:200],
            "sinhalatext": (r.get("sinhalatext") or "")[:200]} for r in rows]
    return json.dumps({"found": True, "count": len(out), "entries": out})


@tool
def search_nissaya_sentences(keyword: str) -> str:
    """
    Search Nissaya Pali sentences of the current book by keyword.
    Handles diacritics (ā/a, ṭ/t), case, punctuation automatically.
    Returns sentences with row_offset for reposition_window(ni_offset=...).

    Args:
        keyword: Any Pali text fragment.
    """
    if not _nissaya_db or not _current_ni_book_id:
        return json.dumps({"error": "Database not bound"})
    rows = _search_sentences_layered(_nissaya_db.conn, _current_ni_book_id, keyword)
    if not rows:
        return json.dumps({"found": False, "message": f"No sentences for '{keyword}'. Try shorter keyword."})
    out = [{"book_id": r["book_id"], "para_id": r["para_id"], "line_id": r["line_id"],
            "row_offset": r["row_offset"],
            "pali_sentence": (r.get("pali_sentence") or "")[:300]} for r in rows]
    return json.dumps({"found": True, "count": len(out), "sentences": out})


@tool
def reposition_window(si_offset: Optional[int] = None, ni_offset: Optional[int] = None) -> str:
    """
    Reposition one or both sliding windows to a specific row offset.
    Call after a search returns a row_offset for the correct alignment point.

    Args:
        si_offset: New 0-based row offset for the sinhala window (omit to keep current).
        ni_offset: New 0-based row offset for the nissaya window (omit to keep current).
    """
    result: dict = {}
    if si_offset is not None:
        result["si_offset"] = int(si_offset)
    if ni_offset is not None:
        result["ni_offset"] = int(ni_offset)
    if not result:
        return json.dumps({"error": "Provide at least one of si_offset or ni_offset."})
    return json.dumps({"reposition": result})


ALL_TOOLS = [search_sinhala_entries, search_nissaya_sentences, reposition_window]


def _invoke_tool(tool_name: str, tool_args: dict) -> str:
    tool_map = {t.name: t for t in ALL_TOOLS}
    if tool_name not in tool_map:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = tool_map[tool_name].invoke(tool_args)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})