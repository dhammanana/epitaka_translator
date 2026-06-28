"""
db.py — Database access layer for all three databases.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SinhalaEntry:
    id: int
    filename: str
    page_num: int
    page_order: int
    entry_order: int
    type: Optional[str]
    palitext: Optional[str]
    sinhalatext: Optional[str]


@dataclass
class NissayaSentence:
    book_id: str
    para_id: int
    line_id: int
    pali_sentence: Optional[str]


@dataclass
class BookPair:
    id: int
    sinhala_filename: str
    sinhala_pali_roman: Optional[str]
    nissaya_book_id: str
    nissaya_book_name: Optional[str]


class SinhalaDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def get_all_filenames_with_pali(self) -> list[dict]:
        """Get all distinct filenames with their full ancestor path (pali_roman joined by ' > ')."""
        cur = self.conn.execute("""
            SELECT DISTINCT e.filename, tn.pali_roman, tn.parent_id
            FROM entries e
            LEFT JOIN tree_nodes tn ON tn.node_id = e.filename
            ORDER BY e.filename
        """)
        rows = [dict(r) for r in cur.fetchall()]

        # Build node_id -> (pali_roman, parent_id) map for ancestor walking
        node_cur = self.conn.execute("SELECT node_id, pali_roman, parent_id FROM tree_nodes")
        node_map = {r[0]: (r[1], r[2]) for r in node_cur.fetchall()}

        def build_path(node_id: str) -> str:
            parts = []
            current = node_id
            seen = set()
            while current and current not in seen:
                seen.add(current)
                info = node_map.get(current)
                if not info:
                    break
                roman, parent = info
                if roman:
                    parts.append(roman)
                current = parent
            parts.reverse()
            return " > ".join(parts)

        for row in rows:
            row["pali_roman"] = build_path(row["filename"]) if row["filename"] else ""
            del row["parent_id"]

        return rows

    def get_entries_window(self, filename: str, offset: int, limit: int = 100) -> list[SinhalaEntry]:
        """Get a window of entries for a given filename, ordered by page/order."""
        cur = self.conn.execute("""
            SELECT id, filename, page_num, page_order, entry_order, type, palitext, sinhalatext
            FROM entries
            WHERE filename = ?
            ORDER BY page_num, page_order, entry_order
            LIMIT ? OFFSET ?
        """, (filename, limit, offset))
        return [SinhalaEntry(**dict(r)) for r in cur.fetchall()]

    def search_entries(self, filename: str, keyword: str, limit: int = 20) -> list[SinhalaEntry]:
        """Search entries by keyword in palitext, return entries + offset for repositioning."""
        cur = self.conn.execute("""
            SELECT id, filename, page_num, page_order, entry_order, type, palitext, sinhalatext
            FROM entries
            WHERE filename = ?
              AND (palitext LIKE ? OR sinhalatext LIKE ?)
            ORDER BY page_num, page_order, entry_order
            LIMIT ?
        """, (filename, f"%{keyword}%", f"%{keyword}%", limit))
        return [SinhalaEntry(**dict(r)) for r in cur.fetchall()]

    def get_entry_offset(self, filename: str, entry_id: int) -> int:
        """Get the row offset of a specific entry_id within the ordered filename entries."""
        cur = self.conn.execute("""
            SELECT COUNT(*) FROM entries
            WHERE filename = ?
              AND (page_num, page_order, entry_order) < (
                  SELECT page_num, page_order, entry_order FROM entries WHERE id = ?
              )
        """, (filename, entry_id))
        row = cur.fetchone()
        return row[0] if row else 0

    def get_total_entries(self, filename: str) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM entries WHERE filename = ?", (filename,))
        return cur.fetchone()[0]

    def close(self):
        self.conn.close()


class NissayaDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def get_all_books(self) -> list[dict]:
        """Get all books with metadata."""
        print(self.path)
        cur = self.conn.execute("""
            SELECT book_id, book_name, vri_id, category, nikaya, sub_nikaya
            FROM books
            ORDER BY book_id
        """)
        return [dict(r) for r in cur.fetchall()]

    def get_sentences_window(self, book_id: str, offset: int, limit: int = 100) -> list[NissayaSentence]:
        """Get a window of sentences for a given book_id."""
        cur = self.conn.execute("""
            SELECT book_id, para_id, line_id, pali_sentence
            FROM sentences
            WHERE book_id = ?
            ORDER BY para_id, line_id
            LIMIT ? OFFSET ?
        """, (book_id, limit, offset))
        return [NissayaSentence(**dict(r)) for r in cur.fetchall()]

    def search_sentences(self, book_id: str, keyword: str, limit: int = 20) -> list[NissayaSentence]:
        """Search sentences by keyword in pali_sentence."""
        cur = self.conn.execute("""
            SELECT book_id, para_id, line_id, pali_sentence
            FROM sentences
            WHERE book_id = ?
              AND pali_sentence LIKE ?
            ORDER BY para_id, line_id
            LIMIT ?
        """, (book_id, f"%{keyword}%", limit))
        return [NissayaSentence(**dict(r)) for r in cur.fetchall()]

    def get_sentence_offset(self, book_id: str, para_id: int, line_id: int) -> int:
        """Get the row offset of a specific sentence within the ordered book sentences."""
        cur = self.conn.execute("""
            SELECT COUNT(*) FROM sentences
            WHERE book_id = ?
              AND (para_id, line_id) < (?, ?)
        """, (book_id, para_id, line_id))
        row = cur.fetchone()
        return row[0] if row else 0

    def get_total_sentences(self, book_id: str) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM sentences WHERE book_id = ?", (book_id,))
        return cur.fetchone()[0]

    def close(self):
        self.conn.close()


class OutputDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS book_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sinhala_filename TEXT NOT NULL,
                sinhala_pali_roman TEXT,
                nissaya_book_id TEXT NOT NULL,
                nissaya_book_name TEXT,
                UNIQUE(sinhala_filename, nissaya_book_id)
            );

            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_pair_id INTEGER NOT NULL REFERENCES book_pairs(id),
                si_cursor INTEGER NOT NULL DEFAULT 0,
                ni_cursor INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(book_pair_id)
            );

            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nissaya_book_id TEXT NOT NULL,
                nissaya_para_id INTEGER NOT NULL,
                line_id INTEGER NOT NULL,
                pali_sentence TEXT,
                sinhala_translation TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(nissaya_book_id, nissaya_para_id, line_id)
            );

            CREATE TABLE IF NOT EXISTS agent_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_pair_id INTEGER,
                level TEXT,
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    # ── Book pairs ──────────────────────────────────────────────────────────

    def save_book_pairs(self, pairs: list[dict]):
        self.conn.executemany("""
            INSERT OR IGNORE INTO book_pairs
                (sinhala_filename, sinhala_pali_roman, nissaya_book_id, nissaya_book_name)
            VALUES (:sinhala_filename, :sinhala_pali_roman, :nissaya_book_id, :nissaya_book_name)
        """, pairs)
        self.conn.commit()

    def get_book_pairs(self, status_filter: Optional[str] = None) -> list[BookPair]:
        if status_filter:
            cur = self.conn.execute("""
                SELECT bp.id, bp.sinhala_filename, bp.sinhala_pali_roman,
                       bp.nissaya_book_id, bp.nissaya_book_name
                FROM book_pairs bp
                LEFT JOIN progress p ON p.book_pair_id = bp.id
                WHERE p.status = ? OR p.status IS NULL
                ORDER BY bp.id
            """, (status_filter,))
        else:
            cur = self.conn.execute("""
                SELECT id, sinhala_filename, sinhala_pali_roman,
                       nissaya_book_id, nissaya_book_name
                FROM book_pairs
                ORDER BY id
            """)
        return [BookPair(**dict(r)) for r in cur.fetchall()]

    def has_book_pairs(self) -> bool:
        cur = self.conn.execute("SELECT COUNT(*) FROM book_pairs")
        return cur.fetchone()[0] > 0

    # ── Progress ─────────────────────────────────────────────────────────────

    def get_progress(self, book_pair_id: int) -> dict:
        cur = self.conn.execute("""
            SELECT * FROM progress WHERE book_pair_id = ?
        """, (book_pair_id,))
        row = cur.fetchone()
        if row:
            return dict(row)
        return {"book_pair_id": book_pair_id, "si_cursor": 0, "ni_cursor": 0, "status": "pending"}

    def save_progress(self, book_pair_id: int, si_cursor: int, ni_cursor: int, status: str = "in_progress"):
        self.conn.execute("""
            INSERT INTO progress (book_pair_id, si_cursor, ni_cursor, status, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(book_pair_id) DO UPDATE SET
                si_cursor = excluded.si_cursor,
                ni_cursor = excluded.ni_cursor,
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
        """, (book_pair_id, si_cursor, ni_cursor, status))
        self.conn.commit()

    # ── Sentences ─────────────────────────────────────────────────────────────

    def save_translations(self, rows: list[dict]):
        """Save translated rows. Each row: {nissaya_book_id, nissaya_para_id, line_id,
           pali_sentence, sinhala_translation, confidence}"""
        self.conn.executemany("""
            INSERT INTO sentences
                (nissaya_book_id, nissaya_para_id, line_id, pali_sentence, sinhala_translation, confidence)
            VALUES (:nissaya_book_id, :nissaya_para_id, :line_id, :pali_sentence, :sinhala_translation, :confidence)
            ON CONFLICT(nissaya_book_id, nissaya_para_id, line_id) DO UPDATE SET
                sinhala_translation = excluded.sinhala_translation,
                confidence = excluded.confidence
        """, rows)
        self.conn.commit()

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, book_pair_id: Optional[int], level: str, message: str):
        self.conn.execute("""
            INSERT INTO agent_log (book_pair_id, level, message)
            VALUES (?, ?, ?)
        """, (book_pair_id, level, message))
        self.conn.commit()

    def close(self):
        self.conn.close()