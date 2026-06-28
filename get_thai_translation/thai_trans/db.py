"""
db.py — Database access layer for Thai, Nissaya, and Output databases.
"""

import sqlite3
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class ThaiDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def get_all_volumes(self) -> List[Dict]:
        """Fetch all Thai volumes from the books table."""
        cur = self.conn.execute("""
            SELECT book_id as volume_id, book_name
            FROM books
            ORDER BY CAST(book_id AS INTEGER)
        """)
        return [dict(r) for r in cur.fetchall()]

    def get_headings(self, volume_id: str) -> List[Dict]:
        """Fetch headings for a specific Thai volume (handles zero-padding)."""
        vid_z = str(volume_id).zfill(2)
        vid_n = str(volume_id)
        cur = self.conn.execute("""
            SELECT page, title
            FROM headings
            WHERE volume_id IN (?, ?)
            ORDER BY page
        """, (vid_z, vid_n))
        return [dict(r) for r in cur.fetchall()]

    def close(self):
        self.conn.close()

class NissayaDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def get_mula_attha_books(self) -> list:
        cur = self.conn.execute("""
            SELECT book_id, book_name, category, nikaya 
            FROM books 
            WHERE category IN ('Mūla', 'Aṭṭhakathā')
            ORDER BY book_id
        """)
        return [dict(r) for r in cur.fetchall()]

    def get_headings(self, book_id: str) -> list:
        """Fetch top-level headings of a Nissaya book."""
        cur = self.conn.execute("""
            SELECT para_id, level, title 
            FROM headings 
            WHERE book_id = ? AND level <= 6
            ORDER BY para_id 
        """, (book_id,))
        return [dict(r) for r in cur.fetchall()]

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
                thai_volume_id TEXT NOT NULL,
                thai_book_name TEXT,
                nissaya_book_id TEXT NOT NULL,
                category TEXT NOT NULL,
                UNIQUE(thai_volume_id, nissaya_book_id)
            );
            
            CREATE TABLE IF NOT EXISTS heading_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_pair_id INTEGER NOT NULL REFERENCES book_pairs(id),
                thai_page TEXT NOT NULL,
                thai_title TEXT,
                nissaya_para_id INTEGER NOT NULL,
                nissaya_title TEXT,
                confidence REAL,
                UNIQUE(book_pair_id, thai_page, nissaya_para_id)
            );
        """)
        self.conn.commit()

    def save_book_pairs(self, pairs: list):
        self.conn.executemany("""
            INSERT OR IGNORE INTO book_pairs
                (thai_volume_id, thai_book_name, nissaya_book_id, category)
            VALUES (:thai_volume_id, :thai_book_name, :nissaya_book_id, :category)
        """, pairs)
        self.conn.commit()

    def get_all_book_pairs(self) -> list:
        cur = self.conn.execute("""
            SELECT id, thai_volume_id, thai_book_name, nissaya_book_id, category
            FROM book_pairs
            ORDER BY thai_volume_id, category DESC
        """)
        return [dict(r) for r in cur.fetchall()]

    def has_book_pairs(self) -> bool:
        cur = self.conn.execute("SELECT COUNT(*) FROM book_pairs")
        return cur.fetchone()[0] > 0
        
    def save_heading_matches(self, matches: list):
        """Save matched headings."""
        self.conn.executemany("""
            INSERT OR IGNORE INTO heading_matches
                (book_pair_id, thai_page, thai_title, nissaya_para_id, nissaya_title, confidence)
            VALUES (:book_pair_id, :thai_page, :thai_title, :nissaya_para_id, :nissaya_title, :confidence)
        """, matches)
        self.conn.commit()
        
    def has_heading_matches(self, book_pair_id: int) -> bool:
        cur = self.conn.execute("SELECT COUNT(*) FROM heading_matches WHERE book_pair_id = ?", (book_pair_id,))
        return cur.fetchone()[0] > 0

    def close(self):
        self.conn.close()
