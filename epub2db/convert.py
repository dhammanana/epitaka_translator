#!/usr/bin/env python3
"""
Parse Visuddhimagga xhtml chapters → SQLite.

Usage:
    python vism_to_sqlite.py --vism-dir /path/to/vism --db vism.db

Table: sections
  id           INTEGER PRIMARY KEY
  chapter      TEXT    (e.g. "ch01")
  chapter_title TEXT   (h1 text)
  heading      TEXT    (h2/h3 text, NULL for intro before first h2)
  heading_level INTEGER (1, 2, or 3)
  content      TEXT    (clean text of paragraphs under this heading)
"""

import argparse
import re
import sqlite3
from pathlib import Path

import warnings
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def clean_text(element) -> str:
    """Return plain text from a BS4 element, collapsing whitespace."""
    text = element.get_text(separator="")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def heading_text(tag) -> str:
    """Clean text of a heading tag (strip footnote refs, etc.)."""
    # Remove <sup> footnote markers
    for sup in tag.find_all("sup"):
        sup.decompose()
    return clean_text(tag)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_chapter(path: Path) -> list[dict]:
    """Return list of section dicts for one chapter file."""
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("body")
    if body is None:
        return []

    chapter_name = path.stem  # e.g. "ch01"

    # Get h1 title
    h1 = body.find("h1")
    chapter_title = heading_text(h1) if h1 else chapter_name

    sections = []
    current_heading = None
    current_level = None
    buf: list[str] = []

    def flush():
        if current_heading is not None or buf:
            sections.append({
                "chapter": chapter_name,
                "chapter_title": chapter_title,
                "heading": current_heading,
                "heading_level": current_level,
                "content": " ".join(buf).strip(),
            })

    for tag in body.children:
        if not isinstance(tag, Tag):
            continue
        name = tag.name
        if name == "h1":
            continue  # already captured
        if name in ("h2", "h3"):
            flush()
            buf = []
            current_heading = heading_text(tag)
            current_level = int(name[1])
        else:
            text = clean_text(tag)
            if text:
                buf.append(text)

    flush()
    return sections


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter       TEXT,
            chapter_title TEXT,
            heading       TEXT,
            heading_level INTEGER,
            content       TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chapter ON sections(chapter)")
    conn.commit()
    return conn


def insert_sections(conn: sqlite3.Connection, sections: list[dict]):
    conn.executemany("""
        INSERT INTO sections (chapter, chapter_title, heading, heading_level, content)
        VALUES (:chapter, :chapter_title, :heading, :heading_level, :content)
    """, sections)
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vism-dir", required=True, help="Folder with ch*.xhtml files")
    parser.add_argument("--db", default="vism.db", help="Output SQLite file")
    args = parser.parse_args()

    vism_dir = Path(args.vism_dir)
    files = sorted(vism_dir.glob("ch*.xhtml"))
    if not files:
        print(f"No ch*.xhtml files found in {vism_dir}")
        return

    conn = init_db(args.db)
    # Clear existing data
    conn.execute("DELETE FROM sections")
    conn.commit()

    total = 0
    for f in files:
        sections = parse_chapter(f)
        insert_sections(conn, sections)
        print(f"  {f.name}: {len(sections)} sections")
        total += len(sections)

    conn.close()
    print(f"\nDone. {total} sections from {len(files)} chapters → {args.db}")


if __name__ == "__main__":
    main()