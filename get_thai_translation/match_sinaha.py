#!/usr/bin/env python3
"""
Match books and headings between sinhala.db and nissaya.db using Gemini API.
Uses key rotation and exponential backoff for quota resilience.
Tracks processed books to skip them on subsequent runs.
"""

import sqlite3
import json
from pathlib import Path
import os
import sys
import re
import time
from itertools import cycle
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    print("ERROR: google-genai not installed. Install with: pip install google-genai")
    sys.exit(1)

load_dotenv()


def _load_gemini_keys():
    """Load all GEMINI_KEY_N from environment."""
    keys = {}
    for k, v in os.environ.items():
        m = re.match(r"GEMINI_KEY_(\d+)$", k)
        if m and v.strip():
            keys[int(m.group(1))] = v.strip()
    if not keys:
        raise RuntimeError("No GEMINI_KEY_[N] found in environment / .env")
    ordered = [keys[i] for i in sorted(keys)]
    print(f"✓ Loaded {len(ordered)} Gemini key(s): " +
          ", ".join(f"GEMINI_KEY_{i+1}" for i in range(len(ordered))))
    import random, time
    random.seed(time.time())
    random.shuffle(ordered)
    return ordered


_gemini_keys = _load_gemini_keys()
_key_cycle = cycle(_gemini_keys)
_key_index = 0
_clients = [genai.Client(api_key=k) for k in _gemini_keys]


def _current_client():
    return _clients[_key_index]


def _rotate_key(reason=""):
    """Rotate to next Gemini API key."""
    global _key_index
    _key_index = (_key_index + 1) % len(_gemini_keys)
    next(_key_cycle)
    print(f"🔑 Rotated to GEMINI_KEY_{_key_index+1}" +
          (f" — {reason}" if reason else ""))


def _call_gemini_with_rotation(prompt: str, tag: str = "MATCH") -> str:
    """Call Gemini with exponential backoff and key rotation on quota errors."""
    attempt = 0
    max_attempts = len(_gemini_keys) * 2
    backoff_base = 2

    while attempt < max_attempts:
        try:
            client = _current_client()
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )
            return response.text

        except Exception as err:
            err_str = str(err)
            attempt += 1

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                wait_time = min(backoff_base ** attempt, 120)
                log(tag, f"Quota hit, sleeping {wait_time}s (attempt {attempt}/{max_attempts})")
                time.sleep(wait_time)
                _rotate_key("quota exceeded")
                continue
            if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower():
                wait_time = min(backoff_base ** attempt, 120)
                log(tag, f"High demand hit, sleeping {wait_time}s (attempt {attempt}/{max_attempts})")
                time.sleep(wait_time)
                _rotate_key("High demand exceeded")
                continue

            if "401" in err_str or "UNAUTHENTICATED" in err_str:
                log(tag, f"Auth failed on current key, rotating...")
                _rotate_key("auth failed")
                continue

            raise

    raise RuntimeError(f"{tag}: exhausted all {max_attempts} attempts")


def log(tag: str, msg: str):
    """Simple logging with emoji tag."""
    tags = {
        "BOOKS": "📚",
        "MATCH": "🤖",
        "HEADINGS": "📑",
        "INSERT": "💾",
        "ERROR": "❌",
        "SUCCESS": "✅",
        "SKIP": "⏭️",
    }
    emoji = tags.get(tag, "•")
    print(f"{emoji} {tag.ljust(10)} {msg}")


class DatabaseMatcher:
    def __init__(self, sinhala_db: str, nissaya_db: str):
        """Initialize with database paths."""
        self.sinhala_db = sinhala_db
        self.nissaya_db = nissaya_db
        self.model = "gemini-3-flash-preview"

    def get_sinhala_books(self) -> list[dict]:
        """Get distinct books from sinhala.db."""
        conn = sqlite3.connect(self.sinhala_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT node_id, pali_roman 
            FROM tree_nodes t 
            INNER JOIN entries e ON e.filename = t.node_id
        """
        cursor.execute(query)
        books = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return books

    def get_nissaya_books(self) -> list[dict]:
        """Get books from nissaya.db."""
        conn = sqlite3.connect(self.nissaya_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT book_id, book_name, category FROM books"
        cursor.execute(query)
        books = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return books

    def match_book_names_with_gemini(self, sinhala_books: list[dict], nissaya_books: list[dict]) -> dict:
        """Use Gemini to match book names between databases."""
        sinhala_names = [(b["node_id"], b["pali_roman"]) for b in sinhala_books]
        nissaya_names = [(b["book_id"], b["book_name"], b["category"]) for b in nissaya_books]

        prompt = f"""
You are an expert in Pāli Buddhist texts. Match the following Tipitaka BJT book names with Tipitaka CST book names.

Sinhala Books (node_id, pali_roman):
{json.dumps(sinhala_names, ensure_ascii=False, indent=2)}

Nissaya Books (book_id, book_name, category):
{json.dumps(nissaya_names, ensure_ascii=False, indent=2)}

For each Tipitaka BJT book, find the best matching Tipitaka CST book(s). Consider:
1. Similar text names (accounting for spelling variations)
2. For Mūla texts, match with Mūla entries in nissaya.db
3. For Aṭṭhakathā/Ṭīkā, match with corresponding commentary entries
4. Some Sinhala books may not have matches - exclude them

Return a JSON object where:
- Keys are sinhala node_id
- Values are objects with: {{"book_id": nissaya_book_id, "book_name": nissaya_name, "category": nissaya_category, "confidence": score}}

Example format:
{{
  "node_id_1": {{"book_id": "b1", "book_name": "...", "category": "Mūla", "confidence": 0.95}},
  "node_id_2": null
}}

Return ONLY valid JSON, no other text.
"""

        response_text = _call_gemini_with_rotation(prompt, tag="MATCH_BOOKS")

        try:
            resp_text = response_text.replace('```json', '').replace('```', '').strip()
            matches = json.loads(resp_text)
            return matches
        except json.JSONDecodeError:
            log("ERROR", f"Gemini response was not valid JSON:\n{response_text}")
            raise

    def get_sinhala_headings(self, filename: str) -> list[dict]:
        """Get headings from sinhala.db for a specific filename."""
        conn = sqlite3.connect(self.sinhala_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
            SELECT filename, page_num, entry_order, palitext 
            FROM entries 
            WHERE filename = ? AND type = 'heading'
            ORDER BY page_num, entry_order
        """
        cursor.execute(query, (filename,))
        headings = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return headings

    def get_nissaya_headings(self, book_id: str) -> list[dict]:
        """Get headings from nissaya.db for a specific book."""
        conn = sqlite3.connect(self.nissaya_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
            SELECT book_id, para_id, level, title 
            FROM headings 
            WHERE book_id = ? 
            ORDER BY para_id
        """
        cursor.execute(query, (book_id,))
        headings = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return headings

    def match_headings_with_gemini(self, sinhala_headings: list[dict], nissaya_headings: list[dict], bookname: str) -> dict:
        """Use Gemini to match headings between sinhala and nissaya versions."""
        if not sinhala_headings or not nissaya_headings:
            return {}

        prompt = f"""
You are an expert in Pāli Buddhist texts. Match the headings of Tipitaka BJT Text with Tipitaka CST headings.
From book: {bookname}
BJT Headings (page_num, para_num, palitext):
{json.dumps([(h["page_num"], h["entry_order"], h["palitext"]) for h in sinhala_headings], ensure_ascii=False, indent=2)}

CST Headings (para_id, level, title):
{json.dumps([(h["para_id"], h["level"], h["title"]) for h in nissaya_headings], ensure_ascii=False, indent=2)}

Match each Sinhala heading with the best corresponding Nissaya heading based on:
1. Similar content/meaning
2. Hierarchical level (level field)
3. Sequential order

Return a JSON object where:
- Key is para_id from CST headings.
- Values are {{"page_num": page_num, "para_num": para_num, "confidence": score}}

If no good match exists for a heading, omit it from results.

Return ONLY valid JSON, no other text.
"""
        with open('test.txt', 'wt') as f:
            f.write(prompt)
            f.write('\n' + '-' * 50 + '\n')

        response = _call_gemini_with_rotation(prompt, tag="MATCH_HEADINGS")

        with open('output.txt', 'wt') as f:
            f.write(prompt)
            f.write('\n' + '-' * 50 + '\n')
            f.write(response)
            f.write('\n' + '=' * 50 + '\n\n')

        try:
            resp = response.replace('```json', '').replace('```', '').strip()
            matches = json.loads(resp)
            return matches
        except json.JSONDecodeError:
            print(f"ERROR: Gemini response was not valid JSON:\n{response}")
            return {}

    def create_sinhala_match_table(self):
        """Create the sinhala_match table in nissaya.db."""
        conn = sqlite3.connect(self.nissaya_db)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sinhala_match (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sinhala_filename TEXT NOT NULL,
                sinhala_page_num INTEGER,
                sinhala_entry_order INTEGER,
                sinhala_palitext TEXT,
                nissaya_book_id TEXT,
                nissaya_para_id INTEGER,
                nissaya_title TEXT,
                match_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def book_id_already_processed(self, nissaya_book_id: str) -> bool:
        """Check if nissaya_book_id already has entries in sinhala_match table."""
        conn = sqlite3.connect(self.nissaya_db)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT 1 FROM sinhala_match WHERE nissaya_book_id = ? LIMIT 1",
            (nissaya_book_id,)
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def insert_sinhala_match(self, sinhala_data: dict, nissaya_data: dict, confidence: float):
        """Insert a match into the sinhala_match table."""
        conn = sqlite3.connect(self.nissaya_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sinhala_match 
            (sinhala_filename, sinhala_page_num, sinhala_entry_order, sinhala_palitext,
             nissaya_book_id, nissaya_para_id, nissaya_title, match_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sinhala_data.get("filename"),
            sinhala_data.get("page_num"),
            sinhala_data.get("entry_order"),
            sinhala_data.get("palitext"),
            nissaya_data.get("book_id"),
            nissaya_data.get("para_id"),
            nissaya_data.get("title"),
            confidence
        ))
        conn.commit()
        conn.close()

    def run(self, output_json: str = "book_matches.json"):
        """Run the full matching pipeline."""
        if not os.path.exists(output_json):
            print("📚 Getting books from sinhala.db...")
            sinhala_books = self.get_sinhala_books()
            print(f"   Found {len(sinhala_books)} sinhala books")

            print("📚 Getting books from nissaya.db...")
            nissaya_books = self.get_nissaya_books()
            print(f"   Found {len(nissaya_books)} nissaya books")

            print("🤖 Matching book names with Gemini...")
            book_matches = self.match_book_names_with_gemini(sinhala_books, nissaya_books)
            print(f"   Matched {sum(1 for v in book_matches.values() if v)} books")

            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(book_matches, f, ensure_ascii=False, indent=2)
            print(f"✅ Book matches saved to {output_json}")
        else:
            book_matches = json.load(open(output_json))

        # Create sinhala_match table
        print("📊 Creating sinhala_match table in nissaya.db...")
        self.create_sinhala_match_table()

        # Match headings for each book pair
        print("🔗 Matching headings...")
        matched_count = 0
        skipped_count = 0

        for sinhala_node_id, match_data in book_matches.items():
            if not match_data:
                continue

            nissaya_book_id = match_data["book_id"]
            
            # Skip if nissaya_book_id already processed
            if self.book_id_already_processed(nissaya_book_id):
                log("SKIP", f"{nissaya_book_id} already processed")
                skipped_count += 1
                continue

            confidence = match_data.get("confidence", 0.5)

            print(f"   Matching headings for {sinhala_node_id} → {nissaya_book_id}")

            sinhala_headings = self.get_sinhala_headings(sinhala_node_id)
            nissaya_headings = self.get_nissaya_headings(nissaya_book_id)

            if not sinhala_headings or not nissaya_headings:
                print(f"      ⚠️  No headings found (sinhala: {len(sinhala_headings)}, nissaya: {len(nissaya_headings)})")
                continue

            heading_matches = self.match_headings_with_gemini(sinhala_headings, nissaya_headings, match_data['book_name'])

            print(f"      heading_matches count: {len(heading_matches)}")

            nissaya_lookup = {h["para_id"]: h for h in nissaya_headings}
            sinhala_lookup = {(h["page_num"], h["entry_order"]): h for h in sinhala_headings}

            for nissaya_para_id, match_info in heading_matches.items():
                if not match_info:
                    continue

                page_num = match_info["page_num"]
                para_num = match_info["para_num"]
                heading_confidence = match_info.get("confidence", 0.5)

                sinhala_heading = sinhala_lookup.get((int(page_num), int(para_num)))
                nissaya_heading = nissaya_lookup.get(int(nissaya_para_id))

                if not sinhala_heading:
                    print(f"      ⚠️  No sinhala heading for page={page_num}, para={para_num}")
                    continue
                if not nissaya_heading:
                    print(f"      ⚠️  No nissaya heading for para_id={nissaya_para_id}")
                    continue

                self.insert_sinhala_match(
                    sinhala_heading,
                    {**nissaya_heading, "book_id": nissaya_book_id},
                    heading_confidence * confidence
                )
                matched_count += 1

        print(f"✅ Inserted {matched_count} heading matches into sinhala_match table")
        if skipped_count:
            print(f"⏭️  Skipped {skipped_count} already-processed books")


def main():
    sinhala_db = "data/sinhala.db"
    nissaya_db = "data/nissaya.db"

    if not Path(sinhala_db).exists():
        print(f"ERROR: {sinhala_db} not found")
        sys.exit(1)
    if not Path(nissaya_db).exists():
        print(f"ERROR: {nissaya_db} not found")
        sys.exit(1)

    try:
        matcher = DatabaseMatcher(sinhala_db, nissaya_db)
        matcher.run()
        print("\n✅ Matching complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()