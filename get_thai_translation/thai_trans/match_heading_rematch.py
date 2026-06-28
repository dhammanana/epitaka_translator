"""
rematch_missing_headings.py — Re-match Nissaya headings that lack a Thai page mapping.

DB roles:
  thaimm.sqlite  — Thai headings (headings table) + book_pairs + heading_matches
  nissaya.db     — Pāli source: books, headings, sentences
  epitaka_th.db  — Translation output: sentences table only (book_id, para_id, line_id, translation)

Strategy:
1. For each book_pair in thaimm.db, fetch all Nissaya headings (nissaya.db).
2. Query epitaka_th.db sentences grouped by (book_id, para_id) — flag para_ids with
   more than `threshold` untranslated rows.
3. Cross-check flagged para_ids against heading_matches in thaimm.db.
4. Send unmatched Nissaya headings + full Thai heading list to LLM for rematch.
5. Save new matches into thaimm.db heading_matches with INSERT OR IGNORE.

Debug files (overwritten on each LLM call):
  debug_input.txt  — exact prompt sent
  debug_output.txt — raw LLM response
"""

import json
import logging
import re
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sin_trans")))
try:
    from keys import current_client, rotate_key, call_with_rotation, log, Colors
except ImportError:
    class Colors:
        BRIGHT_YELLOW = ""
        GREEN = ""
        RESET = ""
    def log(tag, msg):
        print(f"[{tag}] {msg}")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from db import ThaiDB, NissayaDB, OutputDB
from prompts import HEADING_MATCH_SYSTEM, HEADING_MATCH_USER

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SLEEP_BETWEEN = 2
UNTRANS_THRESHOLD = 5
DEBUG_INPUT_FILE  = "debug_input.txt"
DEBUG_OUTPUT_FILE = "debug_output.txt"


# ─── helpers ────────────────────────────────────────────────────────────────

def _extract_text(response) -> str:
    if hasattr(response, "content"):
        c = response.content
        if isinstance(c, str): return c
        if isinstance(c, list):
            return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(response)


def _parse_matches(raw: str) -> list:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list): return data
    except Exception:
        pass
    m = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list): return data
        except Exception:
            pass
    return []


def _write_debug(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ─── DB queries ──────────────────────────────────────────────────────────────

def get_already_matched_para_ids(thai_conn, book_pair_id: int) -> set:
    """
    thaimm.db: return all nissaya_para_ids already in heading_matches for this pair.
    """
    cur = thai_conn.execute("""
        SELECT nissaya_para_id FROM heading_matches WHERE book_pair_id = ?
    """, (book_pair_id,))
    return {row[0] for row in cur.fetchall()}


# ─── main ────────────────────────────────────────────────────────────────────

def run_rematch(
    thai_path: str,       # thaimm.sqlite — Thai headings + book_pairs + heading_matches
    nissaya_path: str,    # nissaya.db    — Pāli books/headings
    model: str = "gemini-2.0-flash-preview",
):
    th_db   = ThaiDB(thai_path)
    ni_db   = NissayaDB(nissaya_path)
    out_db  = OutputDB(thai_path)

    book_pairs = out_db.get_all_book_pairs()
    if not book_pairs:
        log("ERROR", "No book pairs found in thaimm.db. Run pair_books.py first.")
        return

    log("INFO", f"Checking {len(book_pairs)} book pairs for unmatched headings.")

    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    total_new = 0

    for pair in book_pairs:
        pair_id     = pair["id"]
        thai_vol_id = pair["thai_volume_id"]
        ni_book_id  = pair["nissaya_book_id"]
        category    = pair["category"]

        log("CHECK", f"Pair {pair_id}: Vol {thai_vol_id} <-> {ni_book_id} ({category})")

        # Step 1: all Nissaya headings for this book
        all_ni_headings = ni_db.get_headings(ni_book_id)
        if not all_ni_headings:
            log("SKIP", f"  No Nissaya headings found for book {ni_book_id}")
            continue
        log("INFO", f"  Nissaya headings total: {len(all_ni_headings)}")

        # Step 2: para_ids already matched (thaimm.db heading_matches)
        already_matched = get_already_matched_para_ids(th_db.conn, pair_id)
        log("INFO", f"  Already matched para_ids: {len(already_matched)}")

        # Step 3: any nissaya heading whose para_id is not yet in heading_matches
        unmatched_headings = [
            h for h in all_ni_headings
            if h["para_id"] not in already_matched
        ]
        log("INFO", f"  Unmatched headings: {len(unmatched_headings)}")

        if not unmatched_headings:
            log("OK", f"  No unmatched headings to rematch.")
            continue

        log("REMATCH", f"  {len(unmatched_headings)} unmatched heading(s):")

        # Step 4: full Thai headings for this volume (thaimm.db)
        th_headings = th_db.get_headings(thai_vol_id)
        if not th_headings:
            log("WARN", f"  No Thai headings for vol {thai_vol_id}. Skipping.")
            continue
        log("INFO", f"  Thai headings: {len(th_headings)}")

        th_headings_str = "\n".join(f"Page {h['page']}: {h['title']}" for h in th_headings)

        # Split into chunks of 30 if there are more than 30 unmatched headings
        CHUNK_SIZE = 30
        if len(unmatched_headings) > CHUNK_SIZE:
            chunks = [
                unmatched_headings[i:i + CHUNK_SIZE]
                for i in range(0, len(unmatched_headings), CHUNK_SIZE)
            ]
            log("CHUNK", f"  Splitting into {len(chunks)} chunks of up to {CHUNK_SIZE} headings each.")
        else:
            chunks = [unmatched_headings]

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunks) > 1:
                log("CHUNK", f"  Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} headings)")

            ni_headings_str = "\n".join(
                f"Para {h['para_id']} (Lvl {h['level']}): {h['title']}" for h in chunk
            )

            # Step 5: build prompt (same templates as match_headings.py)
            system_msg = HEADING_MATCH_SYSTEM.format(category=category)
            user_msg   = HEADING_MATCH_USER.format(
                nissaya_book_id=ni_book_id,
                category=category,
                thai_volume_id=thai_vol_id,
                thai_book_name=pair["thai_book_name"],
                nissaya_headings=ni_headings_str,
                thai_headings=th_headings_str,
            )

            debug_suffix = f"_chunk{chunk_idx + 1}" if len(chunks) > 1 else ""
            debug_input  = DEBUG_INPUT_FILE.replace(".txt", f"{debug_suffix}.txt")
            debug_output = DEBUG_OUTPUT_FILE.replace(".txt", f"{debug_suffix}.txt")

            debug_prompt = f"=== SYSTEM ===\n{system_msg}\n\n=== USER ===\n{user_msg}\n"
            _write_debug(debug_input, debug_prompt)

            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ]

            # Step 6: call LLM
            try:
                if 'call_with_rotation' in globals():
                    response = call_with_rotation(lambda: llm.invoke(messages), tag="rematch")
                else:
                    response = llm.invoke(messages)
            except Exception as e:
                log("ERROR", f"  LLM error on chunk {chunk_idx + 1}: {e}")
                _write_debug(debug_output, f"ERROR: {e}")
                continue

            raw = _extract_text(response)
            _write_debug(debug_output, raw)

            # Step 7: parse + inject pair_id + save into thaimm.db
            matches = _parse_matches(raw)
            if not matches:
                log("WARN", f"  No valid JSON matches extracted. Check {debug_output}.")
                continue

            for m in matches:
                m["book_pair_id"] = pair_id

            out_db.save_heading_matches(matches)
            total_new += len(matches)
            log("SAVED", f"  +{len(matches)} new matches saved to thaimm.db for pair {pair_id}")

            if chunk_idx < len(chunks) - 1:
                time.sleep(SLEEP_BETWEEN)

        time.sleep(SLEEP_BETWEEN)
    th_db.close(); ni_db.close(); out_db.close()
    log("DONE", f"Rematch finished. Total new matches saved: {total_new}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-match missing Nissaya headings via untranslated sentence detection")
    parser.add_argument("--thai",      default="../data/thaimm.sqlite")
    parser.add_argument("--nissaya",   default="../data/nissaya.db")
    parser.add_argument("--model",     default="gemini-2.0-flash-preview")
    parser.add_argument("--threshold", type=int, default=5,
                        help="Min unmatched headings per pair before skipping (default: 5)")
    args = parser.parse_args()

    run_rematch(args.thai, args.nissaya, args.model)