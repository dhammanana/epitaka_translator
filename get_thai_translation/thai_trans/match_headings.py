"""
match_headings.py — Link structural headings between Thai and Nissaya books.
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
    print("Warning: Could not import keys.py from ../sin_trans. Running with standard printing.")
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

def run_heading_matching(thai_path: str, nissaya_path: str, output_path: str, model: str = "gemini-2.5-flash"):
    th_db  = ThaiDB(thai_path)
    ni_db  = NissayaDB(nissaya_path)
    out_db = OutputDB(output_path)

    book_pairs = out_db.get_all_book_pairs()
    if not book_pairs:
        log("ERROR", "No book pairs found. Run pair_books.py first.")
        return

    log("INFO", f"Found {len(book_pairs)} paired books to map headings for.")

    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    for pair in book_pairs:
        pair_id = pair["id"]
        thai_vol_id = pair["thai_volume_id"]
        ni_book_id = pair["nissaya_book_id"]
        category = pair["category"]

        if out_db.has_heading_matches(pair_id):
            log("SKIP", f"Headings already mapped for {thai_vol_id} <-> {ni_book_id}")
            continue

        log("TURN", f"Matching Headings: Vol {thai_vol_id} <-> {ni_book_id} ({category})")

        # Fetch headings
        th_headings = th_db.get_headings(thai_vol_id)
        ni_headings = ni_db.get_headings(ni_book_id)

        if not th_headings or not ni_headings:
            log("WARN", f"Missing headings for either Thai or Nissaya. Skipping.")
            continue

        # Format strings for prompt
        th_headings_str = "\n".join(f"Page {h['page']}: {h['title']}" for h in th_headings)
        ni_headings_str = "\n".join(f"Para {h['para_id']} (Lvl {h['level']}): {h['title']}" for h in ni_headings)

        system_msg = HEADING_MATCH_SYSTEM.format(category=category)
        user_msg = HEADING_MATCH_USER.format(
            nissaya_book_id=ni_book_id,
            category=category,
            thai_volume_id=thai_vol_id,
            thai_book_name=pair["thai_book_name"],
            nissaya_headings=ni_headings_str,
            thai_headings=th_headings_str
        )

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ]

        try:
            if 'call_with_rotation' in globals():
                response = call_with_rotation(lambda: llm.invoke(messages), tag="headings")
            else:
                response = llm.invoke(messages)
        except Exception as e:
            log("ERROR", f"LLM error: {e}")
            continue

        raw = _extract_text(response)
        matches = _parse_matches(raw)

        if not matches:
            log("WARN", f"No valid JSON matches extracted. Raw output:\n{raw[:300]}")
            continue

        # Inject the pair_id into the dictionaries before saving
        for m in matches:
            m["book_pair_id"] = pair_id

        out_db.save_heading_matches(matches)
        log("SAVED", f"Mapped {len(matches)} headings for {thai_vol_id} <-> {ni_book_id}")

        time.sleep(SLEEP_BETWEEN)

    th_db.close(); ni_db.close(); out_db.close()
    log("DONE", "Heading mapping finished!")


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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map structural headings between Thai and Nissaya")
    parser.add_argument("--thai",     default="../data/thaimm.sqlite")
    parser.add_argument("--nissaya",  default="../data/nissaya.db")
    parser.add_argument("--output",   default="../data/epitaka_th.db")
    parser.add_argument("--model",    default="gemini-2.0-flash-preview") # 1M context handles long heading lists easily
    args = parser.parse_args()


    run_heading_matching(args.thai, args.nissaya, args.output, args.model)
