"""
pair_books.py — Use Gemini tool-calling to pair Thai volumes with Nissaya books.
"""

import json
import logging
import re
import os
import sys
import time

# Attempt to reuse keys.py from the Sinhala project folder
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
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from db import ThaiDB, NissayaDB, OutputDB
from tools import ALL_TOOLS, bind_nissaya_db, _invoke_tool
from prompts import PAIRING_SYSTEM, PAIRING_USER

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5
SLEEP_BETWEEN   = 2

def run_pairing(thai_path: str, nissaya_path: str, output_path: str, model: str = "gemini-2.5-flash"):
    th_db  = ThaiDB(thai_path)
    ni_db  = NissayaDB(nissaya_path)
    out_db = OutputDB(output_path)

    if out_db.has_book_pairs():
        log("INFO", "Book pairs already exist in output DB. Clear the table to re-run.")
        th_db.close(); ni_db.close(); out_db.close()
        return

    bind_nissaya_db(ni_db)

    thai_volumes  = th_db.get_all_volumes()
    nissaya_books = ni_db.get_mula_attha_books()

    log("INFO", f"Found {len(thai_volumes)} Thai Volumes and {len(nissaya_books)} Nissaya books.")

    # Format Nissaya candidate list once
    ni_list_str = "\n".join(
        f"[{b['category']}] ID: {b['book_id']} | Name: {b['book_name']} | Nikaya: {b['nikaya']}"
        for b in nissaya_books
    )

    llm = ChatGoogleGenerativeAI(model=model, temperature=0).bind_tools(ALL_TOOLS)

    all_pairs = []

    for vol in thai_volumes:
        vol_id   = str(vol['volume_id'])
        vol_name = vol['book_name']
        
        # Format Thai headings for context
        headings = th_db.get_headings(vol_id)
        headings_str = "\n".join(f"Pg {h['page']}: {h['title']}" for h in headings[:80])
        if not headings_str:
            headings_str = "(No headings found for this volume)"

        log("TURN", f"Evaluating Thai Vol {vol_id}: {vol_name} ...")

        user_msg = PAIRING_USER.format(
            vol_id=vol_id,
            vol_name=vol_name,
            thai_headings=headings_str,
            nissaya_list=ni_list_str
        )

        messages = [
            SystemMessage(content=PAIRING_SYSTEM),
            HumanMessage(content=user_msg),
        ]

        # --- Tool Calling Loop ---
        tool_rounds = 0
        final_response = None

        while tool_rounds < MAX_TOOL_ROUNDS:
            try:
                # Assuming call_with_rotation exists from keys.py; if not, use llm.invoke directly.
                if 'call_with_rotation' in globals():
                    response = call_with_rotation(lambda: llm.invoke(messages), tag="pairing")
                else:
                    response = llm.invoke(messages)
            except Exception as e:
                log("ERROR", f"LLM error: {e}")
                break

            if not response.tool_calls:
                final_response = response
                break

            tool_rounds += 1
            messages.append(response)
            
            for tc in response.tool_calls:
                name = tc["name"]
                args = tc["args"]
                log("TOOL_CALL", f"{Colors.BRIGHT_YELLOW}{name}{Colors.RESET}({json.dumps(args)})")

                result_str = _invoke_tool(name, args)
                log("TOOL_RESULT", f"{Colors.GREEN}{name}{Colors.RESET} -> fetched {len(result_str)} chars")
                messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))
            
            time.sleep(SLEEP_BETWEEN)

        if not final_response:
            log("WARN", f"Failed to get final response for Vol {vol_id}")
            continue

        raw = _extract_text(final_response)
        pairs = _parse_pairs(raw)

        if not pairs:
            log("WARN", f"No valid JSON pairs extracted for Vol {vol_id}. Raw output:\n{raw[:300]}")
            continue

        # Force injecting the thai metadata just to be safe
        for p in pairs:
            p['thai_volume_id'] = vol_id
            p['thai_book_name'] = vol_name

        all_pairs.extend(pairs)
        log("SAVED", f"Mapped Vol {vol_id} to: " + ", ".join([p.get('nissaya_book_id', '?') for p in pairs]))

        # Save as we go to preserve progress if script crashes
        out_db.save_book_pairs(pairs)

    th_db.close(); ni_db.close(); out_db.close()
    log("DONE", f"Pairing finished! Total pairs generated: {len(all_pairs)}")


def _extract_text(response) -> str:
    if hasattr(response, "content"):
        c = response.content
        if isinstance(c, str): return c
        if isinstance(c, list):
            return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(response)


def _parse_pairs(raw: str) -> list:
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
    parser = argparse.ArgumentParser(description="Pair Thai volumes with Nissaya Mūla/Aṭṭhakathā books")
    parser.add_argument("--thai",     default="../data/thaimm.sqlite")
    parser.add_argument("--nissaya",  default="../data/nissaya.db")
    parser.add_argument("--output",   default="../data/thaitrans_pairs.db")
    parser.add_argument("--model",    default="gemini-2.0-flash-preview")
    args = parser.parse_args()

    run_pairing(args.thai, args.nissaya, args.output, args.model)
