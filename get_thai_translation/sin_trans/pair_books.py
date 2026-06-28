"""
pair_books.py — One-time step: use Gemini to pair sinhala filenames ↔ nissaya book_ids.
Run this before the main agent. Safe to re-run (uses INSERT OR IGNORE).
"""

import json
import logging
import re
import os
import sys
import time
import threading
from headings import run_link_building, init_rotator, call_gemini
from headings import logger

from google import genai
from dotenv import load_dotenv

from db import SinhalaDB, NissayaDB, OutputDB
from prompts import PAIRING_SYSTEM, PAIRING_USER

load_dotenv()



BATCH_SIZE = 300

def run_pairing(sinhala_path: str, nissaya_path: str, output_path: str,
                model: str = "gemini-2.0-flash-preview"):
    init_rotator(model)

    si_db  = SinhalaDB(sinhala_path)
    ni_db  = NissayaDB(nissaya_path)
    out_db = OutputDB(output_path)

    if out_db.has_book_pairs():
        logger.info(f"Book pairs already exist — skipping. Delete rows to re-run.")
        si_db.close(); ni_db.close(); out_db.close()
        return

    sinhala_filenames = si_db.get_all_filenames_with_pali()
    nissaya_books     = ni_db.get_all_books()

    logger.info(f"Sinhala filenames: {len(sinhala_filenames)}")
    logger.info(f"Nissaya books:     {len(nissaya_books)}")

    ni_list_str = "\n".join(
        f"  {b['book_id']} | {b['book_name']} | {b['category']} | {b['nikaya']}"
        for b in nissaya_books
    )

    all_pairs = []

    for batch_start in range(0, len(sinhala_filenames), BATCH_SIZE):
        batch = sinhala_filenames[batch_start: batch_start + BATCH_SIZE]
        si_list_str = "\n".join(
            f"  {r['filename']} | pali_roman={r['pali_roman']}"
            for r in batch
        )

        logger.info(f"Pairing batch {batch_start}–{batch_start+len(batch)-1} …")

        prompt = PAIRING_SYSTEM + "\n\n" + PAIRING_USER.format(
            sinhala_list=si_list_str,
            nissaya_list=ni_list_str,
        )

        open('input.txt', 'wt').write(prompt)

        try:
            raw = call_gemini(prompt)
            open('output.txt', 'wt').write(prompt + "\n=====\n\n" + raw)
        except AllKeysExhaustedError:
            logger.critical("All API keys exhausted — aborting pairing.")
            break

        pairs = _parse_pairs(raw)
        if not pairs:
            logger.warning(f"No pairs parsed from batch {batch_start}. Raw[:400]:\n{raw[:400]}")
            continue

        logger.info(f"  Got {len(pairs)} pairs from this batch.")
        all_pairs.extend(pairs)

    if all_pairs:
        out_db.save_book_pairs(all_pairs)
        logger.info(f"Saved {len(all_pairs)} book pairs to output DB.")
    else:
        logger.error("No book pairs produced — check LLM responses above.")

    si_db.close(); ni_db.close(); out_db.close()


def _parse_pairs(raw: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    m = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pair sinhala filenames with nissaya book_ids")
    parser.add_argument("--sinhala",  default="../../data/sinhala.db")
    parser.add_argument("--nissaya",  default="../../data/epitaka.db")
    parser.add_argument("--output",   default="../../data/epitaka_si.db")
    parser.add_argument("--model",    default="gemini-2.0-flash-preview")
    parser.add_argument("--link",     action="store_true", help="Run heading link building instead of book pairing")
    args = parser.parse_args()

    if args.link:
        run_link_building(args.sinhala, args.nissaya, args.output,
                          call_gemini=call_gemini, parse_response=_parse_pairs,
                          init_rotator=init_rotator, model=args.model)
    else:
        run_pairing(args.sinhala, args.nissaya, args.output, args.model)