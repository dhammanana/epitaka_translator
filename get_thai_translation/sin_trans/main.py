"""
main.py — Main entrypoint for the Sinhala translation agent.

Usage:
    # Step 1 — pair books (run once)
    python main.py --step pair

    # Step 2 — translate (runs for days, safe to Ctrl-C and resume)
    python main.py --step translate

    # Translate a single book pair by id (for testing)
    python main.py --step translate --pair-id 5

    # Start from a specific book pair (skip already-done ones)
    python main.py --step translate --from-pair 10
"""

import argparse
import logging
import os
import sys
import signal
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import SinhalaDB, NissayaDB, OutputDB
from agent import build_graph, make_llm
from pair_books import run_pairing

# ── Logging setup ──────────────────────────────────────────────────────────────
def setup_logging(log_file: str = "agent.log"):
    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    # suppress noisy third-party loggers
    for name in ("httpx", "httpcore", "google", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── Graceful shutdown ──────────────────────────────────────────────────────────
_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    logger.info("Shutdown signal received — will stop after current window.")
    _shutdown = True

# signal.signal(signal.SIGINT,  _handle_signal)
# signal.signal(signal.SIGTERM, _handle_signal)


# ── Translation orchestrator ───────────────────────────────────────────────────

def run_translation(sinhala_path: str, nissaya_path: str, output_path: str,
                    model: str, pair_id: int = None, from_pair: int = None):

    si_db  = SinhalaDB(sinhala_path)
    ni_db  = NissayaDB(nissaya_path)
    out_db = OutputDB(output_path)

    llm   = make_llm(model)
    graph = build_graph(si_db, ni_db, out_db, llm)

    book_pairs = out_db.get_book_pairs()
    if not book_pairs:
        logger.error("No book pairs found. Run: python main.py --step pair")
        return

    # Filter if specific pair requested
    if pair_id is not None:
        book_pairs = [bp for bp in book_pairs if bp.id == pair_id]
        if not book_pairs:
            logger.error(f"No book pair with id={pair_id}")
            return

    if from_pair is not None:
        book_pairs = [bp for bp in book_pairs if bp.id >= from_pair]

    logger.info(f"Starting translation for {len(book_pairs)} book pair(s).")

    for bp in book_pairs:
        if _shutdown:
            logger.info("Shutdown requested — stopping.")
            break

        prog = out_db.get_progress(bp.id)
        if prog.get("status") == "exhausted":
            logger.info(f"[{bp.id}] {bp.sinhala_filename}↔{bp.nissaya_book_id} — already done, skipping.")
            continue

        si_total = si_db.get_total_entries(bp.sinhala_filename)
        ni_total = ni_db.get_total_sentences(bp.nissaya_book_id)

        if si_total == 0 or ni_total == 0:
            logger.warning(
                f"[{bp.id}] {bp.sinhala_filename}↔{bp.nissaya_book_id} — "
                f"empty source (si={si_total} ni={ni_total}), skipping."
            )
            out_db.save_progress(bp.id, 0, 0, "exhausted")
            continue

        logger.info(
            f"\n{'='*70}\n"
            f"[{bp.id}] {bp.sinhala_filename} ↔ {bp.nissaya_book_id}\n"
            f"       sinhala entries={si_total}  nissaya sentences={ni_total}\n"
            f"       resuming from si={prog['si_cursor']} ni={prog['ni_cursor']}\n"
            f"{'='*70}"
        )

        initial_state = {
            "book_pair":   bp,
            "si_total":    si_total,
            "ni_total":    ni_total,
            "si_cursor":   prog["si_cursor"],
            "ni_cursor":   prog["ni_cursor"],
            "si_entries":  [],
            "ni_sentences": [],
            "tool_rounds": 0,
            "status":      "running",
            "error":       None,
            "pending_translations": [],
        }

        try:
            start = time.time()
            graph.invoke(initial_state)
            elapsed = time.time() - start
            logger.info(
                f"[{bp.id}] {bp.sinhala_filename}↔{bp.nissaya_book_id} — "
                f"finished in {elapsed/60:.1f} min"
            )
        except Exception as e:
            logger.exception(f"[{bp.id}] Unhandled error: {e}")
            out_db.log(bp.id, "ERROR", f"Unhandled exception: {e}")
            # Save progress so we can resume
            out_db.save_progress(bp.id,
                                  prog["si_cursor"],
                                  prog["ni_cursor"],
                                  "error")

        if _shutdown:
            logger.info("Stopping after completing current book pair.")
            break

    si_db.close(); ni_db.close(); out_db.close()
    logger.info("Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sinhala ↔ Nissaya translation agent"
    )
    parser.add_argument("--step",     choices=["pair", "translate", "status"],
                        default="translate")
    parser.add_argument("--sinhala",  default="../data/sinhala.db",    help="Path to sinhala.db")
    parser.add_argument("--nissaya",  default="../data/nissaya.db",    help="Path to nissaya.db")
    parser.add_argument("--output",   default="../data/epitaka_si.db", help="Path to output DB")
    parser.add_argument("--model",    default="gemini-3.1-flash-lite")
    parser.add_argument("--pair-id",  type=int, default=None,
                        help="Translate only this book pair id")
    parser.add_argument("--from-pair", type=int, default=None,
                        help="Start translating from this pair id onwards")
    parser.add_argument("--log",      default="agent.log")
    args = parser.parse_args()

    setup_logging(args.log)

    if args.step == "pair":
        run_pairing(args.sinhala, args.nissaya, args.output, args.model)

    elif args.step == "translate":
        run_translation(
            args.sinhala, args.nissaya, args.output, args.model,
            pair_id=args.pair_id, from_pair=args.from_pair
        )

    elif args.step == "status":
        out_db = OutputDB(args.output)
        pairs  = out_db.get_book_pairs()
        print(f"\n{'ID':>4}  {'Sinhala filename':<30} {'Nissaya book':<12} {'Status':<15} {'si':>8} {'ni':>8}")
        print("-" * 85)
        for bp in pairs:
            prog = out_db.get_progress(bp.id)
            print(
                f"{bp.id:>4}  {bp.sinhala_filename:<30} {bp.nissaya_book_id:<12} "
                f"{prog.get('status','pending'):<15} "
                f"{prog.get('si_cursor',0):>8} {prog.get('ni_cursor',0):>8}"
            )
        out_db.close()


if __name__ == "__main__":
    main()
