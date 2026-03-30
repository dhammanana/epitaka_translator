#!/usr/bin/env python3
"""
Standalone test for fetch_commentary_block() in st_data.py.

Usage:
    python test_commentary.py --db /path/to/nissaya.db --lines 101 102 103
    python test_commentary.py --db /path/to/nissaya.db --lines 101 102 103 --max-chars 3000
    python test_commentary.py --db /path/to/nissaya.db --lines 101 102 103 --max-chars 0

Options:
    --db          Path to nissaya.db  (required)
    --lines       One or more source line_ids to look up  (required)
    --max-chars   Max chars for the result block (default: 3000; 0 = unlimited)
    --show-links  Also dump the raw book_links rows before building the block
"""

import argparse
import sqlite3
import sys
import textwrap
from contextlib import contextmanager


# ── Minimal stubs so we don't need to import the whole Flask app ──────────────

@contextmanager
def _connect(path: str):
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        yield conn
    finally:
        conn.close()


def nissaya_path(params: dict) -> str:
    return params["nissaya_db"]


# ── Paste the real fetch_commentary_block here (copy from st_data.py) ─────────
# (kept in sync — this is the version under test)

def fetch_commentary_block(
    params:    dict,
    src_lines: list[tuple],   # (book_id, para_id, line_id)
    max_chars: int = 3000,
    log_info=None,
    log_warn=None,
) -> str:
    if log_info is None:
        log_info = lambda x: None
    if log_warn is None:
        log_warn = lambda x: None

    if not src_lines:
        return "(no commentary available)"

    path = nissaya_path(params)

    try:
        with _connect(path) as conn:

            # ── 0. Guard ──────────────────────────────────────────────────
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='book_links'"
            ).fetchone()
            if not has_table:
                log_warn("[DB] book_links table not found.")
                return "(no commentary available)"

            # 1. Match src_lines against book_links via temp table.
            #    line_id is only unique within a paragraph — must match
            #    on (src_book, src_para, src_line), all three columns.
            conn.execute("DROP TABLE IF EXISTS _tmp_src")
            conn.execute(
                "CREATE TEMP TABLE _tmp_src"
                " (book_id TEXT, para_id INTEGER, line_id INTEGER)"
            )
            conn.executemany("INSERT INTO _tmp_src VALUES (?,?,?)", src_lines)

            link_rows = conn.execute(
                """SELECT bl.src_book, bl.src_para, bl.src_line,
                          bl.dst_book, bl.dst_para, bl.dst_line
                   FROM   book_links bl
                   JOIN   _tmp_src s
                          ON  bl.src_book = s.book_id
                          AND bl.src_para = s.para_id
                          AND bl.src_line = s.line_id
                   ORDER  BY bl.dst_book, bl.dst_para, bl.dst_line"""
            ).fetchall()
            conn.execute("DROP TABLE IF EXISTS _tmp_src")

            if not link_rows:
                log_info(f"[DB] Commentary: no book_links for {len(src_lines)} source line(s).")
                return "(no commentary available)"

            log_info(
                f"[DB] Commentary: {len(link_rows)} link row(s) for "
                f"{len(src_lines)} source line(s)."
            )

            # Unique (dst_book, dst_para, dst_line) targets, order preserved
            seen_targets: dict[tuple, None] = {}
            for r in link_rows:
                seen_targets[(r["dst_book"], r["dst_para"], r["dst_line"])] = None
            targets = list(seen_targets.keys())
            log_info(f"[DB] Commentary: {len(targets)} unique target(s) after dedup.")


            # ── Helper: fetch rows using a temp table to avoid variable limits ──
            def _fetch_exact(triples: list[tuple[str, int, int]]) -> list[sqlite3.Row]:
                """Fetch sentences for exact (book_id, para_id, line_id) triples."""
                if not triples:
                    return []
                conn.execute("DROP TABLE IF EXISTS _tmp_targets")
                conn.execute(
                    "CREATE TEMP TABLE _tmp_targets (book_id TEXT, para_id INTEGER, line_id INTEGER)"
                )
                conn.executemany(
                    "INSERT INTO _tmp_targets VALUES (?,?,?)", triples
                )
                rows = conn.execute(
                    """SELECT s.book_id, s.para_id, s.line_id, s.pali_sentence
                       FROM   sentences s
                       JOIN   _tmp_targets t
                              ON  s.book_id = t.book_id
                              AND s.para_id = t.para_id
                              AND s.line_id = t.line_id
                       ORDER  BY s.book_id, s.para_id, s.line_id"""
                ).fetchall()
                conn.execute("DROP TABLE IF EXISTS _tmp_targets")
                return rows

            def _fetch_paragraphs(pairs: list[tuple[str, int]]) -> list[sqlite3.Row]:
                """Fetch all sentences for (book_id, para_id) pairs."""
                if not pairs:
                    return []
                conn.execute("DROP TABLE IF EXISTS _tmp_paras")
                conn.execute(
                    "CREATE TEMP TABLE _tmp_paras (book_id TEXT, para_id INTEGER)"
                )
                conn.executemany(
                    "INSERT INTO _tmp_paras VALUES (?,?)", pairs
                )
                rows = conn.execute(
                    """SELECT s.book_id, s.para_id, s.line_id, s.pali_sentence
                       FROM   sentences s
                       JOIN   _tmp_paras t
                              ON  s.book_id = t.book_id
                              AND s.para_id = t.para_id
                       ORDER  BY s.book_id, s.para_id, s.line_id"""
                ).fetchall()
                conn.execute("DROP TABLE IF EXISTS _tmp_paras")
                return rows

            # ── Helper: render rows → formatted string ────────────────────
            def _render(rows: list[sqlite3.Row], label: str) -> str:
                if not rows:
                    return ""
                sections: dict[tuple, list[str]] = {}
                for r in rows:
                    key = (r["book_id"], r["para_id"])
                    sections.setdefault(key, []).append(
                        f"  [{r['line_id']}] {r['pali_sentence'] or ''}"
                    )
                parts = [label]
                for (book, para), sent_lines in sections.items():
                    parts.append(f"\n[{book} §{para}]")
                    parts.extend(sent_lines)
                return "\n".join(parts)

            # ══ Tier 1: full paragraphs ═══════════════════════════════════
            unique_paras: dict[tuple, None] = {}
            for dst_book, dst_para, _ in targets:
                unique_paras[(dst_book, dst_para)] = None
            log_info(f"[DB] Commentary tier-1: fetching {len(unique_paras)} unique paragraph(s).")

            tier1_rows = _fetch_paragraphs(list(unique_paras.keys()))
            tier1_text = _render(tier1_rows, "[Commentary — full paragraphs]")

            if max_chars <= 0 or len(tier1_text) <= max_chars:
                log_info(
                    f"[DB] Commentary tier-1 OK: "
                    f"{len(tier1_text)} chars, {len(tier1_rows)} sentence(s)."
                )
                return tier1_text or "(no commentary available)"

            # ══ Tier 2: 3-line window (i-1, i, i+1) ══════════════════════
            log_info(
                f"[DB] Commentary tier-1 too long ({len(tier1_text)} > {max_chars}); "
                f"falling back to 3-line window."
            )
            window_set: dict[tuple, None] = {}
            for dst_book, dst_para, dst_line in targets:
                for offset in (-1, 0, 1):
                    window_set[(dst_book, dst_para, dst_line + offset)] = None
            log_info(f"[DB] Commentary tier-2: {len(window_set)} window line(s) to fetch.")

            tier2_rows = _fetch_exact(list(window_set.keys()))
            tier2_text = _render(tier2_rows, "[Commentary — 3-line window]")

            if len(tier2_text) <= max_chars:
                log_info(
                    f"[DB] Commentary tier-2 OK: "
                    f"{len(tier2_text)} chars, {len(tier2_rows)} sentence(s)."
                )
                return tier2_text or "(no commentary available)"

            # ══ Tier 3: exact linked lines only ═══════════════════════════
            log_info(
                f"[DB] Commentary tier-2 too long ({len(tier2_text)} > {max_chars}); "
                f"falling back to exact linked lines only."
            )
            log_info(f"[DB] Commentary tier-3: {len(targets)} exact target(s).")

            tier3_rows = _fetch_exact(targets)
            tier3_text = _render(tier3_rows, "[Commentary — linked lines only]")

            log_info(
                f"[DB] Commentary tier-3 final: "
                f"{len(tier3_text)} chars, {len(tier3_rows)} sentence(s)."
            )
            return tier3_text or "(no commentary available)"

    except Exception as exc:
        log_warn(f"[DB] Commentary fetch failed: {exc}")
        import traceback; traceback.print_exc()
        return "(no commentary available)"


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test fetch_commentary_block() in isolation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )
    parser.add_argument("--db",         required=True,  help="Path to nissaya.db")
    parser.add_argument("--book",        required=True,  help="Source book_id (e.g. DN1)")
    parser.add_argument("--para",        required=True,  type=int, help="Source para_id")
    parser.add_argument("--lines",       required=True,  nargs="+", type=int,
                        help="Source line_ids within that paragraph")
    parser.add_argument("--max-chars",  default=3000,   type=int,
                        help="Max chars for the result (0 = unlimited)")
    parser.add_argument("--show-links", action="store_true",
                        help="Dump raw book_links rows before building block")
    args = parser.parse_args()

    params = {"nissaya_db": args.db}

    # Simple coloured-ish logging to stdout
    def log_info(msg):  print(f"  [INFO] {msg}")
    def log_warn(msg):  print(f"  [WARN] {msg}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"DB        : {args.db}")
    print(f"book/para : {args.book} / {args.para}")
    print(f"line_ids  : {args.lines}")
    print(f"max_chars : {args.max_chars}  (0 = unlimited)")
    print(f"{'='*60}\n")

    # Optional: show raw links first
    if args.show_links:
        print("── Raw book_links rows ──────────────────────────────────")
        try:
            src = [(args.book, args.para, lid) for lid in args.lines]
            with _connect(args.db) as conn:
                conn.execute("DROP TABLE IF EXISTS _tmp_src")
                conn.execute("CREATE TEMP TABLE _tmp_src (book_id TEXT, para_id INTEGER, line_id INTEGER)")
                conn.executemany("INSERT INTO _tmp_src VALUES (?,?,?)", src)
                rows = conn.execute(
                    """SELECT bl.src_book, bl.src_para, bl.src_line,
                              bl.dst_book, bl.dst_para, bl.dst_line
                       FROM   book_links bl
                       JOIN   _tmp_src s
                              ON  bl.src_book = s.book_id
                              AND bl.src_para = s.para_id
                              AND bl.src_line = s.line_id
                       ORDER  BY bl.dst_book, bl.dst_para, bl.dst_line"""
                ).fetchall()
                conn.execute("DROP TABLE IF EXISTS _tmp_src")
            for r in rows:
                print(f"  {r['src_book']} §{r['src_para']} line {r['src_line']}  ->  "
                      f"{r['dst_book']} §{r['dst_para']} line {r['dst_line']}")
            print(f"\n  Total: {len(rows)} link row(s)\n")
        except Exception as e:
            print(f"  ERROR reading book_links: {e}", file=sys.stderr)

    # Run the function
    src_lines = [(args.book, args.para, lid) for lid in args.lines]
    result = fetch_commentary_block(
        params    = params,
        src_lines = src_lines,
        max_chars = args.max_chars,
        log_info  = log_info,
        log_warn  = log_warn,
    )

    print(f"\n{'='*60}")
    print("── Result ───────────────────────────────────────────────────")
    print(result)
    print(f"\n── Stats ────────────────────────────────────────────────────")
    print(f"  Total chars : {len(result)}")
    print(f"  Total lines : {result.count(chr(10))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()