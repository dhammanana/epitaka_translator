#!/usr/bin/env python3
"""
cleanup_script_bleed.py — one-off repair tool for Thai/Sinhala/Myanmar/
Devanagari-script contamination that was already saved into a language's
glossary/sentences DB before the script-bleed guard fix (see
book_translator.py / common_utils.py changes).

Built to run on modest hardware (1 CPU / 1GB RAM): rows are streamed from
SQLite one at a time rather than loaded with fetchall(), and detection uses
common_utils.describe_context_script_bleed(), which does one fast combined
regex.search() prefilter before any detailed per-script scan — so the
overwhelming majority of (clean) rows cost almost nothing to check.

This does NOT re-translate anything itself. It only:
  1. Deletes glossary rows whose "translation" field looks like Thai,
     Sinhala, Myanmar, or Devanagari script (glossary_<lang>.db) — these are
     the ones that get fed back into every future prompt as "ESTABLISHED
     GLOSSARY ... apply exactly", so removing them is what stops the
     contamination from continuing to spread.
  2. NULLs out the "translation" field of contaminated rows in
     epitaka_<lang>.db's sentences table, which puts those lines back into
     "pending" state so a normal (non --overwrite) book_translator.py run
     will naturally retranslate them.

Usage:
    python cleanup_script_bleed.py --lang km --epitaka-db ./data/epitaka.db --dry-run

    python cleanup_script_bleed.py --lang km --epitaka-db ./data/epitaka.db --apply
Always run with --dry-run first and read the report before --apply.
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import common.common_utils as cu

# How often to print a progress line while streaming a large table.
_PROGRESS_EVERY = 50_000
# How many DELETE/UPDATE rows to batch per commit, to avoid one giant
# transaction (which on 1GB RAM can blow up SQLite's undo/rollback journal).
_WRITE_BATCH = 2_000


def _low_memory_pragmas(conn: sqlite3.Connection, read_heavy: bool = False) -> None:
    """
    Keep SQLite's own memory footprint small on a 1GB-RAM box.

    `read_heavy=True` is for connections that are about to do a full
    sequential table scan with no SQL-side filter available (the REGEXP
    fallback path) — a bigger page cache and mmap actually reduce disk
    I/O for that specific access pattern, and since it's a plain read
    connection (no undo/rollback journal growth risk like the batched
    write connections have), there's no rollback-journal downside to
    loosening it a bit. Still well within a 1GB budget.
    """
    if read_heavy:
        conn.execute("PRAGMA cache_size = -20000")      # ~20MB page cache
        conn.execute("PRAGMA mmap_size = 268435456")    # 256MB mmap window
    else:
        conn.execute("PRAGMA cache_size = -4000")   # ~4MB page cache
        conn.execute("PRAGMA mmap_size = 0")        # don't mmap the whole DB file into memory
    conn.execute("PRAGMA temp_store = FILE")    # spill temp b-trees to disk, not RAM


def _sql_regexp_available(conn: sqlite3.Connection) -> bool:
    """
    Probe whether this SQLite build has a REGEXP function registered (e.g.
    a loadable regexp extension). If it does, we can push the cheap
    "any watched script present" prefilter down into SQL and let SQLite's
    C-level scan reject the (huge majority of) clean rows itself, instead
    of streaming every single row across the DB-API boundary into Python
    just to run the same prefilter there. This is what actually gets the
    win — the current code already does the right thing algorithmically
    (fast combined-regex prefilter, fetchmany batching); the cost that's
    left is the per-row Python<->SQLite round trip itself.
    """
    try:
        conn.execute("SELECT 1 WHERE 'x' REGEXP 'x'").fetchone()
        return True
    except sqlite3.OperationalError:
        return False


def _scan_table(conn: sqlite3.Connection, sql: str, params: tuple, lang: str,
                 text_field: str, table_desc: str):
    """
    Stream rows via a raw cursor with small server-side batches (NOT
    fetchall — that would materialize the entire result set in RAM, which
    is exactly what we can't afford on a 1.2M-row table with 1GB total
    RAM). Yields (row, reason) for every contaminated row found, printing
    periodic progress since a full-table scan can take a while on a single
    CPU.
    """
    cur = conn.cursor()
    cur.execute(sql, params)
    scanned = 0
    found = 0
    t0 = time.time()
    while True:
        chunk = cur.fetchmany(1000)   # small batches, not the whole table
        if not chunk:
            break
        for row in chunk:
            scanned += 1
            reason = cu.describe_context_script_bleed(lang, row[text_field] or "")
            if reason:
                found += 1
                yield row, reason
            if scanned % _PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = scanned / elapsed if elapsed > 0 else 0
                print(f"    ...scanned {scanned:,} rows of {table_desc} "
                      f"({found} contaminated so far, {rate:,.0f} rows/sec)", file=sys.stderr)
    elapsed = time.time() - t0
    print(f"    scanned {scanned:,} rows of {table_desc} in {elapsed:,.1f}s "
          f"({found} contaminated)", file=sys.stderr)


def find_bad_glossary_rows(glossary_db: str, lang: str) -> list[tuple[sqlite3.Row, str]]:
    if not Path(glossary_db).exists():
        return []
    conn = sqlite3.connect(glossary_db)
    conn.row_factory = sqlite3.Row
    _low_memory_pragmas(conn)
    try:
        regexp_ok = _sql_regexp_available(conn)
        if regexp_ok:
            # Let SQLite reject the clean rows itself; only rows that trip
            # the wide "any watched script" net ever cross into Python.
            sql = ("SELECT id, pali, translation FROM glossary "
                   "WHERE translation REGEXP ?")
            params = (cu.ANY_WATCHED_SCRIPT_SQL_REGEX,)
            print("    [glossary] REGEXP available — filtering in SQL", file=sys.stderr)
        else:
            sql, params = "SELECT id, pali, translation FROM glossary", ()
            _low_memory_pragmas(conn, read_heavy=True)  # full scan, no filter to lean on
            print("    [glossary] no SQLite REGEXP function — streaming full table "
                  "(pass --dry-run and check `sqlite3 -version` / your build's "
                  "regexp extension if you expected SQL-side filtering)", file=sys.stderr)
        return list(_scan_table(conn, sql, params, lang, "translation", "glossary"))
    finally:
        conn.close()


def find_bad_sentence_rows(lang_db: str, lang: str, book_id: str | None) -> list[tuple[sqlite3.Row, str]]:
    if not Path(lang_db).exists():
        return []
    conn = sqlite3.connect(lang_db)
    conn.row_factory = sqlite3.Row
    _low_memory_pragmas(conn)
    try:
        regexp_ok = _sql_regexp_available(conn)
        base_cols = "book_id, para_id, line_id, translation"
        base_where = "translation IS NOT NULL AND translation != ''"
        if regexp_ok:
            base_where += " AND translation REGEXP :pat"
            print("    [sentences] REGEXP available — filtering in SQL", file=sys.stderr)
        else:
            _low_memory_pragmas(conn, read_heavy=True)  # full scan, no filter to lean on
            print("    [sentences] no SQLite REGEXP function — streaming full table "
                  "(pass --dry-run and check `sqlite3 -version` / your build's "
                  "regexp extension if you expected SQL-side filtering)", file=sys.stderr)
        if book_id:
            sql = f"SELECT {base_cols} FROM sentences WHERE book_id=:book AND {base_where}"
            params = {"book": book_id}
        else:
            sql = f"SELECT {base_cols} FROM sentences WHERE {base_where}"
            params = {}
        if regexp_ok:
            params["pat"] = cu.ANY_WATCHED_SCRIPT_SQL_REGEX
        return list(_scan_table(conn, sql, params, lang, "translation", "sentences"))
    finally:
        conn.close()


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="Target language code, e.g. km")
    ap.add_argument("--epitaka-db", default=cu.EPITAKA_DB, help="Path to epitaka.db (used to derive per-lang DB paths)")
    ap.add_argument("--book", default=None, help="Limit sentence cleanup to one book_id (default: all books)")
    ap.add_argument("--apply", action="store_true", help="Actually delete/null the rows (default is dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="Explicit dry-run (default behaviour if --apply not given)")
    ap.add_argument("--report-limit", type=int, default=50, help="Max contaminated rows to print in the report")
    args = ap.parse_args()

    apply_changes = args.apply and not args.dry_run

    glossary_db = cu.glossary_db_path(args.epitaka_db, args.lang)
    lang_db     = cu.lang_db_path(args.epitaka_db, args.lang)

    print(f"[cleanup] lang={args.lang!r}")
    print(f"[cleanup] glossary DB : {glossary_db}")
    print(f"[cleanup] sentences DB: {lang_db}")
    print(f"[cleanup] mode        : {'APPLY (will modify DBs)' if apply_changes else 'DRY RUN (no changes will be made)'}")
    print(f"[cleanup] scanning is streamed (low memory) — this can take a while on large tables, progress logged to stderr")
    print()

    bad_gloss = find_bad_glossary_rows(glossary_db, args.lang)
    print(f"Found {len(bad_gloss)} contaminated glossary entr(y/ies):")
    for row, reason in bad_gloss[:args.report_limit]:
        print(f"  id={row['id']:<6} {row['pali']!r} -> {row['translation']!r}  [{reason}]")
    if len(bad_gloss) > args.report_limit:
        print(f"  ... and {len(bad_gloss) - args.report_limit} more")
    print()

    bad_sent = find_bad_sentence_rows(lang_db, args.lang, args.book)
    print(f"Found {len(bad_sent)} contaminated sentence translation(s):")
    for row, reason in bad_sent[:args.report_limit]:
        print(f"  {row['book_id']} p{row['para_id']}L{row['line_id']}: {row['translation'][:80]!r}  [{reason}]")
    if len(bad_sent) > args.report_limit:
        print(f"  ... and {len(bad_sent) - args.report_limit} more")
    print()

    if not apply_changes:
        print("Dry run only — nothing was changed. Re-run with --apply to fix these.")
        return

    if bad_gloss:
        conn = sqlite3.connect(glossary_db)
        _low_memory_pragmas(conn)
        try:
            ids = [(row["id"],) for row, _reason in bad_gloss]
            for batch in _batched(ids, _WRITE_BATCH):
                conn.executemany("DELETE FROM glossary WHERE id=?", batch)
                conn.commit()   # commit each batch, not one giant transaction
        finally:
            conn.close()
        print(f"Deleted {len(bad_gloss)} glossary row(s) from {glossary_db}")

    if bad_sent:
        conn = sqlite3.connect(lang_db)
        _low_memory_pragmas(conn)
        try:
            keys = [(row["book_id"], row["para_id"], row["line_id"]) for row, _reason in bad_sent]
            for batch in _batched(keys, _WRITE_BATCH):
                conn.executemany(
                    "UPDATE sentences SET translation=NULL, translation_confidence=NULL, "
                    "confidence_note=NULL WHERE book_id=? AND para_id=? AND line_id=?",
                    batch,
                )
                conn.commit()
        finally:
            conn.close()
        print(f"Cleared {len(bad_sent)} sentence translation(s) in {lang_db} — "
              f"they are now pending again and will be retranslated on the next "
              f"normal (non --overwrite) book_translator.py run.")


if __name__ == "__main__":
    main()