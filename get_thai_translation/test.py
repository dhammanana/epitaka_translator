"""
heading_ranges.py
-----------------
Computes thai_page and nissaya_para ranges for each heading match,
then writes results into the matching table in thaimm.sqlite (created if needed).

Key design:
  - Thai page ranges  → partitioned by (thai_volume_id, nissaya_book_id)
                        so Mūla (Vin-i) and Aṭṭhakathā (Sp-i) rows
                        never look at each other's pages.
  - Nissaya para ranges → partitioned by (nissaya_book_id)
                        same reason — each CST book is its own sequence.

Usage:
    python heading_ranges.py                         # writes to matching table
    python heading_ranges.py --db /path/to/thaimm.sqlite
    python heading_ranges.py --csv out.csv           # also dump CSV
    python heading_ranges.py --dry-run               # print only, no DB write
"""

import sqlite3
import argparse
import csv

DB_PATH = "thaimm.sqlite"

QUERY = """
SELECT
    hm.id,
    hm.book_pair_id,
    bp.thai_volume_id,
    bp.nissaya_book_id,
    hm.thai_page,
    hm.thai_title,
    hm.nissaya_para_id,
    hm.nissaya_title
FROM heading_matches hm
JOIN book_pairs bp ON bp.id = hm.book_pair_id
ORDER BY bp.thai_volume_id, bp.nissaya_book_id, hm.thai_page, hm.nissaya_para_id
"""


def load_rows(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(QUERY).fetchall()]
    con.close()
    print(f"  Loaded {len(rows)} rows from heading_matches.")
    return rows


def next_distinct_map(groups, sort_key, value_key):
    """
    For each row in each group, find the next row that has a DIFFERENT value
    for value_key (skipping duplicates).

    Returns: dict of row["id"] → next distinct value (or None if last distinct)

    Example: rows with nissaya_para_id = [4,4,4,4,4,4,4,4,4,35,36,36,36,36,67]
             every row with para=4  → next distinct = 35
             every row with para=35 → next distinct = 36
             every row with para=36 → next distinct = 67
             last row with para=67  → None
    """
    result = {}
    for group in groups.values():
        group_sorted = sorted(group, key=sort_key)
        # Build list of (value, id) pairs in order
        n = len(group_sorted)
        for i, r in enumerate(group_sorted):
            current_val = r[value_key]
            # Scan forward to find the first row with a different value
            next_val = None
            for j in range(i + 1, n):
                if group_sorted[j][value_key] != current_val:
                    next_val = group_sorted[j][value_key]
                    break
            result[r["id"]] = next_val
    return result


def compute_ranges(rows):
    """
    For each row compute Thai page range and Nissaya para range.

    Thai pages   : partitioned by (thai_volume_id, nissaya_book_id)
                   → Mūla and Aṭṭhakathā never share a sequence
                   → end = next DISTINCT thai_page - 1

    Nissaya paras: partitioned by (nissaya_book_id,)
                   → end = next DISTINCT nissaya_para_id - 1
                   → Many Thai headings can share the same nissaya_para_id
                     (e.g. all headings inside Verañjakaṇḍaṃ share para 4).
                     They all get the same nissaya end = next_distinct_para - 1.

    Last distinct value in each partition → end = None, count = None
    """

    # ── Thai page ranges ──────────────────────────────────────────────────────
    # Strategy: for each row, thai_page_end = min(next_own_page, next_any_page) - 1
    #
    # "next_own_page"  = next distinct thai_page within the same (vol, book)
    # "next_any_page"  = next distinct thai_page across ALL books in the same volume
    #
    # This handles interleaving correctly: a Vin-i heading at page 575 ends at
    # page 587 (next Vin-i page 588), not at 18 (where Sp-i first appeared).
    # But a Vin-i heading at page 16 ends at 18 because Sp-i page 19 comes next
    # in the global sequence before the next Vin-i heading.

    thai_groups = {}
    for r in rows:
        key = (r["thai_volume_id"], r["nissaya_book_id"])
        thai_groups.setdefault(key, []).append(r)

    # next distinct page within same (vol, book)
    thai_next_own = next_distinct_map(
        thai_groups,
        sort_key=lambda x: (x["thai_page"], x["id"]),
        value_key="thai_page",
    )

    # next distinct page across ALL books in same volume
    vol_groups = {}
    for r in rows:
        vol_groups.setdefault(r["thai_volume_id"], []).append(r)

    thai_next_any = next_distinct_map(
        vol_groups,
        sort_key=lambda x: (x["thai_page"], x["id"]),
        value_key="thai_page",
    )

    # For each row, take the smaller of the two next pages
    thai_next = {}
    for r in rows:
        own = thai_next_own[r["id"]]
        any_ = thai_next_any[r["id"]]
        if own is None and any_ is None:
            thai_next[r["id"]] = None
        elif own is None:
            thai_next[r["id"]] = any_
        elif any_ is None:
            thai_next[r["id"]] = own
        else:
            thai_next[r["id"]] = min(own, any_)

    # ── Nissaya para ranges ───────────────────────────────────────────────────
    nissaya_groups = {}
    for r in rows:
        nissaya_groups.setdefault(r["nissaya_book_id"], []).append(r)

    nissaya_next = next_distinct_map(
        nissaya_groups,
        sort_key=lambda x: (x["nissaya_para_id"], x["id"]),
        value_key="nissaya_para_id",
    )

    # ── Assemble results ──────────────────────────────────────────────────────
    results = []
    for r in rows:
        tp_start = r["thai_page"]
        next_tp  = thai_next[r["id"]]
        tp_end   = (next_tp) if next_tp is not None else None
        tp_count = (tp_end - tp_start + 1) if tp_end is not None else None

        np_start = r["nissaya_para_id"]
        next_np  = nissaya_next[r["id"]]
        np_end   = (next_np - 1) if next_np is not None else None
        np_count = (np_end - np_start + 1) if np_end is not None else None

        results.append({
            "thai_volume_id":     r["thai_volume_id"],
            "nissaya_book_id":    r["nissaya_book_id"],
            "thai_title":         r["thai_title"],
            "thai_page_start":    tp_start,
            "thai_page_end":      tp_end,
            "thai_page_count":    tp_count,
            "nissaya_title":      r["nissaya_title"],
            "nissaya_para_start": np_start,
            "nissaya_para_end":   np_end,
            "nissaya_para_count": np_count,
        })

    return results


def check_anomalies(results):
    """Report rows with count = 0, negative, or None."""
    bad_thai  = [r for r in results if r["thai_page_count"] is not None and r["thai_page_count"] <= 0]
    bad_niss  = [r for r in results if r["nissaya_para_count"] is not None and r["nissaya_para_count"] <= 0]
    null_thai = [r for r in results if r["thai_page_end"] is None]
    null_niss = [r for r in results if r["nissaya_para_end"] is None]

    print(f"\n── Anomaly report ────────────────────────────────────────────")
    print(f"  thai_page_count   <= 0 : {len(bad_thai):>4}  (duplicate/overlapping pages)")
    print(f"  nissaya_para_count <= 0: {len(bad_niss):>4}  (duplicate/overlapping paras)")
    print(f"  thai_page_end = NULL   : {len(null_thai):>4}  (last heading per vol+book group)")
    print(f"  nissaya_para_end = NULL: {len(null_niss):>4}  (last heading per CST book)")

    if bad_thai:
        print("\n  Bad Thai page ranges (first 10):")
        for r in bad_thai[:10]:
            print(f"    [{r['nissaya_book_id']}] vol={r['thai_volume_id']} "
                  f"page {r['thai_page_start']}→{r['thai_page_end']} "
                  f"(count={r['thai_page_count']})  {r['thai_title'][:50]}")

    if bad_niss:
        print("\n  Bad nissaya para ranges (first 10):")
        for r in bad_niss[:10]:
            print(f"    [{r['nissaya_book_id']}] para {r['nissaya_para_start']}→{r['nissaya_para_end']} "
                  f"(count={r['nissaya_para_count']})  {r['nissaya_title'][:50]}")
    print()


def write_db(results, db_path):
    """Create matching table if needed, then replace its contents."""
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS matching (
            thai_volume_id      TEXT,
            nissaya_book_id     TEXT,
            thai_title          TEXT,
            thai_page_start     INTEGER,
            thai_page_end       INTEGER,
            thai_page_count     INTEGER,
            nissaya_title       TEXT,
            nissaya_para_start  INTEGER,
            nissaya_para_end    INTEGER,
            nissaya_para_count  INTEGER
        )
    """)
    con.execute("DELETE FROM matching")

    fields = [
        "thai_volume_id", "nissaya_book_id",
        "thai_title", "thai_page_start", "thai_page_end", "thai_page_count",
        "nissaya_title", "nissaya_para_start", "nissaya_para_end", "nissaya_para_count",
    ]
    placeholders = ", ".join("?" * len(fields))
    sql = f"INSERT INTO matching ({', '.join(fields)}) VALUES ({placeholders})"
    con.executemany(sql, [[r[f] for f in fields] for r in results])
    con.commit()
    con.close()
    print(f"  Wrote {len(results)} rows into matching.")


def write_csv(results, path):
    fields = [
        "thai_volume_id", "nissaya_book_id",
        "thai_title", "thai_page_start", "thai_page_end", "thai_page_count",
        "nissaya_title", "nissaya_para_start", "nissaya_para_end", "nissaya_para_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {path}")


def print_sample(results, n=15):
    print(f"\n── First {n} rows ─────────────────────────────────────────────")
    hdr = (f"{'vol':>4}  {'book':<8}  {'thai_title':<35}  "
           f"{'thai pages':<12}  {'pg ct':>5}  {'para range':<12}  {'pa ct':>5}")
    print(hdr)
    print("─" * len(hdr))
    for r in results[:n]:
        tp = f"{r['thai_page_start']}–{r['thai_page_end'] or '?'}"
        np = f"{r['nissaya_para_start']}–{r['nissaya_para_end'] or '?'}"
        print(f"{str(r['thai_volume_id']):>4}  {r['nissaya_book_id']:<8}  "
              f"{r['thai_title'][:34]:<35}  {tp:<12}  "
              f"{str(r['thai_page_count'] or '?'):>5}  {np:<12}  "
              f"{str(r['nissaya_para_count'] or '?'):>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",      default=DB_PATH,  help="Path to thaimm.sqlite")
    ap.add_argument("--csv",     default=None,     help="Also save results to CSV")
    ap.add_argument("--dry-run", action="store_true", help="Print only, don't write to DB")
    args = ap.parse_args()

    print(f"Reading {args.db} …")
    rows    = load_rows(args.db)
    results = compute_ranges(rows)

    print_sample(results)
    check_anomalies(results)

    if args.csv:
        write_csv(results, args.csv)

    if not args.dry_run:
        write_db(results, args.db)
    else:
        print("  --dry-run: DB not modified.")


if __name__ == "__main__":
    main()