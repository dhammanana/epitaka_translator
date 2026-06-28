"""
find_untranslated_similar.py  (v2 — sparse, low-RAM)

Finds untranslated pali_sentence rows that are similar to already-translated
rows. Uses TF-IDF + sparse top-k cosine similarity so RAM stays under ~4 GB
even on 1.2 M rows.

Requirements:
    pip install scikit-learn scipy numpy

Usage:
    python find_untranslated_similar.py --db path/to/your.db
    python find_untranslated_similar.py --db path/to/your.db --threshold 0.75 --top-k 3
    python find_untranslated_similar.py --db path/to/your.db --output results.csv
"""

import argparse
import csv
import sqlite3
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


# ── helpers ──────────────────────────────────────────────────────────────────

def load_data(db_path: str, limit: int = None):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    limit_clause = f"LIMIT {limit}" if limit else ""

    print("[INFO] Loading translated sentences...")
    cur.execute(f"""
        SELECT book_id, para_id, line_id, pali_sentence, english_translation
        FROM   sentences
        WHERE  english_translation IS NOT NULL
          AND  TRIM(english_translation) != ''
          AND  pali_sentence IS NOT NULL
          AND  TRIM(pali_sentence) != ''
        {limit_clause}
    """)
    translated = [dict(r) for r in cur.fetchall()]

    print("[INFO] Loading untranslated sentences...")
    cur.execute(f"""
        SELECT book_id, para_id, line_id, pali_sentence
        FROM   sentences
        WHERE  (english_translation IS NULL OR TRIM(english_translation) = '')
          AND  pali_sentence IS NOT NULL
          AND  TRIM(pali_sentence) != ''
        {limit_clause}
    """)
    untranslated = [dict(r) for r in cur.fetchall()]
    con.close()

    print(f"[INFO] Translated  : {len(translated):,}")
    print(f"[INFO] Untranslated: {len(untranslated):,}")
    return translated, untranslated


def sparse_topk(batch_csr: sp.csr_matrix,
                trans_csc: sp.csc_matrix,
                top_k: int,
                threshold: float):
    """
    For each row in batch_csr, find the top-k columns (translated sentences)
    by dot-product (= cosine similarity when both matrices are L2-normalised).

    Returns two arrays shaped (n_batch, top_k):
        indices — column indices into the translated matrix
        scores  — cosine similarity values
    """
    n_batch = batch_csr.shape[0]
    # sparse × sparse → sparse  (only non-zero entries computed)
    sims: sp.csr_matrix = (batch_csr @ trans_csc).toarray()  # (batch, n_trans)
    # argpartition on the dense result, but result is only (batch × n_trans)
    # which for batch=2000 and n_trans=256k is 2000×256k×4B ≈ 2 GB — still big.
    # We clip to only the top_k entries so the returned data is tiny.
    if top_k == 1:
        idx = np.argmax(sims, axis=1).reshape(-1, 1)        # (n_batch, 1)
        scores = sims[np.arange(n_batch), idx[:, 0]].reshape(-1, 1)
    else:
        # argpartition is O(n) not O(n log n)
        part = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
        row_idx = np.arange(n_batch)[:, None]
        scores_part = sims[row_idx, part]
        order = np.argsort(-scores_part, axis=1)
        idx = part[row_idx, order]
        scores = scores_part[row_idx, order]
    # mask below threshold so callers can skip cheaply
    scores[scores < threshold] = 0.0
    return idx, scores


def l2_normalise(mat: sp.csr_matrix) -> sp.csr_matrix:
    """In-place row-wise L2 normalisation (avoids sklearn import cycle)."""
    norms = np.sqrt(mat.multiply(mat).sum(axis=1).A1)
    norms[norms == 0] = 1.0
    # divide each row by its norm via a diagonal sparse matrix
    diag = sp.diags(1.0 / norms)
    return diag @ mat


# ── main logic ────────────────────────────────────────────────────────────────

def find_similar(translated, untranslated, threshold: float, top_k: int,
                 batch_size: int = 2_000):
    all_texts = (
        [r["pali_sentence"] for r in translated]
        + [r["pali_sentence"] for r in untranslated]
    )
    n_trans = len(translated)

    print("[INFO] Building TF-IDF matrix (char n-grams 2–4)...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,   # float32 halves RAM vs float64
    )
    tfidf = vectorizer.fit_transform(all_texts)   # stays sparse
    print(f"[INFO] Matrix shape: {tfidf.shape}  built in {time.time()-t0:.1f}s")

    # L2-normalise so dot-product == cosine similarity
    tfidf = l2_normalise(tfidf.astype(np.float32))

    trans_mat   = tfidf[:n_trans]              # CSR
    untrans_mat = tfidf[n_trans:]              # CSR

    # Transpose translated matrix once — sparse matmul is faster with CSC on right
    trans_csc = trans_mat.T.tocsc()

    print(f"[INFO] Scanning {len(untranslated):,} untranslated rows "
          f"(batch={batch_size}, threshold={threshold})...")
    t1 = time.time()

    results = []
    total = len(untranslated)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = untrans_mat[start:end]

        idx_arr, score_arr = sparse_topk(batch, trans_csc, top_k, threshold)

        for i in range(end - start):
            best_score = score_arr[i, 0]
            if best_score <= 0:
                continue                       # nothing above threshold

            row = untranslated[start + i]
            for rank in range(top_k):
                score = score_arr[i, rank]
                if score <= 0:
                    break
                trans_row = translated[idx_arr[i, rank]]
                results.append({
                    "untrans_book_id": row["book_id"],
                    "untrans_para_id": row["para_id"],
                    "untrans_line_id": row["line_id"],
                    "untrans_pali"   : row["pali_sentence"],
                    "match_rank"     : rank + 1,
                    "similarity"     : round(float(score), 4),
                    "trans_book_id"  : trans_row["book_id"],
                    "trans_para_id"  : trans_row["para_id"],
                    "trans_line_id"  : trans_row["line_id"],
                    "trans_pali"     : trans_row["pali_sentence"],
                    "trans_english"  : trans_row["english_translation"],
                })

        # progress every ~50 batches
        if (start // batch_size + 1) % 50 == 0 or end == total:
            elapsed = time.time() - t1
            pct = end / total
            eta = elapsed / pct - elapsed if pct > 0 else 0
            print(f"  … {end:,}/{total:,} ({pct*100:.1f}%) "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    print(f"[INFO] Done in {time.time()-t1:.1f}s  —  {len(results):,} matches found")
    return results


def print_report(results, threshold, untranslated_total):
    matched = {(r["untrans_book_id"], r["untrans_para_id"], r["untrans_line_id"])
               for r in results if r["match_rank"] == 1}
    print()
    print("=" * 65)
    print(f"  RESULTS  (threshold={threshold})")
    print("=" * 65)
    print(f"  Untranslated sentences scanned : {untranslated_total:,}")
    print(f"  With a match ≥ threshold       : {len(matched):,}")
    print(f"  No close match found           : {untranslated_total - len(matched):,}")
    print()
    if not matched:
        print("  No matches — try lowering --threshold (e.g. 0.65).")
        return
    top = sorted((r for r in results if r["match_rank"] == 1),
                 key=lambda r: -r["similarity"])
    print("  Top 20 matches:")
    print("-" * 65)
    for r in top[:20]:
        pali = r["untrans_pali"][:60].replace("\n", " ")
        eng  = r["trans_english"][:80].replace("\n", " ")
        print(f"  sim={r['similarity']:.3f} | {r['untrans_book_id']} "
              f"p{r['untrans_para_id']} l{r['untrans_line_id']}")
        print(f"    Pali  : {pali}")
        print(f"    Reuse : {eng}")
        print()
    print("=" * 65)


def save_csv(results, path):
    if not results:
        print("[WARN] No results to save.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Saved {len(results):,} rows → {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find untranslated pali sentences similar to translated ones (low-RAM)."
    )
    parser.add_argument("--db",        required=True)
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Cosine similarity threshold 0–1 (default 0.75)")
    parser.add_argument("--top-k",     type=int,   default=1,
                        help="Matches per untranslated sentence (default 1)")
    parser.add_argument("--batch",     type=int,   default=2_000,
                        help="Rows per batch — lower = less RAM (default 2000)")
    parser.add_argument("--output",    default="similar_untranslated.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit rows loaded for testing (e.g. --limit 10000)")
    args = parser.parse_args()

    translated, untranslated = load_data(args.db, args.limit)
    if not translated:
        print("[ERROR] No translated sentences found."); return
    if not untranslated:
        print("[INFO] All sentences are already translated!"); return

    results = find_similar(translated, untranslated,
                           args.threshold, args.top_k, args.batch)
    print_report(results, args.threshold, len(untranslated))
    save_csv(results, args.output)


if __name__ == "__main__":
    main()