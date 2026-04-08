"""
jobs/at_data.py  —  Data-access helpers for AṭṭhakathāTranslatorJob.

Adds three new public functions on top of st_data's API:

  fetch_headings_no_sc_id(params, log_info)
      → list[dict]
      Like st_data.fetch_headings(), but returns headings where sc_id IS NULL
      or empty — i.e. aṭṭhakathā / ṭīkā / aññā texts not in SuttaCentral.

  fetch_mula_translation_block(params, src_lines, max_chars, log_info, log_warn)
      → str
      Follow book_links in reverse (current text is dst, mūla is src) to find
      the root-text paragraphs that the current commentary passage discusses.
      Retrieve their already-translated English sentences and return them as a
      formatted block for inclusion in the prompt.

  fetch_similar_translations_block(params, book_id, pali_text, max_chars,
                                   log_info, log_warn)
      → str
      Token-overlap search: extract significant tokens from pali_text, find
      already-translated sentences in the same book whose pali_sentence shares
      the most tokens, and return them as a context block for consistent
      phrasing.

Design rule inherited from st_data.py:
  Every function opens its own SQLite connection and closes it on exit.
  No connection object is ever held between calls.
"""

import re
import sqlite3
import logging
from contextlib import contextmanager
from typing import Callable

from config import NISSAYA_DB

logger = logging.getLogger(__name__)

_Log = Callable[[str], None]


# ══════════════════════════════════════════════════════════════════
# Internal helpers  (mirrors st_data._connect / nissaya_path)
# ══════════════════════════════════════════════════════════════════

def _nissaya_path(params: dict) -> str:
    """Re-use same resolution logic as st_data.nissaya_path."""
    path = params.get("nissaya_db") or NISSAYA_DB
    if not path:
        raise RuntimeError(
            "nissaya_db path is not configured. "
            "Set NISSAYA_DB in config.py or pass it as a task param."
        )
    return str(path)


@contextmanager
def _connect(path: str):
    """Open a SQLite connection, yield it, always close on exit."""
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        yield conn
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════
# 1. Headings without sc_id
# ══════════════════════════════════════════════════════════════════

def fetch_headings_no_sc_id(params: dict, log_info: _Log) -> list[dict]:
    """
    Return headings that have a NULL / empty sc_id and still have
    at least one untranslated sentence.

    Params consumed: book_id, heading_id, batch_size, overwrite.

    The `heading_id` param accepts a para_id (int as string) to pin
    a single heading — analogous to st_job's `sc_id` param.
    """
    book_id_filter   = params.get("book_id",    "").strip()
    heading_id_filter = params.get("heading_id", "").strip()
    batch_size       = int(params.get("batch_size", 10))
    overwrite        = bool(params.get("overwrite", False))

    log_info(
        f"[AT-DB] fetch_headings_no_sc_id — "
        f"book_id={book_id_filter!r}, heading_id={heading_id_filter!r}, "
        f"batch_size={batch_size}, overwrite={overwrite}"
    )

    path = _nissaya_path(params)
    try:
        with _connect(path) as conn:

            if heading_id_filter:
                rows = conn.execute(
                    "SELECT * FROM headings WHERE para_id = ?",
                    (int(heading_id_filter),)
                ).fetchall()
                candidates = [dict(r) for r in rows]

            elif book_id_filter:
                rows = conn.execute(
                    """SELECT * FROM headings
                       WHERE book_id = ?
                         AND (sc_id IS NULL OR sc_id = '')
                       ORDER BY para_id""",
                    (book_id_filter,)
                ).fetchall()
                candidates = [dict(r) for r in rows]

            else:
                rows = conn.execute(
                    """SELECT * FROM headings
                       WHERE sc_id IS NULL OR sc_id = ''
                       ORDER BY book_id, para_id"""
                ).fetchall()
                candidates = [dict(r) for r in rows]

            log_info(f"[AT-DB] {len(candidates)} candidate heading(s) found (no sc_id).")

            if overwrite or not candidates:
                result = candidates[:batch_size]
            else:
                result = []
                for h in candidates:
                    if len(result) >= batch_size:
                        break
                    start = h["para_id"]
                    end   = start + int(h.get("chapter_len") or 1)
                    row = conn.execute(
                        """SELECT COUNT(*) FROM sentences
                           WHERE book_id = ?
                             AND para_id >= ? AND para_id < ?
                             AND (english_translation IS NULL
                                  OR english_translation = '')""",
                        (h["book_id"], start, end)
                    ).fetchone()
                    if row[0] > 0:
                        result.append(h)

    except Exception as exc:
        log_info(f"[AT-DB] fetch_headings_no_sc_id EXCEPTION — {type(exc).__name__}: {exc}")
        raise

    log_info(f"[AT-DB] fetch_headings_no_sc_id → {len(result)} heading(s) need work.")
    return result


# ══════════════════════════════════════════════════════════════════
# 2. Mūla translation block
# ══════════════════════════════════════════════════════════════════

def fetch_mula_translation_block(
    params:    dict,
    src_lines: list[tuple],   # (book_id, para_id, line_id) of the commentary text
    max_chars: int = 3000,
    log_info:  _Log = None,
    log_warn:  _Log = None,
) -> str:
    """
    Find the mūla (root-text) paragraphs that the current commentary passage
    comments on, then return their already-translated English sentences.

    Strategy
    ────────
    The existing book_links table is directional:
        src_book / src_para / src_line  →  dst_book / dst_para / dst_line

    For SentenceTranslatorJob the *current text* is always the src (mūla)
    and dst is the commentary.  Here the roles are reversed: the current
    text *is* the dst (commentary), so we look for rows where
    (dst_book, dst_para, dst_line) matches our src_lines, and read the
    corresponding src (mūla) paragraphs.

    We then fetch the `english_translation` for those mūla paragraphs
    (they should already be translated by SentenceTranslatorJob).

    Returns a formatted string ready for prompt injection.
    """
    if log_info is None:
        log_info = lambda x: None
    if log_warn is None:
        log_warn = lambda x: None

    if not src_lines:
        return "(no mūla text available)"

    path = _nissaya_path(params)

    try:
        with _connect(path) as conn:

            # Guard: book_links must exist
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='book_links'"
            ).fetchone()
            if not has_table:
                log_warn("[AT-DB] book_links table not found; skipping mūla lookup.")
                return "(no mūla text available)"

            # Load current commentary lines into temp table
            conn.execute("DROP TABLE IF EXISTS _tmp_at_src")
            conn.execute(
                "CREATE TEMP TABLE _tmp_at_src"
                " (book_id TEXT, para_id INTEGER, line_id INTEGER)"
            )
            conn.executemany("INSERT INTO _tmp_at_src VALUES (?,?,?)", src_lines)

            # Reverse lookup: commentary lines are the *dst* side of book_links
            mula_links = conn.execute(
                """SELECT DISTINCT bl.src_book, bl.src_para
                   FROM   book_links bl
                   JOIN   _tmp_at_src t
                          ON  bl.dst_book = t.book_id
                          AND bl.dst_para = t.para_id
                          AND bl.dst_line = t.line_id
                   ORDER  BY bl.src_book, bl.src_para"""
            ).fetchall()
            conn.execute("DROP TABLE IF EXISTS _tmp_at_src")

            if not mula_links:
                log_info(
                    f"[AT-DB] Mūla lookup: no book_links found for "
                    f"{len(src_lines)} commentary line(s)."
                )
                return "(no mūla text available)"

            log_info(
                f"[AT-DB] Mūla lookup: {len(mula_links)} unique mūla paragraph(s) "
                f"for {len(src_lines)} commentary line(s)."
            )

            # Fetch English translations for those mūla paragraphs
            conn.execute("DROP TABLE IF EXISTS _tmp_mula_paras")
            conn.execute(
                "CREATE TEMP TABLE _tmp_mula_paras (book_id TEXT, para_id INTEGER)"
            )
            conn.executemany(
                "INSERT INTO _tmp_mula_paras VALUES (?,?)",
                [(r["src_book"], r["src_para"]) for r in mula_links]
            )

            rows = conn.execute(
                """SELECT s.book_id, s.para_id, s.line_id,
                          s.pali_sentence, s.english_translation
                   FROM   sentences s
                   JOIN   _tmp_mula_paras m
                          ON  s.book_id = m.book_id
                          AND s.para_id = m.para_id
                   WHERE  s.english_translation IS NOT NULL
                     AND  s.english_translation != ''
                   ORDER  BY s.book_id, s.para_id, s.line_id"""
            ).fetchall()
            conn.execute("DROP TABLE IF EXISTS _tmp_mula_paras")

        if not rows:
            log_info("[AT-DB] Mūla paragraphs found but none have English translations yet.")
            return "(mūla text exists but not yet translated)"

        # Format the block
        sections: dict[tuple, list[str]] = {}
        for r in rows:
            key = (r["book_id"], r["para_id"])
            sections.setdefault(key, []).append(
                f"  [{r['line_id']}] {r['pali_sentence'] or ''}\n"
                f"        EN: {r['english_translation']}"
            )

        parts = ["[Mūla Text with Translations — use these for consistent terminology]"]
        total_chars = len(parts[0])
        for (book, para), sent_lines in sections.items():
            header = f"\n[{book} §{para}]"
            block  = header + "\n" + "\n".join(sent_lines)
            if max_chars > 0 and total_chars + len(block) > max_chars:
                parts.append(f"\n… [mūla block truncated at {max_chars} chars]")
                break
            parts.append(block)
            total_chars += len(block)

        result = "\n".join(parts)
        log_info(
            f"[AT-DB] Mūla block: {len(rows)} sentence(s), "
            f"{len(result)} chars."
        )
        return result

    except Exception as exc:
        log_warn(f"[AT-DB] Mūla translation block failed: {exc}")
        return "(mūla text unavailable)"


# ══════════════════════════════════════════════════════════════════
# 3. Similar sentence translations
# ══════════════════════════════════════════════════════════════════

def _significant_tokens(text: str, min_len: int = 5) -> set[str]:
    """
    Extract lowercased tokens longer than min_len chars from Pāli text.
    Filters out very common grammatical particles (ti, ca, vā, etc.).
    """
    _STOP = {
        "tena", "hetu", "pana", "evam", "tattha", "ettha",
        "yatha", "tatha", "idha", "tattha", "yeva", "ceva",
    }
    tokens = re.split(r"[\s,;.\u2018\u2019\"'()\[\]\u0964\u0965]+", text)
    return {
        t.lower() for t in tokens
        if len(t) >= min_len and t.lower() not in _STOP
    }


def fetch_similar_translations_block(
    params:    dict,
    book_id:   str,
    pali_text: str,
    max_chars: int = 2000,
    top_n:     int = 8,
    log_info:  _Log = None,
    log_warn:  _Log = None,
) -> str:
    """
    Find already-translated sentences in the same book whose Pāli text
    overlaps most with pali_text (by shared token count).

    Returns a formatted block of (pali, english) pairs for prompt injection.

    Algorithm
    ─────────
    1.  Extract significant tokens (≥5 chars) from pali_text.
    2.  For each token, collect the set of (para_id, line_id) that contain it
        via a LIKE scan — kept tractable by limiting to the same book and to
        sentences that already have an English translation.
    3.  Score each sentence by |intersection(query_tokens, sentence_tokens)|.
    4.  Return the top_n highest-scoring sentences.

    This is intentionally lightweight (no embeddings, no FTS index required).
    For texts with FTS5 enabled, a future upgrade could use matchinfo().
    """
    if log_info is None:
        log_info = lambda x: None
    if log_warn is None:
        log_warn = lambda x: None

    query_tokens = _significant_tokens(pali_text)
    if not query_tokens:
        return "(no similar sentences found)"

    path = _nissaya_path(params)

    try:
        with _connect(path) as conn:

            # Fetch all translated sentences in the same book.
            # We intentionally do NOT filter by para_id range — we want to
            # surface similar passages from anywhere in the same book.
            rows = conn.execute(
                """SELECT para_id, line_id, pali_sentence, english_translation
                   FROM   sentences
                   WHERE  book_id = ?
                     AND  english_translation IS NOT NULL
                     AND  english_translation != ''
                     AND  pali_sentence IS NOT NULL
                     AND  pali_sentence != ''
                   ORDER  BY para_id, line_id""",
                (book_id,)
            ).fetchall()

        if not rows:
            log_info(f"[AT-DB] Similar sentences: no translated sentences in book {book_id!r}.")
            return "(no similar sentences found)"

        log_info(
            f"[AT-DB] Similar sentences: scoring {len(rows)} candidate(s) "
            f"against {len(query_tokens)} query token(s)."
        )

        # Score each sentence by token overlap
        scored: list[tuple[int, dict]] = []
        for r in rows:
            sent_tokens = _significant_tokens(r["pali_sentence"] or "")
            overlap     = len(query_tokens & sent_tokens)
            if overlap > 0:
                scored.append((overlap, dict(r)))

        if not scored:
            log_info("[AT-DB] Similar sentences: no overlap found.")
            return "(no similar sentences found)"

        # Sort descending by overlap score, take top_n
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_n]

        log_info(
            f"[AT-DB] Similar sentences: top {len(top)} match(es), "
            f"best overlap={top[0][0]} token(s)."
        )

        # Format
        parts = ["[Similar Already-Translated Sentences — for consistent phrasing]"]
        total_chars = len(parts[0])
        for score, r in top:
            entry = (
                f"\n  [overlap={score}] {r['pali_sentence']}\n"
                f"        EN: {r['english_translation']}"
            )
            if max_chars > 0 and total_chars + len(entry) > max_chars:
                parts.append("\n  … [truncated]")
                break
            parts.append(entry)
            total_chars += len(entry)

        return "\n".join(parts)

    except Exception as exc:
        log_warn(f"[AT-DB] Similar sentence lookup failed: {exc}")
        return "(similar sentences unavailable)"
def fetch_mula_layers_block(
    params:    dict,
    book_id:   str,          # the book currently being translated
    src_lines: list[tuple],  # (book_id, para_id, line_id) of the current text
    log_info:  "_Log" = None,
    log_warn:  "_Log" = None,
) -> str:
    """
    Walk book_links in reverse (dst→src) layer by layer, collecting every
    ancestor paragraph that has already-translated English sentences.

    Algorithm
    ─────────
    Layer 0 = the sentences being translated right now  (src_lines).
    Layer 1 = look in book_links WHERE (dst_book, dst_para, dst_line) ∈ Layer-0
              → get unique (src_book, src_para) pairs.
    Layer 2 = look in book_links WHERE (dst_book, dst_para) ∈ Layer-1 src pairs
              → get the next ancestors.
    …and so on until no new ancestors are found or a book repeats.

    For each layer, we fetch sentences WHERE english_translation IS NOT NULL.

    Returns a formatted multi-section string.  Returns "(no mūla text available)"
    if nothing is found.
    """
    if log_info is None:
        log_info = lambda x: None
    if log_warn is None:
        log_warn = lambda x: None

    if not src_lines:
        return "(no mūla text available)"

    path = _nissaya_path(params)

    try:
        with _connect(path) as conn:

            # Guard
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='book_links'"
            ).fetchone()
            if not has_table:
                log_warn("[AT-DB] book_links not found; skipping mūla layer lookup.")
                return "(no mūla text available)"

            all_parts: list[str] = []
            visited_books: set[str] = {book_id}  # don't revisit the current book

            # Seed: the (dst_book, dst_para, dst_line) triples we look up first
            # are exactly the src_lines of the text being translated.
            current_dst_lines = src_lines   # list of (book_id, para_id, line_id)
            layer_num = 0

            while current_dst_lines:
                layer_num += 1

                # ── Reverse lookup: find ancestor (src_book, src_para) pairs ──
                conn.execute("DROP TABLE IF EXISTS _tmp_ul_dst")
                conn.execute(
                    "CREATE TEMP TABLE _tmp_ul_dst"
                    " (book_id TEXT, para_id INTEGER, line_id INTEGER)"
                )
                conn.executemany(
                    "INSERT INTO _tmp_ul_dst VALUES (?,?,?)",
                    current_dst_lines
                )

                ancestor_rows = conn.execute(
                    """SELECT DISTINCT bl.src_book, bl.src_para
                       FROM   book_links bl
                       JOIN   _tmp_ul_dst t
                              ON  bl.dst_book = t.book_id
                              AND bl.dst_para = t.para_id
                              AND bl.dst_line = t.line_id
                       ORDER  BY bl.src_book, bl.src_para"""
                ).fetchall()
                conn.execute("DROP TABLE IF EXISTS _tmp_ul_dst")

                if not ancestor_rows:
                    log_info(
                        f"[AT-DB] Mūla layer {layer_num}: "
                        f"no further ancestors found. Stopping."
                    )
                    break

                # Skip ancestor books we have already processed (loop guard)
                new_pairs = [
                    (r["src_book"], r["src_para"])
                    for r in ancestor_rows
                    if r["src_book"] not in visited_books
                ]

                if not new_pairs:
                    log_info(
                        f"[AT-DB] Mūla layer {layer_num}: "
                        f"all ancestor books already visited. Stopping."
                    )
                    break

                # Track which books appear at this layer
                layer_books = sorted({b for b, _ in new_pairs})
                for b in layer_books:
                    visited_books.add(b)

                log_info(
                    f"[AT-DB] Mūla layer {layer_num}: "
                    f"{len(new_pairs)} paragraph(s) in book(s) {layer_books}."
                )

                # ── Fetch translated sentences for these ancestor paragraphs ──
                conn.execute("DROP TABLE IF EXISTS _tmp_ul_anc")
                conn.execute(
                    "CREATE TEMP TABLE _tmp_ul_anc (book_id TEXT, para_id INTEGER)"
                )
                conn.executemany("INSERT INTO _tmp_ul_anc VALUES (?,?)", new_pairs)

                sent_rows = conn.execute(
                    """SELECT s.book_id, s.para_id, s.line_id,
                              s.pali_sentence, s.english_translation
                       FROM   sentences s
                       JOIN   _tmp_ul_anc a
                              ON  s.book_id = a.book_id
                              AND s.para_id = a.para_id
                       WHERE  s.english_translation IS NOT NULL
                         AND  s.english_translation != ''
                       ORDER  BY s.book_id, s.para_id, s.line_id"""
                ).fetchall()
                conn.execute("DROP TABLE IF EXISTS _tmp_ul_anc")

                if sent_rows:
                    # Format this layer
                    layer_label = (
                        "Mūla (Root Text)" if layer_num == 1
                        else f"Ancestor Layer {layer_num}"
                    )
                    sections: dict[tuple, list[str]] = {}
                    for r in sent_rows:
                        key = (r["book_id"], r["para_id"])
                        sections.setdefault(key, []).append(
                            f"  [{r['line_id']}] {r['pali_sentence'] or ''}\n"
                            f"        EN: {r['english_translation']}"
                        )

                    layer_parts = [
                        f"[{layer_label} — mirror all technical terms from here]"
                    ]
                    for (bk, para), lines in sections.items():
                        layer_parts.append(f"\n[{bk} §{para}]")
                        layer_parts.extend(lines)

                    all_parts.append("\n".join(layer_parts))
                    log_info(
                        f"[AT-DB] Mūla layer {layer_num}: "
                        f"{len(sent_rows)} translated sentence(s) included."
                    )
                else:
                    log_info(
                        f"[AT-DB] Mūla layer {layer_num}: "
                        f"ancestor paragraphs exist but none are translated yet."
                    )

                # ── Prepare next iteration: use ancestor (book, para) as new dst ──
                # We need (book_id, para_id, line_id) triples for the next reverse
                # lookup. Fetch all line_ids for the ancestor paragraphs.
                conn.execute("DROP TABLE IF EXISTS _tmp_ul_anc2")
                conn.execute(
                    "CREATE TEMP TABLE _tmp_ul_anc2 (book_id TEXT, para_id INTEGER)"
                )
                conn.executemany("INSERT INTO _tmp_ul_anc2 VALUES (?,?)", new_pairs)

                all_lines = conn.execute(
                    """SELECT s.book_id, s.para_id, s.line_id
                       FROM   sentences s
                       JOIN   _tmp_ul_anc2 a
                              ON  s.book_id = a.book_id
                              AND s.para_id = a.para_id
                       ORDER  BY s.book_id, s.para_id, s.line_id"""
                ).fetchall()
                conn.execute("DROP TABLE IF EXISTS _tmp_ul_anc2")

                current_dst_lines = [
                    (r["book_id"], r["para_id"], r["line_id"])
                    for r in all_lines
                ]

                if not current_dst_lines:
                    log_info(
                        f"[AT-DB] Mūla layer {layer_num}: "
                        f"no line_ids found for next iteration. Stopping."
                    )
                    break

        if not all_parts:
            return "(no mūla text available)"

        # Layers are in order ancestor-1, ancestor-2, …
        # Reverse so the deepest (most-root) text comes first in the prompt.
        all_parts.reverse()
        return "\n\n".join(all_parts)

    except Exception as exc:
        log_warn(f"[AT-DB] fetch_mula_layers_block failed: {exc}")
        return "(mūla text unavailable)"