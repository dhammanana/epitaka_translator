"""
ai_jobs/context_builders.py  —  Modular prompt-context builders.
"""

import re
import sqlite3
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable

from database import get_glossary_conn

logger = logging.getLogger(__name__)

_Log = Callable[[str], None]
_noop: _Log = lambda _: None


def _epitaka_path(params: dict) -> str:
    path = params.get("epitaka_db") or '../data/epitaka.db'
    if not path:
        raise RuntimeError("epitaka_db not configured (check EPITAKA_DB in config.py).")
    return str(path)


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


def _para_range(para_start: int, para_end: int) -> range:
    end = para_start if para_end == -1 else para_end
    return range(para_start, end + 1)


class ContextBlock(ABC):
    label: str = "CONTEXT"

    def __init__(self, params: dict, log_info: _Log = _noop, log_warn: _Log = _noop):
        self.params   = params
        self.log_info = log_info
        self.log_warn = log_warn

    @abstractmethod
    def build(self) -> str: ...

    def _wrap(self, content: str) -> str:
        bar = "═" * 30
        return f"{bar}\n{self.label}\n{bar}\n{content}"


# ══════════════════════════════════════════════════════════════════
# 1. NissayaContext
# ══════════════════════════════════════════════════════════════════

class NissayaContext(ContextBlock):
    label = "MYANMAR NISSAYA (word-by-word gloss, romanised)"

    def __init__(
        self,
        params:     dict,
        paragraphs: list[dict] | None = None,
        *,
        book_id:    str  = "",
        para_start: int  = 0,
        para_end:   int  = -1,
        log_info:   _Log = _noop,
        log_warn:   _Log = _noop,
    ):
        super().__init__(params, log_info, log_warn)
        if paragraphs is not None:
            self._paragraphs = paragraphs
        elif book_id:
            self._paragraphs = self._fetch_sentences(book_id, para_start, para_end)
        else:
            self._paragraphs = []

    def _fetch_sentences(self, book_id: str, para_start: int, para_end: int) -> list[dict]:
        """Fetch sentences (including english_translation) for all paragraphs in range."""
        result = []
        path = _epitaka_path(self.params)
        with _connect(path) as conn:
            for pid in _para_range(para_start, para_end):
                rows = conn.execute(
                    "SELECT line_id, pali_sentence, english_translation FROM sentences "
                    "WHERE book_id=? AND para_id=? ORDER BY line_id",
                    (book_id, pid),
                ).fetchall()
                if rows:
                    result.append({
                        "book_id": book_id,
                        "para_id": pid,
                        "sentences": [dict(r) for r in rows],
                    })
        return result

    def _fetch_nissaya_map(self, conn: sqlite3.Connection, book_id: str, para_id: int) -> dict[int, str]:
        rows = conn.execute(
            "SELECT line_id, content FROM nissaya "
            "WHERE book_id=? AND para_id=? ORDER BY line_id",
            (book_id, para_id),
        ).fetchall()
        return {r["line_id"]: (r["content"] or "") for r in rows}

    @staticmethod
    def _translit(text: str) -> str:
        _ROMAN = set("aāiīuūeokKgGcCjJṭḍnpPbBmMyrlLvsh")
        if not text or text[0] in _ROMAN or text[0].isupper():
            return text
        try:
            from aksharamukha import transliterate
            r = transliterate.process("autodetect", "IASTPali", text,
                                      post_options=["AnusvaratoNasalASTISO"])
            return r or text
        except Exception:
            return text

    @staticmethod
    def _format_nissaya_json(raw: str) -> str:
        import json
        if not raw:
            return "(none)"
        raw = raw.strip()
        if not raw.startswith("["):
            return raw
        try:
            entries = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
        parts = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            if "note" in e and "pali" not in e:
                parts.append(f"[Note: {e['note']}]")
                continue
            pali    = NissayaContext._translit(e.get("pali", ""))
            meaning = e.get("meaning", "")
            if pali and meaning:
                parts.append(f"{pali}: {meaning}")
            else:
                parts.append(pali or meaning)
        return " | ".join(p for p in parts if p) or "(none)"

    def build(self) -> str:
        import time
        if not self._paragraphs:
            return self._wrap("(no nissaya available)")

        path = _epitaka_path(self.params)
        parts = []

        t0 = time.perf_counter()
        with _connect(path) as conn:
            print(f"[DEBUG NissayaContext] DB connect: {time.perf_counter()-t0:.3f}s")

            for para in self._paragraphs:
                book_id  = para["book_id"]
                pid      = para["para_id"]

                t_niss0 = time.perf_counter()
                niss_map = self._fetch_nissaya_map(conn, book_id, pid)
                # print(f"[DEBUG NissayaContext] nissaya fetch para={pid}: {time.perf_counter()-t_niss0:.3f}s  ({len(niss_map)} rows)")

                # Skip paragraph entirely if no nissaya content exists for it
                if not niss_map:
                    continue

                section_lines = []
                for s in para["sentences"]:
                    lid      = s["line_id"]
                    raw_niss = niss_map.get(lid, "")
                    niss_fmt = self._format_nissaya_json(raw_niss)
                    en       = (s.get("english_translation") or "").strip()

                    # Skip lines with no nissaya content
                    if niss_fmt == "(none)":
                        continue

                    line = f"  [line_id={lid}] {niss_fmt}"
                    if en:
                        line += f"\n    Translation: {en}"
                    section_lines.append(line)

                if section_lines:
                    parts.append(f"--- para_id={pid} ---\n" + "\n".join(section_lines))

        print(f"[DEBUG NissayaContext] total build: {time.perf_counter()-t0:.3f}s  ({len(parts)} para(s) with content)")
        self.log_info(f"[NissayaContext] Built for {len(parts)} para(s) with nissaya content.")
        if not parts:
            return self._wrap("(no nissaya available)")
        return self._wrap("\n\n".join(parts))


# ══════════════════════════════════════════════════════════════════
# 2. GlossaryContext
# ══════════════════════════════════════════════════════════════════

class GlossaryContext(ContextBlock):
    label = "ESTABLISHED GLOSSARY (apply exactly, including multi-word phrases)"

    def __init__(self, params: dict, pali_text: str, *, max_n: int = 5,
                 log_info: _Log = _noop, log_warn: _Log = _noop):
        super().__init__(params, log_info, log_warn)
        self._pali_text = pali_text
        self._max_n     = max_n

    @staticmethod
    def extract_ngrams(text: str, max_n: int = 5) -> list[str]:
        tokens = [t.lower() for t in re.split(r"[\s,;.\u2018\u2019\"'()\[\]]+", text)
                  if len(t) > 1]
        ngrams: set[str] = set(tokens)
        for n in range(2, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.add(" ".join(tokens[i:i + n]))
        return list(ngrams)

    def build(self) -> str:
        import time
        t0 = time.perf_counter()
        ngrams = self.extract_ngrams(self._pali_text, self._max_n)
        print(f"[DEBUG GlossaryContext] extract_ngrams ({len(ngrams)} ngrams): {time.perf_counter()-t0:.3f}s")
        if not ngrams:
            return self._wrap("(no existing glossary entries)")

        try:
            t1 = time.perf_counter()
            conn = get_glossary_conn()
            conn.row_factory = sqlite3.Row
            print(f"[DEBUG GlossaryContext] glossary DB connect: {time.perf_counter()-t1:.3f}s")
            try:
                t2 = time.perf_counter()
                placeholders = ",".join("?" * len(ngrams))
                rows = conn.execute(
                    f"SELECT pali, english, context FROM glossary "
                    f"WHERE pali IN ({placeholders})",
                    ngrams,
                ).fetchall()
                print(f"[DEBUG GlossaryContext] glossary query ({len(rows)} hits): {time.perf_counter()-t2:.3f}s")
            finally:
                conn.close()
        except Exception as exc:
            self.log_warn(f"[GlossaryContext] Lookup failed: {exc}")
            return self._wrap("(glossary unavailable)")

        print(f"[DEBUG GlossaryContext] total build: {time.perf_counter()-t0:.3f}s")
        if not rows:
            return self._wrap("(no matching glossary entries yet)")

        self.log_info(f"[GlossaryContext] {len(rows)} term(s) found.")
        lines = []
        for r in rows:
            line = f"  {r['pali']} → {r['english']}"
            if r["context"]:
                line += f"  [{r['context']}]"
            lines.append(line)
        return self._wrap("\n".join(lines))


# ══════════════════════════════════════════════════════════════════
# 3. CommentaryContext
# ══════════════════════════════════════════════════════════════════

class CommentaryContext(ContextBlock):
    label = "PALI COMMENTARY & SUB-COMMENTARY"

    def __init__(self, params: dict, book_id: str, para_start: int, para_end: int = -1,
                 *, max_chars: int = 3000, log_info: _Log = _noop, log_warn: _Log = _noop):
        super().__init__(params, log_info, log_warn)
        self._book_id    = book_id
        self._para_start = para_start
        self._para_end   = para_end
        self._max_chars  = max_chars

    def _collect_src_lines(self, conn):
        rows = conn.execute(
            "SELECT para_id, line_id FROM sentences "
            "WHERE book_id=? AND para_id BETWEEN ? AND ? "
            "ORDER BY para_id, line_id",
            (self._book_id, self._para_start,
             self._para_start if self._para_end == -1 else self._para_end),
        ).fetchall()
        return [(self._book_id, r["para_id"], r["line_id"]) for r in rows]

    def _fetch_exact(self, conn, triples):
        if not triples:
            return []
        conn.execute("DROP TABLE IF EXISTS _cb_targets")
        conn.execute("CREATE TEMP TABLE _cb_targets (book_id TEXT, para_id INTEGER, line_id INTEGER)")
        conn.executemany("INSERT INTO _cb_targets VALUES (?,?,?)", triples)
        rows = conn.execute(
            "SELECT s.book_id, s.para_id, s.line_id, s.pali_sentence, s.english_translation "
            "FROM sentences s JOIN _cb_targets t "
            "ON s.book_id=t.book_id AND s.para_id=t.para_id AND s.line_id=t.line_id "
            "ORDER BY s.book_id, s.para_id, s.line_id"
        ).fetchall()
        conn.execute("DROP TABLE IF EXISTS _cb_targets")
        return rows

    def _fetch_paragraphs(self, conn, pairs):
        if not pairs:
            return []
        conn.execute("DROP TABLE IF EXISTS _cb_paras")
        conn.execute("CREATE TEMP TABLE _cb_paras (book_id TEXT, para_id INTEGER)")
        conn.executemany("INSERT INTO _cb_paras VALUES (?,?)", pairs)
        rows = conn.execute(
            "SELECT s.book_id, s.para_id, s.line_id, s.pali_sentence, s.english_translation "
            "FROM sentences s JOIN _cb_paras t "
            "ON s.book_id=t.book_id AND s.para_id=t.para_id "
            "ORDER BY s.book_id, s.para_id, s.line_id"
        ).fetchall()
        conn.execute("DROP TABLE IF EXISTS _cb_paras")
        return rows

    @staticmethod
    def _render(rows, label):
        if not rows:
            return ""
        sections: dict[tuple, list[str]] = {}
        for r in rows:
            pali = r["pali_sentence"] or ""
            en   = (r["english_translation"] or "").strip() if "english_translation" in r.keys() else ""
            line = f"  [{r['line_id']}] {pali}"
            if en:
                line += f"\n          EN: {en}"
            sections.setdefault((r["book_id"], r["para_id"]), []).append(line)
        parts = [label]
        for (book, para), lines in sections.items():
            parts.append(f"\n[{book} §{para}]")
            parts.extend(lines)
        return "\n".join(parts)

    def build(self) -> str:
        import time
        path = _epitaka_path(self.params)
        try:
            t0 = time.perf_counter()
            with _connect(path) as conn:
                print(f"[DEBUG CommentaryContext] DB connect: {time.perf_counter()-t0:.3f}s")

                if not conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='book_links'"
                ).fetchone():
                    self.log_warn("[CommentaryContext] book_links table not found.")
                    return self._wrap("(no commentary available)")

                para_end = self._para_start if self._para_end == -1 else self._para_end

                # ── A. Forward lookup: input is src → find dst (commentary / sub-commentary) ──
                #
                # The classic path: the input paragraph is a mūla/aṭṭhakathā text and we
                # want to pull the commentary paragraphs that annotate it.
                src_lines = self._collect_src_lines(conn)
                print(f"[DEBUG CommentaryContext] collect_src_lines ({len(src_lines)} rows): {time.perf_counter()-t0:.3f}s")

                forward_result = ""
                if src_lines:
                    t2s = time.perf_counter()
                    conn.execute("DROP TABLE IF EXISTS _cb_src")
                    conn.execute("CREATE TEMP TABLE _cb_src (book_id TEXT, para_id INTEGER, line_id INTEGER)")
                    conn.executemany("INSERT INTO _cb_src VALUES (?,?,?)", src_lines)
                    fwd_link_rows = conn.execute(
                        "SELECT bl.dst_book, bl.dst_para, bl.dst_line "
                        "FROM book_links bl JOIN _cb_src s "
                        "ON bl.src_book=s.book_id AND bl.src_para=s.para_id "
                        "AND bl.src_line=s.line_id "
                        "ORDER BY bl.dst_book, bl.dst_para, bl.dst_line"
                    ).fetchall()
                    conn.execute("DROP TABLE IF EXISTS _cb_src")
                    print(f"[DEBUG CommentaryContext] forward links ({len(fwd_link_rows)} links): {time.perf_counter()-t2s:.3f}s")

                    if fwd_link_rows:
                        seen: dict[tuple, None] = {}
                        for r in fwd_link_rows:
                            seen[(r["dst_book"], r["dst_para"], r["dst_line"])] = None
                        fwd_targets = list(seen.keys())

                        t3s = time.perf_counter()
                        unique_paras = list(dict.fromkeys((b, p) for b, p, _ in fwd_targets))
                        forward_result = self._render(
                            self._fetch_paragraphs(conn, unique_paras),
                            "[Commentary / Sub-commentary — full paragraphs]",
                        )
                        print(f"[DEBUG CommentaryContext] forward tier-1 ({len(forward_result)} chars): {time.perf_counter()-t3s:.3f}s")

                        if self._max_chars > 0 and len(forward_result) > self._max_chars:
                            t4s = time.perf_counter()
                            window: dict[tuple, None] = {}
                            for b, p, l in fwd_targets:
                                for off in (-1, 0, 1):
                                    window[(b, p, l + off)] = None
                            forward_result = self._render(
                                self._fetch_exact(conn, list(window.keys())),
                                "[Commentary / Sub-commentary — 3-line window]",
                            )
                            print(f"[DEBUG CommentaryContext] forward tier-2 ({len(forward_result)} chars): {time.perf_counter()-t4s:.3f}s")

                        if self._max_chars > 0 and len(forward_result) > self._max_chars:
                            t5s = time.perf_counter()
                            forward_result = self._render(
                                self._fetch_exact(conn, fwd_targets),
                                "[Commentary / Sub-commentary — linked lines only]",
                            )
                            print(f"[DEBUG CommentaryContext] forward tier-3 ({len(forward_result)} chars): {time.perf_counter()-t5s:.3f}s")

                # ── B. Reverse lookup: input is dst → find src (mūla / parent text) ──────
                #
                # If the input book/para is itself a commentary (i.e. it appears as a
                # dst_book/dst_para in book_links), walk back to the source paragraphs.
                # Surfacing the mūla with its existing English translation lets the
                # translator keep the wording consistent with what was already rendered
                # for the root text.
                t6s = time.perf_counter()
                rev_link_rows = conn.execute(
                    "SELECT DISTINCT bl.src_book, bl.src_para "
                    "FROM book_links bl "
                    "WHERE bl.dst_book=? AND bl.dst_para BETWEEN ? AND ? "
                    "ORDER BY bl.src_book, bl.src_para",
                    (self._book_id, self._para_start, para_end),
                ).fetchall()
                print(f"[DEBUG CommentaryContext] reverse links ({len(rev_link_rows)} src para(s)): {time.perf_counter()-t6s:.3f}s")

                mula_result = ""
                rev_pairs: list[tuple] = []
                if rev_link_rows:
                    rev_pairs = [(r["src_book"], r["src_para"]) for r in rev_link_rows]
                    t7s = time.perf_counter()
                    mula_rows = self._fetch_paragraphs(conn, rev_pairs)
                    mula_result = self._render(
                        mula_rows,
                        "[Mūla / Source Text (for translation consistency)]",
                    )
                    print(f"[DEBUG CommentaryContext] reverse fetch+render ({len(mula_result)} chars): {time.perf_counter()-t7s:.3f}s")

                # ── C. Sibling lookup: forward-links from the same mūla src paragraphs ──
                #
                # From the mūla paragraphs found in step B, resolve *their* forward links
                # to other dst books.  This surfaces sibling commentary paragraphs — e.g.
                # if the input is a ṭīkā, this pulls the aṭṭhakathā that comments on the
                # same mūla lines — giving the translator the full commentary stack.
                sibling_result = ""
                if rev_pairs:
                    t8s = time.perf_counter()
                    conn.execute("DROP TABLE IF EXISTS _cb_rev_src")
                    conn.execute("CREATE TEMP TABLE _cb_rev_src (book_id TEXT, para_id INTEGER)")
                    conn.executemany("INSERT INTO _cb_rev_src VALUES (?,?)", rev_pairs)
                    sibling_link_rows = conn.execute(
                        "SELECT DISTINCT bl.dst_book, bl.dst_para "
                        "FROM book_links bl JOIN _cb_rev_src r "
                        "ON bl.src_book=r.book_id AND bl.src_para=r.para_id "
                        # Exclude the input book itself — already covered by forward_result
                        "WHERE bl.dst_book != ? "
                        "ORDER BY bl.dst_book, bl.dst_para",
                        (self._book_id,),
                    ).fetchall()
                    conn.execute("DROP TABLE IF EXISTS _cb_rev_src")
                    print(f"[DEBUG CommentaryContext] sibling links ({len(sibling_link_rows)} para(s)): {time.perf_counter()-t8s:.3f}s")

                    if sibling_link_rows:
                        sib_pairs = [(r["dst_book"], r["dst_para"]) for r in sibling_link_rows]
                        t9s = time.perf_counter()
                        sib_rows = self._fetch_paragraphs(conn, sib_pairs)
                        sibling_result = self._render(
                            sib_rows,
                            "[Sibling Commentary (shares same mūla source)]",
                        )
                        print(f"[DEBUG CommentaryContext] sibling fetch+render ({len(sibling_result)} chars): {time.perf_counter()-t9s:.3f}s")

            # ── Assemble final output ─────────────────────────────────────────────────
            sections = [s for s in (forward_result, mula_result, sibling_result) if s.strip()]
            print(f"[DEBUG CommentaryContext] total build: {time.perf_counter()-t0:.3f}s  ({len(sections)} section(s))")

            if not sections:
                self.log_info("[CommentaryContext] No commentary, mūla, or sibling content found.")
                return self._wrap("(no commentary available)")

            self.log_info(
                f"[CommentaryContext] Built {len(sections)} section(s): "
                + ", ".join([
                    name for name, val in (
                        ("forward", forward_result),
                        ("mūla",    mula_result),
                        ("sibling", sibling_result),
                    ) if val.strip()
                ])
            )
            return self._wrap("\n\n".join(sections))

        except Exception as exc:
            self.log_warn(f"[CommentaryContext] Failed: {exc}")
            return self._wrap("(no commentary available)")


# ══════════════════════════════════════════════════════════════════
# 4. PaliDefsContext
# ══════════════════════════════════════════════════════════════════

class PaliDefsContext(ContextBlock):
    label = "PALI WORD DEFINITIONS (reference for difficult/rare terms)"

    def __init__(self, params: dict, pali_text: str = "", *, book_id: str = "",
                 para_start: int = 0, para_end: int = -1, max_words: int = 30,
                 log_info: _Log = _noop, log_warn: _Log = _noop):
        super().__init__(params, log_info, log_warn)
        self._pali_text  = pali_text
        self._book_id    = book_id
        self._para_start = para_start
        self._para_end   = para_end
        self._max_words  = max_words

    def _resolve_text(self, conn):
        if self._pali_text:
            return self._pali_text
        if not self._book_id:
            return ""
        para_end = self._para_start if self._para_end == -1 else self._para_end
        rows = conn.execute(
            "SELECT pali_sentence FROM sentences "
            "WHERE book_id=? AND para_id BETWEEN ? AND ? "
            "ORDER BY para_id, line_id",
            (self._book_id, self._para_start, para_end),
        ).fetchall()
        return "\n".join(r["pali_sentence"] or "" for r in rows)

    # _get_usages kept for external callers; not used by build() any more
    def _get_usages(self, conn, stem):
        row = conn.execute(
            "SELECT book_id, para_id, line_id FROM pali_definition WHERE stem=? LIMIT 1",
            (stem,),
        ).fetchone()
        if not row:
            return []
        ctx = conn.execute(
            "SELECT pali_sentence, english_translation FROM sentences "
            "WHERE book_id=? AND para_id=? AND line_id BETWEEN ? AND ? "
            "ORDER BY line_id",
            (row["book_id"], row["para_id"], row["line_id"] - 1, row["line_id"] + 1),
        ).fetchall()
        results = []
        for r in ctx:
            if not r["pali_sentence"]:
                continue
            en = (r["english_translation"] or "").strip()
            entry = r["pali_sentence"]
            if en:
                entry += f" [{en}]"
            results.append(entry)
        return results

    def build(self) -> str:
        import time
        path = _epitaka_path(self.params)
        try:
            t0 = time.perf_counter()
            with _connect(path) as conn:
                print(f"[DEBUG PaliDefsContext] DB connect: {time.perf_counter()-t0:.3f}s")

                if not conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pali_definition'"
                ).fetchone():
                    self.log_warn("[PaliDefsContext] pali_definition table not found.")
                    return self._wrap("(no word definitions available)")

                text = self._resolve_text(conn)
                if not text:
                    return self._wrap("(no word definitions available)")

                words = list({w.lower() for w in re.findall(r"\b[\w\u0900-\u097F]{5,}\b", text)})
                capped = words[:self._max_words]
                print(f"[DEBUG PaliDefsContext] {len(words)} candidate words (using {len(capped)})")

                # ── Batch query 1: resolve word/plain → (word, stem) in one shot ──
                t1 = time.perf_counter()
                ph = ",".join("?" * len(capped))
                stem_rows = conn.execute(
                    f"SELECT plain AS matched, stem FROM pali_definition WHERE plain IN ({ph}) "
                    f"UNION SELECT word  AS matched, stem FROM pali_definition WHERE word  IN ({ph})",
                    capped + capped,
                ).fetchall()
                word_to_stem: dict[str, str] = {}
                for r in stem_rows:
                    word_to_stem.setdefault(r["matched"], r["stem"])
                print(f"[DEBUG PaliDefsContext] batch stem lookup ({len(word_to_stem)} matches): {time.perf_counter()-t1:.3f}s")

                if not word_to_stem:
                    print(f"[DEBUG PaliDefsContext] total build: {time.perf_counter()-t0:.3f}s")
                    return self._wrap("(no word definitions found)")

                # ── Batch query 2: one location row per stem ──
                t2 = time.perf_counter()
                stems = list(set(word_to_stem.values()))
                ph2 = ",".join("?" * len(stems))
                loc_rows = conn.execute(
                    f"SELECT stem, book_id, para_id, line_id "
                    f"FROM pali_definition WHERE stem IN ({ph2}) "
                    f"GROUP BY stem",
                    stems,
                ).fetchall()
                stem_to_loc: dict[str, tuple] = {
                    r["stem"]: (r["book_id"], r["para_id"], r["line_id"])
                    for r in loc_rows
                }
                print(f"[DEBUG PaliDefsContext] batch location lookup ({len(stem_to_loc)} stems): {time.perf_counter()-t2:.3f}s")

                # ── Batch query 3: fetch context sentences for all locations ──
                t3 = time.perf_counter()
                conn.execute("DROP TABLE IF EXISTS _pd_locs")
                conn.execute(
                    "CREATE TEMP TABLE _pd_locs "
                    "(stem TEXT, book_id TEXT, para_id INTEGER, lo INTEGER, hi INTEGER)"
                )
                conn.executemany(
                    "INSERT INTO _pd_locs VALUES (?,?,?,?,?)",
                    [
                        (stem, book_id, para_id, line_id - 1, line_id + 1)
                        for stem, (book_id, para_id, line_id) in stem_to_loc.items()
                    ],
                )
                ctx_rows = conn.execute(
                    "SELECT l.stem, s.pali_sentence, s.english_translation "
                    "FROM sentences s "
                    "JOIN _pd_locs l "
                    "ON s.book_id=l.book_id AND s.para_id=l.para_id "
                    "AND s.line_id BETWEEN l.lo AND l.hi "
                    "WHERE s.pali_sentence IS NOT NULL AND s.pali_sentence != \'\' "
                    "ORDER BY l.stem, s.line_id"
                ).fetchall()
                conn.execute("DROP TABLE IF EXISTS _pd_locs")

                from collections import defaultdict
                stem_ctx: dict[str, list[str]] = defaultdict(list)
                for r in ctx_rows:
                    en    = (r["english_translation"] or "").strip()
                    entry = r["pali_sentence"]
                    if en:
                        entry += f" [{en}]"
                    stem_ctx[r["stem"]].append(entry)
                print(f"[DEBUG PaliDefsContext] batch context fetch ({len(ctx_rows)} rows): {time.perf_counter()-t3:.3f}s")

            # ── Assemble output ──
            defs: list[str] = []
            for word in capped:
                stem = word_to_stem.get(word)
                if not stem:
                    continue
                usages = stem_ctx.get(stem, [])
                if usages:
                    defs.append(f"  {word} (stem: {stem}): " + " … ".join(usages))

            print(f"[DEBUG PaliDefsContext] total build: {time.perf_counter()-t0:.3f}s  ({len(defs)} defs)")
            if not defs:
                return self._wrap("(no word definitions found)")

            self.log_info(f"[PaliDefsContext] {len(defs)} definition(s).")
            return self._wrap("Pali Word Definitions:\n" + "\n".join(defs))

        except Exception as exc:
            self.log_warn(f"[PaliDefsContext] Failed: {exc}")
            return self._wrap("(word definitions unavailable)")


# ══════════════════════════════════════════════════════════════════
# 5. PreviousTranslationContext
# ══════════════════════════════════════════════════════════════════

class PreviousTranslationContext(ContextBlock):
    """
    Walk backwards from para_start looking for paragraphs that contain at least
    one translated line.  The search stops as soon as accumulated English text
    reaches ``min_length`` chars, or after scanning ``max_lookback`` rows
    (across all paragraphs), whichever comes first.

    If no translated text is found within the lookback window the block returns
    ``(no previous paragraph translated)`` instead of filling the prompt with
    untranslated Pali.

    Each collected line renders as:
        [line_id=N] Pali: ...
                    EN:   ...          (omitted when not yet translated)
    """

    label = "PREVIOUS PARAGRAPH (for style consistency)"

    def __init__(
        self,
        params:       dict,
        book_id:      str,
        para_start:   int,
        *,
        min_length:   int  = 200,
        max_lookback: int  = 500,
        log_info:     _Log = _noop,
        log_warn:     _Log = _noop,
    ):
        super().__init__(params, log_info, log_warn)
        self._book_id      = book_id
        self._para_start   = para_start
        self._min_length   = min_length
        self._max_lookback = max_lookback

    def build(self) -> str:
        import time
        if self._para_start <= 0:
            return self._wrap("(no previous paragraph translated)")

        path = _epitaka_path(self.params)
        try:
            t0 = time.perf_counter()
            with _connect(path) as conn:
                print(f"[DEBUG PreviousTranslationContext] DB connect: {time.perf_counter()-t0:.3f}s")

                # Single batch fetch: grab up to max_lookback rows preceding
                # para_start, newest-first so we can stop collecting early.
                t1 = time.perf_counter()
                candidate_rows = conn.execute(
                    "SELECT para_id, line_id, pali_sentence, english_translation "
                    "FROM sentences "
                    "WHERE book_id = ? AND para_id < ? "
                    "ORDER BY para_id DESC, line_id DESC "
                    "LIMIT ?",
                    (self._book_id, self._para_start, self._max_lookback),
                ).fetchall()
                print(f"[DEBUG PreviousTranslationContext] batch fetch "
                      f"({len(candidate_rows)} rows): {time.perf_counter()-t1:.3f}s")

            # Group rows by para_id; restore ascending line order inside each para.
            para_map: dict[int, list[dict]] = {}
            for r in candidate_rows:
                para_map.setdefault(r["para_id"], []).append(dict(r))
            for rows in para_map.values():
                rows.sort(key=lambda x: x["line_id"])

            # Walk paragraphs newest-to-oldest; collect only those that have at
            # least one translated line; stop once min_length EN chars gathered.
            collected: list[tuple[int, list[dict]]] = []   # oldest first
            translated_chars = 0

            for pid in sorted(para_map.keys(), reverse=True):
                rows = para_map[pid]
                para_en_chars = sum(
                    len((r["english_translation"] or "").strip())
                    for r in rows
                )
                # Skip paragraphs with no translation at all.
                if para_en_chars == 0:
                    continue

                collected.insert(0, (pid, rows))
                translated_chars += para_en_chars

                if translated_chars >= self._min_length:
                    break

            print(f"[DEBUG PreviousTranslationContext] "
                  f"scanned {len(para_map)} para(s), "
                  f"kept {len(collected)} translated para(s), "
                  f"{translated_chars} EN chars | "
                  f"total: {time.perf_counter()-t0:.3f}s")

            if not collected:
                self.log_info(
                    f"[PreviousTranslationContext] No translated paragraphs found "
                    f"within {self._max_lookback} rows before para {self._para_start}."
                )
                return self._wrap("(no previous paragraph translated)")

            # Render: show Pali for every line; EN only when present.
            parts = []
            for pid, sentences in collected:
                lines = [f"[Para {pid}]"]
                for s in sentences:
                    pali = (s.get("pali_sentence") or "").strip()
                    en   = (s.get("english_translation") or "").strip()
                    line = f"  [line_id={s['line_id']}] Pali: {pali}"
                    if en:
                        line += f"\n                   EN:   {en}"
                    lines.append(line)
                parts.append("\n".join(lines))

            self.log_info(
                f"[PreviousTranslationContext] {len(collected)} para(s), "
                f"{translated_chars} translated chars collected."
            )
            return self._wrap("\n\n".join(parts))

        except Exception as exc:
            self.log_warn(f"[PreviousTranslationContext] Failed: {exc}")
            return self._wrap("(no previous paragraph translated)")


# ══════════════════════════════════════════════════════════════════
# 6. MulaAtthaContext
# ══════════════════════════════════════════════════════════════════

class MulaAtthaContext(ContextBlock):
    label = "TRANSLATED MŪLA / AṬṬHAKATHĀ / TĪKĀ REFERENCES"

    def __init__(self, params: dict, book_id: str, para_start: int, para_end: int = -1,
                 *, log_info: _Log = _noop, log_warn: _Log = _noop):
        super().__init__(params, log_info, log_warn)
        self._book_id    = book_id
        self._para_start = para_start
        self._para_end   = para_end

    def build(self) -> str:
        path = _epitaka_path(self.params)
        para_end = self._para_start if self._para_end == -1 else self._para_end

        try:
            with _connect(path) as conn:
                if not conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='book_links'"
                ).fetchone():
                    return self._wrap("(no linked translations available)")

                src_rows = conn.execute(
                    "SELECT para_id, line_id FROM sentences "
                    "WHERE book_id=? AND para_id BETWEEN ? AND ? "
                    "ORDER BY para_id, line_id",
                    (self._book_id, self._para_start, para_end),
                ).fetchall()

                if not src_rows:
                    return self._wrap("(no linked translations available)")

                src_lines = [(self._book_id, r["para_id"], r["line_id"]) for r in src_rows]

                conn.execute("DROP TABLE IF EXISTS _mat_src")
                conn.execute("CREATE TEMP TABLE _mat_src (book_id TEXT, para_id INTEGER, line_id INTEGER)")
                conn.executemany("INSERT INTO _mat_src VALUES (?,?,?)", src_lines)
                link_rows = conn.execute(
                    "SELECT DISTINCT bl.dst_book, bl.dst_para "
                    "FROM book_links bl JOIN _mat_src s "
                    "ON bl.src_book=s.book_id AND bl.src_para=s.para_id "
                    "AND bl.src_line=s.line_id "
                    "ORDER BY bl.dst_book, bl.dst_para"
                ).fetchall()
                conn.execute("DROP TABLE IF EXISTS _mat_src")

                if not link_rows:
                    return self._wrap("(no linked translations available)")

                blocks: list[str] = []
                for r in link_rows:
                    dst_book = r["dst_book"]
                    dst_para = r["dst_para"]
                    sent_rows = conn.execute(
                        "SELECT line_id, pali_sentence, english_translation "
                        "FROM sentences "
                        "WHERE book_id=? AND para_id=? "
                        "AND english_translation IS NOT NULL "
                        "AND english_translation != '' "
                        "ORDER BY line_id",
                        (dst_book, dst_para),
                    ).fetchall()

                    if not sent_rows:
                        continue

                    lines = [f"[{dst_book} §{dst_para}]"]
                    for sr in sent_rows:
                        lines.append(
                            f"  [{sr['line_id']}] Pāli: {sr['pali_sentence'] or ''}\n"
                            f"          EN:   {sr['english_translation']}"
                        )
                    blocks.append("\n".join(lines))

                if not blocks:
                    return self._wrap("(no linked translations available)")

                self.log_info(f"[MulaAtthaContext] {len(blocks)} linked para(s) with translations.")
                return self._wrap("\n\n".join(blocks))

        except Exception as exc:
            self.log_warn(f"[MulaAtthaContext] Failed: {exc}")
            return self._wrap("(no linked translations available)")


# ══════════════════════════════════════════════════════════════════
# 7. ParallelTranslationContext
# ══════════════════════════════════════════════════════════════════

# Human-readable labels for each known epitaka_*.db file.
# Add new languages here as more databases become available.
_EPITAKA_LABELS: dict[str, str] = {
    "epitaka_si.db":      "Sinhala translation",
    "epitaka_th.db":      "Thai translation",
    "epitaka_en-book.db": "English translation (human, book sources)",
}


class ParallelTranslationContext(ContextBlock):
    """
    Scans the same data folder as the nissaya DB for sibling files matching
    ``epitaka_*.db``.  For each file found it queries::

        SELECT translation FROM sentences
        WHERE book_id = :book_id
          AND para_id = :para_id
          AND line_id = :line_id

    and renders every hit as a labelled parallel-translation block so the AI
    translator can consult existing human renderings for the same passage.

    Parameters
    ----------
    params      : job params dict (must contain or imply ``epitaka_db`` path).
    book_id     : source book identifier.
    para_start  : first paragraph to fetch.
    para_end    : last paragraph (inclusive); -1 means same as para_start.
    db_dir      : optional explicit directory to scan; if omitted the parent
                  directory of the nissaya DB is used.
    only_langs  : optional whitelist of filename stems, e.g.
                  ``["epitaka_si.db", "epitaka_en-book.db"]``.
                  When omitted every ``epitaka_*.db`` file found is included.
    """

    label = "PARALLEL HUMAN TRANSLATIONS (for reference)"

    def __init__(
        self,
        params:     dict,
        book_id:    str,
        para_start: int,
        para_end:   int  = -1,
        *,
        db_dir:     str | None        = None,
        only_langs: list[str] | None  = None,
        log_info:   _Log = _noop,
        log_warn:   _Log = _noop,
    ):
        super().__init__(params, log_info, log_warn)
        self._book_id    = book_id
        self._para_start = para_start
        self._para_end   = para_end
        self._db_dir     = db_dir
        self._only_langs = set(only_langs) if only_langs else None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _data_dir(self) -> "Path":
        from pathlib import Path
        if self._db_dir:
            return Path(self._db_dir)
        nissaya = Path(_epitaka_path(self.params))
        return nissaya.parent

    def _discover_dbs(self) -> list[tuple[str, "Path"]]:
        """Return [(filename, full_path), …] for every epitaka_*.db in the folder."""
        from pathlib import Path
        data_dir = self._data_dir()
        found: list[tuple[str, Path]] = []
        for p in sorted(data_dir.glob("epitaka_*.db")):
            fname = p.name
            if self._only_langs and fname not in self._only_langs:
                continue
            found.append((fname, p))
        return found

    @staticmethod
    def _lang_label(filename: str) -> str:
        return _EPITAKA_LABELS.get(filename, filename)

    def _fetch_translations(
        self, path: "Path", para_end: int
    ) -> list[dict]:
        """
        Query one epitaka DB for all (para_id, line_id, translation) rows in the
        requested para range.  Returns an empty list when the DB or column does
        not exist rather than raising.
        """
        try:
            with _connect(str(path)) as conn:
                # Guard: some builds may not have a 'translation' column yet
                cols = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(sentences)").fetchall()
                }
                if "translation" not in cols:
                    return []

                rows = conn.execute(
                    "SELECT para_id, line_id, translation "
                    "FROM sentences "
                    "WHERE book_id = ? "
                    "  AND para_id BETWEEN ? AND ? "
                    "  AND translation IS NOT NULL "
                    "  AND translation != '' "
                    "ORDER BY para_id, line_id",
                    (self._book_id, self._para_start, para_end),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            # Non-fatal: log and skip this DB
            return []

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self) -> str:
        import time
        t0 = time.perf_counter()

        para_end = self._para_start if self._para_end == -1 else self._para_end

        db_files = self._discover_dbs()
        print(f"[DEBUG ParallelTranslationContext] found {len(db_files)} epitaka DB(s): "
              f"{[f for f, _ in db_files]}")

        if not db_files:
            return self._wrap("(no parallel translation databases found)")

        all_blocks: list[str] = []

        for filename, db_path in db_files:
            label = self._lang_label(filename)
            t1 = time.perf_counter()
            rows = self._fetch_translations(db_path, para_end)
            print(f"[DEBUG ParallelTranslationContext] {filename}: "
                  f"{len(rows)} row(s) in {time.perf_counter()-t1:.3f}s")

            if not rows:
                self.log_info(f"[ParallelTranslationContext] {filename}: no translations for "
                              f"book={self._book_id} para={self._para_start}–{para_end}")
                continue

            # Group by para_id for tidy output
            para_groups: dict[int, list[dict]] = {}
            for r in rows:
                para_groups.setdefault(r["para_id"], []).append(r)

            section_lines = [f"[{label}]"]
            for pid in sorted(para_groups):
                section_lines.append(f"  --- para_id={pid} ---")
                for r in para_groups[pid]:
                    section_lines.append(
                        f"    [line_id={r['line_id']}] {r['translation']}"
                    )
            all_blocks.append("\n".join(section_lines))

        print(f"[DEBUG ParallelTranslationContext] total build: "
              f"{time.perf_counter()-t0:.3f}s  ({len(all_blocks)} source(s) with content)")

        if not all_blocks:
            self.log_info("[ParallelTranslationContext] No parallel translations found.")
            return self._wrap("(no parallel translations found)")

        self.log_info(f"[ParallelTranslationContext] {len(all_blocks)} translation source(s) included.")
        return self._wrap("\n\n".join(all_blocks))


# ══════════════════════════════════════════════════════════════════
# 8. TranslationWriter
# ══════════════════════════════════════════════════════════════════

class TranslationWriter:
    def __init__(self, params: dict, log_info: _Log = _noop,
                 log_warn: _Log = _noop, log_error: _Log = _noop):
        self.params    = params
        self.log_info  = log_info
        self.log_warn  = log_warn
        self.log_error = log_error

    def save(self, book_id: str, para_id: int, translations: list[dict]) -> int:
        updated = 0
        with _connect(_epitaka_path(self.params)) as conn:
            for entry in translations:
                line_id = entry.get("line_id")
                text    = str(entry.get("english_translation") or "").strip()
                if line_id is None or not text:
                    self.log_warn(f"[TranslationWriter] Skipping bad entry: {entry}")
                    continue
                try:
                    conn.execute(
                        "UPDATE sentences SET english_translation=? "
                        "WHERE book_id=? AND para_id=? AND line_id=?",
                        (text, book_id, para_id, line_id),
                    )
                    updated += conn.execute("SELECT changes()").fetchone()[0]
                except Exception as exc:
                    self.log_error(f"[TranslationWriter] Error on line_id={line_id}: {exc}")
            conn.commit()

        # self.log_info(f"[TranslationWriter] {updated} row(s) updated (book={book_id}, para={para_id}).")
        return updated


# ══════════════════════════════════════════════════════════════════
# 9. GlossaryWriter
# ══════════════════════════════════════════════════════════════════

class GlossaryWriter:
    def __init__(self, log_info: _Log = _noop, log_warn: _Log = _noop, log_error: _Log = _noop):
        self.log_info  = log_info
        self.log_warn  = log_warn
        self.log_error = log_error

    def upsert(self, terms: list[dict], sc_id: str = "") -> int:
        required = {"pali", "english"}
        inserted = 0

        try:
            conn = get_glossary_conn()
        except Exception as exc:
            self.log_warn(f"[GlossaryWriter] Cannot open glossary DB: {exc}")
            return 0

        try:
            for term in terms:
                if not required.issubset(term):
                    self.log_warn(f"[GlossaryWriter] Skipping bad entry: {term}")
                    continue
                pali    = str(term.get("pali",    "")).strip()
                english = str(term.get("english", "")).strip()
                if not pali or not english:
                    continue
                try:
                    conn.execute(
                        "INSERT INTO glossary "
                        "(pali, english, domain, sub_domain, context, note, source_id) "
                        "VALUES (?,?,?,?,?,?,?) "
                        "ON CONFLICT(pali, english) DO NOTHING",
                        (
                            pali, english,
                            str(term.get("domain",     "") or ""),
                            str(term.get("sub_domain", "") or ""),
                            str(term.get("context",    "") or ""),
                            str(term.get("note",       "") or ""),
                            sc_id,
                        ),
                    )
                    inserted += conn.execute("SELECT changes()").fetchone()[0]
                except Exception as exc:
                    self.log_error(f"[GlossaryWriter] Insert error for '{pali}': {exc}")
            conn.commit()
        finally:
            conn.close()

        self.log_info(f"[GlossaryWriter] {inserted} new term(s) inserted (sc_id={sc_id!r}).")
        return inserted


# ══════════════════════════════════════════════════════════════════
# 10. RemarkWriter
# ══════════════════════════════════════════════════════════════════

class RemarkWriter:
    """
    Persists "translation remarks" — short notes the AI raises when the
    chosen English translation conflicts with one of the parallel human
    translations (Sinhala / Thai / English-book) or with the Pāli
    commentary.  These are meant to be rare: only genuinely useful notes
    should be saved, never a remark for every sentence.

    Table (created on first use, in the same DB as `sentences`):

        CREATE TABLE translation_remarks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id     TEXT NOT NULL,
            para_id     INTEGER NOT NULL,
            line_id     INTEGER NOT NULL,
            pali        TEXT,
            translation TEXT,
            conflict    TEXT,
            note        TEXT,
            source_id   TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """

    _DDL = """
        CREATE TABLE IF NOT EXISTS translation_remarks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id     TEXT NOT NULL,
            para_id     INTEGER NOT NULL,
            line_id     INTEGER NOT NULL,
            pali        TEXT,
            translation TEXT,
            conflict    TEXT,
            note        TEXT,
            source_id   TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """

    def __init__(self, params: dict, log_info: _Log = _noop,
                 log_warn: _Log = _noop, log_error: _Log = _noop):
        self.params    = params
        self.log_info  = log_info
        self.log_warn  = log_warn
        self.log_error = log_error

    def save(self, book_id: str, remarks: list[dict], source_id: str = "") -> int:
        """
        Each remark dict may contain:
            para_id, line_id, pali, translation, conflict, note
        `para_id` / `line_id` are required; everything else is optional.
        Returns the number of rows inserted.
        """
        if not remarks:
            return 0

        inserted = 0
        with _connect(_epitaka_path(self.params)) as conn:
            conn.execute(self._DDL)
            for r in remarks:
                para_id = r.get("para_id")
                line_id = r.get("line_id")
                if para_id is None or line_id is None:
                    self.log_warn(f"[RemarkWriter] Skipping bad remark (missing ids): {r}")
                    continue
                try:
                    conn.execute(
                        "INSERT INTO translation_remarks "
                        "(book_id, para_id, line_id, pali, translation, conflict, note, source_id) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (
                            book_id, para_id, line_id,
                            str(r.get("pali", "") or ""),
                            str(r.get("translation", "") or ""),
                            str(r.get("conflict", "") or ""),
                            str(r.get("note", "") or ""),
                            source_id,
                        ),
                    )
                    inserted += 1
                except Exception as exc:
                    self.log_error(f"[RemarkWriter] Insert error for para={para_id} line={line_id}: {exc}")
            conn.commit()

        self.log_info(f"[RemarkWriter] {inserted} remark(s) saved (book={book_id}).")
        return inserted