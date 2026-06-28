"""
tests/test_context_builders_real.py  —  Integration tests against real databases.

Set the two path constants below before running.  All tests are READ-ONLY
(no INSERT / UPDATE / DELETE).  Writer classes are also tested but only in
dry-run mode (they use in-memory DBs so nothing touches disk).

Run:
    pytest tests/test_context_builders_real.py -v
or just:
    python tests/test_context_builders_real.py
"""

import sqlite3
import sys
import types
import unittest

# ══════════════════════════════════════════════════════════════════
# ▶▶  SET THESE BEFORE RUNNING  ◀◀
# ══════════════════════════════════════════════════════════════════
NISSAYA_DB  = "/Volumes/Data/code/translator/epitaka_translator/data/nissaya.db"
GLOSSARY_DB = "/Volumes/Data/code/translator/epitaka_translator/data/glossary.db"
# ══════════════════════════════════════════════════════════════════

# One sample location to anchor the tests — pick any heading that has:
#   • nissaya rows
#   • at least one translated sentence  (for PreviousTranslationContext)
# Adjust to a real book_id / para_id from your data.
SAMPLE_BOOK_ID  = "Sp-i"
SAMPLE_PARA     = 20      # para_start  (must have sentences in DB)
SAMPLE_PARA_END = 30      # para_end    (inclusive)


# ══════════════════════════════════════════════════════════════════
# Stubs  (replace `database` and `config` modules)
# ══════════════════════════════════════════════════════════════════

def _install_stubs():
    db_mod = types.ModuleType("database")
    db_mod.get_glossary_conn = lambda: sqlite3.connect(GLOSSARY_DB, timeout=30)
    sys.modules["database"] = db_mod

    cfg_mod = types.ModuleType("config")
    cfg_mod.NISSAYA_DB  = NISSAYA_DB
    cfg_mod.SC_DATA_DB  = ""
    sys.modules["config"] = cfg_mod

_install_stubs()
import context_builders as cb   # noqa: E402  (must come after stubs)


# ══════════════════════════════════════════════════════════════════
# Helper — open real nissaya.db read-only
# ══════════════════════════════════════════════════════════════════

def _open_nissaya() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{NISSAYA_DB}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


# params dict that points at the real DB
PARAMS = {"nissaya_db": NISSAYA_DB}


# ══════════════════════════════════════════════════════════════════
# Sanity checks — abort early if DB is unreachable / misconfigured
# ══════════════════════════════════════════════════════════════════

class TestDBSanity(unittest.TestCase):
    """Fails fast with a clear message if the DB paths are wrong."""

    def test_nissaya_db_reachable(self):
        try:
            conn = _open_nissaya()
            tables = {r[0] for r in
                      conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                      .fetchall()}
            conn.close()
        except Exception as exc:
            self.fail(
                f"Cannot open nissaya.db at {NISSAYA_DB!r}: {exc}\n"
                "Set NISSAYA_DB at the top of this file."
            )
        for t in ("sentences", "nissaya", "book_links", "pali_definition"):
            self.assertIn(t, tables, f"Expected table '{t}' not found in nissaya.db")

    def test_glossary_db_reachable(self):
        try:
            conn = sqlite3.connect(GLOSSARY_DB, timeout=30)
            tables = {r[0] for r in
                      conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                      .fetchall()}
            conn.close()
        except Exception as exc:
            self.fail(
                f"Cannot open glossary.db at {GLOSSARY_DB!r}: {exc}\n"
                "Set GLOSSARY_DB at the top of this file."
            )
        self.assertIn("glossary", tables, "Expected table 'glossary' not found")

    def test_sample_sentences_exist(self):
        """Confirm the SAMPLE_* constants point at real data."""
        conn = _open_nissaya()
        row = conn.execute(
            "SELECT COUNT(*) FROM sentences "
            "WHERE book_id=? AND para_id BETWEEN ? AND ?",
            (SAMPLE_BOOK_ID, SAMPLE_PARA, SAMPLE_PARA_END),
        ).fetchone()
        conn.close()
        self.assertGreater(
            row[0], 0,
            f"No sentences found for book_id={SAMPLE_BOOK_ID!r} "
            f"para {SAMPLE_PARA}–{SAMPLE_PARA_END}. "
            "Adjust SAMPLE_BOOK_ID / SAMPLE_PARA at the top of this file."
        )


# ══════════════════════════════════════════════════════════════════
# 1. NissayaContext
# ══════════════════════════════════════════════════════════════════

class TestNissayaContextReal(unittest.TestCase):

    def _paragraphs(self) -> list[dict]:
        """Fetch real sentences for the sample range."""
        conn = _open_nissaya()
        result = []
        for pid in range(SAMPLE_PARA, SAMPLE_PARA_END + 1):
            rows = conn.execute(
                "SELECT line_id, pali_sentence FROM sentences "
                "WHERE book_id=? AND para_id=? ORDER BY line_id",
                (SAMPLE_BOOK_ID, pid),
            ).fetchall()
            if rows:
                result.append({
                    "book_id":   SAMPLE_BOOK_ID,
                    "para_id":   pid,
                    "sentences": [dict(r) for r in rows],
                })
        conn.close()
        return result

    def test_build_from_paragraphs(self):
        paras = self._paragraphs()
        self.assertTrue(paras, "No paragraphs fetched — check SAMPLE_* constants.")
        ctx = cb.NissayaContext(PARAMS, paragraphs=paras)
        result = ctx.build()

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        # Should contain section header and at least one para marker
        self.assertIn(f"para_id={SAMPLE_PARA}", result)
        print(f"\n[NissayaContext] build_from_paragraphs — {len(result)} chars")
        print(result[:600])

    def test_build_by_range(self):
        ctx = cb.NissayaContext(
            PARAMS, book_id=SAMPLE_BOOK_ID,
            para_start=SAMPLE_PARA, para_end=SAMPLE_PARA_END,
        )
        result = ctx.build()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        print(f"\n[NissayaContext] build_by_range — {len(result)} chars")

    def test_label_in_output(self):
        paras = self._paragraphs()
        ctx = cb.NissayaContext(PARAMS, paragraphs=paras)
        result = ctx.build()
        self.assertIn("NISSAYA", result)


# ══════════════════════════════════════════════════════════════════
# 2. GlossaryContext
# ══════════════════════════════════════════════════════════════════

class TestGlossaryContextReal(unittest.TestCase):

    def _sample_pali(self) -> str:
        conn = _open_nissaya()
        rows = conn.execute(
            "SELECT pali_sentence FROM sentences "
            "WHERE book_id=? AND para_id BETWEEN ? AND ? "
            "ORDER BY para_id, line_id",
            (SAMPLE_BOOK_ID, SAMPLE_PARA, SAMPLE_PARA_END),
        ).fetchall()
        conn.close()
        return "\n".join(r["pali_sentence"] or "" for r in rows)

    def test_returns_string(self):
        pali = self._sample_pali()
        self.assertTrue(pali.strip(), "No pali text found.")
        ctx = cb.GlossaryContext(PARAMS, pali)
        result = ctx.build()
        self.assertIsInstance(result, str)
        print(f"\n[GlossaryContext] {len(result)} chars")
        print(result[:400])

    def test_ngrams_cover_text(self):
        pali = self._sample_pali()
        ngrams = cb.GlossaryContext.extract_ngrams(pali)
        # Should contain some tokens
        self.assertGreater(len(ngrams), 0)
        # Every token should be lowercase
        for ng in ngrams:
            self.assertEqual(ng, ng.lower(), f"n-gram not lowercase: {ng!r}")

    def test_glossary_label_present(self):
        pali = self._sample_pali()
        ctx = cb.GlossaryContext(PARAMS, pali)
        result = ctx.build()
        self.assertIn("GLOSSARY", result)


# ══════════════════════════════════════════════════════════════════
# 3. CommentaryContext
# ══════════════════════════════════════════════════════════════════

class TestCommentaryContextReal(unittest.TestCase):

    def test_returns_string(self):
        ctx = cb.CommentaryContext(
            PARAMS, SAMPLE_BOOK_ID, SAMPLE_PARA, SAMPLE_PARA_END,
            max_chars=3000,
        )
        result = ctx.build()
        self.assertIsInstance(result, str)
        print(f"\n[CommentaryContext] {len(result)} chars")
        print(result[:600])

    def test_unlimited_max_chars(self):
        """max_chars=-1 should never trigger tier fallback."""
        ctx = cb.CommentaryContext(
            PARAMS, SAMPLE_BOOK_ID, SAMPLE_PARA, SAMPLE_PARA_END,
            max_chars=-1,
        )
        result = ctx.build()
        # Should NOT contain the tier-2/3 labels
        self.assertNotIn("3-line window", result)
        self.assertNotIn("linked lines only", result)

    def test_tiny_max_chars_still_returns_string(self):
        """Even with max_chars=1, tier-3 fires and returns something."""
        ctx = cb.CommentaryContext(
            PARAMS, SAMPLE_BOOK_ID, SAMPLE_PARA, SAMPLE_PARA_END,
            max_chars=1,
        )
        result = ctx.build()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_single_paragraph(self):
        """para_end=-1 should work (single para)."""
        ctx = cb.CommentaryContext(PARAMS, SAMPLE_BOOK_ID, SAMPLE_PARA)
        result = ctx.build()
        self.assertIsInstance(result, str)


# ══════════════════════════════════════════════════════════════════
# 4. PaliDefsContext
# ══════════════════════════════════════════════════════════════════

class TestPaliDefsContextReal(unittest.TestCase):

    def _sample_pali(self) -> str:
        conn = _open_nissaya()
        rows = conn.execute(
            "SELECT pali_sentence FROM sentences "
            "WHERE book_id=? AND para_id BETWEEN ? AND ? "
            "ORDER BY para_id, line_id",
            (SAMPLE_BOOK_ID, SAMPLE_PARA, SAMPLE_PARA_END),
        ).fetchall()
        conn.close()
        return "\n".join(r["pali_sentence"] or "" for r in rows)

    def test_from_pali_text(self):
        pali = self._sample_pali()
        ctx = cb.PaliDefsContext(PARAMS, pali_text=pali)
        result = ctx.build()
        self.assertIsInstance(result, str)
        print(f"\n[PaliDefsContext from text] {len(result)} chars")
        print(result[:400])

    def test_from_range(self):
        ctx = cb.PaliDefsContext(
            PARAMS, book_id=SAMPLE_BOOK_ID,
            para_start=SAMPLE_PARA, para_end=SAMPLE_PARA_END,
        )
        result = ctx.build()
        self.assertIsInstance(result, str)
        print(f"\n[PaliDefsContext from range] {len(result)} chars")

    def test_label_present(self):
        pali = self._sample_pali()
        ctx = cb.PaliDefsContext(PARAMS, pali_text=pali)
        result = ctx.build()
        self.assertIn("PALI WORD DEFINITIONS", result)


# ══════════════════════════════════════════════════════════════════
# 5. PreviousTranslationContext
# ══════════════════════════════════════════════════════════════════

class TestPreviousTranslationContextReal(unittest.TestCase):

    def _first_translated_para(self) -> int | None:
        """Find the first para_id that has translated sentences, return para+1."""
        conn = _open_nissaya()
        row = conn.execute(
            "SELECT para_id FROM sentences "
            "WHERE book_id=? AND english_translation IS NOT NULL "
            "AND english_translation != '' "
            "ORDER BY para_id LIMIT 1",
            (SAMPLE_BOOK_ID,),
        ).fetchone()
        conn.close()
        return (row["para_id"] + 1) if row else None

    def test_returns_string(self):
        para = self._first_translated_para()
        if para is None:
            self.skipTest(
                f"No translated sentences in book {SAMPLE_BOOK_ID!r} yet."
            )
        ctx = cb.PreviousTranslationContext(
            PARAMS, SAMPLE_BOOK_ID, para_start=para, min_length=200
        )
        result = ctx.build()
        self.assertIsInstance(result, str)
        print(f"\n[PreviousTranslationContext] para_start={para}, {len(result)} chars")
        print(result[:600])

    def test_min_length_respected(self):
        """With min_length=0 only the immediately preceding para is fetched."""
        para = self._first_translated_para()
        if para is None:
            self.skipTest("No translated sentences.")

        ctx_small = cb.PreviousTranslationContext(
            PARAMS, SAMPLE_BOOK_ID, para_start=para, min_length=0
        )
        ctx_large = cb.PreviousTranslationContext(
            PARAMS, SAMPLE_BOOK_ID, para_start=para, min_length=2000
        )
        result_small = ctx_small.build()
        result_large = ctx_large.build()
        # Large budget should be >= small (may collect more paragraphs)
        self.assertGreaterEqual(len(result_large), len(result_small))

    def test_label_present(self):
        para = self._first_translated_para()
        if para is None:
            self.skipTest("No translated sentences.")
        ctx = cb.PreviousTranslationContext(PARAMS, SAMPLE_BOOK_ID, para_start=para)
        result = ctx.build()
        self.assertIn("PREVIOUS PARAGRAPH", result)

    def test_no_previous_at_start(self):
        ctx = cb.PreviousTranslationContext(PARAMS, SAMPLE_BOOK_ID, para_start=0)
        result = ctx.build()
        self.assertIn("no previous paragraph", result)


# ══════════════════════════════════════════════════════════════════
# 6. MulaAtthaContext
# ══════════════════════════════════════════════════════════════════

class TestMulaAtthaContextReal(unittest.TestCase):

    def _para_with_links(self) -> int | None:
        """Find a source para that has book_links entries."""
        conn = _open_nissaya()
        # Check book_links exists first
        has = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='book_links'"
        ).fetchone()
        if not has:
            conn.close()
            return None
        row = conn.execute(
            "SELECT src_para FROM book_links WHERE src_book=? LIMIT 1",
            (SAMPLE_BOOK_ID,),
        ).fetchone()
        conn.close()
        return row["src_para"] if row else None

    def test_returns_string(self):
        ctx = cb.MulaAtthaContext(
            PARAMS, SAMPLE_BOOK_ID, SAMPLE_PARA, SAMPLE_PARA_END
        )
        result = ctx.build()
        self.assertIsInstance(result, str)
        print(f"\n[MulaAtthaContext] {len(result)} chars")
        print(result[:600])

    def test_linked_para_shows_translations(self):
        para = self._para_with_links()
        if para is None:
            self.skipTest(
                "No book_links entries for this book. "
                "Adjust SAMPLE_BOOK_ID or add book_links data."
            )
        ctx = cb.MulaAtthaContext(PARAMS, SAMPLE_BOOK_ID, para)
        result = ctx.build()
        # Either found translations or the standard fallback message
        self.assertTrue(
            "§" in result or "no linked translations available" in result,
            f"Unexpected output: {result[:200]!r}"
        )

    def test_single_para(self):
        ctx = cb.MulaAtthaContext(PARAMS, SAMPLE_BOOK_ID, SAMPLE_PARA)
        result = ctx.build()
        self.assertIsInstance(result, str)

    def test_label_present(self):
        ctx = cb.MulaAtthaContext(PARAMS, SAMPLE_BOOK_ID, SAMPLE_PARA, SAMPLE_PARA_END)
        result = ctx.build()
        self.assertIn("MŪLA", result)


# ══════════════════════════════════════════════════════════════════
# 7. All blocks together  (smoke test — simulates a real job)
# ══════════════════════════════════════════════════════════════════

class TestFullPromptBlock(unittest.TestCase):
    """
    Build all context blocks for the sample range and concatenate them,
    exactly as a _job.py would before calling the AI.
    """

    def test_assemble_prompt_body(self):
        conn = _open_nissaya()
        # Fetch sentences for NissayaContext
        paras = []
        pali_lines = []
        for pid in range(SAMPLE_PARA, SAMPLE_PARA_END + 1):
            rows = conn.execute(
                "SELECT line_id, pali_sentence FROM sentences "
                "WHERE book_id=? AND para_id=? ORDER BY line_id",
                (SAMPLE_BOOK_ID, pid),
            ).fetchall()
            if rows:
                paras.append({
                    "book_id":   SAMPLE_BOOK_ID,
                    "para_id":   pid,
                    "sentences": [dict(r) for r in rows],
                })
                pali_lines.extend(r["pali_sentence"] or "" for r in rows)
        conn.close()

        pali_text = "\n".join(pali_lines)

        blocks = [
            cb.NissayaContext(PARAMS, paragraphs=paras).build(),
            cb.GlossaryContext(PARAMS, pali_text).build(),
            cb.CommentaryContext(PARAMS, SAMPLE_BOOK_ID,
                                 SAMPLE_PARA, SAMPLE_PARA_END).build(),
            cb.PaliDefsContext(PARAMS, pali_text=pali_text).build(),
            cb.PreviousTranslationContext(PARAMS, SAMPLE_BOOK_ID,
                                          SAMPLE_PARA, min_length=200).build(),
            cb.MulaAtthaContext(PARAMS, SAMPLE_BOOK_ID,
                                SAMPLE_PARA, SAMPLE_PARA_END).build(),
        ]

        prompt_body = "\n\n".join(b for b in blocks if b)

        self.assertIsInstance(prompt_body, str)
        self.assertGreater(len(prompt_body), 50)
        print(f"\n[Full prompt body] total {len(prompt_body)} chars, "
              f"{len(blocks)} block(s)")
        print("─" * 60)
        # Print first 100 chars of each block
        for blk in blocks:
            print(blk[:100].replace("\n", " "))
        print("─" * 60)


# ══════════════════════════════════════════════════════════════════
# Writers — tested read-only using in-memory DBs (no writes to disk)
# ══════════════════════════════════════════════════════════════════

class TestWritersReadOnly(unittest.TestCase):
    """
    Verify writer logic (column mapping, change counting, ON CONFLICT handling)
    against fresh in-memory DBs.  Nothing touches the real nissaya.db or
    glossary.db.
    """

    def _mem_nissaya(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE sentences (
                book_id TEXT, para_id INTEGER, line_id INTEGER,
                pali_sentence TEXT, english_translation TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sentences VALUES ('TEST',1,1,'Pali 1.', '')"
        )
        conn.execute(
            "INSERT INTO sentences VALUES ('TEST',1,2,'Pali 2.', '')"
        )
        conn.commit()
        return conn

    def _mem_glossary(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE glossary (
                pali TEXT, english TEXT, domain TEXT, sub_domain TEXT,
                context TEXT, note TEXT, source_id TEXT,
                UNIQUE(pali, english)
            )
        """)
        conn.commit()
        return conn

    def test_translation_writer_updates_rows(self):
        from unittest.mock import patch
        from contextlib import contextmanager

        mem = self._mem_nissaya()

        @contextmanager
        def _mock_connect(path):
            yield mem

        with patch.multiple("context_builders",
                            _nissaya_path=lambda p: "x",
                            _connect=_mock_connect):
            writer = cb.TranslationWriter(PARAMS)
            n = writer.save("TEST", 1, [
                {"line_id": 1, "english_translation": "First."},
                {"line_id": 2, "english_translation": "Second."},
            ])

        self.assertEqual(n, 2)
        rows = mem.execute(
            "SELECT line_id, english_translation FROM sentences ORDER BY line_id"
        ).fetchall()
        self.assertEqual(rows[0]["english_translation"], "First.")
        self.assertEqual(rows[1]["english_translation"], "Second.")

    def test_glossary_writer_inserts_and_deduplicates(self):
        from unittest.mock import patch

        mem = self._mem_glossary()

        with patch("context_builders.get_glossary_conn", return_value=mem):
            writer = cb.GlossaryWriter()
            n1 = writer.upsert([
                {"pali": "dhamma", "english": "teaching", "domain": "sutta"},
            ], sc_id="DN1")

        # Re-open for second upsert (first call closes the conn)
        mem2 = sqlite3.connect(":memory:")
        mem2.row_factory = sqlite3.Row
        mem2.execute("""
            CREATE TABLE glossary (
                pali TEXT, english TEXT, domain TEXT, sub_domain TEXT,
                context TEXT, note TEXT, source_id TEXT,
                UNIQUE(pali, english)
            )
        """)
        # Pre-insert the duplicate
        mem2.execute(
            "INSERT INTO glossary (pali,english,domain,sub_domain,context,note,source_id) "
            "VALUES ('dhamma','teaching','sutta','','','','')"
        )
        mem2.commit()

        with patch("context_builders.get_glossary_conn", return_value=mem2):
            writer2 = cb.GlossaryWriter()
            n2 = writer2.upsert([
                {"pali": "dhamma", "english": "teaching"},  # duplicate
            ], sc_id="DN1")

        self.assertEqual(n1, 1, "First insert should create 1 row.")
        self.assertEqual(n2, 0, "Duplicate should be ignored (ON CONFLICT DO NOTHING).")


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)