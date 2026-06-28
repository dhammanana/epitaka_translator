"""
Quick smoke test — runs all context builders against the real DBs and prints output.
Usage:  python run_context_test.py
"""

import sqlite3
import sys
import types

NISSAYA_DB  = "/Volumes/Data/code/translator/epitaka_translator/data/nissaya.db"
GLOSSARY_DB = "/Volumes/Data/code/translator/epitaka_translator/data/glossary.db"

# SAMPLE_BOOK_ID  = "Ps-ii"
SAMPLE_BOOK_ID  = "Sp-i"
SAMPLE_PARA     = 615
SAMPLE_PARA_END = 620

# ── stubs ────────────────────────────────────────────────────────
db_mod = types.ModuleType("database")
db_mod.get_glossary_conn = lambda: sqlite3.connect(GLOSSARY_DB, timeout=30)
sys.modules["database"] = db_mod

cfg_mod = types.ModuleType("config")
cfg_mod.NISSAYA_DB = NISSAYA_DB
cfg_mod.SC_DATA_DB = ""
sys.modules["config"] = cfg_mod

from context_builders import (   # noqa: E402
    NissayaContext, GlossaryContext, CommentaryContext,
    PaliDefsContext, PreviousTranslationContext, MulaAtthaContext,
    ParallelTranslationContext,
)

# ── params ───────────────────────────────────────────────────────
params     = {"nissaya_db": NISSAYA_DB}
book_id    = SAMPLE_BOOK_ID
para_start = SAMPLE_PARA
para_end   = SAMPLE_PARA_END

# ── fetch sentences for NissayaContext ───────────────────────────
conn = sqlite3.connect(f"file:{NISSAYA_DB}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
paragraphs = []
pali_lines = []
for pid in range(para_start, para_end + 1):
    rows = conn.execute(
        "SELECT line_id, pali_sentence FROM sentences "
        "WHERE book_id=? AND para_id=? ORDER BY line_id",
        (book_id, pid),
    ).fetchall()
    if rows:
        paragraphs.append({
            "book_id":   book_id,
            "para_id":   pid,
            "sentences": [dict(r) for r in rows],
        })
        pali_lines.extend(r["pali_sentence"] or "" for r in rows)
conn.close()

pali_text = "\n".join(pali_lines)
print(pali_text)

# ── build all blocks ─────────────────────────────────────────────
blocks = [
    NissayaContext(params, paragraphs).build(),
    GlossaryContext(params, pali_text).build(),
    CommentaryContext(params, book_id, para_start, para_end).build(),
    PaliDefsContext(params, pali_text=pali_text).build(),
    PreviousTranslationContext(params, book_id, para_start).build(),
    MulaAtthaContext(params, book_id, para_start, para_end).build(),
    ParallelTranslationContext(params, book_id, para_start, para_end).build(),
]

prompt_body = "\n\n".join(b for b in blocks if b)
print(prompt_body)