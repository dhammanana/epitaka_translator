"""
study_builder.py — Turn mūla sections + their linked commentary into study material.

For every level-10 heading section of a mūla book (e.g. M-i), this script:

  1. Gathers the section's mūla text (Pāli + English translation).
  2. Gathers every related section: the aṭṭhakathā (commentary) and ṭīkā
     (sub-commentary) paragraphs linked from those mūla lines via the
     `book_links` table (forward chain: mūla → aṭṭhakathā → ṭīkā), each with
     Pāli + English translation and every line labelled [book:para:line].
  3. Adds Pāli word definitions for the rarest words in the section
     (PaliDefsContext, capped to a few words/definitions).
  4. Sends everything to Gemini (gemini-3.7-flash), asking it to write
     comprehensive study material in English: ALL information covered, no
     compression, full reasoning sequences preserved for any debated or
     controversial point (with a final “🔎 Controversies, questions & things
     to dig deeper” section); lists rendered as lists; quotes cited as
     [book:para:line]; stories summarised.

     API budget: `--batch-size` consecutive sections are sent in ONE call
     (default 3), and the model gets AT MOST ONE get_text_range tool round per
     batch (all fetches issued as parallel calls in that single round, then it
     must answer). Each batch therefore costs at most 2 API requests (the
     initial request + one tool round), instead of up to 9 requests per
     section when tool rounds were allowed to chain.
  5. Saves the result into epitaka_en.db (default ./data) — the English
     translation DB — so the web server and the app read study guides from
     the same DB they already ship (the `summaries` table).

Example:
    python study_builder.py --book M-i
    python study_builder.py --book M-i --start-para 6 --end-para 6   # one section
    python study_builder.py --book M-i --max-sections 2 --dry-run    # inspect prompts
    python study_builder.py --book M-i --batch-size 1 --no-tools     # strict 1 request/section

Resumable: sections already present in the summary DB are skipped unless
--overwrite is given.
"""

import argparse
import json
import os
import re
import sys
import types
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

import common.common_utils as cu
import common.ai_client as ai
# import common.ai_client_bai as ai

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

EPITAKA_DB  = cu.EPITAKA_DB
EN_DB       = os.environ.get("EPITAKA_EN_DB") or cu.lang_db_path(EPITAKA_DB, "en")
# Study guides now live INSIDE the English translation DB (epitaka_en.db),
# next to the translations they quote. The web server and the app read the
# `summaries` table from that same file, so there is no separate summary DB
# to deploy. EPITAKA_SUMMARY_DB can still override this to a custom path.
SUMMARY_DB  = os.environ.get("EPITAKA_SUMMARY_DB", EN_DB)
DEFAULT_LOG_DIR = "/tmp/study_builder_logs"

# Models used when --model is not given. The first model is preferred; when a
# model's quota is exhausted (3× HTTP 429) it is removed from the list and
# the NEXT model takes over automatically (see ai_client.ModelPool).
MODEL_FALLBACK_LIST = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
]

PRESET_BOOKS = (
    "D-i,D-ii,D-iii,M-i,M-ii,M-iii,S-i,S-ii,S-iii,S-iv,S-v,A-i,A-ii,A-iii,A-iv,A-v,A-vi,A-vii,"
    "A-viii,A-ix,A-x,A-xi,Khp,Dhp,Ud,It,Sn,Vv,Pv,Th,Thī,Ap-i,Ap-ii,Ap-iii,Bv,Cp,Ja-i,Ja-ii,Nidd-i,"
    "Nidd-ii,Paṭis,Nett,Mil,Pet,Vin-i,Vin-ii,Vin-ii-b,Vin-iii,Vin-iv,Vin-v,Dvem-bhk,Dvem-bhni,Dhs,"
    "Vibh,Dhatuk,Pp,Kv,Yam-i,Yam-ii,Yam-iii,Paṭṭh-i,Paṭṭh-ii,Paṭṭh-iii,Paṭṭh-iv,Paṭṭh-v,Sv-i,Sv-ii,"
    "Sv-iii,Ps-i,Ps-ii,Ps-iii,Spk-i,Spk-ii,Spk-iii,Spk-iv,Spk-v,Mp-i,Mp-ii,Mp-v,Mp-viii,Mp-iii,Mp-iv,"
    "Mp-vi,Mp-vii,Mp-ix,Mp-x,Mp-xi,Pj-i")


# Hard cap on the assembled user prompt (UTF-8 bytes) per section. M-i's
# largest level-10 section stays well under this (~400k chars incl. EN), so
# nothing is ever truncated in practice. If a section ever exceeds it, the
# ṭīkā block is dropped first, then the aṭṭhakathā block, keeping the mūla
# whole — and it is logged loudly so it can be reviewed.
MAX_INPUT_CHARS = int(os.environ.get("STUDY_MAX_INPUT_CHARS", 700_000))

# Hard cap on the COMBINED prompt (UTF-8 bytes) of one batched call. Batches
# that exceed it are split in half until they fit, so a handful of giant
# sections never blow the model's context window.
BATCH_MAX_CHARS = int(os.environ.get("STUDY_BATCH_MAX_CHARS", 1_500_000))

# Pāli definition limits (see PaliDefsContext): at most MAX_PALI_WORDS rare
# words are looked up; at most MAX_PALI_DEFS of them are shown, each capped
# at MAX_PALI_DEF_WORDS words.
MAX_PALI_WORDS    = int(os.environ.get("STUDY_PALI_WORDS", 10))
MAX_PALI_DEFS     = int(os.environ.get("STUDY_PALI_DEFS", 5))
MAX_PALI_DEF_WORDS = int(os.environ.get("STUDY_PALI_DEF_WORDS", 100))

# Level of the heading that delimits the study-material sections.
SECTION_HEADING_LEVEL = 10


# ══════════════════════════════════════════════════════════════════
# Stub modules (same pattern as book_translator.py) — context_builders
# imports `database.get_glossary_conn` at import time, so stub it before
# importing it. Study material only reads; the glossary connector is unused
# here but must exist.
# ══════════════════════════════════════════════════════════════════

if "database" not in sys.modules:
    _db_mod = types.ModuleType("database")
    # Study material only reads; the glossary connector is unused here but
    # must exist. Point it at the real per-language file (English), never at
    # a bare glossary.db.
    _db_mod.get_glossary_conn = lambda: sqlite3.connect(
        cu.glossary_db_path(EPITAKA_DB, "en"), timeout=30
    )
    sys.modules["database"] = _db_mod

if "config" not in sys.modules:
    _cfg_mod = types.ModuleType("config")
    _cfg_mod.EPITAKA_DB = EPITAKA_DB
    _cfg_mod.SC_DATA_DB = ""
    sys.modules["config"] = _cfg_mod

from common.context_builders import PaliDefsContext  # noqa: E402

_connect = cu.connect


# ══════════════════════════════════════════════════════════════════
# DB helpers
# ══════════════════════════════════════════════════════════════════

_SUMMARY_DDL = """
CREATE TABLE IF NOT EXISTS summaries (
    book_id       TEXT NOT NULL,
    section_id    INTEGER NOT NULL,
    para_start    INTEGER NOT NULL,
    para_end      INTEGER NOT NULL,
    heading_title TEXT NOT NULL DEFAULT '',
    sutta_title   TEXT NOT NULL DEFAULT '',
    vagga_title   TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    content       TEXT NOT NULL,
    sources       TEXT NOT NULL DEFAULT '[]',
    model         TEXT NOT NULL DEFAULT '',
    input_chars   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT DEFAULT (datetime('now')),
    updated_at    TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (book_id, section_id)
);
CREATE INDEX IF NOT EXISTS idx_summaries_book ON summaries(book_id);
"""


def ensure_summary_db(path: str):
    with _connect(path) as conn:
        conn.executescript(_SUMMARY_DDL)
        conn.commit()


def summary_exists(path: str, book_id: str, section_id: int) -> bool:
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT 1 FROM summaries WHERE book_id=? AND section_id=?",
            (book_id, section_id),
        ).fetchone()
    return row is not None


def save_summary(path: str, row: dict):
    with _connect(path) as conn:
        conn.execute(
            """INSERT INTO summaries
               (book_id, section_id, para_start, para_end, heading_title,
                sutta_title, vagga_title, title, content, sources, model,
                input_chars, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
               ON CONFLICT(book_id, section_id) DO UPDATE SET
                   para_start=excluded.para_start, para_end=excluded.para_end,
                   heading_title=excluded.heading_title,
                   sutta_title=excluded.sutta_title,
                   vagga_title=excluded.vagga_title,
                   title=excluded.title, content=excluded.content,
                   sources=excluded.sources, model=excluded.model,
                   input_chars=excluded.input_chars, updated_at=datetime('now')""",
            (
                row["book_id"], row["section_id"], row["para_start"],
                row["para_end"], row["heading_title"], row["sutta_title"],
                row["vagga_title"], row["title"], row["content"],
                json.dumps(row["sources"], ensure_ascii=False),
                row["model"], row["input_chars"],
            ),
        )
        conn.commit()


# ══════════════════════════════════════════════════════════════════
# Section + link gathering
# ══════════════════════════════════════════════════════════════════

def fetch_sections(conn, book_id: str) -> list[dict]:
    """
    All level-10 heading sections in document order. Each dict:
      {para_id, title, next (exclusive end para), sutta_title, vagga_title}
    The section spans paragraphs [para_id, next).
    """
    rows = conn.execute(
        "SELECT para_id, title, parent FROM headings "
        "WHERE book_id=? AND level=? ORDER BY para_id",
        (book_id, SECTION_HEADING_LEVEL),
    ).fetchall()

    # next boundary = next level-10 heading para, else end of book
    by_id = {r["para_id"]: dict(r) for r in rows}
    paras = sorted(by_id.keys())

    # heading lookup for ancestor chain
    by_para = {}
    for h in conn.execute(
        "SELECT para_id, level, title, parent FROM headings WHERE book_id=?",
        (book_id,),
    ).fetchall():
        by_para[h["para_id"]] = dict(h)

    def _ancestors(pid: int) -> list[dict]:
        chain = []
        seen = set()
        while pid in by_para and pid not in seen:
            seen.add(pid)
            h = by_para[pid]
            chain.append(h)
            pid = h.get("parent")
        return chain

    sections = []
    for i, p in enumerate(paras):
        nxt = paras[i + 1] if i + 1 < len(paras) else 999_999_999
        # sutta / vagga titles from the ancestor chain of the heading
        sutta_title = ""
        vagga_title = ""
        for h in _ancestors(p):
            if h["level"] == 4 and not sutta_title:
                sutta_title = h["title"] or ""
            elif h["level"] == 2 and not vagga_title:
                vagga_title = h["title"] or ""
        sections.append({
            "para_id":      p,
            "next":         nxt,
            "heading_title": by_id[p].get("title") or "",
            "sutta_title":  sutta_title,
            "vagga_title":  vagga_title,
        })
    return sections


def linked_dst_paras(conn, src_book: str, src_lo: int, src_hi: int, dst_book: str) -> list[int]:
    """Distinct dst paragraphs in `dst_book` linked from src lines in [src_lo, src_hi)."""
    rows = conn.execute(
        "SELECT DISTINCT dst_para FROM book_links "
        "WHERE src_book=? AND src_para>=? AND src_para<? AND dst_book=? "
        "ORDER BY dst_para",
        (src_book, src_lo, src_hi, dst_book),
    ).fetchall()
    return [r["dst_para"] for r in rows]


def linked_dst_paras_from_paras(conn, src_book: str, src_paras: list[int], dst_book: str) -> list[int]:
    """Distinct dst paragraphs in `dst_book` linked from the given src paragraphs."""
    if not src_paras:
        return []
    out: set[int] = set()
    step = 400
    for i in range(0, len(src_paras), step):
        chunk = src_paras[i:i + step]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT DISTINCT dst_para FROM book_links "
            f"WHERE src_book=? AND src_para IN ({ph}) AND dst_book=?",
            [src_book, *chunk, dst_book],
        ).fetchall()
        out.update(r["dst_para"] for r in rows)
    return sorted(out)


def fetch_lines(conn, en_conn, book_id: str, paras: list[int]) -> list[dict]:
    """
    All lines (Pāli + English translation) for the given paragraphs, in
    document order. Returns [{"para_id", "line_id", "pali", "en"}, ...].
    """
    if not paras:
        return []
    out = []
    step = 400
    for i in range(0, len(paras), step):
        chunk = paras[i:i + step]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT para_id, line_id, pali FROM sentences "
            f"WHERE book_id=? AND para_id IN ({ph}) ORDER BY para_id, line_id",
            [book_id, *chunk],
        ).fetchall()
        out.extend(dict(r) for r in rows)

    en_map: dict[tuple[int, int], str] = {}
    if en_conn is not None:
        for i in range(0, len(paras), step):
            chunk = paras[i:i + step]
            ph = ",".join("?" * len(chunk))
            erows = en_conn.execute(
                f"SELECT para_id, line_id, translation FROM sentences "
                f"WHERE book_id=? AND para_id IN ({ph})",
                [book_id, *chunk],
            ).fetchall()
            for er in erows:
                if er["translation"]:
                    en_map[(er["para_id"], er["line_id"])] = er["translation"]

    return [
        {
            "para_id": ln["para_id"],
            "line_id": ln["line_id"],
            "pali":    ln["pali"] or "",
            "en":      en_map.get((ln["para_id"], ln["line_id"]), ""),
        }
        for ln in out
    ]


def render_block(book_id: str, book_name: str, lines: list[dict]) -> str:
    """Render lines with [book:para:line] citation prefixes."""
    parts = [f"══ {book_id} — {book_name} ══"]
    for ln in lines:
        cit = f"[{book_id}:{ln['para_id']}:{ln['line_id']}]"
        parts.append(f"{cit} {ln['pali']}")
        if ln["en"]:
            parts.append(f"     EN: {ln['en']}")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════
# Pāli definitions (capped)
# ══════════════════════════════════════════════════════════════════

_DEF_ENTRY_RE = re.compile(r"^  \S")


def cap_pali_defs(raw: str, max_defs: int, max_words_per_def: int) -> str:
    """
    PaliDefsContext output is blocks of the form::

          word:
            - stem: usage … usage

    Keep at most `max_defs` blocks, each capped at `max_words_per_def` words
    (first line — the headword — always kept intact).
    """
    if not raw or max_defs <= 0:
        return raw
    lines = raw.split("\n")
    entries: list[list[str]] = []
    cur: list[str] | None = None
    for ln in lines:
        if _DEF_ENTRY_RE.match(ln) and not ln.startswith("    "):
            if cur is not None:
                entries.append(cur)
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
    if cur is not None:
        entries.append(cur)

    capped = []
    for entry in entries[:max_defs]:
        first, _, rest = entry[0], None, entry[1:]
        rest_words = " ".join(rest).split()
        if len(rest_words) > max_words_per_def:
            rest = ["    " + " ".join(rest_words[:max_words_per_def]) + " …[truncated]"]
        capped.append("\n".join([first, *rest]))
    return "\n".join(capped)


def build_pali_defs_block(params: dict, pali_text: str) -> str:
    """Rare-word definitions for the section, capped (10 words / 5 defs / 100 words)."""
    ctx = PaliDefsContext(
        params, pali_text=pali_text, max_words=MAX_PALI_WORDS,
        log_info=lambda *a, **k: None, log_warn=print,
    )
    raw = ctx.build()
    if "(no " in raw or "(word definitions unavailable)" in raw:
        return ""
    return cap_pali_defs(raw, MAX_PALI_DEFS, MAX_PALI_DEF_WORDS)


# ══════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════

# Shared writing rules — used verbatim by both the single-section and the
# batched system prompts.
_RULES = """Write comprehensive study material in English. Follow these rules strictly:

1. COMPLETENESS — Do not omit any information from the mūla, aṭṭhakathā, or ṭīkā. Cover every doctrinal explanation, list, detail, definition, and nuance. The ONLY exception: if the aṭṭhakathā or ṭīkā merely glosses a word whose meaning is already fully conveyed by the translation itself (a plain word-for-word gloss adding no new information), you may skip it. If the gloss adds ANY extra information — a further meaning, an etymology, a doctrinal point, or a cross-reference — include it.
   Do NOT compress. Where the source is long and detailed, your study material must be correspondingly long and detailed. Never replace a chain of argument, a series of alternatives, or a step-by-step explanation with a single summary sentence. Err decisively on the side of including too much rather than too little.

2. REASONING SEQUENCES & CONTROVERSIES — Whenever the aṭṭhakathā or ṭīkā raises a question, an objection, an alternative interpretation, an apparent contradiction, or an interpretive difficulty, preserve the ENTIRE reasoning sequence, step by step:
   (a) the question or problem,
   (b) why it is a problem,
   (c) the proposed interpretation(s) or answer(s),
   (d) the objections raised against those proposals,
   (e) the final resolution — or, if the text leaves it unresolved, say so explicitly,
   (f) its doctrinal significance.
   Never flatten a debate into a single conclusion. If the commentators argue for and against a position, show both sides and how the argument proceeds, with citations for each step.

3. DIG DEEPER SECTION — At the very end of the study material, add a final section headed exactly:
   ## 🔎 Controversies, questions & things to dig deeper
   List every point from the aṭṭhakathā/ṭīkā that is debated, problematic, ambiguous, or doctrinally significant — one bullet per point — each with its citation(s) [book:para:line], a one-to-three-sentence account of what is at stake and what the commentators say, and why it is worth digging deeper. If there are none, write exactly: "None identified in this section."

4. LISTS — Whenever the source presents a list (e.g. seven factors, five aggregates, the steps of a practice), render it as a real bulleted or numbered list in your study material, so the list structure is unmistakable.

5. CITATIONS — Whenever you quote or closely paraphrase an excerpt, cite its source immediately after it in the exact form [book:para:line] (e.g. [M-i:6:1], [Ps-i:49:2]). Use the citation labels exactly as given in the input — book_id, para_id, line_id.

6. STORIES — If a passage contains a story or narrative (a jātaka tale, an anecdote about a person, a simile elaborated as a scene), summarise it concisely in your own words — do not quote it at length. Include the point or moral of the story.

7. STRUCTURE — Organise the material clearly: a short overview of the section, then structured subsections. Use markdown headings (##), bold for key terms, and keep Pāli terms in italics with an English gloss on first use (e.g. *dukkha* — suffering).

8. TERMINOLOGY — Preserve doctrinal precision; do not conflate distinct Pāli terms or concepts."""

# Tool strategy shared by both prompts: the model gets AT MOST ONE tool round
# per call, so it must issue every fetch as parallel calls in that single
# round and then answer. Each round is a full API request (rpm/tpm budget).
_TOOL_STRATEGY = """TOOL USE — You have a get_text_range tool that can fetch more lines (Pāli + English) from the same books; use it whenever a discussion refers to a passage you do not have — e.g. the passage a ṭīkā argument is about, or the wider context of a debate. You get AT MOST ONE tool round: in that single round, issue EVERY fetch you need as parallel calls (multiple calls in one response), then after the results come back you must write the final answer — there is no further round. Never fetch one range at a time. If the results are still insufficient, work with what you have and note the gap in the Dig Deeper section."""

_OUTPUT_SINGLE = """Return ONLY valid JSON with exactly two keys: "title" (a short descriptive title for this study section) and "content" (the full study material in Markdown, including the Dig Deeper section). No markdown fences, no prose outside the JSON."""

_OUTPUT_BATCH = """Return ONLY valid JSON with exactly one key, "sections": an ARRAY with one object per section, IN THE ORDER GIVEN, covering EVERY section:
{"sections": [{"section_id": <the section's paragraph id, as given in its header>, "title": "...", "content": "..."}, ...]}
Each "content" is the full study material in Markdown for that section, including its own Dig Deeper section. Do not merge sections or skip any. No markdown fences, no prose outside the JSON."""

_HEADER = "You are an expert scholar of Pāli Buddhist literature, creating STUDY MATERIAL for a mūla (canonical) text, using the mūla text itself, its aṭṭhakathā (commentary), and its ṭīkā (sub-commentary). English translations of the Pāli are provided alongside every passage."

_BLOCKS_SINGLE = """You will be given up to four blocks for THE section:
  1. MŪLA — the canonical text section (Pāli + English), every line cited [book:para:line].
  2. AṬṬHAKATHĀ — commentary paragraphs linked to this section (Pāli + English), cited the same way.
  3. ṬĪKĀ — sub-commentary paragraphs linked to this section (Pāli + English), cited the same way.
  4. PALI WORD DEFINITIONS — dictionary/grammar usages of rare words found in the section."""

_BLOCKS_BATCH = """This request contains SEVERAL study sections. You will be given up to four blocks for EACH of them, clearly separated:
  1. MŪLA — the canonical text section (Pāli + English), every line cited [book:para:line].
  2. AṬṬHAKATHĀ — commentary paragraphs linked to that section (Pāli + English), cited the same way.
  3. ṬĪKĀ — sub-commentary paragraphs linked to that section (Pāli + English), cited the same way.
  4. PALI WORD DEFINITIONS — dictionary/grammar usages of rare words found in that section.
You must produce study material for EVERY section, in the order given."""

SYSTEM_PROMPT = _HEADER + "\n\n" + _BLOCKS_SINGLE + "\n\n" + _TOOL_STRATEGY + "\n\n" + _RULES + "\n\n" + _OUTPUT_SINGLE

BATCH_SYSTEM_PROMPT = _HEADER + "\n\n" + _BLOCKS_BATCH + "\n\n" + _TOOL_STRATEGY + "\n\n" + _RULES + "\n\n" + _OUTPUT_BATCH


def build_section_block_text(
    book_id: str,
    book_name: str,
    sec: dict,
    mula_block: str,
    attha_block: str,
    tika_block: str,
    pali_defs: str,
) -> str:
    """One section's source text (blocks + header), WITHOUT the final
    instructions — several of these are assembled into a batch prompt."""
    lines = [
        f"Book: {book_id} — {book_name}",
        f"Section (heading level {SECTION_HEADING_LEVEL}) at paragraph {sec['para_id']}"
        + (f" — “{sec['heading_title']}”" if sec["heading_title"] else ""),
    ]
    if sec["sutta_title"]:
        lines.append(f"Sutta: {sec['sutta_title']}")
    if sec["vagga_title"]:
        lines.append(f"Vagga: {sec['vagga_title']}")
    lines.append(f"Paragraphs: {sec['para_id']}–{sec['next'] - 1}")
    lines.append("")
    lines.append(mula_block)
    if attha_block:
        lines.append("")
        lines.append(attha_block)
    if tika_block:
        lines.append("")
        lines.append(tika_block)
    if pali_defs:
        lines.append("")
        lines.append("══ PALI WORD DEFINITIONS ══")
        lines.append(pali_defs)
    return "\n".join(lines)


def build_batch_user_prompt(book_id: str, book_name: str, inputs: list[dict]) -> str:
    """One prompt covering several sections. Each `input` is a dict with
    'sec' (the section dict) and 'text' (its size-guarded section text)."""
    lines = [
        f"Book: {book_id} — {book_name}",
        f"This request contains {len(inputs)} study section(s). "
        "Write study material for EVERY one of them, in the order given.",
    ]
    for i, inp in enumerate(inputs, 1):
        lines.append("")
        lines.append("═" * 72)
        lines.append(f"SECTION {i}/{len(inputs)}")
        lines.append("═" * 72)
        lines.append(inp["text"])
    lines.append("")
    lines.append(
        "Now write the study material for EVERY section above, according to the "
        "instructions (completeness without compression, reasoning sequences "
        "preserved, lists as lists, citations [book:para:line], stories summarised, "
        "a final Dig Deeper section in every section). If you need more source text "
        "than is provided, call get_text_range — you get AT MOST ONE tool round, so "
        "issue every fetch as parallel calls in that single round, then produce the "
        "final answer once the results come back. Return the JSON object."
    )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# get_text_range tool — lets the model fetch more source text mid-answer
# ══════════════════════════════════════════════════════════════════

def make_tool_executor(
    epitaka_db:    str,
    en_db:         str,
    allowed_books: list[str],
    max_lines:     int = 800,
    max_chars:     int = 200_000,
):
    """
    Build the executor behind the `get_text_range` tool declaration. Each
    call returns the requested lines (Pāli + English, cited [book:para:line])
    as a plain dict for the model, capped so a single tool response never
    blows the context window. Books outside `allowed_books` are rejected.
    """
    allowed = set(allowed_books)
    book_names: dict[str, str] = {}

    def _book_name(book_id: str) -> str:
        if book_id not in book_names:
            with _connect(epitaka_db) as conn:
                row = conn.execute(
                    "SELECT book_name FROM books WHERE book_id=?", (book_id,),
                ).fetchone()
            book_names[book_id] = (row["book_name"] if row and row["book_name"]
                                   else book_id)
        return book_names[book_id]

    def execute(name: str, args: dict) -> dict:
        if name != "get_text_range":
            return {"error": f"Unknown tool: {name}"}
        book_id = str(args.get("book_id") or "").strip()
        if book_id not in allowed:
            return {"error": (
                f"book_id must be a single book id (one of: "
                f"{', '.join(sorted(allowed))}) — call get_text_range once per "
                f"book, never combine multiple ids in one string."
            )}
        try:
            para_start = int(args.get("para_start") or 1)
            para_end   = int(args.get("para_end") or para_start)
            line_start = int(args.get("line_start") or 1)
            line_end   = int(args.get("line_end") or 0)
        except (TypeError, ValueError):
            return {"error": "para_start/para_end/line_start/line_end must be integers"}
        if para_start < 1 or para_end < para_start:
            return {"error": "para_start >= 1 and para_end >= para_start are required"}
        if para_end - para_start + 1 > 200:
            return {"error": f"Range too large: {para_end - para_start + 1} paragraphs; "
                             "request at most 200 at a time."}

        paras = list(range(para_start, para_end + 1))
        with _connect(epitaka_db) as conn, _connect(en_db) as en_conn:
            lines = fetch_lines(conn, en_conn, book_id, paras)
        if line_start > 1:
            lines = [ln for ln in lines if ln["line_id"] >= line_start]
        if line_end > 0:
            lines = [ln for ln in lines if ln["line_id"] <= line_end]
        if not lines:
            return {"text": f"No lines found for {book_id} paras {para_start}–{para_end} "
                            f"(lines {line_start}–{line_end or 'end'})."}

        truncated = False
        buf: list[str] = []
        total = 0
        for ln in lines:
            piece = f"[{book_id}:{ln['para_id']}:{ln['line_id']}] {ln['pali']}"
            if ln["en"]:
                piece += f"\n     EN: {ln['en']}"
            total += len(piece) + 1
            if total > max_chars and buf:
                truncated = True
                break
            if len(buf) >= max_lines:
                truncated = True
                break
            buf.append(piece)
        text = "\n".join(buf)
        if truncated:
            text += (f"\n\n[Note: response capped at {len(buf)} lines — "
                     "request a smaller range if you need the rest.]")
        return {"text": text, "line_count": len(buf), "truncated": truncated}

    return execute


# ══════════════════════════════════════════════════════════════════
# Per-batch processing (several sections in one AI call)
# ══════════════════════════════════════════════════════════════════

def parse_batch_response(raw: str) -> list[dict]:
    """
    Parse the AI's JSON reply into a list of {section_id, title, content}.

    Expected shape: {"sections": [{"section_id": <para>, "title": ...,
    "content": ...}, ...]}. Tolerates markdown fences, a single-section
    object (no "sections" key), and "sections" given as a dict instead of
    a list. Entries without usable content are dropped.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^\s*```[a-zA-Z]*\s*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        obj = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return []

    sections = obj.get("sections")
    if isinstance(sections, dict):
        sections = [sections]
    if not isinstance(sections, list):
        # Model ignored the batch format and returned a single-section object.
        if isinstance(obj.get("content"), str) and obj.get("content").strip():
            sections = [obj]
        else:
            return []

    out = []
    for s in sections:
        if not isinstance(s, dict):
            continue
        content = s.get("content")
        if isinstance(content, list):
            content = "\n\n".join(str(c) for c in content)
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            sid = int(s.get("section_id") or 0)
        except (TypeError, ValueError):
            sid = 0
        out.append({
            "section_id": sid,
            "title":      str(s.get("title") or "").strip(),
            "content":    content.strip(),
        })
    return out


def build_section_inputs(
    book_id: str,
    book_name: str,
    attha_ref: str,
    tika_ref: str,
    sec: dict,
    args,
    epitaka_db: str,
    en_db: str,
) -> dict:
    """
    Fetch and render all source blocks for one section into its prompt text
    (with the ṭīkā-then-aṭṭhakathā size guard applied). Returns a dict with
    'sec', 'text', 'chars', 'sources', and mūla/aṭṭhakathā/ṭīkā line counts.
    """
    lo, hi = sec["para_id"], sec["next"]
    params = {"epitaka_db": epitaka_db, "lang_db": en_db}
    sources = [book_id]

    with _connect(epitaka_db) as conn, _connect(en_db) as en_conn:
        # 1) mūla lines of this section
        mula_lines = fetch_lines(conn, en_conn, book_id, list(range(lo, hi)))

        # 2) aṭṭhakathā paragraphs linked from the mūla lines
        attha_paras = []
        if attha_ref:
            attha_paras = linked_dst_paras(conn, book_id, lo, hi, attha_ref)
            if attha_paras:
                sources.append(attha_ref)
        attha_lines = fetch_lines(conn, en_conn, attha_ref, attha_paras) if attha_paras else []

        # 3) ṭīkā paragraphs linked from the mūla lines AND from those
        #    aṭṭhakathā lines (the ṭīkā-on-aṭṭhakathā chain)
        tika_paras = []
        if tika_ref:
            tika_paras = linked_dst_paras(conn, book_id, lo, hi, tika_ref)
            tika_paras = sorted(set(tika_paras) | set(
                linked_dst_paras_from_paras(conn, attha_ref, attha_paras, tika_ref)
            ))
            if tika_paras:
                sources.append(tika_ref)
        tika_lines = fetch_lines(conn, en_conn, tika_ref, tika_paras) if tika_paras else []

        # 4) Pāli word definitions for the rarest words in the whole section
        pali_all = "\n".join(
            ln["pali"]
            for ln in (mula_lines + attha_lines + tika_lines)
        )
        pali_defs = build_pali_defs_block(params, pali_all)

        mula_block  = render_block(book_id, "MŪLA", mula_lines)
        attha_block = render_block(attha_ref, "AṬṬHAKATHĀ", attha_lines) if attha_lines else ""
        tika_block  = render_block(tika_ref, "ṬĪKĀ", tika_lines) if tika_lines else ""

    text = build_section_block_text(
        book_id, book_name, sec, mula_block, attha_block, tika_block, pali_defs,
    )
    size = len(text.encode("utf-8"))

    # Size guard: drop ṭīkā first, then aṭṭhakathā, keep mūla whole.
    if size > args.max_input_chars and tika_block:
        print(f"  [size] section@{lo}: {size:,} bytes > {args.max_input_chars:,} — dropping ṭīkā block (REVIEW)")
        text = build_section_block_text(book_id, book_name, sec, mula_block, attha_block, "", pali_defs)
        size = len(text.encode("utf-8"))
    if size > args.max_input_chars and attha_block:
        print(f"  [size] section@{lo}: {size:,} bytes — dropping aṭṭhakathā block (REVIEW)")
        text = build_section_block_text(book_id, book_name, sec, mula_block, "", "", pali_defs)
        size = len(text.encode("utf-8"))

    return {
        "sec":     sec,
        "text":    text,
        "chars":   len(text),
        "sources": sources,
        "mula":    len(mula_lines),
        "attha":   len(attha_lines),
        "tika":    len(tika_lines),
    }


def process_batch(
    book_id: str,
    book_name: str,
    sections: list[dict],
    args,
    rotator: ai.KeyRotator,
    epitaka_db: str,
    en_db: str,
    summary_db: str,
    attha_ref: str,
    tika_ref: str,
    models: list[str],
) -> list[dict]:
    """
    Process several sections in ONE AI call (at most one tool round), save
    each section's result, and return one result dict per section.
    Oversized batches are split in half recursively until they fit
    `args.batch_max_chars`.
    """
    if not sections:
        return []

    inputs = [
        build_section_inputs(book_id, book_name, attha_ref, tika_ref, sec, args, epitaka_db, en_db)
        for sec in sections
    ]
    prompt = build_batch_user_prompt(book_id, book_name, inputs)

    if len(prompt.encode("utf-8")) > args.batch_max_chars and len(inputs) > 1:
        mid = len(inputs) // 2
        print(f"  [size] batch {len(prompt):,} chars > {args.batch_max_chars:,} — "
              f"splitting into {len(inputs[:mid])}+{len(inputs[mid:])} sections")
        return (
            process_batch(book_id, book_name, [i["sec"] for i in inputs[:mid]], args,
                          rotator, epitaka_db, en_db, summary_db, attha_ref, tika_ref, models)
            + process_batch(book_id, book_name, [i["sec"] for i in inputs[mid:]], args,
                            rotator, epitaka_db, en_db, summary_db, attha_ref, tika_ref, models)
        )

    lo_first = inputs[0]["sec"]["para_id"]
    lo_last  = inputs[-1]["sec"]["para_id"]
    chunk_id = f"sec{lo_first}" if len(inputs) == 1 else f"sec{lo_first}-{lo_last}"
    total_chars = sum(i["chars"] for i in inputs)

    print(f"  Prompt: {total_chars:,} chars ({len(inputs)} section(s)) — "
          + ", ".join(f"sec@{i['sec']['para_id']}: {i['mula']}m/{i['attha']}a/{i['tika']}t" for i in inputs)
          + (" + get_text_range tool" if not args.no_tools else ""))

    if args.dry_run:
        print(prompt)
        return [{"section_id": i["sec"]["para_id"], "status": "dry-run", "input_chars": i["chars"]} for i in inputs]

    tools, tool_executor = None, None
    if not args.no_tools:
        allowed = sorted({b for b in (book_id, attha_ref, tika_ref) if b})
        if allowed:
            tools = [ai.make_get_text_range_tool(allowed, single_round=True)]
            tool_executor = make_tool_executor(epitaka_db, en_db, allowed)

    used_holder: list[str] = []
    raw = ai.call_ai_with_logging(
        rotator=rotator, prompt=prompt, book_id=book_id, chunk_id=chunk_id,
        log_dir=args.log_dir, model=args.model, models=models,
        system_prompt=BATCH_SYSTEM_PROMPT,
        tools=tools, tool_executor=tool_executor,
        max_output_tokens=args.max_output_tokens,
        max_tool_rounds=args.max_tool_rounds,
        used_model=used_holder,
    )
    if raw is None:
        print(f"  Batch @sec{lo_first} returned no response. Skipping {len(inputs)} section(s).")
        return [{"section_id": i["sec"]["para_id"], "status": "no-response", "input_chars": i["chars"]} for i in inputs]

    parsed = parse_batch_response(raw)
    model_used = used_holder[0] if used_holder else (models[0] if models else "")
    by_id = {p["section_id"]: p for p in parsed if p.get("section_id")}

    results = []
    for idx, inp in enumerate(inputs):
        lo = inp["sec"]["para_id"]
        entry = by_id.get(lo) or (parsed[idx] if idx < len(parsed) else None)
        if not entry or not entry.get("content"):
            print(f"  Section {lo}: missing/empty in batch response.")
            results.append({"section_id": lo, "status": "no-response", "input_chars": inp["chars"]})
            continue
        title = entry.get("title", "")
        content = entry["content"]
        save_summary(summary_db, {
            "book_id":      book_id,
            "section_id":   lo,
            "para_start":   lo,
            "para_end":     inp["sec"]["next"] - 1,
            "heading_title": inp["sec"]["heading_title"],
            "sutta_title":  inp["sec"]["sutta_title"],
            "vagga_title":  inp["sec"]["vagga_title"],
            "title":        title,
            "content":      content,
            "sources":      inp["sources"],
            "model":        model_used,
            "input_chars":  inp["chars"],
        })
        print(f"  Saved section {lo} — title: {title[:80]!r}, content {len(content):,} chars")
        results.append({"section_id": lo, "status": "saved", "input_chars": inp["chars"], "title": title})
    return results


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True, help="Comma-separated mūla book_ids, e.g. \"M-i\"")
    parser.add_argument("--epitaka-db",  default=EPITAKA_DB)
    parser.add_argument("--lang-db",     default=EN_DB, help="DB holding English translations (epitaka_en.db)")
    parser.add_argument("--summary-db",  default=SUMMARY_DB,
                        help="DB to write the summaries table into "
                             "(default: the English translation DB, epitaka_en.db)")
    parser.add_argument("--model",       default="",
                        help="single model to use (default: MODEL_FALLBACK_LIST, with "
                             "automatic fallback when a model's quota is exhausted)")
    parser.add_argument("--log-dir",     default=DEFAULT_LOG_DIR)
    parser.add_argument("--api-keys",    default="")
    parser.add_argument("--start-para",  type=int, default=0,  help="only sections whose heading para_id >= this")
    parser.add_argument("--end-para",    type=int, default=0,  help="only sections whose heading para_id <= this (0 = all)")
    parser.add_argument("--max-sections", type=int, default=-1, help="stop after N sections per book")
    parser.add_argument("--batch-size",   type=int, default=3,
                        help="sections processed per AI call (default 3). Each call = 1 API request "
                             "(+1 if the model uses its single tool round)")
    parser.add_argument("--batch-max-chars", type=int, default=BATCH_MAX_CHARS,
                        help="max combined prompt bytes per batch; oversized batches are split")
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--max-input-chars", type=int, default=MAX_INPUT_CHARS)
    parser.add_argument("--max-output-tokens", type=int, default=65_536,
                        help="max output tokens per AI response (model ceiling is 65536)")
    parser.add_argument("--max-tool-rounds", type=int, default=1,
                        help="max get_text_range tool ROUNDS per batch — each round is a full API "
                             "request, and the final answer costs one more, so a batch uses at most "
                             "rounds+1 requests total (default 1 → ≤2 requests/batch)")
    parser.add_argument("--no-tools",     action="store_true",
                        help="disable the get_text_range tool → exactly 1 API request per batch")
    args = parser.parse_args()

    if args.book == 'preset':
        args.book = PRESET_BOOKS
    book_list = [b.strip() for b in args.book.split(",") if b.strip()]
    if not book_list:
        print("No books specified.")
        return 1

    epitaka_db = args.epitaka_db
    en_db      = args.lang_db
    summary_db = args.summary_db

    if not Path(en_db).exists():
        print(f"[ERROR] English translation DB not found: {en_db}")
        return 1
    if not Path(epitaka_db).exists():
        print(f"[ERROR] epitaka DB not found: {epitaka_db}")
        return 1

    if not args.dry_run:
        ensure_summary_db(summary_db)

    explicit_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    rotator = ai.make_rotator(explicit_keys)

    models = [args.model] if args.model else MODEL_FALLBACK_LIST
    print(f"[study] Models      : {' → '.join(models)}")
    print(f"[study] Source DB   : {epitaka_db}")
    print(f"[study] Translation : {en_db}")
    print(f"[study] Summary DB  : {summary_db}")
    print(f"[study] Max input   : {args.max_input_chars:,} bytes/section")
    print(f"[study] Max output  : {args.max_output_tokens:,} tokens/response")
    print(f"[study] Batch       : {args.batch_size} section(s)/call "
          f"(max {args.batch_max_chars:,} chars/batch)")
    print(f"[study] Tools       : get_text_range "
          + ("(disabled — exactly 1 request per batch)" if args.no_tools
             else f"(enabled, ≤{args.max_tool_rounds} tool round(s)/batch "
                  f"→ ≤{args.max_tool_rounds + 1} requests/batch)"))

    grand = {"saved": 0, "skipped": 0, "dry": 0, "no_response": 0}

    for book_idx, book_id in enumerate(book_list, 1):
        print(f"\n{'#' * 60}\n# BOOK {book_idx}/{len(book_list)}: {book_id}\n{'#' * 60}")

        with _connect(epitaka_db) as conn:
            brow = conn.execute("SELECT book_name, attha_ref, tika_ref FROM books WHERE book_id=?", (book_id,)).fetchone()
            if not brow:
                print(f"[ERROR] Unknown book_id {book_id}. Skipping.")
                continue
            book_name = brow["book_name"] or book_id
            attha_ref = brow["attha_ref"] or ""
            tika_ref  = brow["tika_ref"] or ""
            sections  = fetch_sections(conn, book_id)

        print(f"  {len(sections)} level-{SECTION_HEADING_LEVEL} section(s) found. "
              f"attha_ref={attha_ref or '-'} tika_ref={tika_ref or '-'}")

        if args.start_para:
            sections = [s for s in sections if s["para_id"] >= args.start_para]
        if args.end_para:
            sections = [s for s in sections if s["para_id"] <= args.end_para]

        # Collect the sections that still need processing (already-summarised
        # ones are skipped), keeping the original --max-sections semantics
        # (counts every encountered section, skipped or not).
        todo = []
        for idx, sec in enumerate(sections, 1):
            if args.max_sections != -1 and idx > args.max_sections:
                print(f"  Reached --max-sections={args.max_sections}; stopping book.")
                break
            lo = sec["para_id"]
            if (not args.dry_run and not args.overwrite
                    and summary_exists(summary_db, book_id, lo)):
                grand["skipped"] += 1
                continue
            todo.append(sec)

        n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
        for b_idx in range(0, len(todo), args.batch_size):
            batch = todo[b_idx:b_idx + args.batch_size]
            print(f"\n[BATCH {b_idx // args.batch_size + 1}/{n_batches}] "
                  f"{len(batch)} section(s) @paras {[s['para_id'] for s in batch]}")

            try:
                results = process_batch(
                    book_id, book_name, batch, args, rotator,
                    epitaka_db, en_db, summary_db, attha_ref, tika_ref,
                    models,
                )
            except Exception as exc:
                print(f"  [ERROR] batch @sec{batch[0]['para_id']} failed: {exc}")
                continue

            for res in results:
                status = res.get("status")
                if status == "saved":
                    grand["saved"] += 1
                elif status == "no-response":
                    grand["no_response"] += 1
                elif status == "dry-run":
                    grand["dry"] += 1

    print("\n" + "=" * 60)
    print(f"DONE. Books: {len(book_list)}  saved: {grand['saved']}  "
          f"skipped(existing): {grand['skipped']}  no-response: {grand['no_response']}  dry: {grand['dry']}")
    print(f"Summaries written to: {summary_db}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
