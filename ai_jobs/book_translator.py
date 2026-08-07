"""
book_translator.py — Standalone "translate a whole book, part by part" runner.

    # Translate books:
    python book_translator.py --lang en --books Sp-i,Sp-ii --start 615 --end 700 \
        --part-size 4 --max-tokens 3000 --log-dir /tmp/book_logs

    # Use the preset book list:
    python book_translator.py --lang en --books preset

    # Build glossary only from already-translated books (no new translations):
    python book_translator.py --lang vi --books Sp-i,Sp-ii --glossary-only

What it does
------------
Accepts a --lang code (e.g. "en", "si", "th", "vi") and one or more book_ids.
Translations are saved to epitaka_<lang>.db (same directory as epitaka.db).
Glossary terms are saved to glossary_<lang>.db (same directory).
Once saved there, the corresponding fields are cleared from epitaka.db.

epitaka_<lang>.db/sentences schema:
  book_id, para_id, line_id  — primary key
  translation                — the translated text
  translation_confidence     — "high" | "low"
  confidence_note            — reason when low

glossary_<lang>.db/glossary schema:
  pali (stem form), translation, domain, sub_domain, context, note,
  source_id, para_id_start, para_id_end  — provenance range in the source book

For each book it:

  1. Finds every paragraph whose (book_id, para_id, line_id) is not yet
     present in epitaka_<lang>.db (unless --overwrite is given).
  2. Groups paragraphs into sections based on headings (merged to min_lines).
  3. For each section, splits into token-safe chunks and builds a prompt using
     context_builders.py:

        - GlossaryContext        (reads from glossary_<lang>.db)
        - GlossaryContextStemmed
        - CommentaryContext
        - PaliDefsContext
        - PreviousTranslationContext
        - MulaAtthaContext
        - NissayaContext

  4. Calls Gemini, parses the JSON response:
     {"translations": [...], "glossary": [...], "remarks": [...]}

     Each translation carries:
       "confidence": "high" | "low"
       "confidence_note": "<reason if low>"

     Each glossary entry carries:
       "pali": stem form of the Pāli word
       "translation": translation in the target language
       "domain", "sub_domain", "context", "note"

  5. Saves:
        - translations + confidence → epitaka_<lang>.db / sentences
        - remarks                   → epitaka_<lang>.db / translation_remarks
        - glossary                  → glossary_<lang>.db / glossary
     Then clears translation / translation_confidence / confidence_note
     and removes matching translation_remarks rows from epitaka.db.

  --glossary-only mode:
     Reads existing pali + translations from epitaka_<lang>.db for the
     given books and sends them to Gemini to extract glossary terms only.
     No new translations are produced or saved.

File layout
-----------
This script owns everything specific to "translate a book": heading-based
sectioning, token-safe chunking, the translation prompt/schema, and saving
translations/remarks. Infrastructure shared with glossary_builder.py (DB
paths, schema, glossary upserts, Pāli stem lookup) lives in common_utils.py.
Everything about *how* we talk to the AI (Gemini calls, key rotation,
retries, response-JSON parsing) lives in ai_client.py — change that file,
not this one, when the AI logic needs to change.
"""

import argparse
import json
import os
import sys
import types
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

import common.common_utils as cu
import common.ai_client as ai

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# PRESET BOOK LIST
# ══════════════════════════════════════════════════════════════════

PRESET_BOOKS = (
    "D-i,D-ii,D-iii,M-i,M-ii,M-iii,S-i,S-ii,S-iii,S-iv,S-v,A-i,A-ii,A-iii,A-iv,A-v,A-vi,A-vii,"
    "A-viii,A-ix,A-x,A-xi,Khp,Dhp,Ud,It,Sn,Vv,Pv,Th,Thī,Ap-i,Ap-ii,Ap-iii,Bv,Cp,Ja-i,Ja-ii,Nidd-i,"
    "Nidd-ii,Paṭis,Nett,Mil,Pet,Vin-i,Vin-ii,Vin-ii-b,Vin-iii,Vin-iv,Vin-v,Dvem-bhk,Dvem-bhni,Dhs,"
    "Vibh,Dhatuk,Pp,Kv,Yam-i,Yam-ii,Yam-iii,Paṭṭh-i,Paṭṭh-ii,Paṭṭh-iii,Paṭṭh-iv,Paṭṭh-v,Sv-i,Sv-ii,"
    "Sv-iii,Ps-i,Ps-ii,Ps-iii,Spk-i,Spk-ii,Spk-iii,Spk-iv,Spk-v,Mp-i,Mp-ii,Mp-v,Mp-viii,Mp-iii,Mp-iv,"
    "Mp-vi,Mp-vii,Mp-ix,Mp-x,Mp-xi,Pj-i,Dhp-a,Ud-a,It-a,Pj-ii,Vv-a,Pv-a,Th-a-i,Th-a-ii,Thī-a,Ap-a,Bv-a,"
    "Cp-a,Ja-a-i,Ja-a-ii,Ja-a-iii,Ja-a-iv,Ja-a-v,Ja-a-vi-b,Ja-a-vi,Ja-a-vii,Nidd-a-i,Nidd-a-ii,"
    "Paṭis-a,Nett-a,Sp-i,Sp-ii,Sp-ii-b,Sp-iii,Sp-iv,Sp-v,Kkh,VinSaṅg-a,As,Vibh-a,Dhatuk-a,Pp-a,Kv-a,"
    "Yam-a,Paṭṭh-a,Sv-pt-i,Sv-pt-ii,Sv-pt-iii,Sv-nt-i,Sv-nt-ii,Ps-t-i,Ps-t-ii,Ps-t-iii,Spk-t-i,"
    "Spk-t-ii,Spk-t-iii,Spk-t-iv,Spk-t-v,Mp-t-i,Mp-t-ii,Mp-t-iii,Mp-t-iv,Mp-t-v,Mp-t-vi,Mp-t-vii,"
    "Mp-t-viii,Mp-t-ix,Mp-t-x,Mp-t-xi,Nett-ṭ,Nett-vbh,Sp-t-i-a,Sp-t-i-b,Sp-t-ii,Sp-t-iii,Sp-t-iv,"
    "Sp-t-v,Vjb-prj,Vjb-pac,Vjb-bkn,Vjb-mv,Vjb-cv,Vjb-pr,Kkh-pt,Vmv-i,Vmv-ii,Vmv-iii,Vmv-iv,Vmv-v,"
    "Vin-vn,Vin-vn-t,Pācity-y,Khuddas,Utt-vn,Utt-vn-t,Khuddas-pt,Khuddas-nt,Mūlasikk,Mūlasikk-t,"
    "Kkh-nt,As-mt,Vibh-mt,Dhatuk-mt,Dhs-anuṭ,Dhatuk-anuṭ,Abhidh-av,Abhidh-s,Abhidh-av-pt,AbhMāt,"
    "Vibh-anuṭ,Pp-mt,Kv-mt,Yam-mt,Paṭṭh-mt,Pp-anuṭ,Kv-anuṭ,Yam-anuṭ,Paṭṭh-anuṭ,Namar-p,Abhidh-av-vinich,"
    "Abhidh-av-sacc,Abhidh-s-t,Abhidh-av-nt,Moh,Vism-i,Vism-ii,Vism-mht-i,Vism-mht-ii,Vism-nid,DN-pv,"
    "MN-pv,SN-pv,AN-pv,Vin-pv,Abh-pv,Att-pv,Nirud,PmtDīp,Anudīp,Paṭṭhuddes,Nāmakkp,MhPaṇām,LakkhBudth,"
    "Sutvan,Jinal,Kamal,Pajjm,Buddhaguṇ,Cūḷgv,Sas,Mhv,Mogg,Kacc,Sadd-pad,Sadd-dh,PadRūp,MoggPañc,Payog,"
    "Vutt,Abh,Abh-ṭ,Subodh,Subodh-t,Bālāv,Kavid,Nītim,Dhammn,Mhran,Lokan,Suttn,Sūrn,Cāṇn,Narad,Catur,"
    "Rasav,Sīmav,Vessg,MoggVutt,Thup,Dat.h,Dhātpvil,Dhātv,Hattv,Jina-c,Jinvdīp,Tel,Mil-t,Padamañj,Padsādh,"
    "Saddbind,Dhatup,Samantak,Nāmakkṭ,Tigumb,Vāsamāl,Mogg-byk,Kacc-sadd,Vasala,Vin-alaṅke,"
)


# ══════════════════════════════════════════════════════════════════
# CONFIG (script-specific: shared paths/lang-names/glossary logic live in
# common_utils.py; AI defaults live alongside the AI code in ai_client.py)
# ══════════════════════════════════════════════════════════════════

EPITAKA_DB   = cu.EPITAKA_DB
GLOSSARY_DB  = cu.GLOSSARY_DB
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
DEFAULT_LOG_DIR = "/tmp/book_translator_logs"

# Prompts above this size (system + user, UTF-8 bytes) trigger the
# size-reduction cascade below. Below this size nothing is touched — full
# commentary, full parallel-translation set, full chunk — so normal-sized
# requests never lose any quality.
PROMPT_SIZE_LIMIT_BYTES = int(os.environ.get("PROMPT_SIZE_LIMIT_BYTES", 1_000_000))

# Heading-based sections often end up with very little *pending* content each
# (most of the section may already be translated), which used to mean one
# tiny AI call per section. Adjacent sections are merged into one bigger
# batch as long as the batch's pending content stays under these two caps.
SECTION_MERGE_MAX_BYTES = int(os.environ.get("SECTION_MERGE_MAX_BYTES", 300_000))
SECTION_MERGE_MAX_LINES = int(os.environ.get("SECTION_MERGE_MAX_LINES", 50))


# ══════════════════════════════════════════════════════════════════
# Stub modules
# ══════════════════════════════════════════════════════════════════
#
# NOTE: context_builders.py may do `from database import get_glossary_conn`,
# which binds the *current* function object at import time. If we set a
# generic lambda here and only patch sys.modules["database"].get_glossary_conn
# later (in main(), once --lang is parsed), any module that imported the name
# directly will keep using this early lambda forever — i.e. it will keep
# reading from the default glossary.db no matter what --lang is. So we must
# resolve --lang / --epitaka-db / --glossary-db *before* importing
# common.context_builders, and bake the correct lang-specific path into the
# stub from the very first definition.

def _early_arg(flag: str, default: str = "") -> str:
    """Cheap pre-parse of a single --flag value/=value from argv, before argparse runs."""
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


_early_lang        = _early_arg("--lang", "")
_early_epitaka_db  = _early_arg("--epitaka-db", EPITAKA_DB)
_early_glossary_db = _early_arg("--glossary-db", "")

if _early_glossary_db:
    _RESOLVED_GLOSSARY_DB = _early_glossary_db
elif _early_lang:
    _RESOLVED_GLOSSARY_DB = cu.glossary_db_path(_early_epitaka_db, _early_lang)
else:
    # No --lang available yet (e.g. introspection/help); fall back to default.
    _RESOLVED_GLOSSARY_DB = GLOSSARY_DB

if "database" not in sys.modules:
    _db_mod = types.ModuleType("database")
    _db_mod.get_glossary_conn = lambda: sqlite3.connect(_RESOLVED_GLOSSARY_DB, timeout=30)
    sys.modules["database"] = _db_mod
else:
    # Module already present (e.g. re-imported) — still make sure it points
    # at the language-specific glossary DB rather than whatever it had before.
    sys.modules["database"].get_glossary_conn = lambda: sqlite3.connect(_RESOLVED_GLOSSARY_DB, timeout=30)

if "config" not in sys.modules:
    _cfg_mod = types.ModuleType("config")
    _cfg_mod.EPITAKA_DB  = _early_epitaka_db
    _cfg_mod.SC_DATA_DB  = ""
    sys.modules["config"] = _cfg_mod
else:
    sys.modules["config"].EPITAKA_DB = _early_epitaka_db

from common.context_builders import (  # noqa: E402
    GlossaryContext,
    GlossaryContextStemmed,
    CommentaryContext,
    PaliDefsContext,
    PreviousTranslationContext,
    MulaAtthaContext,
    NissayaContext,
    TranslationWriter,
    GlossaryWriter,
    RemarkWriter,
)


# ══════════════════════════════════════════════════════════════════
# DB helpers (translation-specific; shared schema/connect helpers are in
# common_utils.py)
# ══════════════════════════════════════════════════════════════════

_connect = cu.connect  # local alias, kept short since it's used throughout this file


def fetch_headings(epitaka_db: str, book_id: str) -> list[dict]:
    """All headings for a book, in document order — used to draw section boundaries."""
    with _connect(epitaka_db) as conn:
        rows = conn.execute(
            "SELECT para_id, level, title, chapter_len "
            "FROM headings WHERE book_id=? ORDER BY para_id",
            (book_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_paragraphs_range(
    epitaka_db: str,
    book_id:    str,
    para_start: int,
    para_end:   int,
    overwrite:  bool,
    lang_db:    str | None = None,
) -> list[dict]:
    """
    Load every paragraph in [para_start, para_end] for `book_id`, each
    annotated with which of its sentences are still "pending" translation.

    A sentence counts as pending when:
      - --overwrite was given (everything is pending, regardless of lang_db), or
      - it has no non-empty translation yet in lang_db, AND its Pāli text is
        at least 3 characters (skips bare punctuation/number placeholder lines).

    Paragraphs with zero pending sentences are dropped entirely so downstream
    sectioning/chunking never has to special-case "nothing to do here".
    """
    # Collect already-translated (para_id, line_id) pairs from lang_db, if supplied
    already_translated: set[tuple[int, int]] = set()
    if lang_db and not overwrite and Path(lang_db).exists():
        with _connect(lang_db) as lconn:
            lang_rows = lconn.execute(
                "SELECT para_id, line_id FROM sentences "
                "WHERE book_id=? AND para_id BETWEEN ? AND ? "
                "AND translation IS NOT NULL AND translation != ''",
                (book_id, para_start, para_end),
            ).fetchall()
            already_translated = {(r["para_id"], r["line_id"]) for r in lang_rows}

    with _connect(epitaka_db) as conn:
        rows = conn.execute(
            "SELECT DISTINCT para_id FROM sentences "
            "WHERE book_id=? AND para_id BETWEEN ? AND ? ORDER BY para_id",
            (book_id, para_start, para_end),
        ).fetchall()
        para_ids = [r["para_id"] for r in rows]

        result = []
        for pid in para_ids:
            srows = conn.execute(
                "SELECT line_id, pali "
                "FROM sentences WHERE book_id=? AND para_id=? ORDER BY line_id",
                (book_id, pid),
            ).fetchall()
            sentences = [dict(r) for r in srows]
            if overwrite:
                pending = sentences
            else:
                pending = [
                    s for s in sentences
                    if (pid, s["line_id"]) not in already_translated
                    and len((s["pali"] or "").strip()) >= 3
                ]
            if pending:
                result.append({
                    "book_id":   book_id,
                    "para_id":   pid,
                    "sentences": sentences,
                    "pending":   pending,
                })
    return result


def count_lines_range(epitaka_db: str, book_id: str, para_start: int, para_end: int) -> int:
    """Total sentence count (translated or not) across a paragraph range — used for section sizing."""
    with _connect(epitaka_db) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM sentences "
            "WHERE book_id=? AND para_id BETWEEN ? AND ?",
            (book_id, para_start, para_end),
        ).fetchone()
    return row["n"] if row else 0


def build_sections_from_headings(
    epitaka_db:   str,
    book_id:      str,
    para_start:   int,
    para_end:     int,
    min_lines:    int,
    overwrite:    bool,
    lang_db:      str | None = None,
) -> list[list[dict]]:
    """
    Split [para_start, para_end] into sections aligned to heading boundaries,
    merging consecutive headings together until each section has at least
    `min_lines` total sentences (so we don't fire one AI call per tiny
    sub-heading). Returns a list of sections, each a list of paragraph dicts
    (as produced by fetch_paragraphs_range).
    """
    headings = fetch_headings(epitaka_db, book_id)

    with _connect(epitaka_db) as conn:
        row = conn.execute(
            "SELECT MAX(para_id) AS m FROM sentences WHERE book_id=?", (book_id,)
        ).fetchone()
        book_max = row["m"] if row and row["m"] is not None else para_start

    if para_end == -1:
        para_end = book_max

    boundaries = [h["para_id"] for h in headings if para_start <= h["para_id"] <= para_end]
    if not boundaries or boundaries[0] > para_start:
        boundaries.insert(0, para_start)
    boundaries.append(para_end + 1)
    boundaries = sorted(set(boundaries))

    merged_ranges: list[tuple[int, int]] = []
    pending_start: int | None = None
    pending_lines: int = 0

    for i in range(len(boundaries) - 1):
        sec_start = boundaries[i]
        sec_end   = boundaries[i + 1] - 1

        if pending_start is None:
            pending_start = sec_start

        n = count_lines_range(epitaka_db, book_id, sec_start, sec_end)
        pending_lines += n

        is_last = (i == len(boundaries) - 2)
        if pending_lines >= min_lines or is_last:
            merged_ranges.append((pending_start, sec_end))
            pending_start = None
            pending_lines = 0

    sections: list[list[dict]] = []
    for (sec_start, sec_end) in merged_ranges:
        paras = fetch_paragraphs_range(epitaka_db, book_id, sec_start, sec_end, overwrite, lang_db=lang_db)
        if paras:
            sections.append(paras)

    return sections


def merge_small_sections(
    sections: list[list[dict]],
    max_bytes: int = SECTION_MERGE_MAX_BYTES,
    max_lines: int = SECTION_MERGE_MAX_LINES,
) -> list[list[dict]]:
    """Merge consecutive heading-based sections together so one AI call can
    cover more pending content, instead of firing a separate call per tiny
    section (common when most of a section is already translated and only
    a handful of new sentences remain).

    Sections are appended to the current running batch as long as doing so
    keeps the batch's pending content under BOTH `max_bytes` (UTF-8 bytes of
    the pending Pāli text) and `max_lines` (pending sentence count). As soon
    as adding the next section would exceed either cap, the current batch is
    closed off and a new one starts. A single section that already exceeds
    the caps by itself is kept as its own batch (chunk_paragraphs/_handle_chunk
    further down still applies token-based and byte-based splitting on top
    of this, so nothing overflows).
    """
    if not sections:
        return sections

    merged: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    current_lines = 0

    for section in sections:
        sec_bytes = sum(
            len(s["pali"].encode("utf-8"))
            for para in section
            for s in para["pending"]
        )
        sec_lines = sum(len(para["pending"]) for para in section)

        if current and (
            current_bytes + sec_bytes > max_bytes
            or current_lines + sec_lines > max_lines
        ):
            merged.append(current)
            current = []
            current_bytes = 0
            current_lines = 0

        current.extend(section)
        current_bytes += sec_bytes
        current_lines += sec_lines

    if current:
        merged.append(current)

    return merged


# ══════════════════════════════════════════════════════════════════
# Token-safe chunking
# ══════════════════════════════════════════════════════════════════

def _prompt_bytes(system_prompt: str, user_prompt: str) -> int:
    return len((system_prompt + user_prompt).encode("utf-8"))


# ══════════════════════════════════════════════════════════════════
# Script-bleed guard
# ══════════════════════════════════════════════════════════════════

# Unicode block ranges for target-language scripts we've seen the model
# confuse with a closely-related neighbour (Lao <-> Thai being the main
# offender: same family, Thai is far better-resourced in the model).
_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "lo": [(0x0E80, 0x0EFF)],   # Lao
    "th": [(0x0E00, 0x0E7F)],   # Thai
    "km": [(0x1780, 0x17FF)],   # Khmer
    "my": [(0x1000, 0x109F)],   # Myanmar
    "si": [(0x0D80, 0x0DFF)],   # Sinhala
}
# For each of the above, the "confusable" script(s) that must NOT dominate.
_CONFUSABLE_WITH: dict[str, list[str]] = {
    "lo": ["th"],
    "th": ["lo"],
}


def _script_mismatch(lang: str, text: str) -> str | None:
    """
    Returns a short warning string if `text` looks like it was written in a
    confusable neighbour script instead of `lang`'s own script. Returns None
    if the text looks fine (including for languages we don't check, or
    lines too short/symbol-only to judge).
    """
    own_ranges = _SCRIPT_RANGES.get(lang)
    confusables = _CONFUSABLE_WITH.get(lang)
    if not own_ranges or not confusables or not text:
        return None

    def _count_in_ranges(s: str, ranges: list[tuple[int, int]]) -> int:
        return sum(1 for ch in s if any(lo <= ord(ch) <= hi for lo, hi in ranges))

    own_count = _count_in_ranges(text, own_ranges)
    for other_lang in confusables:
        other_ranges = _SCRIPT_RANGES.get(other_lang, [])
        other_count = _count_in_ranges(text, other_ranges)
        # Only fire if there's a meaningful amount of "other" script and it
        # dominates (or entirely replaces) the target script.
        if other_count >= 5 and other_count > own_count:
            return f"looks like {other_lang!r} script, not {lang!r}"
    return None


def check_translations_for_script_bleed(lang: str, translations: list[dict]) -> list[dict]:
    """
    Scan a parsed 'translations' list for entries whose script doesn't match
    the target language (e.g. Lao target coming back in Thai script). Any
    flagged entries are forced to confidence='low' with a confidence_note,
    and returned separately so the caller can log/retry them.
    """
    flagged = []
    for t in translations:
        text = t.get("translation", "")
        if text in ("", "~"):
            continue
        reason = _script_mismatch(lang, text)
        if reason:
            t["confidence"] = "low"
            note = f"[SCRIPT-BLEED] {reason}"
            existing_note = t.get("confidence_note")
            t["confidence_note"] = f"{note}; {existing_note}" if existing_note else note
            flagged.append(t)
    return flagged


def _split_chunk_in_half(chunk: list[dict]) -> tuple[list[dict], list[dict]] | None:
    """
    Split a chunk (list of paragraph dicts, each carrying a 'pending'
    sentence list) into two smaller halves so a single API call has fewer
    sentences to translate. Returns None when the chunk is already as small
    as it can get (one paragraph, one pending sentence) — the caller should
    stop trying to shrink the sentence count at that point.
    """
    if len(chunk) > 1:
        mid = len(chunk) // 2
        return chunk[:mid], chunk[mid:]

    para = chunk[0]
    sentences = para["pending"]
    if len(sentences) <= 1:
        return None
    mid = len(sentences) // 2
    left  = {**para, "pending": sentences[:mid]}
    right = {**para, "pending": sentences[mid:]}
    return [left], [right]


def _para_tokens(para: dict) -> int:
    text = "\n".join(s.get("pali", "") for s in para.get("pending", []))
    return ai.estimate_tokens(text)


def _split_oversized_para(para: dict, max_tokens: int) -> list[dict]:
    """
    A single paragraph whose pending sentences alone exceed max_tokens can't
    be handled by para-level chunking (chunk_paragraphs only ever groups or
    breaks *between* paragraphs). Split its sentences into sub-paragraph
    pieces that each stay within budget, keeping the same book_id/para_id/
    sentences metadata so downstream code (chunk[0]["para_id"], etc.) still
    works.
    """
    pieces: list[dict] = []
    piece_sentences: list[dict] = []
    piece_tokens = 0

    for s in para.get("pending", []):
        s_tokens = ai.estimate_tokens(s.get("pali", ""))
        if piece_sentences and (piece_tokens + s_tokens > max_tokens):
            pieces.append({**para, "pending": piece_sentences})
            piece_sentences = []
            piece_tokens = 0
        piece_sentences.append(s)
        piece_tokens += s_tokens

    if piece_sentences:
        pieces.append({**para, "pending": piece_sentences})

    return pieces


def chunk_paragraphs(paragraphs: list[dict], max_tokens: int = 3000) -> list[list[dict]]:
    """Group paragraphs into token-budgeted chunks, splitting any single oversized paragraph on its own."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _para_tokens(para)

        if para_tokens > max_tokens:
            # This single paragraph is already too long by itself (long
            # comment/note, run-on line, etc.) — flush what's pending, then
            # split it into its own token-safe piece(s) at the sentence level
            # instead of letting it blow past max_tokens in one chunk.
            if current:
                chunks.append(current)
                current = []
                current_tokens = 0
            for piece in _split_oversized_para(para, max_tokens):
                chunks.append([piece])
            continue

        if current and (current_tokens + para_tokens > max_tokens):
            chunks.append(current)
            current = []
            current_tokens = 0

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append(current)

    return chunks


# ══════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════

def _build_system_prompt(lang: str) -> str:
    """Build the system prompt with the correct target language injected."""
    lang_name = cu.lang_name(lang)
    return f"""You are an expert scholar-translator of Pāli Buddhist literature
(canonical texts, commentaries [aṭṭhakathā] and sub-commentaries [ṭīkā]).

TARGET LANGUAGE: {lang_name}
All "translations" output MUST be in {lang_name} — and ONLY {lang_name}. The Pāli
source text is in Pāli; your job is to produce {lang_name} renderings that are both
ACCURATE and READABLE for a GENERAL but serious audience. Try to minimize the use
of pali term in translation except commonly accepted terms like nibbāna, tathāgata, etc.
All glossary "translation" fields must also be in {lang_name}. Return '~' for lines that are
number, signs, or things not to be translated.

⚠ LANGUAGE-BLEED WARNING: You will be shown reference translations in OTHER
languages (see block 6 below), which may include a language closely related to,
or sharing a script family with, {lang_name} (e.g. Thai reference text when
translating to Lao, Khmer, etc.). Those are reference material ONLY — for
meaning and terminology, never for wording. Do NOT let the wording, script, or
orthography of any reference language leak into your output. Every single
character you write in "translation" fields must belong to {lang_name}. Before
finalizing each translation, double-check it is not accidentally in a
different (even closely related) language.

You will be given several reference blocks:
  1. ESTABLISHED GLOSSARY — accumulated translation memory containing
   previously selected Pāli → {lang_name} renderings. Maintain consistency with
   these terms unless the context requires a different meaning. A line like
   "(3 more existing variant(s) for 'X' omitted — reuse one of the above
   rather than adding another)" means the term already has several accepted
   renderings; pick the closest one instead of proposing a new variant.
  2. PALI COMMENTARY & SUB-COMMENTARY — aṭṭhakathā / ṭīkā explaining these lines.
  The Mūla may also included, if the word in commnetary is a definition of the word in mūla
  use the translation in mūla for that pali term.
  3. PALI WORD DEFINITIONS       — definition of a word in other area in tipitaka, 
  it may not related to the term being translate.
  4. PREVIOUS PARAGRAPH          — the immediately preceding paragraph's translation,
                                    for tone/terminology continuity.
  5. TRANSLATED MŪLA / AṬṬHAKATHĀ / ṬĪKĀ REFERENCES — other already-translated
                                    paragraphs linked to this passage.
  6. MYANMAR NISSAYA              — word-by-word gloss (romanised) for each sentence.
  7. SENTENCES TO TRANSLATE      — JSON array of Pāli sentences (para_id + line_id).

Return ONE JSON object with exactly three keys: "translations", "glossary",
and "remarks".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A. "translations" — array, ONE entry per input sentence, SAME ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {{
    "para_id": <int>,
    "line_id": <int>,
    "translation": "<text in {lang_name}>",
    "confidence": "high" | "low",
    "confidence_note": "<brief reason — ONLY when confidence is low, else omit>"
  }}

CONFIDENCE RULES — be honest, not conservative:

  Mark confidence "low" when ANY of the following apply:
    • The sentence contains rare compounds, technical terms, or ambiguous
      syntax that the commentary/nissaya does not clearly resolve.
    • The nissaya for this sentence is missing or incomplete.
    • The commentary/ṭīkā directly contradicts or is inconsistent with what
      another parallel translation says, and you had to make a judgment call.
    • A Pāli compound or term has multiple plausible meanings and context
      does not clearly decide between them.
    • The sentence is part of a highly technical Abhidhamma, Vinaya procedure,
      or grammatical passage where a mistake is easy and context is thin.

  Mark confidence "high" when:
    • The sentence is straightforward prose or verse with clear vocabulary.
    • OR the commentary/nissaya clearly resolves any difficult points.

  Do NOT mark everything "low" out of caution. Simple sentences with no
  ambiguity should be "high".
  The confidence field is for the human reviewer, not a disclaimer.

Translation style — read carefully:

  • Write natural, idiomatic {lang_name} that a literate non-specialist
    can follow. Prefer clear prose over a word-for-word rendering, but never
    drift from the actual meaning of the Pāli.
  • Use the PALI COMMENTARY (and ṭīkā, if present) as the primary authority
    for understanding difficult meanings, compounds, technical terms, and
    ambiguous syntax. The nissaya and word definitions support the commentary.
  • When translating COMMENTARIES or SUB-COMMENTARIES:
      - If the commentary explains or comments on a word, phrase, or technical
        term from the source text, use the established {lang_name} translation of
        that source term if it is provided inside the commentary translation
        references.
      - Preserve the terminology relationship between the commented word and
        the explanation.
      - The translation of the commentary should remain consistent with the
        translation of the original passage being explained.
  • Apply every ESTABLISHED GLOSSARY term/phrase exactly as given, including
    multi-word phrases.
  • Reference the PREVIOUS PARAGRAPH and TRANSLATED REFERENCES for consistency
    of terminology, names, and register.
  • Preserve important doctrinal distinctions between related Pāli terms.
    Do not merge different technical concepts merely because words overlap.
  • Keep the html tags like <b>, <i> in the translation same as original pali.
  • For the definition of a word (word in <b> wrapped in pali), make a translation
    for that term based on translated text from "PALI COMMENTARY & SUB-COMMENTARY"
    block if it has, or from glossary. The pali will be quoted in this style after
    the translation (<i>pali term</i>)
  • No verse numbers, footnotes, sentence numbering, or meta-commentary —
    output only the translated text for each sentence.
  • If a number provided only (eg. "20."), just returns an empty string "".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B. "glossary" — NEW TRANSLATION TERMS FOR FUTURE CONSISTENCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {{ "pali": "…", "translation": "…", "domain": "…",
    "sub_domain": "…", "context": "…", "note": "…" }}

  domain ∈ {{sutta, vinaya, abhidhamma, grammar, story}}

  CRITICAL RULES for the "pali" field:
    • Always supply the STEM (dictionary headword) of the Pāli term, NOT the
      inflected form found in the text. E.g. use "bhikkhu" not "bhikkhūnaṃ";
      use "samādhi" not "samādhiṃ"; use "sīla" not "sīlāni".
    • If a compound, give the whole compound in its uninflected/stem form.

  CRITICAL RULES for the "translation" field:
    • Must be in {lang_name}.

  What TO include:
    • Technical terms that a translator could easily render inconsistently
      or confuse with a similar term (e.g. "samādhi" vs "samāpatti",
      "sīla" vs "vinaya", "paññā" vs "vijjā").
    • Named doctrinal concepts, proper nouns, and set terms specific to
      Buddhist philosophy or Vinaya procedure.
    • Terms whose {lang_name} rendering is non-obvious or debatable.

  What NOT to include:
    • Common grammatical particles and conjunctions such as:
      ca, va, vā, pi, api, tu, pana, hi, eva, kho, ti, iti, atha, yeva,
      hoti, hotu, honti, ahosi, atthi, natthi, kacci, kiṃ, na, mā, evaṃ,
      seyyathā, tattha, tatra, tato, yathā, tathā, idaṃ, ayaṃ, so, sā, yo, yā.
    • Plain verbs of being/doing with no doctrinal significance.
    • Any term already present in the ESTABLISHED GLOSSARY block.

  AVOID DUPLICATE / NEAR-SYNONYM ENTRIES — this matters:
    • Before adding a new glossary entry, check whether the ESTABLISHED
      GLOSSARY already has an entry for that Pāli stem (or a very close
      synonym rendering). If an existing entry already conveys the meaning
      adequately for this context, REUSE it — do not add another slightly
      different phrasing of the same rendering just because the wording here
      differs a little.
    • Only add a new entry when the meaning in THIS context is genuinely
      different from every existing entry for that term (e.g. a different
      sense of a polysemous word), not merely a stylistic rewording of the
      same sense.
    • When in doubt, prefer inferring the translation from the closest
      existing glossary entry over minting a new one. The glossary is a
      shared, growing resource — treat near-duplicates as noise to avoid,
      not as helpful additional context.

  DO NOT OVER-APPLY THE GLOSSARY:
    • The ESTABLISHED GLOSSARY is guidance for terminology consistency, not
      a mandatory verbatim substitution list. If forcing a glossary term
      into this specific sentence would make the {lang_name} read awkwardly
      or unnaturally, prefer natural, idiomatic phrasing and only keep the
      glossary term's core sense, not necessarily its exact wording.
    • This applies especially to common words that happen to have a glossary
      entry from a specific technical context — do not force that technical
      rendering onto every plain, non-technical occurrence of the word.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C. "remarks" — ONLY for genuine, worth-noting CONFLICTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {{ "para_id": <int>, "line_id": <int>, "pali": "<short excerpt>",
    "translation": "<the {lang_name} translation you chose>",
    "conflict": "<what the other source says, briefly>",
    "note": "<why you went with your choice, 1 short sentence>" }}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — critical
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown fences, no prose outside the JSON.
{{ "translations": [...], "glossary": [...], "remarks": [...] }}
"""


USER_TEMPLATE = """Book: {book_id}  —  paragraphs {para_start}–{para_end}

{glossary_block}

{commentary_block}

{pali_defs_block}

{prev_para_block}

{nissaya_block}

══════════════════════════════
SENTENCES TO TRANSLATE (JSON array)
══════════════════════════════
{sentences_json}
"""


def build_prompt(
    book_id:          str,
    para_start:       int,
    para_end:         int,
    chunk:            list[dict],
    glossary_block:   str,
    commentary_block: str,
    pali_defs_block:  str,
    prev_para_block:  str,
    mula_block:       str,
    nissaya_block:    str,
) -> tuple[str, list[dict]]:
    """Fill in USER_TEMPLATE for one chunk. Returns (prompt_text, flat_sentence_list)."""
    flat_sentences = []
    for para in chunk:
        for s in para["pending"]:
            flat_sentences.append({
                "para_id":       para["para_id"],
                "line_id":       s["line_id"],
                "pali": s["pali"],
            })

    prompt = USER_TEMPLATE.format(
        book_id          = book_id,
        para_start       = para_start,
        para_end         = para_end,
        glossary_block   = glossary_block,
        commentary_block = commentary_block,
        pali_defs_block  = pali_defs_block,
        prev_para_block  = prev_para_block,
        mula_block       = mula_block,
        nissaya_block    = nissaya_block,
        sentences_json   = json.dumps(flat_sentences, ensure_ascii=False, indent=2),
    )
    return prompt, flat_sentences


# ══════════════════════════════════════════════════════════════════
# Per-book processing
# ══════════════════════════════════════════════════════════════════

def process_book(
    book_id:        str,
    args,
    rotator:        ai.KeyRotator,
    epitaka_db:     str,
    lang_db:        str,
    glossary_db:    str,
    system_prompt:  str,
    translation_writer,
    glossary_writer,
    remark_writer,
) -> tuple[int, int, int]:
    """Translate one book. Returns (sentences_updated, glossary_added, remarks_saved)."""
    params = {"epitaka_db": epitaka_db, "lang_db": lang_db}

    print("=" * 60)
    print(f"BOOK: {book_id}  paras={args.start}..{'end' if args.end == -1 else args.end} "
          f"min_lines={args.min_lines} overwrite={args.overwrite}")
    print("=" * 60)

    sections = build_sections_from_headings(
        epitaka_db  = epitaka_db,
        book_id     = book_id,
        para_start  = args.start,
        para_end    = args.end,
        min_lines   = args.min_lines,
        overwrite   = args.overwrite,
        lang_db     = lang_db,
    )
    print(f"{len(sections)} section(s) from headings.")

    sections = merge_small_sections(sections)
    print(f"{len(sections)} section(s) after merging small ones together.")

    if not sections:
        print("Nothing to do.")
        return 0, 0, 0

    total_updated  = 0
    total_glossary = 0
    total_remarks  = 0

    for part_idx, part in enumerate(sections, 1):
        if args.max_parts != -1 and part_idx > args.max_parts:
            print(f"Reached --max-parts={args.max_parts}; stopping.")
            break

        pid_start = part[0]["para_id"]
        pid_end   = part[-1]["para_id"]
        n_pending = sum(len(p["pending"]) for p in part)

        print("-" * 60)
        print(f"[Section {part_idx}/{len(sections)}] paras {pid_start}-{pid_end} "
              f"({n_pending} pending sentence(s))")

        # NOTE: glossary_block / commentary_block / pali_defs_block used to be
        # built once here for the *whole section* (pid_start..pid_end) and
        # reused unchanged across every chunk. That meant when a chunk's
        # prompt came out oversized, splitting it in half only shrank the
        # blocks that are genuinely per-chunk (nissaya/mula/parallel/prev
        # para) -- the section-wide commentary/pali-defs/glossary payload
        # never got smaller, so oversized sections just kept recursively
        # splitting down to single-sentence chunks instead of converging,
        # burning far more API calls than before. They're now built
        # per-chunk inside _handle_chunk instead, scoped to that chunk's
        # own paragraph range, so splitting actually shrinks them too.

        chunks = chunk_paragraphs(part, max_tokens=args.max_tokens)
        print(f"  -> {len(chunks)} chunk(s)")

        def _handle_chunk(chunk: list[dict], depth: int = 0) -> tuple[int, int, int]:
            """
            Build the prompt for `chunk` and send it. If the assembled prompt
            comes out over PROMPT_SIZE_LIMIT_BYTES, apply reductions in order
            — each step only runs when the previous one wasn't enough, so a
            normal-sized prompt is completely untouched:

              1. Split the chunk in half and recurse. Fewer sentences per
                 call shrinks every context block (nissaya, previous
                 paragraph, mūla, commentary, word-defs, glossary),
                 since all of them are now scoped to the chunk's own
                 paragraph range rather than the whole section.
              2. Last resort — the chunk is already a single sentence and the
                 commentary / word-def context for that one sentence alone
                 is still oversized: hard-truncate those blocks for this
                 call only and log it clearly so it can be reviewed later.
            """
            chunk_start = chunk[0]["para_id"]
            chunk_end   = chunk[-1]["para_id"]
            ctx_start   = max(1, chunk_start - 1)
            ctx_end     = chunk_end + 1

            prev_para_block = PreviousTranslationContext(
                params, book_id, chunk_start,
                min_length=600, max_lookback=1500,
            ).build()
            mula_block     = MulaAtthaContext(params, book_id, ctx_start, ctx_end).build()
            nissaya_block  = NissayaContext(params, chunk).build()

            # Scoped to this chunk's own pending sentences (not the whole
            # section) so that splitting the chunk in half actually shrinks
            # these blocks too, instead of leaving a fixed section-wide
            # payload baked into every split.
            pali_text_for_chunk = "\n".join(
                s["pali"]
                for para in chunk
                for s in para["pending"]
            )
            glossary_block = GlossaryContextStemmed(params, pali_text_for_chunk).build()
            local_commentary_block = CommentaryContext(
                params, book_id, ctx_start, ctx_end, log_info=print, log_warn=print,
            ).build()
            local_pali_defs_block = PaliDefsContext(
                params, pali_text=pali_text_for_chunk, log_info=print, log_warn=print,
            ).build()

            def _build():
                return build_prompt(
                    book_id=book_id, para_start=chunk_start, para_end=chunk_end,
                    chunk=chunk, glossary_block=glossary_block,
                    commentary_block=local_commentary_block, pali_defs_block=local_pali_defs_block,
                    prev_para_block=prev_para_block, mula_block=mula_block,
                    nissaya_block=nissaya_block,
                )

            prompt, flat_sentences = _build()
            size = _prompt_bytes(system_prompt, prompt)

            if size > PROMPT_SIZE_LIMIT_BYTES:
                halves = _split_chunk_in_half(chunk)
                if halves is not None and depth < 8:
                    print(f"  [size] p{chunk_start}-{chunk_end}: still {size:,} bytes — "
                          f"splitting into 2 smaller chunks (fewer sentences per call)")
                    left, right = halves
                    u1, g1, r1 = _handle_chunk(left, depth + 1)
                    u2, g2, r2 = _handle_chunk(right, depth + 1)
                    return u1 + u2, g1 + g2, r1 + r2
                # Can't reduce sentence count any further (already one
                # sentence) — the static context itself is the problem.
                print(f"  [size] p{chunk_start}-{chunk_end}: {size:,} bytes with a single "
                      f"pending sentence — truncating commentary/word-def context "
                      f"as a last resort (translation quality may be affected here)")
                cap = PROMPT_SIZE_LIMIT_BYTES // 3
                local_commentary_block = local_commentary_block[:cap]
                local_pali_defs_block  = local_pali_defs_block[:cap]
                prompt, flat_sentences = _build()
                size = _prompt_bytes(system_prompt, prompt)

            n_sentences = len(flat_sentences)
            line_ids = [s["line_id"] for s in flat_sentences]
            chunk_label = f"p{chunk_start}-{chunk_end}_L{min(line_ids)}-{max(line_ids)}" if line_ids else f"p{chunk_start}-{chunk_end}"
            print(f"  Chunk {chunk_label}: {n_sentences} sentence(s) "
                  f"across {len(chunk)} para(s), {size:,} bytes")

            if args.dry_run:
                print(prompt)
                print(f"  [dry-run] {n_sentences} sentence(s) would be sent.")
                return 0, 0, 0

            raw = ai.call_ai_with_logging(
                rotator       = rotator,
                prompt        = prompt,
                book_id       = book_id,
                chunk_id      = chunk_label,
                log_dir       = args.log_dir,
                model         = args.model,
                system_prompt = system_prompt,
            )
            if raw is None:
                print(f"  Chunk {chunk_label} returned no response. Skipping.")
                return 0, 0, 0

            try:
                result = ai.parse_ai_json_response(raw, ("translations", "glossary", "remarks"))
            except Exception as exc:
                print(f"  parse_response failed for chunk {chunk_label}: {exc}. Skipping.")
                return 0, 0, 0

            translations = result.get("translations", [])
            new_terms    = result.get("glossary",     [])
            remarks      = result.get("remarks",      [])

            bleed_flagged = check_translations_for_script_bleed(args.lang, translations)
            if bleed_flagged:
                bad_ids = ", ".join(f"p{t.get('para_id')}L{t.get('line_id')}" for t in bleed_flagged)
                print(f"  ⚠ SCRIPT BLEED in chunk {chunk_label}: {len(bleed_flagged)} "
                      f"line(s) look like wrong-language output ({bad_ids}). "
                      f"Marked low-confidence; re-run with --overwrite for these paragraphs.")

            low_conf = sum(1 for t in translations if t.get("confidence") == "low")
            print(f"  Parsed: {len(translations)} translation(s) "
                  f"({low_conf} low-confidence, {len(bleed_flagged)} script-bleed), "
                  f"{len(new_terms)} glossary term(s), "
                  f"{len(remarks)} remark(s)")

            saved_trans = save_translations_to_lang_db(lang_db, book_id, translations)
            saved_rem   = save_remarks_to_lang_db(lang_db, book_id, remarks)

            saved_gloss = 0
            if new_terms:
                saved_gloss = cu.save_glossary_terms(
                    glossary_db   = glossary_db,
                    new_terms     = new_terms,
                    source_id     = book_id,
                    para_id_start = chunk_start,
                    para_id_end   = chunk_end,
                    epitaka_db    = epitaka_db,
                )

            return saved_trans, saved_gloss, saved_rem

        for c_idx, chunk in enumerate(chunks, 1):
            u, g, r = _handle_chunk(chunk)
            total_updated  += u
            total_glossary += g
            total_remarks  += r

        print(f"  Section done. Running: sentences={total_updated}, "
              f"glossary={total_glossary}, remarks={total_remarks}.")

    return total_updated, total_glossary, total_remarks


# ══════════════════════════════════════════════════════════════════
# Saving results (translation-specific; glossary saving is shared —
# see common_utils.save_glossary_terms)
# ══════════════════════════════════════════════════════════════════

def save_translations_to_lang_db(lang_db: str, book_id: str, translations: list[dict]) -> int:
    """Upsert translated sentences (+ confidence/confidence_note) into the lang-specific DB."""
    if not translations:
        return 0
    rows = [
        (
            book_id,
            t["para_id"],
            t["line_id"],
            t.get("translation") or "",
            t.get("confidence", "high"),
            t.get("confidence_note") or None,
        )
        for t in translations
        if "para_id" in t and "line_id" in t
    ]
    if not rows:
        return 0
    with _connect(lang_db) as conn:
        conn.executemany(
            """INSERT INTO sentences
               (book_id, para_id, line_id, translation, translation_confidence, confidence_note)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(book_id, para_id, line_id) DO UPDATE SET
                   translation            = excluded.translation,
                   translation_confidence = excluded.translation_confidence,
                   confidence_note        = excluded.confidence_note""",
            rows,
        )
        conn.commit()
    return len(rows)


def save_remarks_to_lang_db(lang_db: str, book_id: str, remarks: list[dict]) -> int:
    """Insert conflict remarks (produced alongside translations) into the lang-specific DB."""
    if not remarks:
        return 0
    rows = [
        (
            book_id,
            r.get("para_id"),
            r.get("line_id"),
            r.get("pali"),
            r.get("translation"),
            r.get("conflict"),
            r.get("note"),
            book_id,
        )
        for r in remarks
    ]
    with _connect(lang_db) as conn:
        conn.executemany(
            """INSERT INTO translation_remarks
               (book_id, para_id, line_id, pali, translation, conflict, note, source_id)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
    return len(rows)


def clear_from_epitaka_db(epitaka_db: str, book_id: str, translations: list[dict], remarks: list[dict]):
    """Nullify translation fields in epitaka.db for rows now saved in lang_db (frees up epitaka.db)."""
    if not translations:
        return

    trans_keys = [
        (book_id, t["para_id"], t["line_id"])
        for t in translations
        if "para_id" in t and "line_id" in t
    ]
    remark_keys = [
        (book_id, r.get("para_id"), r.get("line_id"))
        for r in remarks
        if r.get("para_id") is not None
    ]

    with _connect(epitaka_db) as conn:
        conn.executemany(
            """UPDATE sentences
               SET translation    = NULL,
                   translation_confidence = NULL,
                   confidence_note        = NULL
               WHERE book_id=? AND para_id=? AND line_id=?""",
            trans_keys,
        )
        if remark_keys:
            conn.executemany(
                """DELETE FROM translation_remarks
                   WHERE book_id=? AND para_id=? AND line_id=?""",
                remark_keys,
            )
        conn.commit()


def ensure_confidence_columns(epitaka_db: str):
    """Add translation_confidence and confidence_note columns to epitaka.db/sentences if absent."""
    with _connect(epitaka_db) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sentences)").fetchall()]
        added = []
        if "translation_confidence" not in cols:
            conn.execute("ALTER TABLE sentences ADD COLUMN translation_confidence TEXT")
            added.append("translation_confidence")
        if "confidence_note" not in cols:
            conn.execute("ALTER TABLE sentences ADD COLUMN confidence_note TEXT")
            added.append("confidence_note")
        if added:
            conn.commit()
            print(f"[DB] Added column(s): {', '.join(added)}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lang", required=True,
        help=(
            'Target language code, e.g. "en", "si", "th", "vi". '
            'Translations are saved to epitaka_<lang>.db; glossary to glossary_<lang>.db.'
        ),
    )
    parser.add_argument(
        "--books", required=True,
        help=(
            'Comma-separated book_ids to translate, e.g. "Sp-i,Sp-ii". '
            'Use "preset" to run the full preset list.'
        ),
    )
    parser.add_argument("--start",          type=int, default=1,   help="first para_id (applies to every book, default 1)")
    parser.add_argument("--end",            type=int, default=-1,  help="last para_id, -1 = end of book")
    parser.add_argument("--min-lines",      type=int, default=50)
    parser.add_argument("--max-tokens",     type=int, default=3000)
    parser.add_argument("--overwrite",      action="store_true")
    parser.add_argument("--epitaka-db",     default=EPITAKA_DB)
    parser.add_argument("--glossary-db",    default="",
                        help="Override path to glossary_<lang>.db (default: auto-derived next to epitaka.db).")
    parser.add_argument("--model",          default=GEMINI_MODEL)
    parser.add_argument("--api-keys",       default="")
    parser.add_argument("--log-dir",        default=DEFAULT_LOG_DIR)
    parser.add_argument("--max-parts",      type=int, default=-1,  help="stop after N parts per book (for testing)")
    parser.add_argument("--dry-run",        action="store_true")
    args = parser.parse_args()

    epitaka_db  = args.epitaka_db
    lang_db     = cu.lang_db_path(epitaka_db, args.lang)
    glossary_db = args.glossary_db or cu.glossary_db_path(epitaka_db, args.lang)

    global _RESOLVED_GLOSSARY_DB
    if glossary_db != _RESOLVED_GLOSSARY_DB:
        # Argparse saw something the early pre-parse didn't catch (e.g. --lang
        # given via a different form). Re-resolve so every consumer agrees.
        print(f"[lang] Re-resolving glossary DB: {_RESOLVED_GLOSSARY_DB} -> {glossary_db}")
    _RESOLVED_GLOSSARY_DB = glossary_db

    # Point the glossary connector stub at the language-specific DB. (This is
    # belt-and-suspenders: _RESOLVED_GLOSSARY_DB above is what the lambda
    # actually reads, but we also keep this attribute override for any code
    # that calls database.get_glossary_conn() via module-attribute lookup.)
    sys.modules["config"].EPITAKA_DB = epitaka_db
    sys.modules["database"].get_glossary_conn = lambda: sqlite3.connect(glossary_db, timeout=30)

    lang_name = cu.lang_name(args.lang)
    print(f"[lang] Language   : {args.lang}  ({lang_name})")
    print(f"[lang] Source DB  : {epitaka_db}")
    print(f"[lang] Target DB  : {lang_db}")
    print(f"[lang] Glossary DB: {glossary_db}")

    # Build system prompts
    system_prompt = _build_system_prompt(args.lang)

    # Resolve book list
    if args.books.strip().lower() == "preset":
        book_list = [b.strip() for b in PRESET_BOOKS.split(",") if b.strip()]
        print(f"Using preset: {len(book_list)} books.")
    else:
        book_list = [b.strip() for b in args.books.split(",") if b.strip()]

    if not book_list:
        print("No books specified.")
        return 1

    # Ensure DB structures exist
    if not args.dry_run:
        cu.ensure_lang_db(lang_db)
        cu.ensure_glossary_db(glossary_db)

    explicit_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    rotator = ai.make_rotator(explicit_keys)

    params = {"epitaka_db": epitaka_db, "lang_db": lang_db}
    translation_writer = TranslationWriter(params, log_info=print, log_warn=print, log_error=print)
    glossary_writer    = GlossaryWriter(log_info=print, log_warn=print, log_error=print)
    remark_writer      = RemarkWriter(params, log_info=print, log_warn=print, log_error=print)

    grand_sentences = 0
    grand_glossary  = 0
    grand_remarks   = 0

    for book_idx, book_id in enumerate(book_list, 1):
        print(f"\n{'#' * 60}")
        print(f"# BOOK {book_idx}/{len(book_list)}: {book_id}")
        print(f"{'#' * 60}")

        try:
            s, g, r = process_book(
                book_id            = book_id,
                args               = args,
                rotator            = rotator,
                epitaka_db         = epitaka_db,
                lang_db            = lang_db,
                glossary_db        = glossary_db,
                system_prompt      = system_prompt,
                translation_writer = translation_writer,
                glossary_writer    = glossary_writer,
                remark_writer      = remark_writer,
            )
        except Exception as exc:
            print(f"[ERROR] Book {book_id} failed: {exc}. Continuing with next book.")
            continue

        grand_sentences += s
        grand_glossary  += g
        grand_remarks   += r

        print(f"Book {book_id} done — sentences: {s}, glossary: {g}, remarks: {r}.")

    print("\n" + "=" * 60)
    print(f"ALL DONE.  Books processed: {len(book_list)}")
    print(f"  Total sentences updated : {grand_sentences}")
    print(f"  Total glossary terms    : {grand_glossary}")
    print(f"  Total remarks saved     : {grand_remarks}")
    print("=" * 60)

    ai.send_telegram(
        f"<b>book_translator finished</b>\n"
        f"Lang: {args.lang}\n"
        f"Books: {len(book_list)}\n"
        f"Sentences: {grand_sentences}\n"
        f"Glossary: {grand_glossary}\n"
        f"Remarks: {grand_remarks}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())