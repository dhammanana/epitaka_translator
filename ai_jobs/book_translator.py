"""
book_translator.py — Standalone "translate a whole book, part by part" runner.

    python book_translator.py --lang en --books Sp-i,Sp-ii --start 615 --end 700 \
        --part-size 4 --max-tokens 3000 --log-dir /tmp/book_logs

    # Or use the preset book list:
    python book_translator.py --lang en --books preset

What it does
------------
Accepts a --lang code (e.g. "en", "si", "th") and one or more book_ids.
Translations are saved to epitaka_<lang>.db (same directory as epitaka.db).
Once saved there, the corresponding fields are cleared from epitaka.db.

epitaka_<lang>.db/sentences schema:
  book_id, para_id, line_id  — primary key
  translation                — the translated text
  translation_confidence     — "high" | "low"
  confidence_note            — reason when low

For each book it:

  1. Finds every paragraph whose (book_id, para_id, line_id) is not yet
     present in epitaka_<lang>.db (unless --overwrite is given).
  2. Groups paragraphs into sections based on headings (merged to min_lines).
  3. For each section, splits into token-safe chunks and builds a prompt using
     context_builders.py:

        - GlossaryContext
        - CommentaryContext
        - PaliDefsContext
        - PreviousTranslationContext
        - MulaAtthaContext
        - ParallelTranslationContext
        - NissayaContext

  4. Calls Gemini, parses the JSON response:
     {"translations": [...], "glossary": [...], "remarks": [...]}

     Each translation carries:
       "confidence": "high" | "low"
       "confidence_note": "<reason if low>"

  5. Saves:
        - translations + confidence → epitaka_<lang>.db / sentences
        - remarks                   → epitaka_<lang>.db / translation_remarks
        - glossary                  → GlossaryWriter (glossary.db)
     Then clears english_translation / translation_confidence / confidence_note
     and removes matching translation_remarks rows from epitaka.db.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# PRESET BOOK LIST
# ══════════════════════════════════════════════════════════════════

PRESET_BOOKS = (
    # "Dhp-a,Ja-a-i,A-i,A-iii,A-iv,A-v,A-vii,A-x,Ap-a,Ap-i,D-i,D-ii,D-iii,Dhp,It,It-a,"
    # "Ja-a-ii,Ja-a-iv,Ja-a-vi,Ja-a-vii,Ja-i,Ja-ii,Khp,M-i,M-ii,M-iii,Mp-i,Mp-iii,Mp-iv,"
    # "Mp-v,Mp-vii,Mp-x,Nidd-a-i,Nidd-a-ii,Nidd-i,Nidd-ii,Paṭis,Paṭis-a,Pj-i,Pj-ii,Ps-i,"
    # "Ps-ii,Ps-iii,Pv,Pv-a,S-i,S-ii,S-iii,S-iv,S-v,Sn,Sp-i,Sp-ii,Sp-iii,Sp-iv,Sp-v,"
    # "Spk-i,Spk-ii,Spk-iii,Spk-iv,Spk-v,Sv-i,Sv-ii,Sv-iii,Th,Th-a-i,Thī,Thī-a,Ud,Ud-a,"
    # "Vin-i,Vin-ii,Vin-ii-b,Vin-iii,Vin-iv,Vin-v,Vism-i,Vism-ii,Vv,Vv-a,A-ii,A-ix,A-vi,"
    # "A-viii,A-xi,Abhidh-s,Abhidh-s-t,Ap-ii,Ap-iii,As,Bv,Bv-a,Cp,Cp-a,Dhammn,Dhatuk,"
    # "Dhatuk-a,Dhs,Ja-a-iii,Ja-a-v,Ja-a-vi-b,Jina-c,Kv,Kv-a,Mp-ii,Mp-ix,Mp-vi,Mp-viii,"
    # "Mp-xi,Nett,Nett-a,Paṭṭh-a,Paṭṭh-i,Pet,Pp,Pp-a,Sp-ii-b,Th-a-ii,Vibh,Vibh-a,Yam-a,Yam-i"
    "Paṭṭh-ii,Paṭṭh-iii,Paṭṭh-iv,Paṭṭh-v,Kkh,VinSaṅg-a,Mp-t-i,Mp-t-ii,Mp-t-iii,Mp-t-iv,Mp-t-ix,"
    "Mp-t-v,Mp-t-vi,Mp-t-vii,Mp-t-viii,Mp-t-x,Mp-t-xi,Nett-vbh,Nett-ṭ,Ps-t-i,Ps-t-ii,Ps-t-iii,Spk-t-i,"
    "Spk-t-ii,Spk-t-iii,Spk-t-iv,Spk-t-v,Sv-nt-i,Sv-nt-ii,Sv-pt-i,Sv-pt-ii,Sv-pt-iii,Khuddas,Khuddas-nt,"
    "Khuddas-pt,Kkh-nt,Kkh-pt,Mūlasikk,Mūlasikk-t,Pācity-y,Sp-t-i-a,Sp-t-i-b,Sp-t-ii,Sp-t-iii,Sp-t-iv,Sp-t-v,"
    "Utt-vn,Utt-vn-t,Vin-alaṅke,Vin-vn,Vin-vn-t,Vjb-bkn,Vjb-cv,Vjb-mv,Vjb-pac,Vjb-pr,Vjb-prj,Vmv-i,Vmv-ii,"
    "Vmv-iii,Vmv-iv,Vmv-v,AbhMāt,Abhidh-av,Abhidh-av-nt,Abhidh-av-pt,Abhidh-av-sacc,Abhidh-av-vinich,As-mt,"
    "Dhatuk-anuṭ,Dhatuk-mt,Dhs-anuṭ,Kv-anuṭ,Kv-mt,Moh,Namar-p,Paṭṭh-anuṭ,Paṭṭh-mt,Pp-anuṭ,Pp-mt,Vibh-anuṭ,"
    "Vibh-mt,Vism-mht-i,Vism-mht-ii,Vism-nid,Yam-anuṭ,Yam-mt,Buddhaguṇ,Jinal,Kamal,LakkhBudth,MhPaṇām,Nāmakkp,"
    "Nāmakkṭ,Pajjm,Sutvan,Tigumb,Vāsamāl,Abh,Abh-ṭ,Bālāv,Kacc,Kacc-sadd,Mogg,Mogg-byk,MoggPañc,PadRūp,Payog,"
    "Sadd-dh,Sadd-pad,Subodh,Subodh-t,Vutt,Anudīp,Nirud,Paṭṭhuddes,PmtDīp,Catur,Cāṇn,Kavid,Lokan,Mhran,Narad,"
    "Nītim,Suttn,Sūrn,Vasala,Rasav,Sīmav,Vessg,AN-pv,Abh-pv,Att-pv,DN-pv,MN-pv,SN-pv,Vin-pv,Dat.h,Dhatup,"
    "Dhātpvil,Dhātv,Hattv,Jinvdīp,Mil-t,MoggVutt,Padamañj,Padsādh,Saddbind,Samantak,Tel,Thup,Cūḷgv,Mhv,Sas"
)


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

EPITAKA_DB  = os.environ.get("EPITAKA_DB",  "../data/epitaka.db")
GLOSSARY_DB = os.environ.get("GLOSSARY_DB", "../data/glossary.db")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
DEFAULT_LOG_DIR = "/tmp/book_translator_logs"


def _lang_db_path(epitaka_db: str, lang: str) -> str:
    """Derive the language-specific DB path, e.g. epitaka.db → epitaka_en.db."""
    p = Path(epitaka_db)
    return str(p.parent / f"{p.stem}_{lang}{p.suffix}")


# ══════════════════════════════════════════════════════════════════
# Stub modules
# ══════════════════════════════════════════════════════════════════

if "database" not in sys.modules:
    _db_mod = types.ModuleType("database")
    _db_mod.get_glossary_conn = lambda: sqlite3.connect(GLOSSARY_DB, timeout=30)
    sys.modules["database"] = _db_mod

if "config" not in sys.modules:
    _cfg_mod = types.ModuleType("config")
    _cfg_mod.EPITAKA_DB  = EPITAKA_DB
    _cfg_mod.SC_DATA_DB  = ""
    sys.modules["config"] = _cfg_mod

from common.context_builders import (  # noqa: E402
    GlossaryContext,
    CommentaryContext,
    PaliDefsContext,
    PreviousTranslationContext,
    MulaAtthaContext,
    ParallelTranslationContext,
    NissayaContext,
    TranslationWriter,
    GlossaryWriter,
    RemarkWriter,
)


# ══════════════════════════════════════════════════════════════════
# DB helpers
# ══════════════════════════════════════════════════════════════════

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


def ensure_confidence_column(epitaka_db: str):
    """Add translation_confidence column to sentences if not present."""
    with _connect(epitaka_db) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sentences)").fetchall()]
        if "translation_confidence" not in cols:
            conn.execute(
                "ALTER TABLE sentences ADD COLUMN translation_confidence TEXT"
            )
            conn.commit()
            print("[DB] Added column: sentences.translation_confidence")


def fetch_headings(epitaka_db: str, book_id: str) -> list[dict]:
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
    # Collect already-translated (book_id, para_id, line_id) from lang_db, if supplied
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
                "SELECT line_id, pali_sentence "
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
                    and len((s["pali_sentence"] or "").strip()) >= 3
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


# ══════════════════════════════════════════════════════════════════
# Token-safe chunking
# ══════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    return len(text) // 4


def chunk_paragraphs(paragraphs: list[dict], max_tokens: int = 3000) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for para in paragraphs:
        para_text = "\n".join(s.get("pali_sentence", "") for s in para.get("pending", []))
        para_tokens = estimate_tokens(para_text)

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

SYSTEM_PROMPT = """You are an expert scholar-translator of Pāli Buddhist literature
(canonical texts, commentaries [aṭṭhakathā] and sub-commentaries [ṭīkā]),
producing English translations that are both ACCURATE and READABLE for a
general but serious audience.

You will be given several reference blocks:
  1. ESTABLISHED GLOSSARY — accumulated translation memory containing
   previously selected Pāli → English renderings. Maintain consistency with
   these terms unless the context clearly requires a different meaning.
  2. PALI COMMENTARY & SUB-COMMENTARY — aṭṭhakathā / ṭīkā explaining these lines.
  3. PALI WORD DEFINITIONS       — dictionary entries + example usages for hard words.
  4. PREVIOUS PARAGRAPH          — the immediately preceding paragraph's translation,
                                    for tone/terminology continuity.
  5. TRANSLATED MŪLA / AṬṬHAKATHĀ / ṬĪKĀ REFERENCES — other already-translated
                                    paragraphs linked to this passage.
  6. PARALLEL HUMAN TRANSLATIONS — existing Sinhala, Thai, and/or published
                                    English (book) translations of THIS SAME passage.
  7. MYANMAR NISSAYA              — word-by-word gloss (romanised) for each sentence.
  8. SENTENCES TO TRANSLATE      — JSON array of Pāli sentences (para_id + line_id).

Return ONE JSON object with exactly three keys: "translations", "glossary",
and "remarks".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A. "translations" — array, ONE entry per input sentence, SAME ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    "para_id": <int>,
    "line_id": <int>,
    "english_translation": "<text>",
    "confidence": "high" | "low",
    "confidence_note": "<brief reason — ONLY when confidence is low, else omit>"
  }

CONFIDENCE RULES — be honest, not conservative:

  Mark confidence "low" when ANY of the following apply:
    • No parallel translation (Thai / Sinhala / English book) was provided
      AND the sentence contains rare compounds, technical terms, or ambiguous
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
    • OR parallel translations agree and confirm your reading.
    • OR the commentary/nissaya clearly resolves any difficult points.

  Do NOT mark everything "low" out of caution. Simple sentences with no
  ambiguity should be "high" even without parallel translations.
  The confidence field is for the human reviewer, not a disclaimer.

Translation style — read carefully:

  • Write natural, idiomatic, modern English that a literate non-specialist
    can follow. Prefer clear prose over a word-for-word rendering, but never
    drift from the actual meaning of the Pāli.
  • Use the PALI COMMENTARY (and ṭīkā, if present) as the primary authority
    for understanding difficult meanings, compounds, technical terms, and
    ambiguous syntax. The nissaya and word definitions support the commentary.
  • When translating COMMENTARIES or SUB-COMMENTARIES:
      - If the commentary explains or comments on a word, phrase, or technical
        term from the source text, use the established English translation of
        that source term if it is provided inside the commentary translation
        references.
      - Preserve the terminology relationship between the commented word and
        the explanation. The explanation should not introduce a different
        English rendering for the same technical term unless there is a clear
        reason.
      - The translation of the commentary should remain consistent with the
        translation of the original passage being explained.
  • If a PARALLEL HUMAN TRANSLATION (English book source) is supplied:
      use it as a reference for terminology and tone, but do not blindly copy.
      Where it is unnecessarily literal, archaic, or unclear, rewrite it in
      clearer modern English while preserving doctrinal precision.
  • Apply every ESTABLISHED GLOSSARY term/phrase exactly as given, including
    multi-word phrases.
  • Reference the PREVIOUS PARAGRAPH and TRANSLATED REFERENCES for consistency
    of terminology, names, and register. Do not unnecessarily change the
    translation of recurring Pāli terms.
  • Preserve important doctrinal distinctions between related Pāli terms.
    Do not merge different technical concepts merely because English words
    overlap.
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
  { "pali": "…", "english": "…", "domain": "…",
    "sub_domain": "…", "context": "…", "note": "…" }

  domain ∈ {sutta, vinaya, abhidhamma, grammar, story}

  (Full glossary rules as established in previous instructions apply.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C. "remarks" — ONLY for genuine, worth-noting CONFLICTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  { "para_id": <int>, "line_id": <int>, "pali": "<short excerpt>",
    "translation": "<the english_translation you chose>",
    "conflict": "<what the other source says, briefly>",
    "note": "<why you went with your choice, 1 short sentence>" }

  (Full remarks rules as established in previous instructions apply.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — critical
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown fences, no prose outside the JSON.
{ "translations": [...], "glossary": [...], "remarks": [...] }
"""


USER_TEMPLATE = """Book: {book_id}  —  paragraphs {para_start}–{para_end}

{glossary_block}

{commentary_block}

{pali_defs_block}

{prev_para_block}

{mula_block}

{parallel_block}

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
    parallel_block:   str,
    nissaya_block:    str,
) -> tuple[str, list[dict]]:
    flat_sentences = []
    for para in chunk:
        for s in para["pending"]:
            flat_sentences.append({
                "para_id":       para["para_id"],
                "line_id":       s["line_id"],
                "pali_sentence": s["pali_sentence"],
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
        parallel_block   = parallel_block,
        nissaya_block    = nissaya_block,
        sentences_json   = json.dumps(flat_sentences, ensure_ascii=False, indent=2),
    )
    return prompt, flat_sentences


# ══════════════════════════════════════════════════════════════════
# Response parser
# ══════════════════════════════════════════════════════════════════

def parse_response(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^\s*```[a-zA-Z]*\s*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?\s*```\s*$",           "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start:end + 1])
            obj.setdefault("translations", [])
            obj.setdefault("glossary",     [])
            obj.setdefault("remarks",      [])
            return obj
        except json.JSONDecodeError:
            pass

    obj = {"translations": [], "glossary": [], "remarks": []}
    for key in ("translations", "glossary", "remarks"):
        m = re.search(rf'"{key}"\s*:\s*\[', cleaned)
        if not m:
            continue
        array_start = m.end() - 1
        depth = 0
        obj_start = None
        items = []
        for i, ch in enumerate(cleaned[array_start:], start=array_start):
            if ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        items.append(json.loads(cleaned[obj_start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
            elif ch == "]" and depth == 0:
                break
        obj[key] = items

    return obj


# ══════════════════════════════════════════════════════════════════
# Gemini client
# ══════════════════════════════════════════════════════════════════

import logging
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


class KeyRotator:
    def __init__(self, keys: list[str]):
        self._lock = threading.Lock()
        if not keys:
            raise RuntimeError(
                "No Gemini API keys configured. "
                "Set GEMINI_KEY_<N> env vars or pass --api-keys."
            )
        self._keys  = list(keys)
        self._index = 0
        log.info(f"Loaded {len(self._keys)} Gemini key(s).")

    def next(self) -> str:
        with self._lock:
            if not self._keys:
                raise RuntimeError("All keys exhausted.")
            k = self._keys[self._index % len(self._keys)]
            self._index = (self._index + 1) % len(self._keys)
            return k

    def remove(self, key: str):
        with self._lock:
            if key in self._keys:
                self._keys.remove(key)
                log.warning(f"Key removed. {len(self._keys)} remaining.")


def _make_rotator(api_keys: list[str]) -> KeyRotator:
    if api_keys:
        return KeyRotator(api_keys)
    env_keys = [
        v.strip()
        for k, v in os.environ.items()
        if re.match(r"^GEMINI_KEY_\d+$", k) and v.strip()
    ]
    return KeyRotator(env_keys)


def call_gemini(
    rotator: KeyRotator,
    prompt: str,
    system_prompt: str,
    model: str = GEMINI_MODEL,
    max_output_tokens: int = 65_000,
    timeout: int = 300,
) -> str | None:
    from google import genai
    from google.genai import types as genai_types

    for attempt in range(10):
        key = rotator.next()
        result: dict = {"response": None, "error": None}

        def _call():
            try:
                open('input.txt', 'wt').write(f"{system_prompt}\n---\n{prompt}")
                client = genai.Client(api_key=key)
                r = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                result["response"] = r.text
                open('output.txt', 'wt').write(f"{r.text}")
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            log.error(f"[Gemini] Timeout on attempt {attempt + 1}")
            continue

        if result["error"]:
            e = result["error"]
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status == 429:
                rotator.remove(key)
                time.sleep(20)
                continue
            if status in (401, 403):
                time.sleep(20)
                continue
            log.warning(f"[Gemini] Error attempt {attempt + 1}: {e}")
            time.sleep(20)
            continue

        if result["response"] is not None:
            return result["response"]

    log.error("[Gemini] All retry attempts exhausted.")
    return None


def call_ai_with_logging(
    rotator:   KeyRotator,
    prompt:    str,
    book_id:   str,
    chunk_id:  str,
    log_dir:   str,
    model:     str = GEMINI_MODEL,
) -> str | None:
    os.makedirs(log_dir, exist_ok=True)
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    safe_id    = re.sub(r"[^\w\-]", "_", f"{book_id}_{chunk_id}")
    base_name  = f"{timestamp}_{safe_id}"

    prompt_path = os.path.join(log_dir, f"{base_name}_prompt.txt")
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("=== SYSTEM ===\n")
            f.write(SYSTEM_PROMPT)
            f.write("\n\n=== USER ===\n")
            f.write(prompt)
    except OSError as exc:
        print(f"[LOG] could not write prompt log: {exc}")

    n_tokens = estimate_tokens(prompt)
    print(f"[AI] calling: book={book_id} chunk={chunk_id} "
          f"{len(prompt)} chars (~{n_tokens} tokens)")

    raw = call_gemini(rotator, prompt, SYSTEM_PROMPT, model=model)
    if raw is None:
        return None

    response_path = os.path.join(log_dir, f"{base_name}_response.txt")
    try:
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(raw)
    except OSError as exc:
        print(f"[LOG] could not write response log: {exc}")

    print(f"[AI] response: {len(raw)} chars")
    return raw


# ══════════════════════════════════════════════════════════════════
# Per-book processing
# ══════════════════════════════════════════════════════════════════

def process_book(
    book_id:        str,
    args,
    rotator:        KeyRotator,
    epitaka_db:     str,
    lang_db:        str,
    translation_writer,
    glossary_writer,
    remark_writer,
) -> tuple[int, int, int]:
    """Translate one book. Returns (sentences_updated, glossary_added, remarks_saved)."""
    params = {"epitaka_db": epitaka_db}

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
    print(f"{len(sections)} section(s).")

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

        pali_text_for_glossary = "\n".join(
            s["pali_sentence"]
            for para in part
            for s in para["pending"]
        )

        glossary_block   = GlossaryContext(params, pali_text_for_glossary).build()
        commentary_block = CommentaryContext(params, book_id, pid_start, pid_end).build()
        pali_defs_block  = PaliDefsContext(params, pali_text=pali_text_for_glossary).build()
        prev_para_block  = PreviousTranslationContext(params, book_id, pid_start).build()
        mula_block       = MulaAtthaContext(params, book_id, pid_start, pid_end).build()
        parallel_block   = ParallelTranslationContext(params, book_id, pid_start, pid_end).build()

        chunks = chunk_paragraphs(part, max_tokens=args.max_tokens)
        print(f"  -> {len(chunks)} chunk(s)")

        for c_idx, chunk in enumerate(chunks, 1):
            n_sentences = sum(len(p["pending"]) for p in chunk)
            print(f"  Chunk {c_idx}/{len(chunks)}: {n_sentences} sentence(s) "
                  f"across {len(chunk)} para(s)")

            nissaya_block = NissayaContext(params, chunk).build()

            prompt, flat_sentences = build_prompt(
                book_id          = book_id,
                para_start       = chunk[0]["para_id"],
                para_end         = chunk[-1]["para_id"],
                chunk            = chunk,
                glossary_block   = glossary_block,
                commentary_block = commentary_block,
                pali_defs_block  = pali_defs_block,
                prev_para_block  = prev_para_block,
                mula_block       = mula_block,
                parallel_block   = parallel_block,
                nissaya_block    = nissaya_block,
            )

            if args.dry_run:
                print(prompt)
                print(f"  [dry-run] {len(flat_sentences)} sentence(s) would be sent.")
                continue

            raw = call_ai_with_logging(
                rotator  = rotator,
                prompt   = prompt,
                book_id  = book_id,
                chunk_id = f"p{chunk[0]['para_id']}-{chunk[-1]['para_id']}_c{c_idx}",
                log_dir  = args.log_dir,
                model    = args.model,
            )
            if raw is None:
                print(f"  Chunk {c_idx} returned no response. Skipping.")
                continue

            try:
                result = parse_response(raw)
            except Exception as exc:
                print(f"  parse_response failed for chunk {c_idx}: {exc}. Skipping.")
                continue

            translations = result.get("translations", [])
            new_terms    = result.get("glossary",     [])
            remarks      = result.get("remarks",      [])

            low_conf = sum(1 for t in translations if t.get("confidence") == "low")
            print(f"  Parsed: {len(translations)} translation(s) "
                  f"({low_conf} low-confidence), "
                  f"{len(new_terms)} glossary term(s), "
                  f"{len(remarks)} remark(s)")

            # ── Save to language-specific DB ──────────────────────────
            saved_trans = save_translations_to_lang_db(lang_db, book_id, translations)
            total_updated += saved_trans

            saved_rem = save_remarks_to_lang_db(lang_db, book_id, remarks)
            total_remarks += saved_rem

            # ── Clear corresponding fields from epitaka.db ────────────
            clear_from_epitaka_db(epitaka_db, book_id, translations, remarks)

            if new_terms:
                inserted = glossary_writer.upsert(new_terms, sc_id=book_id)
                total_glossary += inserted

        print(f"  Section done. Running: sentences={total_updated}, "
              f"glossary={total_glossary}, remarks={total_remarks}.")

    return total_updated, total_glossary, total_remarks


def ensure_lang_db(lang_db: str):
    """Create epitaka_<lang>.db with the required tables if they don't exist."""
    with _connect(lang_db) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sentences (
                book_id                TEXT NOT NULL,
                para_id                INTEGER NOT NULL,
                line_id                INTEGER NOT NULL,
                translation            TEXT,
                translation_confidence TEXT,
                confidence_note        TEXT,
                PRIMARY KEY (book_id, para_id, line_id)
            );
            CREATE TABLE IF NOT EXISTS translation_remarks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id    TEXT NOT NULL,
                para_id    INTEGER NOT NULL,
                line_id    INTEGER NOT NULL,
                pali       TEXT,
                translation TEXT,
                conflict   TEXT,
                note       TEXT,
                source_id  TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
    print(f"[lang_db] Ready: {lang_db}")


def save_translations_to_lang_db(lang_db: str, book_id: str, translations: list[dict]):
    """Upsert translated sentences into the lang-specific DB."""
    if not translations:
        return 0
    rows = [
        (
            book_id,
            t["para_id"],
            t["line_id"],
            t.get("english_translation") or t.get("translation") or "",
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


def save_remarks_to_lang_db(lang_db: str, book_id: str, remarks: list[dict]):
    """Insert remarks into the lang-specific DB."""
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
    """Nullify translation fields in epitaka.db for rows now saved in lang_db."""
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
               SET english_translation    = NULL,
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



    """Write confidence + confidence_note back to sentences table."""
    if not translations:
        return
    rows = [
        (
            t.get("confidence", "high"),
            t.get("confidence_note") or None,
            book_id,
            t["para_id"],
            t["line_id"],
        )
        for t in translations
        if "para_id" in t and "line_id" in t
    ]
    if not rows:
        return
    with _connect(epitaka_db) as conn:
        conn.executemany(
            "UPDATE sentences SET translation_confidence=?, confidence_note=? "
            "WHERE book_id=? AND para_id=? AND line_id=?",
            rows,
        )
        conn.commit()


def ensure_confidence_columns(epitaka_db: str):
    """Add translation_confidence and confidence_note columns if absent."""
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
            'Target language code, e.g. "en", "si", "th". '
            'Translations are saved to epitaka_<lang>.db next to epitaka.db.'
        ),
    )
    parser.add_argument(
        "--books", required=True,
        help=(
            'Comma-separated book_ids to translate, e.g. "Sp-i,Sp-ii". '
            'Use "preset" to run the full preset list.'
        ),
    )
    parser.add_argument("--start",      type=int, default=1,   help="first para_id (applies to every book, default 1)")
    parser.add_argument("--end",        type=int, default=-1,  help="last para_id, -1 = end of book (applies to every book)")
    parser.add_argument("--min-lines",  type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--epitaka-db", default=EPITAKA_DB)
    parser.add_argument("--glossary-db", default=GLOSSARY_DB)
    parser.add_argument("--model",      default=GEMINI_MODEL)
    parser.add_argument("--api-keys",   default="")
    parser.add_argument("--log-dir",    default=DEFAULT_LOG_DIR)
    parser.add_argument("--max-parts",  type=int, default=-1,  help="stop after N parts per book (for testing)")
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    epitaka_db = args.epitaka_db
    lang_db    = _lang_db_path(epitaka_db, args.lang)
    sys.modules["config"].EPITAKA_DB = epitaka_db
    sys.modules["database"].get_glossary_conn = lambda: sqlite3.connect(args.glossary_db, timeout=30)

    print(f"[lang] Language  : {args.lang}")
    print(f"[lang] Source DB : {epitaka_db}")
    print(f"[lang] Target DB : {lang_db}")

    # Resolve book list
    if args.books.strip().lower() == "preset":
        book_list = [b.strip() for b in PRESET_BOOKS.split(",") if b.strip()]
        print(f"Using preset: {len(book_list)} books.")
    else:
        book_list = [b.strip() for b in args.books.split(",") if b.strip()]

    if not book_list:
        print("No books specified.")
        return 1

    # Ensure DB columns exist before processing
    if not args.dry_run:
        ensure_confidence_columns(epitaka_db)
        ensure_lang_db(lang_db)

    explicit_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    rotator = _make_rotator(explicit_keys)

    params = {"epitaka_db": epitaka_db}
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
    return 0


if __name__ == "__main__":
    sys.exit(main())