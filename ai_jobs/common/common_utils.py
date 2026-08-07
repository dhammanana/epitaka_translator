"""
common_utils.py — Shared, non-AI infrastructure used by both
book_translator.py and glossary_builder.py.

What lives here
----------------
  - Default DB paths (EPITAKA_DB / GLOSSARY_DB) and path derivation helpers
    (epitaka.db -> epitaka_<lang>.db, glossary_<lang>.db)
  - The shared LANG_NAMES table (language code -> human-readable name used
    in prompts) and GLOSSARY_SKIP_TERMS (Pāli particles never worth adding
    to the glossary)
  - A tiny SQLite connection helper with sane WAL/timeout pragmas
  - Table creation (ensure_glossary_db / ensure_lang_db)
  - Pāli stem resolution (best-effort lookup into epitaka.db's dictionary
    tables so glossary entries store dictionary headwords, not inflected forms)
  - save_glossary_terms(): the upsert used whenever new glossary entries
    come back from the AI, whether from a translation run or a glossary-only
    build run

Nothing in this file talks to the AI — see ai_client.py for that. Keeping
the split this way means: changing how Gemini is called never requires
touching this file, and changing DB schema/paths never requires touching
ai_client.py.
"""

import os
import re
from contextlib import contextmanager
from pathlib import Path
import sqlite3

# ══════════════════════════════════════════════════════════════════
# Default DB locations (overridable via env vars or --epitaka-db / --glossary-db)
# ══════════════════════════════════════════════════════════════════

EPITAKA_DB  = os.environ.get("EPITAKA_DB",  "../data/epitaka.db")
GLOSSARY_DB = os.environ.get("GLOSSARY_DB", "../data/glossary.db")


# ══════════════════════════════════════════════════════════════════
# Human-readable language names, used inside prompts so the AI knows the
# target language it should be writing in.
# ══════════════════════════════════════════════════════════════════

LANG_NAMES: dict[str, str] = {
    "en": "English",
    "si": "Sinhala (සිංහල)",
    "ta": "Tamil (தமிழ்)",
    "hi": "Hindi (हिन्दी)",
    "ne": "Nepali (नेपाली)",
    "bn": "Bengali (বাংলা)",
    "mr": "Marathi (मराठी)",
    "gu": "Gujarati (ગુજરાતી)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "or": "Odia (ଓଡ଼ିଆ)",

    "th": "Thai (ภาษาไทย)",
    "lo": "Lao (ພາສາລາວ)",
    "km": "Khmer (ភាសាខ្មែរ)",
    "my": "Burmese (မြန်မာဘာသာ)",
    "vi": "Vietnamese (Tiếng Việt)",
    "id": "Indonesian (Bahasa Indonesia)",
    "ms": "Malay (Bahasa Melayu)",
    "tl": "Filipino (Tagalog)",

    "zh": "Chinese Simplified (简体中文)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",

    "de": "German (Deutsch)",
    "fr": "French (Français)",
    "es": "Spanish (Español)",
    "pt": "Portuguese (Português)",
    "it": "Italian (Italiano)",
    "nl": "Dutch (Nederlands)",
    "pl": "Polish (Polski)",
    "ru": "Russian (Русский)",
    "uk": "Ukrainian (Українська)",
    "tr": "Turkish (Türkçe)",
    "el": "Greek (Ελληνικά)",
    "ro": "Romanian (Română)",
    "cs": "Czech (Čeština)",
    "hu": "Hungarian (Magyar)",
    "sv": "Swedish (Svenska)",
    "da": "Danish (Dansk)",
    "fi": "Finnish (Suomi)",
    "no": "Norwegian (Norsk)",

    "ar": "Arabic (العربية)",
    "he": "Hebrew (עברית)",
    "fa": "Persian (فارسی)",
}


def lang_name(lang: str) -> str:
    """Human-readable name for a language code, falling back to its uppercased code."""
    return LANG_NAMES.get(lang, lang.upper())


# Common Pāli particles / inflections that must NOT be added to the glossary
# (pure grammar, no doctrinal content worth tracking translation choices for).
GLOSSARY_SKIP_TERMS = {
    "ca", "va", "vā", "pi", "api", "tu", "pana", "hi", "eva", "kho",
    "ti", "iti", "atha", "atha kho", "yeva", "neva",
    "hoti", "hotu", "honti", "ahosi",
    "atthi", "natthi", "asi",
    "kacci", "kiṃ", "ko", "kā", "kaṃ",
    "so", "sā", "taṃ", "te", "tā", "tassa", "tasma", "tasmā",
    "yo", "yā", "yaṃ", "ye", "yā",
    "idaṃ", "imaṃ", "imehi", "imesaṃ", "ayaṃ",
    "na", "no", "mā",
    "iti", "evaṃ", "seyyathā", "seyyathīdaṃ",
    "tattha", "tatra", "tato", "tada", "tadā",
    "kathaṃ", "yathā", "tathā",
    "vā", "vāpi", "nevā",
    "ime", "imo", "imasmiṃ",
}


# ══════════════════════════════════════════════════════════════════
# Path helpers
# ══════════════════════════════════════════════════════════════════

def lang_db_path(epitaka_db: str, lang: str) -> str:
    """Derive the language-specific sentences DB path, e.g. epitaka.db -> epitaka_en.db."""
    p = Path(epitaka_db)
    return str(p.parent / f"{p.stem}_{lang}{p.suffix}")


def glossary_db_path(epitaka_db: str, lang: str) -> str:
    """Derive the language-specific glossary DB path, e.g. epitaka.db -> glossary_en.db."""
    p = Path(epitaka_db)
    return str(p.parent / f"glossary_{lang}{p.suffix}")


def state_file_path(epitaka_db: str, lang: str) -> str:
    """Derive the glossary-build resume-state file path (used only by glossary_builder.py)."""
    p = Path(epitaka_db)
    return str(p.parent / f"glossary_build_{lang}_state.json")


# ══════════════════════════════════════════════════════════════════
# SQLite connection helper
# ══════════════════════════════════════════════════════════════════

@contextmanager
def connect(path: str):
    """
    Open a SQLite connection with WAL journaling and a generous busy timeout
    (so two scripts touching the same DB concurrently back off instead of
    raising "database is locked"). Rows come back as sqlite3.Row so callers
    can index by column name.
    """
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        yield conn
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════
# Schema setup
# ══════════════════════════════════════════════════════════════════

def ensure_glossary_db(glossary_db: str):
    """Create glossary_<lang>.db with the `glossary` table if it doesn't exist yet."""
    with connect(glossary_db) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS glossary (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                pali          TEXT NOT NULL,
                translation   TEXT NOT NULL,
                domain        TEXT,
                sub_domain    TEXT,
                context       TEXT,
                note          TEXT,
                source_id     TEXT,
                para_id_start INTEGER,
                para_id_end   INTEGER,
                created_at    TEXT DEFAULT (datetime('now')),
                UNIQUE (pali, translation)
            );
            CREATE INDEX IF NOT EXISTS idx_glossary_pali ON glossary(pali);
        """)
        conn.commit()
    print(f"[glossary_db] Ready: {glossary_db}")


def ensure_lang_db(lang_db: str):
    """
    Create epitaka_<lang>.db with the `sentences` and `translation_remarks`
    tables if they don't exist yet. Used both when actively translating
    (book_translator.py, which fills `sentences`) and when only building the
    glossary from already-translated text (glossary_builder.py, which mainly
    needs `translation_remarks` to exist but the two tables always travel
    together).
    """
    with connect(lang_db) as conn:
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
            );
        """)
        conn.commit()
    print(f"[lang_db] Ready: {lang_db}")


# ══════════════════════════════════════════════════════════════════
# Pāli stem resolution
# ══════════════════════════════════════════════════════════════════

def resolve_pali_stem(epitaka_db: str, word: str) -> str:
    """
    Resolve a Pāli word to its dictionary stem, mirroring dictionary.py's
    lookup order (_get_dpd_headwords / _get_dpr_stem):

      1. dpd_inflections_to_headwords — exact inflection -> headword(s);
         take the first headword and strip bracket/digit/whitespace noise
         DPD sometimes encodes into the field (e.g. "kata[1]" -> "kata").
      2. pali_definition.plain -> pali_definition.word (pali_definition no
         longer has a separate `stem` column; `word` is now the canonical
         headword for that table).
      3. dpr_stem.word -> stem

    Falls back to the input word unchanged if none of the three tables have
    it, or if epitaka_db can't be read for any reason (best-effort only —
    this must never crash a save).
    """
    word = word.strip().lower()
    if not word:
        return word
    try:
        with connect(epitaka_db) as conn:
            row = conn.execute(
                "SELECT headwords FROM dpd_inflections_to_headwords WHERE inflection = ?",
                (word,),
            ).fetchone()
            if row and row["headwords"]:
                parts = row["headwords"].split(",")
                dpd_word = re.sub(r"['\[\]\d\s]", "", parts[0])
                if dpd_word:
                    return dpd_word

            row2 = conn.execute(
                "SELECT word FROM pali_definition WHERE plain = ? OR word = ? LIMIT 1",
                (word, word),
            ).fetchone()
            if row2 and row2["word"]:
                return row2["word"]

            row3 = conn.execute(
                "SELECT stem FROM dpr_stem WHERE word = ?", (word,)
            ).fetchone()
            if row3 and row3["stem"]:
                return row3["stem"]
    except Exception:
        pass
    return word


# ══════════════════════════════════════════════════════════════════
# Glossary writes
# ══════════════════════════════════════════════════════════════════

def save_glossary_terms(
    glossary_db:   str,
    new_terms:     list[dict],
    source_id:     str = "",
    para_id_start: int | None = None,
    para_id_end:   int | None = None,
    epitaka_db:    str = "",
) -> int:
    """
    Upsert AI-extracted glossary terms into glossary_<lang>.db.

    The "pali" field from the AI is treated as a stem already (the prompt
    instructs the AI to return stems), but we additionally try to
    canonicalise it via `resolve_pali_stem()` against epitaka.db's
    dictionary tables when `epitaka_db` is supplied, and drop known
    grammar-only particles (GLOSSARY_SKIP_TERMS).

    para_id_start / para_id_end record provenance — which passage range was
    being processed when this entry was generated. On conflict (same
    pali+translation pair already exists), the row is updated in place
    rather than duplicated, and the provenance range is only ever widened
    (COALESCE keeps whichever bound was already set).

    Returns the number of rows actually inserted/updated (via SQLite's
    `changes()`), not just len(new_terms), since duplicates/skips don't count.
    """
    if not new_terms:
        return 0

    rows = []
    for term in new_terms:
        pali = str(term.get("pali") or "").strip().lower()
        translation = str(term.get("translation") or term.get("english") or "").strip()
        if not pali or not translation:
            continue

        if pali in GLOSSARY_SKIP_TERMS:
            continue

        if epitaka_db:
            pali = resolve_pali_stem(epitaka_db, pali)

        rows.append((
            pali,
            translation,
            str(term.get("domain",     "") or ""),
            str(term.get("sub_domain", "") or ""),
            str(term.get("context",    "") or ""),
            str(term.get("note",       "") or ""),
            source_id,
            para_id_start,
            para_id_end,
        ))

    if not rows:
        return 0

    inserted = 0
    with connect(glossary_db) as conn:
        for row in rows:
            try:
                conn.execute(
                    """INSERT INTO glossary
                       (pali, translation, domain, sub_domain, context, note,
                        source_id, para_id_start, para_id_end)
                       VALUES (?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(pali, translation) DO UPDATE SET
                           domain        = excluded.domain,
                           sub_domain    = excluded.sub_domain,
                           context       = excluded.context,
                           note          = excluded.note,
                           source_id     = excluded.source_id,
                           para_id_start = COALESCE(para_id_start, excluded.para_id_start),
                           para_id_end   = COALESCE(excluded.para_id_end, para_id_end)""",
                    row,
                )
                inserted += conn.execute("SELECT changes()").fetchone()[0]
            except Exception as exc:
                print(f"[GlossaryDB] Insert error for '{row[0]}': {exc}")
        conn.commit()

    return inserted