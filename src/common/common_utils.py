"""
common_utils.py — Shared, non-AI infrastructure used by both
book_translator.py and glossary_builder.py.

What lives here
----------------
  - Default DB paths (EPITAKA_DB, plus the legacy GLOSSARY_DB env override)
    and path derivation helpers
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
#
# Resolved relative to the translator project root (the directory containing
# runner.sh), not the current working directory, so the scripts work no
# matter where they are invoked from:
#   translator/src/common/common_utils.py -> parents[3] == translator/
# Default data dir is then translator/data (i.e. ./data next to ./src).
# A legacy ../data (epitaka/data) is still honoured as a fallback when
# ./data/epitaka.db is missing but ../data/epitaka.db exists.
# ══════════════════════════════════════════════════════════════════

def _default_data_dir() -> Path:
    here = Path(__file__).resolve()
    # src/common/common_utils.py -> translator/
    translator_dir = here.parent.parent.parent
    data_dir = translator_dir / "data"
    if (data_dir / "epitaka.db").exists():
        return data_dir
    legacy = translator_dir.parent / "data"
    if (legacy / "epitaka.db").exists():
        return legacy
    # Neither has the DB yet — default to ./data so runner.sh downloads there.
    return data_dir


_DEFAULT_DATA_DIR = _default_data_dir()

EPITAKA_DB  = os.environ.get("EPITAKA_DB",  str(_DEFAULT_DATA_DIR / "epitaka.db"))
# Legacy explicit override only. There is deliberately NO built-in default
# filename here: every real code path derives the per-language glossary via
# glossary_db_path(epitaka_db, lang) -> glossary_<lang>.db. Defaulting to ""
# (instead of the old bare "glossary.db") means a misconfigured caller fails
# loudly in _glossary_db_path() instead of silently reading/writing a
# language-blind glossary.db that nothing else uses.
GLOSSARY_DB = os.environ.get("GLOSSARY_DB", "")


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
# Script-bleed guard (shared by book_translator.py's live translation
# checks and by save_glossary_terms() below).
#
# The prompt context (PARALLEL HUMAN TRANSLATIONS, MYANMAR NISSAYA, etc.)
# deliberately includes reference text in Thai, Sinhala, Myanmar, and
# English so the model has terminology/meaning support. That means any of
# those FOUR scripts can bleed into a translation whose target language is
# none of them — not just Thai into Khmer. We check for all three
# non-Latin reference scripts (English is Latin-script and can't be
# reliably distinguished from many legitimate target languages that also
# use Latin script, e.g. Vietnamese/Indonesian/German, so it is not
# script-checked here).
# ══════════════════════════════════════════════════════════════════

# lang code -> (list of Unicode ranges, human label, langs it's native for).
# If the TARGET language uses that script, that script is naturally allowed
# and excluded from the check for that run. Some scripts are shared by more
# than one target language code (e.g. Devanagari is used by Hindi,
# Marathi, AND Nepali), so this maps script -> the *set* of lang codes it's
# legitimate for, not just a single code.
#
# Each script lists its FULL set of relevant Unicode blocks, not just the
# main block — Pali-in-[script] texts sometimes use characters from
# "Extended" blocks that the base block alone misses (e.g. Vedic accent
# marks used in some Devanagari Pali/Sanskrit editions, or Mon/Shan-derived
# conjuncts sometimes present in Myanmar-script Pali texts).
_CONTEXT_SCRIPT_RANGES: dict[str, tuple[list[tuple[int, int]], str, frozenset[str]]] = {
    "th": ([(0x0E00, 0x0E7F)], "Thai", frozenset({"th"})),
    "si": ([
        (0x0D80, 0x0DFF),   # Sinhala (main block, used for Pali-in-Sinhala)
        (0x111E0, 0x111FF), # Sinhala Archaic Numbers (rare, but real in some editions)
    ], "Sinhala", frozenset({"si"})),
    "my": ([
        (0x1000, 0x109F),   # Myanmar (main block)
        (0xAA60, 0xAA7F),   # Myanmar Extended-A
        (0xA9E0, 0xA9FF),   # Myanmar Extended-B
    ], "Myanmar", frozenset({"my"})),
    # Devanagari isn't part of the prompt's own reference context, but it's
    # heavily over-represented in the model's training data (Hindi/Sanskrit/
    # Marathi/Nepali all use it), so the model sometimes defaults to it as a
    # "safe-looking" script when translating into OTHER Indic-family or
    # unrelated languages. Shared by hi/mr/ne, so all three are excluded.
    "hi": ([
        (0x0900, 0x097F),   # Devanagari (main block)
        (0xA8E0, 0xA8FF),   # Devanagari Extended
        (0x1CD0, 0x1CFF),   # Vedic Extensions (accent marks used in some Pali/Sanskrit editions)
        (0x11B00, 0x11B7F), # Devanagari Extended-A
    ], "Devanagari", frozenset({"hi", "mr", "ne"})),
}


def _ranges_to_char_class(ranges: list[tuple[int, int]]) -> str:
    return "".join(f"\\U{lo:08X}-\\U{hi:08X}" for lo, hi in ranges)


# Precompiled once at import time — regex (C-level) scanning is orders of
# magnitude faster than a Python per-character loop, which matters a lot
# when scanning tables with 1M+ rows on modest hardware.
_SCRIPT_PATTERNS: dict[str, re.Pattern] = {
    key: re.compile(f"[{_ranges_to_char_class(ranges)}]")
    for key, (ranges, _name, _langs) in _CONTEXT_SCRIPT_RANGES.items()
}

# A single combined pattern covering every watched script, used as a cheap
# first-pass filter: the overwhelming majority of rows are clean (pure
# target script), so most rows can be rejected with one fast regex.search()
# instead of running four separate scans on every row.
_ANY_WATCHED_SCRIPT_PATTERN = re.compile(
    "[" + "".join(_ranges_to_char_class(ranges) for ranges, _n, _l in _CONTEXT_SCRIPT_RANGES.values()) + "]"
)


def _ranges_to_sql_char_class(ranges: list[tuple[int, int]]) -> str:
    # \x{HHHH} is the codepoint-escape syntax understood by the regex
    # engines commonly wired up as a SQLite REGEXP() function (PCRE,
    # PCRE2, Oniguruma, RE2) — NOT the same as Python's \UHHHHHHHH syntax
    # used in _ranges_to_char_class() above, so this needs its own builder.
    return "".join(f"\\x{{{lo:X}}}-\\x{{{hi:X}}}" for lo, hi in ranges)


# Same combined "any watched script" filter as _ANY_WATCHED_SCRIPT_PATTERN,
# but as a plain string in a syntax portable to external SQLite REGEXP
# extensions, for callers that want SQLite itself to do the first-pass
# filtering (e.g. `WHERE translation REGEXP ?`) instead of streaming every
# row into Python just to reject the overwhelming majority of them.
# NOTE: this is only useful if the SQLite build in use actually has a
# REGEXP function registered (some distro sqlite3 CLIs / extensions ship
# one, plain Python sqlite3 does not, by default). Callers should probe for
# support (e.g. run a harmless `SELECT 1 WHERE 'x' REGEXP 'x'`) and fall
# back to Python-side filtering on sqlite3.OperationalError.
ANY_WATCHED_SCRIPT_SQL_REGEX = (
    "[" + "".join(_ranges_to_sql_char_class(ranges) for ranges, _n, _l in _CONTEXT_SCRIPT_RANGES.values()) + "]"
)


def any_watched_script_present(text: str) -> bool:
    """
    Cheap, lang-agnostic prefilter: True if `text` contains ANY character
    from ANY watched reference script, regardless of target language. This
    intentionally does NOT do the per-language exclusion that
    find_context_script_bleed() does — it's meant to be a fast, wide net
    (in Python, or registered as a SQLite scalar function) that narrows a
    huge table down to a small set of candidates, which then get the full
    find_context_script_bleed()/describe_context_script_bleed() treatment
    to apply the real per-language + threshold logic.
    """
    return bool(text) and bool(_ANY_WATCHED_SCRIPT_PATTERN.search(text))


def count_chars_in_range(text: str, lo: int, hi: int) -> int:
    if not text:
        return 0
    return sum(1 for ch in text if lo <= ord(ch) <= hi)


def count_thai_chars(text: str) -> int:
    """Count characters in `text` that fall in the Thai Unicode block."""
    return len(_SCRIPT_PATTERNS["th"].findall(text)) if text else 0


def has_thai_bleed(lang: str, text: str, threshold: int = 3) -> bool:
    """
    True if `text` contains a suspicious run of Thai-script characters even
    though `lang` is not Thai. Kept as a thin wrapper around
    find_context_script_bleed() for backwards compatibility.
    """
    hits = find_context_script_bleed(lang, text, threshold=threshold)
    return any(name == "Thai" for name, _ in hits)


def find_context_script_bleed(lang: str, text: str, threshold: int = 3) -> list[tuple[str, int]]:
    """
    Check `text` for a suspicious amount of characters from any of the
    watched reference/high-resource scripts (Thai, Sinhala, Myanmar,
    Devanagari) OTHER than the target language's own script. Returns a
    list of (script_name, count) pairs for every script that crossed
    `threshold` — usually empty. `threshold` guards against false
    positives from a single stray character (e.g. a quoted Sanskrit term
    in a footnote).

    Fast path: most text is completely clean, so we first run one cheap
    combined regex.search() (stops at the first hit, doesn't scan the
    whole string) before doing the more expensive per-script findall()
    breakdown that's only needed to build the detailed report.
    """
    if not text or not _ANY_WATCHED_SCRIPT_PATTERN.search(text):
        return []
    hits = []
    for key, (_ranges, name, allowed_langs) in _CONTEXT_SCRIPT_RANGES.items():
        if lang in allowed_langs:
            continue  # this script IS (one of) the target language's own — allowed
        n = len(_SCRIPT_PATTERNS[key].findall(text))
        if n >= threshold:
            hits.append((name, n))
    return hits


def describe_context_script_bleed(lang: str, text: str, threshold: int = 3) -> str | None:
    """Convenience: human-readable one-line reason string, or None if clean."""
    hits = find_context_script_bleed(lang, text, threshold=threshold)
    if not hits:
        return None
    parts = ", ".join(f"{name} ({n} char(s))" for name, n in hits)
    return f"looks like {parts} script, not {lang!r}"


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
    lang:          str = "",
    log_warn=print,
) -> int:
    """
    Upsert AI-extracted glossary terms into glossary_<lang>.db.

    The "pali" field from the AI is treated as a stem already (the prompt
    instructs the AI to return stems), but we additionally try to
    canonicalise it via `resolve_pali_stem()` against epitaka.db's
    dictionary tables when `epitaka_db` is supplied, and drop known
    grammar-only particles (GLOSSARY_SKIP_TERMS).

    Entries whose "translation" text looks like it bled into Thai script
    (see has_thai_bleed) are rejected outright rather than saved — this is
    the critical guard, because glossary entries are fed back into every
    future prompt as "apply exactly", so a single contaminated entry that
    slips through here will keep re-poisoning the rest of the book. When
    `lang` is not supplied, no script check is performed (callers should
    always pass it going forward).

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

        if lang:
            reason = describe_context_script_bleed(lang, translation)
            if reason:
                log_warn(f"[GlossaryDB] Rejected '{pali}' -> '{translation}': {reason}. Not saved.")
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