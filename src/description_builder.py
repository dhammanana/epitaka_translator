"""
description_builder.py — Generate a short SEO description + a longer readable
summary for every section of a book (canonical text, commentary,
sub-commentary, and ancillary works).

    # English descriptions for the whole preset book list (mūla → aṭṭhakathā
    # → ṭīkā → añña order), resuming whatever is already saved:
    python description_builder.py --lang en

    # A few books only, para range, one section for a quick test:
    python description_builder.py --lang en --books M-i,D-i --max-sections 1

    # Same content described in Sinhala instead of English:
    python description_builder.py --lang si --books M-i

    # Re-generate rows that already exist:
    python description_builder.py --lang en --books M-i --overwrite

What it does
------------
For each book:

  1. Reads `headings` from epitaka.db and builds one section per heading,
     skipping the excluded heading levels (default: level 10, which is the
     per-paragraph numbering, not a real section). A section runs from its
     own heading's para_id up to the next non-excluded heading's para_id.

  2. Counts the section's sentences (lines). Sections with fewer than
     `--min-lines` lines (default 20) are skipped — they are too short to
     deserve a description.

  3. Builds the section text the AI will read, preferring the target
     language's translation (epitaka_<lang>.db), then the English reference
     (epitaka_en.db), then the Pāli source when no translation exists.

  4. Truncates that text to `--max-input-tokens` (default 100,000 estimated
     tokens) so a giant section can never blow the request budget, then asks
     Gemini for a JSON object:

        {
          "description": "<one or two sentences, ~150 chars, hard max 200>",
          "summary":     "<2–4 easy-to-read paragraphs explaining the section>"
        }

     The description is written for use BOTH as the intro shown at the top of
     the section on the website AND as that page's <meta name="description">.
     The summary is the longer "if you need an explanation" version.

  5. Saves both into the `descriptions` table (see DDL below), keyed by
     (book_id, para_id, lang). `lang` defaults to "en", so several languages
     can coexist in the same table and the English row is the fallback.

descriptions schema (created on demand):
  book_id, para_id        — section identity (para_id = the heading's para_id)
  lang                    — output language of description/summary ("en" default)
  level, heading_title    — heading metadata, so the web server can build
                            breadcrumbs / pages without another lookup
  para_start, para_end    — the section's paragraph range (para_end inclusive)
  line_count              — sentences in the section (what --min-lines tested)
  description             — short SEO description (≤ 200 chars)
  summary                 — longer, easy-to-read explanation
  source                  — which text was sent: "si"/"th"/"en"/"pali"
  model                   — model that produced the row
  created_at, updated_at

Where it is saved
-----------------
The table lives in the ENGLISH translation DB (epitaka_en.db) by default —
the same file the web server already ships and reads (like `summaries`) — so
there is no extra DB to deploy. Override with --description-db or the
EPITAKA_DESCRIPTION_DB env var.

File layout
-----------
This script owns everything specific to "describe a book's sections":
heading-based sectioning, line counting, section-text selection/truncation,
the description/summary prompt, and saving rows. Shared infrastructure
(DB paths, connection helper) lives in common_utils.py; the AI call, key
rotation, retries and logging live in ai_client.py.
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import sqlite3
from dotenv import load_dotenv

import common.common_utils as cu
import common.ai_client as ai

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

EPITAKA_DB = cu.EPITAKA_DB
# Descriptions live inside the English translation DB by default (same file
# the web server already reads for `summaries`). Overridable.
DESCRIPTION_DB = (
    os.environ.get("EPITAKA_DESCRIPTION_DB")
    or cu.lang_db_path(EPITAKA_DB, "en")
)
# Empty by default so MODEL_FALLBACK_LIST is used (like study_builder.py).
# Set GEMINI_MODEL or pass --model to pin one model instead.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "")
DEFAULT_LOG_DIR = "/tmp/description_builder_logs"

# Models used when --model is not given. First is preferred; when its quota is
# exhausted for every key it is dropped and the next model takes over
# (ai_client.ModelPool). Pass --model to pin a single model instead.
MODEL_FALLBACK_LIST = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
]

# Heading levels that are NOT sections worth describing. Level 10 is the
# per-paragraph numbering row (e.g. "1", "2", "3") that exists in every book,
# so every level-10 heading is skipped by default (--exclude-levels).
DEFAULT_EXCLUDE_LEVELS = "10"

# Hard cap on the section text sent to the AI, measured in estimated tokens
# (ai.estimate_tokens: ~4 chars/token). Longer sections are cut at a line
# boundary. This is a request-budget guard, not a quality knob.
MAX_INPUT_TOKENS = int(os.environ.get("DESCRIPTION_MAX_INPUT_TOKENS", 100_000))

# Description length. The prompt asks for ~150 chars; anything longer than
# MAX_DESCRIPTION_CHARS (200) is trimmed at a word boundary before saving.
TARGET_DESCRIPTION_CHARS = 150
MAX_DESCRIPTION_CHARS = 200

# Safety cap on the long summary (never expected to be hit).
MAX_SUMMARY_CHARS = int(os.environ.get("DESCRIPTION_MAX_SUMMARY_CHARS", 8_000))

_connect = cu.connect  # local alias, kept short since it's used throughout this file


# ══════════════════════════════════════════════════════════════════
# Schema
# ══════════════════════════════════════════════════════════════════

_DESCRIPTION_DDL = """
CREATE TABLE IF NOT EXISTS descriptions (
    book_id       TEXT    NOT NULL,
    para_id       INTEGER NOT NULL,
    lang          TEXT    NOT NULL DEFAULT 'en',
    level         INTEGER,
    heading_title TEXT    NOT NULL DEFAULT '',
    para_start    INTEGER NOT NULL,
    para_end      INTEGER NOT NULL,
    line_count    INTEGER NOT NULL DEFAULT 0,
    description   TEXT    NOT NULL DEFAULT '',
    summary       TEXT    NOT NULL DEFAULT '',
    source        TEXT    NOT NULL DEFAULT '',
    model         TEXT    NOT NULL DEFAULT '',
    created_at    TEXT    DEFAULT (datetime('now')),
    updated_at    TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (book_id, para_id, lang)
);
CREATE INDEX IF NOT EXISTS idx_descriptions_book ON descriptions(book_id, lang);
"""


def ensure_description_db(path: str):
    """Create the `descriptions` table (and its index) if it doesn't exist yet."""
    with _connect(path) as conn:
        conn.executescript(_DESCRIPTION_DDL)
        conn.commit()
    print(f"[description_db] Ready: {path}")


def fetch_existing_para_ids(description_db: str, book_id: str, lang: str) -> set[int]:
    """para_ids already described for this book/language (used to resume)."""
    if not Path(description_db).exists():
        return set()
    with _connect(description_db) as conn:
        try:
            rows = conn.execute(
                "SELECT para_id FROM descriptions WHERE book_id=? AND lang=?",
                (book_id, lang),
            ).fetchall()
        except sqlite3.Error:
            return set()
    return {r["para_id"] for r in rows}


def save_description(path: str, row: dict):
    with _connect(path) as conn:
        conn.execute(
            """INSERT INTO descriptions
               (book_id, para_id, lang, level, heading_title, para_start,
                para_end, line_count, description, summary, source, model,
                updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
               ON CONFLICT(book_id, para_id, lang) DO UPDATE SET
                   level=excluded.level,
                   heading_title=excluded.heading_title,
                   para_start=excluded.para_start,
                   para_end=excluded.para_end,
                   line_count=excluded.line_count,
                   description=excluded.description,
                   summary=excluded.summary,
                   source=excluded.source,
                   model=excluded.model,
                   updated_at=datetime('now')""",
            (
                row["book_id"], row["para_id"], row["lang"], row["level"],
                row["heading_title"], row["para_start"], row["para_end"],
                row["line_count"], row["description"], row["summary"],
                row["source"], row["model"],
            ),
        )
        conn.commit()


# ══════════════════════════════════════════════════════════════════
# Book list
# ══════════════════════════════════════════════════════════════════

def load_preset_books() -> list[str]:
    """
    Read PRESET_BOOKS out of book_translator.py without importing it (that
    module sets up context stubs and pulls in context_builders at import
    time — unnecessary here). The list is already ordered
    mūla → aṭṭhakathā → ṭīkā → añña, which is also the order sections are
    described in: earlier books' descriptions are the ones most likely to be
    linked to from later ones.
    """
    src_path = Path(__file__).with_name("book_translator.py")
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(getattr(t, "id", None) == "PRESET_BOOKS" for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                break
            if isinstance(value, str):
                return [b.strip() for b in value.split(",") if b.strip()]
    raise RuntimeError(f"Could not read PRESET_BOOKS from {src_path}")


def fetch_book_categories(epitaka_db: str) -> dict[str, str]:
    """{book_id: category} from the books table, case-insensitive lookups made easy."""
    with _connect(epitaka_db) as conn:
        rows = conn.execute("SELECT book_id, category FROM books").fetchall()
    return {r["book_id"]: (r["category"] or "") for r in rows}


def fetch_book_names(epitaka_db: str) -> dict[str, str]:
    """{book_id: book_name} for prompt headers and log output."""
    with _connect(epitaka_db) as conn:
        rows = conn.execute("SELECT book_id, book_name FROM books").fetchall()
    return {r["book_id"]: (r["book_name"] or "") for r in rows}


def resolve_books(args, epitaka_db: str) -> list[str]:
    """
    Turn --books/--categories into the ordered list of books to process.
    `preset` (the default) expands to book_translator.py's PRESET_BOOKS, then
    everything not present in the DB is dropped with a note.
    """
    if args.books.strip().lower() in ("", "preset"):
        books = load_preset_books()
    else:
        books = [b.strip() for b in args.books.split(",") if b.strip()]

    if args.categories.strip():
        wanted = {c.strip().casefold() for c in args.categories.split(",") if c.strip()}
        cats = fetch_book_categories(epitaka_db)
        kept = [b for b in books if cats.get(b, "").casefold() in wanted]
        skipped = len(books) - len(kept)
        if skipped:
            print(f"[books] {skipped} book(s) dropped by --categories filter.")
        books = kept

    with _connect(epitaka_db) as conn:
        known = {
            r["book_id"] for r in
            conn.execute("SELECT DISTINCT book_id FROM books").fetchall()
        }
    missing = [b for b in books if b not in known]
    if missing:
        print(f"[books] {len(missing)} book(s) not in the DB, skipped: "
              f"{', '.join(missing[:10])}{' …' if len(missing) > 10 else ''}")
    return [b for b in books if b in known]


# ══════════════════════════════════════════════════════════════════
# Section discovery (headings → para ranges)
# ══════════════════════════════════════════════════════════════════

def fetch_headings(epitaka_db: str, book_id: str) -> list[dict]:
    """All headings for a book in document order."""
    with _connect(epitaka_db) as conn:
        rows = conn.execute(
            "SELECT para_id, level, title, parent FROM headings "
            "WHERE book_id=? ORDER BY para_id",
            (book_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_book_max_para(epitaka_db: str, book_id: str) -> int:
    with _connect(epitaka_db) as conn:
        row = conn.execute(
            "SELECT MAX(para_id) AS m FROM sentences WHERE book_id=?", (book_id,)
        ).fetchone()
    return row["m"] if row and row["m"] is not None else 0


def build_sections(
    epitaka_db:     str,
    book_id:        str,
    exclude_levels: set[int],
    para_start:     int,
    para_end:       int,
) -> list[dict]:
    """
    One section per non-excluded heading. A section spans its own heading's
    para_id up to (but not including) the next non-excluded heading's
    para_id, so excluded level-10 numbering rows never cut a section in two.

    Each returned dict: {para_id, para_end, level, title, next_para}.
    """
    headings = fetch_headings(epitaka_db, book_id)
    if not headings:
        return []

    book_max = fetch_book_max_para(epitaka_db, book_id)
    if para_end == -1:
        para_end = book_max

    starts = [
        h for h in headings
        if h["level"] not in exclude_levels
        and para_start <= h["para_id"] <= para_end
    ]
    if not starts:
        return []

    sections = []
    for i, h in enumerate(starts):
        next_para = starts[i + 1]["para_id"] if i + 1 < len(starts) else book_max + 1
        # Cap at the requested --end so a single-section range (--start 5
        # --end 5) never stretches to the end of the book.
        sections.append({
            "para_id":   h["para_id"],
            "para_end":  min(next_para - 1, para_end),
            "level":     h["level"],
            "title":     (h["title"] or "").strip(),
            "next_para": next_para,
        })
    return sections


def count_section_lines(epitaka_db: str, book_id: str, para_start: int, para_end: int) -> int:
    with _connect(epitaka_db) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM sentences "
            "WHERE book_id=? AND para_id BETWEEN ? AND ?",
            (book_id, para_start, para_end),
        ).fetchone()
    return row["n"] if row else 0


# ══════════════════════════════════════════════════════════════════
# Section text
# ══════════════════════════════════════════════════════════════════

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _plain(text: str) -> str:
    """Section text for the prompt: strip HTML tags (translations keep <b>/<i>)."""
    return _HTML_TAG_RE.sub("", text or "").strip()


def fetch_section_lines(
    epitaka_db: str,
    lang_db:    str,
    en_db:      str,
    book_id:    str,
    para_start: int,
    para_end:   int,
) -> list[dict]:
    """
    All sentences of the section with every available rendering:
    [{para_id, line_id, pali, lang, en}, ...] in document order.
    """
    with _connect(epitaka_db) as conn:
        rows = conn.execute(
            "SELECT para_id, line_id, pali FROM sentences "
            "WHERE book_id=? AND para_id BETWEEN ? AND ? ORDER BY para_id, line_id",
            (book_id, para_start, para_end),
        ).fetchall()
    lines = [dict(r) for r in rows]
    if not lines:
        return []

    def _translation_map(path: str) -> dict[tuple[int, int], str]:
        if not path or not Path(path).exists():
            return {}
        try:
            with _connect(path) as conn:
                trows = conn.execute(
                    "SELECT para_id, line_id, translation FROM sentences "
                    "WHERE book_id=? AND para_id BETWEEN ? AND ?",
                    (book_id, para_start, para_end),
                ).fetchall()
        except sqlite3.Error:
            return {}
        return {
            (r["para_id"], r["line_id"]): (r["translation"] or "")
            for r in trows
        }

    lang_map = _translation_map(lang_db)
    en_map = _translation_map(en_db) if os.path.abspath(en_db or "") != os.path.abspath(lang_db or "") else lang_map

    for ln in lines:
        key = (ln["para_id"], ln["line_id"])
        ln["pali"] = ln["pali"] or ""
        ln["lang"] = lang_map.get(key, "")
        ln["en"] = en_map.get(key, "")
    return lines


def pick_source(lines: list[dict], lang: str) -> str:
    """
    Which text to send: the target language's own translation when it covers
    at least half the section, else the English reference under the same
    rule, else the Pāli source. Keeps one consistent language per prompt.
    """
    total = len(lines) or 1
    if lang != "en":
        if sum(1 for l in lines if l["lang"].strip()) / total >= 0.5:
            return "lang"
    if sum(1 for l in lines if l["en"].strip()) / total >= 0.5:
        return "en"
    return "pali"


def build_section_text(lines: list[dict], source: str) -> str:
    """Render the section as plain text (with para/line anchors for context)."""
    out = []
    for ln in lines:
        if source == "lang":
            text = ln["lang"] or ln["en"] or ln["pali"]
        elif source == "en":
            text = ln["en"] or ln["pali"]
        else:
            text = ln["pali"]
        text = _plain(text)
        if text:
            out.append(text)
    return "\n".join(out)


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """
    Cut `text` down to `max_tokens` ESTIMATED tokens (~4 chars/token, matching
    ai.estimate_tokens) at a line boundary. Returns (text, was_truncated).
    """
    if max_tokens <= 0 or ai.estimate_tokens(text) <= max_tokens:
        return text, False
    max_chars = max_tokens * 4
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    if nl > max_chars // 2:
        cut = cut[:nl]
    return cut.rstrip() + "\n\n[... content truncated ...]", True


# ══════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════

_SYSTEM_TEMPLATE = """You are an expert editor and SEO writer for E-Piṭaka, a free\
 online reader of the Pāli Tipiṭaka and its commentaries (aṭṭhakathā),\
 sub-commentaries (ṭīkā) and ancillary works.

You are given ONE section of one book: its book id, its heading, and the\
 section's own text. The text may be Pāli, or a translation of it, or a mix;\
 read it carefully and write in {lang_name}.

Write for a general but serious reader — someone who is interested in early\
 Buddhism and can handle clarity, not jargon.

Return ONE JSON object with exactly two keys: "description" and "summary".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A. "description" — the short one (used as the page's meta description AND
   as the one-line introduction shown at the top of the section)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • ONE sentence, occasionally two very short ones.
  • Aim for about 150 characters. NEVER more than 200 characters.
  • Plain text only: no quotation marks around it, no markdown, no newlines,
    no numbering, no citations like [M-i:12:3], no trailing ellipsis.
  • Be specific and unique: name the subject, teaching, person, place, or
    procedure the section is about, so two sections of a book never share a
    description. Do NOT write generic filler such as "This section discusses
    Buddhist teachings."
  • Do NOT start with "This section", "This passage", "In this text", or
    "The passage". Start with the actual subject.
  • Do not use the word "section" more than once (ideally not at all).

Examples of the right shape (English):
  "The Buddha explains why beings are reborn in different realms, tracing
   each destination to the deeds that lead there."
  "Aṭṭhakathā on the opening of the Sutta: the reasons for the First
   Council and how the canon was preserved."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B. "summary" — the long one (an easy explanation, shown on request)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 2–4 short paragraphs, roughly 600–1200 characters in total.
  • Easy to read, flowing prose. Use complete, plain sentences. Do NOT
    compress the content into a dense note or telegraphic list.
  • Explain what the section covers and why it matters: the main teaching,
    argument, story or procedure, the key terms (explain them in a few words
    on first use), and the setting or context when the section has one.
  • Enough detail that a reader understands the section without opening it,
    but do NOT summarise every sentence line by line.
  • Plain text only: no markdown headings, no bullet points, no citations,
    no labels like "Summary:".
  • If the text is Pāli and you are confident of the meaning, explain it;
    if some detail is genuinely unclear, simply leave it out rather than
    guessing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — critical
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown fences, no prose outside the JSON.
{{ "description": "...", "summary": "..." }}
"""

_USER_TEMPLATE = """Book: {book_id} — {book_title}
Heading: {heading}  (level {level}, paragraphs {para_start}–{para_end}, {line_count} lines)
Text sent: {source_note}

══════════════════════════════
SECTION TEXT
══════════════════════════════
{section_text}

Write the JSON object ({lang_name} "description" and "summary") for this section.
"""


def build_prompts(
    book_id:     str,
    book_title:  str,
    section:     dict,
    lines:       list[dict],
    lang:        str,
    source:      str,
    section_text: str,
    truncated:   bool,
) -> tuple[str, str]:
    lang_name = cu.lang_name(lang)
    source_note = {
        "lang": f"the {lang_name} translation of this section"
                + (" (early lines only — the rest was truncated)" if truncated else ""),
        "en":   "the English translation of this section"
                + (" (early lines only — the rest was truncated)" if truncated else ""),
        "pali": "the original Pāli text (no translation available)"
                + (" (early lines only — the rest was truncated)" if truncated else ""),
    }[source]
    system = _SYSTEM_TEMPLATE.format(lang_name=lang_name)
    user = _USER_TEMPLATE.format(
        book_id     = book_id,
        book_title  = book_title or book_id,
        heading     = section["title"] or "(untitled heading)",
        level       = section["level"],
        para_start  = section["para_id"],
        para_end    = section["para_end"],
        line_count  = len(lines),
        source_note = source_note,
        section_text = section_text,
        lang_name   = lang_name,
    )
    return system, user


# ══════════════════════════════════════════════════════════════════
# Response parsing / cleanup
# ══════════════════════════════════════════════════════════════════

def parse_response(raw: str) -> dict:
    """
    Parse {"description": "...", "summary": "..."} defensively: strip code
    fences, try straight JSON, then fall back to pulling the two string
    values out with a regex (model output is occasionally loose JSON).
    """
    cleaned = (raw or "").strip()
    cleaned = re.sub(r"^\s*```[a-zA-Z]*\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned).strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start:end + 1])
            if isinstance(obj, dict):
                return {
                    "description": _as_text(obj.get("description")),
                    "summary":     _as_text(obj.get("summary")),
                }
        except json.JSONDecodeError:
            pass

    out = {}
    for key in ("description", "summary"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL)
        if m:
            try:
                out[key] = json.loads(f'"{m.group(1)}"')
            except json.JSONDecodeError:
                out[key] = m.group(1)
        else:
            out[key] = ""
    return {"description": _as_text(out["description"]), "summary": _as_text(out["summary"])}


def _as_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    if value is None:
        return ""
    return str(value).strip()


def clean_description(text: str) -> str:
    """
    Normalise the short description: single line, no wrapping quotes, no
    stray markdown, trimmed at a word boundary to MAX_DESCRIPTION_CHARS.
    """
    text = _as_text(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"^[#>*\-\s]+", "", text)
    text = _plain(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"“”\'').strip()
    if len(text) <= MAX_DESCRIPTION_CHARS:
        return text
    cut = text[:MAX_DESCRIPTION_CHARS]
    space = cut.rfind(" ")
    if space > MAX_DESCRIPTION_CHARS // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-—–").strip()


def clean_summary(text: str) -> str:
    """
    Normalise the long summary: drop markdown heading/list markers, tidy
    whitespace, and keep blank-line paragraph breaks intact (the web template
    splits paragraphs on a blank line).
    """
    text = _plain(_as_text(text))
    # Leading "## ", "> ", "- " markers only — never consume real line breaks,
    # which are what separate the summary's paragraphs.
    text = re.sub(r"^[ \t]*(?:[#>]+|[-*+])\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    # Any line break the model used separates paragraphs — the template splits
    # on the blank line, so normalise 1+ newlines to exactly that.
    text = re.sub(r"\n+", "\n\n", text)
    return text.strip()[:MAX_SUMMARY_CHARS].strip()


# ══════════════════════════════════════════════════════════════════
# Per-book processing
# ══════════════════════════════════════════════════════════════════

def process_book(
    book_id:         str,
    book_title:      str,
    args,
    rotator:         ai.KeyRotator,
    epitaka_db:      str,
    lang_db:         str,
    en_db:           str,
    description_db:  str,
    models:          list[str],
    budget:          dict,
) -> tuple[int, int, int]:
    """
    Describe every eligible section of one book.
    Returns (saved, skipped_short, failed). `budget` is a mutable dict with
    {"remaining"} shared across books so --max-sections is global.
    """
    lang = args.lang
    exclude_levels = {
        int(x) for x in str(args.exclude_levels).split(",") if x.strip().lstrip("-").isdigit()
    }

    print("=" * 60)
    print(f"BOOK: {book_id} — {book_title}  (lang={lang})")
    print("=" * 60)

    sections = build_sections(
        epitaka_db     = epitaka_db,
        book_id        = book_id,
        exclude_levels = exclude_levels,
        para_start     = args.start,
        para_end       = args.end,
    )
    if not sections:
        print("  No headings found. Skipping.")
        return 0, 0, 0

    existing = set() if args.overwrite else fetch_existing_para_ids(description_db, book_id, lang)
    if existing:
        print(f"  {len(existing)} section(s) already described in {lang} — skipping (use --overwrite to redo).")

    saved = skipped_short = failed = 0
    considered = 0

    for section in sections:
        if budget["remaining"] == 0:
            print("  Reached --max-sections; stopping this book.")
            break
        if section["para_id"] in existing:
            continue

        n_lines = count_section_lines(
            epitaka_db, book_id, section["para_id"], section["para_end"]
        )
        if n_lines < args.min_lines:
            skipped_short += 1
            continue

        if budget["remaining"] != -1:
            budget["remaining"] -= 1
        considered += 1

        lines = fetch_section_lines(
            epitaka_db = epitaka_db,
            lang_db    = lang_db,
            en_db      = en_db,
            book_id    = book_id,
            para_start = section["para_id"],
            para_end   = section["para_end"],
        )
        if not lines:
            print(f"  [sec {section['para_id']}] no sentences found; skipped.")
            continue

        source = pick_source(lines, lang)
        section_text = build_section_text(lines, source)
        if not section_text.strip():
            print(f"  [sec {section['para_id']}] empty text; skipped.")
            continue
        section_text, truncated = truncate_to_tokens(section_text, args.max_input_tokens)

        system_prompt, user_prompt = build_prompts(
            book_id      = book_id,
            book_title   = book_title,
            section      = section,
            lines        = lines,
            lang         = lang,
            source       = source,
            section_text = section_text,
            truncated    = truncated,
        )

        heading = section["title"] or "(untitled)"
        print("-" * 60)
        print(f"  [sec {considered}] para {section['para_id']}–{section['para_end']} "
              f"level {section['level']} “{heading[:60]}” — {n_lines} lines, "
              f"source={source}{', TRUNCATED' if truncated else ''}")

        if args.dry_run:
            print(user_prompt[:800], "\n...")
            continue

        used_holder: list[str] = []
        raw = ai.call_ai_with_logging(
            rotator       = rotator,
            prompt        = user_prompt,
            book_id       = book_id,
            chunk_id      = f"desc_para{section['para_id']}",
            log_dir       = args.log_dir,
            model         = args.model,
            models        = models,
            system_prompt = system_prompt,
            used_model    = used_holder,
        )
        if raw is None:
            print(f"  [sec {section['para_id']}] no response; skipped.")
            failed += 1
            continue

        try:
            parsed = parse_response(raw)
        except Exception as exc:                                  # pragma: no cover
            print(f"  [sec {section['para_id']}] parse failed: {exc}; skipped.")
            failed += 1
            continue

        description = clean_description(parsed.get("description", ""))
        summary     = clean_summary(parsed.get("summary", ""))
        if not description:
            print(f"  [sec {section['para_id']}] empty description; skipped.")
            failed += 1
            continue
        if len(description) > TARGET_DESCRIPTION_CHARS:
            print(f"  [note] description is {len(description)} chars (target "
                  f"{TARGET_DESCRIPTION_CHARS}, max {MAX_DESCRIPTION_CHARS}).")

        save_description(description_db, {
            "book_id":       book_id,
            "para_id":       section["para_id"],
            "lang":          lang,
            "level":         section["level"],
            "heading_title": section["title"],
            "para_start":    section["para_id"],
            "para_end":      section["para_end"],
            "line_count":    n_lines,
            "description":   description,
            "summary":       summary,
            "source":        source,
            "model":         used_holder[0] if used_holder else (args.model or (models[0] if models else "")),
        })
        saved += 1
        print(f"  saved ({len(description)} chars): {description}")
        print(f"  summary ({len(summary)} chars)")

    print(f"  => {saved} saved, {skipped_short} too short (<{args.min_lines} lines), {failed} failed.")
    return saved, skipped_short, failed


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lang", default="en",
        help='Output language code, e.g. "en", "si", "th" (default: en). '
             'Saved in the `lang` column; en is the default/fallback row.',
    )
    parser.add_argument(
        "--books", default="preset",
        help='Comma-separated book_ids, or "preset" (default) for the whole '
             'book_translator.py PRESET_BOOKS list (mūla → aṭṭhakathā → ṭīkā → añña).',
    )
    parser.add_argument(
        "--categories", default="",
        help='Optional filter on books.category, e.g. "Mūla,Aṭṭhakathā" '
             '(case-insensitive; empty = no filter).',
    )
    parser.add_argument("--start", type=int, default=0,
                        help="First para_id (default 0 = beginning)")
    parser.add_argument("--end", type=int, default=-1,
                        help="Last para_id, -1 = end of book")
    parser.add_argument("--min-lines", type=int, default=20,
                        help="Skip sections with fewer than this many lines (default 20)")
    parser.add_argument("--exclude-levels", default=DEFAULT_EXCLUDE_LEVELS,
                        help='Heading levels that are not sections (default "10")')
    parser.add_argument("--max-input-tokens", type=int, default=MAX_INPUT_TOKENS,
                        help=f"Truncate section text to this many estimated tokens "
                             f"(default {MAX_INPUT_TOKENS})")
    parser.add_argument("--max-sections", type=int, default=-1,
                        help="Stop after this many sections in total (-1 = no limit; for testing)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate sections that already have a description")
    parser.add_argument("--epitaka-db", default=EPITAKA_DB,
                        help="Source DB with headings/sentences (default ./data/epitaka.db)")
    parser.add_argument("--lang-db", default="",
                        help="Translation DB for --lang (default epitaka_<lang>.db)")
    parser.add_argument("--description-db", default=DESCRIPTION_DB,
                        help="DB that holds the `descriptions` table (default epitaka_en.db)")
    parser.add_argument("--model", default=GEMINI_MODEL,
                        help="Single model to use (default: MODEL_FALLBACK_LIST, "
                             "first model preferred, next one takes over when its quota runs out)")
    parser.add_argument("--api-keys", default="")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts, call no AI, save nothing")
    args = parser.parse_args()

    epitaka_db     = args.epitaka_db
    lang_db        = args.lang_db or cu.lang_db_path(epitaka_db, args.lang)
    en_db          = cu.lang_db_path(epitaka_db, "en")
    description_db = args.description_db

    print(f"[lang] Language       : {args.lang} ({cu.lang_name(args.lang)})")
    print(f"[lang] Source DB      : {epitaka_db}")
    print(f"[lang] Section text DB: {lang_db}")
    print(f"[lang] English ref DB : {en_db}")
    print(f"[lang] Description DB : {description_db}")
    print(f"[mode] min-lines={args.min_lines}  exclude-levels={args.exclude_levels}  "
          f"max-input-tokens={args.max_input_tokens}  overwrite={args.overwrite}")

    if not args.dry_run:
        ensure_description_db(description_db)

    book_list = resolve_books(args, epitaka_db)
    if not book_list:
        print("No books to process.")
        return 1
    print(f"[books] {len(book_list)} book(s) to process.")

    rotator = ai.make_rotator([k.strip() for k in args.api_keys.split(",") if k.strip()])
    models = [args.model] if args.model else MODEL_FALLBACK_LIST

    book_categories = fetch_book_categories(epitaka_db)
    book_names = fetch_book_names(epitaka_db)
    budget = {"remaining": args.max_sections}
    grand_saved = grand_short = grand_failed = 0

    for idx, book_id in enumerate(book_list, 1):
        print(f"\n{'#' * 60}")
        print(f"# BOOK {idx}/{len(book_list)}: {book_id}  [{book_categories.get(book_id, '?')}]")
        print(f"{'#' * 60}")
        try:
            saved, short, failed = process_book(
                book_id        = book_id,
                book_title     = book_names.get(book_id, ""),
                args           = args,
                rotator        = rotator,
                epitaka_db     = epitaka_db,
                lang_db        = lang_db,
                en_db          = en_db,
                description_db = description_db,
                models         = models,
                budget         = budget,
            )
        except Exception as exc:
            print(f"[ERROR] Book {book_id} failed: {exc}. Continuing.")
            continue
        grand_saved  += saved
        grand_short  += short
        grand_failed += failed
        if budget["remaining"] == 0:
            print("Reached --max-sections; stopping.")
            break

    print("\n" + "=" * 60)
    print(f"ALL DONE. Descriptions saved: {grand_saved}  "
          f"(skipped short: {grand_short}, failed: {grand_failed})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
