"""
glossary_builder.py — Build / refresh the glossary and check translation quality
             for already-translated books in epitaka_<lang>.db.

    # Build glossary for explicit books:
    python glossary_builder.py --lang th --books Sp-i,Sp-ii

    # Auto-discover all books that have translations in epitaka_th.db:
    python glossary_builder.py --lang th

    # Resume from last saved position (default behaviour — state is always saved):
    python glossary_builder.py --lang th

    # Force restart from the beginning:
    python glossary_builder.py --lang th --restart

    # Check-only: skip glossary extraction, just flag wrong translations.
    # Output per call is much smaller, so you can raise --chunk-size to
    # check longer stretches of content per Gemini call:
    python glossary_builder.py --lang th --books Sp-ii --check-only --chunk-size 150

What it does
------------
For each book it:

  1. Reads every (pali, translation) pair from epitaka_<lang>.db that has a
     non-empty translation.

  2. Sends chunks to Gemini with:
       • the current ESTABLISHED GLOSSARY (so the AI can skip duplicates)
       • the pali+translation pairs for this chunk

  3. Gemini returns:
       "glossary"  — new terms whose meaning is NOT yet covered in the
                     established glossary (context and note in target language)
       "remarks"   — sentences where the translation appears wrong, with a
                     suggested correction and the reason

  4. New glossary terms are saved to glossary_<lang>.db.

  5. Wrong translations are saved to translation_remarks in epitaka_<lang>.db
     with note="wrong" and the AI's suggested correction in the translation
     field.  book_id and pali are filled from the DB — not invented by the AI.

Progress tracking
-----------------
A JSON state file is saved next to epitaka_<lang>.db as
  glossary_build_<lang>_state.json
It stores the last completed (book_id, para_id_end) so the script can
resume exactly where it left off after interruption.

File layout
-----------
This script owns everything specific to "build/refresh the glossary from
already-translated text": resume-state tracking, book auto-discovery,
established-glossary lookup for prompts, the check-only/full quality-check
prompts, and saving flagged remarks. Infrastructure shared with
book_translator.py (DB paths, schema, glossary upserts, Pāli stem lookup)
lives in common_utils.py. Everything about *how* we talk to the AI (Gemini
calls, key rotation, retries, response-JSON parsing) lives in ai_client.py —
change that file, not this one, when the AI logic needs to change.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import common.common_utils as cu
import common.ai_client as ai

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# CONFIG (script-specific: shared paths/lang-names/glossary logic live in
# common_utils.py; AI defaults live alongside the AI code in ai_client.py)
# ══════════════════════════════════════════════════════════════════

EPITAKA_DB   = cu.EPITAKA_DB
GLOSSARY_DB  = cu.GLOSSARY_DB
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
DEFAULT_LOG_DIR = "/tmp/build_glossary_logs"
DEFAULT_CHUNK_SIZE = 60   # sentence pairs per Gemini call

_connect = cu.connect  # local alias, kept short since it's used throughout this file


# ══════════════════════════════════════════════════════════════════
# Progress state (resume support)
# ══════════════════════════════════════════════════════════════════

class BuildState:
    """
    Persists progress to a JSON file so the script can resume after interruption.

    Schema:
      {
        "lang": "th",
        "completed_books": ["Sp-i", "Sp-ii"],
        "current_book": "Sp-iii",
        "current_para_end": 450    # last para_id chunk that finished
      }
    """

    def __init__(self, path: str, lang: str):
        self._path = path
        self._lang = lang
        self._data: dict = {}
        self._load()

    def _load(self):
        if Path(self._path).exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                print(f"[state] Loaded from {self._path}")
            except Exception as exc:
                print(f"[state] Could not load state file: {exc}. Starting fresh.")
                self._data = {}

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[state] Could not save state: {exc}")

    def reset(self):
        self._data = {"lang": self._lang, "completed_books": [], "current_book": None, "current_para_end": 0}
        self._save()
        print("[state] Reset.")

    @property
    def completed_books(self) -> list[str]:
        return self._data.get("completed_books", [])

    @property
    def current_book(self) -> str | None:
        return self._data.get("current_book")

    @property
    def current_para_end(self) -> int:
        return self._data.get("current_para_end", 0)

    def is_book_done(self, book_id: str) -> bool:
        return book_id in self.completed_books

    def resume_para_for(self, book_id: str) -> int:
        """Return the para_id to resume from (exclusive — i.e. start AFTER this)."""
        if self._data.get("current_book") == book_id:
            return self._data.get("current_para_end", 0)
        return 0

    def mark_chunk_done(self, book_id: str, para_id_end: int):
        self._data["current_book"]     = book_id
        self._data["current_para_end"] = para_id_end
        self._save()

    def mark_book_done(self, book_id: str):
        done = self._data.get("completed_books", [])
        if book_id not in done:
            done.append(book_id)
        self._data["completed_books"]  = done
        self._data["current_book"]     = None
        self._data["current_para_end"] = 0
        self._save()
        print(f"[state] Book '{book_id}' marked complete.")


# ══════════════════════════════════════════════════════════════════
# Book / pair discovery (glossary-build-specific; schema setup itself is
# shared — see common_utils.ensure_lang_db / ensure_glossary_db)
# ══════════════════════════════════════════════════════════════════

def discover_books(lang_db: str) -> list[str]:
    """Return distinct book_ids that have at least one translation in lang_db."""
    if not Path(lang_db).exists():
        return []
    with _connect(lang_db) as conn:
        rows = conn.execute(
            """SELECT DISTINCT book_id FROM sentences
               WHERE translation IS NOT NULL AND translation != ''
               ORDER BY book_id"""
        ).fetchall()
    books = [r["book_id"] for r in rows]
    print(f"[discover] Found {len(books)} book(s) with translations in {lang_db}.")
    return books


def fetch_translated_pairs(
    lang_db:    str,
    epitaka_db: str,
    book_id:    str,
    para_start: int,
    para_end:   int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[list[dict]]:
    """
    Fetch (pali, translation) pairs for a book from lang_db + epitaka_db.
    Returns a list of chunks (each chunk is a list of sentence dicts).
    para_start is inclusive-exclusive as a resume cursor: rows with
    para_id > para_start are returned, so pass the last completed para_id
    (0 = from the very beginning).
    """
    if not Path(lang_db).exists():
        return []

    with _connect(lang_db) as lconn:
        with _connect(epitaka_db) as econn:
            rows = lconn.execute(
                """SELECT book_id, para_id, line_id, translation
                   FROM sentences
                   WHERE book_id = ?
                     AND para_id > ?
                     AND (? = -1 OR para_id <= ?)
                     AND translation IS NOT NULL AND translation != ''
                   ORDER BY para_id, line_id""",
                (book_id, para_start, para_end, para_end),
            ).fetchall()

            pairs = []
            for r in rows:
                prow = econn.execute(
                    "SELECT pali_sentence FROM sentences "
                    "WHERE book_id=? AND para_id=? AND line_id=?",
                    (r["book_id"], r["para_id"], r["line_id"]),
                ).fetchone()
                pali = (prow["pali_sentence"] or "").strip() if prow else ""
                if pali:
                    pairs.append({
                        "para_id":     r["para_id"],
                        "line_id":     r["line_id"],
                        "pali":        pali,
                        "translation": r["translation"],
                    })

    if not pairs:
        return []
    return [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]


def fetch_established_glossary_block(glossary_db: str, pali_text: str) -> str:
    """
    Build a compact ESTABLISHED GLOSSARY block for the prompt, limited to
    terms whose stems appear in the current pali_text.  Falls back to the
    top-500 most-recent entries if stem matching returns nothing useful.
    """
    if not Path(glossary_db).exists():
        return ""

    try:
        import re
        # Collect pali tokens from pali_text
        tokens = set(re.findall(r"[a-zāīūṃṅñṭḍṇḷṣśḥ]+", pali_text.lower()))

        with _connect(glossary_db) as conn:
            rows = conn.execute(
                "SELECT pali, translation, domain, context, note FROM glossary "
                "ORDER BY id DESC LIMIT 2000"
            ).fetchall()

        if not rows:
            return ""

        matched = [r for r in rows if r["pali"] in tokens]
        if not matched:
            matched = rows[:500]  # fallback: most recent 500

        lines = ["══════════════════════════════",
                 "ESTABLISHED GLOSSARY",
                 "══════════════════════════════"]
        for r in matched:
            line = f"  {r['pali']} → {r['translation']}"
            if r["domain"]:
                line += f"  [{r['domain']}]"
            if r["context"]:
                line += f"  ctx: {r['context']}"
            if r["note"]:
                line += f"  note: {r['note']}"
            lines.append(line)
        return "\n".join(lines)
    except Exception as exc:
        print(f"[glossary_block] Could not build: {exc}")
        return ""


# ══════════════════════════════════════════════════════════════════
# Save helpers (glossary term saving is shared — see
# common_utils.save_glossary_terms — but "flag a wrong translation" is
# specific to this script, since it needs its own severity/critical logic)
# ══════════════════════════════════════════════════════════════════

def save_remarks(
    lang_db:    str,
    book_id:    str,
    epitaka_db: str,
    remarks:    list[dict],
) -> tuple[int, int]:
    """
    Save AI-flagged wrong translations to translation_remarks.

    The AI returns: para_id, line_id, translation (corrected), conflict (reason),
    and optionally severity ("wrong" | "critical"). book_id comes from our own
    variable. Unknown/missing severity values fall back to "wrong".

    Returns (rows_saved, critical_count).
    """
    if not remarks:
        return 0, 0

    rows = []
    critical_count = 0
    with _connect(epitaka_db) as econn:
        for r in remarks:
            para_id = r.get("para_id")
            line_id = r.get("line_id")
            if para_id is None or line_id is None:
                continue

            # Fetch pali from epitaka.db — do not trust AI-supplied pali
            prow = econn.execute(
                "SELECT pali_sentence FROM sentences "
                "WHERE book_id=? AND para_id=? AND line_id=?",
                (book_id, para_id, line_id),
            ).fetchone()
            pali = (prow["pali_sentence"] or "").strip() if prow else ""

            severity = str(r.get("severity") or "").strip().lower()
            if severity != "critical":
                severity = "wrong"
            else:
                critical_count += 1

            rows.append((
                book_id,
                para_id,
                line_id,
                pali,
                str(r.get("translation") or ""),   # AI-suggested correct translation
                str(r.get("conflict")    or ""),    # reason it is wrong
                severity,                           # "wrong" or "critical"
                book_id,                            # source_id
            ))

    if not rows:
        return 0, 0

    with _connect(lang_db) as conn:
        conn.executemany(
            """INSERT INTO translation_remarks
               (book_id, para_id, line_id, pali, translation, conflict, note, source_id)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
    return len(rows), critical_count


# ══════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════

_QUALITY_CHECK_INSTRUCTIONS = """Review each translation pair for clear errors: mistranslated terms, wrong
sense of a compound, a critical doctrinal term rendered by a generic word,
obvious grammatical misreading, or a translation that contradicts what the
Pāli actually says.

Only flag genuine errors — do NOT flag stylistic differences, synonyms, or
minor wording variations that preserve the meaning.

In addition, ALSO check specifically for these four error types. Each one,
if found, gets "severity": "critical" instead of the default "wrong". Be
conservative: only flag these when you are genuinely confident — if in
doubt, do NOT flag it.

  a) WRONG LANGUAGE — the translation is not actually written in {lang_name}
     (e.g. it came out in English or some other language by mistake). Flag
     only when the whole sentence (or the large majority of it) is in the
     wrong language — not when a single Pāli/Sanskrit proper noun or loanword
     is left untranslated inside an otherwise correct {lang_name} sentence.

  b) LINE MISALIGNMENT — the translation clearly belongs to a *different*
     Pāli line than the one it's paired with (content shifted, so a run of
     consecutive lines is off). Flag only when you're confident the
     translation's content matches a different line's Pāli. Do NOT flag
     verse (gāthā) lines where the translator simply reordered clauses
     within the SAME line for a smoother reading flow — that is acceptable
     and not an error.

  c) GĀTHĀ OVERRUN — for verse (gāthā) specifically: a single line's
     translation contains content that actually belongs to a later line
     (i.e. it translates ahead into the next line(s), leaving those lines'
     own translations duplicated or hollow). Flag only when you can point
     to specific extra clauses that correspond to a different, later line's
     Pāli — not merely because a line's translation reads long.

  d) DISRESPECTFUL REGISTER — when the subject is the Buddha, an Arahant,
     or another highly venerated figure, the translation uses a word/phrase
     that is casual, slangy, or carries an unintended modern double meaning
     that reads as disrespectful. Flag only clear, unambiguous cases — do
     NOT flag translations that are merely plain or informal in general
     register when no venerated figure is involved.

For each error return:
  {{
    "para_id": <int>,
    "line_id": <int>,
    "translation": "<your corrected {lang_name} translation of that sentence>",
    "conflict": "<brief explanation in {lang_name} of what is wrong and why>",
    "severity": "wrong" | "critical"
  }}

  • "translation": provide the full corrected {lang_name} sentence (not a diff).
  • "conflict": write in {lang_name}. Be specific: quote the problematic Pāli
    term or phrase and explain the correct meaning.
  • "severity": "critical" ONLY for cases (a)-(d) above. Everything else is
    "wrong". If unsure whether something qualifies as (a)-(d), use "wrong"
    rather than "critical".
  • Do NOT include para_id / line_id of sentences that look correct."""


def _build_system_prompt(lang: str, check_only: bool = False) -> str:
    """
    Build the system prompt. In --check-only mode the AI only performs the
    quality-check task (cheaper/faster, no glossary extraction); otherwise
    it performs both glossary extraction and the quality check in one call.
    """
    lang_name = cu.lang_name(lang)

    if check_only:
        return f"""You are an expert in Pāli Buddhist terminology and {lang_name} translation.

You will receive Pāli sentences paired with their {lang_name} translations from
Buddhist scriptures (canonical texts, commentaries, sub-commentaries).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK — "remarks": Flag WRONG translations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{_QUALITY_CHECK_INSTRUCTIONS.format(lang_name=lang_name)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — critical
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown fences, no prose outside the JSON.
{{ "remarks": [...] }}
"""

    return f"""You are an expert in Pāli Buddhist terminology and {lang_name} translation.

You will receive Pāli sentences paired with their {lang_name} translations from
Buddhist scriptures (canonical texts, commentaries, sub-commentaries).

Your tasks are TWO:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 1 — "glossary": Extract NEW glossary terms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract significant Pāli technical terms and their {lang_name} renderings
to help future translators choose the correct, consistent translation.

Return one object per NEW term:
  {{ "pali": "…", "translation": "…", "domain": "…",
    "sub_domain": "…", "context": "…", "note": "…" }}

domain ∈ {{sutta, vinaya, abhidhamma, grammar, story}}

CRITICAL — "pali" field:
  • Always the STEM (dictionary headword), NOT the inflected form.
    E.g. "bhikkhu" not "bhikkhūnaṃ"; "samādhi" not "samādhiṃ".
  • For compounds, give the whole compound in stem/uninflected form.

CRITICAL — "translation" field:
  • Must be in {lang_name}.
  • Extract the rendering that appears in the supplied translations.

CRITICAL — "context" and "note" fields:
  • Write BOTH in {lang_name} (not in English or Pāli).
  • "context": describe in {lang_name} when/where this term appears and how
    it is used — enough that a future translator understands the doctrinal
    or textual setting.
  • "note": write in {lang_name} any guidance that helps a future translator
    DECIDE the correct {lang_name} rendering — e.g. how this term differs from
    a similar one, why this particular {lang_name} word was chosen, what
    nuance would be lost with an alternative, or common pitfalls to avoid.
    This is the most important field: make it genuinely useful for decision-making.

CRITICAL — DO NOT add duplicates:
  • Check the ESTABLISHED GLOSSARY block carefully before adding any term.
  • If the pali stem (or a very close variant) already appears in the
    established glossary AND the {lang_name} meaning is substantially the same,
    DO NOT add it again — even if the wording differs slightly.
  • Only add a term if its meaning or usage is NOT already covered.

What TO include:
  • Technical terms easily rendered inconsistently or confused with similar
    terms (e.g. "samādhi" vs "samāpatti", "sīla" vs "vinaya").
  • Named doctrinal concepts, proper nouns, set Vinaya/Abhidhamma terms.
  • Terms whose {lang_name} rendering is non-obvious or debatable.

What NOT to include:
  • Grammatical particles / conjunctions:
    ca, va, vā, pi, api, tu, pana, hi, eva, kho, ti, iti, atha, yeva,
    hoti, hotu, honti, ahosi, atthi, natthi, kacci, kiṃ, na, mā, evaṃ,
    seyyathā, tattha, tatra, tato, yathā, tathā, idaṃ, ayaṃ, so, sā, yo, yā.
  • Plain verbs of being/doing with no doctrinal significance.
  • Any term already covered in the ESTABLISHED GLOSSARY.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 2 — "remarks": Flag WRONG translations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{_QUALITY_CHECK_INSTRUCTIONS.format(lang_name=lang_name)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — critical
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown fences, no prose outside the JSON.
{{ "glossary": [...], "remarks": [...] }}
"""


def _build_user_prompt(
    book_id:          str,
    pid_start:        int,
    pid_end:          int,
    lang_name:        str,
    glossary_block:   str,
    pairs:            list[dict],
    check_only:       bool = False,
) -> str:
    pairs_text = "\n".join(
        f"  para={s['para_id']} line={s['line_id']}\n"
        f"  Pāli: {s['pali']}\n"
        f"  {lang_name}: {s['translation']}"
        for s in pairs
    )

    if check_only:
        return (
            f"Book: {book_id}  —  paragraphs {pid_start}–{pid_end}\n\n"
            f"══════════════════════════════\n"
            f"PĀLI + {lang_name.upper()} TRANSLATION PAIRS\n"
            f"══════════════════════════════\n"
            f"{pairs_text}\n\n"
            f"Check the translations for errors as instructed.\n"
            f"Return only JSON: {{ \"remarks\": [...] }}"
        )

    return (
        f"Book: {book_id}  —  paragraphs {pid_start}–{pid_end}\n\n"
        f"{glossary_block}\n\n"
        f"══════════════════════════════\n"
        f"PĀLI + {lang_name.upper()} TRANSLATION PAIRS\n"
        f"══════════════════════════════\n"
        f"{pairs_text}\n\n"
        f"Perform both tasks (glossary + remarks) as instructed.\n"
        f"Do NOT add any glossary term already present in the ESTABLISHED GLOSSARY above.\n"
        f"Return only JSON: {{ \"glossary\": [...], \"remarks\": [...] }}"
    )


# ══════════════════════════════════════════════════════════════════
# Per-book processing
# ══════════════════════════════════════════════════════════════════

def process_book(
    book_id:     str,
    args,
    rotator:     ai.KeyRotator,
    epitaka_db:  str,
    lang_db:     str,
    glossary_db: str,
    lang:        str,
    state:       BuildState,
    check_only:  bool = False,
) -> tuple[int, int, int]:
    """
    Process one book.  Returns (glossary_inserted, remarks_saved, critical_saved).
    """
    lang_name = cu.lang_name(lang)
    system_prompt = _build_system_prompt(lang, check_only=check_only)

    print("=" * 60)
    print(f"BOOK: {book_id}  lang={lang}")
    print("=" * 60)

    # Determine where to resume
    resume_para = state.resume_para_for(book_id)
    if resume_para > 0:
        print(f"  [resume] Continuing from after para_id={resume_para}")

    para_end = args.end
    if para_end == -1:
        with _connect(epitaka_db) as conn:
            row = conn.execute(
                "SELECT MAX(para_id) AS m FROM sentences WHERE book_id=?", (book_id,)
            ).fetchone()
            para_end = row["m"] if row and row["m"] is not None else 0

    chunks = fetch_translated_pairs(
        lang_db    = lang_db,
        epitaka_db = epitaka_db,
        book_id    = book_id,
        para_start = resume_para,   # exclusive lower bound
        para_end   = para_end,
        chunk_size = args.chunk_size,
    )

    if not chunks:
        print("  No translated pairs found (or all already processed). Skipping.")
        state.mark_book_done(book_id)
        return 0, 0, 0

    print(f"  {sum(len(c) for c in chunks)} pairs in {len(chunks)} chunk(s).")
    total_glossary = 0
    total_remarks  = 0
    total_critical = 0

    for c_idx, chunk in enumerate(chunks, 1):
        pid_start = chunk[0]["para_id"]
        pid_end   = chunk[-1]["para_id"]
        print(f"  Chunk {c_idx}/{len(chunks)}: {len(chunk)} pairs "
              f"(paras {pid_start}-{pid_end})")

        if check_only:
            glossary_block = ""
        else:
            pali_text      = "\n".join(s["pali"] for s in chunk)
            glossary_block = fetch_established_glossary_block(glossary_db, pali_text)

        prompt = _build_user_prompt(
            book_id        = book_id,
            pid_start      = pid_start,
            pid_end        = pid_end,
            lang_name      = lang_name,
            glossary_block = glossary_block,
            pairs          = chunk,
            check_only     = check_only,
        )

        if args.dry_run:
            print(prompt[:500], "...")
            state.mark_chunk_done(book_id, pid_end)
            continue

        raw = ai.call_ai_with_logging(
            rotator       = rotator,
            prompt        = prompt,
            book_id       = book_id,
            chunk_id      = f"glos_p{pid_start}-{pid_end}_c{c_idx}",
            log_dir       = args.log_dir,
            model         = args.model,
            system_prompt = system_prompt,
        )
        if raw is None:
            print(f"  Chunk {c_idx} returned no response. Skipping.")
            continue

        expected_keys = ("remarks",) if check_only else ("glossary", "remarks")
        try:
            result = ai.parse_ai_json_response(raw, expected_keys)
        except Exception as exc:
            print(f"  parse_response failed for chunk {c_idx}: {exc}. Skipping.")
            continue

        new_terms = result.get("glossary", [])
        remarks   = result.get("remarks",  [])
        print(f"  Parsed: {len(new_terms)} glossary term(s), {len(remarks)} remark(s)")

        # Save glossary terms
        if new_terms and not check_only:
            inserted = cu.save_glossary_terms(
                glossary_db   = glossary_db,
                new_terms     = new_terms,
                source_id     = book_id,
                para_id_start = pid_start,
                para_id_end   = pid_end,
                epitaka_db    = epitaka_db,
            )
            total_glossary += inserted
            print(f"  Saved {inserted} new glossary term(s). Running total: {total_glossary}.")

        # Save wrong-translation remarks
        if remarks:
            saved, critical = save_remarks(
                lang_db    = lang_db,
                book_id    = book_id,
                epitaka_db = epitaka_db,
                remarks    = remarks,
            )
            total_remarks  += saved
            total_critical += critical
            crit_note = f" ({critical} critical)" if critical else ""
            print(f"  Saved {saved} remark(s){crit_note}. "
                  f"Running total: {total_remarks} ({total_critical} critical).")

        # Persist progress
        state.mark_chunk_done(book_id, pid_end)

    state.mark_book_done(book_id)
    return total_glossary, total_remarks, total_critical


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lang", required=True,
        help='Target language code, e.g. "en", "si", "th", "vi".',
    )
    parser.add_argument(
        "--books", default="",
        help=(
            'Comma-separated book_ids, e.g. "Sp-i,Sp-ii". '
            'Omit to auto-discover all books with translations in epitaka_<lang>.db.'
        ),
    )
    parser.add_argument("--start",       type=int, default=0,   help="First para_id (default 0 = beginning)")
    parser.add_argument("--end",         type=int, default=-1,  help="Last para_id, -1 = end of book")
    parser.add_argument("--chunk-size",  type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Sentence pairs per Gemini call (default {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--restart",     action="store_true",
                        help="Ignore saved state and restart from the beginning.")
    parser.add_argument("--check-only",  action="store_true",
                        help=(
                            "Only check translations for errors (remarks) — skip glossary "
                            "extraction entirely. Since the model doesn't have to also extract "
                            "and dedupe glossary terms, output is much smaller, so you can safely "
                            "raise --chunk-size to check longer stretches of content per call."
                        ))
    parser.add_argument("--epitaka-db",  default=EPITAKA_DB)
    parser.add_argument("--glossary-db", default="",
                        help="Override path to glossary_<lang>.db (default: auto-derived).")
    parser.add_argument("--model",       default=GEMINI_MODEL)
    parser.add_argument("--api-keys",    default="")
    parser.add_argument("--log-dir",     default=DEFAULT_LOG_DIR)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    epitaka_db  = args.epitaka_db
    lang_db     = cu.lang_db_path(epitaka_db, args.lang)
    glossary_db = args.glossary_db or cu.glossary_db_path(epitaka_db, args.lang)
    state_path  = cu.state_file_path(epitaka_db, args.lang)

    lang_name = cu.lang_name(args.lang)
    print(f"[lang] Language   : {args.lang}  ({lang_name})")
    print(f"[lang] Source DB  : {epitaka_db}")
    print(f"[lang] Lang DB    : {lang_db}")
    print(f"[lang] Glossary DB: {glossary_db}")
    print(f"[lang] State file : {state_path}")
    print(f"[mode] {'CHECK-ONLY (remarks only, no glossary extraction)' if args.check_only else 'Full (glossary + remarks)'}")

    # Ensure DB structures
    if not args.dry_run:
        if not args.check_only:
            cu.ensure_glossary_db(glossary_db)
        cu.ensure_lang_db(lang_db)

    # Load (or reset) progress state
    state = BuildState(state_path, args.lang)
    if args.restart:
        state.reset()
        print("[state] Restarted from scratch.")

    # Resolve book list
    if args.books.strip():
        book_list = [b.strip() for b in args.books.split(",") if b.strip()]
        print(f"[books] Explicit list: {len(book_list)} book(s).")
    else:
        book_list = discover_books(lang_db)
        if not book_list:
            print("No books found with translations. Nothing to do.")
            return 1

    # Filter already-completed books (unless --restart already cleared state)
    pending = [b for b in book_list if not state.is_book_done(b)]
    skipped = len(book_list) - len(pending)
    if skipped:
        print(f"[state] Skipping {skipped} already-completed book(s).")
    book_list = pending

    if not book_list:
        print("All books already processed. Use --restart to reprocess.")
        return 0

    rotator = ai.make_rotator([k.strip() for k in args.api_keys.split(",") if k.strip()])

    grand_glossary = 0
    grand_remarks  = 0
    grand_critical = 0

    for book_idx, book_id in enumerate(book_list, 1):
        print(f"\n{'#' * 60}")
        print(f"# BOOK {book_idx}/{len(book_list)}: {book_id}")
        print(f"{'#' * 60}")

        try:
            g, r, c = process_book(
                book_id     = book_id,
                args        = args,
                rotator     = rotator,
                epitaka_db  = epitaka_db,
                lang_db     = lang_db,
                glossary_db = glossary_db,
                lang        = args.lang,
                state       = state,
                check_only  = args.check_only,
            )
        except Exception as exc:
            print(f"[ERROR] Book {book_id} failed: {exc}. Continuing.")
            continue

        grand_glossary += g
        grand_remarks  += r
        grand_critical += c
        print(f"Book {book_id} done — glossary: {g}, remarks: {r} ({c} critical).")

    print("\n" + "=" * 60)
    print(f"ALL DONE.  Books processed: {len(book_list)}")
    print(f"  Total glossary terms saved : {grand_glossary}")
    print(f"  Total wrong-trans remarks  : {grand_remarks} ({grand_critical} critical)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())