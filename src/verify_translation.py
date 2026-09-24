"""
verify_translation.py — Verify existing translations against the Pāli
source AND its linked texts (commentary / sub-commentary / mūla / sibling
commentary, pulled via CommentaryContext), using the B.AI API
(deepseek-v4-flash, single key — see ai_client_bai.py).

Why CommentaryContext
----------------------
A wrong translation is often only detectable by cross-checking against
what the *commentary* says the passage means, or against the *mūla* text
a sub-commentary passage is explaining. CommentaryContext already knows
how to resolve all of that from `book_links`:

  - forward:  input paragraph  -> its commentary / sub-commentary
  - reverse:  input paragraph  -> the mūla / source text it comments on
  - sibling:  the mūla's other commentaries (e.g. aṭṭhakathā alongside a ṭīkā)

This script re-uses CommentaryContext as-is (it already overlays existing
English/target-language translations of the linked lines too, via
_fetch_lang_translations_with_fallback) rather than re-implementing any
of that lookup.

Usage
-----
    # Dry run: print the prompt for the first chunk, don't call the AI
    python verify_translation.py --lang th --book Sp-i --start 1 --end 40 --dry-run

    # Verify a whole book. Findings are ALWAYS saved to translation_remarks
    # (note="wrong"/"critical"); sentences.translation is left untouched:
    python verify_translation.py --lang th --book Sp-i

    # Verify AND overwrite sentences.translation with the AI's corrections:
    python verify_translation.py --lang th --book Sp-i --apply

    # Auto-discover every book that has translations in epitaka_th.db:
    python verify_translation.py --lang th

What it does, per chunk of paragraphs
--------------------------------------
  1. Reads every (pali, translation) line pair in that paragraph range
     from epitaka.db / epitaka_<lang>.db.
  2. Builds the LINKED-TEXT context for that same paragraph range via
     CommentaryContext (forward + mūla + sibling).
  3. Sends both to B.AI and asks it to flag lines whose translation
     conflicts with the Pāli itself OR with the linked texts, returning a
     corrected translation + a short explanation for each flagged line.
  4. Saves every flagged line to `translation_remarks` via RemarkWriter.
  5. With --apply, flagged lines' `sentences.translation` is also
     overwritten with the AI's corrected text, via TranslationWriter.

File layout
-----------
This script owns: paragraph chunking, the verification prompt, and the
per-chunk save/apply flow. DB paths/schema/connection helpers live in
common_utils.py. Linked-text lookup lives in context_builders.py
(CommentaryContext / TranslationWriter / RemarkWriter). All B.AI calling
and JSON-response parsing lives in ai_client_bai.py — change that file,
not this one, if the AI-calling logic needs to change.
"""

import argparse
import os
import sys
from pathlib import Path

# Support both `python -m ai_jobs.verify_translation` (package import, with
# ai_client_bai.py / common_utils.py / context_builders.py in a `common/`
# subpackage next to this file) and running this file directly as a script
# (`python verify_translation.py`) from inside the ai_jobs directory.
try:
    from .common.common_utils import connect as _connect, lang_name as _lang_name
    from .common.context_builders import CommentaryContext, TranslationWriter, RemarkWriter
    from .common import ai_client_bai as ai
except ImportError:
    # No parent package — this is the "attempted relative import with no
    # known parent package" case you get from `python verify_translation.py`
    # directly. Fall back to a plain absolute import of `common.*` instead;
    # this only works if `common/` sits right next to this file, i.e. you're
    # running from inside ai_jobs/. sys.path[0] is normally already this
    # file's directory when run as a script, but we add it explicitly to be
    # safe (e.g. if something else imports this module from elsewhere).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common.common_utils import connect as _connect, lang_name as _lang_name  # type: ignore
    from common.context_builders import CommentaryContext, TranslationWriter, RemarkWriter  # type: ignore
    from common import ai_client_bai as ai  # type: ignore

EPITAKA_DB = os.environ.get("EPITAKA_DB", str(Path(__file__).resolve().parent.parent / "data" / "epitaka.db"))

DEFAULT_CHUNK_PARAS            = 5     # paragraphs per B.AI call
DEFAULT_MAX_CONTEXT_CHARS      = 4000   # budget passed to CommentaryContext
DEFAULT_MAX_TOKENS             = 10000   # thinking + answer combined, per B.AI call
DEFAULT_THINKING_BUDGET_TOKENS = 500   # cap on deepseek-v4-flash's reasoning tokens
DEFAULT_LOG_DIR                = "/tmp/verify_translation_logs"


def lang_db_path(epitaka_db: str, lang: str) -> str:
    p = Path(epitaka_db)
    return str(p.parent / f"{p.stem}_{lang}{p.suffix}")


def glossary_db_path(epitaka_db: str, lang: str) -> str:
    p = Path(epitaka_db)
    return str(p.parent / f"glossary_{lang}{p.suffix}")


# ══════════════════════════════════════════════════════════════════
# Book / paragraph discovery
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


def book_para_bounds(epitaka_db: str, book_id: str) -> tuple[int, int]:
    """(min_para_id, max_para_id) for a book in epitaka.db."""
    with _connect(epitaka_db) as conn:
        row = conn.execute(
            "SELECT MIN(para_id) AS lo, MAX(para_id) AS hi FROM sentences WHERE book_id=?",
            (book_id,),
        ).fetchone()
    lo = row["lo"] if row and row["lo"] is not None else 0
    hi = row["hi"] if row and row["hi"] is not None else 0
    return lo, hi


def chunk_para_ranges(para_lo: int, para_hi: int, chunk_paras: int) -> list[tuple[int, int]]:
    """Split [para_lo, para_hi] into contiguous (lo, hi) ranges of at most chunk_paras each."""
    if para_hi < para_lo:
        return []
    ranges = []
    p = para_lo
    while p <= para_hi:
        hi = min(p + chunk_paras - 1, para_hi)
        ranges.append((p, hi))
        p = hi + 1
    return ranges


def fetch_translated_lines(
    lang_db: str,
    epitaka_db: str,
    book_id: str,
    para_lo: int,
    para_hi: int,
) -> list[dict]:
    """Every (para_id, line_id, pali, translation) triple with a non-empty translation in range."""
    if not Path(lang_db).exists():
        return []
    with _connect(lang_db) as lconn, _connect(epitaka_db) as econn:
        trans_rows = lconn.execute(
            """SELECT para_id, line_id, translation FROM sentences
               WHERE book_id=? AND para_id BETWEEN ? AND ?
                 AND translation IS NOT NULL AND translation != ''
               ORDER BY para_id, line_id""",
            (book_id, para_lo, para_hi),
        ).fetchall()
        if not trans_rows:
            return []

        pali_rows = econn.execute(
            "SELECT para_id, line_id, pali FROM sentences WHERE book_id=? AND para_id BETWEEN ? AND ?",
            (book_id, para_lo, para_hi),
        ).fetchall()
        pali_map = {(r["para_id"], r["line_id"]): (r["pali"] or "").strip() for r in pali_rows}

    lines = []
    for r in trans_rows:
        pali = pali_map.get((r["para_id"], r["line_id"]), "")
        if not pali:
            continue
        lines.append({
            "para_id": r["para_id"],
            "line_id": r["line_id"],
            "pali": pali,
            "translation": r["translation"],
        })
    return lines


# ══════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════

_VERIFY_INSTRUCTIONS = """You are verifying already-published {lang_name} translations of Pāli
Buddhist scripture against BOTH the Pāli source AND its linked texts
(commentary / sub-commentary / mūla / sibling commentary, given below).

Flag a line ONLY when you are genuinely confident it is wrong:
  - The {lang_name} contradicts what the Pāli line actually says.
  - The {lang_name} contradicts the meaning given by the linked
    commentary / mūla / sibling text for that same content.
  - A doctrinal term is rendered by a generic word that loses its
    technical meaning.
  - Obvious grammatical misreading (wrong subject/object, wrong tense,
    negation dropped or added).

Do NOT flag: stylistic differences, synonyms, or wording variations that
preserve the meaning. Do NOT flag a line just because the linked context
uses different phrasing — only flag when the {lang_name} translation
itself is actually incorrect.

Additionally use "severity": "critical" (instead of the default "wrong")
ONLY for these, and only when genuinely confident:
  a) WRONG LANGUAGE — the {lang_name} came out in the wrong language
     entirely (not just one untranslated Pāli proper noun/loanword).
  b) LINE MISALIGNMENT — the translation clearly belongs to a different
     Pāli line than the one it's paired with.
  c) GĀTHĀ OVERRUN — a verse line's translation contains content that
     belongs to a later line, leaving that later line's own translation
     hollow or duplicated.
  d) DISRESPECTFUL REGISTER — casual/slangy/double-meaning wording applied
     to the Buddha, an Arahant, or another venerated figure.

For each error return:
  {{
    "para_id": <int>,
    "line_id": <int>,
    "translation": "<your corrected {lang_name} translation of that line>",
    "conflict": "<brief explanation in {lang_name}: what is wrong, and whether
                  it conflicts with the Pāli itself or with the linked text>",
    "severity": "wrong" | "critical"
  }}

Return ONLY valid JSON, no markdown fences, no prose outside the JSON:
  {{ "remarks": [...] }}

Do NOT include para_id/line_id for lines that look correct."""


def build_system_prompt(lang: str) -> str:
    return (
        f"You are an expert in Pāli Buddhist terminology and {_lang_name(lang)} translation, "
        f"cross-checking translations against canonical commentary.\n\n"
        + _VERIFY_INSTRUCTIONS.format(lang_name=_lang_name(lang))
    )


def build_user_prompt(
    book_id: str,
    para_lo: int,
    para_hi: int,
    lang: str,
    linked_context: str,
    lines: list[dict],
) -> str:
    lname = _lang_name(lang)
    pairs_text = "\n".join(
        f"  para={ln['para_id']} line={ln['line_id']}\n"
        f"  Pāli: {ln['pali']}\n"
        f"  {lname}: {ln['translation']}"
        for ln in lines
    )
    return (
        f"Book: {book_id}  —  paragraphs {para_lo}–{para_hi}\n\n"
        f"{linked_context}\n\n"
        f"══════════════════════════════\n"
        f"PĀLI + {lname.upper()} TRANSLATION LINES TO VERIFY\n"
        f"══════════════════════════════\n"
        f"{pairs_text}\n\n"
        f"Verify these lines against the Pāli and the linked texts above, as instructed.\n"
        f'Return only JSON: {{ "remarks": [...] }}'
    )


# ══════════════════════════════════════════════════════════════════
# Per-chunk processing
# ══════════════════════════════════════════════════════════════════

def process_chunk(
    *,
    book_id: str,
    para_lo: int,
    para_hi: int,
    lang: str,
    params: dict,
    log_dir: str,
    max_context_chars: int,
    max_tokens: int,
    thinking_budget_tokens: int,
    apply: bool,
    dry_run: bool,
    chunk_tag: str,
) -> tuple[int, int]:
    """Returns (remarks_saved, critical_count) for this chunk."""
    lines = fetch_translated_lines(
        lang_db=params["lang_db"],
        epitaka_db=params["epitaka_db"],
        book_id=book_id,
        para_lo=para_lo,
        para_hi=para_hi,
    )
    if not lines:
        print(f"  [{chunk_tag}] No translated lines in paras {para_lo}-{para_hi}. Skipping.")
        return 0, 0

    linked_context = CommentaryContext(
        params, book_id, para_lo, para_hi,
        max_chars=max_context_chars,
    ).build()

    system_prompt = build_system_prompt(lang)
    user_prompt = build_user_prompt(book_id, para_lo, para_hi, lang, linked_context, lines)

    print(f"  [{chunk_tag}] {len(lines)} line(s), "
          f"prompt ~{len(system_prompt) + len(user_prompt)} chars "
          f"(linked context ~{len(linked_context)} chars)")

    if dry_run:
        print(user_prompt[:800], "...\n[dry-run — not calling B.AI]")
        return 0, 0

    raw = ai.call_ai_with_logging(
        prompt=user_prompt,
        system_prompt=system_prompt,
        book_id=book_id,
        chunk_id=chunk_tag,
        log_dir=log_dir,
        max_tokens=max_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
    )
    if raw is None:
        print(f"  [{chunk_tag}] B.AI call failed (no response). Skipping.")
        return 0, 0

    try:
        result = ai.parse_ai_json_response(raw, expected_keys=("remarks",))
    except Exception as exc:
        print(f"  [{chunk_tag}] Could not parse response: {exc}. Skipping.")
        return 0, 0

    remarks = result.get("remarks", [])
    if not remarks:
        print(f"  [{chunk_tag}] No issues flagged.")
        return 0, 0

    critical = sum(1 for r in remarks if str(r.get("severity", "")).lower() == "critical")
    print(f"  [{chunk_tag}] {len(remarks)} line(s) flagged ({critical} critical).")

    # Always log findings as translation_remarks.
    remark_writer = RemarkWriter(params)
    saved = remark_writer.save(book_id, remarks, source_id=book_id)

    # Optionally overwrite sentences.translation with the AI's corrections,
    # grouped by para_id since TranslationWriter.save() takes one para at a time.
    if apply:
        by_para: dict[int, list[dict]] = {}
        for r in remarks:
            pid = r.get("para_id")
            if pid is None or r.get("line_id") is None or not str(r.get("translation") or "").strip():
                continue
            by_para.setdefault(pid, []).append({
                "line_id": r["line_id"],
                "translation": r["translation"],
            })
        writer = TranslationWriter(params)
        updated = 0
        for pid, entries in by_para.items():
            updated += writer.save(book_id, pid, entries)
        print(f"  [{chunk_tag}] --apply: {updated} line(s) overwritten in sentences.translation.")

    return saved, critical


# ══════════════════════════════════════════════════════════════════
# Per-book processing
# ══════════════════════════════════════════════════════════════════

def process_book(book_id: str, args, params: dict) -> tuple[int, int]:
    print("=" * 60)
    print(f"BOOK: {book_id}  lang={args.lang}")
    print("=" * 60)

    if args.start is not None or args.end is not None:
        lo = args.start if args.start is not None else book_para_bounds(params["epitaka_db"], book_id)[0]
        hi = args.end if args.end is not None else book_para_bounds(params["epitaka_db"], book_id)[1]
    else:
        lo, hi = book_para_bounds(params["epitaka_db"], book_id)

    if hi < lo:
        print(f"  Empty/invalid paragraph range ({lo}-{hi}). Skipping.")
        return 0, 0

    ranges = chunk_para_ranges(lo, hi, args.chunk_paras)
    if args.limit:
        ranges = ranges[: args.limit]
    print(f"  Paragraphs {lo}-{hi}, {len(ranges)} chunk(s) of up to {args.chunk_paras} paras each.")

    total_saved = 0
    total_critical = 0
    for idx, (plo, phi) in enumerate(ranges, 1):
        saved, critical = process_chunk(
            book_id=book_id,
            para_lo=plo,
            para_hi=phi,
            lang=args.lang,
            params=params,
            log_dir=args.log_dir,
            max_context_chars=args.max_context_chars,
            max_tokens=args.max_tokens,
            thinking_budget_tokens=args.thinking_budget_tokens,
            apply=args.apply,
            dry_run=args.dry_run,
            chunk_tag=f"verify_p{plo}-{phi}_c{idx}",
        )
        total_saved += saved
        total_critical += critical

    print(f"  Book {book_id} done — {total_saved} remark(s) saved ({total_critical} critical).")
    return total_saved, total_critical


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", required=True, help='Target language code, e.g. "en", "si", "th".')
    parser.add_argument("--book", default="", help="Single book_id to verify.")
    parser.add_argument("--books", default="", help='Comma-separated book_ids, e.g. "Sp-i,Sp-ii".')
    parser.add_argument("--start", type=int, default=None, help="First para_id (default: book's minimum).")
    parser.add_argument("--end", type=int, default=None, help="Last para_id (default: book's maximum).")
    parser.add_argument("--chunk-paras", type=int, default=DEFAULT_CHUNK_PARAS,
                         help=f"Paragraphs per B.AI call (default {DEFAULT_CHUNK_PARAS}).")
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS,
                         help="Char budget passed to CommentaryContext for linked texts.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                         help=f"Total B.AI max_tokens per call, thinking + answer combined "
                              f"(default {DEFAULT_MAX_TOKENS}).")
    parser.add_argument("--thinking-budget-tokens", type=int, default=DEFAULT_THINKING_BUDGET_TOKENS,
                         help="Cap on deepseek-v4-flash's reasoning tokens per call — reasoning "
                              f"stays on, just bounded (default {DEFAULT_THINKING_BUDGET_TOKENS}). "
                              "If chunks keep coming back empty, raise --max-tokens (headroom for "
                              "the answer is max-tokens minus this value) before raising this.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N chunks (testing).")
    parser.add_argument("--epitaka-db", default=EPITAKA_DB)
    parser.add_argument("--lang-db", default="", help="Override path to epitaka_<lang>.db.")
    parser.add_argument("--nissaya-db", default="", help="Override path to the nissaya DB, if used.")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--apply", action="store_true",
                         help="Overwrite sentences.translation with the AI's corrections. "
                              "Without this flag, findings are only logged to translation_remarks.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Build prompts and print the first one per chunk; never call B.AI.")
    args = parser.parse_args()

    epitaka_db = args.epitaka_db
    lang_db = args.lang_db or lang_db_path(epitaka_db, args.lang)
    glossary_db = glossary_db_path(epitaka_db, args.lang)

    params = {
        "epitaka_db": epitaka_db,
        "lang_db": lang_db,
        "glossary_db": glossary_db,
    }
    if args.nissaya_db:
        params["nissaya_db"] = args.nissaya_db

    print(f"[lang] Language  : {args.lang}  ({_lang_name(args.lang)})")
    print(f"[lang] Source DB : {epitaka_db}")
    print(f"[lang] Lang DB   : {lang_db}")
    print(f"[mode] {'APPLY (corrections written to sentences.translation)' if args.apply else 'REPORT-ONLY (remarks logged, nothing overwritten)'}")
    if args.dry_run:
        print("[mode] DRY RUN — no B.AI calls will be made.")

    # Resolve book list
    if args.book.strip():
        book_list = [args.book.strip()]
    elif args.books.strip():
        book_list = [b.strip() for b in args.books.split(",") if b.strip()]
    else:
        book_list = discover_books(lang_db)
        if not book_list:
            print("No books found with translations. Nothing to do.")
            return 1

    grand_saved = 0
    grand_critical = 0
    for book_idx, book_id in enumerate(book_list, 1):
        print(f"\n{'#' * 60}\n# BOOK {book_idx}/{len(book_list)}: {book_id}\n{'#' * 60}")
        try:
            saved, critical = process_book(book_id, args, params)
        except Exception as exc:
            print(f"[ERROR] Book {book_id} failed: {exc}. Continuing.")
            continue
        grand_saved += saved
        grand_critical += critical

    print("\n" + "=" * 60)
    print(f"ALL DONE. Books processed: {len(book_list)}")
    print(f"  Total remarks saved: {grand_saved} ({grand_critical} critical)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
