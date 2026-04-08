"""
jobs/ut_job.py  —  UnifiedTranslatorJob

Translates ALL headings regardless of whether they have an sc_id or not.
This supersedes the split between SentenceTranslatorJob (sc_id required)
and AṭṭhakathāTranslatorJob (sc_id absent).

Key behaviour
─────────────
• Fetches headings the same way as at_job — no sc_id filter.
• For every heading, walks book_links IN REVERSE (current text = dst, look
  up src) to collect all ancestor layers that already have English translations:
    – Translating aṭṭhakathā  →  mūla layer included automatically
    – Translating ṭīkā         →  mūla + aṭṭhakathā layers included
    – Translating mūla sutta   →  no ancestors found, block omitted
  No character limit is applied to the ancestor block.
• Also injects SIMILAR ALREADY-TRANSLATED SENTENCES from the same book
  (token-overlap search, unchanged from at_job).
• Everything else — nissaya, glossary, commentary (forward links), pāli
  definitions, previous paragraph, chunking, AI dispatch, save, upsert —
  is called unchanged from st_data / st_prompt / at_data / at_prompt.

Code reuse
──────────
  st_data   — all DB reads/writes except headings + mūla layers
  at_data   — fetch_headings_no_sc_id (with sc_id filter removed),
               fetch_mula_layers_block (new multi-layer reverse lookup),
               fetch_similar_translations_block
  st_prompt — chunk_paragraphs, extract_pali_ngrams, call_ai_with_logging,
               parse_response, build_nissaya_block_for_prompt
  at_prompt — SYSTEM_PROMPT (commentary-aware), build_prompt, USER_TEMPLATE

Params (job_params JSON)
────────────────────────
  book_id    : str   — restrict to one book (e.g. "DN-a1")     — optional
  heading_id : str   — single heading by para_id               — optional
  batch_size : int   — headings per run                        (default 10)
  max_tokens : int   — soft token cap per AI call              (default 3000)
  max_sim_chars : int — chars of similar-sentence block        (default 2000)
  overwrite  : bool  — re-translate existing translations      (default false)
  key_ids    : str   — comma-sep API key IDs, blank = all
  log_dir    : str   — folder for debug logs                   (default /tmp/ut_logs)
"""

import logging
from typing import Any

from ai_jobs.base_job import BaseJob
import ai_jobs.st_data  as data
import ai_jobs.st_prompt as prompt_lib
import ai_jobs.at_data   as at_data
import ai_jobs.at_prompt as at_prompt

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "/tmp/ut_logs"


class UnifiedTranslatorJob(BaseJob):
    display_name = "Unified Translator"
    param_schema = {
        "book_id":      {"type": "string",  "label": "Book ID (e.g. DN1 or DN-a1)",                "default": ""},
        "heading_id":   {"type": "string",  "label": "Single heading para_id (optional)",           "default": ""},
        "batch_size":   {"type": "integer", "label": "Headings per run",                            "default": 10},
        "max_tokens":   {"type": "integer", "label": "Soft token cap per AI call",                  "default": -1},
        "max_sim_chars":{"type": "integer", "label": "Max chars of similar-sentence block",         "default": 2000},
        "overwrite":    {"type": "boolean", "label": "Re-translate already-translated sentences",   "default": False},
        "key_ids":      {"type": "string",  "label": "API key IDs (comma-sep, blank = all)",        "default": ""},
        "log_dir":      {"type": "string",  "label": "Folder for prompt/response debug logs",       "default": DEFAULT_LOG_DIR},
    }

    def run(self) -> Any:
        max_tokens    = int(self.params.get("max_tokens",    -1))
        max_sim_chars = int(self.params.get("max_sim_chars", 2000))
        overwrite     = bool(self.params.get("overwrite",    False))
        log_dir       = self.params.get("log_dir") or DEFAULT_LOG_DIR

        self.log_info("=" * 60)
        self.log_info("UnifiedTranslator — START")
        self.log_info(f"Params: {self.params}")
        self.log_info(f"Debug logs → {log_dir}")
        self.log_info("=" * 60)

        # ── Validate DB ────────────────────────────────────────────
        try:
            data.validate_nissaya_db(self.params, self.log_info, self.log_warn)
        except RuntimeError as exc:
            self.log_error(f"DB config error: {exc}")
            raise

        # ── Fetch headings (all — regardless of sc_id) ─────────────
        # at_data.fetch_headings_no_sc_id handles book_id / heading_id
        # filters and the overwrite / batch_size logic. We reuse it
        # unchanged; sc_id being null or not is simply not checked here.
        # If you also want to process sc_id headings, use _fetch_all_headings
        # (defined below) instead.
        self.log_info("[RUN] Fetching headings ...")
        try:
            headings = _fetch_all_headings(self.params, self.log_info)
        except Exception as exc:
            self.log_error(f"[RUN] fetch_headings raised {type(exc).__name__}: {exc}")
            raise
        self.log_info(f"[RUN] {len(headings)} heading(s) to process.")

        if not headings:
            self.log_info("No headings to process. Done.")
            return 0

        total_updated  = 0
        total_glossary = 0

        # ── Per-heading loop ───────────────────────────────────────
        for h_idx, heading in enumerate(headings, 1):
            heading_label = (
                heading.get("title")
                or heading.get("sc_id")
                or f"{heading['book_id']}§{heading['para_id']}"
            )

            self.log_info("-" * 60)
            self.log_info(
                f"[{h_idx}/{len(headings)}] label={heading_label!r}  "
                f"book={heading['book_id']}  para={heading['para_id']}  "
                f"chapter_len={heading.get('chapter_len') or 1}"
            )
            self.heartbeat()

            # ── 1. Paragraphs / sentences ──────────────────────────
            try:
                paragraphs = data.fetch_paragraphs_for_heading(
                    self.params, heading, overwrite,
                    self.log_info, self.log_warn,
                )
            except Exception as exc:
                self.log_error(f"fetch_paragraphs_for_heading failed: {exc}. Skipping.")
                continue

            total_pending = sum(len(p.get("pending", [])) for p in paragraphs)
            if total_pending == 0:
                self.log_info("  No pending sentences. Skipping heading.")
                continue

            self.log_debug(f"  Found {total_pending} sentences to translate.")

            # ── 2. Composite src_lines ─────────────────────────────
            # (book_id, para_id, line_id) triples for every sentence under
            # this heading.  Used for both forward (commentary) and reverse
            # (mūla ancestor) book_links lookups.
            src_lines: list[tuple] = list(dict.fromkeys(
                (para["book_id"], para["para_id"], sent["line_id"])
                for para in paragraphs
                for sent in para.get("sentences", [])
                if sent.get("line_id") is not None
            ))

            # ── 3. Pāli text for glossary / definitions / similar ──
            pali_text_for_context = "\n".join(
                sent["pali_sentence"]
                for para in paragraphs
                for sent in para.get("pending", [])
            )

            # ── 4. Glossary ────────────────────────────────────────
            pali_ngrams    = prompt_lib.extract_pali_ngrams(pali_text_for_context)
            glossary_block = data.fetch_glossary_block(
                pali_ngrams, self.log_info, self.log_warn
            )

            # ── 5. Nissaya maps ────────────────────────────────────
            nissaya_blocks: dict[int, str] = {}
            for para in paragraphs:
                niss_map = data.fetch_nissaya_map(
                    self.params, para["book_id"], para["para_id"], self.log_info
                )
                nissaya_blocks[para["para_id"]] = data.build_nissaya_block(
                    para["sentences"], niss_map
                )

            # ── 6. Commentary block (forward: sub-commentary of this text) ─
            # book_links src=this text → dst=deeper commentary/ṭīkā
            commentary_block = data.fetch_commentary_block(
                self.params, src_lines,
                max_chars=-1,   # no limit — let the AI see everything
                log_info=self.log_info, log_warn=self.log_warn,
            )

            # ── 7. Pāli definitions ────────────────────────────────
            pali_defs_block = data.fetch_pali_definitions_block(
                pali_text_for_context, self.params,
                self.log_info, self.log_warn,
            )

            # ── 8. Mūla ancestor layers (reverse book_links walk) ──
            # Walk book_links in reverse: current text is dst, find src.
            # Continues layer by layer (mūla → aṭṭhakathā → ṭīkā order)
            # until no further ancestors exist or all ancestor books have
            # been visited.  No character cap.
            mula_block = at_data.fetch_mula_layers_block(
                params    = self.params,
                book_id   = heading["book_id"],
                src_lines = src_lines,
                log_info  = self.log_info,
                log_warn  = self.log_warn,
            )

            # ── 9. Similar sentence translations (same book) ───────
            similar_block = at_data.fetch_similar_translations_block(
                self.params,
                book_id   = heading["book_id"],
                pali_text = pali_text_for_context,
                max_chars = max_sim_chars,
                log_info  = self.log_info,
                log_warn  = self.log_warn,
            )

            # ── 10. Chunk paragraphs into token-safe groups ────────
            chunks = prompt_lib.chunk_paragraphs(paragraphs, max_tokens=max_tokens)
            self.log_info(
                f"  {len(paragraphs)} paragraph(s) → "
                f"{len(chunks)} chunk(s) (max_tokens={max_tokens})."
            )

            # ── 11. Per-chunk: build prompt → AI → save ────────────
            for c_idx, chunk in enumerate(chunks, 1):
                n_sentences = sum(len(p["pending"]) for p in chunk)
                self.log_info(
                    f"  Chunk {c_idx}/{len(chunks)}: "
                    f"{n_sentences} sentence(s) across {len(chunk)} para(s)."
                )
                self.heartbeat()

                # Previous paragraph for style continuity
                prev_para_translation = data.fetch_previous_paragraph_translation(
                    self.params, heading["book_id"],
                    chunk[0]["para_id"],
                    self.log_info,
                )

                # Build prompt using at_prompt.build_prompt (commentary-aware
                # template with mūla + similar blocks).
                user_prompt, flat_sentences = at_prompt.build_prompt(
                    heading_label    = heading_label,
                    paragraphs       = chunk,
                    nissaya_blocks   = nissaya_blocks,
                    glossary_block   = glossary_block,
                    commentary_block = commentary_block,
                    pali_defs_block  = pali_defs_block,
                    mula_block       = mula_block,
                    similar_block    = similar_block,
                    prev_para_text   = prev_para_translation,
                )

                # Use at_prompt.SYSTEM_PROMPT (commentary-aware).
                # call_ai_with_logging passes its own SYSTEM_PROMPT from
                # st_prompt — we override via the wrapper below.
                _system = at_prompt.SYSTEM_PROMPT

                def _ask_ai(user_msg: str, _ignored_system: str) -> str:
                    return self.ask_ai(user_msg, _system)

                raw = prompt_lib.call_ai_with_logging(
                    ask_ai_fn  = _ask_ai,
                    prompt     = user_prompt,
                    sc_id      = heading_label,   # only used for log filename
                    chunk_idx  = c_idx,
                    log_dir    = log_dir,
                    log_info   = self.log_info,
                    log_sucess = self.log_sucess,
                    log_error  = self.log_error,
                )
                if raw is None:
                    self.log_warn(f"  Chunk {c_idx} timed out. Skipping.")
                    continue

                try:
                    result = prompt_lib.parse_response(raw)
                except Exception as exc:
                    self.log_error(
                        f"  parse_response failed for chunk {c_idx}: {exc}. Skipping."
                    )
                    continue

                translations = result.get("translations", [])
                new_terms    = result.get("glossary",     [])
                self.log_info(
                    f"  Parsed: {len(translations)} translation(s), "
                    f"{len(new_terms)} new glossary term(s)."
                )

                # Save translations grouped by para_id
                by_para: dict[int, list[dict]] = {}
                for t in translations:
                    pid = t.get("para_id")
                    if pid is None:
                        matched = [
                            s for s in flat_sentences
                            if s["line_id"] == t.get("line_id")
                        ]
                        pid = matched[0]["para_id"] if matched else None
                    if pid is not None:
                        by_para.setdefault(pid, []).append(t)
                    else:
                        self.log_warn(f"  Cannot determine para_id for: {t}")

                for para_id, para_translations in by_para.items():
                    book_id = next(
                        (p["book_id"] for p in chunk if p["para_id"] == para_id),
                        heading["book_id"],
                    )
                    updated = data.save_translations(
                        self.params, book_id, para_id, para_translations,
                        self.log_info, self.log_warn, self.log_error,
                    )
                    total_updated += updated

                if new_terms:
                    inserted = data.upsert_glossary(
                        new_terms, heading_label,
                        self.log_info, self.log_warn, self.log_error,
                    )
                    total_glossary += inserted

            self.log_info(
                f"  Heading done. "
                f"Running totals — sentences: {total_updated}, "
                f"glossary: {total_glossary}."
            )

        self.log_info("=" * 60)
        self.log_info(
            f"UnifiedTranslator — DONE. "
            f"Sentences updated: {total_updated}. "
            f"Glossary terms added: {total_glossary}."
        )
        self.log_info("=" * 60)
        return total_updated


# ══════════════════════════════════════════════════════════════════
# Internal: fetch ALL headings (sc_id present or absent)
# ══════════════════════════════════════════════════════════════════

def _fetch_all_headings(params: dict, log_info) -> list[dict]:
    """
    Variant of at_data.fetch_headings_no_sc_id that returns headings
    regardless of sc_id value.

    Params consumed: book_id, heading_id, batch_size, overwrite.
    """
    import sqlite3
    from contextlib import contextmanager
    from config import NISSAYA_DB

    def _nissaya_path(p: dict) -> str:
        path = p.get("nissaya_db") or NISSAYA_DB
        if not path:
            raise RuntimeError("nissaya_db path is not configured.")
        return str(path)

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

    book_id_filter    = params.get("book_id",    "").strip()
    heading_id_filter = params.get("heading_id", "").strip()
    batch_size        = int(params.get("batch_size", 10))
    overwrite         = bool(params.get("overwrite", False))

    log_info(
        f"[UT-DB] _fetch_all_headings — "
        f"book_id={book_id_filter!r}, heading_id={heading_id_filter!r}, "
        f"batch_size={batch_size}, overwrite={overwrite}"
    )

    path = _nissaya_path(params)
    try:
        with _connect(path) as conn:

            if heading_id_filter:
                rows = conn.execute(
                    "SELECT * FROM headings WHERE para_id = ?",
                    (int(heading_id_filter),)
                ).fetchall()
                candidates = [dict(r) for r in rows]

            elif book_id_filter:
                rows = conn.execute(
                    "SELECT * FROM headings WHERE book_id = ? ORDER BY para_id",
                    (book_id_filter,)
                ).fetchall()
                candidates = [dict(r) for r in rows]

            else:
                rows = conn.execute(
                    "SELECT * FROM headings ORDER BY book_id, para_id"
                ).fetchall()
                candidates = [dict(r) for r in rows]

            log_info(f"[UT-DB] {len(candidates)} candidate heading(s).")

            if overwrite or not candidates:
                result = candidates[:batch_size]
            else:
                result = []
                for h in candidates:
                    if len(result) >= batch_size:
                        break
                    start = h["para_id"]
                    end   = start + int(h.get("chapter_len") or 1)
                    row = conn.execute(
                        """SELECT COUNT(*) FROM sentences
                           WHERE book_id = ?
                             AND para_id >= ? AND para_id < ?
                             AND (english_translation IS NULL
                                  OR english_translation = '')""",
                        (h["book_id"], start, end)
                    ).fetchone()
                    if row[0] > 0:
                        result.append(h)

    except Exception as exc:
        log_info(f"[UT-DB] _fetch_all_headings EXCEPTION — {type(exc).__name__}: {exc}")
        raise

    log_info(f"[UT-DB] _fetch_all_headings → {len(result)} heading(s) need work.")
    return result