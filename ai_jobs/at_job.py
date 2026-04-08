"""
jobs/at_job.py  —  AṭṭhakathāTranslatorJob

Translates sentences that have NO sc_id  (aṭṭhakathā, ṭīkā, and any other
commentary texts that were not imported from SuttaCentral).

Key differences from SentenceTranslatorJob (st_job.py)
───────────────────────────────────────────────────────
1.  fetch_headings uses WHERE sc_id IS NULL / '' instead of NOT NULL.
2.  No SC-reference block.  Instead two new context sources are injected:
      a.  MŪLA TRANSLATIONS — for every sentence in the chunk, follow
          book_links backward (dst→src) to find the mūla (root text)
          paragraph and include its already-translated English sentences.
          This ensures the commentary translation uses identical terminology
          to the root-text translation.
      b.  SIMILAR SENTENCE TRANSLATIONS — a lightweight token-overlap search
          over already-translated sentences in the *same book* to surface
          comparable passages for consistent phrasing.
3.  The system prompt is adjusted to reflect these new context sources.
4.  Everything else (chunking, glossary, nissaya, commentary, save, upsert)
    is called unchanged from st_data / st_prompt.

Params (job_params JSON)
────────────────────────
  book_id        : str   — restrict to one book                  — optional
  heading_id     : str   — process a single heading (para_id)    — optional
  batch_size     : int   — headings per run                      (default 10)
  max_tokens     : int   — soft token cap per AI call            (default 3000)
  max_mula_chars : int   — chars of mūla translation to include  (default 3000)
  max_sim_chars  : int   — chars of similar-sentence block       (default 2000)
  overwrite      : bool  — re-translate existing translations    (default false)
  key_ids        : str   — comma-sep API key IDs, blank = all
  log_dir        : str   — folder for prompt/response debug logs
                           (default /tmp/at_logs)
"""

import logging
from typing import Any

from ai_jobs.base_job import BaseJob
import ai_jobs.st_data  as data
import ai_jobs.st_prompt as prompt_lib
import ai_jobs.at_data   as at_data   # new data helpers (see at_data.py)
import ai_jobs.at_prompt as at_prompt  # new prompt helpers (see at_prompt.py)

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "/tmp/at_logs"


class AṭṭhakathāTranslatorJob(BaseJob):
    display_name = "Aṭṭhakathā Translator"
    param_schema = {
        "book_id":        {"type": "string",  "label": "Book ID (e.g. DN-a1)",                        "default": ""},
        "heading_id":     {"type": "string",  "label": "Single heading para_id (optional)",           "default": ""},
        "batch_size":     {"type": "integer", "label": "Headings per run",                            "default": 10},
        "max_tokens":     {"type": "integer", "label": "Soft token cap per AI call",                  "default": -1},
        "max_mula_chars": {"type": "integer", "label": "Max chars of mūla translation to include",    "default": -1},
        "max_sim_chars":  {"type": "integer", "label": "Max chars of similar-sentence block",         "default": 2000},
        "overwrite":      {"type": "boolean", "label": "Re-translate already-translated sentences",   "default": False},
        "key_ids":        {"type": "string",  "label": "API key IDs (comma-sep, blank = all)",        "default": ""},
        "log_dir":        {"type": "string",  "label": "Folder for prompt/response debug logs",       "default": DEFAULT_LOG_DIR},
    }

    def run(self) -> Any:
        max_tokens     = int(self.params.get("max_tokens",     -1))
        max_mula_chars = int(self.params.get("max_mula_chars", -1))
        max_sim_chars  = int(self.params.get("max_sim_chars",  2000))
        overwrite      = bool(self.params.get("overwrite",     False))
        log_dir        = self.params.get("log_dir") or DEFAULT_LOG_DIR

        self.log_info("=" * 60)
        self.log_info("AṭṭhakathāTranslator — START")
        self.log_info(f"Params: {self.params}")
        self.log_info(f"Debug logs → {log_dir}")
        self.log_info("=" * 60)

        # ── Validate DB ────────────────────────────────────────────
        try:
            data.validate_nissaya_db(self.params, self.log_info, self.log_warn)
        except RuntimeError as exc:
            self.log_error(f"DB config error: {exc}")
            raise

        # ── Fetch headings (no sc_id) ──────────────────────────────
        self.log_info("[RUN] Calling at_data.fetch_headings_no_sc_id ...")
        try:
            headings = at_data.fetch_headings_no_sc_id(self.params, self.log_info)
        except Exception as exc:
            self.log_error(f"[RUN] fetch_headings_no_sc_id raised {type(exc).__name__}: {exc}")
            raise
        self.log_info(f"[RUN] fetch_headings_no_sc_id returned {len(headings)} heading(s).")

        if not headings:
            self.log_info("No headings to process. Done.")
            return 0

        total_updated  = 0
        total_glossary = 0

        # ── Per-heading loop ───────────────────────────────────────
        for h_idx, heading in enumerate(headings, 1):
            # sc_id is NULL/empty for these texts — use book+para as identifier
            heading_label = (
                heading.get("title")
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
                self.log_info(f"  No pending sentences. Skipping heading.")
                continue

            self.log_debug(f"  Found {total_pending} sentences to translate.")

            # ── 2. Composite src_lines for this heading ────────────
            src_lines: list[tuple] = list(dict.fromkeys(
                (para["book_id"], para["para_id"], sent["line_id"])
                for para in paragraphs
                for sent in para.get("sentences", [])
                if sent.get("line_id") is not None
            ))

            # ── 3. Glossary ────────────────────────────────────────
            pali_text_for_glossary = "\n".join(
                sent["pali_sentence"]
                for para in paragraphs
                for sent in para.get("pending", [])
            )
            pali_ngrams     = prompt_lib.extract_pali_ngrams(pali_text_for_glossary)
            glossary_block  = data.fetch_glossary_block(
                pali_ngrams, self.log_info, self.log_warn
            )

            # ── 4. Nissaya maps ────────────────────────────────────
            nissaya_blocks: dict[int, str] = {}
            for para in paragraphs:
                niss_map = data.fetch_nissaya_map(
                    self.params, para["book_id"], para["para_id"], self.log_info
                )
                nissaya_blocks[para["para_id"]] = data.build_nissaya_block(
                    para["sentences"], niss_map
                )

            # ── 5. Commentary block (sub-commentary of this text) ──
            # For aṭṭhakathā/ṭīkā we still look forward in book_links to find
            # any deeper sub-commentary that comments on *these* lines.
            commentary_block = data.fetch_commentary_block(
                self.params, src_lines,
                max_chars=3000, log_info=self.log_info, log_warn=self.log_warn
            )

            # ── 6. Pāli definitions ────────────────────────────────
            pali_defs_block = data.fetch_pali_definitions_block(
                pali_text_for_glossary, self.params,
                self.log_info, self.log_warn
            )

            # ── 7. NEW: Mūla translation block ────────────────────
            # Follow book_links *in reverse* (dst→src) to find the mūla paragraph
            # that this commentary passage comments on, then retrieve its
            # already-translated English sentences.
            mula_block = at_data.fetch_mula_translation_block(
                self.params, src_lines,
                max_chars=max_mula_chars,
                log_info=self.log_info, log_warn=self.log_warn,
            )

            # ── 8. NEW: Similar sentence translations ──────────────
            # Token-overlap search over already-translated sentences in the
            # same book — surfaces comparable passages for consistent phrasing.
            similar_block = at_data.fetch_similar_translations_block(
                self.params,
                book_id   = heading["book_id"],
                pali_text = pali_text_for_glossary,
                max_chars = max_sim_chars,
                log_info  = self.log_info, log_warn=self.log_warn,
            )

            # ── 9. Chunk paragraphs ────────────────────────────────
            chunks = prompt_lib.chunk_paragraphs(paragraphs, max_tokens=max_tokens)
            self.log_info(
                f"  {len(paragraphs)} paragraph(s) → "
                f"{len(chunks)} chunk(s) (max_tokens={max_tokens})."
            )

            # ── 10. Per-chunk: build prompt → AI → save ────────────
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
                    self.log_info
                )

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

                # call_ai_with_logging bakes in SYSTEM_PROMPT from st_prompt.
                # We override by wrapping ask_ai so it always receives the
                # commentary-specific system prompt from at_prompt regardless
                # of what call_ai_with_logging passes as the second argument.
                _at_system = at_prompt.SYSTEM_PROMPT

                def _ask_ai_at(user_msg: str, _ignored_system: str) -> str:
                    return self.ask_ai(user_msg, _at_system)

                raw = prompt_lib.call_ai_with_logging(
                    ask_ai_fn = _ask_ai_at,
                    prompt    = user_prompt,
                    sc_id     = heading_label,   # used only for log filename
                    chunk_idx = c_idx,
                    log_dir   = log_dir,
                    log_info  = self.log_info,
                    log_sucess= self.log_sucess,
                    log_error = self.log_error,
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
            f"AṭṭhakathāTranslator — DONE. "
            f"Sentences updated: {total_updated}. "
            f"Glossary terms added: {total_glossary}."
        )
        self.log_info("=" * 60)
        return total_updated