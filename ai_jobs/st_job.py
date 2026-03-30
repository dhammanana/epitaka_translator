"""
jobs/st_job.py  —  SentenceTranslatorJob orchestrator.

All DB I/O is in st_data.py (each function opens/closes its own connection).
All prompt/AI logic is in st_prompt.py.
This file only contains the run() loop.

Params (job_params JSON)
------------------------
  book_id       : str   — restrict to one book (e.g. "DN1")  — optional
  sc_id         : str   — process a single heading sc_id     — optional
  batch_size    : int   — headings per run                   (default 5)
  max_ref_chars : int   — chars of SC reference sent to AI   (default 4000)
  max_tokens    : int   — soft token cap per AI call         (default 5000)
  overwrite     : bool  — re-translate existing translations (default false)
  key_ids       : str   — comma-sep API key IDs, blank = all
  log_dir       : str   — folder for prompt/response debug logs
                          (default /tmp/st_logs)
"""

import logging
from typing import Any

from ai_jobs.base_job import BaseJob
import ai_jobs.st_data as data
import ai_jobs.st_prompt as prompt_lib

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "/tmp/st_logs"


class SentenceTranslatorJob(BaseJob):
    display_name = "Sentence Translator"
    param_schema = {
        "book_id":       {"type": "string",  "label": "Book ID (e.g. DN1)",                        "default": ""},
        "sc_id":         {"type": "string",  "label": "Single heading SC-ID (optional)",            "default": ""},
        "batch_size":    {"type": "integer", "label": "Headings per run",                           "default": 500},
        "max_ref_chars": {"type": "integer", "label": "Max chars of SC reference text sent to AI",  "default": -1},
        "max_tokens":    {"type": "integer", "label": "Soft token cap per AI call",                 "default": 3000},
        "overwrite":     {"type": "boolean", "label": "Re-translate already-translated sentences",  "default": False},
        "key_ids":       {"type": "string",  "label": "API key IDs (comma-sep, blank = all)",       "default": ""},
        "log_dir":       {"type": "string",  "label": "Folder for prompt/response debug logs",      "default": DEFAULT_LOG_DIR},
    }

    def run(self) -> Any:
        max_ref_chars = int(self.params.get("max_ref_chars") or self.params.get("max_chars", 4000))
        max_tokens    = int(self.params.get("max_tokens",    3000))
        overwrite     = bool(self.params.get("overwrite",    False))
        log_dir       = self.params.get("log_dir") or DEFAULT_LOG_DIR

        self.log_info("=" * 60)
        self.log_info("SentenceTranslator — START")
        self.log_info(f"Params: {self.params}")
        self.log_info(f"Debug logs → {log_dir}")
        self.log_info("=" * 60)

        # ── Validate DBs (quick open/close, no connection kept) ────
        try:
            data.validate_nissaya_db(self.params, self.log_info, self.log_warn)
            data.validate_sc_db(self.params, self.log_info, self.log_warn)
        except RuntimeError as exc:
            self.log_error(f"DB config error: {exc}")
            raise

        # ── Fetch headings (open/close inside) ────────────────────
        self.log_info("[RUN] Calling fetch_headings ...")
        try:
            headings = data.fetch_headings(self.params, self.log_info)
        except Exception as exc:
            self.log_error(f"[RUN] fetch_headings raised {type(exc).__name__}: {exc}")
            raise
        self.log_info(f"[RUN] fetch_headings returned {len(headings)} heading(s).")

        if not headings:
            self.log_info("No headings to process. Done.")
            return 0

        self.log_info(f"Will process {len(headings)} heading(s).")

        total_updated  = 0
        total_glossary = 0

        # ── Per-heading loop ───────────────────────────────────────
        for h_idx, heading in enumerate(headings, 1):
            sc_id  = heading["sc_id"]
            title  = heading.get("title") or sc_id

            self.log_info("-" * 60)
            self.log_info(
                f"[{h_idx}/{len(headings)}] sc_id={sc_id!r}  title={title!r}  "
                f"book={heading['book_id']}  para={heading['para_id']}  "
                f"chapter_len={heading.get('chapter_len') or 1}"
            )
            self.heartbeat()

            # ── 1. Collect all paragraphs under this heading ───────
            # (open/close inside fetch_paragraphs_for_heading)
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
                self.log_info(f"  No pending sentences in {len(paragraphs)} paragraphs. Skipping heading.")
                continue
            
            self.log_debug(f"  Found {total_pending} sentences to translate.")


            # ── 2. SC reference text (open/close inside) ───────────
            try:
                ref = data.fetch_sc_reference(
                    self.params, sc_id, self.log_info, self.log_warn
                )
            except Exception as exc:
                self.log_error(f"fetch_sc_reference failed: {exc}. Skipping.")
                continue

            # ── 3. Glossary block — phrase-aware ngram lookup ─────
            pali_text_for_glossary = "\n".join(
                sent["pali_sentence"] 
                for para in paragraphs 
                for sent in para.get("pending", [])
            )
            pali_ngrams = prompt_lib.extract_pali_ngrams(pali_text_for_glossary)
            
            glossary_block = data.fetch_glossary_block(
                pali_ngrams, self.log_info, self.log_warn
            )

            # ── 4. Nissaya maps for each para (open/close each) ────
            nissaya_blocks: dict[int, str] = {}
            for para in paragraphs:
                niss_map = data.fetch_nissaya_map(
                    self.params, para["book_id"], para["para_id"], self.log_info
                )
                nissaya_blocks[para["para_id"]] = data.build_nissaya_block(
                    para["sentences"], niss_map
                )

            # ── 5. Commentary & sub-commentary blocks (via book_links) ────
            # Build the full composite key (book_id, para_id, line_id) for every
            # sentence in this heading. line_id is only unique within a paragraph,
            # so we must pass all three columns to avoid cross-paragraph false matches.
            src_lines: list[tuple] = list(dict.fromkeys(
                (para["book_id"], para["para_id"], sent["line_id"])
                for para in paragraphs
                for sent in para.get("sentences", [])
                if sent.get("line_id") is not None
            ))
            commentary_block = data.fetch_commentary_block(
                self.params, src_lines,
                max_chars=3000, log_info=self.log_info, log_warn=self.log_warn
            )

            # ── 6. Pali definitions (dictionary lookup for difficult terms) ─
            pali_defs_block = data.fetch_pali_definitions_block(
                pali_text_for_glossary, self.params,
                self.log_info, self.log_warn
            )

            # ── 7. Split paragraphs into token-safe chunks ─────────
            # NOTE: max_sentences parameter removed — we now chunk by paragraph
            # to limit at paragraph level instead of sentence level
            chunks = prompt_lib.chunk_paragraphs(
                paragraphs,
                max_tokens = max_tokens,
            )
            self.log_info(
                f"  {len(paragraphs)} paragraph(s) → "
                f"{len(chunks)} chunk(s) (max_tokens={max_tokens})."
            )

            # ── 8. Per-chunk: build prompt → call AI → save ────────
            for c_idx, chunk in enumerate(chunks, 1):
                n_sentences = sum(len(p["pending"]) for p in chunk)
                self.log_info(
                    f"  Chunk {c_idx}/{len(chunks)}: "
                    f"{n_sentences} sentence(s) across {len(chunk)} para(s)."
                )
                self.heartbeat()

                # ── Get previous translated paragraph for context ───
                prev_para_translation = data.fetch_previous_paragraph_translation(
                    self.params, heading["book_id"],
                    chunk[0]["para_id"],  # para_id of first para in chunk
                    self.log_info
                )

                user_prompt, flat_sentences = prompt_lib.build_prompt(
                    sc_id                = sc_id,
                    sutta_name           = ref["sutta_name"],
                    paragraphs           = chunk,
                    pali_text            = ref["pali_text"],
                    en_text              = ref["en_text"],
                    nissaya_blocks       = nissaya_blocks,
                    glossary_block       = glossary_block,
                    commentary_block     = commentary_block,
                    pali_defs_block      = pali_defs_block,
                    prev_para_text       = prev_para_translation,
                    max_ref_chars        = max_ref_chars,
                )

                # AI call — prompt and response written to log_dir
                raw = prompt_lib.call_ai_with_logging(
                    ask_ai_fn = self.ask_ai,
                    prompt    = user_prompt,
                    sc_id     = sc_id,
                    chunk_idx = c_idx,
                    log_dir   = log_dir,
                    log_info  = self.log_info,
                    log_sucess= self.log_sucess,
                    log_error = self.log_error,
                )
                if raw is None:
                    self.log_warn(f"  Chunk {c_idx} timed out. Skipping.")
                    continue

                # Parse
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

                # Save translations — group by para_id, each save opens/closes DB
                by_para: dict[int, list[dict]] = {}
                for t in translations:
                    pid = t.get("para_id")
                    if pid is None:
                        # AI dropped para_id — infer from flat_sentences by line_id
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

                # Upsert glossary — opens/closes glossary DB inside
                if new_terms:
                    inserted = data.upsert_glossary(
                        new_terms, sc_id,
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
            f"SentenceTranslator — DONE. "
            f"Sentences updated: {total_updated}. "
            f"Glossary terms added: {total_glossary}."
        )
        self.log_info("=" * 60)
        return total_updated