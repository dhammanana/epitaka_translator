"""
ai_jobs/sinhala_job/si_job.py  —  SinhalaTranslatorJob orchestrator.

All DB I/O goes through si_data (which delegates to NissayaContext).
All prompt / AI logic is in si_prompt.py.
This file contains only the run() loop.

Params (job_params JSON)
------------------------
  book_id       : str   — restrict to one nissaya book_id (e.g. "A-i")   — optional
  sc_id         : str   — process a single heading (matched by sc_id)     — optional
  batch_size    : int   — headings per run                                (default 5)
  max_ref_chars : int   — max chars of Sinhala reference sent to AI       (default -1 = unlimited)
  max_tokens    : int   — soft token cap per AI call                      (default 3000)
  overwrite     : bool  — re-translate already-translated sentences        (default false)
  nissaya_db    : str   — path to nissaya.db
  sinhala_db    : str   — path to sinhala.db                              (default "sinhala.db")
  output_db     : str   — path to epitaka_si.db                           (default "epitaka_si.db")
  key_ids       : str   — comma-sep API key IDs, blank = all
  log_dir       : str   — folder for prompt/response debug logs            (default /tmp/si_logs)
"""

import logging
import threading
from typing import Any

from ai_jobs.base_job import BaseJob
import ai_jobs.sinhala_job.si_data   as data
import ai_jobs.sinhala_job.si_prompt as prompt_lib

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "/tmp/si_logs"


class SinhalaTranslatorJob(BaseJob):
    display_name = "Sinhala Translator"
    param_schema = {
        "book_id":       {"type": "string",  "label": "Book ID (e.g. A-i)",                        "default": ""},
        "sc_id":         {"type": "string",  "label": "Single heading SC-ID (optional)",            "default": ""},
        "batch_size":    {"type": "integer", "label": "Headings per run",                           "default": 5},
        "max_ref_chars": {"type": "integer", "label": "Max chars of Sinhala reference (-1 = all)", "default": -1},
        "max_tokens":    {"type": "integer", "label": "Soft token cap per AI call",                 "default": 3000},
        "overwrite":     {"type": "boolean", "label": "Re-translate already-translated sentences",  "default": False},
        "nissaya_db":    {"type": "string",  "label": "Path to nissaya.db",                         "default": ""},
        "sinhala_db":    {"type": "string",  "label": "Path to sinhala.db",                         "default": "sinhala.db"},
        "output_db":     {"type": "string",  "label": "Path to epitaka_si.db",                      "default": "epitaka_si.db"},
        "key_ids":       {"type": "string",  "label": "API key IDs (comma-sep, blank = all)",       "default": ""},
        "log_dir":       {"type": "string",  "label": "Folder for prompt/response debug logs",      "default": DEFAULT_LOG_DIR},
    }

    def run(self) -> Any:
        max_ref_chars = int(self.params.get("max_ref_chars") or -1)
        max_tokens    = int(self.params.get("max_tokens",   2000))
        overwrite     = bool(self.params.get("overwrite",   False))
        log_dir       = self.params.get("log_dir") or DEFAULT_LOG_DIR

        self.log_info("=" * 60)
        self.log_info("SinhalaTranslator — START")
        self.log_info(f"Params: {self.params}")
        self.log_info(f"Debug logs → {log_dir}")
        self.log_info("=" * 60)

        # ── Validate / prepare databases ───────────────────────────
        try:
            data.validate_nissaya_db(self.params, self.log_info, self.log_warn)
            data.validate_sinhala_db(self.params, self.log_info, self.log_warn)
            data.ensure_output_db(self.params, self.log_info)
            data.ensure_sinhala_glossary_table(self.log_info)
        except RuntimeError as exc:
            self.log_error(f"DB config error: {exc}")
            raise

        # ── Fetch headings ─────────────────────────────────────────
        self.log_info("[RUN] Fetching headings ...")
        try:
            headings = data.fetch_headings(self.params, self.log_info)
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
            book_id = heading["book_id"]
            para_id = heading["para_id"]
            title   = heading.get("title") or heading.get("sc_id") or f"{book_id}:{para_id}"

            self.log_info("-" * 60)
            self.log_info(
                f"[{h_idx}/{len(headings)}] book_id={book_id!r}  "
                f"para_id={para_id}  title={title!r}  "
                f"chapter_len={heading.get('chapter_len') or 1}"
            )
            self.heartbeat()

            # ── 1. Paragraphs under this heading ───────────────────
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
                self.log_info(
                    f"  No pending sentences in {len(paragraphs)} paragraph(s). Skipping."
                )
                continue
            self.log_info(f"  {total_pending} sentence(s) to translate.")

            # ── 2. Sinhala reference (authentic published text) ────
            try:
                ref = data.fetch_sinhala_reference(
                    self.params, book_id, para_id,
                    self.log_info, self.log_warn,
                )
            except Exception as exc:
                self.log_error(f"fetch_sinhala_reference failed: {exc}. Skipping.")
                continue

            section_title = ref["title"]
            pali_block    = ref["pali_block"]
            sinhala_block = ref["sinhala_block"]

            # ── 3. Pāli text for downstream lookups ───────────────
            pali_text_for_lookups = "\n".join(
                sent["pali_sentence"]
                for para in paragraphs
                for sent in para.get("pending", [])
            )

            # ── 4. Glossary (Pāli → Sinhala, phrase-aware) ────────
            self.heartbeat()
            pali_ngrams    = prompt_lib.extract_pali_ngrams(pali_text_for_lookups)
            glossary_block = data.fetch_glossary_block(
                pali_ngrams, self.log_info, self.log_warn
            )
            self.heartbeat()

            # ── 5. Nissaya maps for each paragraph ─────────────────
            nissaya_blocks: dict[int, str] = {}
            for para in paragraphs:
                niss_map = data.fetch_nissaya_map(
                    self.params, para["book_id"], para["para_id"], self.log_info
                )
                nissaya_blocks[para["para_id"]] = data.build_nissaya_block(
                    para["sentences"], niss_map
                )
            self.heartbeat()

            # ── 6. Commentary via book_links ───────────────────────
            src_lines: list[tuple] = list(dict.fromkeys(
                (para["book_id"], para["para_id"], sent["line_id"])
                for para in paragraphs
                for sent in para.get("sentences", [])
                if sent.get("line_id") is not None
            ))
            self.heartbeat()
            commentary_block = data.fetch_commentary_block(
                self.params, src_lines,
                max_chars=100000,
                log_info=self.log_info,
                log_warn=self.log_warn,
            )
            self.heartbeat()

            # ── 7. Pāli word definitions ───────────────────────────
            pali_defs_block = data.fetch_pali_definitions_block(
                pali_text_for_lookups, self.params,
                self.log_info, self.log_warn,
            )
            self.heartbeat()

            # ── 8. Split paragraphs into token-safe chunks ─────────
            chunks = prompt_lib.chunk_paragraphs(paragraphs, max_tokens=max_tokens)
            self.log_info(
                f"  {len(paragraphs)} paragraph(s) → "
                f"{len(chunks)} chunk(s) (max_tokens={max_tokens})."
            )

            # ── 9. Per-chunk: build prompt → call AI → save ────────
            for c_idx, chunk in enumerate(chunks, 1):
                n_sentences = sum(len(p["pending"]) for p in chunk)
                self.log_info(
                    f"  Chunk {c_idx}/{len(chunks)}: "
                    f"{n_sentences} sentence(s) across {len(chunk)} para(s)."
                )
                self.heartbeat()

                prev_para_translation = data.fetch_previous_paragraph_translation(
                    self.params, book_id, chunk[0]["para_id"], self.log_info
                )
                self.heartbeat()

                user_prompt, flat_sentences = prompt_lib.build_prompt(
                    book_id         = book_id,
                    para_id         = para_id,
                    section_title   = section_title,
                    paragraphs      = chunk,
                    pali_block      = pali_block,
                    sinhala_block   = sinhala_block,
                    nissaya_blocks  = nissaya_blocks,
                    glossary_block  = glossary_block,
                    commentary_block = commentary_block,
                    pali_defs_block  = pali_defs_block,
                    prev_para_text   = prev_para_translation,
                    max_ref_chars    = max_ref_chars,
                )

                # Wrapper to inject heartbeats during AI wait
                def ask_ai_with_heartbeat(prompt: str, system: str = "") -> str:
                    """Call ask_ai in a separate thread and heartbeat while waiting."""
                    import threading
                    result = [None]
                    error = [None]
                    
                    def _ai_call():
                        try:
                            result[0] = self.ask_ai(prompt, system)
                        except Exception as e:
                            error[0] = e
                    
                    thread = threading.Thread(target=_ai_call, daemon=False)
                    thread.start()
                    
                    # Heartbeat every 10 seconds while thread is alive
                    while thread.is_alive():
                        thread.join(timeout=10)
                        if thread.is_alive():
                            self.heartbeat()
                    
                    if error[0]:
                        raise error[0]
                    return result[0]

                raw = prompt_lib.call_ai_with_logging(
                    ask_ai_fn  = ask_ai_with_heartbeat,
                    prompt     = user_prompt,
                    book_id    = book_id,
                    para_id    = para_id,
                    chunk_idx  = c_idx,
                    log_dir    = log_dir,
                    log_info   = self.log_info,
                    log_sucess = self.log_sucess,
                    log_error  = self.log_error,
                    heartbeat_fn = self.heartbeat,
                )
                self.heartbeat()
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

                # Group by para_id, infer missing para_id from flat_sentences
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
                        self.log_warn(
                            f"  Cannot determine para_id for entry: {t}"
                        )

                for chunk_para_id, para_translations in by_para.items():
                    chunk_book_id = next(
                        (p["book_id"] for p in chunk if p["para_id"] == chunk_para_id),
                        book_id,
                    )
                    # Attach pali_sentence so save_translations can store it
                    for t in para_translations:
                        if not t.get("pali_sentence"):
                            matched = [
                                s for s in flat_sentences
                                if s["line_id"] == t.get("line_id")
                                and s["para_id"] == chunk_para_id
                            ]
                            if matched:
                                t["pali_sentence"] = matched[0]["pali_sentence"]

                    updated = data.save_translations(
                        self.params, chunk_book_id, chunk_para_id,
                        para_translations,
                        self.log_info, self.log_warn, self.log_error,
                    )
                    total_updated += updated
                    self.heartbeat()

                if new_terms:
                    source_id = f"{book_id}:{para_id}"
                    inserted  = data.upsert_glossary(
                        new_terms, source_id,
                        self.log_info, self.log_warn, self.log_error,
                    )
                    total_glossary += inserted
                    self.heartbeat()

            self.log_info(
                f"  Heading done. "
                f"Running totals — sentences: {total_updated}, "
                f"glossary: {total_glossary}."
            )

        self.log_info("=" * 60)
        self.log_info(
            f"SinhalaTranslator — DONE. "
            f"Sentences saved: {total_updated}. "
            f"Glossary terms added: {total_glossary}."
        )
        self.log_info("=" * 60)
        return total_updated