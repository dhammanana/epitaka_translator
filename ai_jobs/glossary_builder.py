"""
jobs/glossary_builder.py — Build a Pāli/English glossary from the en_translation table.

Params (job_params JSON)
------------------------
  book_id      : str  — filter by book (optional, e.g. "DN", "MN")
  sc_id        : str  — process a single sutta (optional)
  batch_size   : int  — how many suttas per run (default 10)
  max_terms    : int  — soft cap on new terms per sutta (default 30)

Algorithm
---------
1. Load a batch of suttas not yet fully processed.
2. For each sutta:
   a. Fetch existing glossary terms → send to AI so it skips duplicates.
   b. Build a prompt with the Pāli + English parallel text.
   c. Ask AI to return JSON list of new glossary entries.
   d. Parse response, validate, upsert into glossary.db.
   e. Log progress.
"""

import json
import re
import logging
import sqlite3
from typing import Any

from ai_jobs.base_job import BaseJob
from database import get_glossary_conn, get_runner_conn
from config import SOURCE_DB

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Pāli language expert and scholar of Theravāda Buddhism.
Your task is to extract a glossary of Pāli terms and phrases from the provided sutta text.

Return ONLY a valid JSON array. No markdown, no explanation, no code blocks.
Each element must have these fields (all strings, empty string if unknown):
  pali       — the Pāli word or phrase (IAST diacritics)
  english    — the English translation or gloss used in THIS text
  domain     — broad category (e.g. "meditation", "ethics", "cosmology", "psychology", "monastic", "general")
  sub_domain — finer category (e.g. "jhāna", "precepts", "deva realms")
  context    — brief note on how the term is used in this sutta (1–2 sentences)
  note       — any additional scholarly note (empty string if none)

Rules:
- Include technical Pāli terms, compound terms, and significant phrases.
- DO NOT include terms already in the existing glossary (provided below).
- DO NOT include plain common words with obvious translations.
- Prefer terms that are doctrinally significant or commonly untranslated.
- Maximum {max_terms} new terms.
"""

USER_TEMPLATE = """
Sutta: {sc_id} — {sutta_name}

=== EXISTING GLOSSARY TERMS (do NOT repeat these) ===
{existing_terms}

=== PĀLI TEXT ===
{pali_text}

=== ENGLISH TRANSLATION ===
{en_text}

Extract new glossary terms as a JSON array:
"""


class GlossaryBuilderJob(BaseJob):
    display_name = "Glossary Builder"
    param_schema = {
        "book_id":    {"type": "string",  "label": "Book ID (e.g. MN)",                    "default": ""},
        "sc_id":      {"type": "string",  "label": "Single SC-ID (optional)",               "default": ""},
        "batch_size": {"type": "integer", "label": "Suttas per run",                        "default": 10},
        "max_terms":  {"type": "integer", "label": "Max new terms per sutta",               "default": 15},
        "max_chars":  {"type": "integer", "label": "Max chars of text per side",            "default": 3000},
        "key_ids":    {"type": "string",  "label": "API key IDs to use (comma-sep, or blank for all)", "default": ""},
    }

    # ── Helpers ──────────────────────────────────────────────────

    def _get_source_conn(self) -> sqlite3.Connection:
        source_db = self.params.get("source_db") or SOURCE_DB
        if not source_db:
            raise RuntimeError(
                "source_db path is not configured. "
                "Set it in Settings → source_db or pass it as a param."
            )
        conn = sqlite3.connect(source_db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _fetch_suttas(self, src_conn) -> list:
        book_id    = self.params.get("book_id", "").strip()
        sc_id_filter = self.params.get("sc_id", "").strip()
        batch_size = int(self.params.get("batch_size", 10))

        if sc_id_filter:
            rows = src_conn.execute(
                "SELECT * FROM en_translation WHERE sc_id=?", (sc_id_filter,)
            ).fetchall()
        elif book_id:
            rows = src_conn.execute(
                "SELECT * FROM en_translation WHERE book_id=? AND palitext!='' AND entext!='' LIMIT ?",
                (book_id, batch_size)
            ).fetchall()
        else:
            rows = src_conn.execute(
                "SELECT * FROM en_translation WHERE palitext!='' AND entext!='' LIMIT ?",
                (batch_size,)
            ).fetchall()

        return [dict(r) for r in rows]

    def _get_existing_terms(self, pali_words: list[str]) -> list[str]:
        """
        Return existing Pāli glossary entries that overlap with words in this sutta.
        We do a broad match to avoid the AI re-generating them.
        """
        if not pali_words:
            return []
        g_conn = get_glossary_conn()
        placeholders = ",".join("?" * len(pali_words))
        rows = g_conn.execute(
            f"SELECT pali, english FROM glossary WHERE pali IN ({placeholders})",
            pali_words
        ).fetchall()
        g_conn.close()
        return [f"{r['pali']} = {r['english']}" for r in rows]

    @staticmethod
    def _extract_pali_words(text: str) -> list[str]:
        """Very rough tokenisation of Pāli text for pre-existing-term lookup."""
        # Normalise whitespace; split on spaces/punctuation
        tokens = re.split(r"[\s,;.\u2018\u2019\"'()\[\]]+", text)
        # Keep tokens that look like Pāli (non-trivial length, may have diacritics)
        return list({t.lower() for t in tokens if 2 < len(t) < 40})

    @staticmethod
    def _parse_ai_response(raw: str) -> list[dict]:
        """
        Extract JSON array from AI response, tolerating:
        - markdown fences (with or without leading spaces)
        - truncated responses (salvage complete objects from a cut-off array)
        """
        raw = raw.strip()

        # Strip markdown code fences — allow optional leading whitespace
        raw = re.sub(r"^\s*```[a-z]*\s*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?\s*```\s*$",        "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        start = raw.find("[")
        if start == -1:
            raise ValueError(f"No JSON array found. First 200 chars: {raw[:200]}")

        end = raw.rfind("]")

        # ── Happy path: complete array ────────────────────────────
        if end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass   # fall through to salvage attempt

        # ── Salvage path: response was truncated mid-array ────────
        # Extract every complete {...} object we can find before the cut-off.
        salvaged = []
        depth    = 0
        obj_start = None

        for i, ch in enumerate(raw[start:], start=start):
            if ch == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        obj = json.loads(raw[obj_start:i + 1])
                        salvaged.append(obj)
                    except json.JSONDecodeError:
                        pass
                    obj_start = None

        if salvaged:
            return salvaged

        raise ValueError(
            f"Could not parse JSON array (response may be truncated). "
            f"First 300 chars: {raw[start:start+300]}"
        )

    def _upsert_terms(self, terms: list[dict], sc_id: str) -> int:
        """Upsert glossary terms; return count of newly inserted rows."""
        required_fields = {"pali", "english"}
        inserted = 0
        g_conn = get_glossary_conn()

        for term in terms:
            if not required_fields.issubset(term):
                self.log_warn(f"Skipping malformed term (missing pali/english): {term}")
                continue
            pali    = str(term.get("pali", "")).strip()
            english = str(term.get("english", "")).strip()
            if not pali or not english:
                continue

            try:
                g_conn.execute(
                    """INSERT INTO glossary(pali, english, domain, sub_domain, context, note, source_id)
                       VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(pali, english) DO NOTHING""",
                    (
                        pali,
                        english,
                        str(term.get("domain",     "") or ""),
                        str(term.get("sub_domain", "") or ""),
                        str(term.get("context",    "") or ""),
                        str(term.get("note",       "") or ""),
                        sc_id,
                    )
                )
                inserted += g_conn.execute("SELECT changes()").fetchone()[0]
            except Exception as exc:
                self.log_error(f"DB error inserting term '{pali}': {exc}")

        g_conn.commit()
        g_conn.close()
        return inserted

    # ── Main entry point ─────────────────────────────────────────

    def run(self) -> Any:
        max_terms = int(self.params.get("max_terms", 15))
        max_chars = int(self.params.get("max_chars", 3000))
        self.log_info("GlossaryBuilder starting.")
        self.log_info(f"Params: {self.params}")

        # 1. Connect to source DB
        try:
            src_conn = self._get_source_conn()
        except Exception as exc:
            self.log_error(f"Cannot open source DB: {exc}")
            raise

        # 2. Fetch suttas to process
        suttas = self._fetch_suttas(src_conn)
        src_conn.close()

        if not suttas:
            self.log_info("No suttas found matching the given parameters. Nothing to do.")
            return

        self.log_info(f"Processing {len(suttas)} sutta(s).")
        total_inserted = 0

        for idx, sutta in enumerate(suttas, 1):
            sc_id      = sutta["sc_id"]
            sutta_name = sutta.get("sutta_name") or sc_id
            pali_text  = sutta.get("palitext", "") or ""
            en_text    = sutta.get("entext",   "") or ""

            self.log_info(
                f"[{idx}/{len(suttas)}] Processing: {sc_id} — {sutta_name}"
            )
            # Keep watchdog happy between potentially long AI calls
            self.heartbeat()

            if not pali_text.strip() or not en_text.strip():
                self.log_warn(f"  Skipping {sc_id}: missing Pāli or English text.")
                continue

            # 3. Build existing-terms list for de-dup
            pali_words     = self._extract_pali_words(pali_text)
            existing_terms = self._get_existing_terms(pali_words)
            existing_str   = "\n".join(existing_terms) if existing_terms else "(none yet)"

            self.log_debug(
                f"  Existing matching terms: {len(existing_terms)} — "
                f"sending as context to AI."
            )

            # 4. Build prompts
            system = SYSTEM_PROMPT.format(max_terms=max_terms)
            user   = USER_TEMPLATE.format(
                sc_id      = sc_id,
                sutta_name = sutta_name,
                existing_terms = existing_str,
                pali_text  = pali_text[:max_chars],
                en_text    = en_text[:max_chars],
            )

            # 5. Call AI
            try:
                raw_response = self.ask_ai(user, system=system)
            except TimeoutError as exc:
                self.log_error(f"  AI timed out for {sc_id}: {exc}")
                continue   # move on to next sutta
            except Exception as exc:
                # Let caller (worker) decide if it's a rate-limit
                raise

            # 6. Parse & store
            try:
                terms = self._parse_ai_response(raw_response)
                self.log_info(f"  AI returned {len(terms)} term(s).")
            except (ValueError, json.JSONDecodeError) as exc:
                self.log_error(
                    f"  Failed to parse AI response for {sc_id}: {exc} "
                    f"(response was {len(raw_response)} chars)"
                )
                self.log_debug(f"  Raw response tail (last 300 chars): {raw_response[-300:]}")
                continue

            inserted = self._upsert_terms(terms, sc_id)
            total_inserted += inserted
            self.log_info(
                f"  Saved {inserted} new term(s) to glossary "
                f"(total this run: {total_inserted})."
            )

        self.log_info(
            f"GlossaryBuilder complete. "
            f"Total new glossary entries this run: {total_inserted}."
        )
        return total_inserted