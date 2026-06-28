"""
ai_jobs/prompt_composer.py  —  Shared prompt assembly, chunking, AI dispatch,
                                and response parsing for translation jobs (ST, SI, etc).

This class factors out common functionality:
  • IAST transliteration (Myanmar → Roman Pāli)
  • Nissaya JSON formatting
  • Glossary n-gram extraction
  • Paragraph chunking by token count
  • AI calls with retry + logging
  • Response parsing with truncation salvage

Language-specific prompt templates and system prompts are subclassed.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Callable

# ══════════════════════════════════════════════════════════════════════════════
# IAST transliteration helper
# ══════════════════════════════════════════════════════════════════════════════

_PALI_ALPHABET = set("aāiīuūeokKgGcCjJṭṭḍḍNnpPbBmMyrlLvśṣshḥṃṅñṇḷ")


def translit_to_roman(text: str) -> str:
    """
    Convert Myanmar (or any non-Roman) Pāli to IAST using aksharamukha.
    If aksharamukha is not installed, returns text unchanged.
    Falls back gracefully on any error.
    """
    if not text:
        return ""
    # Already Roman — leave alone
    if text[0] in _PALI_ALPHABET or text[0].isupper():
        return text
    try:
        from aksharamukha import transliterate
        result = transliterate.process(
            "autodetect", "IASTPali", text,
            post_options=["AnusvaratoNasalASTISO"]
        )
        if result:
            return (
                result
                .replace("ï", "i")
                .replace("ü", "u")
                .replace("ĕ", "e")
                .replace("ŏ", "o")
                .replace("l̤", "ḷ")
            )
    except Exception:
        pass
    return text


def format_nissaya_entry(entry: dict) -> str:
    """
    Turn one nissaya JSON object into  'pali_roman: meaning'
    or just the note text if it is a note-only entry.
    """
    if "note" in entry and "pali" not in entry:
        return f"[Note: {entry['note']}]"
    pali    = translit_to_roman(entry.get("pali", ""))
    meaning = entry.get("meaning", "")
    if pali and meaning:
        return f"{pali}: {meaning}"
    return pali or meaning


def format_nissaya_line(raw_nissaya: str) -> str:
    """
    Parse the nissaya content (stored as a JSON array of objects) and
    return a compact human-readable string:
      pali1: meaning1 | pali2: meaning2 | [Note: ...]
    If the content is not valid JSON, return it as-is.
    """
    if not raw_nissaya:
        return "(none)"
    raw_nissaya = raw_nissaya.strip()
    if not raw_nissaya.startswith("["):
        return raw_nissaya   # plain text — return unchanged
    try:
        entries = json.loads(raw_nissaya)
        parts   = [format_nissaya_entry(e) for e in entries if isinstance(e, dict)]
        return " | ".join(p for p in parts if p) or "(none)"
    except (json.JSONDecodeError, TypeError):
        return raw_nissaya


# ══════════════════════════════════════════════════════════════════════════════
# Glossary — phrase-aware n-gram extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_pali_ngrams(text: str, max_n: int = 5) -> list[str]:
    """
    Return all unique lowercased tokens AND n-grams (up to max_n words)
    from the Pāli text. This ensures multi-word glossary phrases are matched.
    """
    # Split on whitespace and punctuation, keep only non-trivial tokens
    tokens = [t.lower() for t in re.split(r"[\s,;.\u2018\u2019\"'()\[\]]+", text)
              if len(t) > 1]
    ngrams = set(tokens)
    for n in range(2, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[i:i + n]))
    return list(ngrams)


# ══════════════════════════════════════════════════════════════════════════════
# Token estimation (simple heuristic)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


# ══════════════════════════════════════════════════════════════════════════════
# Paragraph chunking by token count
# ══════════════════════════════════════════════════════════════════════════════

def chunk_paragraphs(
    paragraphs: list[dict],
    max_tokens: int = 3000,
) -> list[list[dict]]:
    """
    Split a list of paragraphs into chunks such that each chunk's token count
    stays under max_tokens. Keeps paragraphs intact (no splitting within a para).

    Returns a list of chunks, where each chunk is a list of paragraphs.
    """
    if not paragraphs:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        # Estimate tokens in this paragraph (simple: count Pali sentences)
        para_token_estimate = sum(
            len(sent.get("pali_sentence", "")) // 4
            for sent in para.get("sentences", [])
        )

        # If adding this para would exceed the limit, start a new chunk
        if current_chunk and current_tokens + para_token_estimate > max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(para)
        current_tokens += para_token_estimate

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Base PromptComposer class
# ══════════════════════════════════════════════════════════════════════════════

class PromptComposer(ABC):
    """
    Abstract base class for translation prompt composition.
    Subclasses (ST, SI, etc.) provide:
      - system_prompt() → str
      - user_template() → str (for formatting user prompts)
      - call_ai_with_logging() — customizable AI dispatch logic
    """

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this translator."""
        pass

    @property
    @abstractmethod
    def user_template(self) -> str:
        """Return the user prompt template with {placeholder} keys."""
        pass

    @abstractmethod
    def build_user_prompt(self, **kwargs) -> tuple[str, list[dict]]:
        """
        Build the user prompt and return (prompt_str, flat_sentence_list).
        Implementation depends on the language/source (ST vs SI, etc).
        """
        pass

    # ── Shared AI call with logging and retry logic ────────────────────────

    def call_ai_with_logging(
        self,
        ask_ai_fn:  Callable[[str, str], str],
        prompt:     str,
        identifier: str,  # sc_id, book_id.para_id, etc.
        chunk_idx:  int,
        log_dir:    str,
        log_info:   Callable[[str], None],
        log_sucess: Callable[[str], None],
        log_error:  Callable[[str], None],
        max_chunk_retries: int = 4,
        base_wait_sec:     int = 60,
    ) -> str | None:
        """
        Call the AI, write prompt+response to log_dir, return raw response.

        Retry policy (per-chunk):
          - 503 / quota / rate-limit errors: exponential backoff
          - TimeoutError: returns None so the caller can skip this chunk.
          - Other exceptions: re-raised immediately.
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        safe_id    = re.sub(r"[^\w\-]", "_", identifier)
        base_name  = f"{timestamp}_{safe_id}_chunk{chunk_idx:02d}"

        prompt_path = os.path.join(log_dir, f"{base_name}_prompt.txt")
        try:
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("=== SYSTEM ===\n")
                f.write(self.system_prompt)
                f.write("\n\n=== USER ===\n")
                f.write(prompt)
            log_info(f"[AI] Prompt → {prompt_path}")
        except OSError as exc:
            log_error(f"[AI] Could not write prompt log: {exc}")

        n_tokens = estimate_tokens(prompt)
        log_info(
            f"[AI] Calling AI: id={identifier!r}, chunk={chunk_idx}, "
            f"{len(prompt)} chars (~{n_tokens} tokens)."
        )

        raw = None
        attempt = 0
        wait_sec = base_wait_sec

        while attempt <= max_chunk_retries:
            try:
                raw = ask_ai_fn(prompt, self.system_prompt)
                break   # success — exit retry loop
            except TimeoutError as exc:
                log_error(f"[AI] Timeout: id={identifier!r} chunk={chunk_idx}: {exc}")
                return None
            except Exception as exc:
                if self._is_retryable_error(exc) and attempt < max_chunk_retries:
                    attempt += 1
                    log_error(
                        f"[AI] Retryable error (attempt {attempt}/{max_chunk_retries}): {exc}. "
                        f"Waiting {wait_sec}s before retry…"
                    )
                    time.sleep(wait_sec)
                    wait_sec *= 2
                else:
                    raise

        if raw is None:
            return None

        log_sucess(f"[AI] Response: {len(raw)} chars.")

        response_path = os.path.join(log_dir, f"{base_name}_response.txt")
        try:
            with open(response_path, "w", encoding="utf-8") as f:
                f.write(raw)
            log_info(f"[AI] Response → {response_path}")
        except OSError as exc:
            log_error(f"[AI] Could not write response log: {exc}")

        return raw

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """Check if error is retryable (503, quota, rate limit, etc)."""
        patterns = (
            "503", "unavailable", "resource_exhausted", "quota", "rate",
            "limit", "429", "exhausted", "overloaded", "service unavailable",
            "too many requests",
        )
        msg = str(exc).lower()
        return any(p in msg for p in patterns)

    # ── Shared response parsing ────────────────────────────────────────────

    @staticmethod
    def parse_response(raw: str) -> dict:
        """
        Extract {"translations": [...], "glossary": [...]} from AI output.

        Handles:
          - Markdown code fences
          - Truncated arrays (salvages all complete objects)
          - Missing "glossary" key (returns empty list)
        """
        cleaned = raw.strip()
        cleaned = re.sub(r"^\s*```[a-zA-Z]*\s*\n?", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\n?\s*```\s*$",           "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        # ── Happy path: full valid JSON ───────────────────────────────
        start = cleaned.find("{")
        end   = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(cleaned[start:end + 1])
                obj.setdefault("translations", [])
                obj.setdefault("glossary",     [])
                return obj
            except json.JSONDecodeError:
                pass

        # ── Salvage path: extract translations array even if truncated ─
        obj = {"translations": [], "glossary": []}

        for key in ("translations", "glossary"):
            # Find the opening bracket of this key's array
            m = re.search(rf'"{key}"\s*:\s*\[', cleaned)
            if not m:
                continue
            array_start = m.end() - 1   # position of '['

            # Collect all complete {...} objects inside the array
            depth     = 0
            obj_start = None
            items     = []

            for i, ch in enumerate(cleaned[array_start:], start=array_start):
                if ch == "{":
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        try:
                            items.append(json.loads(cleaned[obj_start:i + 1]))
                        except json.JSONDecodeError:
                            pass
                        obj_start = None
                elif ch == "]" and depth == 0:
                    break   # end of array

            obj[key] = items

        return obj