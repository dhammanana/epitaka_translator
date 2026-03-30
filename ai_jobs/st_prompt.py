"""
jobs/st_prompt.py  —  Prompt assembly, chunking, AI dispatch, and response
                       parsing for SentenceTranslatorJob.

Changes vs previous version
----------------------------
1. SC reference now shows Pāli + English interleaved line-by-line so the AI
   can match them.
2. Nissaya JSON is transliterated from Myanmar script to IAST Roman using
   aksharamukha, then formatted as  pali: meaning | pali2: meaning2 | …
3. Glossary lookup is phrase-aware: in addition to single-token matching it
   also checks every n-gram (up to 5 words) in the Pāli text against the
   glossary so multi-word phrases are found.
4. Gemini thinking budget is capped (via generation_config) and
   max_output_tokens raised to avoid truncation.
5. chunk_paragraphs now limits by token count only (removed sentence limit),
   and limits at paragraph level for consistency.
6. parse_response salvages truncated arrays properly.
7. NEW: Commentary (aṭṭhakathā + tīka) from book_links.
8. NEW: Pali word definitions from pali_definition table.
9. NEW: Previous paragraph translation for style consistency.
"""

import json
import os
import re
import time
from typing import Callable

# ══════════════════════════════════════════════════════════════════
# IAST transliteration helper
# ══════════════════════════════════════════════════════════════════

_PALI_ALPHABET = set("aāiīuūeokKgGcCjJṭṭḍḍNnpPbBmMyrlLvśṣshḥṃṅñṇḷ")

def _translit_to_roman(text: str) -> str:
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


def _format_nissaya_entry(entry: dict) -> str:
    """
    Turn one nissaya JSON object into  'pali_roman: meaning'
    or just the note text if it is a note-only entry.
    """
    if "note" in entry and "pali" not in entry:
        return f"[Note: {entry['note']}]"
    pali    = _translit_to_roman(entry.get("pali", ""))
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
        parts   = [_format_nissaya_entry(e) for e in entries if isinstance(e, dict)]
        return " | ".join(p for p in parts if p) or "(none)"
    except (json.JSONDecodeError, TypeError):
        return raw_nissaya

from itertools import zip_longest

def build_sc_reference_block(pali_text: str, en_text: str, max_chars: int) -> str:
    """
    Groups the entire Pāli text followed by the entire English text.
    English is prioritized and never truncated. 
    Pāli is truncated to fit the remaining budget.
    """
    unlimited = (max_chars == -1)
    
    p_prefix = "[Pāli] "
    e_prefix = "[English] "
    
    # 1. Clean inputs
    p_raw = pali_text.strip().replace('ṁ', 'ṃ')
    e_raw = en_text.strip()

    # 2. Prepare English (The "Sacred" priority text)
    en_content = f"{e_prefix}{e_raw}"
    
    # If not unlimited, check if English alone exceeds the budget
    if not unlimited and len(en_content) >= max_chars:
        # If English is already too big, we show as much of English as possible 
        # (though your rule says never truncate English, we must respect the physical limit)
        # and provide a minimal truncated marker for Pāli.
        p_content = f"{p_prefix}... [truncated]"
        return f"{p_content}\n\n{en_content}"

    # 3. Calculate remaining budget for Pāli
    # Budget = Total limit - English length - 2 (for the \n\n separator)
    if unlimited:
        p_content = f"{p_prefix}{p_raw}"
    else:
        available_for_pali = max_chars - len(en_content) - 2
        
        full_pali = f"{p_prefix}{p_raw}"
        
        if available_for_pali <= len(p_prefix):
            # Not even enough room for the Pāli header
            p_content = f"{p_prefix}... [truncated]"
        elif len(full_pali) > available_for_pali:
            # Truncate Pāli to fit the remaining space
            # Subtracting 15 to ensure room for the " … [truncated]" string
            trunc_point = max(len(p_prefix), available_for_pali - 15)
            p_content = full_pali[:trunc_point].strip() + " … [truncated]"
        else:
            p_content = full_pali

    # 4. Join the two blocks
    return f"{p_content}\n\n{en_content}"

# ══════════════════════════════════════════════════════════════════
# Glossary — phrase-aware lookup
# ══════════════════════════════════════════════════════════════════

def extract_pali_ngrams(text: str, max_n: int = 5) -> list[str]:
    """
    Return all unique lowercased tokens AND n-grams (up to max_n words)
    from the Pāli text.  This ensures multi-word glossary phrases are matched.
    """
    # Split on whitespace and punctuation, keep only non-trivial tokens
    tokens = [t.lower() for t in re.split(r"[\s,;.\u2018\u2019\"'()\[\]]+", text)
              if len(t) > 1]
    ngrams = set(tokens)
    for n in range(2, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[i:i + n]))
    return list(ngrams)


# ══════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a scholar-translator specialising in Pāli Buddhist texts.
You will be given:
  1. SENTENCES TO TRANSLATE — a JSON array of Pāli sentences.
  2. SC REFERENCE — the authoritative Pāli + English parallel text for context.
  3. MYANMAR NISSAYA — word-by-word gloss for each sentence (already romanised).
  4. ESTABLISHED GLOSSARY — fixed translations for specific Pāli terms/phrases.
  5. PALI COMMENTARY & SUB-COMMENTARY — Aṭṭhakathā and Tīka texts (when available).
  6. PALI WORD DEFINITIONS — Example sentences for difficult/rare words.
  7. PREVIOUS PARAGRAPH — the translation of the immediately preceding paragraph (for style consistency).

Your task is to return a single JSON object with exactly two keys:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A. "translations"  — array, one entry per input sentence (same order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  { "para_id": <int>, "line_id": <int>, "english_translation": "<text>" }

Rules:
  • Follow the SC English closely in meaning and register; adapt to sentence level.
  • If available, use the Pāli commentary as the primary reference for meaning.
  • Use the nissaya glossary to resolve ambiguous compounds or unusual forms.
  • Apply every glossary term exactly — including multi-word phrases.
  • Reference the previous paragraph for consistent style and terminology.
  • Keep untranslated Pāli (with diacritics) only when no English equivalent exists;
    gloss in parentheses on first occurrence only.
  • No verse numbers, footnotes, or meta-commentary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B. "glossary"  — NEW terms only (not already in the supplied glossary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  { "pali": "…", "english": "…", "domain": "…",
    "sub_domain": "…", "context": "…", "note": "…" }
domain should be one of these: sutta, vinaya, abhidhamma, grammar, story.
Include technical terms, formulaic phrases, and compound terms you had to decide.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — critical
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown fences, no prose outside the JSON.
{ "translations": [...], "glossary": [...] }
"""

USER_TEMPLATE = """Sutta: {sc_id} — {sutta_name}

══════════════════════════════
ESTABLISHED GLOSSARY (apply exactly, including multi-word phrases)
══════════════════════════════
{glossary_block}

══════════════════════════════
SC REFERENCE — Pāli + English parallel text
══════════════════════════════
{sc_reference_block}

══════════════════════════════
PALI COMMENTARY & SUB-COMMENTARY
══════════════════════════════
{commentary_block}

══════════════════════════════
PALI WORD DEFINITIONS (reference for difficult/rare terms)
══════════════════════════════
{pali_defs_block}

══════════════════════════════
PREVIOUS PARAGRAPH (for style consistency)
══════════════════════════════
{prev_para_block}

══════════════════════════════
MYANMAR NISSAYA (word-by-word gloss, romanised)
══════════════════════════════
{nissaya_block}

══════════════════════════════
SENTENCES TO TRANSLATE (JSON array)
══════════════════════════════
{sentences_json}
"""


# ══════════════════════════════════════════════════════════════════
# Chunking — token-safe, paragraph-aware
# ══════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
    return len(text) // 4


def chunk_paragraphs(
    paragraphs: list[dict],
    max_tokens: int = 2000,
) -> list[list[dict]]:
    """
    Group paragraphs into chunks such that each chunk stays under max_tokens.
    Each chunk is a list of paragraph dicts.
    Maintains paragraph-level boundaries (never splits within a paragraph).
    """
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        # Estimate tokens for this paragraph (including all sentences)
        para_text = "\n".join(s.get("pali_sentence", "") for s in para.get("pending", []))
        para_tokens = estimate_tokens(para_text)

        # If this single paragraph exceeds max_tokens, still include it
        # (avoid chunks with zero paragraphs)
        if current_chunk and (current_tokens + para_tokens > max_tokens):
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(para)
        current_tokens += para_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ══════════════════════════════════════════════════════════════════
# Nissaya block builder  (transliterated, compact format)
# ══════════════════════════════════════════════════════════════════

def build_nissaya_block_for_prompt(
    paragraphs:     list[dict],   # chunk paragraphs, each has "sentences"
    nissaya_blocks: dict[int, str],  # para_id → raw nissaya block string from st_data
) -> str:
    """
    Rebuild the nissaya section for the prompt using the compact format:
      [para_id=X / line_id=Y] pali_sentence
        Nissaya: pali1: meaning1 | pali2: meaning2 | …

    The raw nissaya block stored in nissaya_blocks was built by
    st_data.build_nissaya_block() which already pairs line_id with content.
    We re-parse it here to apply transliteration and compact formatting.
    """
    parts = []
    for para in paragraphs:
        pid   = para["para_id"]
        raw   = nissaya_blocks.get(pid, "")
        if not raw or raw.startswith("(no nissaya"):
            parts.append(f"--- para_id={pid} ---\n(no nissaya)")
            continue

        # raw is already split into blocks like:
        #   [line_id=1] pali_sentence\n  Nissaya: <content>
        #   [line_id=2] …
        section_lines = []
        for sent in para["sentences"]:
            lid        = sent["line_id"]
            pali_sent  = sent.get("pali_sentence", "")
            # Extract the nissaya content for this line_id from the raw block
            m = re.search(
                rf"\[line_id={lid}\][^\n]*\n\s*Nissaya:\s*(.*?)(?=\n\[line_id=|\Z)",
                raw, re.DOTALL
            )
            niss_raw = m.group(1).strip() if m else ""
            niss_fmt = format_nissaya_line(niss_raw)
            section_lines.append(
                f"  [line_id={lid}] {pali_sent}\n"
                f"    Nissaya: {niss_fmt}"
            )
        parts.append(f"--- para_id={pid} ---\n" + "\n".join(section_lines))

    return "\n\n".join(parts) or "(no nissaya available)"


# ══════════════════════════════════════════════════════════════════
# Prompt builder
# ══════════════════════════════════════════════════════════════════

def build_prompt(
    sc_id:            str,
    sutta_name:       str,
    paragraphs:       list[dict],
    pali_text:        str,
    en_text:          str,
    nissaya_blocks:   dict[int, str],
    glossary_block:   str,
    commentary_block: str = "(no commentary available)",
    pali_defs_block:  str = "(no word definitions available)",
    prev_para_text:   str = "",
    max_ref_chars:    int = -1,
) -> tuple[str, list[dict]]:
    """
    Build the user prompt for one chunk of paragraphs.

    Returns (prompt_str, flat_sentence_list).
    flat_sentence_list contains all pending sentences tagged with para_id.
    """
    # Flat list of sentences to translate
    flat_sentences = []
    for para in paragraphs:
        for s in para["pending"]:
            flat_sentences.append({
                "para_id":       para["para_id"],
                "line_id":       s["line_id"],
                "pali_sentence": s["pali_sentence"],
            })

    sc_reference_block = build_sc_reference_block(pali_text, en_text, max_ref_chars)
    nissaya_block      = build_nissaya_block_for_prompt(paragraphs, nissaya_blocks)
    
    # Format previous paragraph block
    prev_para_block = prev_para_text or "(no previous paragraph)"

    prompt = USER_TEMPLATE.format(
        sc_id              = sc_id,
        sutta_name         = sutta_name,
        glossary_block     = glossary_block,
        sc_reference_block = sc_reference_block,
        commentary_block   = commentary_block,
        pali_defs_block    = pali_defs_block,
        prev_para_block    = prev_para_block,
        nissaya_block      = nissaya_block,
        sentences_json     = json.dumps(flat_sentences, ensure_ascii=False, indent=2),
    )
    return prompt, flat_sentences


# ══════════════════════════════════════════════════════════════════
# AI call  — with debug log dump
# ══════════════════════════════════════════════════════════════════

_RETRYABLE_PATTERNS = (
    "503", "unavailable", "resource_exhausted", "quota", "rate",
    "limit", "429", "exhausted", "overloaded", "service unavailable",
    "too many requests",
)

def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _RETRYABLE_PATTERNS)


def call_ai_with_logging(
    ask_ai_fn:  Callable[[str, str], str],
    prompt:     str,
    sc_id:      str,
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
        waits: 60s, 120s, 240s, 480s  (doubles each attempt)
      - TimeoutError: returns None so the caller can skip this chunk.
      - Other exceptions: re-raised immediately.

    max_chunk_retries : how many extra attempts after the first failure
    base_wait_sec     : initial wait in seconds (doubles each retry)
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    safe_sc_id = re.sub(r"[^\w\-]", "_", sc_id)
    base_name  = f"{timestamp}_{safe_sc_id}_chunk{chunk_idx:02d}"

    prompt_path = os.path.join(log_dir, f"{base_name}_prompt.txt")
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("=== SYSTEM ===\n")
            f.write(SYSTEM_PROMPT)
            f.write("\n\n=== USER ===\n")
            f.write(prompt)
        log_info(f"[AI] Prompt → {prompt_path}")
    except OSError as exc:
        log_error(f"[AI] Could not write prompt log: {exc}")

    n_tokens = estimate_tokens(prompt)
    log_info(
        f"[AI] Calling AI: sc_id={sc_id!r}, chunk={chunk_idx}, "
        f"{len(prompt)} chars (~{n_tokens} tokens)."
    )

    raw = None
    attempt = 0
    wait_sec = base_wait_sec

    while attempt <= max_chunk_retries:
        try:
            raw = ask_ai_fn(prompt, SYSTEM_PROMPT)
            break   # success — exit retry loop
        except TimeoutError as exc:
            log_error(f"[AI] Timeout: sc_id={sc_id!r} chunk={chunk_idx}: {exc}")
            return None
        except Exception as exc:
            if _is_retryable_error(exc) and attempt < max_chunk_retries:
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


# ══════════════════════════════════════════════════════════════════
# Response parser  — with truncation salvage
# ══════════════════════════════════════════════════════════════════

def parse_response(raw: str) -> dict:
    """
    Extract {"translations": [...], "glossary": [...]} from AI output.

    Handles:
      - Markdown code fences
      - Truncated "translations" array (salvages all complete objects)
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