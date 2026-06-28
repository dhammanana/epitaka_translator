"""
ai_jobs/sinhala_job/si_prompt.py  —  Prompt assembly, AI dispatch, chunking,
                                      and response parsing for
                                      SinhalaTranslatorJob.

All prompts are written in English so they are easy to read and edit.

The AI is asked to produce Sinhala translations of individual Pāli sentences,
guided by:
  1. The authentic Sinhala translation published in the Sinhala Tipiṭaka
     (ePiṭaka project) for context — this is the main reference.
  2. The Nissaya word-by-word gloss for each sentence.
  3. The established Sinhala glossary (Pāli → Sinhala term mappings).
  4. Pāli word definitions from the nisssaya dictionary.
  5. The previous paragraph's translation for style consistency.

Output format: a single JSON object with keys "translations" and "glossary".
"""

import json
import os
import re
import time
from typing import Callable

# ══════════════════════════════════════════════════════════════════════════════
# Nissaya formatter  (identical logic to st_prompt.py — copied to keep modules
# independent; the Sinhala job lives in its own folder)
# ══════════════════════════════════════════════════════════════════════════════

_PALI_ALPHABET = set("aāiīuūeokKgGcCjJṭṭḍḍNnpPbBmMyrlLvśṣshḥṃṅñṇḷ")


def _translit_to_roman(text: str) -> str:
    """Convert Myanmar (or other non-Roman) Pāli to IAST using aksharamukha."""
    if not text:
        return ""
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
    if "note" in entry and "pali" not in entry:
        return f"[Note: {entry['note']}]"
    pali    = _translit_to_roman(entry.get("pali", ""))
    meaning = entry.get("meaning", "")
    if pali and meaning:
        return f"{pali}: {meaning}"
    return pali or meaning


def format_nissaya_line(raw_nissaya: str) -> str:
    """
    Parse the nissaya JSON array and return a compact string:
      pali1: meaning1 | pali2: meaning2 | [Note: …]
    Falls back to the raw string if not valid JSON.
    """
    if not raw_nissaya:
        return "(none)"
    raw_nissaya = raw_nissaya.strip()
    if not raw_nissaya.startswith("["):
        return raw_nissaya
    try:
        entries = json.loads(raw_nissaya)
        parts   = [_format_nissaya_entry(e) for e in entries if isinstance(e, dict)]
        return " | ".join(p for p in parts if p) or "(none)"
    except (json.JSONDecodeError, TypeError):
        return raw_nissaya


# ══════════════════════════════════════════════════════════════════════════════
# Glossary — phrase-aware n-gram extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_pali_ngrams(text: str, max_n: int = 5) -> list[str]:
    """
    Return all unique lowercased tokens AND n-grams (up to max_n words)
    from the Pāli text so that multi-word glossary phrases are matched.
    """
    tokens = [
        t.lower()
        for t in re.split(r"[\s,;.\u2018\u2019\"'()\[\]]+", text)
        if len(t) > 1
    ]
    ngrams = set(tokens)
    for n in range(2, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[i : i + n]))
    return list(ngrams)


# ══════════════════════════════════════════════════════════════════════════════
# Token estimation
# ══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
    return len(text) // 4


# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a scholar specialising in Pāli Buddhist texts and the classical Sinhala translation tradition.

You will receive:
  1. SENTENCES TO TRANSLATE  — a JSON array of Pāli sentences that need Sinhala translations.
  2. SINHALA REFERENCE       — the authentic published Sinhala Tipiṭaka text (ePiṭaka) for the same section,
                               together with its Pāli source. Use this as the primary guide for meaning,
                               style, and Sinhala terminology.
  3. NISSAYA GLOSS           — a word-by-word Pāli-to-vernacular gloss for each sentence (already romanised).
                               Use this to resolve individual words and compounds.
  4. ESTABLISHED GLOSSARY    — fixed Pāli → Sinhala translations for specific terms.
                               Apply these exactly whenever the Pāli term appears.
  5. PĀLI COMMENTARY & SUB-COMMENTARY — Aṭṭhakathā and Tīkā texts linked to the sentences being
                               translated (when available). Use these to understand doctrinal meaning.
  6. PĀLI WORD DEFINITIONS   — example sentences for rare or difficult Pāli words.
  7. PREVIOUS PARAGRAPH      — the Sinhala translation of the immediately preceding paragraph,
                               for style and terminology consistency.

Your task: return a single JSON object with exactly two top-level keys.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A. "translations"  — one entry per input sentence, in the same order
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    "para_id":             <int>,
    "line_id":             <int>,
    "pali_sentence":       "<original Pāli>",
    "sinhala_translation": "<Sinhala translation>"
  }

Rules for translation:
  • If available, use the Pāli commentary (aṭṭhakathā / tīkā) as the primary reference for meaning.
  • Match the style and register of the published Sinhala Tipiṭaka reference as closely as possible.
  • Use the Nissaya gloss to resolve ambiguous Pāli compounds or inflected forms.
  • Apply every term from the established glossary — including multi-word Pāli phrases.
  • Maintain the same style and terminology as the previous paragraph where possible.
  • Keep untranslated Pāli (with diacritics) only when no standard Sinhala equivalent exists;
    gloss in parentheses on first occurrence only.
  • Do not add verse numbers, footnotes, or editorial comments.
  • para_id and line_id must be copied exactly from the input — do not change them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B. "glossary"  — NEW Pāli → Sinhala terms discovered during translation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    "pali":       "…",
    "sinhala":    "…",
    "domain":     "…",    (one of: sutta, vinaya, abhidhamma, grammar, story)
    "sub_domain": "…",
    "context":    "…",    (1–2 sentences on how the term is used)
    "note":       "…"     (optional scholarly note, or empty string)
  }

Include: technical terms, formulaic phrases, and compounds you had to decide.
Do NOT repeat terms already in the supplied established glossary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — critical
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.
{ "translations": [...], "glossary": [...] }
"""


USER_TEMPLATE = """Section: {book_id} — paragraph {para_id}
Title: {section_title}

══════════════════════════════
ESTABLISHED GLOSSARY  (apply exactly, including multi-word phrases)
══════════════════════════════
{glossary_block}

══════════════════════════════
SINHALA REFERENCE  — authentic published text for this section
(Pāli original followed by the Sinhala Tipiṭaka translation)
══════════════════════════════
{sinhala_reference_block}

══════════════════════════════
PĀLI COMMENTARY & SUB-COMMENTARY  (aṭṭhakathā / tīkā)
══════════════════════════════
{commentary_block}

══════════════════════════════
PĀLI WORD DEFINITIONS  (reference for difficult / rare terms)
══════════════════════════════
{pali_defs_block}

══════════════════════════════
PREVIOUS PARAGRAPH  (for style consistency)
══════════════════════════════
{prev_para_block}

══════════════════════════════
NISSAYA GLOSS  (word-by-word, romanised)
══════════════════════════════
{nissaya_block}

══════════════════════════════
SENTENCES TO TRANSLATE  (JSON array)
══════════════════════════════
{sentences_json}
"""


# ══════════════════════════════════════════════════════════════════════════════
# Sinhala reference block builder
# ══════════════════════════════════════════════════════════════════════════════

def build_sinhala_reference_block(
    pali_block:    str,
    sinhala_block: str,
    max_chars:     int = -1,
) -> str:
    """
    Format the Sinhala reference for the prompt.
    The Sinhala translation is shown first (it is the primary guide).
    The Pāli source is shown below it, truncated to fit if necessary.

    If max_chars == -1 no truncation is applied.
    """
    unlimited = (max_chars == -1)

    si_section = f"[Sinhala]\n{sinhala_block.strip()}" if sinhala_block.strip() \
                 else "[Sinhala]\n(not available)"
    pa_section = f"[Pāli]\n{pali_block.strip()}" if pali_block.strip() \
                 else "[Pāli]\n(not available)"

    if unlimited:
        return f"{si_section}\n\n{pa_section}"

    # Sinhala is never truncated — it is the primary reference.
    if len(si_section) >= max_chars:
        return si_section + "\n\n[Pāli]\n... [truncated]"

    remaining = max_chars - len(si_section) - 2  # -2 for the \n\n
    if len(pa_section) > remaining:
        pa_section = pa_section[: max(10, remaining - 15)].rstrip() + " … [truncated]"

    return f"{si_section}\n\n{pa_section}"


# ══════════════════════════════════════════════════════════════════════════════
# Nissaya block builder for prompt
# ══════════════════════════════════════════════════════════════════════════════

def build_nissaya_block_for_prompt(
    paragraphs:     list[dict],
    nissaya_blocks: dict[int, str],
) -> str:
    """
    Rebuild the nissaya section for the prompt:
      --- para_id=X ---
        [line_id=Y] pali_sentence
          Nissaya: pali1: meaning1 | pali2: meaning2 | …
    """
    parts = []
    for para in paragraphs:
        pid = para["para_id"]
        raw = nissaya_blocks.get(pid, "")
        if not raw or raw.startswith("(no nissaya"):
            parts.append(f"--- para_id={pid} ---\n(no nissaya)")
            continue

        section_lines = []
        for sent in para["sentences"]:
            lid       = sent["line_id"]
            pali_sent = sent.get("pali_sentence", "")
            m = re.search(
                rf"\[line_id={lid}\][^\n]*\n\s*Nissaya:\s*(.*?)(?=\n\[line_id=|\Z)",
                raw,
                re.DOTALL,
            )
            niss_raw = m.group(1).strip() if m else ""
            niss_fmt = format_nissaya_line(niss_raw)
            section_lines.append(
                f"  [line_id={lid}] {pali_sent}\n"
                f"    Nissaya: {niss_fmt}"
            )
        parts.append(f"--- para_id={pid} ---\n" + "\n".join(section_lines))

    return "\n\n".join(parts) or "(no nissaya available)"


# ══════════════════════════════════════════════════════════════════════════════
# Chunking — token-safe, paragraph-aware
# ══════════════════════════════════════════════════════════════════════════════

def chunk_paragraphs(
    paragraphs: list[dict],
    max_tokens: int = 2000,
) -> list[list[dict]]:
    """
    Group paragraphs into chunks so that each chunk stays under max_tokens.
    Never splits within a paragraph.
    """
    chunks        = []
    current_chunk : list[dict] = []
    current_tokens = 0

    for para in paragraphs:
        para_text   = "\n".join(
            s.get("pali_sentence", "") for s in para.get("pending", [])
        )
        para_tokens = estimate_tokens(para_text)

        if current_chunk and (current_tokens + para_tokens > max_tokens):
            chunks.append(current_chunk)
            current_chunk  = []
            current_tokens = 0

        current_chunk.append(para)
        current_tokens += para_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(
    book_id:               str,
    para_id:               int,
    section_title:         str,
    paragraphs:            list[dict],
    pali_block:            str,
    sinhala_block:         str,
    nissaya_blocks:        dict[int, str],
    glossary_block:        str,
    commentary_block:      str  = "(no commentary available)",
    pali_defs_block:       str  = "(no word definitions available)",
    prev_para_text:        str  = "",
    max_ref_chars:         int  = -1,
) -> tuple[str, list[dict]]:
    """
    Build the user prompt for one chunk of paragraphs.

    Returns (prompt_str, flat_sentence_list).
    flat_sentence_list has one dict per pending sentence with para_id tagged.
    """
    flat_sentences = []
    for para in paragraphs:
        for s in para["pending"]:
            flat_sentences.append({
                "para_id":       para["para_id"],
                "line_id":       s["line_id"],
                "pali_sentence": s.get("pali_sentence", ""),
            })

    sinhala_ref_block = build_sinhala_reference_block(
        pali_block, sinhala_block, max_ref_chars
    )
    nissaya_section = build_nissaya_block_for_prompt(paragraphs, nissaya_blocks)
    prev_block      = prev_para_text or "(no previous paragraph)"

    prompt = USER_TEMPLATE.format(
        book_id                 = book_id,
        para_id                 = para_id,
        section_title           = section_title,
        glossary_block          = glossary_block,
        sinhala_reference_block = sinhala_ref_block,
        commentary_block        = commentary_block,
        pali_defs_block         = pali_defs_block,
        prev_para_block         = prev_block,
        nissaya_block           = nissaya_section,
        sentences_json          = json.dumps(
            flat_sentences, ensure_ascii=False, indent=2
        ),
    )
    return prompt, flat_sentences


# ══════════════════════════════════════════════════════════════════════════════
# AI call  — with debug log dump and retry logic
# ══════════════════════════════════════════════════════════════════════════════

_RETRYABLE_PATTERNS = (
    "503", "unavailable", "resource_exhausted", "quota", "rate",
    "limit", "429", "exhausted", "overloaded", "service unavailable",
    "too many requests",
)


def _is_retryable_error(exc: Exception) -> bool:
    return any(p in str(exc).lower() for p in _RETRYABLE_PATTERNS)


def _heartbeat_aware_sleep(
    total_seconds: int,
    heartbeat_fn: "Callable[[], None] | None" = None,
    log_info: "Callable[[str], None] | None" = None,
    heartbeat_interval: int = 10,
):
    """
    Sleep for total_seconds, sending heartbeat every heartbeat_interval seconds.
    If heartbeat_fn is None, regular sleep is used.
    """
    if heartbeat_fn is None:
        time.sleep(total_seconds)
        return

    elapsed = 0
    while elapsed < total_seconds:
        chunk = min(heartbeat_interval, total_seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        if elapsed < total_seconds:
            heartbeat_fn()
            if log_info:
                remaining = total_seconds - elapsed
                log_info(f"[AI] Waiting for API recovery... ({remaining}s remaining)")


def call_ai_with_logging(
    ask_ai_fn:         Callable[[str, str], str],
    prompt:            str,
    book_id:           str,
    para_id:           int,
    chunk_idx:         int,
    log_dir:           str,
    log_info:          Callable[[str], None],
    log_sucess:        Callable[[str], None],
    log_error:         Callable[[str], None],
    heartbeat_fn:      Callable[[], None] = None,
    max_chunk_retries: int = 4,
    base_wait_sec:     int = 60,
) -> "str | None":
    """
    Call the AI with retry/backoff; write prompt and response to log_dir.

    Retry policy:
      - Rate-limit / quota / 503 errors: exponential backoff
        waits: 60 s, 120 s, 240 s, 480 s (doubles each attempt).
      - TimeoutError: returns None so the caller can skip this chunk.
      - Other exceptions: re-raised immediately.
    
    If heartbeat_fn is provided, it's called every 10 seconds during waits.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    safe_id    = re.sub(r"[^\w\-]", "_", f"{book_id}_p{para_id}")
    base_name  = f"{timestamp}_{safe_id}_chunk{chunk_idx:02d}"

    prompt_path = os.path.join(log_dir, f"{base_name}_prompt.txt")
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("=== SYSTEM ===\n")
            f.write(SYSTEM_PROMPT)
            f.write("\n\n=== USER ===\n")
            f.write(prompt)
        log_info(f"[AI] Prompt written → {prompt_path}")
    except OSError as exc:
        log_error(f"[AI] Could not write prompt log: {exc}")

    n_tokens = estimate_tokens(prompt)
    log_info(
        f"[AI] Calling AI: {book_id}/para={para_id}, chunk={chunk_idx}, "
        f"{len(prompt)} chars (~{n_tokens} tokens)."
    )

    raw      = None
    attempt  = 0
    wait_sec = base_wait_sec

    while attempt <= max_chunk_retries:
        try:
            raw = ask_ai_fn(prompt, SYSTEM_PROMPT)
            break
        except TimeoutError as exc:
            log_error(
                f"[AI] Timeout: {book_id}/para={para_id} chunk={chunk_idx}: {exc}"
            )
            return None
        except Exception as exc:
            if _is_retryable_error(exc) and attempt < max_chunk_retries:
                attempt += 1
                log_error(
                    f"[AI] Retryable error (attempt {attempt}/{max_chunk_retries}): "
                    f"{exc}. Waiting {wait_sec}s before retry…"
                )
                # Heartbeat-aware sleep: send heartbeat every 10s during wait
                _heartbeat_aware_sleep(wait_sec, heartbeat_fn, log_info)
                wait_sec *= 2
            else:
                raise

    if raw is None:
        return None

    log_sucess(f"[AI] Response received: {len(raw)} chars.")

    response_path = os.path.join(log_dir, f"{base_name}_response.txt")
    try:
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(raw)
        log_info(f"[AI] Response written → {response_path}")
    except OSError as exc:
        log_error(f"[AI] Could not write response log: {exc}")

    return raw


# ══════════════════════════════════════════════════════════════════════════════
# Response parser  — with truncation salvage
# ══════════════════════════════════════════════════════════════════════════════

def parse_response(raw: str) -> dict:
    """
    Extract {"translations": [...], "glossary": [...]} from the AI output.

    Handles:
      - Markdown code fences
      - Truncated "translations" array (salvages all complete objects)
      - Missing "glossary" key (returns empty list)
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^\s*```[a-zA-Z]*\s*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?\s*```\s*$",           "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # ── Happy path: full valid JSON ───────────────────────────────────────────
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            obj.setdefault("translations", [])
            obj.setdefault("glossary",     [])
            return obj
        except json.JSONDecodeError:
            pass

    # ── Salvage path: extract arrays even if the response was truncated ───────
    obj = {"translations": [], "glossary": []}

    for key in ("translations", "glossary"):
        m = re.search(rf'"{key}"\s*:\s*\[', cleaned)
        if not m:
            continue
        array_start = m.end() - 1

        depth      = 0
        obj_start  = None
        items: list = []

        for i, ch in enumerate(cleaned[array_start:], start=array_start):
            if ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        items.append(json.loads(cleaned[obj_start : i + 1]))
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
            elif ch == "]" and depth == 0:
                break

        obj[key] = items

    return obj