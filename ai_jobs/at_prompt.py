"""
jobs/at_prompt.py  —  Prompt assembly for AṭṭhakathāTranslatorJob.

Differences from st_prompt.py
──────────────────────────────
•  No SC-reference block (these texts have no SuttaCentral parallel).
•  Two new context sources injected instead:
     – MŪLA TEXT WITH TRANSLATIONS  (root text + already-translated English)
     – SIMILAR ALREADY-TRANSLATED SENTENCES  (from the same book)
•  System prompt revised to reflect commentary-translation task and the
   new context sources.
•  build_prompt() has a different signature matching the new sources.
•  All chunking / AI dispatch / response parsing is delegated to st_prompt.
"""

import json
import re
from typing import Callable

# Re-use unchanged helpers from st_prompt
from ai_jobs.st_prompt import (
    build_nissaya_block_for_prompt,
    estimate_tokens,
    chunk_paragraphs,           # re-exported so callers can import from here
    call_ai_with_logging,       # re-exported
    parse_response,             # re-exported
    extract_pali_ngrams,        # re-exported
    SYSTEM_PROMPT as _ST_SYSTEM_PROMPT,  # kept for reference only
)

# ══════════════════════════════════════════════════════════════════
# System prompt — commentary / sub-commentary variant
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a scholar-translator specialising in Pāli Buddhist commentarial literature.
You will be given:
  1. SENTENCES TO TRANSLATE — a JSON array of Pāli sentences from an aṭṭhakathā or ṭīkā.
  2. MŪLA TEXT WITH TRANSLATIONS — the root-text (mūla) paragraph(s) that this commentary
     passage discusses, together with their existing English translations.
     ► Use this as your primary guide to terminology, register, and doctrinal framing.
       The commentary translation MUST use the same rendering for every technical term
       that appears in the mūla translation.
  3. SIMILAR ALREADY-TRANSLATED SENTENCES — other translated passages from the same book
     that share significant vocabulary with the current sentences.
     ► Use these for consistent phrasing and idiomatic style.
  4. MYANMAR NISSAYA — word-by-word gloss for each sentence (already romanised).
  5. ESTABLISHED GLOSSARY — fixed translations for specific Pāli terms/phrases.
  6. PALI COMMENTARY & SUB-COMMENTARY — any deeper commentary that comments on these lines.
  7. PALI WORD DEFINITIONS — example sentences for difficult/rare words.
  8. PREVIOUS PARAGRAPH — the translation of the immediately preceding paragraph.

Your task is to return a single JSON object with exactly two keys:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A. "translations"  — array, one entry per input sentence (same order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  { "para_id": <int>, "line_id": <int>, "english_translation": "<text>" }

Rules:
  • Mirror the terminology of the mūla translation exactly — this is mandatory.
  • Use the nissaya gloss to resolve uncertain compounds and syntax.
  • Apply every glossary term exactly, including multi-word phrases.
  • Reference similar sentences for consistent register and phrasing.
  • Reference the previous paragraph for style continuity.
  • Keep untranslated Pāli (with diacritics) only when no English equivalent
    exists; gloss in parentheses on first occurrence only.
  • Commentarial hedges (tīkā style: "here … means …", "the meaning is …")
    should be rendered naturally in English without over-formalising.
  • No verse numbers, footnotes, or meta-commentary in the output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B. "glossary"  — NEW terms only (not already in the supplied glossary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  { "pali": "…", "english": "…", "domain": "…",
    "sub_domain": "…", "context": "…", "note": "…" }
domain should be one of: sutta, vinaya, abhidhamma, grammar, story.
Include technical terms, commentarial formulae, and compound terms you decided.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — critical
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON. No markdown fences, no prose outside the JSON.
{ "translations": [...], "glossary": [...] }
"""

# ══════════════════════════════════════════════════════════════════
# User prompt template
# ══════════════════════════════════════════════════════════════════

USER_TEMPLATE = """Text: {heading_label}

══════════════════════════════
MŪLA TEXT WITH TRANSLATIONS (mirror all technical terms from here)
══════════════════════════════
{mula_block}

══════════════════════════════
SIMILAR ALREADY-TRANSLATED SENTENCES (same book — for consistent phrasing)
══════════════════════════════
{similar_block}

══════════════════════════════
ESTABLISHED GLOSSARY (apply exactly, including multi-word phrases)
══════════════════════════════
{glossary_block}

══════════════════════════════
PALI COMMENTARY & SUB-COMMENTARY (deeper commentary on these lines, if any)
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
# Prompt builder
# ══════════════════════════════════════════════════════════════════

def build_prompt(
    heading_label:   str,
    paragraphs:      list[dict],
    nissaya_blocks:  dict[int, str],
    glossary_block:  str,
    commentary_block: str = "(no commentary available)",
    pali_defs_block:  str = "(no word definitions available)",
    mula_block:       str = "(no mūla text available)",
    similar_block:    str = "(no similar sentences found)",
    prev_para_text:   str = "",
) -> tuple[str, list[dict]]:
    """
    Build the user prompt for one chunk of commentary paragraphs.

    Returns (prompt_str, flat_sentence_list).
    """
    flat_sentences = []
    for para in paragraphs:
        for s in para["pending"]:
            flat_sentences.append({
                "para_id":       para["para_id"],
                "line_id":       s["line_id"],
                "pali_sentence": s["pali_sentence"],
            })

    nissaya_block  = build_nissaya_block_for_prompt(paragraphs, nissaya_blocks)
    prev_para_block = prev_para_text or "(no previous paragraph)"

    prompt = USER_TEMPLATE.format(
        heading_label    = heading_label,
        mula_block       = mula_block,
        similar_block    = similar_block,
        glossary_block   = glossary_block,
        commentary_block = commentary_block,
        pali_defs_block  = pali_defs_block,
        prev_para_block  = prev_para_block,
        nissaya_block    = nissaya_block,
        sentences_json   = json.dumps(flat_sentences, ensure_ascii=False, indent=2),
    )
    return prompt, flat_sentences