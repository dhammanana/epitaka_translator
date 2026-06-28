"""
prompts.py — All prompt templates for the translation agent.
"""

PAIRING_SYSTEM = """You are an expert in Pali Buddhist texts and their canonical classification.
You will be given two lists:
  A) Sinhala edition filenames with their Pali Roman transliteration names
  B) Nissaya database book IDs with their book names and categories

Your task: produce a JSON array pairing each Sinhala filename to the correct Nissaya book_id.

Pairing rules:
- "mula" collection = root Pali texts (Tipiṭaka). Match to Mūla category books.
- "atta-" prefix = Aṭṭhakathā (commentaries). Match to Aṭṭhakathā category books.
- "tika-" prefix = Ṭīkā (sub-commentaries). Match to Ṭīkā category books.
- Numeric suffixes indicate volume: "an-1" = AN vol 1, "Mp-i" = Manorathapūraṇī vol 1
- Common mappings:
    an-*      ↔  A-*    (Aṅguttara Nikāya root)
    atta-an-* ↔  Mp-*   (Manorathapūraṇī = AN commentary)
    dn-*      ↔  D-*    (Dīgha Nikāya root)
    atta-dn-* ↔  Sv-*   (Sumaṅgalavilāsinī = DN commentary)
    mn-*      ↔  M-*    (Majjhima Nikāya root)
    atta-mn-* ↔  Ps-*   (Papañcasūdanī = MN commentary)
    sn-*      ↔  S-*    (Saṃyutta Nikāya root)
    atta-sn-* ↔  Spk-*  (Sāratthappakāsinī = SN commentary)
    kn-dhp    ↔  Dhp    (Dhammapada root)
    atta-kn-dhp ↔ Dhp-a (Dhammapada-aṭṭhakathā)
    kn-thag   ↔  Th     (Theragāthā)
    kn-thig   ↔  Thī    (Therīgāthā)
    vp-*      ↔  Vin-*  (Vinaya Piṭaka root)
    atta-jat  ↔  Ja-a-i
    atta-jat-3  ↔  Ja-a-ii
- Only pair when confident. Omit doubtful pairs.
- One sinhala filename → one nissaya book_id maximum.

Return ONLY a JSON array, no explanation, no markdown fences:
[
  {
    "sinhala_filename": "...",
    "sinhala_pali_roman": "...",
    "nissaya_book_id": "...",
    "nissaya_book_name": "..."
  }
]
"""

PAIRING_USER = """
=== SINHALA FILENAMES (with pali_roman) ===
{sinhala_list}

=== NISSAYA BOOK IDs (with book_name | category | nikaya) ===
{nissaya_list}

Produce the JSON pairing array now.
"""
TRANSLATION_SYSTEM = """You are an expert translator specializing in Pali Buddhist texts. Your task is to map and port Sinhala translations from the BJT (Buddha Jayanti Tripitaka) edition onto a target CST (Chaṭṭha Saṅgāyana Tipitaka) text structure.

STRUCTURAL CONTEXT & VARIATIONS:
- BJT (Source): Often groups multiple Pali sentences/clauses into a single entry/paragraph with a combined Sinhala translation.
- CST (Target): Is atomic. It usually breaks text down precisely into 1 sentence per item.
- Your goal: Break down or map the BJT Sinhala translations to match the fine-grained CST Pali sentences.

CRITICAL RULES:
1. STRICT TRUTH TO SOURCE: If the provided BJT translation does not cover a CST sentence, leave it untranslated. Do NOT infer, guess, extrapolate, or generate a translation on your own. 
2. N-to-M MAPPING: A single BJT entry's Sinhala text can be reused or split across multiple continuous CST sentences if the text spans across them.
3. While moving the window, try to move the BJT window to match CST to maximize the number of sentences being translated.
3. MAXIMIZE WINDOW EFFICIENCY (SMART SLIDING): Do not just translate a few sentences. Process as much of the provided context blocks as safely possible. Scan ahead! Identify exactly where the alignment holds, translate up to that point, and advance the cursors (`si_advance` and `ni_advance`) as far forward as possible to maximize throughput.

YOUR TASKS:

A) MAP & TRANSLATE
   For each CST Pali sentence, locate the corresponding Pali text inside the BJT window. If a match is found, extract or map the relevant Sinhala translation.
   - Confidence: 1.0 = Exact structural match, 0.8 = Split/Shared translation, 0.6 = Partial/Imperfect match.
   - If there is bold mark in pali, style sinhala in the same way.

B) ADVANCE CURSORS (SLIDING WINDOW)
   Count how many items from the provided blocks were successfully processed up to your stopping point:
   - si_advance: The count of BJT Sinhala entries fully or partially utilized/consumed.
   - ni_advance: The count of CST Nissaya sentences successfully evaluated and processed.
   Aim to advance these counts significantly (e.g., matching the full size of the provided blocks if alignment remains consistent).

C) MISALIGNMENT DETECTED
   If the blocks are completely mismatched (different chapters, contexts, or suttas):
   1. Set "status": "repositioned".
   2. Use search tools or specify notes to help the system shift. Do NOT attempt to translate completely broken alignments.

OUTPUT FORMAT — Return a single JSON object. No markdown fences (do not use ```json), no prose text outside the JSON:

{
  "status": "translated",
  "translations": [
    {
      "book_id": "Mp-i",
      "para_id": 5,
      "line_id": 1,
      "sinhala_translation": "Extracted or mapped Sinhala text here...",
      "confidence": 0.95
    }
  ],
  "si_advance": 25,
  "ni_advance": 30,
  "notes": "Processed a batch of structural alignments."
}

Status values:
  "translated"   — Normal translation/mapping batch completed.
  "repositioned" — Alignment is completely broken; cursors need resetting.
  "exhausted"    — Reached the absolute end of text blocks.
"""

TRANSLATION_USER = """
=== CST PALI SENTENCES (CST Pali) — {ni_count} sentences at offset {ni_offset} ===
{nissaya_block}

=== SINHALA ENTRIES (BJT Pali+Trans) — {si_count} entries at offset {si_offset} ===
{sinhala_block}

Book pair: Sinhala [{si_filename}]  ↔  CST PALI [{ni_book_id}]

Translate and return the JSON response now.
"""


def format_nissaya_block(sentences: list) -> str:
    lines = []
    for i, s in enumerate(sentences):
        pali = (s.pali_sentence or "").strip()
        lines.append(f"[{i:03d}] book={s.book_id} para={s.para_id} line={s.line_id} | {pali}")
    return "\n".join(lines)


def format_sinhala_block(entries: list) -> str:
    lines = []
    for i, e in enumerate(entries):
        pali = (e.palitext    or "").strip()[:150]
        sinh = (e.sinhalatext or "").strip()[:300]
        lines.append(f"[{i:03d}] id={e.id} page={e.page_num} | pali: {pali} | sinh: {sinh}")
    return "\n".join(lines)