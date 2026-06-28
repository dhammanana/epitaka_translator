"""
prompts.py — Prompt templates for pairing Thai Volumes.
"""

PAIRING_SYSTEM = """You are an expert in Pali Tipitaka and Thai Buddhist texts.
Your task is to map a single Thai Tipitaka volume to its corresponding CST (Nissaya) Pali books.

CRITICAL RULES:
1. Thai volumes often contain BOTH the root text (Mūla) and its commentary (Aṭṭhakathā).
2. Sometimes a Thai volume corresponds to 1-3 different books in the Nissaya database (e.g., spanning across sections).
3. You MUST use the `get_nissaya_headings` tool to look up candidate Nissaya book IDs to verify that their headings align with the provided Thai headings. Look for phonetic/structural matches (e.g., Thai "มหาวิภังค์" -> Pali "Mahāvibhaṅga").
4. Once you have confidently verified the exact Mūla and Aṭṭhakathā books, return a final JSON array.

Return ONLY a JSON array, with no markdown fences or explanation:
[
  {
    "thai_volume_id": "...",
    "thai_book_name": "...",
    "nissaya_book_id": "...",
    "category": "Mūla"
  },
  {
    "thai_volume_id": "...",
    "thai_book_name": "...",
    "nissaya_book_id": "...",
    "category": "Aṭṭhakathā"
  }
]
"""

PAIRING_USER = """
=== THAI VOLUME DATA ===
Volume ID: {vol_id}
Book Name: {vol_name}

--- THAI HEADINGS ---
{thai_headings}

=== CANDIDATE NISSAYA BOOKS (Mūla & Aṭṭhakathā) ===
{nissaya_list}

Please use the `get_nissaya_headings` tool to investigate the most likely candidate book_ids.
When you are certain of both the Mūla and Aṭṭhakathā mappings for this Thai Volume, output the final JSON array.
"""

HEADING_MATCH_SYSTEM = """You are an expert in Pali Tipitaka and Thai Buddhist texts.
Your task is to accurately map the structural headings of a Thai Tipitaka volume to the headings of a specific CST (Nissaya) Pali book.

CRITICAL RULES:
1. The provided Thai volume list might contain headings for BOTH the root text (Mūla) and commentary (Aṭṭhakathā). 
2. You are currently mapping ONLY for the specified Nissaya book ({category}). IGNORE Thai headings that belong to a different text/category.
3. Look for phonetic/structural matches (e.g., "เวรัญชกัณฑ์" -> "Verañjakaṇḍa", "มหาวรรค" -> "Mahāvaggo").
4. Return a JSON array linking the Thai page number to the Nissaya para_id. Include the titles for verification.
5. Provide a confidence score (0.0 to 1.0). 1.0 = exact match. 

Return ONLY a JSON array, with no markdown fences or explanation:
[
  {{
    "thai_page": "15",
    "thai_title": "๑. เวรัญชกัณฑ์",
    "nissaya_para_id": 4,
    "nissaya_title": "Verañjakaṇḍa",
    "confidence": 0.98
  }}
]
"""

HEADING_MATCH_USER = """
=== TASK INFO ===
Target Nissaya Book: [{nissaya_book_id}] ({category})
Thai Volume: [{thai_volume_id}] {thai_book_name}

=== NISSAYA HEADINGS ({nissaya_book_id}) ===
{nissaya_headings}

=== THAI HEADINGS (Volume {thai_volume_id}) ===
{thai_headings}

Produce the JSON mapping array now.
"""
