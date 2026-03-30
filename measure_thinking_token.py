#!/usr/bin/env python3
"""
tools/measure_thinking_tokens.py
---------------------------------
One-time diagnostic: sends a small sample of real prompts to Gemini and
reports how many output tokens (including thinking tokens) each call uses.

Run from the project root:
    python tools/measure_thinking_tokens.py --samples 5

Requirements:
  pip install google-generativeai
  GOOGLE_API_KEY env var set (or pass --api-key)

Output:
  sample  sentences  prompt_chars  output_chars  output_tokens  finish_reason
  -----------------------------------------------------------------------
  1       8          4321          2187          547            STOP
  ...
  Average output tokens per sentence: 68.3
  Recommended max_sentences for 8192 token output budget: 119
"""

import argparse
import json
import os
import sys
import time

# ── Allow running from project root without installing the package ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NISSAYA_DB, SC_DATA_DB
import sqlite3


def get_sample_prompts(n_samples: int, max_sentences: int = 15) -> list[dict]:
    """Pull n_samples real headings from the DB and build minimal prompts."""
    from ai_jobs.st_prompt import (
        build_prompt, build_sc_reference_block, SYSTEM_PROMPT
    )
    from ai_jobs.st_data import (
        fetch_nissaya_map, build_nissaya_block, nissaya_path, sc_path,
        _connect,
    )

    params = {}   # use config defaults

    samples = []

    with _connect(str(NISSAYA_DB)) as nis_conn:
        headings = nis_conn.execute(
            """SELECT * FROM headings
               WHERE sc_id IS NOT NULL AND sc_id != ''
               ORDER BY RANDOM()
               LIMIT ?""",
            (n_samples * 3,)   # over-fetch in case some have no sentences
        ).fetchall()

        for h in headings:
            if len(samples) >= n_samples:
                break
            book_id    = h["book_id"]
            para_id    = h["para_id"]
            sc_id      = h["sc_id"]
            chapter_len = int(h["chapter_len"] or 1)

            # Collect sentences
            paragraphs = []
            for pid in range(para_id, para_id + chapter_len):
                rows = nis_conn.execute(
                    """SELECT line_id, pali_sentence, english_translation
                       FROM sentences WHERE book_id=? AND para_id=?
                       ORDER BY line_id""",
                    (book_id, pid)
                ).fetchall()
                if not rows:
                    continue
                sentences = [dict(r) for r in rows]
                pending   = [s for s in sentences
                             if not (s.get("english_translation") or "").strip()]
                if pending:
                    paragraphs.append({
                        "book_id": book_id, "para_id": pid,
                        "sentences": sentences, "pending": pending,
                    })

            if not paragraphs:
                continue

            # Limit to max_sentences
            flat = []
            trimmed_paras = []
            for para in paragraphs:
                if len(flat) >= max_sentences:
                    break
                room = max_sentences - len(flat)
                trimmed = {**para, "pending": para["pending"][:room]}
                trimmed_paras.append(trimmed)
                flat.extend(trimmed["pending"])

            # SC reference
            with _connect(str(SC_DATA_DB)) as sc_conn:
                row = sc_conn.execute(
                    "SELECT * FROM en_translation WHERE sc_id=?", (sc_id,)
                ).fetchone()
            pali_text = row["palitext"] if row else ""
            en_text   = row["entext"]   if row else ""
            sutta_name = row["sutta_name"] if row else sc_id

            # Nissaya
            nissaya_blocks = {}
            for para in trimmed_paras:
                niss_map = fetch_nissaya_map(params, para["book_id"], para["para_id"], lambda x: None)
                nissaya_blocks[para["para_id"]] = build_nissaya_block(para["sentences"], niss_map)

            prompt, _ = build_prompt(
                sc_id          = sc_id,
                sutta_name     = sutta_name,
                paragraphs     = trimmed_paras,
                pali_text      = pali_text,
                en_text        = en_text,
                nissaya_blocks = nissaya_blocks,
                glossary_block = "(no glossary for diagnostic run)",
                max_ref_chars  = 3000,
            )

            samples.append({
                "sc_id":       sc_id,
                "n_sentences": len(flat),
                "prompt":      prompt,
                "system":      SYSTEM_PROMPT,
            })

    return samples


def call_gemini_and_measure(api_key: str, model: str, sample: dict) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    gen_cfg = {
        "temperature":       0.3,
        "max_output_tokens": 8192,
    }
    safety = [
        {"category": c, "threshold": "BLOCK_NONE"}
        for c in [
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
    ]

    full_prompt = f"{sample['system']}\n\n{sample['prompt']}"
    m = genai.GenerativeModel(model, generation_config=gen_cfg, safety_settings=safety)

    t0   = time.time()
    resp = m.generate_content(full_prompt, request_options={"timeout": 120})
    elapsed = time.time() - t0

    output_text = resp.text or ""

    # Try to get usage metadata
    usage = {}
    try:
        usage = {
            "prompt_tokens":    resp.usage_metadata.prompt_token_count,
            "candidates_tokens": resp.usage_metadata.candidates_token_count,
            "total_tokens":     resp.usage_metadata.total_token_count,
        }
    except Exception:
        pass

    finish_reason = "?"
    try:
        finish_reason = str(resp.candidates[0].finish_reason)
    except Exception:
        pass

    return {
        "sc_id":             sample["sc_id"],
        "n_sentences":       sample["n_sentences"],
        "prompt_chars":      len(sample["prompt"]),
        "output_chars":      len(output_text),
        "finish_reason":     finish_reason,
        "elapsed_s":         round(elapsed, 1),
        **usage,
    }


def main():
    ap = argparse.ArgumentParser(description="Measure Gemini output token usage for translation prompts")
    ap.add_argument("--samples",  type=int, default=5,                    help="Number of sample prompts to test")
    ap.add_argument("--api-key",  default=os.environ.get("GOOGLE_API_KEY"), help="Gemini API key")
    ap.add_argument("--model",    default="gemini-2.5-flash",             help="Model name")
    ap.add_argument("--max-sentences", type=int, default=15,              help="Sentences per sample prompt")
    ap.add_argument("--out",      default=None,                           help="Optional JSON output file")
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: provide --api-key or set GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)

    print(f"Building {args.samples} sample prompt(s) ({args.max_sentences} sentences each) …")
    samples = get_sample_prompts(args.samples, args.max_sentences)
    print(f"Got {len(samples)} sample(s).\n")

    header = f"{'#':<4} {'sc_id':<20} {'sents':>5} {'p_chars':>8} {'o_chars':>8} {'p_tok':>6} {'o_tok':>6} {'total':>7} {'finish':<12} {'sec':>5}"
    print(header)
    print("-" * len(header))

    results = []
    for i, sample in enumerate(samples, 1):
        try:
            r = call_gemini_and_measure(args.api_key, args.model, sample)
            results.append(r)
            print(
                f"{i:<4} {r['sc_id']:<20} {r['n_sentences']:>5} "
                f"{r['prompt_chars']:>8} {r['output_chars']:>8} "
                f"{r.get('prompt_tokens','?'):>6} {r.get('candidates_tokens','?'):>6} "
                f"{r.get('total_tokens','?'):>7} {r['finish_reason']:<12} {r['elapsed_s']:>5}"
            )
        except Exception as exc:
            print(f"{i:<4} {sample['sc_id']:<20} ERROR: {exc}")
        time.sleep(2)   # avoid rate limit

    if not results:
        print("No results.")
        return

    # Summary
    o_toks = [r.get("candidates_tokens") for r in results if isinstance(r.get("candidates_tokens"), int)]
    n_sents = [r["n_sentences"] for r in results]

    if o_toks:
        avg_toks        = sum(o_toks) / len(o_toks)
        avg_sents       = sum(n_sents) / len(n_sents)
        toks_per_sent   = avg_toks / avg_sents if avg_sents else 0
        budget          = 8192
        recommended     = int(budget * 0.85 / toks_per_sent) if toks_per_sent else "?"
        print(f"\nAverage output tokens : {avg_toks:.0f}")
        print(f"Average sentences     : {avg_sents:.1f}")
        print(f"Tokens per sentence   : {toks_per_sent:.1f}")
        print(f"Recommended max_sentences for {budget}-token budget (85% safety): {recommended}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
