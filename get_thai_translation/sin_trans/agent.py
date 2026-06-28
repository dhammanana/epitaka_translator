"""
agent.py — LangGraph-based translation agent.

Graph: load_window → translate → save → load_window (loop) / END (exhausted)

Provider priority: Groq (llama-3.1-8b-instant) → Gemini fallback.
Key rotation and retry are handled by keys.py:call_with_rotation().
"""

import json
import logging
import time
import re
from typing import TypedDict, Optional

# Silence SDK retry loggers before any imports touch them
logging.getLogger("google.genai._api_client").setLevel(logging.CRITICAL)
logging.getLogger("groq").setLevel(logging.CRITICAL)

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

from db import SinhalaDB, NissayaDB, OutputDB, BookPair
from tools import ALL_TOOLS, bind_databases, bind_current_book, _invoke_tool
from prompts import (
    TRANSLATION_SYSTEM, TRANSLATION_USER,
    format_nissaya_block, format_sinhala_block,
)
from keys import (
    call_with_rotation, rotate_key, log, Colors,
    make_groq_llm, make_gemini_llm,
    _groq_keys, _gemini_keys,
    current_groq_key, current_gemini_key,
    _rotate_groq, _rotate_gemini,
)

logger = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
# Groq: llama-3.1-8b-instant has 14,400 req/day free — use as primary
# Gemini: 2.5-flash-lite has 1,000 req/day free — use as fallback
GROQ_MODEL   = "llama-3.1-8b-instant"
GEMINI_MODEL = "gemini-2.5-flash-lite"

WINDOW_SIZE     = 50
MAX_TOOL_ROUNDS = 6
SLEEP_BETWEEN   = 2   # seconds between API calls


# ── State ──────────────────────────────────────────────────────────────────────

class TranslationState(TypedDict):
    book_pair:    BookPair
    si_total:     int
    ni_total:     int
    si_cursor:    int
    ni_cursor:    int
    si_entries:   list
    ni_sentences: list
    tool_rounds:  int
    status:       str        # "running" | "exhausted" | "error"
    error:        Optional[str]
    pending_translations: list


# ── LLM factory ───────────────────────────────────────────────────────────────

def make_llm(model: str = None):
    """
    Return a LangChain LLM bound with tools.
    If model is given, provider is detected from name:
      - llama / gemma / mixtral / qwen → Groq
      - gemini → Gemini
    If no model given, defaults to GROQ_MODEL or GEMINI_MODEL based on available keys.
    """
    _groq_pat   = re.compile(r"llama|gemma|mixtral|qwen|whisper", re.IGNORECASE)
    _gemini_pat = re.compile(r"gemini", re.IGNORECASE)

    if model is None:
        model = GROQ_MODEL if _groq_keys else GEMINI_MODEL

    if _gemini_pat.search(model):
        if not _gemini_keys:
            raise RuntimeError(f"Model '{model}' requires Gemini keys but none are loaded.")
        log("MODEL", f"{Colors.CYAN}Using Gemini / {model}{Colors.RESET}")
        return make_gemini_llm(model, ALL_TOOLS)
    elif _groq_pat.search(model):
        if not _groq_keys:
            raise RuntimeError(f"Model '{model}' requires Groq keys but none are loaded.")
        log("MODEL", f"{Colors.CYAN}Using Groq / {model}{Colors.RESET}")
        return make_groq_llm(model, ALL_TOOLS)
    else:
        # Unknown model name — try Groq first, fall back to Gemini
        if _groq_keys:
            log("MODEL", f"{Colors.CYAN}Using Groq / {model}{Colors.RESET}")
            return make_groq_llm(model, ALL_TOOLS)
        else:
            log("MODEL", f"{Colors.CYAN}Using Gemini / {model}{Colors.RESET}")
            return make_gemini_llm(model, ALL_TOOLS)


def _rebuild_llm_with_next_key():
    """
    After a 429, rotate to the next key and rebuild the LLM so it picks
    up the new API key. Falls through Groq keys first, then Gemini.
    """
    if _groq_keys:
        _rotate_groq(reason="rebuild after 429")
        return make_groq_llm(GROQ_MODEL, ALL_TOOLS)
    elif _gemini_keys:
        _rotate_gemini(reason="rebuild after 429 (Groq exhausted)")
        return make_gemini_llm(GEMINI_MODEL, ALL_TOOLS)
    return None


def _llm_invoke(llm, messages, tag: str = "translate"):
    """
    Invoke LLM with automatic key rotation on quota / server errors.
    Rebuilds the LLM object on each rotation so the new key is used.
    """
    is_quota     = lambda e: "429" in e or "RESOURCE_EXHAUSTED" in e or "rate_limit_exceeded" in e
    is_retryable = lambda e: is_quota(e) or "503" in e or "504" in e or "UNAVAILABLE" in e or "DEADLINE_EXCEEDED" in e

    total_keys  = len(_groq_keys) + len(_gemini_keys)
    max_retries = total_keys * 2
    current_llm = llm

    for attempt in range(1, max_retries + 1):
        try:
            return current_llm.invoke(messages)
        except Exception as e:
            err = str(e)
            if not is_retryable(err):
                log("ERROR", f"{Colors.BRIGHT_RED}LLM non-retryable: {err[:120]}{Colors.RESET}")
                return None

            if attempt >= max_retries:
                log("ERROR", f"{Colors.BRIGHT_RED}All keys exhausted after {max_retries} attempts{Colors.RESET}")
                return None

            if is_quota(err):
                log("RATE_LIMIT", f"Quota hit — rotating key (attempt {attempt}/{max_retries})")
                current_llm = _rebuild_llm_with_next_key()
                if current_llm is None:
                    log("ERROR", f"{Colors.BRIGHT_RED}No more keys available{Colors.RESET}")
                    return None
            else:
                wait = 20 * attempt
                log("RATE_LIMIT",
                    f"Server error — sleeping {Colors.YELLOW}{wait}s{Colors.RESET} "
                    f"(attempt {attempt}/{max_retries})")
                time.sleep(wait)

    return None


# ── Nodes ───────────────────────────────────────────────────────────────────────

def node_load_window(state: TranslationState, *, si_db: SinhalaDB, ni_db: NissayaDB) -> dict:
    bp        = state["book_pair"]
    si_cursor = state["si_cursor"]
    ni_cursor = state["ni_cursor"]

    si_entries   = si_db.get_entries_window(bp.sinhala_filename, si_cursor, WINDOW_SIZE)
    ni_sentences = ni_db.get_sentences_window(bp.nissaya_book_id, ni_cursor, WINDOW_SIZE)

    log("TURN",
        f"[{bp.sinhala_filename}↔{bp.nissaya_book_id}] "
        f"si={si_cursor}/{state['si_total']}  ni={ni_cursor}/{state['ni_total']}  "
        f"loaded si={len(si_entries)} ni={len(ni_sentences)}")

    if not si_entries and not ni_sentences:
        return {"status": "exhausted", "si_entries": [], "ni_sentences": []}
    if not si_entries:
        log("INFO", "Sinhala entries exhausted — done")
        return {"status": "exhausted", "si_entries": [], "ni_sentences": ni_sentences}
    if not ni_sentences:
        log("INFO", "Nissaya sentences exhausted — done")
        return {"status": "exhausted", "si_entries": si_entries, "ni_sentences": []}

    bind_current_book(bp.sinhala_filename, bp.nissaya_book_id)

    return {
        "si_entries":   si_entries,
        "ni_sentences": ni_sentences,
        "tool_rounds":  0,
        "status":       "running",
        "pending_translations": [],
    }


def node_translate(state: TranslationState, *, llm, out_db: OutputDB) -> dict:
    bp           = state["book_pair"]
    si_entries   = state["si_entries"]
    ni_sentences = state["ni_sentences"]

    user_msg = TRANSLATION_USER.format(
        ni_count      = len(ni_sentences),
        ni_offset     = state["ni_cursor"],
        nissaya_block = format_nissaya_block(ni_sentences),
        si_count      = len(si_entries),
        si_offset     = state["si_cursor"],
        sinhala_block = format_sinhala_block(si_entries),
        si_filename   = bp.sinhala_filename,
        ni_book_id    = bp.nissaya_book_id,
    )

    messages = [
        SystemMessage(content=TRANSLATION_SYSTEM),
        HumanMessage(content=user_msg),
    ]

    new_si_cursor = state["si_cursor"]
    new_ni_cursor = state["ni_cursor"]
    tool_rounds   = state.get("tool_rounds", 0)
    repositioned  = False

    tag = f"{bp.sinhala_filename}↔{bp.nissaya_book_id}"

    # Save a readable version of the input messages
    with open('input.txt', 'wt', encoding='utf-8') as f:
        for msg in messages:
            f.write(f"=== {type(msg).__name__} ===\n{msg.content}\n\n")

    response = _llm_invoke(llm, messages, tag=tag)

    # Save a readable version of the output response
    if response is not None:
        with open('output.txt', 'wt', encoding='utf-8') as f:
            f.write(str(response.content if hasattr(response, 'content') else response))

    if response is None:
        out_db.log(bp.id, "ERROR",
                   f"LLM call failed at si={state['si_cursor']} ni={state['ni_cursor']}. "
                   "Progress NOT advanced — book not marked finished.")
        log("ERROR",
            f"{Colors.BRIGHT_RED}Aborting window — cursors held at "
            f"si={state['si_cursor']} ni={state['ni_cursor']}{Colors.RESET}")
        return {
            "status":      "error",
            "error":       "LLM failed",
            "si_cursor":   state["si_cursor"],
            "ni_cursor":   state["ni_cursor"],
            "pending_translations": [],
        }

    # ── Tool-call loop ──────────────────────────────────────────────────────
    while response.tool_calls and tool_rounds < MAX_TOOL_ROUNDS:
        tool_rounds += 1
        messages.append(response)
        tool_messages = []

        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            log("TOOL_CALL", f"{Colors.BRIGHT_YELLOW}{name}{Colors.RESET}({json.dumps(args)[:80]})")

            result_str  = _invoke_tool(name, args)
            result_data = _safe_json(result_str)

            log("TOOL_RESULT", f"{Colors.GREEN}{name}{Colors.RESET} → {result_str[:120]}")

            if name == "reposition_window" and result_data and "reposition" in result_data:
                repo = result_data["reposition"]
                if "si_offset" in repo:
                    new_si_cursor = int(repo["si_offset"])
                    log("INFO", f"si_cursor → {new_si_cursor}")
                if "ni_offset" in repo:
                    new_ni_cursor = int(repo["ni_offset"])
                    log("INFO", f"ni_cursor → {new_ni_cursor}")
                repositioned = True

            tool_messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

        messages.extend(tool_messages)

        if repositioned:
            out_db.log(bp.id, "INFO", f"Repositioned si={new_si_cursor} ni={new_ni_cursor}")
            return {
                "si_cursor":   new_si_cursor,
                "ni_cursor":   new_ni_cursor,
                "tool_rounds": tool_rounds,
                "status":      "running",
                "pending_translations": [],
            }

        time.sleep(SLEEP_BETWEEN)
        response = _llm_invoke(llm, messages, tag=tag)
        if response is None:
            out_db.log(bp.id, "ERROR",
                       f"LLM call failed mid-tool-loop at si={state['si_cursor']} "
                       f"ni={state['ni_cursor']}. Progress NOT advanced.")
            return {
                "status":    "error",
                "error":     "LLM failed in tool loop",
                "si_cursor": state["si_cursor"],
                "ni_cursor": state["ni_cursor"],
                "pending_translations": [],
            }

    # ── Parse final JSON ────────────────────────────────────────────────────
    raw  = _extract_text(response)
    data = _parse_json(raw)

    if data is None:
        out_db.log(bp.id, "WARN",
                   f"Unparseable at si={state['si_cursor']} ni={state['ni_cursor']}. "
                   f"Forcing +50. Raw[:200]={raw[:200]}")
        log("WARN", f"JSON parse failed — forcing +50 advance")
        return {
            "si_cursor": state["si_cursor"] + 50,
            "ni_cursor": state["ni_cursor"] + 50,
            "status":    "running",
            "pending_translations": [],
        }

    if data.get("status") == "exhausted":
        return {"status": "exhausted", "pending_translations": []}

    translations = data.get("translations", [])
    si_advance   = max(1, int(data.get("si_advance", 50)))
    ni_advance   = max(1, int(data.get("ni_advance", 50)))
    notes        = data.get("notes", "")

    if notes:
        out_db.log(bp.id, "INFO", f"si={state['si_cursor']} ni={state['ni_cursor']}: {notes}")

    log("SAVED",
        f"{Colors.BRIGHT_GREEN}{len(translations)} translations{Colors.RESET} | "
        f"si+{si_advance}  ni+{ni_advance}| {Colors.BRIGHT_YELLOW}{notes}{Colors.RESET}")

    ni_map   = {(s.para_id, s.line_id): s.pali_sentence for s in ni_sentences}
    enriched = []
    for t in translations:
        enriched.append({
            "nissaya_book_id":     t.get("book_id", bp.nissaya_book_id),
            "nissaya_para_id":     t["para_id"],
            "line_id":             t["line_id"],
            "pali_sentence":       ni_map.get((t["para_id"], t["line_id"]), ""),
            "sinhala_translation": t.get("sinhala_translation", ""),
            "confidence":          float(t.get("confidence", 0.8)),
        })

    return {
        "si_cursor":   state["si_cursor"] + si_advance,
        "ni_cursor":   state["ni_cursor"] + ni_advance,
        "status":      "running",
        "pending_translations": enriched,
        "tool_rounds": tool_rounds,
    }


def node_save(state: TranslationState, *, out_db: OutputDB) -> dict:
    bp           = state["book_pair"]
    translations = state.get("pending_translations", [])

    if translations:
        out_db.save_translations(translations)

    if state["status"] == "exhausted":
        db_status = "exhausted"
    elif state["status"] == "error":
        db_status = "in_progress"   # ← do NOT mark finished on LLM failure
    else:
        db_status = "in_progress"

    out_db.save_progress(bp.id, state["si_cursor"], state["ni_cursor"], db_status)
    return {}


# ── Routing ─────────────────────────────────────────────────────────────────────

def _route_after_load(state):
    return "done" if state["status"] == "exhausted" else "translate"

def _route_after_translate(state):
    return "save_done" if state["status"] in ("exhausted", "error") else "save_continue"

def _route_after_save(state):
    return "done" if state["status"] in ("exhausted", "error") else "load_window"


# ── Graph ────────────────────────────────────────────────────────────────────────

def build_graph(si_db: SinhalaDB, ni_db: NissayaDB, out_db: OutputDB, llm):
    bind_databases(si_db, ni_db)

    g = StateGraph(TranslationState)
    g.add_node("load_window", lambda s: node_load_window(s, si_db=si_db, ni_db=ni_db))
    g.add_node("translate",   lambda s: node_translate(s,   llm=llm,    out_db=out_db))
    g.add_node("save",        lambda s: node_save(s,                     out_db=out_db))

    g.set_entry_point("load_window")

    g.add_conditional_edges("load_window", _route_after_load,
                            {"translate": "translate", "done": "save"})
    g.add_conditional_edges("translate", _route_after_translate,
                            {"save_continue": "save", "save_done": "save"})
    g.add_conditional_edges("save", _route_after_save,
                            {"load_window": "load_window", "done": END})

    return g.compile()


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _extract_text(response) -> str:
    if hasattr(response, "content"):
        c = response.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(response)


def _safe_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_json(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None