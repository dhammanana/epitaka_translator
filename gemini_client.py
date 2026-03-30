"""
gemini_client.py — Multi-key Gemini client (google-genai SDK v1+).

Features:
  • Reads all GEMINI_KEY_<N> from .env automatically
  • Per-key RPM throttling  (15 req/min free tier)
  • Per-key RPD tracking    (1 500 req/day free tier)
  • Rotates to next key on 429 / quota errors
  • Waits with countdown when ALL keys are exhausted
  • State persisted to progress/key_state.json (survives restarts)
  • embed_texts()     — batch embedding via gemini-embedding-001
  • embed_query()     — single query embedding (RETRIEVAL_QUERY task)
  • generate()        — text generation with flash model (default)
  • generate_lite()   — text generation with flash-lite + thinking_budget=300
                        Use for structural tasks like chunking that don't need
                        deep reasoning but benefit from a little deliberation.
"""

import sys
import json
import time
import datetime
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types as genai_types
from db_tools import DB_TOOLS, dispatch_tool

from config_tmp import (
    GEMINI_MODEL, GEMINI_MAX_TOKENS,
    GEMINI_FREE_RPM, GEMINI_FREE_RPD,
    GEMINI_EMBED_MODEL, GEMINI_EMBED_DIMENSION, GEMINI_EMBED_BATCH_SIZE,
    QUOTA_EXHAUSTED_WAIT, PROGRESS_DIR,
    get_gemini_keys,
    GEMINI_LITE_MODEL, CHUNKING_THINKING_BUDGET
)



def _log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)


def _is_quota_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(x in s for x in [
        "429", "quota", "resource_exhausted",
        "rate limit", "too many requests", "ratelimitexceeded",
    ])


def _is_server_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(x in s for x in ["500", "503", "unavailable", "internal server"])


class GeminiClient:
    """Thread-safe(-ish) Gemini client with automatic key rotation."""

    STATE_FILE = f"{PROGRESS_DIR}/key_state.json"


    def generate_with_tools(
        self,
        prompt        : str,
        system        : str = "",
        max_tokens    : int = GEMINI_MAX_TOKENS,
        thinking      : int = 0,
        max_tool_calls: int = 10,
    ) -> tuple[str, list[dict]]:
        """
        Agentic generation with automatic DB tool use.
 
        Gemini may call any of the tools declared in DB_TOOLS to browse the
        corpus (list books, read paragraphs, search by keyword, …) before
        writing its final answer.  Tool calls are dispatched automatically;
        the full conversation — including every tool result — is replayed back
        to the model until it produces a final text response.
 
        Parameters
        ----------
        prompt         : User question or instruction.
        system         : Optional system-prompt prefix.
        max_tokens     : Max output tokens for each model turn.
        thinking       : thinking_budget (0 = off, >0 = extended thinking).
        max_tool_calls : Safety limit on total tool calls per request.
 
        Returns
        -------
        (answer_text, tool_call_log)
 
        answer_text   : The model\'s final text response.
        tool_call_log : List of {name, args, result} dicts — one entry per
                        tool call made during the turn, useful for debugging
                        or displaying "sources consulted" in the UI.
        """
        full_prompt = (system.strip() + "\\n\\n" + prompt.strip()).strip() if system else prompt
 
        # Build message history for multi-turn tool loop
        messages = [{"role": "user", "parts": [{"text": full_prompt}]}]
 
        tool_call_log: list[dict] = []
        calls_made = 0
 
        thinking_cfg = genai_types.ThinkingConfig(thinking_budget=thinking if thinking else 0)
        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            thinking_config=thinking_cfg,
            tools=[DB_TOOLS],
            tool_config=genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(mode="AUTO")
            ),
        )
 
        while True:
            # ── pick a key ────────────────────────────────────────────────
            key = self._next_key()
            if key is None:
                self._wait_all_exhausted()
                continue
 
            self._throttle(key)
            client = self._clients[key]
 
            try:
                resp = client.models.generate_content(
                    model   =GEMINI_MODEL,
                    contents=messages,
                    config  =config,
                )
                self._mark_used(key)
            except Exception as e:
                if _is_quota_error(e):
                    self._mark_exhausted(key)
                    continue
                elif _is_server_error(e):
                    import time as _time
                    _time.sleep(30)
                    continue
                else:
                    raise
 
            # ── inspect the response ──────────────────────────────────────
            candidate = resp.candidates[0]
            parts     = candidate.content.parts
 
            # Collect any function calls in this turn
            fn_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]
            text_parts = [p.text for p in parts if hasattr(p, "text") and p.text]
 
            # If no function calls — we have our final answer
            if not fn_calls:
                final_text = "\\n".join(text_parts).strip()
                return final_text, tool_call_log
 
            # Safety guard
            if calls_made >= max_tool_calls:
                _log(f"[TOOLS] Reached max_tool_calls={max_tool_calls}; forcing final answer.")
                return "\\n".join(text_parts).strip(), tool_call_log
 
            # ── append assistant turn with function calls ─────────────────
            messages.append({"role": "model", "parts": [p._raw for p in parts]})
 
            # ── dispatch each tool call and build tool-result parts ────────
            tool_result_parts = []
            for fc_part in fn_calls:
                fc   = fc_part.function_call
                name = fc.name
                args = dict(fc.args or {})
 
                result = dispatch_tool(name, args)
                calls_made += 1
 
                tool_call_log.append({"name": name, "args": args, "result": result})
                _log(f"[TOOLS] {name}({args}) → {str(result)[:120]}")
 
                tool_result_parts.append({
                    "function_response": {
                        "name"    : name,
                        "response": result,
                    }
                })
 
            # ── append tool results as user turn ──────────────────────────
            messages.append({"role": "user", "parts": tool_result_parts})
            # Loop — model will now continue with tool results in context


    def __init__(self):
        self.keys = get_gemini_keys()
        if not self.keys:
            raise ValueError(
                "No Gemini API keys found.\n"
                "Add to .env:\n"
                "  GEMINI_KEY_1=AIza...\n"
                "  GEMINI_KEY_2=AIza..."
            )

        self.n      = len(self.keys)
        self._idx   = 0
        self._state = self._load_state()

        # One genai.Client per key (new SDK style)
        self._clients: dict[str, genai.Client] = {
            k: genai.Client(api_key=k) for k in self.keys
        }

        today = datetime.date.today().isoformat()
        for key in self.keys:
            if key not in self._state or self._state[key].get("date") != today:
                self._state[key] = {
                    "date"         : today,
                    "requests_day" : 0,
                    "exhausted"    : False,
                    "last_request" : 0.0,
                }
        self._save_state()
        _log(f"[GEMINI] {self.n} key(s) loaded. "
             f"Quota: {GEMINI_FREE_RPD} req/day, {GEMINI_FREE_RPM} req/min per key.")

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        Path(PROGRESS_DIR).mkdir(parents=True, exist_ok=True)
        try:
            with open(self.STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(self):
        with open(self.STATE_FILE, "w") as f:
            json.dump(self._state, f, indent=2)

    # ── Key rotation ───────────────────────────────────────────────────────────

    def _is_available(self, key: str) -> bool:
        s = self._state[key]
        today = datetime.date.today().isoformat()
        if s.get("date") != today:
            s["date"]         = today
            s["requests_day"] = 0
            s["exhausted"]    = False
        return not s["exhausted"] and s["requests_day"] < GEMINI_FREE_RPD

    def _next_key(self) -> Optional[str]:
        for _ in range(self.n):
            key       = self.keys[self._idx]
            self._idx = (self._idx + 1) % self.n
            if self._is_available(key):
                return key
        return None

    def _throttle(self, key: str):
        min_gap = 60.0 / GEMINI_FREE_RPM
        elapsed = time.time() - self._state[key].get("last_request", 0.0)
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)

    def _mark_used(self, key: str):
        self._state[key]["requests_day"] += 1
        self._state[key]["last_request"]  = time.time()
        self._save_state()

    def _mark_exhausted(self, key: str):
        self._state[key]["exhausted"] = True
        self._save_state()
        used = self._state[key]["requests_day"]
        _log(f"[GEMINI] Key ...{key[-6:]} exhausted ({used} req today). Rotating.")

    def _wait_all_exhausted(self):
        _log("[GEMINI] ALL keys exhausted — waiting for quota reset ...")
        waited = 0
        while True:
            if self._next_key() is not None:
                return
            remaining = QUOTA_EXHAUSTED_WAIT - (waited % QUOTA_EXHAUSTED_WAIT)
            sys.stdout.write(
                f"\r  All keys exhausted. Rechecking in {remaining}s ..." + " " * 10
            )
            sys.stdout.flush()
            time.sleep(1)
            waited += 1
            if waited % QUOTA_EXHAUSTED_WAIT == 0:
                today = datetime.date.today().isoformat()
                for k in self.keys:
                    if self._state[k].get("date") != today:
                        self._state[k].update({
                            "date": today, "requests_day": 0, "exhausted": False
                        })
                self._save_state()

    # ── Internal generation core ───────────────────────────────────────────────

    def _generate_raw(
        self,
        prompt     : str,
        model      : str,
        max_tokens : int,
        thinking_budget: Optional[int] = None,
        system     : str = "",
    ) -> str:
        """
        Low-level text generation — shared by generate() and generate_lite().
        Handles key rotation, throttling, and retries transparently.
        """
        full_prompt = (system.strip() + "\n\n" + prompt.strip()).strip() if system else prompt

        attempts = 0
        while True:
            key = self._next_key()
            if key is None:
                self._wait_all_exhausted()
                continue

            self._throttle(key)
            client = self._clients[key]

            try:
                # Build thinking config
                if thinking_budget is not None:
                    thinking_cfg = genai_types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                    )
                else:
                    thinking_cfg = genai_types.ThinkingConfig(
                        thinking_budget=0,   # disable thinking entirely
                    )

                config = genai_types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    thinking_config=thinking_cfg,
                )
                resp = client.models.generate_content(
                    model   =model,
                    contents=full_prompt,
                    config  =config,
                )

                self._mark_used(key)

                # Optional debug log
                open('log.txt', 'a+').write(full_prompt + '\n')
                open('log.txt', 'a+').write('-' * 50 + '\n')
                open('log.txt', 'a+').write(resp.text.strip() + '\n')
                open('log.txt', 'a+').write('=' * 50 + '\n\n')

                return resp.text.strip()

            except Exception as e:
                if _is_quota_error(e):
                    self._mark_exhausted(key)
                    attempts += 1
                    continue
                elif _is_server_error(e):
                    wait = min(30 * (attempts + 1), 120)
                    _log(f"[GEMINI] Server error ({e}). Retrying in {wait}s ...")
                    time.sleep(wait)
                    attempts += 1
                    continue
                else:
                    raise

    # ── Public: text generation — flash (default) ──────────────────────────────

    def generate(
        self,
        prompt     : str,
        max_tokens : int = GEMINI_MAX_TOKENS,
        thinking   : int = 0,
        system     : str = "",
    ) -> str:
        """
        Generate text using the primary flash model (GEMINI_MODEL in config).
        `thinking=True` enables extended thinking with an automatic budget.
        Use for enrichment, summarisation, and any task needing strong reasoning.
        """
        return self._generate_raw(
            prompt=prompt,
            model=GEMINI_MODEL,
            max_tokens=max_tokens,
            thinking_budget=thinking,
            system=system,
        )

    # ── Public: text generation — flash-lite (chunking / structural) ───────────

    def generate_lite(
        self,
        prompt     : str,
        max_tokens : int = GEMINI_MAX_TOKENS,
        system     : str = "",
    ) -> str:
        """
        Generate text using flash-lite with a fixed thinking_budget of 300 tokens.

        Ideal for structural / classification tasks that don't need deep reasoning
        but benefit from a small deliberation budget — specifically:
          • Deciding where paragraph boundaries fall (chunking)
          • Classifying semantic roles
          • Any task that must be fast and cheap

        The 300-token thinking budget lets the model briefly plan before answering
        without burning quota on extended chain-of-thought.
        """
        return self._generate_raw(
            prompt=prompt,
            model=GEMINI_LITE_MODEL,
            max_tokens=max_tokens,
            thinking_budget=CHUNKING_THINKING_BUDGET,
            system=system,
        )

    # ── Public: batch embedding ────────────────────────────────────────────────

    def embed_texts(self, texts: list[str],
                    task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """
        Embed a list of texts using gemini-embedding-001.
        Batches to GEMINI_EMBED_BATCH_SIZE; rotates keys on quota errors.
        Returns list of float vectors (dim=768).
        """
        results: list[list[float]] = []
        for start in range(0, len(texts), GEMINI_EMBED_BATCH_SIZE):
            batch = texts[start : start + GEMINI_EMBED_BATCH_SIZE]
            results.extend(self._embed_batch(batch, task_type))
        return results

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        attempts = 0
        while True:
            key = self._next_key()
            if key is None:
                self._wait_all_exhausted()
                continue

            self._throttle(key)
            client = self._clients[key]

            try:
                resp = client.models.embed_content(
                    model   =GEMINI_EMBED_MODEL,
                    contents=texts,
                    config  =genai_types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=GEMINI_EMBED_DIMENSION,
                    ),
                )
                self._mark_used(key)
                # resp.embeddings is a list of ContentEmbedding objects
                return [list(emb.values) for emb in resp.embeddings]

            except Exception as e:
                if _is_quota_error(e):
                    self._mark_exhausted(key)
                    attempts += 1
                    continue
                elif _is_server_error(e):
                    wait = min(30 * (attempts + 1), 120)
                    _log(f"[GEMINI] Embed error ({e}). Retrying in {wait}s ...")
                    time.sleep(wait)
                    attempts += 1
                    continue
                else:
                    raise

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string with RETRIEVAL_QUERY task type."""
        return self._embed_batch([text], task_type="RETRIEVAL_QUERY")[0]

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> list[dict]:
        today = datetime.date.today().isoformat()
        out   = []
        for key in self.keys:
            s    = self._state[key]
            used = s.get("requests_day", 0) if s.get("date") == today else 0
            out.append({
                "key_suffix"    : f"...{key[-8:]}",
                "requests_today": used,
                "remaining"     : max(0, GEMINI_FREE_RPD - used),
                "exhausted"     : s.get("exhausted", False),
            })
        return out