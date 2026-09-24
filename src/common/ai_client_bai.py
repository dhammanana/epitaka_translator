"""
ai_client_bai.py — Minimal B.AI API client.

Drop-in alternative to the Gemini-based ai_client.py, but deliberately
stripped down per request: ONE model, ONE key, no swapping.

Everything that made ai_client.py complicated is gone on purpose:
  - No GEMINI_KEY_1..N rotation, no per-key rpm/tpm throttling, no
    file-backed rotation state store, no Telegram-on-exhaustion.
  - No model selection logic. This client only ever calls
    "deepseek-v4-flash" (overridable via B_AI_MODEL, but there's nowhere
    in this file that swaps models on failure — a bad call just fails).

If a call fails it retries a few times with backoff against the SAME key
and the SAME model, then gives up and returns None. That's the entire
failure-handling story.

Endpoint used: POST {B_AI_BASE_URL}/v1/messages (Claude-compatible), per the
B.AI API reference — chosen over /v1/chat/completions specifically because
deepseek-v4-flash is a REASONING model, and the Messages endpoint is the
only one in the B.AI docs that exposes an explicit `"thinking": {"type":
"disabled"}` switch. Without that switch, the model was observed spending
its entire max_tokens budget on internal reasoning_content and returning
an empty final answer (finish_reason="length") on the chat/completions
endpoint — raising max_tokens doesn't fix that, turning thinking off does.

Env vars
--------
B_AI_API_KEY   Required. The single B.AI API key to use.
B_AI_BASE_URL  Optional. Default "https://api.b.ai".
B_AI_MODEL     Optional. Default "deepseek-v4-flash".
BT_DEBUG       Optional. "1" turns on verbose per-call debug logging
               (full request params, raw usage dict) — same convention as
               context_builders.py's BT_DEBUG flag. NOT required for timing:
               see below.

Logging
-------
Three independent log prefixes, all to stdout:
  [bai-debug]   Gated by BT_DEBUG=1. Verbose — request params, raw usage.
  [bai-diag]    Always on. Failure-path detail (why an attempt failed).
  [bai-timing]  Always on. One line per attempt (HTTP status + elapsed
                seconds) and one line per call (total elapsed, tokens/sec,
                which attempt it succeeded on, or total time before giving
                up). This is what to grep for latency/slowness questions —
                it does NOT require BT_DEBUG.
  [bai-stats]   Always on. Printed after every call_ai() call (success or
                failure): running totals for this process — call counts,
                attempt counts broken down by outcome (timeout / network
                error / 429 / 5xx / empty-due-to-max_tokens), cumulative
                wall time (total / success-only / failed-only), and
                cumulative input/output tokens. Use this to tell, over a
                whole run, whether time is going to genuinely slow (but
                successful) calls or to retried failures — call
                get_session_stats() to grab the same numbers programmatically
                (e.g. to log one summary line per book).

Public API
----------
call_ai(system_prompt, user_prompt, **kw) -> str | None
call_ai_with_logging(prompt=..., system_prompt=..., **kw) -> str | None
    Same as call_ai(), plus writes a per-call prompt/response .log file
    when log_dir is given. Accepts (and ignores) a `rotator` kwarg purely
    so scripts written against the old ai_client.py call signature don't
    need to change their call sites.
parse_ai_json_response(raw, expected_keys=()) -> dict
list_models() -> list[dict]
    Quick sanity check that B_AI_API_KEY works (GET /v1/models).
get_session_stats() -> dict
    Snapshot of the cumulative [bai-stats] counters for this process.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("B_AI_BASE_URL", "https://api.b.ai").rstrip("/")
MODEL    = os.environ.get("B_AI_MODEL", "deepseek-v4-flash")
API_KEY  = os.environ.get("B_AI_API_KEY", "")

_MESSAGES_ENDPOINT = f"{BASE_URL}/v1/messages"
_MODELS_ENDPOINT   = f"{BASE_URL}/v1/models"

_DEBUG = os.environ.get("BT_DEBUG", "0") == "1"


def _debug(msg: str) -> None:
    if _DEBUG:
        print(f"[bai-debug] {msg}")


def _diag(msg: str) -> None:
    """Diagnostic output for failure paths — always printed (not gated by
    BT_DEBUG), since 'the call failed' is exactly when you need to see why,
    not only when you happened to already have verbose mode on."""
    print(f"[bai-diag] {msg}")


def _ts() -> str:
    """Wall-clock timestamp for log lines, so slow stretches can be lined up
    against external events (rate-limit windows, network blips, etc.)."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _timing(msg: str) -> None:
    """Always-on timing/latency log — separate from _debug (verbose, opt-in)
    and _diag (failure-path only). This one prints for every attempt and
    every call, success or failure, specifically so call latency is visible
    without needing BT_DEBUG on."""
    print(f"[bai-timing] {_ts()} {msg}")


# ══════════════════════════════════════════════════════════════════
# Running session stats — cumulative across every call_ai() in this
# process, so you can see at a glance (without grepping timestamps)
# whether slowness this run is dominated by timeouts, 429s, 5xx, or
# genuinely slow-but-successful calls.
# ══════════════════════════════════════════════════════════════════

_stats = {
    "calls_started": 0,
    "calls_succeeded": 0,
    "calls_failed": 0,
    "attempts_total": 0,
    "attempts_timeout": 0,
    "attempts_network_error": 0,
    "attempts_429": 0,
    "attempts_5xx": 0,
    "attempts_empty": 0,
    "attempts_max_tokens": 0,
    "wall_time_total": 0.0,      # sum of call_ai() durations (all attempts + backoff)
    "wall_time_success": 0.0,    # only calls that eventually returned text
    "wall_time_failed": 0.0,     # only calls that gave up / raised
    "input_tokens_total": 0,
    "output_tokens_total": 0,
}


def _print_session_summary() -> None:
    s = _stats
    avg_call = s["wall_time_total"] / s["calls_started"] if s["calls_started"] else 0.0
    print(
        f"[bai-stats] {_ts()} "
        f"calls={s['calls_started']} (ok={s['calls_succeeded']} failed={s['calls_failed']})  "
        f"attempts={s['attempts_total']} "
        f"(timeout={s['attempts_timeout']} net_err={s['attempts_network_error']} "
        f"429={s['attempts_429']} 5xx={s['attempts_5xx']} "
        f"empty_max_tokens={s['attempts_max_tokens']})  "
        f"time: total={s['wall_time_total']:.1f}s success={s['wall_time_success']:.1f}s "
        f"failed={s['wall_time_failed']:.1f}s avg/call={avg_call:.1f}s  "
        f"tokens: in={s['input_tokens_total']} out={s['output_tokens_total']}"
    )


def get_session_stats() -> dict:
    """Snapshot of the running totals, for scripts that want to log a final
    summary themselves (e.g. verify_translation.py at end of a book/run)."""
    return dict(_stats)


class BaiError(RuntimeError):
    """Raised when a B.AI call fails in a way that isn't worth retrying."""


def _require_api_key() -> str:
    if not API_KEY:
        raise BaiError(
            "B_AI_API_KEY is not set. Export it, or put it in your .env file. "
            "This client intentionally supports only a single key — no rotation."
        )
    return API_KEY


# ══════════════════════════════════════════════════════════════════
# Core call
# ══════════════════════════════════════════════════════════════════

def call_ai(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = MODEL,
    max_tokens: int = 6000,
    thinking_budget_tokens: int = 2000,
    temperature: float = 0.2,
    json_mode: bool = True,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
    timeout: float = 120.0,
    debug_dump_path: Optional[str] = None,
) -> Optional[str]:
    """
    One call to B.AI's /v1/messages (Claude-compatible) endpoint using the
    fixed model and the single configured API key. Retries on network
    errors / 429 / 5xx with linear backoff; does NOT retry (fails fast) on
    4xx client errors other than 429, since those mean the request itself
    is wrong and retrying won't help.

    Returns the assistant's final answer text (the `text` content block,
    not the thinking block), or None if every retry was exhausted.

    thinking_budget_tokens caps — but does not disable — deepseek-v4-flash's
    reasoning. Left unbounded, this model was observed spending its ENTIRE
    max_tokens budget on internal reasoning and returning an empty final
    answer (finish_reason="length"). Passing thinking={"type": "enabled",
    "budget_tokens": thinking_budget_tokens} keeps reasoning switched on but
    puts a ceiling on it, so max_tokens - thinking_budget_tokens is always
    left over for the actual answer. Must satisfy
    1024 <= thinking_budget_tokens < max_tokens (enforced below).

    json_mode is accepted for call-signature compatibility with the old
    chat/completions-based version of this client, but does nothing here:
    the Messages API has no response_format param. JSON output is enforced
    purely by prompt instructions (every system prompt in this codebase
    already says "return only JSON"), and parse_ai_json_response() below
    is tolerant of stray markdown fences around the JSON.

    `debug_dump_path`, if given, gets the FULL raw response body written to
    it on every failed attempt (overwritten each attempt, so it always
    holds the most recent one).
    """
    api_key = _require_api_key()

    if thinking_budget_tokens < 1024:
        thinking_budget_tokens = 1024
    if thinking_budget_tokens >= max_tokens:
        # Leave at least ~1500 tokens of headroom for the actual answer —
        # a translation-verification JSON payload needs real room.
        max_tokens = thinking_budget_tokens + 1500

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "thinking": {"type": "enabled", "budget_tokens": thinking_budget_tokens},
    }

    prompt_chars = len(system_prompt) + len(user_prompt)
    est_input_tokens = prompt_chars // 4  # rough chars->tokens guess; real count comes back in usage

    call_t0 = time.perf_counter()
    _stats["calls_started"] += 1
    _timing(
        f"call start  model={model}  max_tokens={max_tokens} thinking_budget={thinking_budget_tokens}  "
        f"prompt_chars={prompt_chars} (~{est_input_tokens} est. input tokens)  "
        f"timeout={timeout}s  max_retries={max_retries}"
    )
    _debug(
        f"POST {_MESSAGES_ENDPOINT}  model={model}  max_tokens={max_tokens} "
        f"thinking_budget_tokens={thinking_budget_tokens}  temperature={temperature}  "
        f"prompt_chars={prompt_chars}"
    )

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        attempt_t0 = time.perf_counter()
        try:
            resp = requests.post(_MESSAGES_ENDPOINT, headers=headers, json=body, timeout=timeout)
            attempt_elapsed = time.perf_counter() - attempt_t0
            _stats["attempts_total"] += 1
            _timing(f"attempt {attempt}/{max_retries}: HTTP {resp.status_code} in {attempt_elapsed:.2f}s")

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as exc:
                    _diag(f"attempt {attempt}: HTTP 200 but body isn't valid JSON after {attempt_elapsed:.2f}s: {exc}")
                    _diag(f"attempt {attempt}: raw body (first 1000 chars): {resp.text[:1000]!r}")
                    if debug_dump_path:
                        _dump_raw(debug_dump_path, resp.text)
                    last_exc = BaiError(f"Non-JSON 200 response: {exc}")
                    if attempt < max_retries:
                        time.sleep(retry_backoff * attempt)
                    continue

                content_blocks = data.get("content") or []
                stop_reason = data.get("stop_reason")
                usage = data.get("usage") or {}
                in_tok = usage.get("input_tokens")
                out_tok = usage.get("output_tokens")

                text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
                thinking_parts = [b.get("thinking", "") for b in content_blocks if b.get("type") == "thinking"]
                text = "".join(text_parts).strip()

                _debug(
                    f"usage: input={in_tok} output={out_tok} "
                    f"stop_reason={stop_reason}  block_types={[b.get('type') for b in content_blocks]}"
                )

                if text:
                    call_elapsed = time.perf_counter() - call_t0
                    _stats["calls_succeeded"] += 1
                    _stats["wall_time_total"] += call_elapsed
                    _stats["wall_time_success"] += call_elapsed
                    if isinstance(in_tok, int):
                        _stats["input_tokens_total"] += in_tok
                    if isinstance(out_tok, int):
                        _stats["output_tokens_total"] += out_tok
                    tok_per_s = (out_tok / attempt_elapsed) if (out_tok and attempt_elapsed > 0) else 0.0
                    _timing(
                        f"call OK in {call_elapsed:.2f}s (succeeded on attempt {attempt}/{max_retries})  "
                        f"input_tokens={in_tok} output_tokens={out_tok} (~{tok_per_s:.0f} tok/s)  "
                        f"stop_reason={stop_reason}"
                    )
                    _print_session_summary()
                    return text

                # Empty final text — most likely stop_reason == "max_tokens" and
                # the thinking block ate the whole budget anyway (raise
                # thinking_budget_tokens's headroom or max_tokens further).
                _stats["attempts_empty"] += 1
                _diag(
                    f"attempt {attempt}: HTTP 200, empty text content after {attempt_elapsed:.2f}s. "
                    f"stop_reason={stop_reason!r}  content block types={[b.get('type') for b in content_blocks]}  "
                    f"thinking_chars={sum(len(t) for t in thinking_parts)}  usage={usage}"
                )
                if stop_reason == "max_tokens":
                    _stats["attempts_max_tokens"] += 1
                    _diag(
                        f"attempt {attempt}: hit max_tokens={max_tokens} before finishing the answer. "
                        f"thinking_budget_tokens={thinking_budget_tokens} leaves only "
                        f"{max_tokens - thinking_budget_tokens} tokens of headroom for the answer — "
                        f"raise max_tokens (or lower thinking_budget_tokens) if this keeps happening."
                    )
                if debug_dump_path:
                    _dump_raw(debug_dump_path, json.dumps(data, ensure_ascii=False, indent=2))
                last_exc = BaiError(f"Empty text content in B.AI response (stop_reason={stop_reason!r}).")

            elif resp.status_code == 429:
                _stats["attempts_429"] += 1
                _diag(f"attempt {attempt}: rate limited (429) after {attempt_elapsed:.2f}s: {resp.text[:500]}")
                last_exc = BaiError(f"429 rate limited: {resp.text[:300]}")

            elif resp.status_code in (500, 502, 503):
                _stats["attempts_5xx"] += 1
                _diag(f"attempt {attempt}: upstream error {resp.status_code} after {attempt_elapsed:.2f}s: {resp.text[:500]}")
                last_exc = BaiError(f"{resp.status_code}: {resp.text[:300]}")

            else:
                # 400 / 401 / 403 / etc — the request itself is wrong (bad key,
                # bad model name, bad params). Retrying won't fix that.
                _diag(f"attempt {attempt}: non-retryable error {resp.status_code} after {attempt_elapsed:.2f}s: {resp.text[:1500]}")
                if debug_dump_path:
                    _dump_raw(debug_dump_path, f"HTTP {resp.status_code}\n{resp.text}")
                call_elapsed = time.perf_counter() - call_t0
                _stats["calls_failed"] += 1
                _stats["wall_time_total"] += call_elapsed
                _stats["wall_time_failed"] += call_elapsed
                _print_session_summary()
                raise BaiError(f"B.AI request failed ({resp.status_code}): {resp.text[:500]}")

        except requests.exceptions.Timeout as exc:
            # Broken out from the generic RequestException branch below so
            # timeouts (the dominant failure mode against api.b.ai in
            # practice) are counted and logged separately from DNS/connection
            # resets and the like — that split is exactly what tells you
            # whether it's "the model is slow" vs "the network/gateway is
            # flaky" when you're staring at a wall of retries.
            attempt_elapsed = time.perf_counter() - attempt_t0
            _stats["attempts_total"] += 1
            _stats["attempts_timeout"] += 1
            _diag(f"attempt {attempt}/{max_retries}: TIMEOUT after {attempt_elapsed:.2f}s (limit was {timeout}s): {exc}")
            last_exc = exc

        except requests.RequestException as exc:
            attempt_elapsed = time.perf_counter() - attempt_t0
            _stats["attempts_total"] += 1
            _stats["attempts_network_error"] += 1
            _diag(f"attempt {attempt}/{max_retries}: network error after {attempt_elapsed:.2f}s: {exc}")
            last_exc = exc

        if attempt < max_retries:
            sleep_s = retry_backoff * attempt
            _timing(f"retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    call_elapsed = time.perf_counter() - call_t0
    _stats["calls_failed"] += 1
    _stats["wall_time_total"] += call_elapsed
    _stats["wall_time_failed"] += call_elapsed
    _timing(f"call FAILED after {max_retries} attempt(s), total {call_elapsed:.2f}s")
    _print_session_summary()
    logger.warning(f"[ai_client_bai] call_ai failed after {max_retries} attempt(s): {last_exc}")
    return None


def _dump_raw(path: str, content: str) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as exc:
        logger.warning(f"[ai_client_bai] Could not write debug dump to {path}: {exc}")


def call_ai_with_logging(
    *,
    prompt: str,
    system_prompt: str,
    book_id: str = "",
    chunk_id: str = "",
    log_dir: str = "",
    model: str = MODEL,
    max_tokens: int = 6000,
    thinking_budget_tokens: int = 2000,
    temperature: float = 0.2,
    json_mode: bool = True,
    rotator=None,  # accepted-and-ignored: kept only for call-signature
                   # compatibility with code written against ai_client.py.
                   # This client has exactly one key; it never rotates.
) -> Optional[str]:
    """
    Same as call_ai(), plus writes the prompt/response pair to
    {log_dir}/{book_id}_{chunk_id}.log when log_dir is given.
    """
    if rotator is not None:
        _debug("call_ai_with_logging() received a `rotator` argument — ignoring it "
               "(ai_client_bai has a single key and never rotates).")

    debug_dump_path = None
    if log_dir:
        debug_dump_path = str(Path(log_dir) / f"{book_id or 'nobook'}_{chunk_id or 'nochunk'}.raw_response.json")

    raw = call_ai(
        system_prompt=system_prompt,
        user_prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
        temperature=temperature,
        json_mode=json_mode,
        debug_dump_path=debug_dump_path,
    )

    if log_dir:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(log_dir) / f"{book_id or 'nobook'}_{chunk_id or 'nochunk'}.log"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\nSYSTEM PROMPT\n" + "=" * 60 + "\n")
                f.write(system_prompt + "\n\n")
                f.write("=" * 60 + "\nUSER PROMPT\n" + "=" * 60 + "\n")
                f.write(prompt + "\n\n")
                f.write("=" * 60 + "\nRESPONSE\n" + "=" * 60 + "\n")
                f.write((raw or "(no response — see .raw_response.json next to this file)") + "\n")
        except Exception as exc:
            logger.warning(f"[ai_client_bai] Could not write log to {log_dir}: {exc}")

    if raw is None and debug_dump_path and not Path(debug_dump_path).exists():
        # Every attempt failed before we ever got a 200-with-empty-content or
        # a non-retryable error body to dump (e.g. pure network errors) —
        # leave a note so it's clear there's nothing more to inspect.
        _dump_raw(debug_dump_path, "(no response body was ever received — check network/timeout errors above)")

    return raw


# ══════════════════════════════════════════════════════════════════
# Response parsing
# ══════════════════════════════════════════════════════════════════

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_ai_json_response(raw: Optional[str], expected_keys: tuple[str, ...] = ()) -> dict:
    """
    Parse the model's raw text into a dict, tolerating the common ways
    models decorate JSON (markdown fences, stray prose around the object).

    Raises ValueError if no valid JSON object can be recovered, or if
    `expected_keys` is non-empty and none of those keys are present.
    """
    if raw is None:
        raise ValueError("No response text to parse (raw is None).")

    text = raw.strip()
    text = _JSON_FENCE_RE.sub("", text).strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse JSON from AI response: {exc}\n"
            f"Raw (first 300 chars): {raw[:300]!r}"
        )

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}.")

    if expected_keys and not any(k in data for k in expected_keys):
        raise ValueError(
            f"None of expected keys {expected_keys} found in response: {list(data.keys())}"
        )

    return data


# ══════════════════════════════════════════════════════════════════
# Misc
# ══════════════════════════════════════════════════════════════════

def list_models() -> list[dict]:
    """GET /v1/models — quick sanity check that B_AI_API_KEY works."""
    api_key = _require_api_key()
    resp = requests.get(_MODELS_ENDPOINT, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    resp.raise_for_status()
    return (resp.json() or {}).get("data", [])


if __name__ == "__main__":
    # `python ai_client_bai.py` — smoke test: list models, then one tiny call.
    logging.basicConfig(level=logging.INFO)
    print(f"[ai_client_bai] BASE_URL={BASE_URL}  MODEL={MODEL}  "
          f"API_KEY={'set' if API_KEY else 'MISSING'}")
    try:
        models = list_models()
        ids = [m.get("id") for m in models]
        print(f"[ai_client_bai] {len(models)} model(s) available. "
              f"deepseek-v4-flash present: {MODEL in ids}")
    except Exception as exc:
        print(f"[ai_client_bai] list_models() failed: {exc}")

    out = call_ai(
        system_prompt="Reply with only JSON: {\"ok\": true}",
        user_prompt="ping",
        max_tokens=50,
    )
    print(f"[ai_client_bai] test call raw response: {out!r}")
