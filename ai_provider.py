"""
ai_provider.py — Thin abstraction over AI backends.

Adding a new provider:
  1. Subclass AIProvider and implement `complete(prompt, system) -> str`
  2. Register in PROVIDERS dict at bottom of file.
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────

class AIProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, system: str = "", timeout: int = 300) -> str:
        """Return the model's text response or raise an exception."""

    def is_rate_limit_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in ("quota", "rate", "limit", "429", "exhausted", "resource_exhausted"))


# ─────────────────────────────────────────────────────────────────
# Gemini (updated)
# ─────────────────────────────────────────────────────────────────

class Gemini1Provider(AIProvider):
    name = "gemini"
 
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 65536, thinking_budget: int = 2048):
        self.api_key = api_key
        self.model   = model
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
 
    def complete(self, prompt: str, system: str = "", timeout: int = 300) -> str:
        """
        Call Gemini via the google-generativeai SDK with a hard thread-based
        timeout. Gemini can silently hang; the thread kill is the real safety net.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )
 
        genai.configure(api_key=self.api_key)
 
        generation_config = {
            "temperature":     0.7,
            "max_output_tokens": self.max_output_tokens,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
 
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
 
        result_box  = [None]
        error_box   = [None]
        done_event  = threading.Event()
 
        def _call():
            try:
                model = genai.GenerativeModel(
                    self.model,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
                resp = model.generate_content(full_prompt, request_options={"timeout": timeout})
 
                # Log finish reason so truncation is visible in task logs
                finish_reason = None
                try:
                    finish_reason = resp.candidates[0].finish_reason
                except Exception:
                    pass
 
                if finish_reason and str(finish_reason) not in ("FinishReason.STOP", "1", "STOP"):
                    # Still return the partial text — the job's parser will salvage
                    # complete objects. Wrap in a warning prefix that the caller can detect.
                    import logging as _log
                    _log.getLogger(__name__).warning(
                        "Gemini finish_reason=%s (not STOP) — response truncated at %d chars. "
                        "Salvage parser will extract complete terms.",
                        finish_reason, len(resp.text)
                    )
                    result_box[0] = resp.text   # return partial, let parser salvage
                else:
                    result_box[0] = resp.text
 
            except Exception as e:
                error_box[0] = e
            finally:
                done_event.set()
 
        t = threading.Thread(target=_call, daemon=True)
        t.start()
 
        finished = done_event.wait(timeout=timeout + 5)   # 5-sec grace over SDK timeout
        if not finished:
            logger.warning("Gemini call timed out after %ds — thread left as daemon", timeout)
            raise TimeoutError(f"Gemini did not respond within {timeout}s")
 
        if error_box[0]:
            raise error_box[0]
 
        return result_box[0]
     
class GeminiProvider(AIProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 65536, thinking_budget: int = 2048):
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget

    def complete(self, prompt: str, system: str = "", timeout: int = 300) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError("google-genai not installed. Run: pip install google-genai")

        client = genai.Client(api_key=self.api_key)

        config_params = {
            "temperature": 0.7,
            "max_output_tokens": self.max_output_tokens,
            "system_instruction": system if system else None,
        }

        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]
        config_params["safety_settings"] = safety_settings

        # Always set thinking_config explicitly.
        # For non-thinking models, budget=0 disables it entirely.
        # For thinking models, use the configured budget.
        is_thinking_model = "thinking" in self.model.lower()
        if is_thinking_model:
            if self.max_output_tokens <= self.thinking_budget:
                self.max_output_tokens = self.thinking_budget + 4096
                config_params["max_output_tokens"] = self.max_output_tokens
            config_params["thinking_config"] = types.ThinkingConfig(
                include_thoughts=False,       # don't include thought tokens in output
                max_thinking_tokens=self.thinking_budget,
            )
        else:
            # Explicitly disable thinking for models like gemini-2.5-flash
            # that support it but shouldn't use it here (saves quota)
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0,
            )

        result_box = [None]
        error_box  = [None]
        done_event = threading.Event()

        def _call():
            try:
                resp = client.models.generate_content(   # <-- was .generate(), must be .generate_content()
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_params),
                )

                if resp.usage_metadata:
                    u = resp.usage_metadata
                    logger.info(
                        f"Gemini usage: prompt={u.prompt_token_count}, "
                        f"candidates={u.candidates_token_count}, total={u.total_token_count}"
                    )

                text = resp.text or ""

                finish_reason = resp.candidates[0].finish_reason if resp.candidates else "UNKNOWN"
                if str(finish_reason) not in ("FinishReason.STOP", "1", "STOP"):
                    logger.warning(f"Gemini stopped early: {finish_reason}")
                    text += f"\n\n[TERMINATED_BY_{finish_reason}]"

                result_box[0] = text

            except Exception as e:
                error_box[0] = e
            finally:
                done_event.set()

        t = threading.Thread(target=_call, daemon=True)
        t.start()

        if not done_event.wait(timeout=timeout + 5):
            raise TimeoutError(f"Gemini did not respond within {timeout}s")

        if error_box[0]:
            raise error_box[0]

        return result_box[0]
# ─────────────────────────────────────────────────────────────────
# OpenAI (stub — easy to complete later)
# ─────────────────────────────────────────────────────────────────

class OpenAIProvider(AIProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model   = model

    def complete(self, prompt: str, system: str = "", timeout: int = 300) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai not installed. Run: pip install openai")

        client = OpenAI(api_key=self.api_key, timeout=timeout)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content


# ─────────────────────────────────────────────────────────────────
# Anthropic (stub)
# ─────────────────────────────────────────────────────────────────

class AnthropicProvider(AIProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key
        self.model   = model

    def complete(self, prompt: str, system: str = "", timeout: int = 300) -> str:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic not installed. Run: pip install anthropic")

        client  = anthropic.Anthropic(api_key=self.api_key, timeout=timeout)
        kwargs  = {"model": self.model, "max_tokens": 8192,
                   "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system

        msg = client.messages.create(**kwargs)
        return msg.content[0].text


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────

PROVIDERS: dict[str, type[AIProvider]] = {
    "gemini":    GeminiProvider,
    "openai":    OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def build_provider(provider_name: str, api_key: str, model: str) -> AIProvider:
    cls = PROVIDERS.get(provider_name)
    if not cls:
        raise ValueError(f"Unknown provider: {provider_name!r}. Known: {list(PROVIDERS)}")
    return cls(api_key=api_key, model=model)