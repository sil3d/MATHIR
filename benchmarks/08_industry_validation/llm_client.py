#!/usr/bin/env python3
"""
Thin OpenAI-compatible chat client for the benchmarks/08_industry_validation/
LongMemEval and LoCoMo runners.

Follows the same style as benchmarks/01_cross_llm_benchmark/benchmark.py's
LLMClient (urllib-based POST to {api_base}/chat/completions with a Bearer
token) and the env-var conventions in benchmarks/.env.example:
MATHIR_LLM_BACKEND (auto/openrouter/api), MATHIR_API_KEY, MATHIR_API_BASE,
MATHIR_API_MODEL -- also honors the standard OPENAI_API_KEY / OPENAI_BASE_URL
/ OPENAI_MODEL env vars if already set globally in the shell (a real,
already-working gateway, e.g. one serving MiniMax or any other real model),
so a working setup outside the MATHIR_* scheme isn't silently ignored.

No local Ollama backend: it's not fast enough for benchmark-scale runs
(hundreds to thousands of LLM calls), so this module only ever talks to a
real hosted API (OpenRouter, OpenCode Zen, MiniMax's native endpoint, or
any other OpenAI-compatible provider via MATHIR_API_BASE/OPENAI_BASE_URL).

No model name is ever hardcoded as a fallback: if a backend resolves an
api_base/api_key but no model is set anywhere (env var or `model=` override),
`chat()` raises a clear RuntimeError telling you which env var to set,
rather than silently substituting an unrelated model name.

IMPORTANT for comparability with published Mem0/Zep LongMemEval/LoCoMo
scores: those benchmarks were originally scored using GPT-4o as the judge
model. If you want numbers that are directly apples-to-apples with the
published competitor numbers, you need an OpenAI API key and must force the
judge model to gpt-4o. Using a different judge model (e.g. a free
OpenRouter model) still gives a real, internally-consistent signal for
comparing MATHIR against itself over time, but is NOT apples-to-apples with
published competitor scores.

Set MATHIR_BENCHMARK_JUDGE_MODEL to force a specific model for judge calls
(e.g. "gpt-4o") while leaving MATHIR_API_MODEL / the backend default in
place for answer-generation calls. Default is empty, meaning "use whatever
the resolved backend/model normally resolves to" -- callers decide which
model to request per `chat()` call via the `model` parameter if they need
per-call overrides (e.g. judge vs. generation); this module doesn't
special-case that beyond exposing the env var and reading it where the
caller asks for it.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Auto-load .env from benchmarks/ root via shared helper (same pattern as
# benchmarks/01_cross_llm_benchmark/benchmark.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import _env  # noqa: F401 -- populates os.environ
except ImportError:
    pass


def _resolve_backend_config(model_override: str | None = None) -> tuple[str, str, str]:
    """Resolve (api_base, api_key, model) from MATHIR_LLM_BACKEND + env vars.

    Mirrors the "auto/openrouter/api" convention documented in
    benchmarks/.env.example, but ALSO checks the standard OPENAI_API_KEY /
    OPENAI_BASE_URL / OPENAI_MODEL env vars first -- a real global gateway
    (any OpenAI-compatible endpoint, e.g. a proxy serving MiniMax or other
    real models) set directly in the shell, not just the bespoke MATHIR_*
    scheme that requires a separate benchmarks/.env file. Without this, a
    real configured OPENAI_API_KEY was being silently ignored.

    No Ollama backend -- deliberately not fast enough for benchmark-scale
    runs (hundreds to thousands of calls per full LongMemEval/LoCoMo pass).
    """
    backend = os.environ.get("MATHIR_LLM_BACKEND", "auto").strip().lower()

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_base = os.environ.get("OPENAI_BASE_URL", "")
    openai_model = os.environ.get("OPENAI_MODEL", "")

    # MATHIR_API_KEY is a deliberately generic name -- it's THIS repo's
    # internal env var for "whichever OpenAI-compatible key you've pointed
    # MATHIR_API_BASE at", not a provider name. But that's genuinely
    # confusing (a user asked "why MATHIR_API_KEY, I'm not a provider,
    # give me the real name" -- fair complaint), and benchmarks/.env.example
    # already defines real, named per-provider key slots
    # (OPENCODE_ZEN_KEY, MINIMAX_API_KEY, GOOGLE_AI_STUDIO_KEY, NVIDIA_API_KEY)
    # for 01_cross_llm_benchmark that were silently NOT read here before --
    # so pasting a key into e.g. OPENCODE_ZEN_KEY had no effect on this
    # module. Fixed: fall back through every real provider-named key too,
    # in addition to the generic MATHIR_API_KEY/OPENROUTER_API_KEY, so
    # whichever named slot you actually fill in just works.
    mathir_key = (
        os.environ.get("MATHIR_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENCODE_ZEN_KEY")
        or os.environ.get("MINIMAX_API_KEY")
        or os.environ.get("GOOGLE_AI_STUDIO_KEY")
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or ""
    )

    if backend == "auto":
        if openai_key:
            backend = "openai_env"
        elif mathir_key:
            backend = "openrouter"
        else:
            raise RuntimeError(
                "No LLM backend configured: set OPENAI_API_KEY/OPENAI_BASE_URL/"
                "OPENAI_MODEL (a real gateway, e.g. MiniMax's native API or "
                "OpenCode Zen), or MATHIR_API_KEY/MATHIR_API_BASE/MATHIR_API_MODEL "
                "in benchmarks/.env. There is no local fallback."
            )

    if backend == "openai_env":
        # A real, already-working OpenAI-compatible endpoint set globally
        # in the shell -- use it as-is, model name included, rather than
        # substituting a hardcoded guess.
        api_base = openai_base or "https://api.openai.com/v1"
        model = model_override or openai_model
        if not model:
            raise RuntimeError(
                "No model resolved for the 'openai_env' backend: OPENAI_API_KEY is "
                "set but OPENAI_MODEL is not, and no model override was passed. "
                "Set OPENAI_MODEL to the real model your gateway/provider serves "
                "(e.g. a MiniMax or OpenRouter model id) -- this code does not "
                "guess a model name for you."
            )
        return api_base, openai_key, model

    if backend == "openrouter":
        api_base = os.environ.get("MATHIR_API_BASE") or "https://openrouter.ai/api/v1"
        model = model_override or os.environ.get("MATHIR_API_MODEL")
        if not model:
            raise RuntimeError(
                "No model resolved for the 'openrouter' backend: set MATHIR_API_MODEL "
                "to a real OpenRouter model id (e.g. a MiniMax model, or any current "
                "free-tier model from https://openrouter.ai/models) -- this code does "
                "not guess a model name for you."
            )
        return api_base, mathir_key, model

    # backend == "api": direct OpenAI-compatible API. Fall back through
    # MATHIR_* first (explicit opt-in wins), then the real OPENAI_* env --
    # never silently drop a real key/model that's actually set, and never
    # substitute a hardcoded guess if neither is set.
    api_base = os.environ.get("MATHIR_API_BASE") or openai_base
    if not api_base:
        raise RuntimeError(
            "No api_base resolved for the 'api' backend: set MATHIR_API_BASE or "
            "OPENAI_BASE_URL to your provider's endpoint."
        )

    # Match the key to the actual configured api_base rather than picking
    # the first key that happens to be set in a fixed priority order. With
    # several provider keys configured at once (OpenCode Zen, MiniMax,
    # OpenRouter, ...), a base-independent fallback chain can silently pair
    # e.g. an OpenRouter key with an OpenCode Zen endpoint -- a real bug
    # that was caught here (401/403 HTTP errors, not a config-missing
    # situation) before this fix.
    explicit_mathir_key = os.environ.get("MATHIR_API_KEY")
    if explicit_mathir_key:
        api_key = explicit_mathir_key
    elif "opencode.ai" in api_base:
        api_key = os.environ.get("OPENCODE_ZEN_KEY") or mathir_key or openai_key
    elif "minimax.io" in api_base:
        api_key = os.environ.get("MINIMAX_API_KEY") or mathir_key or openai_key
    elif "openrouter.ai" in api_base:
        api_key = os.environ.get("OPENROUTER_API_KEY") or mathir_key or openai_key
    elif "groq.com" in api_base:
        api_key = os.environ.get("GROQ_API_KEY") or mathir_key or openai_key
    else:
        api_key = mathir_key or openai_key

    model = model_override or os.environ.get("MATHIR_API_MODEL") or openai_model
    if not model:
        raise RuntimeError(
            "No model resolved for the 'api' backend: set MATHIR_API_MODEL or "
            "OPENAI_MODEL to the real model your provider serves -- this code "
            "does not guess a model name for you."
        )
    return api_base, api_key, model


def chat(messages: list, temperature: float = 0.0, max_tokens: int = 1024, model: str = None) -> str:
    """Send an OpenAI-compatible chat completion request.

    Uses whichever backend benchmarks/.env configures (MATHIR_LLM_BACKEND).
    `model` lets a caller override the resolved model for this one call
    (e.g. pass os.environ.get("MATHIR_BENCHMARK_JUDGE_MODEL") for judge
    calls, or leave None to use the backend's normal default).

    Returns the assistant's text content. Raises RuntimeError with the
    response body on failure -- never silently returns an empty string.
    """
    api_base, api_key, resolved_model = _resolve_backend_config(model_override=model)
    url = f"{api_base.rstrip('/')}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "MATHIR-Benchmark/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    referer = os.environ.get("MATHIR_OPENROUTER_REFERER")
    title = os.environ.get("MATHIR_OPENROUTER_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    payload = json.dumps(
        {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")

    import time as _time

    max_retries = 3
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code in (429, 413) and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"[{e.code} rate-limit, waiting {wait}s] ", end="", flush=True)
                _time.sleep(wait)
                continue
            raise RuntimeError(
                f"LLM chat completion failed: HTTP {e.code} from {url} (model={resolved_model}): {body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"LLM chat completion failed: could not reach {url} ({e})"
            ) from e

    try:
        data = json.loads(body)
        raw = data["choices"][0]["message"]["content"]
        # Strip <thinking>...</thinking> tags that reasoning models emit.
        # Handles both the full block form and any leftover content.
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        # If cleaning removed everything (no thinking tags found), return original
        return cleaned if cleaned else raw
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        raise RuntimeError(
            f"LLM chat completion returned an unparseable response from {url} "
            f"(model={resolved_model}): {body!r}"
        ) from e
