#!/usr/bin/env python3
"""
Thin OpenAI-compatible chat client for the benchmarks/08_industry_validation/
LongMemEval and LoCoMo runners.

Follows the same style as benchmarks/01_cross_llm_benchmark/benchmark.py's
LLMClient (urllib-based POST to {api_base}/chat/completions with a Bearer
token) and the env-var conventions in benchmarks/.env.example:
MATHIR_LLM_BACKEND (auto/openrouter/ollama/api), MATHIR_API_KEY,
MATHIR_API_BASE, MATHIR_API_MODEL -- also honors the standard OPENAI_API_KEY
/ OPENAI_BASE_URL / OPENAI_MODEL env vars if already set globally in the
shell (a real, already-working gateway, e.g. one serving MiniMax or any
other real model), so a working setup outside the MATHIR_* scheme isn't
silently ignored.

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

    Mirrors the "auto/openrouter/ollama/api" convention documented in
    benchmarks/.env.example, but ALSO checks the standard OPENAI_API_KEY /
    OPENAI_BASE_URL / OPENAI_MODEL env vars first -- a real global gateway
    (any OpenAI-compatible endpoint, e.g. a proxy serving MiniMax or other
    real models) set directly in the shell, not just the bespoke MATHIR_*
    scheme that requires a separate benchmarks/.env file. Without this, a
    real configured OPENAI_API_KEY was being silently ignored and "auto"
    fell back to a local Ollama server that usually isn't running.
    """
    backend = os.environ.get("MATHIR_LLM_BACKEND", "auto").strip().lower()

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_base = os.environ.get("OPENAI_BASE_URL", "")
    openai_model = os.environ.get("OPENAI_MODEL", "")

    mathir_key = os.environ.get("MATHIR_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")

    if backend == "auto":
        if openai_key:
            backend = "openai_env"
        elif mathir_key:
            backend = "openrouter"
        else:
            backend = "ollama"

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

    if backend == "ollama":
        api_base = os.environ.get("MATHIR_OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/v1"
        model = model_override or os.environ.get("MATHIR_OLLAMA_MODEL")
        if not model:
            raise RuntimeError(
                "No model resolved for the 'ollama' backend: set MATHIR_OLLAMA_MODEL "
                "to whichever model you've actually pulled into your local Ollama "
                "server -- this code does not guess a model name for you."
            )
        return api_base, "ollama", model

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

    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "ollama":
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

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM chat completion failed: HTTP {e.code} from {url} (model={resolved_model}): {body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"LLM chat completion failed: could not reach {url} ({e})"
        ) from e

    try:
        data = json.loads(body)
        return data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        raise RuntimeError(
            f"LLM chat completion returned an unparseable response from {url} "
            f"(model={resolved_model}): {body!r}"
        ) from e
