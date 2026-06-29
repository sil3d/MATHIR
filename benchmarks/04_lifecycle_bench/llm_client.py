"""
LLM client for the AI-driven cognitive benchmark.

NEVER embeds API keys. Reads from environment at call time.
Falls back to local Ollama if no API key is set.

Environment variables (read at runtime, never stored in this file):
  MATHIR_LLM_BACKEND  : "api" | "openrouter" | "ollama" | "auto" (default: auto)
  MATHIR_API_KEY      : API key for the remote provider
  MATHIR_API_BASE     : API base URL (default: https://api.minimax.chat/v1)
  MATHIR_API_MODEL    : Model name (default: MiniMax-M2.7)
  MATHIR_OPENROUTER_REFERER : Optional. Site URL for OpenRouter ranking.
  MATHIR_OPENROUTER_TITLE   : Optional. Site title for OpenRouter ranking.
  MATHIR_OLLAMA_URL   : Ollama base URL (default: http://127.0.0.1:11434)
  MATHIR_OLLAMA_MODEL : Ollama model (default: qwen3.5:2b)

Loading priority (first non-empty wins):
  1. Real environment variables ($env: in PowerShell, export in bash)
  2. .env file in benchmarks/04_lifecycle_bench/  (auto-loaded if present)
  3. Built-in defaults

Backend auto-resolution (when MATHIR_LLM_BACKEND=auto):
  - If MATHIR_API_KEY is set -> "api" (uses MATHIR_API_BASE/MODEL)
  - Else -> "ollama"

For OpenRouter specifically, recommended free model:
  meta-llama/llama-3.3-70b-instruct:free
"""
import os
import json
import time
import urllib.request
import urllib.error
from typing import Optional, List, Dict


# Auto-load .env from benchmarks/ root via shared helper
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import _env as _env_loader  # noqa: F401
except ImportError:
    pass  # _env.py missing — fall back to no .env loading


class LLMUnavailable(RuntimeError):
    """Raised when neither API nor Ollama is reachable."""


# OpenRouter free model presets (verified working on 2026-06-23)
# 26 free models tested, 9 actually respond. Models that return HTTP 200
# with content=null are multimodal-only (vision/audio) and excluded.
# Models returning HTTP 429 are rate-limited upstream and skipped.
# Latency measured on a single ping ("Reply with exactly one word: pong").
OPENROUTER_FREE_MODELS: List[Dict] = [
    {
        "id": "liquid/lfm-2.5-1.2b-instruct:free",
        "label": "Liquid LFM 1.2B (fastest, small)",
        "ctx": 32768,
        "latency_ms": 601,
        "best_for": "smoke tests, low-latency benchmarks",
        "verified_works": True,
    },
    {
        "id": "openrouter/free",
        "label": "OpenRouter auto-route (200k ctx)",
        "ctx": 200000,
        "latency_ms": 838,
        "best_for": "let OpenRouter pick, general use",
        "verified_works": True,
    },
    {
        "id": "nvidia/nemotron-3-super-120b-a12b:free",
        "label": "Nemotron Super 120B (1M ctx)",
        "ctx": 1000000,
        "latency_ms": 917,
        "best_for": "very long context, large documents",
        "verified_works": True,
    },
    {
        "id": "nvidia/nemotron-3-nano-30b-a3b:free",
        "label": "Nemotron Nano 30B (256k ctx)",
        "ctx": 256000,
        "latency_ms": 923,
        "best_for": "balanced speed/quality, good default",
        "verified_works": True,
    },
    {
        "id": "openai/gpt-oss-20b:free",
        "label": "GPT-OSS 20B (OpenAI-class, small)",
        "ctx": 131072,
        "latency_ms": 1503,
        "best_for": "OpenAI-style responses at small size",
        "verified_works": True,
    },
    {
        "id": "google/gemma-4-31b-it:free",
        "label": "Gemma 4 31B (Google)",
        "ctx": 262144,
        "latency_ms": 1536,
        "best_for": "Q&A, factual, long context",
        "verified_works": True,
    },
    {
        "id": "openai/gpt-oss-120b:free",
        "label": "GPT-OSS 120B (OpenAI-class, large)",
        "ctx": 131072,
        "latency_ms": 2565,
        "best_for": "instruction following, JSON, complex reasoning",
        "verified_works": True,
    },
    {
        "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "label": "Nemotron Ultra 550B (huge MoE)",
        "ctx": 1000000,
        "latency_ms": 6378,
        "best_for": "biggest free model, slow but capable",
        "verified_works": True,
    },
    {
        "id": "openrouter/owl-alpha",
        "label": "OpenRouter owl-alpha (1M ctx)",
        "ctx": 1048756,
        "latency_ms": 21369,
        "best_for": "huge context, slow",
        "verified_works": True,
    },
    # NOT VERIFIED / of rate-limited as of 2026-06-23:
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B (429 rate-limited)", "ctx": 131072, "verified_works": False},
    {"id": "qwen/qwen3-next-80b-a3b-instruct:free", "label": "Qwen3 Next 80B (429 rate-limited)", "ctx": 262144, "verified_works": False},
    {"id": "meta-llama/llama-3.2-3b-instruct:free", "label": "Llama 3.2 3B (429 rate-limited)", "ctx": 131072, "verified_works": False},
]


def list_openrouter_free_models() -> List[Dict]:
    """Return the curated list of OpenRouter free models for the bench."""
    return OPENROUTER_FREE_MODELS


class LLMClient:
    """Thin LLM client that supports OpenAI-compatible APIs (incl. OpenRouter) and Ollama."""

    def __init__(self, backend: Optional[str] = None, timeout: float = 120.0):
        self.timeout = timeout
        forced = (backend or os.environ.get("MATHIR_LLM_BACKEND", "auto")).lower()
        has_api = bool(os.environ.get("MATHIR_API_KEY"))
        if forced == "ollama":
            self.backend = "ollama"
        elif forced == "openrouter":
            if not has_api:
                raise LLMUnavailable(
                    "MATHIR_LLM_BACKEND=openrouter but MATHIR_API_KEY is empty. "
                    "Get a free key at https://openrouter.ai/"
                )
            self.backend = "openrouter"
        elif forced == "api":
            if not has_api:
                raise LLMUnavailable("MATHIR_LLM_BACKEND=api but MATHIR_API_KEY is empty")
            self.backend = "api"
        else:  # auto
            if has_api:
                # Heuristic: if the key starts with "sk-or-" it's OpenRouter
                api_key = os.environ.get("MATHIR_API_KEY", "")
                if api_key.startswith("sk-or-"):
                    self.backend = "openrouter"
                else:
                    self.backend = "api"
            else:
                self.backend = "ollama"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, prompt: str, system: str = "", max_tokens: int = 512,
             temperature: float = 0.7) -> str:
        """Single-turn chat. Returns assistant text (stripped)."""
        if self.backend in ("api", "openrouter"):
            text = self._call_api(prompt, system, max_tokens, temperature)
        else:
            text = self._call_ollama(prompt, system, max_tokens, temperature)
        # Strip thinking tags (qwen3, lfm2.5-thinking emit <think>...</think>)
        import re
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        return text

    def health(self) -> dict:
        """Returns a small status dict — never includes secrets."""
        info = {"backend": self.backend, "ok": False, "error": None}
        try:
            if self.backend in ("api", "openrouter"):
                base = self._api_base().rstrip("/")
                req = urllib.request.Request(f"{base}/models", method="GET")
                try:
                    urllib.request.urlopen(req, timeout=5).read()
                    info["ok"] = True
                except urllib.error.HTTPError as e:
                    # 401/403 means the endpoint is reachable but key is bad — still useful
                    info["ok"] = e.code in (401, 403)
                    if e.code not in (401, 403):
                        info["error"] = f"HTTP {e.code}"
            else:
                url = os.environ.get("MATHIR_OLLAMA_URL", "http://127.0.0.1:11434")
                req = urllib.request.Request(f"{url}/api/tags", method="GET")
                urllib.request.urlopen(req, timeout=5).read()
                info["ok"] = True
        except Exception as e:
            info["error"] = str(e)[:120]
        return info

    # ------------------------------------------------------------------
    # Backend config helpers
    # ------------------------------------------------------------------

    def _api_base(self) -> str:
        if self.backend == "openrouter":
            return os.environ.get("MATHIR_API_BASE", "https://openrouter.ai/api/v1")
        return os.environ.get("MATHIR_API_BASE", "https://api.minimax.chat/v1")

    def _api_model(self) -> str:
        if self.backend == "openrouter":
            return os.environ.get("MATHIR_API_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        return os.environ.get("MATHIR_API_MODEL", "MiniMax-M2.7")

    # ------------------------------------------------------------------
    # OpenAI-compatible API backend (works with OpenRouter, MiniMax, etc.)
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        api_key = os.environ.get("MATHIR_API_KEY", "")
        if not api_key:
            raise LLMUnavailable("MATHIR_API_KEY is empty")
        base = self._api_base().rstrip("/")
        model = self._api_model()

        payload = {
            "model": model,
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # OpenRouter-specific: ranking headers (optional but recommended)
        if self.backend == "openrouter":
            referer = os.environ.get("MATHIR_OPENROUTER_REFERER")
            title = os.environ.get("MATHIR_OPENROUTER_TITLE")
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title

        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        elapsed = (time.perf_counter() - t0) * 1000
        text = data["choices"][0]["message"]["content"].strip()
        self._last_elapsed_ms = elapsed
        return text

    # ------------------------------------------------------------------
    # Ollama backend
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        url = os.environ.get("MATHIR_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        model = os.environ.get("MATHIR_OLLAMA_MODEL", "qwen3.5:2b")
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        req = urllib.request.Request(
            f"{url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        elapsed = (time.perf_counter() - t0) * 1000
        text = (data.get("response") or "").strip()
        self._last_elapsed_ms = elapsed
        return text


# -----------------------------------------------------------------------------
# CLI: list free models + smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "list-free":
        print("OpenRouter free models curated for MATHIR benchmark:")
        print("=" * 90)
        for m in OPENROUTER_FREE_MODELS:
            print(f'  {m["id"]:55s} ctx={m["ctx"]:>7}  {m["label"]}')
        print()
        print("Usage:")
        print("  $env:MATHIR_LLM_BACKEND = 'openrouter'")
        print("  $env:MATHIR_API_KEY = 'sk-or-v1-...'")
        print("  $env:MATHIR_API_MODEL = 'meta-llama/llama-3.3-70b-instruct:free'")
        print("  python benchmarks/04_lifecycle_bench/run_all.py --duration 20")
        sys.exit(0)

    c = LLMClient()
    h = c.health()
    print(f"backend={h['backend']} ok={h['ok']} error={h['error']}")
    if h["ok"]:
        try:
            ans = c.chat("Reply with one word: ok", max_tokens=10, temperature=0)
            print(f"smoke-test reply: {ans!r}")
        except Exception as e:
            print(f"smoke-test error: {e}")
