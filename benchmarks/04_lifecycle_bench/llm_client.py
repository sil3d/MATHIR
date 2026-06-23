"""
LLM client for the AI-driven cognitive benchmark.

NEVER embeds API keys. Reads from environment at call time.
Falls back to local Ollama if no API key is set.

Environment variables (read at runtime, never stored in this file):
  MATHIR_LLM_BACKEND  : "api" | "ollama" | "auto" (default: auto)
  MATHIR_API_KEY      : API key for the remote provider
  MATHIR_API_BASE     : API base URL (default: https://api.minimax.chat/v1)
  MATHIR_API_MODEL    : Model name (default: MiniMax-M2.7)
  MATHIR_OLLAMA_URL   : Ollama base URL (default: http://127.0.0.1:11434)
  MATHIR_OLLAMA_MODEL : Ollama model (default: qwen3.5:2b)
"""
import os
import json
import time
import urllib.request
import urllib.error
from typing import Optional


class LLMUnavailable(RuntimeError):
    """Raised when neither API nor Ollama is reachable."""


class LLMClient:
    """Thin LLM client that prefers API when MATHIR_API_KEY is set, else Ollama."""

    def __init__(self, backend: Optional[str] = None, timeout: float = 120.0):
        self.timeout = timeout
        forced = (backend or os.environ.get("MATHIR_LLM_BACKEND", "auto")).lower()
        has_api = bool(os.environ.get("MATHIR_API_KEY"))
        if forced == "api" and not has_api:
            raise LLMUnavailable("MATHIR_LLM_BACKEND=api but MATHIR_API_KEY is empty")
        if forced == "ollama":
            self.backend = "ollama"
        elif forced == "api":
            self.backend = "api"
        else:  # auto
            self.backend = "api" if has_api else "ollama"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, prompt: str, system: str = "", max_tokens: int = 512,
             temperature: float = 0.7) -> str:
        """Single-turn chat. Returns assistant text (stripped)."""
        if self.backend == "api":
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
            if self.backend == "api":
                base = os.environ.get("MATHIR_API_BASE", "https://api.minimax.chat/v1")
                req = urllib.request.Request(f"{base}/models", method="GET")
                # No auth header needed for /models usually; if 401 we still treat
                # the call as reachable so the bench can proceed.
                try:
                    urllib.request.urlopen(req, timeout=5).read()
                    info["ok"] = True
                except urllib.error.HTTPError as e:
                    info["ok"] = e.code in (401, 403)  # reachable but unauth is fine
            else:
                url = os.environ.get("MATHIR_OLLAMA_URL", "http://127.0.0.1:11434")
                req = urllib.request.Request(f"{url}/api/tags", method="GET")
                urllib.request.urlopen(req, timeout=5).read()
                info["ok"] = True
        except Exception as e:
            info["error"] = str(e)[:120]
        return info

    # ------------------------------------------------------------------
    # API backend (OpenAI-compatible chat/completions)
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        api_key = os.environ.get("MATHIR_API_KEY", "")
        if not api_key:
            raise LLMUnavailable("MATHIR_API_KEY is empty")
        base = os.environ.get("MATHIR_API_BASE", "https://api.minimax.chat/v1").rstrip("/")
        model = os.environ.get("MATHIR_API_MODEL", "MiniMax-M2.7")

        payload = {
            "model": model,
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        elapsed = (time.perf_counter() - t0) * 1000
        text = data["choices"][0]["message"]["content"].strip()
        # Attach timing so callers can log
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
# Self-test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    c = LLMClient()
    h = c.health()
    print(f"backend={h['backend']} ok={h['ok']} error={h['error']}")
    if h["ok"]:
        try:
            ans = c.chat("Reply with one word: ok", max_tokens=10, temperature=0)
            print(f"smoke-test reply: {ans!r}")
        except Exception as e:
            print(f"smoke-test error: {e}")
