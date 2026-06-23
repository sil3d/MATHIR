"""MATHIR Playground — quick test of all 4 configured free models.

Tests both:
- Chat (text-only) with prompt "Reply with the single word: pong"
- Vision (image) with a tiny inline PNG + prompt "What color is this image? Reply in one word."

Outputs a markdown table with per-model results.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from env_config import load_env, get_openrouter_api_key
load_env()

from vision_test import OpenRouterClient, load_config

# Inline 1x1 red PNG (89 bytes, well-known smallest valid PNG)
RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEAQH/"
    "FFGKkQAAAABJRU5ErkJggg=="
)


def test_model_chat(client, timeout_s=60):
    """Test text-only chat. Returns (latency_s, response_text, error_str)."""
    t0 = time.time()
    try:
        # Use higher max_tokens for Zen models (some are quirky with low limits)
        max_tok = 100 if client.provider == "opencode_zen" else 20
        result = client.chat(
            messages=[{"role": "user", "content": "Reply with exactly two words: yes pong"}],
            max_tokens=max_tok,
            temperature=0.0,
        )
        dt = time.time() - t0
        if isinstance(result, str) and result.startswith("ERROR"):
            return dt, None, result
        return dt, result[:200], None
    except Exception as e:
        return time.time() - t0, None, f"{type(e).__name__}: {e}"


def test_model_vision(client, timeout_s=60):
    """Test vision chat with a tiny red PNG. Returns (latency_s, response_text, error_str)."""
    t0 = time.time()
    try:
        # Build the multimodal message manually (chat_with_image needs a file path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this 1x1 pixel image? Reply in one word."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{RED_PNG_B64}"}},
            ],
        }]
        result = client.chat(messages, max_tokens=20, temperature=0.0)
        dt = time.time() - t0
        if isinstance(result, str) and result.startswith("ERROR"):
            return dt, None, result
        return dt, result[:200], None
    except Exception as e:
        return time.time() - t0, None, f"{type(e).__name__}: {e}"


def main():
    api_key = get_openrouter_api_key()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Edit .env or set env var.")
        sys.exit(1)
    print(f"OpenRouter API key loaded: ...{api_key[-8:]}")
    print()

    cfg = load_config()
    models = cfg.get("models", {})

    print("=" * 90)
    print(f"{'Model':<55} {'Mode':<8} {'Latency':<10} {'Status':<10}")
    print("=" * 90)

    results = []
    for name, m in models.items():
        # Skip non-dict entries (e.g. "_comment" string)
        if not isinstance(m, dict):
            continue
        if not m.get("enabled", True):
            continue
        # Make sure m has 'id' (it does per config.json)
        client = OpenRouterClient(m)
        # Override retries to 3 with longer waits (1+3+9=13s max per call)
        client.max_retries = 3
        if not client.api_key:
            status = "NO KEY"
            results.append((name, "chat", None, status))
            results.append((name, "vision", None, status))
            print(f"  [{client.provider:<8}] {name:<45} {'chat':<8} {'-':<10} {'NO KEY':<10}")
            print(f"  [{client.provider:<8}] {name:<45} {'vision':<8} {'-':<10} {'NO KEY':<10}")
            continue

        # Test chat
        dt_chat, resp_chat, err_chat = test_model_chat(client)
        # Override max_tokens for tests (some models need more)
        if client.provider == "opencode_zen":
            # MiMo and similar models need higher max_tokens
            pass  # already 20 in helper, may need adjustment
        if err_chat:
            chat_status = f"FAIL ({err_chat[:30]})"
        else:
            chat_status = "OK"
        print(f"  [{client.provider:<8}] {name:<45} {'chat':<8} {dt_chat:>6.2f}s   {chat_status}")
        results.append((name, "chat", dt_chat, chat_status, resp_chat, err_chat))

        # Sleep 2s between models to avoid rate limits
        time.sleep(2)

        # Test vision (only for models that claim to support it)
        if m.get("supports_vision"):
            dt_vis, resp_vis, err_vis = test_model_vision(client)
            if err_vis:
                vis_status = f"FAIL ({err_vis[:30]})"
            else:
                vis_status = "OK"
            print(f"  [{client.provider:<8}] {name:<45} {'vision':<8} {dt_vis:>6.2f}s   {vis_status}")
            results.append((name, "vision", dt_vis, vis_status, resp_vis, err_vis))
        else:
            print(f"  [{client.provider:<8}] {name:<45} {'vision':<8} {'-':<10} {'SKIP (text-only)'}")

    print("=" * 90)
    print()
    print("Sample responses (for OK cases):")
    for r in results:
        if len(r) == 6 and r[3] == "OK":
            name, mode, dt, _, resp, _ = r
            print(f"  [{mode}] {name}:")
            print(f"     {resp!r}")
    print()
    print("=" * 90)
    ok = sum(1 for r in results if len(r) == 6 and r[3] == "OK")
    total = sum(1 for r in results if len(r) == 6)
    print(f"RESULT: {ok}/{total} tests passed")
    print("=" * 90)
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()