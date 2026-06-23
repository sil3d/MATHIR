"""Test all 5 OpenCode Zen free models (with correct endpoint, auth, model ID)."""
import sys
import json
import time
import urllib.request
import urllib.error

sys.path.insert(0, ".")
from env_config import load_env, get_opencode_zen_api_key
load_env()

api_key = get_opencode_zen_api_key()
if not api_key:
    print("No key")
    sys.exit(1)
print(f"Key: ...{api_key[-8:]}")

models = [
    "mimo-v2.5-free",
    "big-pickle",
    "deepseek-v4-flash-free",
    "north-mini-code-free",
    "nemotron-3-ultra-free",
]

endpoint = "https://opencode.ai/zen/v1/responses"
print()
print("=" * 90)
print(f"{'Model':<35} {'Latency':<10} {'Status'}")
print("=" * 90)

results = []
for model in models:
    t0 = time.time()
    body = json.dumps({
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Reply with exactly two words: yes pong"}]}],
        "max_tokens": 100,
        "temperature": 0.0,
    }).encode()
    try:
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "opencode-cli/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            dt = time.time() - t0
            text = "?"
            if "output" in data and isinstance(data["output"], list):
                for item in data["output"]:
                    if isinstance(item, dict) and "content" in item:
                        content = item["content"]
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    text = c.get("text", "?")
                                    break
                        elif isinstance(content, str):
                            text = content
                        if text != "?":
                            break
            print(f"  {model:<35} {dt:>6.2f}s   OK: {text[:60]!r}")
            results.append((model, dt, "OK", text))
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        body = e.read()[:200].decode(errors="replace")
        print(f"  {model:<35} {dt:>6.2f}s   FAIL: HTTP {e.code} - {body[:80]}")
        results.append((model, dt, f"FAIL {e.code}", body[:80]))
    except Exception as e:
        dt = time.time() - t0
        print(f"  {model:<35} {dt:>6.2f}s   FAIL: {type(e).__name__}: {str(e)[:80]}")
        results.append((model, dt, "FAIL", str(e)[:80]))
    time.sleep(1)

print()
print("=" * 90)
ok = sum(1 for r in results if r[2] == "OK")
print(f"RESULT: {ok}/{len(results)} Zen models work")
print("=" * 90)