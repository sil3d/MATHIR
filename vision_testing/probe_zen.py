"""Probe OpenCode Zen — find the right endpoint."""
import sys
import json
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
print()

# Try various endpoints
endpoints = [
    "https://opencode.ai/api/v1/models",
    "https://opencode.ai/zen/v1/models",
    "https://opencode.ai/zen/v1/chat/completions",
    "https://opencode.ai/v1/models",
    "https://api.opencode.ai/v1/models",
    "https://api.opencode.ai/zen/v1/models",
]
for url in endpoints:
    try:
        # Try GET on /models
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"  GET  {url} -> {resp.status}")
            data = resp.read()[:2000]
            print(f"    {data[:500]!r}")
    except urllib.error.HTTPError as e:
        print(f"  GET  {url} -> {e.code} {e.reason}")
    except Exception as e:
        print(f"  GET  {url} -> {type(e).__name__}: {e}")
    print()

# Try a simple chat completion test
print("=== Trying a basic chat completion with gpt-oss-120b on Zen ===")
for api_base in ["https://opencode.ai/zen/v1", "https://api.opencode.ai/v1", "https://api.openrouter.ai/api/v1"]:
    try:
        body = json.dumps({
            "model": "opencode/gpt-oss-120b",
            "messages": [{"role": "user", "content": "Reply with just the word: ok"}],
            "max_tokens": 10,
        }).encode()
        req = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "?")
            print(f"  POST {api_base}/chat/completions -> {resp.status} content={content!r}")
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode(errors="replace")
        print(f"  POST {api_base}/chat/completions -> {e.code} {body[:200]}")
    except Exception as e:
        print(f"  POST {api_base}/chat/completions -> {type(e).__name__}: {e}")