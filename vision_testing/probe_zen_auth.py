"""Probe Zen auth methods — try Bearer, x-api-key, and other formats."""
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

# Try with a SINGLE model that's documented to be free
# The doc said MiMo-V2.5 Free, DeepSeek V4 Flash Free etc.
# Try a few different model ID formats
test_models = [
    "mimo-v2.5-free",       # just the suffix
    "opencode/mimo-v2.5-free",  # with prefix
    "MiMo-V2.5",           # display name
    "big-pickle",          # stealth
]

endpoint = "https://opencode.ai/zen/v1/responses"

# Auth methods to try
auth_methods = [
    ("Bearer", {"Authorization": f"Bearer {api_key}"}),
    ("x-api-key", {"x-api-key": api_key}),
    ("x-api-key (x-opencode)", {"x-opencode-key": api_key}),
    ("Bearer + User-Agent", {"Authorization": f"Bearer {api_key}", "User-Agent": "opencode-cli/1.0"}),
    ("cookie-style", {"Authorization": api_key}),
]

for model in test_models:
    print(f"\n=== Model: {model} ===")
    body = json.dumps({
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "ping"}]}],
        "max_tokens": 10,
    }).encode()
    for label, headers in auth_methods:
        headers_full = {"Content-Type": "application/json", **headers}
        try:
            req = urllib.request.Request(endpoint, data=body, headers=headers_full, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()[:500].decode(errors="replace")
                print(f"  [{label}] {resp.status} OK: {data[:100]!r}")
                break  # success, stop trying
        except urllib.error.HTTPError as e:
            err_body = e.read()[:200].decode(errors="replace")
            print(f"  [{label}] {e.code} {e.reason}: {err_body[:80]}")
        except Exception as e:
            print(f"  [{label}] {type(e).__name__}: {e}")