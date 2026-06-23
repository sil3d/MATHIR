"""Debug: print the full response from one Zen model to see the shape."""
import sys
import json
import urllib.request
import sys
from pathlib import Path

sys.path.insert(0, ".")
from env_config import load_env, get_opencode_zen_api_key
load_env()

api_key = get_opencode_zen_api_key()
body = json.dumps({
    "model": "mimo-v2.5-free",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "Reply with the single word: pong"}]}],
    "max_tokens": 20,
}).encode()
req = urllib.request.Request(
    "https://opencode.ai/zen/v1/responses",
    data=body,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "opencode-cli/1.0",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=20) as resp:
    data = json.loads(resp.read())
    print(json.dumps(data, indent=2)[:2000])