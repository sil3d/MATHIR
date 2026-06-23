"""Try MiMo with higher max_tokens + simpler prompt."""
import sys
import json
import urllib.request

sys.path.insert(0, ".")
from env_config import load_env, get_opencode_zen_api_key
load_env()

api_key = get_opencode_zen_api_key()

# Test mimo with different params
for model, max_tok, prompt in [
    ("mimo-v2.5-free", 200, "What is 2+2? Answer with just the number."),
    ("mimo-v2.5-free", 200, "pong"),
    ("mimo-v2.5-free", 50, "ping"),
    ("deepseek-v4-flash-free", 200, "What is 2+2? Answer with just the number."),
    ("deepseek-v4-flash-free", 50, "ping"),
]:
    body = json.dumps({
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_tokens": max_tok,
        "temperature": 0.7,
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
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            # Find any text in the response
            text = None
            for item in data.get("output", []):
                if isinstance(item, dict) and "content" in item:
                    for c in item.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            text = c.get("text", "")
                            break
                if text:
                    break
            print(f"  {model} (max={max_tok}, prompt={prompt!r}): {text!r}")
    except Exception as e:
        print(f"  {model} (max={max_tok}, prompt={prompt!r}): {type(e).__name__}: {e}")