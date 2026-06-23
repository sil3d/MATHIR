"""List available models on OpenCode Zen (the actual curated free list)."""
import sys
import json
import urllib.request
import os

sys.path.insert(0, ".")
from env_config import load_env, get_opencode_zen_api_key
load_env()

api_key = get_opencode_zen_api_key()
if not api_key:
    print("No OPENCODE_ZEN_API_KEY in env/.env")
    sys.exit(1)

# OpenCode Zen endpoint
endpoints = [
    "https://opencode.ai/zen/v1/models",
    "https://opencode.ai/zen/v1",
]
for url in endpoints:
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            print(f"=== {url} ===")
            if isinstance(data, dict) and "data" in data:
                models = data["data"]
            elif isinstance(data, list):
                models = data
            else:
                models = []
            print(f"Found {len(models)} models")
            print()
            # Look for free models
            free = []
            for m in models:
                mid = m.get("id", "?")
                name = m.get("name", mid)
                pricing = m.get("pricing", {})
                # free = prompt = 0
                try:
                    is_free = float(pricing.get("prompt", "1")) == 0 and float(pricing.get("completion", "1")) == 0
                except (ValueError, TypeError):
                    is_free = False
                if is_free or "free" in mid.lower() or "mimo" in mid.lower() or "minimax" in mid.lower():
                    free.append({"id": mid, "name": name, "free": is_free, "pricing": pricing})
            print(f"Free / special models: {len(free)}")
            for f in free[:50]:
                print(f"  {f['id']:<50} {f['name'][:50]} {f['pricing']}")
            break
    except Exception as e:
        print(f"  {url}: {e}")
        continue