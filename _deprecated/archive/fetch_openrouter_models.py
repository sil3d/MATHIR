#!/usr/bin/env python3
"""Fetch free models from OpenRouter API."""
import json
import urllib.request

API_KEY = "***REMOVED***"

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {API_KEY}"}
)

with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read())

all_models = data.get("data", [])
print(f"Total models: {len(all_models)}")

# Free = prompt price is 0
free_models = [
    m for m in all_models
    if float(m.get("pricing", {}).get("prompt", "1") or "1") == 0
]
print(f"Free models: {len(free_models)}")

# Sort by context length
free_models.sort(key=lambda m: m.get("context_length", 0), reverse=True)

print("\n=== FREE MODELS ===")
for m in free_models:
    pid = m["id"]
    name = m.get("name", pid)
    ctx = m.get("context_length", 0)
    desc = m.get("description", "")[:60]
    print(f"{pid} | ctx:{ctx} | {desc}")

# Save to file for later use
with open("D:/SECRET_PROJECT/MATHIR/openrouter_free_models.json", "w") as f:
    json.dump(free_models, f, indent=2)
print(f"\nSaved {len(free_models)} free models to openrouter_free_models.json")
