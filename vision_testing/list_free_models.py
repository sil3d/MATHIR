"""List actual free models on OpenRouter to fix the config.json."""
import sys
import json
import urllib.request

sys.path.insert(0, ".")
from env_config import load_env, get_openrouter_api_key
load_env()

api_key = get_openrouter_api_key()
if not api_key:
    print("No API key")
    sys.exit(1)

# Get /models endpoint
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"},
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
except Exception as e:
    print(f"Failed to list models: {e}")
    sys.exit(1)

# Filter free models (pricing.prompt = "0" or pricing.completion = "0")
free_models = []
for m in data.get("data", []):
    pricing = m.get("pricing", {})
    prompt = pricing.get("prompt", "0")
    completion = pricing.get("completion", "0")
    # Free = both prompt and completion are 0
    try:
        if float(prompt) == 0 and float(completion) == 0:
            free_models.append({
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "context_length": m.get("context_length", 0),
                "modality": m.get("architecture", {}).get("modality", "?"),
            })
    except (ValueError, TypeError):
        pass

# Sort by modality (text+image first, then text-only)
free_models.sort(key=lambda x: (0 if "image" in x["modality"] else 1, x["id"]))

print(f"Found {len(free_models)} free models on OpenRouter")
print()
print(f"{'ID':<60} {'Context':<10} {'Modality'}")
print("=" * 110)
for m in free_models[:30]:  # top 30
    print(f"  {m['id']:<58} {m['context_length']:<10} {m['modality']}")
print()
print("=" * 110)
print(f"Total free: {len(free_models)}")
print("=" * 110)

# Save the full list to a JSON for reference
with open("free_models_actual.json", "w") as f:
    json.dump(free_models, f, indent=2)
print(f"\nFull list saved to free_models_actual.json")