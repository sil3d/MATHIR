import json, urllib.request

API_KEY = "***REMOVED***"
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
with urllib.request.urlopen(req, timeout=15) as resp:
    data = json.loads(resp.read())

models = data.get("data", [])
print(f"Total models: {len(models)}")

# Find truly free models (prompt price = 0)
free_models = []
for m in models:
    pricing = m.get("pricing", {})
    prompt_price = pricing.get("prompt", "999")
    # Check if prompt is "0" or 0
    if prompt_price == "0" or prompt_price == 0:
        free_models.append({
            "id": m["id"],
            "name": m.get("name", ""),
            "prompt": prompt_price,
            "completion": pricing.get("completion"),
            "context_length": m.get("context_length"),
            "top_provider": m.get("top_provider", {}),
            "per_request_limits": m.get("per_request_limits"),
        })

print(f"Truly free models (prompt=0): {len(free_models)}")
print()

# Sort by context length descending
free_models.sort(key=lambda x: x["context_length"] or 0, reverse=True)

print("Top 20 free models by context length:")
print(f"{'ID':<60} {'Context':<10} {'Prompt':<8} {'Completion':<10} {'RPM'}")
print("-" * 120)
for m in free_models[:20]:
    prl = m.get("per_request_limits") or {}
    rpm = prl.get("requests_per_minute", "unlimited")
    print(f"{m['id']:<60} {str(m['context_length']):<10} {str(m['prompt']):<8} {str(m['completion']):<10} {rpm}")

print()
print("All free model IDs:")
for m in free_models:
    print(f"  {m['id']}")