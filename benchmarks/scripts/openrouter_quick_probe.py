#!/usr/bin/env python3
"""
OpenRouter Free Model Quick Probe - Test each free model ONCE.
Fast: no retries, short timeout. Save results after each model.
"""
import json, time, urllib.request, urllib.error
from datetime import datetime

API_KEY = "***REMOVED***"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# All 27 free models
FREE_MODELS = [
    "openrouter/owl-alpha",
    "google/lyria-3-pro-preview",
    "google/lyria-3-clip-preview",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "poolside/laguna-xs.2:free",
    "poolside/laguna-m.1:free",
    "moonshotai/kimi-k2.6:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "z-ai/glm-4.5-air:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3.5-content-safety:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
]


def quick_probe(model):
    """Probe a model once, fast. Returns status."""
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "OK"}],
        "max_tokens": 5,
    }).encode()
    req = urllib.request.Request(
        f"{OPENROUTER_BASE}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mathir.dev",
            "X-Title": "MATHIR Quick Probe",
        },
        method="POST"
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            duration = time.time() - start
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"]
            model_used = result.get("model", model)
            return {
                "model": model,
                "status": "OK",
                "duration_s": round(duration, 3),
                "content": (content or "")[:60],
                "model_used": model_used,
                "usage": result.get("usage", {}),
            }
    except urllib.error.HTTPError as e:
        duration = time.time() - start
        body = e.read().decode("utf-8", errors="replace")[:150]
        return {
            "model": model,
            "status": f"HTTP_{e.code}",
            "duration_s": round(duration, 3),
            "error": body,
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "model": model,
            "status": "ERROR",
            "duration_s": round(duration, 3),
            "error": str(e)[:100],
        }


if __name__ == "__main__":
    print("=" * 60)
    print("OPENROUTER FREE MODEL QUICK PROBE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    results = []
    working = []
    failed = []

    for i, model in enumerate(FREE_MODELS):
        print(f"[{i+1}/27] {model[:55]:<55} ", end="", flush=True)
        r = quick_probe(model)
        results.append(r)

        if r["status"] == "OK":
            working.append(model)
            print(f"OK ({r['duration_s']}s) -> {r.get('model_used', '?')[:30]}")
        else:
            failed.append(model)
            print(f"FAIL ({r['status']})")

        time.sleep(1)  # small delay between probes

        # Save after every model (in case of crash)
        with open("os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "openrouter_free_model_probe.json"), "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "working": working,
                "failed": failed,
            }, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"WORKING ({len(working)}):")
    for m in working:
        print(f"  + {m}")
    print(f"\nFAILED ({len(failed)}):")
    for m in failed:
        r = next(r for r in results if r["model"] == m)
        print(f"  - {m} ({r['status']})")

    print(f"\nSaved to: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'openrouter_free_model_probe.json')}")