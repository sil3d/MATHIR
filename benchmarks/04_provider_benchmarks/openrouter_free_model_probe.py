#!/usr/bin/env python3
"""
OpenRouter Free Model Prober - Test each free model individually.
Free models can be unavailable at any time - test one by one.
"""
import json, time, urllib.request, urllib.error, os, sys
from datetime import datetime

# Auto-load centralized .env at benchmarks/ root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _env  # noqa: F401 — populates os.environ

API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# All 27 free models from our discovery
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

def chat(model, messages, max_tokens=50, timeout=60):
    """Call OpenRouter chat API."""
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{OPENROUTER_BASE}/chat/completions",
        data=json.dumps(data).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mathir.dev",
            "X-Title": "MATHIR Free Model Probe",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"]
            model_used = result.get("model", model)
            return {"success": True, "content": content, "model_used": model_used, "usage": result.get("usage", {})}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        return {"success": False, "error": f"HTTP {e.code}", "body": body}
    except Exception as e:
        return {"success": False, "error": str(e)[:100]}


def probe_model(model, retries=3, delay=2):
    """Probe a model with retries and backoff."""
    for attempt in range(retries):
        result = chat(model, [{"role": "user", "content": "Reply with exactly one word: OK"}], max_tokens=10, timeout=45)
        if result["success"]:
            content = result.get("content") or ""
            return {"model": model, "status": "OK", "content": content[:50], "model_used": result.get("model_used"), "attempts": attempt + 1}
        # If rate limited, wait and retry
        if "HTTP 429" in result.get("error", "") or "HTTP 429" in result.get("body", ""):
            print(f"    Rate limited, waiting {delay}s...")
            time.sleep(delay)
            delay *= 2
        else:
            # Other error, don't retry same thing
            break
    return {"model": model, "status": "FAIL", "error": result.get("error", "unknown"), "attempts": attempt + 1}


def test_free_router():
    """Test the openrouter/free router."""
    print("\n" + "=" * 60)
    print("TESTING: openrouter/free (Free Models Router)")
    print("=" * 60)
    result = probe_model("openrouter/free", retries=5, delay=3)
    if result["status"] == "OK":
        print(f"  SUCCESS: used model '{result.get('model_used')}'")
        print(f"  Response: {result['content']}")
    else:
        print(f"  FAILED: {result.get('error')}")
    return result


def test_all_free_models():
    """Test all 27 free models one by one."""
    print("\n" + "=" * 60)
    print("TESTING: All 27 Free Models (one by one)")
    print("=" * 60)

    results = []
    working = []
    failed = []

    for i, model in enumerate(FREE_MODELS):
        print(f"\n[{i+1}/27] Testing: {model}")
        result = probe_model(model, retries=3, delay=2)
        results.append(result)

        if result["status"] == "OK":
            working.append(model)
            print(f"  OK - Response: {result['content'][:40]}")
        else:
            failed.append(model)
            print(f"  FAIL - {result.get('error')}")

        # Delay between models to avoid burst rate limits
        time.sleep(3)

    return results, working, failed


def run_mathir_benchmark(working_models):
    """Run MATHIR vs FAISS benchmark on working models."""
    print("\n" + "=" * 60)
    print("MATHIR vs FAISS BENCHMARK (on working models)")
    print("=" * 60)

    if not working_models:
        print("No working models to test.")
        return None

    benchmark_results = []

    for model in working_models:
        print(f"\n--- Benchmark: {model} ---")

        # Test connectivity and measure latency
        latencies = []
        for round_i in range(5):
            result = chat(model, [{"role": "user", "content": "What is 2+2?"}], max_tokens=20, timeout=60)
            if result["success"]:
                latencies.append(result.get("latency", 0))
            time.sleep(1)

        avg_latency = sum(latencies) / max(len(latencies), 1) if latencies else None

        # Setup MATHIR
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
            from mathir_dropin import MATHIRMemory
            import torch
            mathir = MATHIRMemory(embedding_dim=1024, db_path=f"mathir_{model.split('/')[-1].replace(':','_')}.db")
            mathir_ready = True
        except Exception as e:
            mathir = None
            mathir_ready = False
            print(f"  MATHIR setup failed: {e}")

        # Setup FAISS
        try:
            import faiss
            import numpy as np
            faiss_index = faiss.IndexFlatIP(1024)
            np_random = np.random.RandomState(42)
            faiss_ready = True
        except Exception as e:
            faiss_index = None
            np_random = None
            faiss_ready = False
            print(f"  FAISS setup failed: {e}")

        # Run memory stress test
        import torch
        scenarios = [50, 100, 200]  # messages per scenario

        mathir_times = []
        faiss_times = []

        for msg_count in scenarios:
            # Generate test data
            texts = [f"Message {i}: content about programming and physics." for i in range(msg_count)]

            # Store in MATHIR
            if mathir_ready and mathir:
                for text in texts:
                    emb = torch.randn(1, 1024)
                    mathir.store(emb, {"text": text})

            # Store in FAISS
            if faiss_ready and faiss_index:
                import numpy as np
                embs = np_random.randn(len(texts), 1024).astype("float32")
                faiss.normalize_L2(embs)
                faiss_index.add(embs)

            # Query MATHIR
            if mathir_ready and mathir:
                t0 = time.time()
                q = torch.randn(1, 1024)
                mathir.recall(q, k=5)
                mathir_times.append(time.time() - t0)

            # Query FAISS
            if faiss_ready and faiss_index:
                import numpy as np
                q_np = np_random.randn(1, 1024).astype("float32")
                faiss.normalize_L2(q_np)
                t0 = time.time()
                faiss_index.search(q_np, k=5)
                faiss_times.append(time.time() - t0)

        bm_result = {
            "model": model,
            "api_latency_s": avg_latency,
            "mathir_ready": mathir_ready,
            "faiss_ready": faiss_ready,
            "scenarios": [],
        }

        if mathir_times:
            bm_result["mathir_avg_s"] = sum(mathir_times) / len(mathir_times)
        if faiss_times:
            bm_result["faiss_avg_s"] = sum(faiss_times) / len(faiss_times)

        if mathir_times and faiss_times:
            bm_result["mathir_wins"] = sum(1 for m, f in zip(mathir_times, faiss_times) if m < f)

        benchmark_results.append(bm_result)

        m_str = f"{bm_result.get('mathir_avg_s', 'N/A'):.4f}s" if bm_result.get("mathir_avg_s") else "N/A"
        f_str = f"{bm_result.get('faiss_avg_s', 'N/A'):.4f}s" if bm_result.get("faiss_avg_s") else "N/A"
        print(f"  MATHIR avg: {m_str}, FAISS avg: {f_str}")
        print(f"  MATHIR wins: {bm_result.get('mathir_wins', 'N/A')}/{len(scenarios)}")

        time.sleep(3)

    return benchmark_results


if __name__ == "__main__":
    print("=" * 60)
    print("OPENROUTER FREE MODEL PROBER")
    print("=" * 60)
    print(f"Testing {len(FREE_MODELS)} free models + free router")
    print(f"Started: {datetime.now().isoformat()}")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "router_test": None,
        "model_tests": [],
        "benchmarks": [],
    }

    # 1. Test free router
    router_result = test_free_router()
    all_results["router_test"] = router_result
    time.sleep(5)

    # 2. Test all free models
    model_results, working_models, failed_models = test_all_free_models()
    all_results["model_tests"] = model_results
    all_results["working_models"] = working_models
    all_results["failed_models"] = failed_models

    # 3. Run MATHIR vs FAISS benchmark on working models
    if working_models:
        benchmarks = run_mathir_benchmark(working_models)
        all_results["benchmarks"] = benchmarks

    # Save results
    output_path = "os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "openrouter_free_model_probe.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Router (openrouter/free): {router_result['status']}")
    print(f"Working models: {len(working_models)}/{len(FREE_MODELS)}")
    print(f"Failed models: {len(failed_models)}/{len(FREE_MODELS)}")
    if working_models:
        print(f"\nWorking model IDs:")
        for m in working_models:
            print(f"  - {m}")
    if failed_models:
        print(f"\nFailed model IDs:")
        for m in failed_models:
            print(f"  - {m}")
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)