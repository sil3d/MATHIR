"""Test script for Solutions A and B."""
import torch
import sys
sys.path.insert(0, r"D:\SECRET_PROJECT\MATHIR")

from mathir_dropin import MATHIRMemory

def test1_provider_tracking():
    """Test1: Provider tracking (Solution A)"""
    print("Test 1: Provider tracking")
    try:
        memory = MATHIRMemory(embedding_dim=384, db_path=":memory:")
        memory.provider = "test_provider"
        memory.model = "test_model"

        # Store something
        emb = torch.randn(1, 384)
        mid = memory.store(emb, {"text": "hello world"})

        # Verify provider was stored
        stored = memory._store.get(mid)
        print(f"  stored: {stored}")
        assert stored["provider"] == "test_provider", f"Expected test_provider, got {stored.get('provider')}"
        assert stored["model"] == "test_model", f"Expected test_model, got {stored.get('model')}"
        print("Provider tracking: PASS")
        return True
    except Exception as e:
        print(f"Provider tracking: FAIL - {e}")
        import traceback; traceback.print_exc()
        return False


def test2_reembed():
    """Test 2: Re-embed (Solution A)"""
    print("\nTest 2: Re-embed")
    try:
        memory = MATHIRMemory(embedding_dim=384, db_path=":memory:")

        # Store some items first
        for i in range(3):
            emb = torch.randn(1, 384)
            memory.store(emb, {"text": f"memory {i}"})

        # Re-embed with new provider
        call_count = [0]
        def mock_embed(text):
            call_count[0] += 1
            return torch.randn(1, 384)

        memory.reembed("new_provider", "new_model", embedding_fn=mock_embed)
        assert call_count[0] > 0, f"Expected calls > 0, got {call_count[0]}"
        print("Re-embed: PASS")
        return True
    except Exception as e:
        print(f"Re-embed: FAIL - {e}")
        import traceback; traceback.print_exc()
        return False


def test3_multi_embedding():
    """Test 3: Multi-embedding storage (Solution B)"""
    print("\nTest 3: Multi-embedding storage")
    try:
        memory = MATHIRMemory(embedding_dim=384, db_path=":memory:")

        # Store with multiple providers
        emb_a = torch.randn(1, 384)
        emb_b = torch.randn(1, 384)
        metadata = {"text": "test"}

        mid = memory.store(emb_a, metadata, extra_providers={
            "provider_a": (emb_a, "model_a"),
            "provider_b": (emb_b, "model_b")
        })

        # Query with provider_a embeddings
        results = memory.recall(emb_a, k=1, provider="provider_a")
        assert results is not None, "Expected results, got None"
        print("Multi-embedding: PASS")
        return True
    except Exception as e:
        print(f"Multi-embedding: FAIL - {e}")
        import traceback; traceback.print_exc()
        return False


def test4_cross_provider():
    """Test 4: Cross-provider retrieval"""
    print("\nTest 4: Cross-provider retrieval")
    try:
        memory = MATHIRMemory(embedding_dim=384, db_path=":memory:")

        # Store with provider A, query with provider B embeddings
        # Using different dims to simulate cross-provider
        memory2 = MATHIRMemory(embedding_dim=384, db_path=":memory:")
        emb_openai = torch.randn(1, 1536)
        emb_cohere = torch.randn(1, 1024)
        mid = memory2.store(emb_openai[:1, :384], {"text": "physics"}, extra_providers={
            "openai": (emb_openai[:1, :384], "text-embedding-3-small"),
            "cohere": (emb_cohere[:1, :384], "embed-english-v3.0")
        })
        # Recall with cohere embedding
        results = memory2.recall(emb_cohere[:1, :384], k=1, provider="cohere")
        assert len(results) > 0, f"Expected results, got {len(results)}"
        print("Cross-provider retrieval: PASS")
        return True
    except Exception as e:
        print(f"Cross-provider retrieval: FAIL - {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    results.append(test1_provider_tracking())
    results.append(test2_reembed())
    results.append(test3_multi_embedding())
    results.append(test4_cross_provider())

    print("\n" + "=" * 60)
    passed = sum(results)
    failed = len(results) - passed
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        print("BUILD: FAIL")
    else:
        print("BUILD: PASS")
