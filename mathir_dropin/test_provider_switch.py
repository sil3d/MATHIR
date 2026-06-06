import torch
from mathir_dropin import MATHIRMemory

def test_provider_tracking():
    memory = MATHIRMemory(embedding_dim=384, db_path=":memory:")
    memory.provider = "openai"
    memory.model = "text-embedding-3-small"

    emb = torch.randn(1, 384)
    mid = memory.store(emb, {"text": "hello world"})

    # Verify stored
    row = memory._store.get(mid)
    assert row is not None, "Memory should be stored"
    assert row["provider"] == "openai", f"Expected openai, got {row['provider']}"
    assert row["model"] == "text-embedding-3-small", f"Expected text-embedding-3-small, got {row['model']}"
    print("Provider tracking: PASS")

def test_store_with_provider_kwarg():
    """Test that store() accepts provider/model kwargs directly."""
    memory = MATHIRMemory(embedding_dim=384, db_path=":memory:")

    emb = torch.randn(1, 384)
    mid = memory.store(emb, {"text": "test"}, provider="cohere", model="embed-english-v3.0")

    row = memory._store.get(mid)
    assert row["provider"] == "cohere", f"Expected cohere, got {row['provider']}"
    assert row["model"] == "embed-english-v3.0", f"Expected embed-english-v3.0, got {row['model']}"
    print("Store with provider kwarg: PASS")

if __name__ == "__main__":
    test_provider_tracking()
    test_store_with_provider_kwarg()
    print("All provider tests passed!")