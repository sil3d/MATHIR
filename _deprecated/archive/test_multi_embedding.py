import torch
from mathir_dropin import MATHIRMemory

def test_multi_embedding_store():
    memory = MATHIRMemory(embedding_dim=384, db_path=":memory:")
    
    emb_primary = torch.randn(1, 384)
    emb_a = torch.randn(1, 384)
    emb_b = torch.randn(1, 384)
    
    mid = memory.store(emb_primary, {"text": "physics"}, extra_providers={
        "provider_a": (emb_a, "model_a"),
        "provider_b": (emb_b, "model_b")
    })
    
    # Verify all stored via _store (internal SQLiteStore instance)
    emb_retrieved_a = memory._store.get_embedding(mid, "provider_a")
    emb_retrieved_b = memory._store.get_embedding(mid, "provider_b")
    
    assert emb_retrieved_a is not None, "provider_a embedding should be stored"
    assert emb_retrieved_b is not None, "provider_b embedding should be stored"
    print("Multi-embedding store: PASS")

def test_cross_provider_recall():
    memory = MATHIRMemory(embedding_dim=384, db_path=":memory:")
    
    emb_primary = torch.randn(1, 384)
    emb_a = torch.randn(1, 384)
    
    mid = memory.store(emb_primary, {"text": "thermodynamics"}, extra_providers={
        "openai": (emb_primary, "text-embedding-3-small"),
        "cohere": (emb_a, "embed-english-v3.0")
    })
    
    # Query with provider
    results = memory.recall(emb_a, k=1, provider="cohere")
    assert results is not None
    assert len(results) > 0
    print("Cross-provider recall: PASS")

if __name__ == "__main__":
    test_multi_embedding_store()
    test_cross_provider_recall()
    print("All Solution B tests passed!")
