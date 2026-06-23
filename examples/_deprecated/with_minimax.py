"""
MATHIR + MiniMax API — Quick Example
====================================

Use MATHIR as a memory layer for the MiniMax API.

Setup:
    export MINIMAX_API_KEY="your-api-key"
    export MINIMAX_BASE_URL="https://api.minimaxi.com/v1"
    python examples/with_minimax.py

What this does:
    1. Connects to MiniMax embedding API
    2. Creates a MATHIR plugin
    3. Has a multi-turn conversation
    4. MATHIR maintains memory between turns
    5. Each new turn uses enhanced context from MATHIR
"""

import os
import sys
import torch
from openai import OpenAI

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathir_lib import MATHIRPlugin


class MiniMaxProvider:
    """Simple MiniMax embedding provider."""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "embo-01"):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        self.base_url = base_url or os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
        self.model = model
        self._dim = None
        
        if not self.api_key:
            raise ValueError(
                "MINIMAX_API_KEY not set. Set it as an environment variable."
            )
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def embed_text(self, text: str) -> torch.Tensor:
        """Embed a single text."""
        resp = self.client.embeddings.create(model=self.model, input=text)
        emb = torch.tensor(resp.data[0].embedding, dtype=torch.float32).unsqueeze(0)
        if self._dim is None:
            self._dim = emb.size(-1)
        return emb
    
    def embedding_dim(self) -> int:
        return self._dim


def main():
    print("=" * 60)
    print("MATHIR + MiniMax Demo")
    print("=" * 60)
    
    # 1. Setup
    print("\n[1] Connecting to MiniMax...")
    try:
        provider = MiniMaxProvider()
        emb = provider.embed_text("test")
        dim = provider.embedding_dim()
        print(f"    ✓ Connected (embedding dim: {dim})")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        print("    Set MINIMAX_API_KEY environment variable and try again.")
        return
    
    # 2. Create MATHIR plugin
    print(f"\n[2] Creating MATHIR plugin...")
    plugin = MATHIRPlugin(embedding_dim=dim)
    print(f"    ✓ Plugin ready")
    print(f"    Config: working={plugin.working_capacity}, episodic={plugin.episodic_capacity}, semantic={plugin.semantic_prototypes}")
    
    # 3. Simulate a conversation
    print(f"\n[3] Multi-turn conversation (10 turns)...")
    print("-" * 60)
    
    conversation = [
        "My name is Alice and I live in Paris.",
        "I work as a data scientist at a startup.",
        "My favorite programming language is Python.",
        "I'm building an autonomous driving project called MATHIR.",
        "I use PyTorch for the neural network implementation.",
        "I have a cat named Schrödinger.",
        "I graduated from MIT in 2020.",
        "I love hiking in the Alps on weekends.",
        "My project is competing against LSTM and Transformer baselines.",
        "The memory layer is the most important part of the system.",
    ]
    
    for turn, user_text in enumerate(conversation, 1):
        # Get embedding
        emb = provider.embed_text(user_text)
        
        # Process through MATHIR
        output = plugin.perceive(emb)
        
        # Store in memory
        plugin.store({"embedding": emb})
        
        # Print what MATHIR learned
        anomaly = output["anomaly_score"].item()
        router = output["router_weights"].squeeze().tolist()
        
        print(f"\nTurn {turn}: \"{user_text}\"")
        print(f"  Anomaly score: {anomaly:.3f}")
        print(f"  Router weights: [work={router[0]:.2f}, epi={router[1]:.2f}, sem={router[2]:.2f}, imm={router[2]:.2f}]")
    
    # 4. Test recall
    print("\n" + "-" * 60)
    print("[4] Testing recall (memory retrieval)...")
    
    queries = [
        "What is the user's name?",
        "Where does the user work?",
        "What is the user's project called?",
    ]
    
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        emb = provider.embed_text(query)
        memories = plugin.recall(emb, k=3)
        
        if memories:
            for i, mem in enumerate(memories, 1):
                print(f"  #{i}: similarity={mem['similarity']:.3f}, index={mem['index']}")
        else:
            print("  (no memories found)")
    
    # 5. Stats
    print("\n" + "-" * 60)
    print("[5] Memory statistics:")
    stats = plugin.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
