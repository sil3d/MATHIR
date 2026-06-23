"""
Cognitive stress test for MATHIR memory systems using real BEIR (SciFact and NFCorpus) dataset embeddings.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from mathir_lib.plugin_v7 import MATHIRPluginV7
from mathir_lib.config import get_default_config

# Setup seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

CACHE_DIR = Path(__file__).parent / "controlled_emb_cache"
RESULTS_FILE = Path(__file__).parent.parent / "results" / "mathir_beir_cognitive_results.json"

def calculate_auc(y_true, y_scores):
    """Calculates AUC-ROC manually."""
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = np.array(y_scores)[desc_score_indices]
    y_true = np.array(y_true)[desc_score_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
        
    tp = 0
    fp = 0
    auc = 0.0
    last_tp = 0
    
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
            auc += (tp + last_tp) / 2.0
            last_tp = tp
            
    auc /= (n_pos * n_neg)
    return auc

def run_real_ebbinghaus_test():
    """
    Stress Test 1: Spaced-repetition memory retention under continuous eviction load using SciFact docs.
    Stores 1000 SciFact embeddings in a 100-capacity Ebbinghaus memory, verifying retention of reinforced items.
    """
    print("\n--- Running SciFact Ebbinghaus Eviction Stress Test ---")
    capacity = 100
    dim = 768 # BGE-base dim
    
    # Load SciFact corpus embeddings
    scifact_emb_path = CACHE_DIR / "scifact_BAAI_bge-base-en-v1.5_test_corpus_emb.npy"
    if not scifact_emb_path.exists():
        print(f"  Error: pre-computed embeddings not found at {scifact_emb_path}")
        return None
        
    corpus_embs = torch.from_numpy(np.load(scifact_emb_path)).float()
    print(f"  Loaded {len(corpus_embs)} SciFact document embeddings.")
    
    config = get_default_config()
    config["memory"]["episodic_capacity"] = capacity
    config["memory"]["internal_dim"] = 128
    config["memory"]["episodic_type"] = "ebbinghaus"
    config["memory"]["ebbinghaus_alpha"] = 1.0
    
    plugin = MATHIRPluginV7(embedding_dim=dim, config=config)
    
    # 1. Fill initial capacity (first 100 documents)
    print(f"  Filling episodic memory to capacity ({capacity} items)...")
    for i in range(capacity):
        plugin.store({"embedding": corpus_embs[i].unsqueeze(0)})
        
    # 2. Select 10 target documents and reinforce them
    target_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    target_embs = [corpus_embs[idx].unsqueeze(0) for idx in target_indices]
    
    print("  Reinforcing 10 target items...")
    for t_emb in target_embs:
        for _ in range(5):
            plugin.perceive(t_emb)
            
    # Verify reinforced items have high stability
    reinforced_stabilities = [plugin.episodic.stability[idx].item() for idx in target_indices]
    print(f"  Target items stabilities after reinforcement (should be high): {reinforced_stabilities}")
    
    # 3. Stream another 800 documents to trigger massive evictions
    print("  Streaming 800 more documents into memory (eviction load)...")
    for i in range(capacity, min(capacity + 800, len(corpus_embs))):
        plugin.store({"embedding": corpus_embs[i].unsqueeze(0)})
        
    # 4. Check if targets are still retained
    stored_values = plugin.episodic.values[:capacity]
    retained_counts = 0
    sims_list = []
    
    # Compute similarity to see if targets are preserved
    for t_emb in target_embs:
        projected_target = plugin.input_proj(t_emb)
        sims = torch.cosine_similarity(projected_target, stored_values, dim=-1)
        max_sim = sims.max().item()
        sims_list.append(max_sim)
        if max_sim > 0.99:
            retained_counts += 1
            
    # Check if early unrecalled items (e.g. index 1) are evicted
    unrecalled_embs = [corpus_embs[i].unsqueeze(0) for i in range(1, 10) if i not in target_indices]
    unrecalled_sims = []
    for u_emb in unrecalled_embs:
        projected_unrecalled = plugin.input_proj(u_emb)
        sims = torch.cosine_similarity(projected_unrecalled, stored_values, dim=-1)
        unrecalled_sims.append(sims.max().item())
        
    avg_retained_sim = np.mean(sims_list)
    avg_evicted_sim = np.mean(unrecalled_sims)
    
    print(f"  Target Items Retained: {retained_counts}/10")
    print(f"  Average similarity of reinforced targets: {avg_retained_sim:.4f}")
    print(f"  Average similarity of unrecalled/evicted items: {avg_evicted_sim:.4f}")
    
    return {
        "retained_count": retained_counts,
        "avg_retained_similarity": float(avg_retained_sim),
        "avg_evicted_similarity": float(avg_evicted_sim),
        "success": retained_counts >= 8
    }

def run_real_anomaly_test():
    """
    Stress Test 2: Immunological Anomaly Detection
    Immunize on SciFact query queries, detect NFCorpus (nutrition queries) as OOD/anomalies.
    """
    print("\n--- Running BEIR Anomaly Detection Stress Test ---")
    dim = 768
    
    scifact_q_path = CACHE_DIR / "scifact_BAAI_bge-base-en-v1.5_query_scifact_71962783feeac720.npy"
    nfcorpus_q_path = CACHE_DIR / "nfcorpus_BAAI_bge-base-en-v1.5_query_nfcorpus_b0f8b5c738108840.npy"
    
    if not scifact_q_path.exists() or not nfcorpus_q_path.exists():
        print("  Error: query embeddings not found.")
        return None
        
    scifact_queries = torch.from_numpy(np.load(scifact_q_path)).float()
    nfcorpus_queries = torch.from_numpy(np.load(nfcorpus_q_path)).float()
    
    print(f"  Loaded {len(scifact_queries)} SciFact (normal) queries and {len(nfcorpus_queries)} NFCorpus (anomaly) queries.")
    
    config = get_default_config()
    config["memory"]["internal_dim"] = 128
    config["memory"]["immune_type"] = "mahalanobis"
    
    plugin = MATHIRPluginV7(embedding_dim=dim, config=config)
    
    # 1. Immunize on first 200 SciFact queries
    train_size = min(200, len(scifact_queries))
    print(f"  Immunizing memory on {train_size} SciFact query patterns...")
    for i in range(train_size):
        plugin.store({"embedding": scifact_queries[i].unsqueeze(0)})
        
    # 2. Test: 100 normal (SciFact follow-up) vs 100 anomaly (NFCorpus nutrition queries)
    test_size = 100
    test_y = []
    anomaly_scores = []
    faiss_scores = []
    
    stored_tensors = plugin.immunological.bank[:plugin.immunological.count]
    
    # Normal queries
    for i in range(train_size, min(train_size + test_size, len(scifact_queries))):
        t_vec = scifact_queries[i].unsqueeze(0)
        out = plugin.perceive(t_vec)
        anomaly_scores.append(out["anomaly_score"].item())
        test_y.append(0)
        
        # FAISS cosine distance baseline
        cos_sims = torch.cosine_similarity(plugin.input_proj(t_vec), stored_tensors, dim=-1)
        faiss_scores.append(1.0 - cos_sims.max().item())
        
    # Anomaly queries
    for i in range(min(test_size, len(nfcorpus_queries))):
        t_vec = nfcorpus_queries[i].unsqueeze(0)
        out = plugin.perceive(t_vec)
        anomaly_scores.append(out["anomaly_score"].item())
        test_y.append(1)
        
        # FAISS cosine distance baseline
        cos_sims = torch.cosine_similarity(plugin.input_proj(t_vec), stored_tensors, dim=-1)
        faiss_scores.append(1.0 - cos_sims.max().item())
        
    # Calculate AUC-ROC
    mathir_auc = calculate_auc(test_y, anomaly_scores)
    faiss_auc = calculate_auc(test_y, faiss_scores)
    
    print(f"  MATHIR Mahalanobis Anomaly AUC-ROC: {mathir_auc:.4f}")
    print(f"  FAISS Cosine Distance Anomaly AUC-ROC: {faiss_auc:.4f}")
    
    return {
        "mathir_auc": mathir_auc,
        "faiss_auc": faiss_auc,
        "auc_gain": mathir_auc - faiss_auc
    }

def main():
    print("=" * 60)
    print("MATHIR BEIR DATASET COGNITIVE STRESS TEST")
    print("=" * 60)
    
    ebbinghaus = run_real_ebbinghaus_test()
    anomaly = run_real_anomaly_test()
    
    results = {
        "timestamp": time.time(),
        "ebbinghaus_stress": ebbinghaus,
        "anomaly_stress": anomaly
    }
    
    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*60)
    print(f"STRESS TEST COMPLETE. Results saved to {RESULTS_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()
