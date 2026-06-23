"""
Empirical capability benchmark verifying MATHIR's core memory features
(Ebbinghaus retention, Mahalanobis anomaly detection, and KL router dynamics)
against static/FIFO baselines.
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

RESULTS_FILE = Path(__file__).parent.parent / "results" / "mathir_real_capabilities_results.json"

def calculate_auc(y_true, y_scores):
    """Calculates AUC-ROC manually without external sklearn dependency."""
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

def run_ebbinghaus_test():
    """
    Test 1: Spaced-Repetition (Ebbinghaus) eviction protection vs FIFO
    Tests if items that are retrieved frequently get stable and resist eviction.
    """
    print("\n--- Running Ebbinghaus Eviction Test ---")
    capacity = 10
    dim = 64
    
    # Configure plugin with Ebbinghaus memory
    config = get_default_config()
    config["memory"]["episodic_capacity"] = capacity
    config["memory"]["internal_dim"] = dim
    config["memory"]["episodic_type"] = "ebbinghaus"
    config["memory"]["ebbinghaus_alpha"] = 1.0 # high boost for test
    
    plugin = MATHIRPluginV7(embedding_dim=dim, config=config)
    
    # 1. Store initial capacity-worth of vectors
    elements = [torch.randn(1, dim) for _ in range(capacity)]
    for idx, x in enumerate(elements):
        plugin.store({"embedding": x})
        
    # Verify we are at full capacity
    assert plugin.episodic.count.item() == capacity
    
    print("  Initial stability:", plugin.episodic.stability[:capacity].tolist())
    
    # Print initial similarities immediately after storing
    print("  --- Debugging initial memory slots immediately after store ---")
    for i in range(capacity):
        val = plugin.episodic.values[i].unsqueeze(0)
        proj_x = plugin.input_proj(elements[i])
        sim = torch.cosine_similarity(proj_x, val, dim=-1).item()
        print(f"    Slot {i} self-sim: {sim:.4f}")
        
    # 2. Perceive element 0 multiple times (reinforce it via active retrieval)
    x_recall = elements[0]
    for i in range(5):
        out = plugin.perceive(x_recall)
        # Diagnostic: check what was retrieved
        retrieved_mem = plugin.recall(x_recall, k=1)
        print(f"    Perceive {i}: top recalled index = {retrieved_mem[0]['index'] if retrieved_mem else 'None'}, similarity = {retrieved_mem[0]['similarity'] if retrieved_mem else 0:.4f}")
        
    print("  Stability after reinforcement:", plugin.episodic.stability[:capacity].tolist())
    print("  Recall count after reinforcement:", plugin.episodic.recall_count[:capacity].tolist())
    
    # 3. Store 5 new vectors (exceed capacity, triggering eviction)
    new_elements = [torch.randn(1, dim) for _ in range(5)]
    for idx, x in enumerate(new_elements):
        # V7 Ebbinghaus store evicts if capacity is reached
        plugin.store({"embedding": x})
        
    # Verify count is still capped at capacity
    assert plugin.episodic.count.item() == capacity
    
    # 4. Check if reinforced element 0 is STILL in memory
    stored_values = plugin.episodic.values[:capacity]
    
    print("\n  --- Debugging memory slots after evictions ---")
    for i in range(capacity):
        val = stored_values[i].unsqueeze(0)
        sim = torch.cosine_similarity(plugin.input_proj(x_recall), val, dim=-1).item()
        print(f"    Slot {i}: sim_to_x_recall={sim:.4f}, stability={plugin.episodic.stability[i].item():.1f}, recall_count={plugin.episodic.recall_count[i].item():.1f}, last_access={plugin.episodic.last_access[i].item():.1f}")
        
    sims = torch.cosine_similarity(plugin.input_proj(x_recall), stored_values, dim=-1)
    max_sim = sims.max().item()
    
    # Check if element 1 (which was NOT recalled) was evicted
    x_no_recall = elements[1]
    sims_no_recall = torch.cosine_similarity(plugin.input_proj(x_no_recall), stored_values, dim=-1)
    max_sim_no_recall = sims_no_recall.max().item()
    
    recalled_retained = max_sim > 0.99
    unrecalled_evicted = max_sim_no_recall < 0.95
    
    print(f"\n  Recalled element similarity in memory: {max_sim:.4f} (Retained: {recalled_retained})")
    print(f"  Unrecalled element similarity in memory: {max_sim_no_recall:.4f} (Evicted: {unrecalled_evicted})")
    
    return {
        "recalled_similarity": max_sim,
        "unrecalled_similarity": max_sim_no_recall,
        "ebbinghaus_success": recalled_retained and unrecalled_evicted
    }

def run_anomaly_test():
    """
    Test 2: Immunological Anomaly Detection
    Compare AUC-ROC for detecting out-of-distribution patterns using Mahalanobis distance.
    """
    print("\n--- Running Anomaly Detection Test ---")
    dim = 64
    
    config = get_default_config()
    config["memory"]["internal_dim"] = dim
    config["memory"]["immune_type"] = "mahalanobis"
    config["memory"]["anomaly_threshold"] = 2.0
    
    plugin = MATHIRPluginV7(embedding_dim=dim, config=config)
    
    # Define a normal covariance distribution
    # Let's say normal vectors are generated from N(0, 1.0) with some correlations
    mean = np.zeros(dim)
    cov = np.eye(dim)
    # add some covariance structure
    for i in range(dim-1):
        cov[i, i+1] = 0.5
        cov[i+1, i] = 0.5
        
    # 1. Immunize: store normal patterns
    print("  Training running covariance on 100 normal vectors...")
    for _ in range(100):
        vec = np.random.multivariate_normal(mean, cov)
        plugin.store({"embedding": torch.from_numpy(vec).float().unsqueeze(0)})
        
    # 2. Generate test set: 50 normal, 50 anomalous
    test_y = []
    anomaly_scores = []
    
    # FAISS baseline cosine distance (we use cosine distance to closest normal vector as baseline)
    faiss_scores = []
    stored_tensors = plugin.immunological.bank[:plugin.immunological.count]
    
    for _ in range(50):
        # Normal
        vec = np.random.multivariate_normal(mean, cov)
        t_vec = torch.from_numpy(vec).float().unsqueeze(0)
        out = plugin.perceive(t_vec)
        anomaly_scores.append(out["anomaly_score"].item())
        test_y.append(0)
        
        # FAISS cosine distance baseline
        cos_sims = torch.cosine_similarity(plugin.input_proj(t_vec), stored_tensors, dim=-1)
        faiss_scores.append(1.0 - cos_sims.max().item())
        
    for _ in range(50):
        # Anomaly: OOD mean or high scaling
        vec = np.random.multivariate_normal(mean + 2.0, cov * 4.0)
        t_vec = torch.from_numpy(vec).float().unsqueeze(0)
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

def run_router_test():
    """
    Test 3: Router dynamics and tier shifts.
    Verify that the KL Router changes its weight allocations depending on input properties.
    """
    print("\n--- Running Router Dynamics Test ---")
    dim = 64
    
    config = get_default_config()
    config["memory"]["internal_dim"] = dim
    
    plugin = MATHIRPluginV7(embedding_dim=dim, config=config)
    
    # Store some vectors first to populate episodic/semantic stores
    for _ in range(20):
        plugin.store({"embedding": torch.randn(1, dim)})
        
    # Query 1: static/noise query (should distribute uniformly or lean semantic)
    q_noise = torch.randn(5, dim)
    out_noise = plugin.perceive(q_noise)
    w_noise = out_noise["router_weights"].mean(dim=0).tolist()
    
    # Query 2: Repeated sequence query (working memory context should rise)
    q_seq = torch.randn(1, dim)
    # perceive 10 times to populate working buffer with it
    for _ in range(10):
         out_seq = plugin.perceive(q_seq)
    
    w_seq = out_seq["router_weights"].mean(dim=0).tolist()
    
    # Memory tiers: 0: Working, 1: Episodic, 2: Semantic, 3: Immunological
    print(f"  Noise Query Routing Weights: Working={w_noise[0]:.3f}, Episodic={w_noise[1]:.3f}, Semantic={w_noise[2]:.3f}, Anomaly={w_noise[3]:.3f}")
    print(f"  Repeated Query Routing Weights: Working={w_seq[0]:.3f}, Episodic={w_seq[1]:.3f}, Semantic={w_seq[2]:.3f}, Anomaly={w_seq[3]:.3f}")
    
    # Calculate routing entropy
    entropy_noise = -sum(p * np.log2(p) for p in w_noise)
    entropy_seq = -sum(p * np.log2(p) for p in w_seq)
    
    return {
        "noise_routing": w_noise,
        "seq_routing": w_seq,
        "entropy_noise": entropy_noise,
        "entropy_seq": entropy_seq
    }

def main():
    print("=" * 60)
    print("MATHIR DYNAMIC CAPABILITY BENCHMARK")
    print("=" * 60)
    
    ebbinghaus = run_ebbinghaus_test()
    anomaly = run_anomaly_test()
    router = run_router_test()
    
    results = {
        "timestamp": time.time(),
        "ebbinghaus_test": ebbinghaus,
        "anomaly_test": anomaly,
        "router_test": router
    }
    
    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*60)
    print(f"BENCHMARK COMPLETE. Results saved to {RESULTS_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()
