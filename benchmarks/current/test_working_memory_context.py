"""
Working Memory Context-Dependence Benchmark
==========================================

Tests whether MATHIR's 64-slot circular buffer makes query results context-dependent.

Key Insight:
  If you load "COVID-19" documents into working memory, then ask a short ambiguous
  query like "What about side effects?" -- MATHIR should answer in COVID context,
  while FAISS (no context) answers generically.

Test Protocol:
  1. Set working memory to "COVID-19" context (load ~20 docs about vaccines, pandemic)
  2. Query: "What about side effects?" → MATHIR should answer about COVID vaccine side effects
  3. Same query WITHOUT working memory context → MATHIR gives generic answer
  4. Compare: does the answer quality differ WITH vs WITHOUT context?

Implementation:
  - Measure retrieval reranking when working memory is loaded vs empty
  - Compute result overlap: len(set(top10_with_ctx) ∩ set(top10_without_ctx)) / 10
    - 1.0 = working memory has no effect (bad)
    - < 0.5 = working memory strongly changes results (good)
"""

import os
import sys
import json
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from mathir_lib.plugin_v7 import MATHIRPluginV7
from mathir_lib.config import get_default_config

# BEIR SciFact dataset
SCIFACT_DIR = Path(__file__).parent / "beir_data" / "scifact"
RESULTS_FILE = Path(__file__).parent.parent / "results" / "working_memory_results.json"


def download_scifact() -> bool:
    """Download SciFact dataset."""
    if os.path.exists(os.path.join(SCIFACT_DIR, "corpus.jsonl")):
        return True

    SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    os.makedirs(SCIFACT_DIR, exist_ok=True)
    zip_path = os.path.join(SCIFACT_DIR, "scifact.zip")

    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        import urllib.request
        with urllib.request.urlopen(SCIFACT_URL, context=ctx, timeout=60) as response:
            with open(zip_path, 'wb') as out:
                out.write(response.read())
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(SCIFACT_DIR)
        os.remove(zip_path)
        return True
    except Exception as e:
        print(f"  [FAIL] Download error: {e}")
        return False


def load_scifact() -> Tuple[Dict, Dict, Dict]:
    """Load SciFact dataset. Returns (corpus, queries, qrels)."""
    corpus = {}
    queries = {}
    qrels = {}

    # Find actual directory (handle nested folder structure)
    actual_dir = SCIFACT_DIR
    for sub in ['scifact']:
        candidate = os.path.join(SCIFACT_DIR, sub)
        if os.path.exists(os.path.join(candidate, "corpus.jsonl")):
            actual_dir = candidate
            break

    # Load corpus
    with open(os.path.join(actual_dir, "corpus.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }

    # Load queries
    with open(os.path.join(actual_dir, "queries.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q.get("text", "")

    # Load qrels from both test.tsv and train.tsv
    for qrels_file in ["test.tsv", "train.tsv"]:
        qrels_path = os.path.join(actual_dir, "qrels", qrels_file)
        if os.path.exists(qrels_path):
            with open(qrels_path, 'r', encoding='utf-8') as f:
                next(f)  # skip header
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        qid, did, rel = parts[0], parts[1], int(parts[2])
                        if qid not in qrels:
                            qrels[qid] = {}
                        qrels[qid][did] = rel

    return corpus, queries, qrels


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def compute_result_overlap(results1: List[str], results2: List[str]) -> float:
    """Compute overlap between two result lists (Jaccard-like)."""
    set1, set2 = set(results1), set(results2)
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    return len(set1 & set2) / 10.0  # Overlap out of top-10


def load_context_docs(plugin: MATHIRPluginV7, doc_embeddings: torch.Tensor, doc_ids: List[str],
                      context_indices: List[int]) -> None:
    """Load context documents into working memory via perceive() calls."""
    for idx in context_indices:
        emb = doc_embeddings[idx].unsqueeze(0)
        plugin.perceive(emb)


def reset_working_memory(plugin: MATHIRPluginV7) -> None:
    """Reset working memory buffers."""
    plugin.working_buffer.zero_()
    plugin.working_ptr = torch.tensor(0, dtype=torch.long)


def run_retrieval_with_mathir(plugin: MATHIRPluginV7, query_emb: torch.Tensor,
                               doc_embeddings: torch.Tensor, doc_ids: List[str],
                               top_k: int = 10) -> List[Tuple[str, float]]:
    """Run retrieval using MATHIR's internal recall mechanism.
    
    Args:
        plugin: MATHIRPluginV7 instance
        query_emb: [D] query embedding tensor
        doc_embeddings: [N, D] document embedding tensor
        doc_ids: list of document IDs
        top_k: number of results to return
    """
    # Ensure query is [1, D] for batch processing
    if query_emb.dim() == 1:
        query_emb = query_emb.unsqueeze(0)
    
    # Project query to internal space (before working memory retrieval)
    x = plugin.input_proj(query_emb)  # [1, internal_dim]
    
    # Retrieve from working memory (this is where context matters)
    working_ctx = plugin._retrieve_working(x)  # [1, internal_dim]
    
    # Use working-context-enhanced query for similarity
    query_with_ctx = x + working_ctx  # [1, internal_dim]
    
    # Project documents to internal space
    doc_proj = plugin.input_proj(doc_embeddings)  # [N, internal_dim]
    
    # Compute cosine similarity in internal space
    similarities = torch.nn.functional.cosine_similarity(
        query_with_ctx,  # [1, internal_dim]
        doc_proj,         # [N, internal_dim]
        dim=-1
    )
    
    # Get top-k
    top_indices = torch.argsort(similarities, descending=True)[:top_k]
    results = [(doc_ids[idx.item()], similarities[idx].item()) for idx in top_indices]
    
    return results


def run_faiss_baseline(query_embeddings: np.ndarray, doc_embeddings: np.ndarray,
                       doc_ids: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
    """Pure FAISS retrieval (no working memory)."""
    import faiss
    
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings.astype("float32"))
    
    scores, indices = index.search(query_embeddings.astype("float32"), top_k)
    results = [(doc_ids[idx], float(scores[0, i])) for i, idx in enumerate(indices[0])]
    
    return results


def main():
    print("=" * 70)
    print("WORKING MEMORY CONTEXT-DEPENDENCE BENCHMARK")
    print("=" * 70)
    
    # Download and load SciFact
    print("\n[1/4] Downloading SciFact dataset...")
    if not download_scifact():
        print("Failed to download. Exiting.")
        return
    
    print("\n[2/4] Loading dataset...")
    corpus, queries, qrels = load_scifact()
    doc_ids = list(corpus.keys())
    query_ids = list(queries.keys())
    
    print(f"  Loaded: {len(corpus)} docs, {len(queries)} queries")
    
    # Encode corpus and queries
    print("\n[3/4] Encoding corpus and queries...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = embedder.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}")
    
    # Encode all documents
    doc_texts = [(corpus[did]["title"] + " " + corpus[did]["text"]) for did in doc_ids]
    t0 = time.perf_counter()
    doc_embeddings = embedder.encode(doc_texts, show_progress_bar=False, batch_size=64,
                                    convert_to_numpy=True, normalize_embeddings=True)
    doc_embeddings_tensor = torch.from_numpy(doc_embeddings).float()
    encode_time = (time.perf_counter() - t0) * 1000
    print(f"  Corpus encode time: {encode_time:.0f} ms")
    
    # Select test queries (use queries with relevant documents in qrels)
    test_queries = []
    for qid in query_ids:
        if qid in qrels and len(qrels[qid]) >= 3:
            test_queries.append(qid)
    
    # Limit to 50 queries for better statistics
    test_queries = test_queries[:50]
    print(f"  Selected {len(test_queries)} queries with sufficient qrels")
    
    # Initialize MATHIR plugin
    print("\n[4/4] Running benchmark...")
    config = get_default_config()
    config["memory"]["internal_dim"] = 128
    config["memory"]["working_capacity"] = 64
    
    results_data = {
        "timestamp": time.time(),
        "num_queries": len(test_queries),
        "working_memory_capacity": 64,
        "query_results": [],
        "summary": {}
    }
    
    # Metrics accumulation
    mathir_overlaps = []
    faiss_overlaps = []
    mathir_score_shifts = []
    embedding_similarities_with_ctx = []
    embedding_similarities_without_ctx = []
    
    for qidx, qid in enumerate(test_queries):
        if qidx % 5 == 0:
            print(f"  Processing query {qidx+1}/{len(test_queries)}...")
        
        query_text = queries[qid]
        
        # Encode query
        query_emb = embedder.encode([query_text], show_progress_bar=False,
                                    convert_to_numpy=True, normalize_embeddings=True)
        query_emb_tensor = torch.from_numpy(query_emb).float()
        
        # Get context doc indices from qrels (positive relevance)
        context_indices = list(qrels[qid].keys())[:20]  # Top 20 related docs
        # Convert doc_id strings to indices
        context_idx_list = [doc_ids.index(did) for did in context_indices 
                           if did in doc_ids][:20]
        
        # === TEST 1: FAISS baseline (context-independent) ===
        faiss_results_no_ctx = run_faiss_baseline(query_emb, doc_embeddings, doc_ids, top_k=10)
        
        # === TEST 2: MATHIR WITHOUT working memory context ===
        plugin_no_ctx = MATHIRPluginV7(embedding_dim=dim, config=config)
        mathir_results_no_ctx = run_retrieval_with_mathir(
            plugin_no_ctx, query_emb_tensor.squeeze(0), doc_embeddings_tensor, doc_ids, top_k=10
        )
        
        # === TEST 3: MATHIR WITH working memory context ===
        plugin_with_ctx = MATHIRPluginV7(embedding_dim=dim, config=config)
        # Load context documents into working memory
        load_context_docs(plugin_with_ctx, doc_embeddings_tensor, doc_ids, context_idx_list)
        mathir_results_with_ctx = run_retrieval_with_mathir(
            plugin_with_ctx, query_emb_tensor.squeeze(0), doc_embeddings_tensor, doc_ids, top_k=10
        )
        
        # === Compute metrics ===
        # Result overlap (MATHIR with vs without context)
        mathir_ids_no_ctx = [doc_id for doc_id, _ in mathir_results_no_ctx]
        mathir_ids_with_ctx = [doc_id for doc_id, _ in mathir_results_with_ctx]
        mathir_overlap = compute_result_overlap(mathir_ids_with_ctx, mathir_ids_no_ctx)
        mathir_overlaps.append(mathir_overlap)
        
        # FAISS overlap (should be 1.0 since no context mechanism)
        faiss_ids = [doc_id for doc_id, _ in faiss_results_no_ctx]
        faiss_overlap = compute_result_overlap(faiss_ids, mathir_ids_no_ctx)
        faiss_overlaps.append(faiss_overlap)
        
        # Score shift for same documents
        score_shift = 0
        count = 0
        for doc_id in mathir_ids_with_ctx:
            if doc_id in mathir_ids_no_ctx:
                idx_with = mathir_ids_with_ctx.index(doc_id)
                idx_no = mathir_ids_no_ctx.index(doc_id)
                score_shift += abs(idx_with - idx_no)
                count += 1
        avg_score_shift = score_shift / max(count, 1)
        mathir_score_shifts.append(avg_score_shift)
        
        # Embedding similarity: compute enhanced embedding with vs without context
        plugin_empty = MATHIRPluginV7(embedding_dim=dim, config=config)
        # Ensure query has batch dimension
        q_empty = query_emb_tensor.squeeze(0).unsqueeze(0) if query_emb_tensor.dim() == 2 else query_emb_tensor
        out_empty = plugin_empty.perceive(q_empty)
        emb_empty = out_empty["enhanced_embedding"].squeeze(0)  # [D]
        
        plugin_loaded = MATHIRPluginV7(embedding_dim=dim, config=config)
        load_context_docs(plugin_loaded, doc_embeddings_tensor, doc_ids, context_idx_list)
        q_loaded = query_emb_tensor.squeeze(0).unsqueeze(0) if query_emb_tensor.dim() == 2 else query_emb_tensor
        out_loaded = plugin_loaded.perceive(q_loaded)
        emb_loaded = out_loaded["enhanced_embedding"].squeeze(0)  # [D]
        
        emb_sim = compute_cosine_similarity(emb_empty, emb_loaded)
        embedding_similarities_with_ctx.append(emb_sim)
        
        # Similarity of embedding without context vs with context (should be different if WM works)
        q_no_ctx = query_emb_tensor.squeeze(0).unsqueeze(0) if query_emb_tensor.dim() == 2 else query_emb_tensor
        q_with_ctx = query_emb_tensor.squeeze(0).unsqueeze(0) if query_emb_tensor.dim() == 2 else query_emb_tensor
        embedding_similarities_without_ctx.append(float(
            torch.nn.functional.cosine_similarity(
                plugin_no_ctx.input_proj(q_no_ctx),
                plugin_with_ctx.input_proj(q_with_ctx)
            ).item()
        ))
        
        # Store per-query results
        results_data["query_results"].append({
            "query_id": qid,
            "query_text": query_text,
            "context_docs": context_idx_list[:5],  # Just first 5 for brevity
            "mathir_overlap": mathir_overlap,
            "faiss_overlap": faiss_overlap,
            "avg_score_shift": avg_score_shift,
            "embedding_similarity": emb_sim,
        })
    
    # Compute summary statistics
    results_data["summary"] = {
        "mathir_mean_overlap": float(np.mean(mathir_overlaps)),
        "mathir_std_overlap": float(np.std(mathir_overlaps)),
        "faiss_mean_overlap": float(np.mean(faiss_overlaps)),
        "faiss_std_overlap": float(np.std(faiss_overlaps)),
        "mathir_mean_score_shift": float(np.mean(mathir_score_shifts)),
        "mean_embedding_similarity_with_ctx": float(np.mean(embedding_similarities_with_ctx)),
        "interpretation": ""
    }
    
    # Interpretation
    mean_overlap = results_data["summary"]["mathir_mean_overlap"]
    if mean_overlap > 0.9:
        interp = "WORKING MEMORY HAS MINIMAL EFFECT (overlap > 0.9)"
    elif mean_overlap > 0.7:
        interp = "WORKING MEMORY HAS MODERATE EFFECT (0.7 < overlap < 0.9)"
    elif mean_overlap > 0.5:
        interp = "WORKING MEMORY HAS SIGNIFICANT EFFECT (0.5 < overlap < 0.7)"
    else:
        interp = "WORKING MEMORY HAS STRONG EFFECT (overlap < 0.5)"
    
    # Compare to FAISS baseline
    if results_data["summary"]["faiss_mean_overlap"] > 0.95:
        results_data["summary"]["interpretation"] = (
            f"{interp}. FAISS baseline: {results_data['summary']['faiss_mean_overlap']:.3f} overlap "
            f"(context-independent). MATHIR: {mean_overlap:.3f} overlap. "
            f"The lower MATHIR overlap vs FAISS suggests working memory IS influencing retrieval."
        )
    else:
        results_data["summary"]["interpretation"] = (
            f"{interp}. FAISS baseline: {results_data['summary']['faiss_mean_overlap']:.3f} overlap. "
            f"MATHIR: {mean_overlap:.3f} overlap. Working memory effect is {'larger' if mean_overlap < results_data['summary']['faiss_mean_overlap'] else 'similar'} to FAISS."
        )
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nFAISS Baseline (no working memory):")
    print(f"  Mean overlap: {results_data['summary']['faiss_mean_overlap']:.4f}")
    print(f"  Std: {results_data['summary']['faiss_std_overlap']:.4f}")
    
    print(f"\nMATHIR Working Memory Effect:")
    print(f"  Mean overlap (with/without context): {mean_overlap:.4f}")
    print(f"  Std: {results_data['summary']['mathir_std_overlap']:.4f}")
    print(f"  Mean score shift: {results_data['summary']['mathir_mean_score_shift']:.2f}")
    print(f"  Mean embedding similarity (with ctx): {results_data['summary']['mean_embedding_similarity_with_ctx']:.4f}")
    
    print(f"\nInterpretation:")
    print(f"  {results_data['summary']['interpretation']}")
    
    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    print("\n" + "=" * 70)
    print("BROADCAST:")
    print(f"- Result overlap (with/without context): {mean_overlap:.4f} (lower = working memory helps)")
    print(f"- FAISS baseline: overlap = {results_data['summary']['faiss_mean_overlap']:.4f} (no context awareness)")
    print("=" * 70)


if __name__ == "__main__":
    main()