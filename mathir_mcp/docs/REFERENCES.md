# MATHIR — Academic References & Research Papers

Papers cited in the MATHIR codebase, grouped by algorithm/feature.

---

## 3-Layer Auto-Cache and Reliability Hardening (v8.9.8)

### L1 Embedding Cache (LRU)

- **O'Neil, E.J., O'Neil, P.E., & Weikum, G. (1993).** "The LRU-K Page Replacement Algorithm for Database Disk Buffering." *Proceedings of the 1993 ACM SIGMOD International Conference on Management of Data*, pp. 297-306.
  - Foundational LRU variant for buffer management. MATHIR's L1 uses classic LRU (K=1) on embedding vectors — deterministic outputs make embeddings ideal LRU candidates.

- **Cao, P. & Irani, S. (1999).** "Cost-Aware WWW Proxy Caching Algorithms." *Proceedings of USENIX USITS*.
  - Cost-aware caching where items have different retrieval costs. Directly applicable: embedding encode (~60ms) vs cached lookup (<1ms) = 60x cost differential justifies aggressive caching.

### L2 Recall Cache (TTL + Write-Invalidation)

- **Nishtala, R., Fugal, H., Grimm, S., et al. (2013).** "Scaling Memcache at Facebook." *Proceedings of the 10th USENIX NSDI*, pp. 385-398.
  - Real-world multi-tier caching at scale with write-through invalidation. Same pattern used by MATHIR's `invalidate_on_write()` — any mutation (save/delete/promote/consolidate) clears the recall cache.

### L3 Session Pre-Warm (Working Set)

- **Denning, P.J. (1968).** "The Working Set Model for Program Behavior." *Communications of the ACM*, 11(5), pp. 323-333. DOI: `10.1145/363095.363141`
  - Classic paper establishing that programs access a small, stable "working set." MATHIR's L3 pre-loads top-20 memories per project — an agent's hot memories are a small subset of the corpus.

- **Denning, P.J. (1980).** "Working Sets Past and Present." *IEEE Transactions on Software Engineering*, SE-6(1), pp. 64-84. DOI: `10.1109/TSE.1980.230464`
  - Retrospective survey of working-set theory with extensions. Provides theoretical backing for session-locality assumptions.

### Multi-Tier Architecture

- **Che, H., Tung, Y., & Wang, Z. (2002).** "Hierarchical Web Caching Systems: Modeling, Design and Experimental Results." *IEEE Journal on Selected Areas in Communications*, 20(7), pp. 1305-1314. DOI: `10.1109/JSAC.2002.801752`
  - Models multi-tier caching with different sizes/policies per tier. Directly analogous to MATHIR's L1 (1024 LRU) / L2 (256 TTL) / L3 (top-20/project) architecture.

---

## Ebbinghaus Decay

- **Ebbinghaus, H. (1885).** *Uber das Gedachtnis: Untersuchungen zur experimentellen Psychologie*. Leipzig: Duncker & Humblot. (English translation: Ruger & Bussenius, 1913, Teachers College, Columbia University.)
  - Original forgetting curve. MATHIR implements Ebbinghaus decay as -5% stability per 30 days without recall, with a floor of 0.05 to prevent total deletion.

---

## Spreading Activation

- **Collins, A.M. & Loftus, E.F. (1975).** "A Spreading-Activation Theory of Semantic Processing." *Psychological Review*, 82(6), pp. 407-428. DOI: `10.1037/0033-295X.82.6.407`
  - Foundational paper for semantic network traversal. MATHIR's `mathir_spread.py` implements activation propagation through the memory link graph with configurable depth and decay.

---

## Hybrid Search (Vector + BM25 + RRF)

- **Robertson, S.E. & Zaragoza, H. (2009).** "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 3(4), pp. 333-389. DOI: `10.1561/1500000019`
  - Comprehensive treatment of BM25. MATHIR uses Okapi BM25 as the lexical component of hybrid search.

- **Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009).** "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." *Proceedings of the 32nd ACM SIGIR*, pp. 758-759. DOI: `10.1145/1571941.1572114`
  - Establishes RRF as a simple, effective rank fusion method. MATHIR uses RRF (k=60) to combine vector cosine and BM25 ranked lists.

---

## Cross-Encoder Reranking

- **Nogueira, R. & Cho, K. (2020).** "Passage Re-ranking with BERT." *arXiv preprint arXiv:1901.04085*.
  - Foundational work on using cross-encoders for reranking. MATHIR uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params) as an optional second-pass reranker (+20pp hit@10).

---

## Vector Quantization & Similarity Search

- **Guo, R., et al. (2020).** "Accelerating Large-Scale Inference with Anisotropic Vector Quantization." *Proceedings of the 37th ICML*.
  - ScaNN paper on vector quantization for faster retrieval. Conceptually related to MATHIR's INT8 scalar quantization (4x compression, zero recall loss).

- **Johnson, J., Douze, M., & Jegou, H. (2021).** "Billion-Scale Similarity Search with GPUs." *IEEE Transactions on Big Data*, 7(3), pp. 535-547. DOI: `10.1109/TBDATA.2019.2921572`
  - FAISS — foundational work on efficient similarity search. MATHIR uses sqlite-vec but shares the same indexing principles.

---

## Anomaly Detection (Immunological Tier)

- **Mahalanobis, P.C. (1936).** "On the Generalized Distance in Statistics." *Proceedings of the National Institute of Sciences of India*, 2(1), pp. 49-55.
  - Original Mahalanobis distance formulation. MATHIR's immunological tier uses Mahalanobis distance (threshold=25.0) to detect anomalous embeddings (prompt injections, threat signatures).

---

## Memory-Augmented LLMs (Related Work)

- **Zhong, W., Guo, L., Gao, Q., Ye, H., & Wang, Y. (2024).** "MemoryBank: Enhancing Large Language Models with Long-Term Memory." *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(17), pp. 19724-19731. DOI: `10.1609/aaai.v38i17.29946`
  - Implements Ebbinghaus forgetting curve for LLM memory. Closest published system to MATHIR's decay mechanism. MATHIR extends this with five adaptive tiers, a push-based guardrail tier, link graphs, and anomaly detection.

- **Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S., Stoica, I., & Gonzalez, J. (2024).** "MemGPT: Towards LLMs as Operating Systems." *arXiv preprint arXiv:2310.08560*.
  - Virtual memory management for LLM context (paging between main context and external storage). MATHIR's five adaptive tiers form an analogous multi-level memory hierarchy; guardrails intentionally bypass adaptive routing.

- **Graves, A., Wayne, G., & Danihelka, I. (2014).** "Neural Turing Machines." *arXiv preprint arXiv:1410.5401*.
  - Early work on external memory for neural networks with read/write heads. Foundational concept for memory-augmented AI systems.

---

## MATHIR Benchmark Results

| Benchmark | Result | Reference |
|---|---|---|
| **Auto-cache speedup** | Cold 37ms -> cached 2ms (**18x**) | v8.7.0 live benchmark |
| **INT8 quantization** | 4x compression, 0% recall loss | v8.6.0, 410 DBs migrated |
| **Cross-encoder reranking** | +20pp hit@10 (50% -> 70%) | v8.6.0 |
| **Multi-agent memory sharing** | 0% -> 53% accuracy with MATHIR | v8.6.0, 3-phase benchmark |
| **Hybrid search (RRF)** | +15pp vs vector-only | v8.5.0 |
| **e5-small + rerank vs e5-large** | 52.9% vs 51.0% at 47x less cost | v8.6.0 |
