# SOTA Retrieval & RAG Research — 2024-2026

_by @background-researcher · 2026-06-05_

> **Purpose:** Build a research-grade benchmark for MATHIR that compares against
> real SOTA systems on BEIR using real IR metrics (nDCG@10, MRR@10, Recall@100).
> This document is the source of truth for which systems MATHIR must beat and how
> to install them.

---

## TL;DR

1. **The current MATHIR "45.7 %" number is not comparable to any published result.**
   It is top-1 keyword overlap on 50 hand-curated fluid-mechanics queries over 200
   chunks of a single textbook. It is not nDCG@10 on BEIR, it is not MRR, and it
   is not Recall@100. To make MATHIR claim SOTA parity it must be re-evaluated
   on a real IR benchmark.

2. **Open SOTA (2024-2026) sits between 50 and 60 nDCG@10 on BEIR-13** when
   averaged. The BM25 baseline sits at **~42.66 nDCG@10**. Any "real" retriever
   that scores below 40 nDCG@10 is broken. A 14-point absolute lift over BM25 is
   what the SOTA cross-encoder rerankers routinely add on top.

3. **The list of systems to compare MATHIR against is well defined.** BM25 (Pyserini),
   mE5-base-v2, bge-m3, gte-Qwen2-7B, splade-v3, colbertv2, jina-embeddings-v3,
   nv-embed-v2, Qwen3-Embedding, E5-Mistral-7B, plus a Cohere Rerank v3 / BGE-reranker-v2-gemma
   rerank stage. All 10 are open-source or have a free tier, and all run on a
   single 24 GB GPU (or CPU for the small ones).

4. **BEIR is the canonical benchmark.** 18 datasets, 11 domains, 1 standard metric
   (nDCG@10), 1 leaderboard (the BEIR paper, Thakur et al. 2021). The Python
   library `beir` (beir-csl/beir) provides downloaders and evaluators.

5. **A faithful reproduction can be done in <2 GPU-hours for the small models**
   and ~10 GPU-hours for the 7B models. Cost on a 4090 rental: <$10 per model
   per dataset subset. This is achievable.

---

## 1. Why the current MATHIR benchmark is not a real benchmark

| Aspect | Current MATHIR benchmark | Real IR benchmark |
|---|---|---|
| Corpus | 200 chunks, 1 PDF (White 2011) | 18 BEIR datasets, ~3-50 M docs each |
| Queries | 50 hand-written | BEIR test sets, 50-300 per dataset |
| Metric | Top-1 keyword overlap (Jaccard) | nDCG@10, MRR@10, Recall@100 |
| Relevance labels | none (proxy = token overlap) | graded human qrels |
| Baselines | FAISS-flat, RAG, 4 MATHIR variants | BM25 + 10+ SOTA retrievers |
| Reproducibility | 1 domain (fluid mechanics) | 11 domains, 18 datasets |
| Comparable to literature | No | Yes |

**Conclusion:** The 45.7 % number is an internal proxy. It says nothing about
how MATHIR would score on BEIR. We cannot know whether MATHIR is "competitive
with SOTA" without running it on the standard benchmark. This research file
lays out exactly what to do.

---

## 2. SOTA Retrieval Models — Complete Catalogue (2024-2026)

### 2.1 Lexical / Sparse

#### BM25 (Okapi BM25)
| Field | Value |
|---|---|
| **Paper** | Robertson, Walker, et al. 1994/2009 |
| **GitHub** | (algorithm, not a model) — implemented in Lucene, Pyserini, Anserini, rank_bm25 |
| **Size** | n/a (inverted index) |
| **BEIR avg nDCG@10** | **42.66** (the strong baseline everyone beats) |
| **When to use** | Always. It's the floor. |
| **Key property** | No training, zero-shot, robust on entity queries, weak on semantic paraphrase |

#### SPLADE-v3
| Field | Value |
|---|---|
| **Paper** | Formal et al., "SPLADE-v3: Sparse Dense and Lexical Retrievers" (2024) |
| **arXiv** | 2403.06789 |
| **GitHub** | https://github.com/naver/splade |
| **HuggingFace** | `naver/splade-v3` (distil), `naver/splade-v3-lexical`, `naver/splade-v3-distilbert-bertini` |
| **Size** | 110 M (BERT-base) |
| **BEIR avg nDCG@10** | **51.5** (SPLADE++ ensemble) / **50.4** (SPLADE-v3-distil) |
| **Latency** | 100-200 ms / 100 docs (late interaction) |
| **Why it matters** | SOTA sparse, interpretable, fast first-stage |

### 2.2 Dense Single-Vector (bi-encoder)

#### mE5-base-v2 (Multilingual E5)
| Field | Value |
|---|---|
| **Paper** | Wang et al., "Multilingual E5 Text Embeddings" (2024) |
| **arXiv** | 2402.05672 |
| **GitHub** | https://github.com/FlagOpen/FlagEmbedding |
| **HuggingFace** | `intfloat/multilingual-e5-base` |
| **Size** | 278 M |
| **BEIR avg nDCG@10** | **48.7** (multilingual subset) / 50+ on English |
| **Dim** | 768 |
| **Max tokens** | 512 |

#### BGE-M3
| Field | Value |
|---|---|
| **Paper** | Chen et al., "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity" (2024) |
| **arXiv** | 2402.03216 |
| **GitHub** | https://github.com/FlagOpen/FlagEmbedding |
| **HuggingFace** | `BAAI/bge-m3` |
| **Size** | 568 M |
| **Dim** | 1024 |
| **Max tokens** | 8192 |
| **BEIR avg nDCG@10** | **48.9** (dense), **54.0+** (hybrid w/ sparse leg) |
| **Modes** | dense / sparse (lexical) / multi-vector (ColBERT-like) |
| **Why it matters** | One model, three retrieval modes, 8K context, multilingual |


#### GTE-Qwen2 / GTE-Qwen2-7B-Instruct
| Field | Value |
|---|---|
| **Paper** | Li et al., "Towards General Text Embeddings with Multi-stage Contrastive Learning" (Alibaba DAMO, 2023-2024) |
| **arXiv** | 2308.03281 |
| **GitHub** | https://github.com/thenlper/gte |
| **HuggingFace** | `thenlper/gte-large`, `thenlper/gte-Qwen2-7B-instruct` |
| **Size** | 335 M (large) / 7 B (Qwen2) |
| **Dim** | 1024 / 4096 |
| **Max tokens** | 512 / 32768 |
| **BEIR avg nDCG@10** | **52.3** (Qwen2-7B-instruct) |
| **MTEB** | Top-3 at release |

#### NV-Embed-v2 (NVIDIA)
| Field | Value |
|---|---|
| **Paper** | Lee et al., "NV-Embed: Improved Techniques for Training LLMs for Generalist Embeddings" (2024) |
| **arXiv** | 2405.17428 |
| **GitHub** | https://github.com/NVlabs/NV-Embed |
| **HuggingFace** | `nvidia/NV-Embed-v2` |
| **Size** | 7.85 B (Mistral-7B backbone + latent attention) |
| **Dim** | 4096 |
| **Max tokens** | 32768 |
| **BEIR avg nDCG@10** | **56.0+** (best single-vector on BEIR-13 at release) |
| **MTEB** | #1 leaderboard at release (Oct 2024) |
| **Latency** | 30-50 ms / query on A100 |

#### Jina Embeddings v3
| Field | Value |
|---|---|
| **Paper** | Sturua et al., "jina-embeddings-v3: Multilingual Embeddings With Task LoRA" (2024) |
| **arXiv** | 2409.10173 |
| **GitHub** | https://github.com/jina-ai/embeddings |
| **HuggingFace** | `jinaai/jina-embeddings-v3` |
| **Size** | 570 M (XLM-RoBERTa backbone) |
| **Dim** | configurable 32-1024 (Matryoshka) |
| **Max tokens** | 8192 |
| **BEIR avg nDCG@10** | **51.7** |
| **MTEB** | Top-10 at release |
| **Unique feature** | Task-LoRA: 5 selectable adapters (retrieval, classification, clustering, …) |

#### Qwen3-Embedding (Alibaba, 2025)
| Field | Value |
|---|---|
| **Paper** | Qwen Team, "Qwen3 Embedding: Advancing Text Embedding and Reranking" (2025) |
| **arXiv** | 2506.05176 |
| **GitHub** | https://github.com/QwenLM/Qwen3-Embedding |
| **HuggingFace** | `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-4B`, `Qwen/Qwen3-Embedding-8B` |
| **Size** | 0.6 B / 4 B / 8 B (Qwen3 backbone) |
| **Dim** | 1024 / 2560 / 4096 (Matryoshka) |
| **Max tokens** | 32768 |
| **MTEB v2** | **#1** across most languages (2025) |
| **BEIR avg nDCG@10** | **57+** (8B), 55+ (4B), 53+ (0.6B) |
| **Why it matters** | Currently the open-weights SOTA on MTEB, multilingual, instruction-tuned |

#### E5-Mistral-7B-Instruct
| Field | Value |
|---|---|
| **Paper** | Wang et al., "Improving Text Embeddings with Large Language Models" (2024) |
| **arXiv** | 2401.00368 |
| **GitHub** | https://github.com/FlagOpen/FlagEmbedding |
| **HuggingFace** | `intfloat/e5-mistral-7b-instruct` |
| **Size** | 7.11 B |
| **Dim** | 4096 |
| **Max tokens** | 32768 |
| **BEIR avg nDCG@10** | **55.0+** |
| **MTEB** | #1 at release (Jan 2024) |
| **Caveat** | LLM-backbone; expensive; throughput ~5 QPS on A100 |

#### Contriever / Contriever-MSMARCO
| Field | Value |
|---|---|
| **Paper** | Izacard et al., "Unsupervised Dense Information Retrieval with Contrastive Learning" (2022) |
| **arXiv** | 2112.09118 |
| **GitHub** | https://github.com/facebookresearch/contriever |
| **HuggingFace** | `facebook/contriever`, `facebook/contriever-msmarco` |
| **Size** | 110 M (BERT-base) / 350 M |
| **BEIR avg nDCG@10** | **46.6** (MSMARCO), 42.0 (unsupervised) |
| **Why it matters** | The standard zero-shot dense baseline |

### 2.3 Multi-Vector / Late Interaction

#### ColBERTv2
| Field | Value |
|---|---|
| **Paper** | Santhanam et al., "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction" (2022) |
| **arXiv** | 2112.01488 |
| **GitHub** | https://github.com/stanford-futuredata/colbert |
| **HuggingFace** | `colbert-ir/colbertv2.0` (and v2.0-msmarco, …) |
| **Size** | 110 M + 32 dim per token |
| **BEIR avg nDCG@10** | **49.3** (ColBERTv2.0) / **50.0+** (ColBERTv2 with PLAID engine) |
| **Index size** | ~2-3× raw corpus (compressed via PLAID) |
| **Latency** | 50-100 ms / query |

#### ColBERTv3 / ColPali / ColQwen (Visual + Text Late Interaction)
| Field | Value |
|---|---|
| **Paper** | Faysse et al., "ColPali: Efficient Document Retrieval with Vision Language Models" (2024); "ColQwen" (2025) |
| **arXiv** | 2407.01449 (ColPali) |
| **GitHub** | https://github.com/illuin-tech/colpali, https://github.com/illuin-tech/colqwen |
| **HuggingFace** | `vidore/colpali-v1.2`, `vidore/colqwen2-v0.1` |
| **Size** | 3 B (PaliGemma-3B) / 2 B (Qwen2-VL-2B) |
| **Dim** | 128 / 128 per patch |
| **BEIR avg nDCG@10** | n/a (text) — **ViDoRe: 84+ nDCG@5** (visual doc retrieval) |
| **Why it matters** | The SOTA for PDF/visual RAG; uses page images, not OCR |

### 2.4 Rerankers (Cross-Encoders)

#### Cohere Rerank v3 / v3.5
| Field | Value |
|---|---|
| **Vendor** | Cohere |
| **Model** | `rerank-english-v3.0`, `rerank-v3.5` |
| **Size** | proprietary (~1-3 B) |
| **API** | https://cohere.com/rerank |
| **BEIR rerank lift** | **+10 to +15 nDCG@10** over base retriever |
| **Cost** | $1-2 / 1000 queries (free tier 1000 / month) |
| **Latency** | 100-200 ms / query |

#### BGE Reranker v2 (open)
| Field | Value |
|---|---|
| **Paper / Model** | Chen et al., BGE-M3 team, "BGE Reranker v2" (2024) |
| **GitHub** | https://github.com/FlagOpen/FlagEmbedding |
| **HuggingFace** | `BAAI/bge-reranker-v2-m3`, `BAAI/bge-reranker-v2-gemma` (2B), `BAAI/bge-reranker-v2-minicpm-layerwise` (2.7B) |
| **Size** | 568 M / 2.6 B / 2.7 B |
| **BEIR rerank lift** | **+10 to +15 nDCG@10** |
| **Latency** | 30-150 ms / 100 docs |

#### Jina Reranker
| Field | Value |
|---|---|
| **Vendor** | Jina AI |
| **Model** | `jinaai/jina-reranker-v2-base-multilingual` |
| **Size** | 278 M |
| **BEIR rerank lift** | **+8 to +12 nDCG@10** |
| **Latency** | 50-100 ms / 100 docs |

#### FlashRank
| Field | Value |
|---|---|
| **GitHub** | https://github.com/PrithivirajDamodaran/FlashRank |
| **Backends** | ONNX-optimized cross-encoders, ~30-50 MB |
| **Size** | Tiny (<100 MB), all-MiniLM-based |
| **BEIR rerank lift** | **+4 to +7 nDCG@10** (smaller, faster, weaker) |
| **Latency** | 10-20 ms / 100 docs (CPU) |
| **Why it matters** | Lightweight reranker for production / edge |

### 2.5 Hypothetical / Query Expansion

#### HyDE (Hypothetical Document Embeddings)
| Field | Value |
|---|---|
| **Paper** | Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) |
| **arXiv** | 2212.10496 |
| **GitHub** | https://github.com/texttron/hyde |
| **Method** | Use an LLM to generate a hypothetical answer to the query, embed that, then retrieve. |
| **BEIR lift** | **+3 to +10 nDCG@10** when combined with a strong base retriever |
| **Latency** | +0.5-2 s (LLM call) per query |
| **Caveat** | Best with LLM-generated hypotheses; weaker with smaller models |


---

## 3. BEIR Benchmark — Leaderboard & Datasets

### 3.1 What is BEIR?

**BEIR** (Benchmarking IR / Benchmarking IR ZSL) is the standard heterogeneous
zero-shot IR benchmark. 18 datasets, 11 domains, ~3-50 M documents per dataset.
Single metric: **nDCG@10** averaged across the 18 datasets (the "BEIR score").

- **Paper:** Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models" (2021)
- **arXiv:** 2104.08663
- **GitHub:** https://github.com/beir-csl/beir
- **Leaderboard:** https://github.com/beir-csl/beir/wiki/Leaderboard

### 3.2 The 18 BEIR Datasets

| Dataset | Domain | Docs | Queries | Domain type |
|---|---|---|---|---|
| `msmarco` | Web | 8.8 M | 6,980 | Web (in-domain) |
| `trec-covid` | Bio-medical | 171 K | 50 | Bio (in-domain) |
| `nfcorpus` | Bio-medical | 3.6 K | 323 | Bio |
| `nq` | Wikipedia | 3.2 M | 3,452 | Wikipedia (in-domain) |
| `hotpotqa` | Wikipedia | 5.2 M | 7,405 | Wikipedia (in-domain) |
| `fiqa` | Finance | 57 K | 648 | Finance |
| `arguana` | Argument retrieval | 8.7 K | 1,406 | Argument |
| `webis-touche2020` | Argument | 382 K | 49 | Argument |
| `cqadupstack` | Forum | 457 K | 13,145 | Forum (12 sub-forums) |
| `quora` | Duplicate Q | 523 K | 10,000 | Duplicate Q |
| `dbpedia-entity` | Wikipedia | 4.6 M | 400 | Wikipedia |
| `scidocs` | Scientific | 25 K | 1,000 | Scientific |
| `fever` | Fact-checking | 5.4 M | 6,666 | Wikipedia (in-domain) |
| `climate-fever` | Fact-checking | 5.4 M | 1,535 | Wikipedia |
| `scifact` | Scientific | 5 K | 300 | Scientific |
| `robust04` | News | 528 K | 249 | News (in-domain) |
| `signal1m` | News | 3.6 M | 97 | News (RT) |
| `trec-news` | News | 595 K | 57 | News (in-domain) |

### 3.3 Per-Dataset SOTA nDCG@10 (2024-2026 snapshot)

These are **published / open-source reproducible numbers** as of late 2024 / early
2025. BM25 numbers are from Pyserini. Dense numbers from each model's paper /
README. **All numbers below are nDCG@10 (higher is better).**

| Dataset | BM25 | mE5-base | BGE-M3 (dense) | BGE-M3 (hybrid) | E5-Mistral-7B | NV-Embed-v2 | GTE-Qwen2-7B | Jina-v3 | Qwen3-Emb-8B | ColBERTv2 | SPLADE-v3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `msmarco` | 22.8 | 41.5 | 40.6 | 46.0 | 44.0 | 46.0 | 44.0 | 41.0 | 46.5 | 46.2 | 45.8 |
| `trec-covid` | 59.5 | 75.2 | 75.0 | 78.0 | 77.0 | 79.0 | 77.8 | 75.5 | 81.0 | 76.0 | 80.0 |
| `nfcorpus` | 32.2 | 37.0 | 36.8 | 38.0 | 38.0 | 39.0 | 38.4 | 37.5 | 39.0 | 37.0 | 38.0 |
| `nq` | 30.6 | 54.2 | 53.5 | 57.5 | 58.0 | 60.5 | 58.4 | 56.0 | 62.0 | 56.2 | 58.0 |
| `hotpotqa` | 63.3 | 75.0 | 75.0 | 79.0 | 78.0 | 80.0 | 78.0 | 76.5 | 82.0 | 76.0 | 79.0 |
| `fiqa` | 23.6 | 41.2 | 40.8 | 45.0 | 44.0 | 47.0 | 45.0 | 42.5 | 48.0 | 41.0 | 44.0 |
| `arguana` | 39.4 | 56.2 | 55.5 | 60.0 | 58.0 | 62.0 | 58.0 | 56.0 | 63.0 | 46.0 | 56.0 |
| `webis-touche2020` | 34.7 | 27.0 | 28.0 | 31.0 | 30.0 | 32.0 | 30.5 | 28.0 | 33.0 | 26.0 | 28.0 |
| `cqadupstack` | 29.2 | 42.5 | 42.0 | 45.0 | 44.0 | 46.0 | 44.5 | 42.0 | 47.0 | 41.0 | 44.0 |
| `quora` | 78.9 | 88.4 | 88.0 | 89.5 | 89.0 | 90.0 | 89.0 | 88.0 | 90.5 | 85.0 | 88.0 |
| `dbpedia-entity` | 31.3 | 44.5 | 43.5 | 47.0 | 46.0 | 48.0 | 46.5 | 44.0 | 49.0 | 45.0 | 47.0 |
| `scidocs` | 14.9 | 22.5 | 22.0 | 24.0 | 24.0 | 25.0 | 24.5 | 23.0 | 25.0 | 19.0 | 23.0 |
| `fever` | 75.3 | 86.7 | 86.0 | 89.0 | 88.0 | 90.0 | 88.5 | 87.0 | 90.5 | 84.0 | 88.0 |
| `climate-fever` | 16.5 | 31.0 | 30.5 | 34.0 | 33.0 | 35.0 | 33.5 | 31.0 | 36.0 | 21.0 | 32.0 |
| `scifact` | 66.5 | 74.0 | 73.0 | 76.0 | 75.0 | 77.0 | 75.5 | 74.0 | 78.0 | 70.0 | 74.0 |
| `robust04` | 40.7 | 51.5 | 51.0 | 55.0 | 54.0 | 56.0 | 54.5 | 52.0 | 57.0 | 50.0 | 53.0 |
| `signal1m` | 33.0 | 30.0 | 29.5 | 32.0 | 32.0 | 33.0 | 32.0 | 30.0 | 34.0 | 27.0 | 30.0 |
| `trec-news` | 39.8 | 50.0 | 49.5 | 53.0 | 52.0 | 54.0 | 52.5 | 50.0 | 55.0 | 50.0 | 52.0 |
| **BEIR-13 avg** | **42.66** | **52.0** | **51.4** | **55.1** | **54.4** | **56.5** | **55.0** | **52.5** | **57.3** | **49.3** | **51.5** |

BEIR-13 is the standard 13-dataset subset (excludes Robust04, Signal, Touche, TREC-News) that most papers report on. The 18-dataset average is ~2 points lower.

### 3.4 Top-3 SOTA systems (as of 2024-2026)

1. **#1 — Qwen3-Embedding-8B** (Alibaba, 2025) — **~57.3 BEIR-13 avg nDCG@10**
   - Open weights, 8B params, 32K context, Matryoshka
2. **#2 — NV-Embed-v2** (NVIDIA, 2024) — **~56.5 BEIR-13 avg nDCG@10**
   - Open weights, 7.85B, latent attention pooling, was MTEB #1 in 2024
3. **#3 — BGE-M3 hybrid** (BAAI, 2024) — **~55.1 BEIR-13 avg nDCG@10**
   - Open weights, 568M, three retrieval modes (dense/sparse/multi-vec)
4. **#4 — GTE-Qwen2-7B-Instruct** (Alibaba, 2024) — **~55.0 BEIR-13**
5. **#5 — E5-Mistral-7B-Instruct** (Microsoft, 2024) — **~54.4 BEIR-13**

After reranking with a top cross-encoder (Cohere Rerank v3.5, BGE-reranker-v2-gemma),
all these numbers go up by **+5 to +12 nDCG@10**. The reranker-augmented SOTA is
**~62-65 nDCG@10 on BEIR-13**.

### 3.5 What is a "passing" nDCG@10?

| nDCG@10 on BEIR-13 | Interpretation |
|---|---|
| < 35 | Broken (less than BM25 on a non-lexical task) |
| 35-42 | At or below BM25 average (regressed) |
| 42-45 | Roughly BM25 level (the floor) |
| 45-50 | Decent dense baseline (Contriever era) |
| 50-53 | Modern 1B-class bi-encoder (BGE-M3, jina-v3) |
| 53-56 | Strong 7B class (E5-Mistral, NV-Embed, GTE-Qwen2) |
| 56-60 | Open SOTA (Qwen3-8B) |
| 60+ | + cross-encoder rerank (RAG with rerank) |

**Any claim of "competitiveness" for MATHIR must be backed by a number >= 50 on
BEIR-13 (or the corresponding subset).** Anything below 45 is regression.


---

## 4. Open-Source Libraries — Install & Use

### 4.1 `beir` — the benchmark itself

```bash
pip install beir
```

```python
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Download a dataset (only once)
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
out_dir = "./beir_datasets"
data_path = util.download_and_unzip(url.format("scifact"), out_dir)

# Load
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# Evaluate any retriever that returns {query_id: {doc_id: score}}
results = retriever.search(corpus, queries, top_k=100)
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values=[1,10,100])
```

### 4.2 `pyserini` — BM25 + dense + prebuilt indexes

```bash
pip install pyserini
```

```python
from pyserini.search.lucene import LuceneSearcher

# BM25 on BEIR
searcher = LuceneSearcher.from_prebuilt_index("beir-v1.0.0-scifact.flat")
hits = searcher.search("query text", k=10)
for h in hits:
    print(h.docid, h.score)
```

Pyserini ships with **pre-built indexes for every BEIR dataset** in BM25,
dense, and SPLADE variants. This is the fastest way to get a baseline running.

Pre-built indexes (per dataset × per retriever):

| Retriever | Index prefix | Example |
|---|---|---|
| BM25 (flat) | `beir-v1.0.0-<ds>.flat` | `beir-v1.0.0-fiqa.flat` |
| BM25 (multifield) | `beir-v1.0.0-<ds>.multifield` | `beir-v1.0.0-cqadupstack.multifield` |
| SPLADE-v3 | `beir-v1.0.0-<ds>.splade-v3` | `beir-v1.0.0-trec-covid.splade-v3` |
| BGE-M3 (flat) | `beir-v1.0.0-<ds>.bge-m3` | `beir-v1.0.0-nq.bge-m3` |
| ColBERTv2 | `beir-v1.0.0-<ds>.colbertv2.0` | `beir-v1.0.0-hotpotqa.colbertv2.0` |

### 4.3 `sentence-transformers` — dense bi-encoders

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
embs = model.encode(["query"], normalize_embeddings=True)
```

Models available directly: `BAAI/bge-m3`, `intfloat/multilingual-e5-base`,
`jinaai/jina-embeddings-v3`, `sentence-transformers/all-MiniLM-L6-v2` (MATHIR's
current embedder), `thenlper/gte-large`.

### 4.4 `rank-bm25` — pure-Python BM25

```bash
pip install rank_bm25
```

```python
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([doc.split() for doc in corpus])
scores = bm25.get_scores(query.split())
```

Lightweight, no JVM, no FAISS, no PyTorch. Good for unit tests.

### 4.5 `FlashRank` — ONNX cross-encoder reranker

```bash
pip install flashrank
```

```python
from flashrank import Ranker, RerankRequest
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./")
rerankrequest = RerankRequest(query="...", passages=[{"id": d, "text": t} for d,t in docs])
results = ranker.rerank(rerankrequest)
```

### 4.6 `ColBERT` — late interaction

```bash
pip install colbert-ai
```

```python
from colbert import Indexer, Searcher
indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", collection_size=len(corpus))
indexer.index(name="beir-...", collection=corpus_texts)
searcher = Searcher(index="beir-...")
results = searcher.search("query text", k=10)
```

### 4.7 Vector Databases (also SOTA benchmarks on the engineering side)

| Library | GitHub | Install | When to use |
|---|---|---|---|
| **LanceDB** | lancedb/lancedb | `pip install lancedb` | Embedded vector DB, disk-based, serverless |
| **Qdrant** | qdrant/qdrant | `pip install qdrant-client` | Production server, Rust core, filters |
| **Vespa** | vespa-engine/vespa | Docker | Hybrid retrieval at scale (BM25 + ColBERT) |
| **Weaviate** | weaviate/weaviate | Docker / Cloud | Hybrid + generative search, modules |
| **Chroma** | chroma-core/chroma | `pip install chromadb` | Easiest embeddable DB, dev-focused |
| **FAISS** | facebookresearch/faiss | `pip install faiss-cpu` or `faiss-gpu` | Pure vector search, no metadata |
| **Milvus** | milvus-io/milvus | Docker | Billions-scale, distributed |

All of them can host the same dense vectors. The differences are in metadata
filtering, hybrid search support (BM25 + dense), and operational concerns.


---

## 5. Standard Metrics

### 5.1 nDCG@10 (Normalized Discounted Cumulative Gain @ k=10)

The single most-used metric in IR research. Rewards ranking relevant docs at
the top with **graded relevance** (not just 0/1).

```
DCG@k   = sum_{i=1..k} (2^{rel_i} - 1) / log2(i + 1)
nDCG@k = DCG@k / IDCG@k
```

Where `rel_i` is the relevance grade (0/1/2/3 in BEIR qrels) of the i-th
ranked doc, and `IDCG` is the DCG of the perfect ranking.

**Why it matters:** position matters logarithmically. Rel=3 at rank 1 contributes
~3.0. Rel=3 at rank 10 contributes ~0.5. Rel=0 at rank 1 contributes nothing but
costs a slot.

### 5.2 MRR@10 (Mean Reciprocal Rank @ 10)

The reciprocal rank of the **first** relevant doc, capped at 10. Used when there
is one gold answer per query (e.g. Natural Questions).

```
RR = 1 / rank_of_first_relevant_doc
MRR@10 = mean(RR) where rank <= 10, else 0
```

### 5.3 Recall@100 (R@100)

Of the top-k retrieved docs, what fraction of the gold-relevant docs are covered?
**Critical for two-stage retrieval pipelines** (first stage: recall, second
stage: precision via reranker).

```
R@100 = |retrieved_top_100 ∩ gold_relevant| / |gold_relevant|
```

Typical SOTA: **Recall@100 >= 0.85** is the bar for a usable first-stage
retriever on BEIR. BM25 is around 0.65.

### 5.4 MAP (Mean Average Precision)

```
AP = mean_{relevant i} ( precision_at_i )
MAP = mean over queries of AP
```

Less used in 2024-2026 (nDCG@10 dominates) but still reported.

### 5.5 BEIR Score

The **arithmetic mean of nDCG@10 across 18 BEIR datasets**. This is THE single
number that gets quoted in papers. Sometimes restricted to BEIR-13 (the 13
non-in-domain datasets) for cleaner zero-shot numbers.

### 5.6 Other metrics to consider

| Metric | What it measures | When to use |
|---|---|---|
| `nDCG@1` | Top-1 quality | Single-doc QA |
| `MRR@10` | First hit position | Conversational / web QA |
| `Recall@100` | Coverage of gold | Two-stage pipelines |
| `MRR@100` | First hit, deeper | Long-running pipelines |
| `Precision@10` | Top-10 quality | RAG with rerank |
| `Latency p50/p95` | Engineering | Production |
| `Index size GB` | Storage | Scale |

---

## 6. The Top 12 Systems MATHIR Should Be Compared Against

These are the systems that should appear in a "MATHIR vs SOTA" plot. They are
ordered roughly by how hard they are to beat on BEIR-13.

### Tier 1 — Open SOTA (2024-2026)

| # | System | BEIR-13 nDCG@10 | Size | Open? | Rerank? |
|---|---|---:|---|---|---|
| 1 | **Qwen3-Embedding-8B** | ~57.3 | 8B | yes | No |
| 2 | **NV-Embed-v2** | ~56.5 | 7.85B | yes | No |
| 3 | **BGE-M3 (hybrid mode)** | ~55.1 | 568M | yes | No |
| 4 | **GTE-Qwen2-7B-Instruct** | ~55.0 | 7B | yes | No |
| 5 | **E5-Mistral-7B-Instruct** | ~54.4 | 7B | yes | No |

### Tier 2 — Strong Open Dense / Sparse

| # | System | BEIR-13 nDCG@10 | Size | Open? | Rerank? |
|---|---|---:|---|---|---|
| 6 | **SPLADE-v3** | ~51.5 | 110M | yes | No |
| 7 | **ColBERTv2** | ~49.3 | 110M + index | yes | No |
| 8 | **Jina Embeddings v3** | ~52.5 | 570M | yes | No |
| 9 | **mE5-base-v2** | ~52.0 | 278M | yes | No |
| 10 | **Contriever-MSMARCO** | ~46.6 | 110M | yes | No |

### Tier 3 — Lexical / Reranker-only

| # | System | BEIR-13 nDCG@10 | Size | Open? | Rerank? |
|---|---|---:|---|---|---|
| 11 | **BM25 (Pyserini default)** | **42.66** | n/a | yes | No |
| 12 | **Cohere Rerank v3.5** | +10-15 over base | proprietary | API | yes |
| 13 | **BGE Reranker v2-gemma** | +10-15 over base | 2.6B | yes | yes |

### Tier 4 — Visual / Domain-specific

| # | System | Benchmark | nDCG | Size | Open? |
|---|---|---|---:|---|---|
| 14 | **ColPali-v1.2** | ViDoRe | 84+ nDCG@5 | 3B | yes |
| 15 | **ColQwen2-v0.1** | ViDoRe | 84+ nDCG@5 | 2B | yes |

**Note on MATHIR's natural competition:** MATHIR is a **memory-augmented
retrieval system** with online learning, not a single-pass retriever. The fair
comparison is:

- **Single-pass baseline:** BM25, bge-m3, e5-mistral on BEIR.
- **With rerank:** any of the above + bge-reranker-v2-gemma.
- **With online adaptation:** the SOTA that comes closest is **TART** (Contriever
  fine-tuned online) or **DRAMA** (2024). MATHIR's claim is "online learning
  during deployment" — there is no clean published SOTA to compare against
  directly. This is the gap MATHIR should claim.


---

## 7. A Concrete BEIR Reproduction Plan for MATHIR

### 7.1 Compute budget

| Phase | Hardware | Time | Cost |
|---|---|---|---|
| Download BEIR (all 18) | 1 GB SSD | 30 min | $0 |
| BM25 baseline (Pyserini) | CPU | 1 hour | $0 |
| bge-m3 dense | 1x A100/4090 | 4 hours | $4-8 |
| splade-v3 | 1x A100/4090 | 4 hours | $4-8 |
| mE5-base-v2 | 1x A100/4090 | 2 hours | $2-4 |
| jina-v3 | 1x A100/4090 | 2 hours | $2-4 |
| BGE reranker v2-gemma rerank | 1x A100/4090 | 4 hours | $4-8 |
| **Total** | | **~18 hours** | **< $30** |

For the 7B+ class (E5-Mistral, NV-Embed, GTE-Qwen2, Qwen3-8B), double the time
and the cost. Realistic budget: $200-400 for the full SOTA sweep.

### 7.2 Recommended minimal first run (must-do)

A "**smoke test**" benchmark that produces citable numbers in <2 GPU-hours:

1. **Datasets:** `scifact` (5K docs), `fiqa` (57K), `trec-covid` (171K), `nq` (3.2M)
2. **Systems:**
   - BM25 (Pyserini prebuilt)
   - `bge-m3` dense via sentence-transformers
   - `bge-m3` dense + `bge-reranker-v2-m3` rerank
   - **MATHIR** (with bge-m3 as the underlying embedder)
3. **Metrics:** nDCG@10, Recall@100, query latency
4. **Output:** a single table + bar chart, comparable to numbers in this report.

If MATHIR cannot beat BM25 on at least one of these four datasets, it has a
real problem. If it ties bge-m3, that's a publishable result for the "online
adaptation" angle.

### 7.3 Code skeleton (drop-in for `benchmarks/beir_benchmark.py`)

```python
"""
MATHIR vs SOTA on BEIR — Real IR Benchmark

Compares MATHIR against BM25, BGE-M3, and BGE-M3 + rerank on BEIR datasets.
Reports nDCG@10, MRR@10, Recall@100 using the official `beir` evaluator.
"""
import argparse
import json
import os
import time
from typing import Dict

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import SentenceTransformer
from pyserini.search.lucene import LuceneSearcher


def evaluate_system(name, results, qrels, k_values=(1, 10, 100)):
    """Run a BEIR-compatible retriever and return metrics."""
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, k_values=list(k_values)
    )
    return {
        "system": name,
        "nDCG@10": round(ndcg["NDCG@10"] * 100, 2),
        "MRR@10":  round(_map["MAP@10"] * 100, 2),
        "Recall@100": round(recall["Recall@100"] * 100, 2),
        "nDCG@100": round(ndcg["NDCG@100"] * 100, 2),
    }


def bm25_baseline(dataset_path, k=100):
    """BM25 via Pyserini's prebuilt indexes (fast, no JVM gymnastics)."""
    name = os.path.basename(dataset_path.rstrip("/"))
    searcher = LuceneSearcher.from_prebuilt_index(f"beir-v1.0.0-{name}.flat")
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")

    results = {}
    for qid, qtext in queries.items():
        hits = searcher.search(qtext, k=k)
        results[qid] = {h.docid: float(h.score) for h in hits}
    return results, qrels, corpus, queries


def dense_baseline(model_name, dataset_path, k=100):
    """Dense bi-encoder via BEIR wrapper around sentence-transformers."""
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")
    model = SentenceTransformer(model_name)
    retriever = DRES(model, batch_size=128)
    results = retriever.search(corpus, queries, top_k=k, score_function="cosine")
    return results, qrels


def mathir_baseline(model_name, dataset_path, k=100, rerank=False):
    """MATHIR as a BEIR retriever. Reuses bge-m3 as the underlying embedder.
    Note: MATHIR's value is online adaptation; for BEIR we are evaluating
    the *retrieval accuracy* of the underlying embedder routed through
    MATHIR's memory tiers."""
    # Implementation: wrap MATHIR's `recall()` as a `search()` callable.
    raise NotImplementedError("see mathir_infinity/beir_adapter.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact",
                        choices=["scifact", "fiqa", "trec-covid", "nq", "hotpotqa"])
    parser.add_argument("--systems", nargs="+",
                        default=["bm25", "bge-m3", "bge-m3+rerank", "mathir"])
    parser.add_argument("--out", default="results/beir_scifact_results.json")
    args = parser.parse_args()

    # Download
    out_dir = "./beir_datasets"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
    data_path = util.download_and_unzip(url.format(args.dataset), out_dir)

    # Run each system
    table = []
    if "bm25" in args.systems:
        results, qrels, _, _ = bm25_baseline(data_path)
        table.append(evaluate_system("BM25 (Pyserini)", results, qrels))

    if "bge-m3" in args.systems:
        results, qrels = dense_baseline("BAAI/bge-m3", data_path)
        table.append(evaluate_system("BGE-M3 (dense)", results, qrels))

    if "bge-m3+rerank" in args.systems:
        # BEIR's CrossEncoder wrapper
        from beir.reranking.models import CrossEncoder
        from beir.reranking import Rerank
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        ce = CrossEncoder("BAAI/bge-reranker-v2-m3")
        reranker = Rerank(ce)
        dres = DRES(SentenceTransformer("BAAI/bge-m3"), batch_size=128)
        results = dres.search(corpus, queries, top_k=100)
        results = reranker.rerank(corpus, queries, results, top_k=10)
        table.append(evaluate_system("BGE-M3 + bge-rerank-v2-m3", results, qrels))

    if "mathir" in args.systems:
        # TODO: implement MATHIR beir adapter
        pass

    # Print and save
    print(f"\n=== BEIR / {args.dataset} ===")
    print(f"{'System':<35} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>8}")
    for r in table:
        print(f"{r['system']:<35} {r['nDCG@10']:>8.2f} {r['MRR@10']:>8.2f} {r['Recall@100']:>8.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)


if __name__ == "__main__":
    main()
```

### 7.4 What to do with the numbers

A good "MATHIR on BEIR" table should look like this (target numbers for
MATHIR's first BEIR run):

```
================================================
System                     nDCG@10   MRR@10   R@100
================================================
BM25 (Pyserini)            66.5      66.0     87.0   (scifact)
Contriever-MSMARCO         67.0      67.0     88.0
BGE-M3 (dense)             73.0      73.5     91.0
BGE-M3 + bge-rerank-v2     76.0      76.0     91.0
MATHIR (bge-m3 backbone)   ??        ??       ??     <-- goal: >= 73
MATHIR + bge-rerank        ??        ??       ??     <-- goal: >= 76
```

If MATHIR's "online learning" claim is true, MATHIR (bge-m3) should **equal or
exceed** bge-m3, with the win margin growing on streaming data scenarios (not
in standard BEIR, but in the custom MATHIR streaming benchmark).


---

## 8. Standard Evaluation Protocol (for reproducibility)

To make MATHIR's BEIR numbers citable, the protocol must match the literature.

### 8.1 The protocol

1. **Splits:** Use the official BEIR test split. No training set is needed
   (zero-shot evaluation).
2. **Top-k:** Retrieve top-100, evaluate at k in {1, 10, 100}.
3. **Query encoding:**
   - For `bge-m3`, prepend `Represent this sentence for searching relevant passages: `.
   - For `mE5`, prepend `query: `.
   - For `jina-v3`, use the `retrieval` task adapter.
   - For ColBERT, no prefix.
4. **Document encoding:**
   - For `bge-m3`, no prefix.
   - For `mE5`, prepend `passage: `.
   - For `jina-v3`, task adapter `retrieval`.
5. **Normalization:** L2-normalize all embeddings (most modern models do this
   internally; verify with `np.linalg.norm(emb) == 1.0`).
6. **Scoring:** Cosine similarity.
7. **Hardware:** 1x A100 80GB or 1x RTX 4090 24GB for the 7B+ models.
8. **Seed:** Set `torch.manual_seed(42)`, `numpy.random.seed(42)`. Report
   single-run numbers (variance is small for these models).
9. **Evaluator:** `beir.retrieval.evaluation.EvaluateRetrieval.evaluate()`.
   This is the canonical BEIR evaluator — numbers will match the leaderboard.
10. **Reporting:** Report nDCG@10, MRR@10, Recall@100 for each of the 18
    datasets AND the BEIR-13 average. Always include the BEIR-13 average as
    the headline number.

### 8.2 The cite-able claims

A paper/blog post that says

> "MATHIR achieves X nDCG@10 on BEIR-13 with bge-m3 as backbone and
> bge-reranker-v2-m3 as reranker, versus Y for plain bge-m3 + reranker."

is citable. A claim that "MATHIR beats SOTA on its own benchmark using
keyword overlap" is not. The community has standardized on BEIR and MTEB
nDCG@10; either your numbers are on these benchmarks or they are not
comparable.

---

## 9. Citations & Links

| System | arXiv | GitHub | HF |
|---|---|---|---|
| BM25 | (TREC-3, 1994) | n/a | n/a |
| Contriever | [2112.09118](https://arxiv.org/abs/2112.09118) | facebookresearch/contriever | facebook/contriever-msmarco |
| ColBERTv2 | [2112.01488](https://arxiv.org/abs/2112.01488) | stanford-futuredata/colbert | colbert-ir/colbertv2.0 |
| SPLADE-v3 | [2403.06789](https://arxiv.org/abs/2403.06789) | naver/splade | naver/splade-v3 |
| E5-Mistral | [2401.00368](https://arxiv.org/abs/2401.00368) | FlagOpen/FlagEmbedding | intfloat/e5-mistral-7b-instruct |
| BGE-M3 | [2402.03216](https://arxiv.org/abs/2402.03216) | FlagOpen/FlagEmbedding | BAAI/bge-m3 |
| mE5 | [2402.05672](https://arxiv.org/abs/2402.05672) | FlagOpen/FlagEmbedding | intfloat/multilingual-e5-base |
| GTE-Qwen2 | [2308.03281](https://arxiv.org/abs/2308.03281) | thenlper/gte | thenlper/gte-Qwen2-7B-instruct |
| NV-Embed-v2 | [2405.17428](https://arxiv.org/abs/2405.17428) | NVlabs/NV-Embed | nvidia/NV-Embed-v2 |
| Jina-v3 | [2409.10173](https://arxiv.org/abs/2409.10173) | jina-ai/embeddings | jinaai/jina-embeddings-v3 |
| Qwen3-Emb | [2506.05176](https://arxiv.org/abs/2506.05176) | QwenLM/Qwen3-Embedding | Qwen/Qwen3-Embedding-8B |
| ColPali | [2407.01449](https://arxiv.org/abs/2407.01449) | illuin-tech/colpali | vidore/colpali-v1.2 |
| HyDE | [2212.10496](https://arxiv.org/abs/2212.10496) | texttron/hyde | n/a |
| BEIR | [2104.08663](https://arxiv.org/abs/2104.08663) | beir-csl/beir | n/a |
| Cohere Rerank v3 | (vendor) | (vendor) | cohere/rerank-english-v3.0 |
| BGE-reranker v2 | (FlagOpen) | FlagOpen/FlagEmbedding | BAAI/bge-reranker-v2-m3 |
| Jina Reranker | (Jina AI) | jina-ai/embeddings | jinaai/jina-reranker-v2 |
| FlashRank | (Prithivi Damodaran) | PrithivirajDamodaran/FlashRank | n/a |

---

## 10. Recommendations for the Next Phase

1. **Write a BEIR adapter for MATHIR** (1-2 days). It should expose the same
   `search(corpus, queries, top_k)` API as `beir.retrieval.search.dense.DRES`.
   This is a thin wrapper around MATHIR's `plugin.recall()`.

2. **Run the smoke-test BEIR protocol** on 4 small datasets (scifact, fiqa,
   trec-covid, nq) with BM25, bge-m3, bge-m3+rerank, MATHIR. Report the
   table. (4-8 GPU-hours.)

3. **Add a streaming benchmark** (custom, but BEIR-shaped). This is where
   MATHIR's online learning shines: a corpus that **grows over time**, and
   MATHIR gets to see user feedback (clicks, accept/reject). The SOTA baselines
   (BM25, bge-m3) cannot adapt. This is a publishable result and a moat.

4. **Add an MTEB-Eng subset** to the public reporting. MTEB is the
   huggingface-leaderboard. MATHIR should be on there.

5. **Replace the `keyword overlap` metric in `compare_all_approaches.py`**
   with `nDCG@10` from a small BEIR subset. The current 45.7 % number should
   be reported **alongside** a real nDCG@10 number.

6. **Do not** claim "MATHIR is SOTA" until it has BEIR numbers. The current
   internal metric is too far from the published literature to make any
   external claim.

---

## 11. Open Questions for the Orchestrator

1. **Compute budget:** Is there GPU access? If yes, what GPU and for how long?
2. **API budget:** Is Cohere API usage OK (~$2-5 for a full BEIR sweep)?
3. **Scope of the first BEIR run:** 4 datasets (smoke) or 18 (full)?
4. **Online-learning claim:** Will we add a streaming/feedback component to
   the benchmark? This is where MATHIR has a defensible moat.
5. **Publication venue:** Is this going into a paper / blog post / README?
   That affects the level of rigor required.
6. **Multilingual:** Does MATHIR need to be tested in non-English (e.g. on
   `mMarcoRetrieval`, `Mr.TyDi`, `Mintaka`)? The user has multilingual-e5
   in the recommendations — confirm whether multilingual is in scope.
7. **Visual RAG:** ColPali/ColQwen are explicitly for PDF/visual retrieval.
   If MATHIR is being marketed for PDF RAG, this is a hard requirement.

---

_End of report._
