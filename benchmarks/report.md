# MATHIR — Handoff Report for Agent Army

**Date:** 2026-07-01
**Written by:** Opencode agent, for other agents to pick up where this session stopped (blocked on LLM API credits, otherwise continuing).
**Read this first**, then go deep on `mathir_mcp/docs/DIMENSIONS.md` for the full technical detail on the retrieval-quality investigation — this report is the map, DIMENSIONS.md is the territory.

---

## 1. What MATHIR is (context for agents with no prior session memory)

MATHIR is a 5-tier cognitive memory system exposed via MCP (Model Context Protocol) — a local daemon (`mathir_mcp/mathir_lib/mathir_server.py`, Flask+Waitress, port 7338) that any coding agent connects to via 24 MCP tools (`memory_save`, `memory_recall`, `memory_hybrid_search`, lifecycle tools, etc.). Backed by SQLite + sqlite-vec for vector search. Explicitly targets **local/edge deployment, no cloud dependency, low resource footprint** — this constraint shaped several decisions below (e.g. we did NOT just swap to a bigger embedding model).

**Read the `mathir-vision-vs-verified-reality-v2` memory in MATHIR itself (`memory_recall` for that label) before assuming any of the following are proven or unproven.** Important nuance on cross-provider portability (MATHIR's headline claim): the CORE mechanism — different LLM-driven agents/tools sharing and accessing the same MATHIR memory DB — IS real and demonstrated (architecturally guaranteed, since the daemon computes all embeddings server-side and the calling LLM never touches them; confirmed by `mathir_dropin/tests/test_multi_agent.py::test_cross_model_plug_and_play`; confirmed by real cross-tool usage across Claude Code/MimoCode/OpenCode with Claude/MiMo/MiniMax backends on this same project's memory). What's still unverified is the NARROWER question of cross-model *answer-consistency* (does model X extract the same correct answer from retrieved memories as model Y) — the `--cross-model` test harness in section 5 below was built for exactly that and hasn't been run yet, blocked on credits. Don't conflate the two. Also still unverified: hallucination reduction (never measured), token reduction (never measured), and "auto-learning" via the promotion/decay/consolidation lifecycle (code exists in `mathir_vec.py` but was never mathematically validated or benchmarked this session). This report and DIMENSIONS.md only cover raw retrieval quality on academic BEIR corpora — that is one slice of MATHIR's vision, not the whole thing.

---

## 2. Real bugs found and fixed this session (already committed, working)

These are done — don't re-investigate them, but know they happened:

1. **Selftest bug**: `mathir_mcp_server.py` never defined `TOOLS`, breaking `--selftest`'s tool-count check. Fixed with `get_tools_info()`.
2. **Anomaly detector (`immunological` tier) was fully disconnected**: the 5th memory tier existed in schema/dashboard but nothing ever wrote to it; the only real Mahalanobis-distance code lived in a dead, unused module. Built a real one (`mathir_lib/mathir_anomaly.py`), wired it into `/api/memory/save`, added `memory_audit_immunological` MCP tool. Found and fixed a **real math bug** along the way: fixed-epsilon covariance regularization doesn't fix rank-deficiency when sample count ≈ embedding dimension (n≈d is the worst case for covariance estimation) — replaced with adaptive Ledoit-Wolf-style shrinkage. Real, honest eval: AUC-ROC=0.8533 on a realistic prompt-injection corpus (not the old disavowed "AUC=1.0" claim from a deprecated module tested on trivially-easy synthetic data).
3. **`/api/memory/hybrid_search` rebuilt its entire BM25 index from scratch on every single query** — no caching at all. Measured cost: 566s for a 5183-doc query set vs 17s for an equivalent cached implementation (~30x, purely architectural, zero correctness benefit). Fixed with a row-count-invalidated cache. This was found via a live MATHIR-vs-FAISS stress benchmark another agent instance ran in parallel (`benchmarks/09_mathir_vs_faiss_stress/` — check that directory too, it has its own findings from a separate concurrent investigation).
4. **Marketing claims didn't match shipped code**: README/OUTREACH docs said the anomaly detector wasn't wired to the MCP server (true before this session, false after) — corrected. Version mismatch between `pyproject.toml` and `__init__.py` — fixed.

---

## 3. The core open question: why does MATHIR trail FAISS on real BEIR benchmarks?

Real, honest benchmark data (see `benchmarks/06_results/current/` and `benchmarks/09_mathir_vs_faiss_stress/`) shows MATHIR's `hybrid_search` losing to plain FAISS dense-only search on scifact and nfcorpus. This session investigated **why**, methodically, with real local experiments (no LLM API needed for any of this — pure embeddings + linear algebra):

### 3a. Ruled out: MATHIR's search mechanism itself

Holding the embedder fixed and comparing raw FAISS `IndexFlatIP` against `VecMemory.search()`'s real code path on **identical embeddings** (nfcorpus) produced an **identical nDCG@10** (0.2345 = 0.2345). sqlite-vec's exact brute-force cosine search is mathematically equivalent to FAISS at these corpus sizes (thousands of docs). **No bug here.** (It IS ~500x slower per-query in this specific unindexed comparison — a real performance gap, not a quality one, and separate from the BM25-cache fix above.)

Script: `benchmarks/07_utilities/isolate_mathir_retrieval_bug.py`

### 3b. Ruled out: RRF fusion weight misconfiguration

`hybrid_search`'s default weights (vector_weight=1.0, bm25_weight=1.0) are already near-optimal for the CURRENT embedder. Sweeping vector_weight from 1 to 10 made nfcorpus nDCG@10 *worse* (0.3056 → 0.2583), not better.

Script: `benchmarks/07_utilities/test_rrf_weights.py`

### 3c. The real, validated factor: embedding model choice

The current default (`paraphrase-multilingual-MiniLM-L12-v2`, 384d, trained for paraphrase/STS similarity, NOT retrieval) is genuinely weaker at retrieval than same-footprint alternatives trained specifically for retrieval. Tested `intfloat/multilingual-e5-small` (same 384d, same param count) head-to-head:

| Dataset | Current default | e5-small | Verdict |
|---|---|---|---|
| scifact | 0.4837 | **0.6770** | +40% |
| nfcorpus | 0.2345 | **0.3100** | +32% |
| arguana | **0.4488** | 0.3908 | -13% |

Not a clean win — e5-small is also ~5x slower to encode (matters for the edge-device constraint), and loses on arguana (argument-similarity retrieval, where the current model's paraphrase training is actually the better fit). **Default was NOT changed** — documented as an opt-in alternative in `mathir.json` config instead. This is a real, disclosed trade-off, not a bug to "fix."

Script: `benchmarks/07_utilities/compare_embedding_models.py`

**⚠️ IMPORTANT CAVEAT FOR THE AGENT ARMY**: all of this testing used BEIR's academic corpora (scifact = scientific fact-checking, nfcorpus = medical/nutrition, arguana = debate arguments). **This is NOT representative of what MATHIR actually stores in real usage** (agent memories: project decisions, bug fixes, code snippets, casual notes). Conclusions here may not transfer. **This is probably the single highest-value thing to test next** — build or find a realistic "developer/agent memory" retrieval benchmark (short informal notes as corpus, natural recall queries) instead of relying solely on academic IR benchmarks. The `benchmarks/Fluid_mecanique_book/` corpus (3208 real chunks from two textbooks, see section 5) is a partial step in a more realistic direction but still book-text, not memory-notes.

---

## 4. Five architecture ideas tested and rejected (I tried to innovate, not just reuse existing algorithms — all failed, documented honestly, don't re-waste time re-discovering these)

All in `benchmarks/07_utilities/`, all tested on nfcorpus + scifact with the real `beir` package's `EvaluateRetrieval`, all local (no LLM/API needed):

1. **Confidence-gated adaptive fusion** (`test_adaptive_fusion_hypothesis.py`) — hypothesis: only fuse BM25 when the dense ranking's top1/top2 score margin is small (ambiguous). **Rejected**: hybrid helped in BOTH the low-confidence bucket (+0.09) AND the high-confidence bucket (+0.05) — no per-query signal to gate on with the current embedder.

2. **Embedding-space pseudo-relevance feedback / PRF** (`novel_algo_embedding_prf.py`) — self-written two-pass Rocchio-style query refinement (blend query embedding with score-weighted centroid of first-pass top-m results, search again). Swept m∈{3,5,10}, β∈{0.1,0.25,0.5,1.0}. **Rejected**: +0.0063 nDCG@10 on nfcorpus but -0.0049 on scifact at best config — classic PRF "query drift" (first-pass false positives pull the refined query away from the true relevant region).

3. **Document-side hubness correction** (`novel_algo_hubness_correction.py`) — self-written: penalize each document's score by its precomputed "hub score" (mean similarity to 500 random other corpus docs), targeting the known high-dimensional "hubness" artifact (some docs become spurious nearest-neighbors to everything). **Rejected**: negligible effect at safe λ (both datasets within noise at λ≤0.5), catastrophic collapse at higher λ (scifact 0.4837→0.1196 at λ=4). Hubness isn't a meaningful effect at these corpus sizes (thousands, not millions of docs).

4. **Anisotropy correction / "all-but-the-top"** (`novel_algo_anisotropy_correction.py`) — self-written: subtract corpus mean + remove top-D principal directions (SVD-fitted) from both corpus and query embeddings before re-ranking, targeting the known anisotropy problem in sentence embeddings. Swept D from 0 to 20. **Rejected**: removing directions degrades quality close to monotonically on both datasets as D increases. One curiosity: mean-centering ALONE (D=0) helps scifact (+0.0049) but hurts nfcorpus (-0.0131) — same dataset-split pattern as everything else.

5. **Hybrid BM25 fusion + cross-encoder rerank** (pre-existing, `multi_dataset_efficient.py`) — both lose to plain dense-only on scifact/nfcorpus with `bge-base-en-v1.5`, but hybrid fusion WINS with the current weaker default embedder (see 3b).

**⚠️ CORRECTION — don't trust a clean "weak embedder → hybrid helps, strong embedder → hybrid hurts" rule.** I initially believed this (see above) and ran a direct test to confirm it: re-ran the RRF hybrid fusion test with `intfloat/multilingual-e5-small` (`benchmarks/07_utilities/retest_with_stronger_embedder.py`), a substantially stronger embedder than the default (0.6770 dense-only baseline on scifact vs the default's 0.4837). **Hybrid fusion still helped** with e5-small (+0.0225 nfcorpus, +0.0146 scifact) — it did NOT flip to hurting, even though it's a much stronger embedder than the default, and even though `bge-base-en-v1.5` (stronger still, 0.744 baseline) DOES show hybrid hurting in the pre-existing benchmark. **The real relationship is not a simple function of embedder strength** — something else differs between e5-small and bge-base-en-v1.5 specifically (possibly how each embedder's score distribution interacts with BM25's score scale inside RRF). This was NOT further characterized this session — a real, still-open question for the agent army (see section 6).

**Honest overall state**: any secondary/augmentation signal (BM25, CE, PRF) sometimes helps and sometimes hurts depending on embedder choice, but the deciding factor is more subtle than "strong vs weak" — it may be embedder-family-specific, or related to score-scale/calibration differences between embedders that a naive RRF fusion doesn't account for. The two purely-corrective techniques (hubness, anisotropy) show no reliable win at any tested setting regardless of embedder. **I did not find a mathematical breakthrough this session** — I ruled out and refined several plausible hypotheses with real evidence instead of leaving them as unverified assumptions, and caught my own oversimplification when new data contradicted it. That's the honest state of things.

---

## 5. What's built and ready but blocked on LLM API credits (pick up here first if you have working credits)

- **`benchmarks/08_industry_validation/`**: full LongMemEval + LoCoMo benchmark runners (the actual academic benchmarks Mem0/Zep report their published scores on — 66.9%/75.1% on LongMemEval respectively). Real ingest+search verified against the live daemon. Generation/judging need a funded LLM backend in `benchmarks/.env` (see `.env.example`, and NOTE: no local Ollama backend — deliberately removed, too slow for benchmark-scale runs; use a real API — OpenCode Zen's `mimo-v2.5-free` is a genuinely free option, MiniMax's native API at `api.minimax.io/v1` if you have a key). Also has a `--cross-model` mode: run the SAME retrieved context through 2+ different models to test whether accuracy depends on the model or the retrieval — direct empirical evidence for/against MATHIR's cross-provider-portability claim.
- **`benchmarks/07_utilities/generate_fluid_mechanics_queries.py`** + the 3208-chunk real corpus already built in `benchmarks/05_test_data/beir_data/fluid_mechanics/`: needs the same funded LLM backend to generate ~280 real synthetic queries, then `multi_dataset_efficient.py` (with `fluid_mechanics` already added to `DATASETS`) gives real nDCG@10/MRR@10/Recall@100 on a much larger, more realistic corpus than the old 200-chunk toy benchmark.
- **`benchmarks/09_mathir_vs_faiss_stress/`**: a parallel investigation (different agent instance, same session) building typo-robustness and long-term-decay stress tests, plus a live MATHIR-vs-FAISS comparison harness. Check its own findings/state — it was still running when this report was written.

---

## 6. Ideas NOT yet tried (genuine gaps, worth a fresh agent's attention)

- **DONE and RESOLVED**: RRF hybrid fusion and PRF were re-tested with `intfloat/multilingual-e5-small` (`benchmarks/07_utilities/retest_with_stronger_embedder.py`), then the full 3-embedder × 2-dataset matrix was completed (`benchmarks/07_utilities/complete_hybrid_flip_matrix.py`) to find the real pattern. **Resolution**: it's not about embedder identity or score-distribution shape (both tested, both rejected) — sorting by the simplest statistic, baseline dense nDCG@10 itself, gives a perfectly monotonic relationship in both datasets: hybrid-fusion delta shrinks and eventually goes negative as baseline quality rises (nfcorpus: +0.0711→+0.0225→-0.0050; scifact: +0.1193→+0.0146→-0.0157, sorted by embedder baseline quality). **Practical rule**: re-check hybrid fusion's weight whenever the embedder changes — there's no universal `bm25_weight` that stays optimal across embedder/corpus combos. Still worth doing: re-test the other 3 rejected techniques (hubness, anisotropy, confidence-gating) against this same "sort by baseline quality" lens — they were only tested with the single default (weak) embedder.
- **Multi-vector / late-interaction retrieval** (ColBERT-style: token-level embeddings + max-sim aggregation instead of one vector per document) — structurally different from every single-vector technique tried this session. Not attempted at all.
- **A small supervised re-ranker fitted on each corpus's own qrels** (e.g. logistic regression on [dense_score, bm25_score, doc_length, ...] features) instead of a fixed heuristic (RRF) or zero-shot cross-encoder — untested, could plausibly beat both since it's fit to the actual data rather than a generic heuristic.
- **Realistic-domain benchmark construction** (see the caveat in section 3c) — the highest-value gap. Build a benchmark corpus that looks like actual MATHIR usage (short project notes, decisions, bug fixes) with real recall-style queries, not academic paper abstracts.
- **Multilingual retrieval quality** — MATHIR's default embedder is explicitly multilingual; none of this session's testing touched non-English retrieval at all. A real gap given that's a stated differentiator.
- **Lifecycle/decay/consolidation mathematics** — completely separate from the retrieval-quality investigation above. The Ebbinghaus decay formula, promotion thresholds, and consolidation dedup logic (`mathir_lib/mathir_vec.py`) have never been benchmarked or mathematically stress-tested this session — worth a dedicated investigation with the same rigor applied here (real data, honest reporting, reject what doesn't work).

---

## 7. How to work in this codebase (learned the hard way this session)

- **Never fabricate results.** Every negative/mixed finding in this report was reported as-is. If an experiment can't run (no API key, etc.), say so plainly — don't invent numbers.
- **Test locally first when possible.** Every technique in section 4 was validated with pure embeddings + linear algebra, zero API calls, before considering any server code change.
- **Read `mathir_mcp/docs/DIMENSIONS.md`** for the full technical detail behind section 3/4 — this report is the summary, that doc has the exact numbers, exact commands, and exact reasoning.
- **Don't restart the live daemon carelessly** — other agent instances may have long-running background jobs against it (check `netstat -ano | grep 7338` and any `run_log.txt` files in `benchmarks/09_mathir_vs_faiss_stress/` before touching it).
- **`benchmarks/_env.py`** auto-loads `benchmarks/.env` (gitignored) — copy `.env.example`, do not hardcode credentials in scripts.
