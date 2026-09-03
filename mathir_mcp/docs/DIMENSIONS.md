# Embedding Dimensions Guide (v8.6.0)

**Current model (as of 2026-07-02): intfloat/multilingual-e5-small (384d, 239MB VRAM, retrieval-trained)**, was paraphrase-multilingual-MiniLM-L12-v2 (same architecture/size, different training objective). See "Real, Independently-Verified Alternative" section below for the investigation that led to this switch; the HotpotQA multi-hop result that finally justified it (+2.5x retrieval quality, ~12% slower not 5x as earlier estimated) is in MATHIR memory embedder-swap-strongest-positive-result-hotpotqa. All comparison tables below that name both models by their historical role (default vs. alternative) reflect what was tested AT THE TIME of each experiment, not the current default -- read dates/context, not just table headers.

---

## Why Change the Model?

The default model (384d) is optimized for **speed and low VRAM**. But you might want to change for:

| Reason | What to do |
|--------|-----------|
| Better recall quality | Upgrade to 768d or 1024d |
| English-only project | Use MiniLM-L6-v2 (smaller, faster) |
| Multilingual priority | Stay with paraphrase-multilingual (50+ langs) |
| Maximum quality | Use Qwen2.5-7B (3584d, GPU required) |
| Edge / minimal RAM | Use Octen-MiniLM-L6-INT8 (22MB) |

---

## Real, Independently-Verified Alternative (Same 384d Footprint)

Unlike the MTEB numbers quoted elsewhere in this doc (borrowed from public
leaderboards, not run by this project), the comparison below was actually
executed against this project's own real BEIR corpora (scifact, nfcorpus,
arguana) using the real `beir.retrieval.evaluation.EvaluateRetrieval`
methodology, on 2026-06-30 (see
`benchmarks/06_results/current/embedding_model_comparison.json` for the raw
output and `benchmarks/07_utilities/compare_embedding_models.py` to
reproduce it):

| Dataset | Default (paraphrase-multilingual-MiniLM-L12-v2) | `intfloat/multilingual-e5-small` | Encoding speed |
|---|---|---|---|
| scifact | nDCG@10 0.4837 | **nDCG@10 0.6770** (+40%) | e5-small ~5x slower |
| nfcorpus | nDCG@10 0.2345 | **nDCG@10 0.3100** (+32%) | e5-small ~5x slower |
| arguana | **nDCG@10 0.4488** | nDCG@10 0.3908 (-13%) | e5-small ~1.8x slower |

Both models are the same size class (384d, ~118M params, similar
edge/low-VRAM footprint), this is a same-footprint quality/speed
trade-off, not an upgrade to a bigger model. `multilingual-e5-small` is
trained specifically for retrieval (requires `"query: "` / `"passage: "`
text prefixes, see the comparison script for the exact usage) and wins
clearly on factual/QA-style retrieval (scifact, nfcorpus, closer to
MATHIR's typical "find a fact in my memories" use case), but loses on
argument-similarity-style retrieval (arguana) where the current default's
paraphrase-training is actually a better fit. It is also meaningfully
slower to encode, which matters for edge/resource-constrained deployments
(MATHIR explicitly targets running locally without cloud dependency).

**UPDATE (2026-07-02): MATHIR's default WAS subsequently changed to
`intfloat/multilingual-e5-small`**, superseding the "deliberately NOT
changed" decision below. What changed the calculus: a later HotpotQA
multi-hop retrieval test (not a BEIR corpus) showed a much larger gain
(+2.5x at both_gold@2, see embedder-swap-strongest-positive-result-hotpotqa
in MATHIR memory) than any BEIR result here, AND the "e5-small ~5x
slower" figure in the table above was never independently re-verified
and turned out to be wrong -- real measured single-query CPU latency is
~12% slower (49.2ms vs 55.0ms), not 5x, because both models share the
same 117.7M-parameter architecture. The original reasoning below
(preserved for history) was sound given the information available at
the time; it was the unverified 5x figure that tipped the decision, and
correcting it changed the conclusion. Existing project databases are
NOT affected by this default change (each DB is pinned to the model it
was created with -- see VecMemory.ensure_embedding_model).

Original 2026-06-30 reasoning (superseded, kept for context): "MATHIR's
default was deliberately NOT changed based on this result, the
current 384d paraphrase-multilingual model remains the default
specifically to preserve edge-device speed and because the trade-off is
task-dependent, not a clean win." If your use case is factual/QA-style
retrieval, set in `mathir.json` (now the default, shown here for
explicitness / for reverting to the old model if desired):

```json
{
  "model": "intfloat/multilingual-e5-small",
  "embedding_dim": 384
}
```

then follow the migration steps below (dimension is unchanged at 384d, so
no vec0 rebuild is needed, but re-embedding existing content is still
recommended for consistent similarity scores across old and new memories,
since the two models produce different embedding spaces at the same
dimensionality).

## Where MATHIR's Retrieval Quality Gap Actually Comes From (Investigation Notes)

A live benchmark (`benchmarks/09_mathir_vs_faiss_stress/`) showed MATHIR's
`hybrid_search` trailing a stronger-embedder FAISS baseline on real BEIR
data. Two follow-up investigations (2026-07-01,
`benchmarks/07_utilities/isolate_mathir_retrieval_bug.py` and
`test_rrf_weights.py`) isolated exactly why, so this doesn't get
misdiagnosed later:

1. **MATHIR's vector search mechanism itself is not the problem.** Holding
   the embedder fixed and comparing raw FAISS `IndexFlatIP` against
   `VecMemory.search()`'s real code path on the exact same embeddings
   (nfcorpus) produced an **identical** nDCG@10 (0.2345 = 0.2345), sqlite-vec's
   exact brute-force cosine search is mathematically equivalent to FAISS at
   these corpus sizes. (It is ~500x slower per-query in this specific
   unindexed comparison, a separate performance question, not a quality one.)
2. **The RRF fusion default weights (vector_weight=1.0, bm25_weight=1.0)
   are already near-optimal for the current embedder**, sweeping
   vector_weight from 1 to 10 (favoring the vector signal more) made
   nfcorpus nDCG@10 *worse* (0.3056 → 0.2583), not better. With a weaker
   embedder, BM25's lexical signal compensates for semantic weakness rather
   than diluting it.
3. **Conclusion: the quality gap is substantially the embedding model's
   retrieval-specific training**, not a search bug, not a fusion-weight
   misconfiguration. The trade-off is real and belongs to whoever picks the
   model (see the table above).

**⚠️ Correction (2026-07-01, later re-test), the "weak embedder → hybrid
helps, strong embedder → hybrid hurts" rule stated in an earlier version of
this section was an oversimplification, contradicted by new data.**
Re-running the exact same RRF hybrid fusion test with
`intfloat/multilingual-e5-small` (a substantially stronger embedder than
the default, e5-small's dense-only baseline is 0.6770 on scifact vs the
default's 0.4837) still showed hybrid fusion **helping**, not hurting:
+0.0225 on nfcorpus, +0.0146 on scifact
(`benchmarks/07_utilities/retest_with_stronger_embedder.py`). This
directly contradicts the simple binary rule, since `bge-base-en-v1.5`
(baseline 0.744 on scifact, even stronger than e5-small) DOES show hybrid
hurting (0.660 RRF < 0.744 dense-only, per `multi_dataset_efficient.py`).
So the real relationship is **not** a clean function of embedder strength
alone, there is a threshold or some other factor (possibly specific to
how each embedder's score distribution interacts with BM25's score scale
in RRF, or something specific to bge-base vs e5-small/the current default)
that this session did not fully characterize. Treat "hybrid fusion's
helpfulness depends on embedder choice" as the honest, narrower claim;
do NOT treat it as a simple strong/weak binary without further testing
across more embedders. PRF's effect became much smaller (roughly neutral,
+0.0006/-0.0049) with e5-small, consistent with, but not a strong
confirmation of, the weak/strong pattern for that specific technique.

**Follow-up hypothesis tested and also falsified**: maybe it's the dense
ranking's *score distribution shape* (how "peaked"/confident the top
results are, measured as a normalized top1-top10 score gap) rather than
overall retrieval quality that predicts whether hybrid fusion helps or
hurts (`benchmarks/07_utilities/investigate_hybrid_flip_factor.py`,
scifact, 3 embedders side by side):

| Embedder | Baseline nDCG@10 | Hybrid nDCG@10 | Delta | top1-top10 gap |
|---|---|---|---|---|
| default | 0.4837 | 0.6029 | **+0.1193** | 0.1897 (largest) |
| e5-small | 0.6770 | 0.6916 | +0.0146 | 0.0362 (smallest) |
| bge-base-en-v1.5 | 0.7376 | 0.7220 | **-0.0157** | 0.1175 (middle) |

If the "peaked ranking → hybrid hurts" hypothesis held, gap and delta
should be inversely ordered. They are not: gap order is
default>bge>e5, but delta order is default>e5>bge, no clean
correlation. **This hypothesis is also rejected.** Note also that
`default`'s hybrid delta here (+0.1193, scifact) is larger than reported
earlier in this document (see point 2's "helps when weak" framing):
this specific number came from re-running with a slightly different
inline harness (`investigate_hybrid_flip_factor.py`) than the original
`test_rrf_weights.py` measurement on nfcorpus; the two scripts aren't
directly comparable line-for-line (different dataset shown here), which
is itself a small methodology lesson: always name which script/dataset a
number came from, since "the hybrid delta" is not a single universal
constant even for the "same" embedder. ~~**What specifically determines
whether hybrid fusion helps or hurts a given embedder remains an open
question**~~ **RESOLVED below.**

### Resolution: it's simply baseline retrieval quality, monotonically

The "top1-top10 gap" statistic (rejected above) was the wrong variable to
look at. Completing the full 3-embedder × 2-dataset matrix
(`benchmarks/07_utilities/complete_hybrid_flip_matrix.py`, 2026-07-01) and
sorting by the simplest possible statistic, the baseline dense nDCG@10
itself, reveals a clean, **perfectly monotonic** relationship within each
dataset:

| Dataset | Embedder (sorted by baseline quality) | Baseline nDCG@10 | Hybrid delta |
|---|---|---|---|
| nfcorpus | default | 0.2345 | **+0.0711** |
| nfcorpus | e5-small | 0.3105 | **+0.0225** |
| nfcorpus | bge-base | 0.3681 | **-0.0050** |
| scifact | default | 0.4837 | **+0.1193** |
| scifact | e5-small | 0.6770 | **+0.0146** |
| scifact | bge-base | 0.7376 | **-0.0157** |

In both datasets, as baseline quality increases, the hybrid-fusion delta
decreases monotonically, crossing from clearly positive to slightly
negative. This is **not** a threshold/binary effect and **not** specific
to embedder identity, it's a continuous function of how good the dense
ranking already is on that corpus. The earlier "e5-small still helps
despite being stronger than default, contradicting the rule" framing was
comparing the wrong things: e5-small's improvement over default simply
wasn't large enough yet to cross into the harmful regime for these two
corpora; bge-base's larger improvement was what crossed it. There was
never a contradiction, just an incomplete matrix (2-3 cherry-picked
points) instead of the full sorted picture.

**Practical, actionable rule**: the marginal value of BM25 hybrid fusion
shrinks as the chosen embedder's baseline retrieval quality on your
corpus rises, and can go negative once that baseline is already strong.
There's no single universal `bm25_weight` that's right for every
embedder/corpus combination, if you upgrade to a stronger embedder,
re-check whether hybrid fusion (or its weight) still helps on YOUR data,
rather than assuming the default 1.0/1.0 weighting remains optimal.

### Rejected idea: confidence-gated adaptive fusion

A natural next architecture idea, given point 2 above: gate BM25 fusion
per-query on the dense ranking's own confidence (e.g. the top1/top2 score
margin), fuse only when the dense ranking looks ambiguous, trust dense-only
otherwise. This was tested empirically
(`benchmarks/07_utilities/test_adaptive_fusion_hypothesis.py`, 2026-07-01,
nfcorpus, median-split by margin) **before** writing any server code, and
the hypothesis did not hold: hybrid fusion improved nDCG@10 in *both* the
low-confidence bucket (0.1946 → 0.2847, +0.09) *and* the high-confidence
bucket (0.2746 → 0.3267, +0.05) with the current embedder. There is no
per-query signal here to gate on, the real, validated factor is the
*embedder's overall strength* (point 2), not per-query ambiguity. Do not
build a confidence-gated fusion mechanism on this premise; it would add
real complexity for no measured benefit. If a stronger embedder is adopted
later, this same test should be re-run with that embedder before deciding
whether hybrid should be disabled outright versus kept adaptive, the two
outcomes are indistinguishable without re-testing.

### Rejected idea: embedding-space pseudo-relevance feedback (PRF)

A genuinely novel (self-written, not FAISS/BM25/RRF/CE-based) two-pass
retrieval idea: refine the query embedding using a score-weighted centroid
of the first pass's own top-m results before searching again (Rocchio-style,
implemented from scratch in
`benchmarks/07_utilities/novel_algo_embedding_prf.py`, 2026-07-01). Swept
m ∈ {3,5,10} and blend weight β ∈ {0.1,0.25,0.5,1.0} on nfcorpus and scifact.

Result: small, inconsistent, dataset-dependent. Best config improved
nfcorpus nDCG@10 by +0.0063 (0.2345→0.2408) but the best config on scifact
still *lost* -0.0049 (0.4837→0.4787) relative to plain single-pass dense
search, and every other tested (m, β) combination on scifact lost more.
This is the classic PRF "query drift" failure mode: when the first pass's
top-m already contains false positives (more likely when the baseline
dense ranking is already fairly good, as on scifact), blending them back
into the query moves it toward noise rather than signal.

This is now the **third** independent technique (after hybrid BM25 fusion
and cross-encoder reranking) that shows the exact same pattern: it helps
when the baseline dense signal is weak and hurts when the baseline dense
signal is already strong. That's a real, generalizable finding in its own
right, any secondary/augmentation signal added on top of this embedder's
dense search inherits this same trade-off. Not adopted as a default; the
script is kept for reproducibility and to save whoever revisits this idea
from re-discovering the same drift problem from scratch.

### Rejected idea: document-side hubness correction

A fourth, structurally different novel idea (self-written, targets a
different failure mode than the three above, a *document-side* ranking
bias rather than a query-side signal):
`benchmarks/07_utilities/novel_algo_hubness_correction.py` (2026-07-01).
High-dimensional embedding spaces can suffer from "hubness", some
documents become disproportionately-frequent nearest neighbors across many
unrelated queries, an intrinsic-dimensionality artifact rather than true
relevance. Precomputed each document's mean similarity to a random sample
of 500 other corpus documents as a "hub score," then penalized ranking
scores proportionally (`adjusted = cos(q,d) - lambda * hub_score(d)`),
sweeping lambda from 0.1 to 4.0.

Result: unlike the other three techniques, this one does **not** show a
"helps weak / hurts strong baseline" pattern, it shows negligible effect
at small, safe lambda values (nfcorpus -0.0015, scifact +0.0028 at best):
both within noise) and **catastrophic** degradation at larger lambda
(scifact nDCG@10 0.4837 → 0.1196 at lambda=4, a near-total collapse, since
a query-independent penalty this large overwhelms the actual
query-document similarity signal entirely). Conclusion: hubness is not a
meaningfully large effect at these corpus sizes (thousands of documents,
not millions) with this embedder, this technique has no safe operating
point where it provides a real benefit. Not adopted.

### Rejected idea: anisotropy correction ("all-but-the-top")

A fifth technique, structurally different again, corrects the embedding
SPACE itself rather than blending a signal or penalizing scores:
`benchmarks/07_utilities/novel_algo_anisotropy_correction.py` (2026-07-01).
Sentence embeddings are known to be anisotropic (a few dominant principal
directions capture most variance but little semantic content, compressing
useful signal into a narrow cone). Self-implemented "all-but-the-top":
fit corpus mean + top principal directions via SVD, subtract the mean and
remove the top-D directions from both corpus and query embeddings before
re-normalizing and re-ranking by cosine similarity. Swept D from 0
(mean-centering only) to 20.

Result: removing principal directions makes things worse, close to
monotonically, on both datasets as D increases (nfcorpus never beats
baseline at any D; scifact degrades from +0.0049 at D=0 down to -0.0700 at
D=20). One notable side-observation: mean-centering ALONE (D=0, no
directions removed) has a small positive effect on scifact (+0.0049) but a
negative effect on nfcorpus (-0.0131), yet another instance of the same
dataset-dependent split seen in every technique tested so far. Not
adopted; the underlying anisotropy hypothesis does not hold up as
implemented here.

### Overall conclusion after five independently-tested novel/existing techniques

BM25 hybrid fusion, cross-encoder reranking, embedding-space PRF,
document-side hubness correction, and embedding-space anisotropy
correction were all tested rigorously against real BEIR data with real,
standard metrics, none of them provide a reliable, corpus-independent
improvement over plain single-pass dense search with a good embedder. Three
of the five (BM25 fusion, CE rerank, PRF) share an identical qualitative
pattern: help when the baseline dense signal is weak, hurt when it's
already strong. The other two (hubness correction, anisotropy correction)
are either negligible-to-harmful or monotonically harmful with no
redeeming operating point. The one validated, generalizable lever for
MATHIR's retrieval quality remains the embedding model choice itself (see
the table above). Further architecture changes to the *search/ranking*
mechanism are not where the effort should go without a genuinely new idea
that breaks this pattern, this is a documented, evidence-based stopping
point, not an assumption.

## How to Change Model (Step by Step)

### Step 1: Choose your model

| Model | Dims | VRAM | Speed | Quality | Languages |
|-------|------|------|-------|---------|-----------|
| **paraphrase-multilingual-MiniLM-L12-v2** | 384 | 239MB | ~104ms | ★★★☆☆ | 50+ |
| MiniLM-L6-v2 | 384 | 80MB | ~22ms | ★★★☆☆ | English |
| nomic-embed-text-v1.5 | 768 | ~500MB | ~21ms | ★★★★☆ | 100+ |
| BAAI/bge-large-en-v1.5 | 1024 | ~1.5GB | ~3ms | ★★★★☆ | English |
| e5-large-v2 | 1024 | ~1.3GB | ~3ms | ★★★★☆ | English |
| Qwen2.5-7B-emb | 3584 | ~4.7GB | ~30ms | ★★★★★ | 100+ |

### Step 2: Install the model

```bash
# For paraphrase-multilingual (default) — already installed
pip install sentence-transformers

# For nomic
pip install sentence-transformers
# Model downloads automatically on first use

# For bge-large
pip install sentence-transformers
# Model downloads automatically on first use

# For Qwen2.5-7B (requires GPU)
pip install sentence-transformers
# Model downloads automatically (~4.7GB)
```

### Step 3: Update config

Edit `~/.config/MATHIR/config/mathir.json`:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "device": "cuda",
  "embedding_dim": 384,
  "port": 7338,
  "db_path": ".mathir/mathir.db"
}
```

**Change to nomic (768d):**
```json
{
  "model": "sentence-transformers/nomic-embed-text-v1.5",
  "device": "cuda",
  "embedding_dim": 768,
  "port": 7338,
  "db_path": ".mathir/mathir.db"
}
```

**Change to bge-large (1024d):**
```json
{
  "model": "sentence-transformers/BAAI/bge-large-en-v1.5",
  "device": "cuda",
  "embedding_dim": 1024,
  "port": 7338,
  "db_path": ".mathir/mathir.db"
}
```

### Step 4: Migrate existing database (IMPORTANT!)

If you have existing memories, you MUST migrate them to the new dimensions:

```bash
# Backup first!
cp .mathir/mathir.db .mathir/mathir.db.backup

# Migrate to new dimensions
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 768

# Or for 1024d
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 1024
```

**Without migration:** Existing memories become unusable (dimension mismatch). The daemon will auto-rebuild the vec0 table, but all vector data is lost.

### Step 5: Restart daemon

```bash
# Kill existing daemon
python -m mathir_lib.mathir_client stop

# Start new daemon
python -m mathir_mcp
```

### Step 6: Verify

```bash
# Check model loaded correctly
python -m mathir_lib.mathir_client ping

# Test save + recall
python -m mathir_lib.mathir_client save "test memory" -a test -t semantic -l test
python -m mathir_lib.mathir_client recall "test" -k 1
```

---

## Quick Reference: Common Upgrades

### Upgrade 384d → 768d (nomic)

```bash
# 1. Update config
# Edit ~/.config/MATHIR/config/mathir.json:
# "model": "sentence-transformers/nomic-embed-text-v1.5"
# "embedding_dim": 768

# 2. Migrate DB
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 768

# 3. Restart daemon
python -m mathir_lib.mathir_client stop
python -m mathir_mcp
```

### Upgrade 384d → 1024d (bge-large)

```bash
# 1. Update config
# Edit ~/.config/MATHIR/config/mathir.json:
# "model": "sentence-transformers/BAAI/bge-large-en-v1.5"
# "embedding_dim": 1024

# 2. Migrate DB
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 1024

# 3. Restart daemon
python -m mathir_lib.mathir_client stop
python -m mathir_mcp
```

### Downgrade 1024d → 384d (if VRAM limited)

```bash
# 1. Update config
# Edit ~/.config/MATHIR/config/mathir.json:
# "model": "intfloat/multilingual-e5-small"
# "embedding_dim": 384

# 2. Migrate DB
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 384

# 3. Restart daemon
python -m mathir_lib.mathir_client stop
python -m mathir_mcp
```

---

## What Are Embedding Dimensions?

Embedding dimensions define the vector size representing each text chunk. Higher dimensions capture more nuance but cost more in speed, storage, and RAM.

**MATHIR default: 384d**, best balance of quality, speed, and memory usage.

| Dimensions | Example Model | Vector Size (bytes) | SQLite Index (1K memories) |
|-----------|---------------|---------------------|---------------------------|
| **384** | **paraphrase-multilingual-MiniLM-L12-v2** | **1,536** | **~2 MB** |
| 768 | nomic-embed-text-v1.5 | 3,072 | ~4 MB |
| 1024 | BAAI/bge-large-en-v1.5 | 4,096 | ~5 MB |
| 3584 | Qwen2.5-7B | 14,336 | ~18 MB |

## Why Dimensions Matter

### Quality vs Speed Trade-off

```
Quality:  384d ████████░░░░░░░ 60%
          768d ██████████░░░░░ 80%
          1024d ███████████░░░ 90%
          3584d ██████████████ 100%

Speed:    384d ██████████████ 100% (fastest)
          768d ████████████░░ 85%
          1024d ██████████░░░ 70%
          3584d █████░░░░░░░░ 35%

Storage:  384d ██░░░░░░░░░░░ 15%
          768d ████░░░░░░░░░ 30%
          1024d ██████░░░░░░ 40%
          3584d █████████████ 100%
```

### Recommendation by Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Default (MATHIR) | 384d (paraphrase-multilingual) | Best VRAM ratio, 50+ langs, ~240MB GPU |
| Balanced alternative | 768d (nomic) | Good speed/quality ratio |
| Maximum quality | 1024d (BAAI/bge-large-en-v1.5) | High quality, needs 1.5GB GPU |
| Edge / minimal | 384d (MiniLM-L6-v2) | Smallest, English-only |

### Speed Benchmarks (RTX 4060, CUDA)

| Model | Dims | Save Latency | Recall Latency (k=10) |
|-------|------|-------------|----------------------|
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | ~104ms/sent | ~140ms (k=3) |
| nomic-embed-text-v1.5 | 768 | 21ms | 27ms |
| BAAI/bge-large-en-v1.5 | 1024 | 3ms (CUDA) | 3ms (CUDA) |
| e5-large-v2 | 1024 | 2.9ms (CUDA) | 2.9ms (CUDA) |
| Qwen2.5-7B | 3584 | ~30ms (GPU) | ~40ms (GPU) |

> paraphrase-multilingual-MiniLM-L12-v2: 384d, 50+ languages, ~240MB VRAM, 0.929 cosine sim FR↔EN verified.

## Matryoshka Embedding

Some models (nomic, bge) support **Matryoshka representation learning (MRL)**, you can truncate embeddings without re-embedding:

```python
# bge-large outputs 1024d, but you can use first 768d or 384d
embedding = model.encode("text")  # shape: (1024,)
truncated = embedding[:768]       # shape: (768,) — still valid!
truncated = embedding[:384]       # shape: (384,) — still valid!
```

This lets you:
1. Start at 1024d for quality
2. Drop to 768d or 384d if speed matters later
3. No re-embedding of existing memories

## Storage Implications

SQLite-vec creates an index per dimension. Storage grows linearly:

```
1,000 memories × 1024d × 4 bytes = 4 MB
10,000 memories × 1024d × 4 bytes = 40 MB
100,000 memories × 1024d × 4 bytes = 400 MB
```

With HNSW index overhead (~1.5x), multiply by 1.5.

## Dimension Change Handling

MATHIR auto-detects dimension mismatches:

1. On startup, reads first embedding from DB
2. Compares dimensions against loaded model
3. If mismatch → drops old `vec0` table
4. Rebuilds vec0 with correct dimensions
5. All memories preserved (content + metadata intact)

**Warning**: vec0 rebuild can take minutes for large databases. Existing memories are not lost, only the vector index is recreated.

```python
# Auto-detection in mathir_server.py
existing_dim = db.execute("SELECT vec_length(embedding) FROM memory LIMIT 1").fetchone()
if existing_dim and existing_dim[0] != model_dim:
    db.execute("DROP TABLE IF EXISTS vec0")
    db.execute(create_vec0_sql(model_dim))  # Recreate with correct dims
```

## Choosing Your Dimension

| Priority | Recommendation |
|----------|---------------|
| Default (MATHIR) | 384d paraphrase-multilingual (50+ langs, low VRAM) |
| Balance first | 768d nomic |
| Quality first | 3584d Qwen2.5-7B (GPU required) |
| Edge / minimal | 384d MiniLM (CPU only) |

---

## Model Benchmarks

*(merged from MODEL_COMPARISON.md, v8.3+)*

### Benchmark Table

| Model | Dims | Size | CPU Save | CPU Recall | GPU Save | GPU Recall | MTEB Avg | License |
|-------|------|------|----------|-----------|----------|------------|----------|---------|
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 471 MB / 239 MB fp16 | ~104ms/sent | ~140ms | ~104ms/sent | ~140ms | ~49.7 (Eng) / 49.4 (Multi) | Apache-2.0 |
| MiniLM-L6-v2 | 384 | 80 MB | 22ms | 53ms |: |: | 56.26 | Apache-2.0 |
| nomic-embed-text-v1.5 | 768 | 137 MB | 21ms | 27ms | ~12ms | ~10ms | 62.38 | Apache-2.0 |
| bge-large-en-v1.5 | 1024 | 335 MB | 43ms | 25ms | **3ms** | **3ms** | 64.23 | MIT |
| e5-large-v2 | 1024 | 1.3 GB | 68ms | 45ms | ~18ms | ~15ms | 63.13 | MIT |
| Octen-MiniLM-L6-INT8 | 384 | 22 MB | 8ms | 18ms |: |: | ~55 | Apache-2.0 |
| Qwen2.5-7B-emb | 3584 | 4.7 GB |: |: | ~30ms | ~25ms | 71.5 | Apache-2.0 |

> GPU times: RTX 4060 Laptop GPU, CUDA 12.4, torch 2.6.0+cu124.
> CPU times: mid-range CPU (Ryzen 7 / i7-12700).

### Model Profiles

#### intfloat/multilingual-e5-small (384d): MATHIR DEFAULT (since 2026-07-02)
- **Best for**: Multilingual projects, retrieval-quality-sensitive use cases
- **Pros**: Retrieval-trained (query:/passage: asymmetric prefixes, applied automatically by MATHIR), same size/architecture as the previous default (117.7M params, 471MB CPU / 239MB fp16 GPU) -- switching cost is ~12% slower single-query latency, NOT 5x as an earlier (incorrect, unverified) estimate claimed. Measured +2.5x retrieval quality on HotpotQA multi-hop (both_gold@2: 15.0%->37.0%).
- **Cons**: Requires the query:/passage: prefix convention (handled internally by MATHIR's get_model_prefixes(); matters only if calling the model directly outside MATHIR)
- **Install**: pip install sentence-transformers
- **GPU**: CUDA fp16 via SentenceTransformer (falls back to CPU automatically on load/encode failure)

#### paraphrase-multilingual-MiniLM-L12-v2 (384d): PREVIOUS MATHIR DEFAULT (pre-2026-07-02)
- **Best for**: Multilingual projects (FR/EN/DE/ES/JA/ZH), low VRAM
- **Pros**: 50+ languages, Apache-2.0, 43.8M downloads, 471MB CPU / 239MB fp16 GPU
- **Cons**: Paraphrase/STS-trained, not retrieval-trained -- measurably weaker on retrieval tasks than e5-small at the same cost. Lower MTEB English (~49.7 vs bge-large 64.2), 128 token max (chunking needed)
- **Install**: pip install sentence-transformers
- **GPU**: CUDA fp16 via SentenceTransformer
- **Verified**: 0.929 cosine sim "Bonjour le monde" ↔ "Hello world" (cross-lingual)
- **Still in use by**: any project DB created before 2026-07-02 (MATHIR pins each DB to the model it was created with -- see VecMemory.ensure_embedding_model -- so existing projects are unaffected by the default change)

#### nomic-embed-text-v1.5 (768d)
- **Best for**: Balanced alternative, most projects
- **Pros**: Best speed/quality ratio, Matryoshka support, Apache-2.0
- **Cons**: Requires Optimum for ONNX export
- **Install**: pip install optimum
- **ONNX**: Export via optimum-cli export onnx

#### MiniLM-L6-v2 (384d)
- **Best for**: Edge deployment, minimal resources
- **Pros**: Tiny, 80MB RAM, fastest on CPU
- **Cons**: Lowest quality, limited nuance
- **Install**: pip install sentence-transformers

#### e5-large-v2 (1024d)
- **Best for**: Research, alternative to bge-large
- **Pros**: Strong retrieval performance
- **Cons**: 1.3GB, slowest on CPU
- **Install**: pip install sentence-transformers

#### Octen-MiniLM-L6-INT8 (384d)
- **Best for**: Edge deployment, minimal resources
- **Pros**: 22MB, fastest inference, INT8 quantized
- **Cons**: Lowest quality, INT8 incompatible with CUDA EP
- **Install**: Pre-quantized ONNX from OctoAI

#### Qwen2.5-7B-emb (3584d)
- **Best for**: Maximum quality, research, GPU servers
- **Pros**: Highest MTEB, 3584 dimensions, best accuracy
- **Cons**: 4.7GB, GPU required, high VRAM usage
- **Install**: pip install transformers accelerate
- **ONNX**: Via Optimum (GPU only)

### Recommendation Matrix

| Scenario | Model | Why |
|----------|-------|-----|
| Default for MATHIR (since 2026-07-02) | intfloat/multilingual-e5-small | 384d, retrieval-trained, low VRAM (239MB fp16), +2.5x retrieval quality vs the previous default |
| Alternative balanced | nomic-embed-text-v1.5 | 768d, best speed/quality ratio, Apache-2.0 |
| Edge / IoT device | Octen-INT8 | 22MB, 8ms CPU |
| GPU server, max quality | Qwen2.5-7B-emb | 71.5 MTEB, 3584d |
| Research benchmarks | e5-large-v2 | Strong MTEB |
| High-quality English (previous default) | bge-large-en-v1.5 | 1024d, CUDA 3ms, 64.23 MTEB |

### MTEB Scores (Retrieval)

| Model | ndcg@10 | Precision@10 | Recall@10 |
|-------|---------|-------------|-----------|
| Qwen2.5-7B-emb | 71.5 | 72.8 | 88.2 |
| bge-large-en-v1.5 | 64.2 | 65.1 | 84.7 |
| e5-large-v2 | 63.1 | 64.3 | 83.9 |
| nomic-embed-text-v1.5 | 62.4 | 63.5 | 82.1 |
| MiniLM-L6-v2 | 56.3 | 57.2 | 76.4 |

### Vector Storage Cost

`
Model               1K memories    10K memories   100K memories
────────────────────────────────────────────────────────────────
MiniLM (384d)       2 MB           20 MB          200 MB
paraphrase-multilingual (384d) 2 MB   20 MB          200 MB  ← MATHIR default
nomic (768d)        4 MB           40 MB          400 MB
bge-large (1024d)   5 MB           50 MB          500 MB
Qwen2.5 (3584d)     18 MB          180 MB         1.8 GB
`

Excludes HNSW index overhead (~1.5x multiplier).
