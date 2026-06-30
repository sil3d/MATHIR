# Immunological Anomaly Detector — Design

## Context

MATHIR's MCP server defines a 5th memory tier, `immunological`, in its schema
(`mathir_lib/__init__.py: TIERS`), and the dashboard/CLI/stats all render it
as a first-class category. In the deployed product, nothing ever writes to
it: `mathir_mcp_server.py` rejects client-supplied `block_type="immunological"`
("reserved for the internal anomaly detector"), but no internal detector
exists in `mathir_lib`. The only real Mahalanobis-distance anomaly detector
in the repo lives in a deprecated, removed v7 module
(`mathir_lib.memory.immunological.MahalanobisImmunologicalMemory`, referenced
only from `benchmarks/_deprecated/v7/test_immunological_anomaly_detection.py`)
and its AUC=1.0 result was measured on synthetic Gaussian clusters separated
by 6 standard deviations — a trivially easy setup, not representative of
real prompt-injection text. The currently-shipped `mathir_dropin` library has
a simpler `_ImmuneMemory` (plain min-L2-distance to a reference bank, not
Mahalanobis at all).

This design implements a real Mahalanobis detector in `mathir_lib`, wires it
into the live `memory_save` path, and validates it against a realistic
(if small) prompt-injection test set instead of a trivial synthetic one.

## Goals

- A real, statistically-grounded anomaly detector (Mahalanobis distance with
  covariance shrinkage) running inside the actual MCP server/daemon that
  agents connect to — not a disconnected library.
- The `immunological` tier becomes genuinely populated when warranted.
- An honestly-measured detection score on realistic data, published as-is.
- A new MCP tool to audit what's been flagged.

## Non-goals

- Replacing or modifying `mathir_dropin`'s own `_ImmuneMemory` (separate
  product, out of scope here).
- Tuning for adversarial/evasive prompt injections (out of scope for v1;
  this targets clearly anomalous content, not obfuscated attacks).
- Per-project anomaly baselines synced across machines — baseline state is
  local to each project's `.mathir/mathir.db`, same as everything else.

## Architecture

### New module: `mathir_mcp/mathir_lib/mathir_anomaly.py`

```
class MahalanobisDetector:
    def __init__(self, dim: int, threshold: float, regularization: float = 1e-4,
                 warmup_count: int = 30, ema_decay: float = 0.01)
    def score(self, embedding: np.ndarray) -> float
    def update(self, embedding: np.ndarray) -> None
    def is_warmed_up(self) -> bool
    def to_dict(self) / from_dict(cls, d)   # persisted in a dedicated SQLite table
```

- Maintains a running mean vector and covariance matrix (EMA-updated, not a
  full history of raw vectors — O(dim²) state, ~590KB for dim=384).
- `Σ_reg = Σ + εI` (shrinkage) keeps the matrix invertible even with few
  samples or near-degenerate covariance.
- `score()` = `sqrt((x-μ)ᵀ Σ_reg⁻¹ (x-μ))`. The inverse is cached and only
  recomputed every N updates (default 10) to bound latency — Mahalanobis
  inversion is O(dim³), not something to redo on every single save.
- Persistence: a new table `anomaly_state` in the same SQLite DB
  (`mean BLOB, covariance BLOB, n_updates INTEGER, last_inverse BLOB`), one
  row per project DB. Loaded on daemon start, saved after each `update()`.

### Wiring into `memory_save` (`mathir_vec.py`)

1. The embedding is already computed before insertion (existing code path) —
   reused as-is, no extra embedding call.
2. Before the INSERT, call `detector.score(embedding)`.
3. If `not detector.is_warmed_up()`: skip detection entirely, proceed
   normally, still call `detector.update()` to build up the baseline.
4. Else if `score > anomaly_threshold` (existing config key, default 2.0):
   - Force `tier = "immunological"` regardless of the requested `block_type`.
   - Do not call `detector.update()` with this embedding (anomalies must not
     pollute the "normal" baseline).
   - Populate the existing `risk_warnings` field in the `memory_save`
     response with `{"anomaly_score": score, "threshold": threshold}`.
5. Else: normal flow, `detector.update()` runs with this embedding.
6. `immunological` remains terminal — excluded from the promotion chain
   (`TIER_ORDER`), consistent with the existing comment at `mathir_vec.py:77-79`.
7. Client-supplied `block_type="immunological"` is still rejected (no
   change to that existing guard) — the detector is the only writer.

### New MCP tool: `memory_incoming_links`-style audit tool

`memory_audit_immunological(project: str | None, k: int = 20) -> dict`
- Lists flagged memories for a project: id, content (truncated), anomaly
  score, timestamp, and the embedding-space distance at time of flagging.
- Read-only, no client write path (consistent with the existing guard).
- Added to the same `@mcp.tool()` registration block in
  `mathir_mcp_server.py`, raising the tool count from 23 to 24.

## Testing

Replace the trivial synthetic 6-std-dev benchmark with a realistic corpus
checked into `mathir_mcp/tests/data/anomaly_eval/`:
- ~50 "normal" texts: realistic memory content (project notes, decisions,
  bug fixes — same register as what this project's own MATHIR memories
  contain).
- ~20 known prompt-injection patterns (publicly documented patterns: fake
  system-block overrides, "ignore previous instructions" variants,
  instruction-exfiltration attempts, suspicious unicode/encoding).
- ~10 benign-but-unusual texts (raw code, number lists) to check the
  detector doesn't fire on harmless outliers.

`mathir_mcp/tests/test_anomaly_detector.py`:
- Unit: covariance shrinkage keeps the matrix invertible at low sample counts.
- Unit: warmup gate (no detection before `warmup_count` normal samples).
- Integration: real AUC-ROC computed on the corpus above via
  `sklearn.metrics.roc_auc_score`, asserted only to be `> 0.5` (better than
  random) — not asserting a specific high number, since the goal is an
  honest measurement, not a target to hit.
- Integration: a `memory_save` call with a known-injection text ends up with
  `tier == 'immunological'` in the DB, and a normal save does not.
- The actual AUC value is printed/logged so it can be reported honestly in
  docs afterward (separately, not part of this change).

## Error handling

- If `covariance` becomes singular even after shrinkage (pathological case,
  e.g. all-identical embeddings during warmup): catch the `LinAlgError`,
  fall back to skipping detection for that call (treat as warmed-up=False),
  log a warning. Never crash `memory_save` because of the detector.
- Detector state load/parse failure on daemon start: start fresh (empty
  mean/covariance), log a warning — same resilience posture as other
  best-effort subsystems in this codebase.
